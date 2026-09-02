"""Tests for the execution-side Web UI publisher facade."""

from __future__ import annotations

import json
import sqlite3
import threading
import time

from agency.observability.agwebui.emitter import agwebui_emitter, ansi_to_hex, _xterm256_hex


def _events(emitter: agwebui_emitter) -> list[dict]:
    emitter._flush_prune()
    connection = sqlite3.connect(emitter._db_path)
    try:
        rows = connection.execute("SELECT data FROM events ORDER BY id").fetchall()
    finally:
        connection.close()
    return [json.loads(row[0]) for row in rows]


def test_xterm256_colors_and_fallback():
    assert _xterm256_hex(0) == "#000000"
    assert _xterm256_hex(15) == "#ffffff"
    assert _xterm256_hex(16) == "#000000"
    assert _xterm256_hex(231) == "#ffffff"
    assert _xterm256_hex(232) == "#080808"
    assert _xterm256_hex(255) == "#eeeeee"
    assert ansi_to_hex("\033[38;5;214m") == _xterm256_hex(214)
    assert ansi_to_hex("") == "#d4d4d4"


def test_emitter_uses_global_database_and_pointer(tmp_path):
    emitter = agwebui_emitter(tmp_path)

    assert emitter._db_path == tmp_path / "agency.sqlite3"
    assert not (tmp_path / "ui_events.db").exists()
    assert (tmp_path / "global_data_path.txt").read_text(encoding="utf-8") == str(
        emitter._db_path.resolve()
    )


def test_emit_is_async_append_only_and_stringifies_values(tmp_path):
    emitter = agwebui_emitter(tmp_path)
    emitter.emit({"type": "a", "value": object()})
    emitter.emit({"type": "b", "value": 2})

    events = _events(emitter)
    assert [event["type"] for event in events] == ["a", "b"]
    assert isinstance(events[0]["value"], str)
    assert [event["sequence"] for event in events] == [1, 2]


def test_emit_accepts_concurrent_publishers(tmp_path):
    emitter = agwebui_emitter(tmp_path)

    threads = [
        threading.Thread(target=emitter.emit, args=({"type": "sample", "index": index},))
        for index in range(100)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    events = _events(emitter)
    assert len(events) == 100
    assert {event["index"] for event in events} == set(range(100))
    assert [event["sequence"] for event in events] == list(range(1, 101))


def test_lightweight_typed_events_are_global(tmp_path):
    emitter = agwebui_emitter(tmp_path)
    emitter.log("hello")
    emitter.team_registered("research", ["a", "b"])
    emitter.resource_update(1, 4, 2.0, 8, 1024, 16384)
    emitter.done()

    events = _events(emitter)
    assert [event["type"] for event in events] == [
        "log",
        "team_registered",
        "resource_update",
        "done",
    ]
    connection = sqlite3.connect(emitter._db_path)
    try:
        team = json.loads(
            connection.execute(
                "SELECT data FROM team_registry WHERE team_name='research'"
            ).fetchone()[0]
        )
        resource = json.loads(
            connection.execute("SELECT data FROM resource_state WHERE id=1").fetchone()[0]
        )
    finally:
        connection.close()
    assert team["agents"] == ["a", "b"]
    assert resource["gpus_acquired"] == 1


def test_detailed_agent_emitters_do_not_enter_global_database(tmp_path):
    emitter = agwebui_emitter(tmp_path)
    emitter.agent_state("a", "llm", "search", None)
    emitter.agent_config("a", {"temperature": 0.2})
    emitter.push_messages("a", [{"role": "user", "content": "hello"}])
    emitter.token_update("a", 1, 2, 3, 4)

    assert _events(emitter) == []
    connection = sqlite3.connect(emitter._db_path)
    try:
        table_names = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        connection.close()
    assert {
        "agent_tokens",
        "agent_messages",
        "agent_state",
        "agent_config_state",
    }.isdisjoint(table_names)


def test_ask_human_returns_reply_and_removes_file(tmp_path):
    emitter = agwebui_emitter(tmp_path)
    reply_file = tmp_path / "ui_replies" / "ask-1.txt"

    def reply():
        time.sleep(0.05)
        reply_file.write_text("continue", encoding="utf-8")

    threading.Thread(target=reply, daemon=True).start()
    assert emitter.ask_human("a", "ask-1", "Continue?") == "continue"
    assert not reply_file.exists()
    event = _events(emitter)[0]
    assert event["type"] == "ask_human"
    assert event["question"] == "Continue?"


def test_ask_human_timeout_is_reported(tmp_path):
    emitter = agwebui_emitter(tmp_path)
    assert emitter.ask_human("a", "ask-2", "Continue?", timeout_s=0) == (emitter._ASK_TIMEOUT_REPLY)
    assert [event["type"] for event in _events(emitter)] == [
        "ask_human",
        "human_reply",
    ]
