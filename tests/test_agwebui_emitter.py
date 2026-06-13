"""Tests for agwebui_emitter — the execution-side event writer."""
import json
import threading
import time
from pathlib import Path

import pytest

from agency.agwebui.emitter import agwebui_emitter, ansi_to_hex, _xterm256_hex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_events(run_dir: Path) -> list[dict]:
    f = run_dir / "ui_events.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# ansi_to_hex / _xterm256_hex
# ---------------------------------------------------------------------------

def test_xterm256_system_colors():
    assert _xterm256_hex(0)  == "#000000"
    assert _xterm256_hex(15) == "#ffffff"

def test_xterm256_cube():
    # index 16 = black cube corner
    assert _xterm256_hex(16) == "#000000"
    # index 231 = white cube corner
    assert _xterm256_hex(231) == "#ffffff"

def test_xterm256_greyscale():
    grey = _xterm256_hex(232)
    assert grey == "#080808"
    assert _xterm256_hex(255) == "#eeeeee"

def test_ansi_to_hex_38_5():
    # \033[38;5;214m → xterm-256 index 214
    result = ansi_to_hex("\033[38;5;214m")
    assert result == _xterm256_hex(214)

def test_ansi_to_hex_fallback():
    assert ansi_to_hex("") == "#d4d4d4"
    assert ansi_to_hex("\033[1m") == "#d4d4d4"


# ---------------------------------------------------------------------------
# emit — core writer
# ---------------------------------------------------------------------------

def test_emit_creates_file(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.emit({"type": "test", "val": 42})
    events = read_events(tmp_path)
    assert len(events) == 1
    assert events[0] == {"type": "test", "val": 42}

def test_emit_appends_lines(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.emit({"type": "a"})
    em.emit({"type": "b"})
    em.emit({"type": "c"})
    events = read_events(tmp_path)
    assert [e["type"] for e in events] == ["a", "b", "c"]

def test_emit_thread_safe(tmp_path):
    em = agwebui_emitter(tmp_path)
    N = 100

    def _write(i):
        em.emit({"type": "t", "i": i})

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    events = read_events(tmp_path)
    assert len(events) == N
    assert {e["i"] for e in events} == set(range(N))

def test_emit_non_serialisable_uses_str(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.emit({"type": "x", "val": object()})   # default=str handles it
    events = read_events(tmp_path)
    assert events[0]["type"] == "x"


# ---------------------------------------------------------------------------
# Typed emitters
# ---------------------------------------------------------------------------

def test_log(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.log("hello world")
    ev = read_events(tmp_path)[0]
    assert ev["type"] == "log"
    assert ev["line"] == "hello world"
    assert "ts" in ev

def test_agent_registered(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.agent_registered("MyAgent", "#ff8800")
    ev = read_events(tmp_path)[0]
    assert ev["type"]   == "agent_registered"
    assert ev["agname"] == "MyAgent"
    assert ev["color"]  == "#ff8800"

def test_agent_state(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.agent_state("A", "llm", "design", None)
    ev = read_events(tmp_path)[0]
    assert ev["type"]  == "agent_state"
    assert ev["state"] == "llm"
    assert ev["skill"] == "design"
    assert ev["tool"]  is None

def test_team_registered(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.team_registered("MyTeam", ["AgA", "AgB"])
    ev = read_events(tmp_path)[0]
    assert ev["type"]      == "team_registered"
    assert ev["team_name"] == "MyTeam"
    assert ev["agents"]    == ["AgA", "AgB"]

def test_push_messages(tmp_path):
    em = agwebui_emitter(tmp_path)
    msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    em.push_messages("AgX", msgs)
    ev = read_events(tmp_path)[0]
    assert ev["type"]     == "messages_snapshot"
    assert ev["agname"]   == "AgX"
    assert ev["messages"] == msgs

def test_push_messages_skips_non_serialisable(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.push_messages("AgX", [{"role": "user", "content": object()}])
    # Should write nothing (silently skipped)
    assert read_events(tmp_path) == []

def test_done(tmp_path):
    em = agwebui_emitter(tmp_path)
    em.done()
    ev = read_events(tmp_path)[0]
    assert ev["type"] == "done"


# ---------------------------------------------------------------------------
# ask_human — file-based request/reply
# ---------------------------------------------------------------------------

def test_ask_human_returns_reply(tmp_path):
    em = agwebui_emitter(tmp_path)
    reply_text = "proceed with option A"

    def _write_reply():
        time.sleep(0.1)
        (tmp_path / "ui_replies" / "abc123.txt").write_text(reply_text)

    threading.Thread(target=_write_reply, daemon=True).start()
    result = em.ask_human("Bot", "abc123", "Which option?")
    assert result == reply_text

def test_ask_human_emits_event(tmp_path):
    em = agwebui_emitter(tmp_path)

    def _write_reply():
        time.sleep(0.05)
        (tmp_path / "ui_replies" / "id99.txt").write_text("yes")

    threading.Thread(target=_write_reply, daemon=True).start()
    em.ask_human("Bot", "id99", "Continue?")

    ev = read_events(tmp_path)[0]
    assert ev["type"]     == "ask_human"
    assert ev["agname"]   == "Bot"
    assert ev["ask_id"]   == "id99"
    assert ev["question"] == "Continue?"

def test_ask_human_removes_reply_file(tmp_path):
    em = agwebui_emitter(tmp_path)
    reply_file = tmp_path / "ui_replies" / "del42.txt"

    def _write_reply():
        time.sleep(0.05)
        reply_file.write_text("done")

    threading.Thread(target=_write_reply, daemon=True).start()
    em.ask_human("Bot", "del42", "Delete test?")
    assert not reply_file.exists()
