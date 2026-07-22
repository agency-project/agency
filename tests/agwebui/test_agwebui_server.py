"""Tests for agwebui server — FastAPI endpoints and WebSocket streaming."""

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_events(db_path: Path, events: list[dict]) -> None:
    """Insert events directly into the SQLite database."""
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id     INTEGER PRIMARY KEY AUTOINCREMENT,
            type   TEXT    NOT NULL,
            agname TEXT,
            ts     REAL    NOT NULL,
            data   TEXT    NOT NULL
        )
    """)
    for ev in events:
        data = json.dumps(ev)
        con.execute(
            "INSERT INTO events(type, agname, ts, data) VALUES(?,?,?,?)",
            (ev.get("type", ""), ev.get("agname"), float(ev.get("ts", 0)), data),
        )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Fixture: isolated server app with its own run_dir
# ---------------------------------------------------------------------------


@pytest.fixture()
def server(tmp_path):
    """Yield (TestClient, run_dir, srv_module) with a fresh server state."""
    import agency.agwebui.server as srv
    from fastapi.testclient import TestClient

    # Snapshot all module-level globals before the app starts
    old_run_dir = srv._run_dir
    old_reply_dir = srv._reply_dir
    old_command_dir = srv._command_dir
    old_clients = srv._clients
    old_last_id = srv._last_event_id
    old_event_count = srv._event_count
    old_first_ts = srv._first_ts
    old_last_ts = srv._last_ts

    # Point the server at a fresh temp directory
    srv._run_dir = tmp_path
    srv._reply_dir = tmp_path / "ui_replies"
    srv._reply_dir.mkdir()
    srv._command_dir = tmp_path / "ui_commands"
    srv._command_dir.mkdir()
    srv._clients = set()
    srv._last_event_id = 0
    srv._event_count = 0
    srv._first_ts = None
    srv._last_ts = None

    with TestClient(srv.app) as client:
        yield client, tmp_path, srv

    # Restore so subsequent tests see a clean state
    srv._run_dir = old_run_dir
    srv._reply_dir = old_reply_dir
    srv._command_dir = old_command_dir
    srv._clients = old_clients
    srv._last_event_id = old_last_id
    srv._event_count = old_event_count
    srv._first_ts = old_first_ts
    srv._last_ts = old_last_ts


def _wait_for(condition, timeout=3.0, interval=0.05):
    """Return True if condition() becomes true within timeout seconds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


def test_health(server):
    client, _, _ = server
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_index_returns_html(server):
    client, _, _ = server
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert b"Agency Web UI" in resp.content


def test_static_css_served(server):
    client, _, _ = server
    resp = client.get("/static/style.css")
    assert resp.status_code == 200


def test_static_js_served(server):
    client, _, _ = server
    resp = client.get("/static/app.js")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tail task — reads ui_events.db and updates globals
# ---------------------------------------------------------------------------


def test_tail_task_reads_events_db(server):
    """Tail task must read ui_events.db and advance _last_event_id."""
    client, run_dir, srv = server
    db_path = run_dir / "ui_events.db"
    _write_events(db_path, [{"type": "log", "line": "hello", "ts": 1.0}])

    assert _wait_for(lambda: srv._last_event_id > 0), (
        "tail task did not process ui_events.db within 3 s"
    )


def test_tail_task_appends_new_events(server):
    """Events inserted into the DB after startup are also picked up."""
    client, run_dir, srv = server
    db_path = run_dir / "ui_events.db"

    _write_events(db_path, [{"type": "log", "line": "first", "ts": 1.0}])
    assert _wait_for(lambda: srv._last_event_id > 0), "tail task did not pick up first event"
    first_id = srv._last_event_id

    _write_events(db_path, [{"type": "done", "ts": 2.0}])

    assert _wait_for(lambda: srv._last_event_id > first_id), (
        "tail task did not pick up second event"
    )


# ---------------------------------------------------------------------------
# WebSocket helpers
# ---------------------------------------------------------------------------


def _recv_n(ws, n, timeout=5.0):
    """Receive exactly n messages from ws using a background thread with timeout."""
    results = []

    def _reader():
        try:
            for _ in range(n):
                results.append(json.loads(ws.receive_text()))
        except Exception as _e:
            # Expected once the test's own timeout below gives up and the
            # connection is torn down while this thread is still receiving.
            print(f"_recv_n reader stopped early: {_e}")

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return results


def _recv_skipping_sync(ws, n, timeout=5.0):
    """Receive n non-timeline_sync messages, discarding the initial sync packet."""
    results = []

    def _reader():
        try:
            while len(results) < n:
                msg = json.loads(ws.receive_text())
                if msg.get("type") != "timeline_sync":
                    results.append(msg)
        except Exception as _e:
            # Expected once the test's own timeout below gives up and the
            # connection is torn down while this thread is still receiving.
            print(f"_recv_skipping_sync reader stopped early: {_e}")

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return results


# ---------------------------------------------------------------------------
# WebSocket — initial timeline_sync on connect
# ---------------------------------------------------------------------------


def test_websocket_sends_timeline_sync_on_connect(server):
    """First message on every WebSocket connection must be timeline_sync."""
    client, _, _ = server
    with client.websocket_connect("/ws") as ws:
        received = _recv_n(ws, 1, timeout=3.0)
    assert received, "no message received"
    assert received[0]["type"] == "timeline_sync"


# ---------------------------------------------------------------------------
# WebSocket — historical replay
# ---------------------------------------------------------------------------


def test_websocket_replays_history_on_connect(server):
    """Client connecting after events exist should receive full replay."""
    client, run_dir, srv = server
    db_path = run_dir / "ui_events.db"

    _write_events(
        db_path,
        [
            {"type": "log", "line": "line one", "ts": 1.0},
            {"type": "agent_registered", "agname": "Bot", "color": "#f00", "ts": 2.0},
        ],
    )

    # Wait for tail task to index the DB so the WebSocket can replay it
    assert _wait_for(lambda: srv._last_event_id > 0), (
        "tail task did not process DB before WebSocket connect"
    )

    with client.websocket_connect("/ws") as ws:
        received = _recv_skipping_sync(ws, 2)

    assert len(received) == 2
    assert received[0]["type"] == "log"
    assert received[1]["type"] == "agent_registered"
    assert received[1]["agname"] == "Bot"


def test_websocket_new_client_sees_all_history(server):
    """A client that connects late gets every event emitted so far."""
    client, run_dir, srv = server
    db_path = run_dir / "ui_events.db"

    _write_events(db_path, [{"type": "log", "line": f"msg{i}", "ts": float(i)} for i in range(3)])
    assert _wait_for(lambda: srv._last_event_id > 0), "tail task did not process DB"

    with client.websocket_connect("/ws") as ws:
        received = _recv_skipping_sync(ws, 3)

    assert [e["line"] for e in received] == ["msg0", "msg1", "msg2"]


# ---------------------------------------------------------------------------
# WebSocket — live broadcast to connected clients
# ---------------------------------------------------------------------------


def test_websocket_receives_live_events(server):
    """Events written to the DB after a client connects are pushed live."""
    client, run_dir, srv = server
    db_path = run_dir / "ui_events.db"

    with client.websocket_connect("/ws") as ws:
        # Consume the initial timeline_sync (sent for the empty DB on connect)
        sync = json.loads(ws.receive_text())
        assert sync["type"] == "timeline_sync"

        # Insert event AFTER connecting — tail task will broadcast it
        _write_events(db_path, [{"type": "done", "ts": 9.0}])

        received = _recv_n(ws, 1, timeout=3.0)

    assert received, "no live event received"
    assert received[0]["type"] == "done"


# ---------------------------------------------------------------------------
# WebSocket — human_reply handling
# ---------------------------------------------------------------------------


def test_websocket_human_reply_writes_file(server):
    client, run_dir, srv = server

    with client.websocket_connect("/ws") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "human_reply",
                    "ask_id": "testask01",
                    "text": "my answer",
                }
            )
        )
        assert _wait_for(lambda: (run_dir / "ui_replies" / "testask01.txt").exists()), (
            "reply file was not written"
        )

    reply_file = run_dir / "ui_replies" / "testask01.txt"
    assert reply_file.read_text() == "my answer"


def test_websocket_human_reply_empty_ask_id_ignored(server):
    """A human_reply with no ask_id should not create any file."""
    client, run_dir, srv = server

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "human_reply", "text": "oops"}))
        time.sleep(0.1)

    assert not any((run_dir / "ui_replies").iterdir())


# ---------------------------------------------------------------------------
# WebSocket — pause/resume command handling
# ---------------------------------------------------------------------------


def _read_command_files(run_dir: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in (run_dir / "ui_commands").glob("*.json")]


def test_websocket_pause_writes_command_file(server):
    client, run_dir, srv = server

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "pause", "agname": "alex_0000"}))
        assert _wait_for(lambda: _read_command_files(run_dir)), "pause command file was not written"

    cmds = _read_command_files(run_dir)
    assert len(cmds) == 1
    assert cmds[0] == {"type": "pause", "agname": "alex_0000"}


def test_websocket_resume_writes_command_file(server):
    client, run_dir, srv = server

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "resume", "agname": "alex_0000"}))
        assert _wait_for(lambda: _read_command_files(run_dir))

    cmds = _read_command_files(run_dir)
    assert cmds == [{"type": "resume", "agname": "alex_0000"}]


@pytest.mark.parametrize("mtype", ["pause_all", "resume_all"])
def test_websocket_pause_all_resume_all_write_command_file(server, mtype):
    client, run_dir, srv = server

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": mtype}))
        assert _wait_for(lambda: _read_command_files(run_dir))

    cmds = _read_command_files(run_dir)
    assert cmds == [{"type": mtype, "agname": None}]


def test_websocket_update_config_writes_command_file(server):
    client, run_dir, srv = server

    with client.websocket_connect("/ws") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "update_config",
                    "agname": "alex_0000",
                    "config": {"agskill": {"react_max_steps": 5}},
                }
            )
        )
        assert _wait_for(lambda: _read_command_files(run_dir))

    cmds = _read_command_files(run_dir)
    assert cmds == [
        {
            "type": "update_config",
            "agname": "alex_0000",
            "config": {"agskill": {"react_max_steps": 5}},
        }
    ]


def test_websocket_update_config_all_writes_command_file(server):
    client, run_dir, srv = server

    with client.websocket_connect("/ws") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "update_config_all",
                    "config": {"agskill": {"react_max_steps": 9}},
                }
            )
        )
        assert _wait_for(lambda: _read_command_files(run_dir))

    cmds = _read_command_files(run_dir)
    assert cmds == [
        {
            "type": "update_config_all",
            "agname": None,
            "config": {"agskill": {"react_max_steps": 9}},
        }
    ]


def test_websocket_update_config_missing_config_defaults_empty(server):
    client, run_dir, srv = server

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "update_config", "agname": "a"}))
        assert _wait_for(lambda: _read_command_files(run_dir))

    cmds = _read_command_files(run_dir)
    assert cmds == [{"type": "update_config", "agname": "a", "config": {}}]


def test_websocket_multiple_pause_commands_each_get_own_file(server):
    """Each command must land in its own file — a single overwritten file
    would silently drop all but the last command between poll cycles."""
    client, run_dir, srv = server

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "pause", "agname": "a"}))
        ws.send_text(json.dumps({"type": "pause", "agname": "b"}))
        assert _wait_for(lambda: len(_read_command_files(run_dir)) == 2)

    agnames = {c["agname"] for c in _read_command_files(run_dir)}
    assert agnames == {"a", "b"}


def test_websocket_malformed_json_ignored(server):
    """Malformed JSON from the client must not crash the server."""
    client, _, _ = server

    with client.websocket_connect("/ws") as ws:
        ws.send_text("not json {{")
        time.sleep(0.1)

    resp = client.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/timeline endpoint
# ---------------------------------------------------------------------------


def test_api_timeline_empty(server):
    """Timeline endpoint returns empty metadata when no DB exists."""
    client, _, _ = server
    resp = client.get("/api/timeline")
    assert resp.status_code == 200
    j = resp.json()
    assert j["index_len"] == 0
    assert j["first_ts"] is None
    assert j["last_ts"] is None


def test_api_timeline_with_events(server):
    """Timeline endpoint returns event count and timestamp range."""
    client, run_dir, _ = server
    db_path = run_dir / "ui_events.db"
    _write_events(
        db_path,
        [
            {"type": "log", "line": "a", "ts": 10.0},
            {"type": "log", "line": "b", "ts": 20.0},
        ],
    )
    resp = client.get("/api/timeline")
    assert resp.status_code == 200
    j = resp.json()
    assert j["first_ts"] == pytest.approx(10.0)
    assert j["last_ts"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# /api/events endpoint
# ---------------------------------------------------------------------------


def test_api_events_range(server):
    """Events endpoint returns events in the requested time range."""
    client, run_dir, _ = server
    db_path = run_dir / "ui_events.db"
    _write_events(
        db_path,
        [
            {"type": "log", "line": "early", "ts": 1.0},
            {"type": "log", "line": "mid", "ts": 5.0},
            {"type": "log", "line": "late", "ts": 9.0},
        ],
    )
    resp = client.get("/api/events?start_ts=3.0&end_ts=7.0")
    assert resp.status_code == 200
    j = resp.json()
    lines = [json.loads(e)["line"] for e in j["events"]]
    assert lines == ["mid"]


def test_api_events_invalid_range_returns_empty(server):
    """end_ts <= start_ts returns empty events list."""
    client, _, _ = server
    resp = client.get("/api/events?start_ts=10.0&end_ts=5.0")
    assert resp.status_code == 200
    assert resp.json()["events"] == []


# ---------------------------------------------------------------------------
# WebSocket — state preamble survives aging out of TAIL_EVENTS
# ---------------------------------------------------------------------------


def test_agent_config_survives_being_pushed_out_of_tail_window(server):
    """Regression test: agent_config used to only ever be inserted into the
    append-only events table, with no latest-value table of its own -- so
    once more than TAIL_EVENTS other events landed after it, a client
    connecting later could never recover it (see agwebui_emitter's
    _UPSERT_TABLES). Uses the real emitter (not the hand-crafted
    _write_events helper) so this exercises the actual upsert path, then
    floods well past TAIL_EVENTS with unrelated log lines before connecting,
    proving the config is still recoverable via the state preamble alone."""
    client, run_dir, srv = server
    from agency.agwebui.emitter import agwebui_emitter

    em = agwebui_emitter(run_dir)
    em.agent_config("LateAgent", {"agskill": {"react_max_steps": 7}})
    for i in range(srv.TAIL_EVENTS + 50):
        em.log(f"filler{i}")

    assert _wait_for(lambda: srv._last_event_id > srv.TAIL_EVENTS), (
        "tail task did not catch up on the flood of filler events"
    )

    with client.websocket_connect("/ws") as ws:
        # TAIL_EVENTS log lines exhaust the tail replay itself; the state
        # preamble (where agent_config actually lives, as the only row in
        # agent_config_state) is sent right after -- exactly one more
        # message to wait for.
        received = _recv_skipping_sync(ws, srv.TAIL_EVENTS + 1)

    configs = [e for e in received if e.get("type") == "agent_config"]
    assert configs, "agent_config was not recovered via the state preamble"
    assert configs[-1]["agname"] == "LateAgent"
    assert configs[-1]["config"]["agskill"]["react_max_steps"] == 7
