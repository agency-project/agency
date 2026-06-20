"""Tests for agwebui server — FastAPI endpoints and WebSocket streaming."""
import json
import threading
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixture: isolated server app with its own run_dir
# ---------------------------------------------------------------------------

@pytest.fixture()
def server(tmp_path):
    """Yield (TestClient, run_dir, srv_module) with a fresh server state."""
    import agency.agwebui.server as srv
    from fastapi.testclient import TestClient

    # Snapshot all module-level globals before the app starts
    old_run_dir   = srv._run_dir
    old_reply_dir = srv._reply_dir
    old_clients   = srv._clients
    old_index     = srv._sparse_index
    old_since_idx = srv._events_since_index
    old_offset    = srv._file_offset
    old_size      = srv._file_size
    old_first_ts  = srv._first_ts
    old_last_ts   = srv._last_ts

    # Point the server at a fresh temp directory
    srv._run_dir            = tmp_path
    srv._reply_dir          = tmp_path / "ui_replies"
    srv._reply_dir.mkdir()
    srv._clients            = set()
    srv._sparse_index       = []
    srv._events_since_index = 0
    srv._file_offset        = 0
    srv._file_size          = 0
    srv._first_ts           = None
    srv._last_ts            = None

    with TestClient(srv.app) as client:
        yield client, tmp_path, srv

    # Restore so subsequent tests see a clean state
    srv._run_dir            = old_run_dir
    srv._reply_dir          = old_reply_dir
    srv._clients            = old_clients
    srv._sparse_index       = old_index
    srv._events_since_index = old_since_idx
    srv._file_offset        = old_offset
    srv._file_size          = old_size
    srv._first_ts           = old_first_ts
    srv._last_ts            = old_last_ts


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
# Tail task — reads ui_events.jsonl and updates file-offset globals
# ---------------------------------------------------------------------------

def _wait_for(condition, timeout=3.0, interval=0.05):
    """Return True if condition() becomes true within timeout seconds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False


def test_tail_task_reads_events_file(server):
    """Tail task must read ui_events.jsonl and advance _file_offset."""
    client, run_dir, srv = server
    event_file = run_dir / "ui_events.jsonl"

    event_file.write_text(
        json.dumps({"type": "log", "line": "hello", "ts": 1.0}) + "\n"
    )

    assert _wait_for(lambda: srv._file_offset > 0), \
        "tail task did not process ui_events.jsonl within 3 s"


def test_tail_task_appends_new_events(server):
    """Events appended to the file after startup are also picked up."""
    client, run_dir, srv = server
    event_file = run_dir / "ui_events.jsonl"

    event_file.write_text(json.dumps({"type": "log", "line": "first", "ts": 1.0}) + "\n")
    assert _wait_for(lambda: srv._file_offset > 0), \
        "tail task did not pick up first event"
    first_offset = srv._file_offset

    with open(event_file, "a") as f:
        f.write(json.dumps({"type": "done", "ts": 2.0}) + "\n")

    assert _wait_for(lambda: srv._file_offset > first_offset), \
        "tail task did not pick up appended event"


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
        except Exception:
            pass

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
        except Exception:
            pass

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
    event_file = run_dir / "ui_events.jsonl"

    lines = [
        json.dumps({"type": "log",              "line": "line one", "ts": 1.0}),
        json.dumps({"type": "agent_registered", "agname": "Bot", "color": "#f00", "ts": 2.0}),
    ]
    event_file.write_text("\n".join(lines) + "\n")

    # Wait for tail task to index the file so the WebSocket can replay it
    assert _wait_for(lambda: srv._file_offset > 0), \
        "tail task did not process file before WebSocket connect"

    with client.websocket_connect("/ws") as ws:
        received = _recv_skipping_sync(ws, 2)

    assert len(received) == 2
    assert received[0]["type"] == "log"
    assert received[1]["type"] == "agent_registered"
    assert received[1]["agname"] == "Bot"


def test_websocket_new_client_sees_all_history(server):
    """A client that connects late gets every event emitted so far."""
    client, run_dir, srv = server
    event_file = run_dir / "ui_events.jsonl"

    lines = "\n".join(
        json.dumps({"type": "log", "line": f"msg{i}", "ts": float(i)})
        for i in range(3)
    ) + "\n"
    event_file.write_text(lines)
    assert _wait_for(lambda: srv._file_offset > 0), \
        "tail task did not process file"

    with client.websocket_connect("/ws") as ws:
        received = _recv_skipping_sync(ws, 3)

    assert [e["line"] for e in received] == ["msg0", "msg1", "msg2"]


# ---------------------------------------------------------------------------
# WebSocket — live broadcast to connected clients
# ---------------------------------------------------------------------------

def test_websocket_receives_live_events(server):
    """Events written to the file after a client connects are pushed live."""
    client, run_dir, srv = server
    event_file = run_dir / "ui_events.jsonl"

    with client.websocket_connect("/ws") as ws:
        # Consume the initial timeline_sync (sent for the empty file on connect)
        sync = json.loads(ws.receive_text())
        assert sync["type"] == "timeline_sync"

        # Write event AFTER connecting — tail task will broadcast it
        event_file.write_text(json.dumps({"type": "done", "ts": 9.0}) + "\n")

        received = _recv_n(ws, 1, timeout=3.0)

    assert received, "no live event received"
    assert received[0]["type"] == "done"


# ---------------------------------------------------------------------------
# WebSocket — human_reply handling
# ---------------------------------------------------------------------------

def test_websocket_human_reply_writes_file(server):
    client, run_dir, srv = server

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({
            "type": "human_reply",
            "ask_id": "testask01",
            "text": "my answer",
        }))
        assert _wait_for(
            lambda: (run_dir / "ui_replies" / "testask01.txt").exists()
        ), "reply file was not written"

    reply_file = run_dir / "ui_replies" / "testask01.txt"
    assert reply_file.read_text() == "my answer"


def test_websocket_human_reply_empty_ask_id_ignored(server):
    """A human_reply with no ask_id should not create any file."""
    client, run_dir, srv = server

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "human_reply", "text": "oops"}))
        time.sleep(0.1)

    assert not any((run_dir / "ui_replies").iterdir())


def test_websocket_malformed_json_ignored(server):
    """Malformed JSON from the client must not crash the server."""
    client, _, _ = server

    with client.websocket_connect("/ws") as ws:
        ws.send_text("not json {{")
        time.sleep(0.1)

    resp = client.get("/health")
    assert resp.status_code == 200
