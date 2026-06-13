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

    # Snapshot and override module-level globals before the app starts
    old = (srv._run_dir, srv._reply_dir, srv._all_events, srv._clients)
    srv._run_dir    = tmp_path
    srv._reply_dir  = tmp_path / "ui_replies"
    srv._reply_dir.mkdir()
    srv._all_events = []
    srv._clients    = set()

    with TestClient(srv.app) as client:
        yield client, tmp_path, srv

    # Restore so other tests see a clean state
    srv._run_dir, srv._reply_dir, srv._all_events, srv._clients = old


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
# Tail task — populates _all_events from the event file
# ---------------------------------------------------------------------------

def _wait_for(condition, timeout=3.0, interval=0.05):
    """Return True if condition() becomes true within timeout seconds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False


def test_tail_task_populates_all_events(server):
    """Tail task must read ui_events.jsonl and add lines to _all_events."""
    client, run_dir, srv = server
    event_file = run_dir / "ui_events.jsonl"

    event_file.write_text(
        json.dumps({"type": "log", "line": "hello", "ts": 1.0}) + "\n"
    )

    assert _wait_for(lambda: len(srv._all_events) >= 1), \
        "tail task did not populate _all_events within 3 s"
    assert json.loads(srv._all_events[0])["line"] == "hello"


def test_tail_task_appends_new_events(server):
    """Events appended to the file after startup are also picked up."""
    client, run_dir, srv = server
    event_file = run_dir / "ui_events.jsonl"

    event_file.write_text(json.dumps({"type": "log", "line": "first", "ts": 1.0}) + "\n")
    assert _wait_for(lambda: len(srv._all_events) >= 1)

    # Append a second event
    with open(event_file, "a") as f:
        f.write(json.dumps({"type": "done", "ts": 2.0}) + "\n")

    assert _wait_for(lambda: len(srv._all_events) >= 2), \
        "tail task did not pick up appended event"
    types = [json.loads(e)["type"] for e in srv._all_events]
    assert types == ["log", "done"]


# ---------------------------------------------------------------------------
# WebSocket — historical replay
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


def test_websocket_replays_history_on_connect(server):
    """Client connecting after events exist should receive full replay."""
    client, run_dir, srv = server

    # Directly populate _all_events (decoupled from tail-task timing)
    lines = [
        json.dumps({"type": "log",              "line": "line one", "ts": 1.0}),
        json.dumps({"type": "agent_registered", "agname": "Bot", "color": "#f00", "ts": 2.0}),
    ]
    srv._all_events.extend(lines)

    with client.websocket_connect("/ws") as ws:
        received = _recv_n(ws, 2)

    assert len(received) == 2
    assert received[0]["type"] == "log"
    assert received[1]["type"] == "agent_registered"
    assert received[1]["agname"] == "Bot"


def test_websocket_new_client_sees_all_history(server):
    """A client that connects late gets every event emitted so far."""
    client, run_dir, srv = server

    for i in range(3):
        srv._all_events.append(json.dumps({"type": "log", "line": f"msg{i}", "ts": float(i)}))

    with client.websocket_connect("/ws") as ws:
        received = _recv_n(ws, 3)

    assert [e["line"] for e in received] == ["msg0", "msg1", "msg2"]


# ---------------------------------------------------------------------------
# WebSocket — live broadcast to connected clients
# ---------------------------------------------------------------------------

def test_websocket_receives_live_events(server):
    """Events written to the file after a client connects are pushed live."""
    client, run_dir, srv = server
    event_file = run_dir / "ui_events.jsonl"

    received = []

    with client.websocket_connect("/ws") as ws:
        # Write event AFTER connecting — tail task will broadcast it
        event_file.write_text(json.dumps({"type": "done", "ts": 9.0}) + "\n")

        # Wait for _all_events to be populated by the tail task
        assert _wait_for(lambda: len(srv._all_events) >= 1), \
            "tail task did not broadcast live event"

        # Receive the broadcast message with a thread+timeout
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
        # Give the server a moment to write the file
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

    # Server should still be alive
    resp = client.get("/health")
    assert resp.status_code == 200
