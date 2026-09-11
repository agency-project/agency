"""Tests for agwebui server — FastAPI endpoints and WebSocket streaming.

Fixtures write directly against agDataLogger's actual schema (events /
latest_values), matching what the live execution process produces, rather
than a hand-rolled schema of their own.
"""

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_events(db_path: Path, events: "list[dict]") -> None:
    """Insert rows directly into agDataLogger's `events` table schema.
    Each dict may carry: type, ts, name, object, call_label, payload (dict).
    Ids keep incrementing across calls on the same db so ordering stays
    correct when a test writes in more than one batch."""
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            timestamp REAL NOT NULL,
            name TEXT,
            object TEXT,
            call_label TEXT,
            payload TEXT NOT NULL,
            term_message TEXT
        )
        """
    )
    start = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    for offset, ev in enumerate(events):
        con.execute(
            "INSERT INTO events (id, type, timestamp, name, object, call_label, payload, "
            "term_message) VALUES (?,?,?,?,?,?,?,?)",
            (
                f"{start + offset:020d}{uuid.uuid4().hex}",
                ev.get("type", ""),
                float(ev.get("ts", 0)),
                ev.get("name"),
                ev.get("object"),
                ev.get("call_label"),
                json.dumps(ev.get("payload", {})),
                None,
            ),
        )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# agterm-style color palette (server.py's _make_color_palette/_agent_color)
# ---------------------------------------------------------------------------


def test_color_palette_has_54_distinct_colors():
    from agency.observability.agwebui.server import _make_color_palette

    palette = _make_color_palette()
    assert len(palette) == 54
    assert len(set(palette)) == 54
    assert all(c.startswith("#") and len(c) == 7 for c in palette)


def test_agent_color_is_stable_and_distinct_per_agent(server):
    from agency.observability.agwebui.server import _agent_color

    first = _agent_color("agent_a")
    second = _agent_color("agent_b")
    assert _agent_color("agent_a") == first
    assert first != second


# ---------------------------------------------------------------------------
# Orchestrator/scheduling events synthesize a shared-log line
# ---------------------------------------------------------------------------


def test_request_lifecycle_events_synthesize_term_message(server):
    """orchestrator.py never gives these a term_message -- _build_envelope
    must synthesize one so scheduling actions show up on the shared log,
    tagged "[scheduler]" (the source) with the agent name in the message
    body, not as the tag -- these are scheduler actions about an agent, not
    the agent's own actions."""
    from agency.observability.agwebui.server import _SCHEDULER_TAG_COLOR, _build_envelope

    envelope = json.loads(
        _build_envelope(
            "request_started",
            1.0,
            "agent_alex_0000",
            json.dumps({"skill": "file_manager", "request_id": "run0"}),
        )
    )
    assert (
        envelope["term_message"]
        == "[scheduler] agent_alex_0000  REQUEST ▶  started     file_manager"
    )
    assert envelope["color"] == _SCHEDULER_TAG_COLOR


def test_scheduler_lifecycle_events_synthesize_term_message(server):
    from agency.observability.agwebui.server import _SCHEDULER_TAG_COLOR, _build_envelope

    started = json.loads(
        _build_envelope(
            "scheduler_started", 1.0, "scheduler", json.dumps({"max_concurrent_engines": 4})
        )
    )
    assert started["term_message"] == "[scheduler] STARTED  max_concurrent_engines=4"
    assert started["color"] == _SCHEDULER_TAG_COLOR

    stopped = json.loads(_build_envelope("scheduler_stopped", 2.0, "scheduler", json.dumps({})))
    assert stopped["term_message"] == "[scheduler] STOPPED"
    assert stopped["color"] == _SCHEDULER_TAG_COLOR


def test_scheduling_events_do_not_consume_the_agent_color_palette(server):
    """ "scheduler" isn't an agent identity -- it must never take a slot in
    the round-robin agent color assignment (it always gets the fixed
    scheduler color instead)."""
    from agency.observability.agwebui.server import _agent_color_assignments, _build_envelope

    _build_envelope("scheduler_started", 1.0, "scheduler", json.dumps({}))
    _build_envelope("request_started", 2.0, "agent_alex_0000", json.dumps({"skill": "s"}))
    assert "scheduler" not in _agent_color_assignments


def test_scheduler_state_does_not_synthesize_a_term_message(server):
    """scheduler_state is a continuous snapshot, not a discrete action --
    logging it on every transition would flood the shared log."""
    from agency.observability.agwebui.server import _build_envelope

    envelope = json.loads(
        _build_envelope(
            "scheduler_state", 1.0, "scheduler", json.dumps({"ready_count": 1, "running_count": 2})
        )
    )
    assert "term_message" not in envelope


def test_db_provided_term_message_wins_over_synthesis(server):
    """team_created already carries a real term_message from agteam.py --
    synthesis must not override it."""
    from agency.observability.agwebui.server import _build_envelope

    envelope = json.loads(
        _build_envelope(
            "team_registered",
            1.0,
            "team_research_0000",
            json.dumps({"team_name": "team_research_0000", "agents": ["a"]}),
            "[team_research_0000] CREATED  parent=None",
        )
    )
    assert envelope["term_message"] == "[team_research_0000] CREATED  parent=None"


def test_long_term_message_is_truncated_only_for_the_webui_envelope(server):
    """The terminal print and the agent's own db keep the full text (see
    orchestrator.py/host_interaction_server.py's writers) -- only what
    _build_envelope hands to the browser gets shortened."""
    from agency.observability.agwebui.server import _build_envelope, _TERM_MESSAGE_DISPLAY_MAX_CHARS

    full_message = "[agent_alex_0000] SKILL ✗  s  error=" + ("x" * 5000)
    envelope = json.loads(
        _build_envelope("skill_error", 1.0, "agent_alex_0000", "{}", full_message)
    )
    assert envelope["term_message"] != full_message
    assert len(envelope["term_message"]) <= _TERM_MESSAGE_DISPLAY_MAX_CHARS + 1
    assert envelope["term_message"].endswith("…")
    assert full_message.startswith(envelope["term_message"][:-1])


def _make_data_logger(db_path: Path):
    from agency.observability.agdatalogger import agDataLogger
    from agency.configs.agconfig import agconfig, dataloggerconfig

    logger = agDataLogger(agconfig(dataloggerconfig(db_path=str(db_path))))
    logger.start()
    return logger


def _strip_ts(messages: list) -> list:
    """Every message _compute_agent_messages() returns now carries a `ts`
    (see its docstring) -- real wall-clock time, so not something a test can
    assert an exact literal for. Validate it's a plausible timestamp, then
    strip it so the rest of a message can still be compared by literal
    equality against the old expected shape."""
    stripped = []
    for message in messages:
        message = dict(message)
        ts = message.pop("ts", None)
        assert isinstance(ts, (int, float)) and ts > 0, message
        stripped.append(message)
    return stripped


# ---------------------------------------------------------------------------
# Fixture: isolated server app with its own run_dir
# ---------------------------------------------------------------------------


@pytest.fixture()
def server(tmp_path, monkeypatch):
    """Yield (TestClient, run_dir, srv_module) with a fresh server state."""
    import agency.observability.agwebui.server as srv
    from fastapi.testclient import TestClient

    # Snapshot all module-level globals before the app starts
    old_run_dir = srv._run_dir
    old_command_dir = srv._command_dir
    old_clients = srv._clients
    old_last_id = srv._last_event_id
    old_event_count = srv._event_count
    old_first_ts = srv._first_ts
    old_last_ts = srv._last_ts
    old_agent_log_cursors = srv._agent_log_cursors
    old_agent_color_assignments = srv._agent_color_assignments
    old_agent_color_next = srv._agent_color_next

    monkeypatch.delenv("AGENCY_PROFILE_DIR", raising=False)

    # Point the server at a fresh temp directory
    srv._run_dir = tmp_path
    srv._command_dir = tmp_path / "ui_commands"
    srv._command_dir.mkdir()
    srv._clients = set()
    srv._last_event_id = ""
    srv._event_count = 0
    srv._first_ts = None
    srv._last_ts = None
    srv._agent_log_cursors = {}
    srv._agent_color_assignments = {}
    srv._agent_color_next = 0

    with TestClient(srv.app) as client:
        yield client, tmp_path, srv

    # Restore so subsequent tests see a clean state
    srv._run_dir = old_run_dir
    srv._command_dir = old_command_dir
    srv._clients = old_clients
    srv._last_event_id = old_last_id
    srv._event_count = old_event_count
    srv._first_ts = old_first_ts
    srv._last_ts = old_last_ts
    srv._agent_log_cursors = old_agent_log_cursors
    srv._agent_color_assignments = old_agent_color_assignments
    srv._agent_color_next = old_agent_color_next


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
# Tail task — reads global_data.sqlite3 and updates globals
# ---------------------------------------------------------------------------


def test_tail_task_reads_events_db(server):
    """Tail task must read global_data.sqlite3 and advance _last_event_id."""
    client, run_dir, srv = server
    db_path = run_dir / "global_data.sqlite3"
    _write_events(db_path, [{"type": "log", "payload": {"line": "hello"}, "ts": 1.0}])

    assert _wait_for(lambda: srv._last_event_id != ""), (
        "tail task did not process global_data.sqlite3 within 3 s"
    )


def test_tail_task_appends_new_events(server):
    """Events inserted into the DB after startup are also picked up."""
    client, run_dir, srv = server
    db_path = run_dir / "global_data.sqlite3"

    _write_events(db_path, [{"type": "log", "payload": {"line": "first"}, "ts": 1.0}])
    assert _wait_for(lambda: srv._last_event_id != ""), "tail task did not pick up first event"
    first_id = srv._last_event_id

    _write_events(db_path, [{"type": "done", "payload": {}, "ts": 2.0}])

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
    """Client connecting after events exist should receive full replay, with
    envelopes reconstructed from the structured row (type/name/payload)."""
    client, run_dir, srv = server
    db_path = run_dir / "global_data.sqlite3"

    _write_events(
        db_path,
        [
            {"type": "log", "payload": {"line": "line one"}, "ts": 1.0},
            {
                "type": "agent_registered",
                "name": "Bot",
                "payload": {"db_path": "/x", "team": None},
                "ts": 2.0,
            },
        ],
    )

    # Wait for tail task to index the DB so the WebSocket can replay it
    assert _wait_for(lambda: srv._last_event_id != ""), (
        "tail task did not process DB before WebSocket connect"
    )

    with client.websocket_connect("/ws") as ws:
        received = _recv_skipping_sync(ws, 2)

    assert len(received) == 2
    assert received[0]["type"] == "log"
    assert received[0]["line"] == "line one"
    assert received[1]["type"] == "agent_registered"
    assert received[1]["agname"] == "Bot"
    assert isinstance(received[1]["color"], str) and received[1]["color"]


def test_websocket_new_client_sees_all_history(server):
    """A client that connects late gets every event emitted so far."""
    client, run_dir, srv = server
    db_path = run_dir / "global_data.sqlite3"

    _write_events(
        db_path,
        [{"type": "log", "payload": {"line": f"msg{i}"}, "ts": float(i)} for i in range(3)],
    )
    assert _wait_for(lambda: srv._last_event_id != ""), "tail task did not process DB"

    with client.websocket_connect("/ws") as ws:
        received = _recv_skipping_sync(ws, 3)

    assert [e["line"] for e in received] == ["msg0", "msg1", "msg2"]


# ---------------------------------------------------------------------------
# WebSocket — live broadcast to connected clients
# ---------------------------------------------------------------------------


def test_websocket_receives_live_events(server):
    """Events written to the DB after a client connects are pushed live."""
    client, run_dir, srv = server
    db_path = run_dir / "global_data.sqlite3"

    with client.websocket_connect("/ws") as ws:
        # Consume the initial timeline_sync (sent for the empty DB on connect)
        sync = json.loads(ws.receive_text())
        assert sync["type"] == "timeline_sync"

        # Insert event AFTER connecting — tail task will broadcast it
        _write_events(db_path, [{"type": "done", "payload": {}, "ts": 9.0}])

        received = _recv_n(ws, 1, timeout=3.0)

    assert received, "no live event received"
    assert received[0]["type"] == "done"


def test_shared_log_relays_new_per_agent_term_messages(server):
    """Almost none of the rich activity lines (SKILL start/success/...) are
    recorded on the global db at all -- they live on each agent's own db --
    so the shared log depends on _tail_and_broadcast() relaying them."""
    client, run_dir, srv = server

    agent_path = run_dir / "Reporter_data.sqlite3"
    agent_logger = _make_data_logger(agent_path)
    global_logger = _make_data_logger(run_dir / "global_data.sqlite3")
    global_logger.record_event(
        "agent_registered",
        {"db_path": str(agent_path), "team": None},
        name="Reporter",
        object="agent",
        update_latest_snapshot=True,
    )
    global_logger.stop()

    with client.websocket_connect("/ws") as ws:
        # Connecting also replays the agent_registered event -- once from
        # the raw event tail, once from the latest_values state preamble --
        # skip past both like everywhere else that only cares about live
        # events, not the connect-time replay.
        agent_logger.record_event(
            "skill_success",
            {"skill": "run"},
            name="Reporter",
            term_message="[Reporter] SKILL OK run",
        )
        agent_logger.stop()

        received = _recv_skipping_sync(ws, 3)

    relayed = next((e for e in received if e.get("term_message")), None)
    assert relayed is not None, f"no relayed per-agent log line received; got {received}"
    assert relayed["term_message"] == "[Reporter] SKILL OK run"
    assert relayed["agname"] == "Reporter"


def test_websocket_connect_replays_per_agent_log_backlog(server):
    """A client connecting after per-agent activity already happened still
    sees it -- not just future updates."""
    client, run_dir, srv = server

    agent_path = run_dir / "Backfilled_data.sqlite3"
    agent_logger = _make_data_logger(agent_path)
    agent_logger.record_event("agent_created", {}, term_message="[Backfilled] CREATED model=m")
    agent_logger.stop()

    global_logger = _make_data_logger(run_dir / "global_data.sqlite3")
    global_logger.record_event(
        "agent_registered",
        {"db_path": str(agent_path), "team": None},
        name="Backfilled",
        object="agent",
        update_latest_snapshot=True,
    )
    global_logger.stop()

    with client.websocket_connect("/ws") as ws:
        # agent_registered replays twice on connect (raw event tail +
        # latest_values state preamble), plus the per-agent log backlog.
        received = _recv_skipping_sync(ws, 3)

    term_messages = [e.get("term_message") for e in received]
    assert "[Backfilled] CREATED model=m" in term_messages


# ---------------------------------------------------------------------------
# WebSocket — pause/resume command handling
# ---------------------------------------------------------------------------


def _read_command_files(run_dir: Path) -> "list[dict]":
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
    db_path = run_dir / "global_data.sqlite3"
    _write_events(
        db_path,
        [
            {"type": "log", "payload": {"line": "a"}, "ts": 10.0},
            {"type": "log", "payload": {"line": "b"}, "ts": 20.0},
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
    db_path = run_dir / "global_data.sqlite3"
    _write_events(
        db_path,
        [
            {"type": "log", "payload": {"line": "early"}, "ts": 1.0},
            {"type": "log", "payload": {"line": "mid"}, "ts": 5.0},
            {"type": "log", "payload": {"line": "late"}, "ts": 9.0},
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
# Selected-agent details come from the per-agent database
# ---------------------------------------------------------------------------


def test_agent_detail_endpoint_reads_selected_agent_database(server):
    """messages/state/config/tokens all come from the agent's own db, found
    via the global db's agent_registered -> db_path pointer."""
    client, run_dir, _srv = server

    agent_path = run_dir / "LateAgent_data.sqlite3"
    agent_logger = _make_data_logger(agent_path)
    agent_logger.record_event(
        "agent_config", {"agskill": {"react_max_steps": 7}}, update_latest_snapshot=True
    )
    # One exchange's worth of llm_block rows -- one row per block, matching
    # record_final_transcript()'s real behavior, with the metadata block carrying
    # the token counts and a sibling text block that must be ignored.
    agent_logger.record_event(
        "llm_block",
        {"type": "metadata", "new_prompt_tokens": 15, "usage": {"completion_tokens": 5}},
    )
    agent_logger.record_event("llm_block", {"type": "text", "index": 0, "text": "hi"})
    agent_logger.stop()

    global_logger = _make_data_logger(run_dir / "global_data.sqlite3")
    global_logger.record_event(
        "agent_registered",
        {"db_path": str(agent_path), "team": None},
        name="LateAgent",
        object="agent",
        update_latest_snapshot=True,
    )
    global_logger.stop()

    response = client.get("/api/agents/LateAgent")
    assert response.status_code == 200
    detail = response.json()
    assert detail["config"]["agskill"]["react_max_steps"] == 7
    # The two llm_block rows share call_label=None and default to
    # role="assistant", so they get reconstructed and grouped into one
    # message (see _reconstruct_in_progress_messages) -- the metadata block
    # is kept (only filtered client-side, behind Full Logs).
    assert _strip_ts(detail["messages"]) == [
        {
            "role": "assistant",
            "blocks": [
                {"type": "metadata", "new_prompt_tokens": 15, "usage": {"completion_tokens": 5}},
                {"type": "text", "index": 0, "text": "hi"},
            ],
        },
    ]
    assert detail["state"] == {}
    assert detail["tokens"] == {"input": 15, "output": 5}


def test_agent_detail_endpoint_sums_tokens_across_multiple_exchanges(server):
    client, run_dir, _srv = server

    agent_path = run_dir / "Multi_data.sqlite3"
    agent_logger = _make_data_logger(agent_path)
    agent_logger.record_event(
        "llm_block",
        {"type": "metadata", "new_prompt_tokens": 10, "usage": {"completion_tokens": 2}},
    )
    agent_logger.record_event(
        "llm_block", {"type": "metadata", "new_prompt_tokens": 3, "usage": {"completion_tokens": 4}}
    )
    agent_logger.stop()

    global_logger = _make_data_logger(run_dir / "global_data.sqlite3")
    global_logger.record_event(
        "agent_registered",
        {"db_path": str(agent_path), "team": None},
        name="Multi",
        object="agent",
        update_latest_snapshot=True,
    )
    global_logger.stop()

    detail = client.get("/api/agents/Multi").json()
    assert detail["tokens"] == {"input": 13, "output": 6}


def test_agent_detail_endpoint_ignores_orchestrator_skill_call_and_live_messages_events(server):
    """live_messages/skill_call (orchestrator._record_execution_results) are
    not the canonical transcript source -- agcontext's own transcript has no
    per-message clock, so every message in one of those snapshots would get
    stamped with a single blanket skill-finish timestamp. Only
    llm_block/tool_result rows (record_final_transcript(), each with its own
    real timestamp) should ever surface in messages."""
    client, run_dir, _srv = server

    agent_path = run_dir / "NoSnapshot_data.sqlite3"
    agent_logger = _make_data_logger(agent_path)
    agent_logger.record_event(
        "skill_call", {"skill": "run", "history_after": [{"role": "user", "content": "hi"}]}
    )
    agent_logger.record_event(
        "live_messages",
        {"messages": [{"role": "assistant", "content": "finished"}]},
        update_latest_snapshot=True,
    )
    agent_logger.stop()

    global_logger = _make_data_logger(run_dir / "global_data.sqlite3")
    global_logger.record_event(
        "agent_registered",
        {"db_path": str(agent_path), "team": None},
        name="NoSnapshot",
        object="agent",
        update_latest_snapshot=True,
    )
    global_logger.stop()

    detail = client.get("/api/agents/NoSnapshot").json()
    assert detail["messages"] == []


# ---------------------------------------------------------------------------
# Transcript reconstruction from llm_block/tool_result events (the
# canonical source -- see _compute_agent_messages)
# ---------------------------------------------------------------------------


def test_reconstruct_in_progress_messages_leading_user_block_gets_its_own_message():
    """A harness-injected user message (e.g. the initial prompt, or a
    mid-run system/user edit) shares the exchange's call_label but carries
    role="user" in its {role, **block} payload (see
    _new_transcript_payloads) -- reconstruction keys on (call_label, role),
    so it must split into its own leading message before the assistant's
    own response blocks, even though both rows share call_label."""
    from agency.observability.agwebui.server import _reconstruct_in_progress_messages

    rows = [
        (
            "llm_block",
            "call_1",
            json.dumps({"role": "user", "type": "text", "text": "do the thing"}),
            100.0,
        ),
        ("llm_block", "call_1", json.dumps({"type": "text", "text": "working on it"}), 101.0),
    ]
    messages = _reconstruct_in_progress_messages(rows)
    assert messages == [
        {
            "role": "user",
            "blocks": [{"type": "text", "text": "do the thing"}],
            "ts": 100.0,
        },
        {
            "role": "assistant",
            "blocks": [{"type": "text", "text": "working on it"}],
            "ts": 101.0,
        },
    ]


def test_reconstruct_in_progress_messages_groups_blocks_by_call_label():
    """Consecutive rows sharing (call_label, role) group into one message --
    metadata blocks are kept here (not dropped), since filtering only
    happens client-side behind the Full Logs toggle."""
    from agency.observability.agwebui.server import _reconstruct_in_progress_messages

    rows = [
        ("llm_block", "call_1", json.dumps({"type": "thinking", "text": "hmm"}), 100.0),
        (
            "llm_block",
            "call_1",
            json.dumps({"type": "tool_use", "name": "write", "arguments": "{}"}),
            101.0,
        ),
        ("llm_block", "call_1", json.dumps({"type": "metadata", "usage": {}}), 102.0),
    ]
    messages = _reconstruct_in_progress_messages(rows)
    assert messages == [
        {
            "role": "assistant",
            "blocks": [
                {"type": "thinking", "text": "hmm"},
                {"type": "tool_use", "name": "write", "arguments": "{}"},
                {"type": "metadata", "usage": {}},
            ],
            "ts": 100.0,
        }
    ]


def test_reconstruct_in_progress_messages_llm_block_tool_result_starts_new_message():
    """A tool_result is logged as an llm_block row with role='tool' (see
    llm_handler_server._new_transcript_payloads()) -- typically under the
    *same* call_label as the next assistant turn it precedes (both get
    logged together once that next exchange finalizes), so it's the
    (call_label, role) key -- not call_label alone -- that must split them
    into separate messages. (host_interaction_server's own separate
    `tool_result` *event* is deliberately not read here at all -- see
    _compute_agent_messages -- since it's a second, differently-formatted
    record of the same result that only produced duplicate messages.)"""
    from agency.observability.agwebui.server import _reconstruct_in_progress_messages

    rows = [
        (
            "llm_block",
            "call_1",
            json.dumps({"role": "assistant", "type": "tool_use", "name": "write"}),
            100.0,
        ),
        (
            "llm_block",
            "call_2",
            json.dumps({"role": "tool", "type": "tool_result", "tool_call_id": "t1", "text": "ok"}),
            101.0,
        ),
        (
            "llm_block",
            "call_2",
            json.dumps({"role": "assistant", "type": "text", "text": "done"}),
            102.0,
        ),
    ]
    messages = _reconstruct_in_progress_messages(rows)
    assert messages == [
        {"role": "assistant", "blocks": [{"type": "tool_use", "name": "write"}], "ts": 100.0},
        {
            "role": "tool",
            "blocks": [{"type": "tool_result", "tool_call_id": "t1", "text": "ok"}],
            "ts": 101.0,
        },
        {"role": "assistant", "blocks": [{"type": "text", "text": "done"}], "ts": 102.0},
    ]


def test_agent_detail_reconstructs_in_progress_content_from_llm_block_events(server):
    """The very first skill call is still running -- messages must come
    entirely from llm_block reconstruction."""
    client, run_dir, _srv = server

    agent_path = run_dir / "InProgress_data.sqlite3"
    agent_logger = _make_data_logger(agent_path)
    agent_logger.record_event("llm_block", {"type": "text", "index": 0, "text": "working on it"})
    agent_logger.record_event(
        "llm_block", {"type": "tool_use", "index": 1, "name": "write", "arguments": "{}"}
    )
    agent_logger.stop()

    global_logger = _make_data_logger(run_dir / "global_data.sqlite3")
    global_logger.record_event(
        "agent_registered",
        {"db_path": str(agent_path), "team": None},
        name="InProgress",
        object="agent",
        update_latest_snapshot=True,
    )
    global_logger.stop()

    detail = client.get("/api/agents/InProgress").json()
    assert _strip_ts(detail["messages"]) == [
        {
            "role": "assistant",
            "blocks": [
                {"type": "text", "index": 0, "text": "working on it"},
                {"type": "tool_use", "index": 1, "name": "write", "arguments": "{}"},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Still-streaming exchange reconstruction (record_final_transcript() hasn't run yet
# for this exchange -- read the raw deltas straight from stream_deltas)
# ---------------------------------------------------------------------------


def test_reconstruct_streaming_messages_merges_text_deltas_by_index():
    from agency.observability.agwebui.server import _reconstruct_streaming_messages

    rows = [
        (
            "call_1",
            json.dumps({"type": "block_delta", "index": 0, "block_type": "text", "text": "Hel"}),
            100.0,
        ),
        (
            "call_1",
            json.dumps({"type": "block_delta", "index": 0, "block_type": "text", "text": "lo"}),
            101.0,
        ),
    ]
    messages = _reconstruct_streaming_messages(rows)
    assert messages == [
        {
            "role": "assistant",
            "blocks": [{"type": "text", "index": 0, "text": "Hello"}],
            "ts": 100.0,
        }
    ]


def test_reconstruct_streaming_messages_merges_tool_use_pieces():
    from agency.observability.agwebui.server import _reconstruct_streaming_messages

    rows = [
        (
            "call_1",
            json.dumps(
                {
                    "type": "block_delta",
                    "index": 0,
                    "block_type": "tool_use",
                    "id": "tc_1",
                    "name": "write",
                    "arguments": '{"path"',
                }
            ),
            100.0,
        ),
        (
            "call_1",
            json.dumps(
                {"type": "block_delta", "index": 0, "block_type": "tool_use", "arguments": ': "x"}'}
            ),
            101.0,
        ),
    ]
    messages = _reconstruct_streaming_messages(rows)
    assert messages == [
        {
            "role": "assistant",
            "blocks": [
                {
                    "type": "tool_use",
                    "index": 0,
                    "text": "",
                    "id": "tc_1",
                    "name": "write",
                    "arguments": '{"path": "x"}',
                }
            ],
            "ts": 100.0,
        }
    ]


def test_reconstruct_streaming_messages_drops_metadata_and_usage_items():
    from agency.observability.agwebui.server import _reconstruct_streaming_messages

    rows = [
        (
            "call_1",
            json.dumps({"type": "block_delta", "index": 0, "block_type": "text", "text": "hi"}),
            100.0,
        ),
        (
            "call_1",
            json.dumps(
                {"type": "block_delta", "index": 2**31 - 1, "block_type": "metadata", "data": {}}
            ),
            101.0,
        ),
        ("call_1", json.dumps({"type": "usage", "usage": {"prompt_tokens": 1}}), 102.0),
    ]
    messages = _reconstruct_streaming_messages(rows)
    assert messages == [
        {
            "role": "assistant",
            "blocks": [{"type": "text", "index": 0, "text": "hi"}],
            "ts": 100.0,
        }
    ]


def test_reconstruct_streaming_messages_no_rows_returns_empty():
    from agency.observability.agwebui.server import _reconstruct_streaming_messages

    assert _reconstruct_streaming_messages([]) == []


def test_agent_detail_includes_currently_streaming_exchange(server):
    """record_final_transcript() hasn't cleared stream_deltas for this exchange yet
    (it's still in progress) -- the panel must show it anyway, not wait for
    completion."""
    client, run_dir, _srv = server

    agent_path = run_dir / "Streaming_data.sqlite3"
    agent_logger = _make_data_logger(agent_path)
    agent_logger.record_stream_delta(
        "llm_stream_delta",
        {"type": "block_delta", "index": 0, "block_type": "text", "text": "Wri"},
        call_label="call_1",
    )
    agent_logger.record_stream_delta(
        "llm_stream_delta",
        {"type": "block_delta", "index": 0, "block_type": "text", "text": "ting..."},
        call_label="call_1",
    )
    agent_logger.stop()

    global_logger = _make_data_logger(run_dir / "global_data.sqlite3")
    global_logger.record_event(
        "agent_registered",
        {"db_path": str(agent_path), "team": None},
        name="Streaming",
        object="agent",
        update_latest_snapshot=True,
    )
    global_logger.stop()

    detail = client.get("/api/agents/Streaming").json()
    assert _strip_ts(detail["messages"]) == [
        {"role": "assistant", "blocks": [{"type": "text", "index": 0, "text": "Writing..."}]}
    ]


def test_agent_detail_endpoint_unknown_agent_returns_404(server):
    client, run_dir, _srv = server
    # A present-but-empty global db, so the lookup reaches the "no such
    # agent registered" branch rather than "global database is not
    # available" (a distinct, non-404 error case).
    _make_data_logger(run_dir / "global_data.sqlite3").stop()

    resp = client.get("/api/agents/__no_such_agent__")
    assert resp.status_code == 404
    assert resp.json()["error"] == "unknown agent"


# ---------------------------------------------------------------------------
# Profiler artifact download
# ---------------------------------------------------------------------------


def test_profiler_files_empty_before_the_run_finishes(server):
    """Nothing exists yet -- agprof only writes output at session stop
    (see _profiler_dir()'s docstring), so a run still in progress must not
    error, just report nothing available."""
    client, _run_dir, _srv = server
    assert client.get("/api/profiler/files").json() == {"files": []}


def test_profiler_files_lists_and_downloads_existing_artifacts(server, monkeypatch):
    client, run_dir, srv = server
    profiler_dir = run_dir / "profiler"
    profiler_dir.mkdir()
    content = b'{"schema_version": 4}'
    (profiler_dir / "summary.json").write_bytes(content)
    monkeypatch.setattr(srv, "_profiler_dir", lambda: profiler_dir)

    assert client.get("/api/profiler/files").json() == {
        "files": [{"name": "summary.json", "size": len(content)}]
    }

    resp = client.get("/api/profiler/download/summary.json")
    assert resp.status_code == 200
    assert resp.content == content
    assert "attachment" in resp.headers["content-disposition"]
    assert "summary.json" in resp.headers["content-disposition"]


def test_profiler_download_rejects_filenames_outside_the_fixed_whitelist(server):
    client, _run_dir, _srv = server
    resp = client.get("/api/profiler/download/not-a-real-artifact.txt")
    assert resp.status_code == 404


def test_profiler_download_reports_404_for_a_whitelisted_but_missing_file(server):
    client, _run_dir, _srv = server
    resp = client.get("/api/profiler/download/summary.md")
    assert resp.status_code == 404


def test_profiler_serves_only_configured_run_trace(server, monkeypatch, tmp_path):
    client, run_dir, srv = server
    monkeypatch.delenv("AGENCY_PROFILE_DIR", raising=False)
    assert client.get("/api/profiler/trace").status_code == 404
    assert client.get("/api/profiler").json()["trace_available"] is False
    profile_dir = run_dir / "profiler"
    profile_dir.mkdir()
    monkeypatch.setenv("AGENCY_PROFILE_DIR", str(profile_dir))
    trace = {"traceEvents": [{"ph": "X", "name": "tool:read"}]}
    (profile_dir / "agprof.trace.json").write_text(json.dumps(trace))
    response = client.get("/api/profiler/trace")
    assert response.json() == trace
    assert client.get("/api/profiler/download/agprof.trace.json").json() == trace
    assert client.get("/api/profiler/files").json()["files"][0]["name"] == "agprof.trace.json"
    assert response.headers["cache-control"] == "no-store"
    assert client.get("/api/profiler").json()["trace_available"] is True
    # An explicit profiler output directory takes precedence over the log dir.
    configured = tmp_path / "separate-profile"
    configured.mkdir()
    monkeypatch.setenv("AGENCY_PROFILE_DIR", str(configured))
    assert client.get("/api/profiler/trace").status_code == 404
    (configured / "agprof.trace.json").write_text('{"traceEvents": []}')
    assert client.get("/api/profiler/trace").json() == {"traceEvents": []}
    # There is no client-controlled filename or traversal route.
    assert client.get("/api/profiler/trace/other.json").status_code == 404


def test_profiler_tab_is_part_of_dashboard(server):
    client, _, _ = server
    html = client.get("/").text
    assert 'id="view-profiler"' in html
    assert 'id="profiler-frame"' in html
    assert "/static/profiler.js" in html
    assert client.get("/static/profiler.js").status_code == 200


def test_profiler_viewer_and_downloads_share_default_sibling_directory(server, monkeypatch):
    client, run_dir, srv = server
    monkeypatch.setattr(srv, "_run_dir", run_dir / "logs")
    monkeypatch.delenv("AGENCY_PROFILE_DIR", raising=False)
    directory = run_dir / "profiler"
    directory.mkdir()
    trace = {"traceEvents": []}
    (directory / "agprof.trace.json").write_text(json.dumps(trace))
    assert client.get("/api/profiler").json()["trace_available"] is True
    assert client.get("/api/profiler/trace").json() == trace
    assert client.get("/api/profiler/download/agprof.trace.json").json() == trace
