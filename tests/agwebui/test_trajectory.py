import json
import sqlite3
from concurrent.futures import Future
from collections import deque

from fastapi.testclient import TestClient

from agency.observability.agwebui.trajectory import LiveSource, Trajectory, replay_events
from agency.observability.agwebui.trajectory_semantics import validate_suggestion


def event(index, kind, ts=None, actor="worker", **payload):
    return {
        "id": f"event/{index}",
        "type": kind,
        "actor": actor,
        "ts": index if ts is None else ts,
        "payload": payload,
    }


def model():
    result = Trajectory("test", "Test execution")
    result.apply(event(0, "workload_started"))
    return result


def test_live_workload_metadata_reaches_snapshot_and_incremental_patch():
    trajectory = Trajectory("live", "Live execution")
    trajectory.apply(
        event(
            0,
            "workload_started",
            task="psf__requests-1963",
            model="gpt-6-luna",
            harness="codex",
            dataset="SWE-bench Lite",
        )
    )
    for result in (trajectory.snapshot(), trajectory.patch()["meta"]):
        assert result["model"] == "gpt-6-luna"
        assert result["harness"] == "codex"
        assert result["dataset"] == "SWE-bench Lite"


def test_call_appears_running_then_keeps_its_action_and_episode_identity():
    trajectory = model()
    trajectory.apply(
        event(1, "tool_call", call_id="call", tool="Bash", arguments={"command": "pytest target"})
    )
    action = trajectory.run["actions"][0]
    episode = action["episode"]
    assert action["outcome"] == "running"
    assert trajectory.episodes[episode]["latest_result"] == "No result recorded yet."
    trajectory.tick(30)
    assert action["duration"] == 29
    assert not trajectory.signals  # A long-running check is not a failure or stall.
    trajectory.apply(
        event(
            2, "tool_result", ts=40, call_id="call", result={"exit_code": 0, "stdout": "1 passed"}
        )
    )
    assert trajectory.run["actions"][0] is action
    assert action["episode"] == episode
    assert action["outcome"] == "success"
    assert "1 passed" in trajectory.episodes[episode]["latest_result"]
    assert trajectory.run["resolved"] is None


def test_duplicates_late_results_and_end_before_start_do_not_duplicate_calls():
    trajectory = model()
    end = event(1, "tool_result", ts=10, call_id="late", tool="Bash", result="done")
    trajectory.apply(end)
    trajectory.apply(end)
    episode = trajectory.run["actions"][0]["episode"]
    trajectory.apply(
        event(
            2, "tool_call", ts=2, call_id="late", tool="Bash", arguments={"command": "echo actual"}
        )
    )
    action = trajectory.run["actions"][0]
    assert len(trajectory.run["actions"]) == 1
    assert action["outcome"] == "success"
    assert not action["missing_start"]
    assert action["duration"] == 8
    assert action["episode"] == episode
    assert action["command"] == "echo actual"


def test_late_start_is_labeled_and_never_reassigns_existing_primary_episodes():
    trajectory = model()
    trajectory.apply(event(1, "tool_call", ts=10, call_id="new", tool="read_file"))
    existing = trajectory.run["episodes"][0]["id"]
    trajectory.apply(event(2, "tool_call", ts=2, call_id="older", tool="read_file"))
    assert trajectory.actions["worker:new"]["episode"] == existing
    assert trajectory.episodes[trajectory.actions["worker:older"]["episode"]]["late"]


def test_actor_request_and_declared_workstream_boundaries_are_structural():
    trajectory = model()
    annotation = {
        "status": "valid",
        "raw": {"purpose": "Investigating retry behavior", "workstream_ids": ["retry"]},
    }
    trajectory.apply(
        event(1, "tool_call", call_id="a", tool="read_file", annotation=annotation, request_id="r1")
    )
    trajectory.apply(
        event(
            2,
            "tool_call",
            actor="second",
            call_id="b",
            tool="read_file",
            annotation=annotation,
            request_id="r1",
        )
    )
    trajectory.apply(
        event(3, "tool_call", call_id="c", tool="edit", annotation=annotation, request_id="r1")
    )
    trajectory.apply(
        event(4, "tool_call", call_id="d", tool="read_file", annotation=annotation, request_id="r2")
    )
    a, b, c, d = trajectory.run["actions"]
    assert a["episode"] == c["episode"]
    assert a["episode"] != b["episode"] != d["episode"]
    assert trajectory.episodes[a["episode"]]["label_source"] == "declared"
    assert all(
        len({trajectory.actions[key]["agent"] for key in e["actions"]}) == 1
        for e in trajectory.run["episodes"]
    )


def test_deterministic_fallback_is_repeatable_and_does_not_claim_discovery():
    first, second = model(), model()
    for trajectory in (first, second):
        trajectory.apply(
            event(1, "tool_call", call_id="a", tool="read_file", annotation={"status": "malformed"})
        )
    assert first.snapshot() == second.snapshot()
    assert first.run["episodes"][0]["title"] == "Inspecting the behavior"
    assert first.run["episodes"][0]["label_source"] == "deterministic"


def check(trajectory, index, call_id, result, command="pytest target"):
    trajectory.apply(
        event(index, "tool_call", call_id=call_id, tool="Bash", arguments={"command": command})
    )
    trajectory.apply(event(index + 1, "tool_result", call_id=call_id, result=result))


def test_repeated_matching_failures_have_evidence_and_edits_break_the_sequence():
    trajectory = model()
    failure = {"exit_code": 1, "stdout": "same assertion"}
    for index in range(3):
        check(trajectory, index * 2 + 1, str(index), failure)
    signal = trajectory.signals["repeat:worker"]
    assert signal["severity"] == "inspect"
    assert len(signal["evidence"]) == 3
    assert "recorded" in signal["explanation"]
    trajectory.apply(event(7, "tool_call", call_id="edit", tool="apply_patch"))
    trajectory.apply(event(8, "tool_result", call_id="edit", result="patched"))
    check(trajectory, 9, "after", failure)
    assert not trajectory.signals["repeat:worker"]["active"]


def test_successful_equivalent_checks_are_inspection_signals_not_action_required():
    trajectory = model()
    for index in range(3):
        check(trajectory, index * 2 + 1, str(index), {"exit_code": 0, "stdout": "passed"})
    assert trajectory.signals["repeat:worker"]["severity"] == "inspect"
    assert not any(s["title"] == "Recorded call failure" for s in trajectory.signals.values())


def test_input_requests_and_dependency_waits_resolve_from_explicit_events():
    trajectory = model()
    trajectory.apply(event(1, "input_required", message="Choose the expected retry limit"))
    assert trajectory.signals["input:worker"]["severity"] == "required"
    trajectory.apply(event(2, "input_resolved"))
    assert not trajectory.signals["input:worker"]["active"]
    trajectory.apply(event(3, "request_submitted", request_id="r"))
    trajectory.apply(event(4, "request_blocked", request_id="r", dependencies=["upstream"]))
    assert trajectory.signals["wait:r"]["evidence"] == ["worker:dependency:r"]
    trajectory.apply(event(5, "request_started", request_id="r"))
    assert not trajectory.signals["wait:r"]["active"]
    assert trajectory.actions["worker:queue:r"]["duration"] == 2
    assert trajectory.actions["worker:dependency:r"]["duration"] == 1


def test_final_completion_failure_and_cancellation_preserve_unfinished_calls():
    for status in ("completed", "failed", "cancelled", "unknown"):
        trajectory = model()
        trajectory.apply(event(1, "tool_call", call_id="open", tool="Bash"))
        trajectory.apply(event(2, "done", status=status))
        assert trajectory.run["status"] == status
        assert trajectory.actions["worker:open"]["outcome"] == "incomplete"
        assert trajectory.actions["worker:open"]["result"] == ""
        trajectory.tick(100)
        assert trajectory.run["duration"] == 2


def test_replay_prefix_contains_no_future_result_usage_or_label():
    saved = {
        "actions": [
            {
                "id": "a",
                "agent": "worker",
                "kind": "test",
                "name": "Bash",
                "command": "pytest target",
                "intent": "",
                "start": 0,
                "duration": 30,
                "outcome": "failed",
                "result": "future assertion",
                "tokens": 99,
                "metadata": {"outcome": "future failure", "error": "future assertion"},
            }
        ],
        "duration": 30,
    }
    rows = replay_events(saved)
    assert "future assertion" not in json.dumps(rows[0])
    assert "future failure" not in json.dumps(rows[0])
    trajectory = model()
    trajectory.apply(rows[0])
    assert "future assertion" not in json.dumps(trajectory.snapshot())
    assert trajectory.actions["a"]["tokens"] is None
    assert (
        trajectory.episodes[trajectory.actions["a"]["episode"]]["title"] == "Checking the behavior"
    )
    trajectory.apply(rows[1])
    assert "future assertion" in json.dumps(trajectory.snapshot())


def write_rows(path, rows):
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,type TEXT,timestamp REAL,name TEXT,call_label TEXT,payload TEXT)"
    )
    count = connection.execute("SELECT count(*) FROM events").fetchone()[0]
    for index, row in enumerate(rows, count):
        connection.execute(
            "INSERT INTO events VALUES(?,?,?,?,?,?)",
            (
                f"{index:020d}",
                row["type"],
                row["ts"],
                row["actor"],
                row.get("call_label"),
                json.dumps(row["payload"]),
            ),
        )
    connection.commit()
    connection.close()


def registered_source(tmp_path):
    actor = tmp_path / "worker.sqlite3"
    write_rows(
        tmp_path / "global_data.sqlite3",
        [event(0, "workload_started"), event(1, "agent_registered", db_path=str(actor))],
    )
    write_rows(
        actor,
        [event(2, "tool_call", call_id="open", tool="Bash", arguments={"command": "pytest real"})],
    )
    return LiveSource(tmp_path), actor


def test_snapshot_subscription_race_and_reconnect_recover_exactly_once(tmp_path):
    source, actor = registered_source(tmp_path)
    source.refresh(force=True)
    snapshot = source.message()
    write_rows(actor, [event(3, "tool_result", call_id="open", result="done")])
    source.refresh(force=True)
    update = source.message(snapshot["cursor"], snapshot["epoch"])
    assert update["type"] == "updates"
    assert update["patches"][-1]["actions"][0]["outcome"] == "success"
    assert not source.message(update["cursor"], update["epoch"])["patches"]
    assert len(source.model.run["actions"]) == 1
    assert (
        source.raw_event(source.model.run["actions"][0]["event_ids"][-1])["payload"]["result"]
        == "done"
    )


def test_reconnect_after_journal_expiry_recovers_a_complete_snapshot(tmp_path):
    source, _actor = registered_source(tmp_path)
    source.journal = deque(maxlen=1)
    source.refresh(force=True)
    first = source.message()
    source.refresh(force=True)
    source.refresh(force=True)
    recovered = source.message(first["cursor"], first["epoch"])
    assert recovered["type"] == "snapshot"
    assert recovered["resynced"]
    assert recovered["run"]["actions"][0]["id"] == first["run"]["actions"][0]["id"]


def test_missing_agent_database_is_disclosed_and_retried_without_losing_cursor(tmp_path):
    path = tmp_path / "missing.sqlite3"
    write_rows(tmp_path / "global_data.sqlite3", [event(0, "agent_registered", db_path=str(path))])
    source = LiveSource(tmp_path)
    source.refresh(force=True)
    assert source.model.run["coverage"]["gaps"]
    write_rows(path, [event(1, "tool_call", call_id="a", tool="read_file")])
    source.refresh(force=True)
    assert not source.model.run["coverage"]["gaps"]
    assert len(source.model.run["actions"]) == 1


def test_many_thousand_events_are_read_in_bounded_batches_and_do_not_regroup_history(tmp_path):
    rows = [
        event(index, "tool_call", call_id=str(index), tool="read_file") for index in range(2500)
    ]
    write_rows(tmp_path / "global_data.sqlite3", rows)
    source = LiveSource(tmp_path)
    source.refresh(force=True)
    assert len(source.model.run["actions"]) == 500
    first = source.model.run["actions"][0]["episode"]
    for _ in range(5):
        source.refresh(force=True)
    assert len(source.model.run["actions"]) == 2500
    assert source.model.run["actions"][0]["episode"] == first
    assert not source.model.run["coverage"]["catching_up"]


def test_optional_semantic_suggestions_cannot_change_membership():
    assert (
        validate_suggestion({"label": "Investigating retries", "evidence": ["other-actor"]}, ["a"])
        is None
    )
    suggestion = validate_suggestion(
        {
            "label": "Investigating retries",
            "summary": "The recorded assertion failed.",
            "evidence": ["a"],
        },
        ["a"],
    )
    assert suggestion["grouping"] == "fixed actor / request / call membership"
    assert (
        validate_suggestion(
            {"label": "Successfully fixed the root cause", "evidence": ["a"]}, ["a"]
        )
        is None
    )
    assert validate_suggestion({"label": "Checking behavior", "evidence": None}, ["a"]) is None


def test_semantic_failure_keeps_deterministic_membership_and_label(monkeypatch):
    from agency.observability.agwebui.trajectory_semantics import SemanticRefiner

    trajectory = model()
    check(trajectory, 1, "a", {"exit_code": 1, "stdout": "assertion"})
    trajectory.apply(event(3, "tool_call", call_id="edit", tool="edit"))
    episode = trajectory.episodes[trajectory.actions["worker:a"]["episode"]]
    refiner = SemanticRefiner("not-configured.json")
    future = Future()
    monkeypatch.setattr(refiner.executor, "submit", lambda *args: future)
    try:
        refiner.update(trajectory)
        assert episode["annotation"] is None
        future.set_exception(RuntimeError("credentials unavailable"))
        refiner.update(trajectory)
        assert "unavailable" in trajectory.run["coverage"]["semantics"]
        assert episode["title"] == "Checking the behavior"
        assert episode["actions"] == ["worker:a"]
    finally:
        refiner.close()


def test_matching_assertions_ignore_reported_elapsed_time_and_success_is_scoped():
    trajectory = model()
    for index in range(3):
        check(
            trajectory,
            index * 2 + 1,
            str(index),
            {
                "exit_code": 1,
                "stderr": f"AssertionError: expected 3 retries\nRan 1 test in {index}.00s\nFAILED (failures=1)",
            },
        )
    assert trajectory.signals["repeat:worker"]["active"]
    check(trajectory, 7, "recovery", {"exit_code": 0, "stderr": "Ran 1 test\nOK"})
    assert not trajectory.signals["repeat:worker"]["active"]
    assert not any(s["active"] for s in trajectory.signals.values())
    assert trajectory.run["resolved"] is None


def test_explicit_model_tool_ids_link_context_without_timestamp_attribution():
    trajectory = model()
    trajectory.apply(
        event(
            1,
            "tool_call",
            call_id="a",
            tool="read_file",
            annotation={"model_tool_call_id": "tool-id"},
        )
    )
    trajectory.apply(
        event(
            2,
            "tool_call",
            actor="other",
            call_id="b",
            tool="read_file",
            annotation={"model_tool_call_id": "tool-id"},
        )
    )
    trajectory.apply(
        {
            **event(3, "model_result", tool_call_ids=["tool-id"], tokens=123),
            "call_label": "exchange",
        }
    )
    assert trajectory.actions["worker:a"]["model_id"] == "worker:model:exchange"
    assert trajectory.actions["other:b"]["model_id"] is None
    assert trajectory.actions["worker:a"]["tokens"] is None
    assert trajectory.actions["worker:model:exchange"]["tokens"] == 123


def test_parallel_equivalent_checks_do_not_form_a_consecutive_failure_alert():
    trajectory = model()
    for index in range(3):
        trajectory.apply(
            event(
                index + 1,
                "tool_call",
                call_id=str(index),
                tool="Bash",
                arguments={"command": "pytest target"},
            )
        )
    for index in range(3):
        trajectory.apply(
            event(
                index + 4,
                "tool_result",
                call_id=str(index),
                result={"exit_code": 1, "stdout": "same assertion"},
            )
        )
    assert "repeat:worker" not in trajectory.signals


def test_websocket_replays_running_calls_and_recovers_between_snapshot_and_subscription(
    tmp_path, monkeypatch
):
    from agency.observability.agwebui import server

    source, actor = registered_source(tmp_path)
    monkeypatch.setattr(server, "_run_dir", tmp_path)
    monkeypatch.setattr(server, "_trajectory_source", lambda: source)
    with TestClient(server.app) as client:
        snapshot = client.get("/api/trajectory/live").json()
        write_rows(
            actor,
            [event(3, "tool_result", call_id="open", result="completed during subscription gap")],
        )
        source.refresh(force=True)
        with client.websocket_connect(
            f"/ws/trajectory?cursor={snapshot['cursor']}&epoch={snapshot['epoch']}"
        ) as websocket:
            update = websocket.receive_json()
            assert update["type"] == "updates"
            assert (
                update["patches"][-1]["actions"][0]["result"] == "completed during subscription gap"
            )


def test_replay_socket_exposes_running_start_before_future_completion(tmp_path, monkeypatch):
    from agency.observability.agwebui import server

    monkeypatch.setattr(server, "_run_dir", tmp_path)
    with TestClient(server.app) as client:
        with client.websocket_connect(
            "/ws/trajectory?mode=replay&run=trajectory-scenarios"
        ) as websocket:
            snapshot = websocket.receive_json()
            assert snapshot["run"]["mode"] == "replay"
            assert snapshot["run"]["actions"] == []
            websocket.send_json({"type": "step"})
            websocket.receive_json()
            websocket.send_json({"type": "step"})
            update = websocket.receive_json()
            assert update["patches"][0]["actions"][0]["outcome"] == "running"
            assert not update["patches"][0]["actions"][0]["result"]
            websocket.send_json({"type": "play", "speed": 1000})
            for _ in range(10):
                update = websocket.receive_json()
                if update["replay"]["finished"]:
                    break
            assert update["replay"]["finished"]
            assert update["replay"]["clock"] == 90
            assert update["patches"][0]["meta"]["duration"] == 90


def test_canonical_producer_flushes_start_and_result_while_logger_is_open(tmp_path):
    from types import SimpleNamespace

    from agency.agpolicy import agpolicy
    from agency.configs.agconfig import agconfig
    from agency.engine.host_servers.host_interaction_server import HostInteractionServer
    from agency.observability.agdatalogger import agDataLogger

    def logger(path, actor):
        config = agconfig()
        config.data_logger.db_path = str(path)
        log = agDataLogger(config, default_name=actor)
        log.start()
        return log

    global_log = logger(tmp_path / "global_data.sqlite3", "workflow")
    actor_log = logger(tmp_path / "worker.sqlite3", "worker")
    try:
        global_log.record_event("workload_started", {"task": "Check behavior"}, flush=True)
        global_log.record_event(
            "agent_registered", {"db_path": actor_log.db_path}, name="worker", flush=True
        )
        source = LiveSource(tmp_path)
        source.refresh(force=True)
        assert source.model.run["status"] == "running"
        assert source.model.run["actions"] == []
        host = HostInteractionServer(
            SimpleNamespace(policy=agpolicy()),
            actor_log,
            "worker",
            profile_attributes={"request_id": "request-1"},
        )
        admission = host.admit_tool_call(
            "Bash",
            {"command": "pytest target"},
            {"status": "valid", "raw": {"purpose": "Checking retry behavior"}},
        )
        source.refresh(force=True)
        action = source.model.run["actions"][0]
        assert action["outcome"] == "running"
        assert action["request_id"] == "request-1"
        assert source.model.episodes[action["episode"]]["title"] == "Checking retry behavior"
        host.complete_tool_call(admission["call_id"], {"exit_code": 1, "stdout": "assertion"})
        source.refresh(force=True)
        assert action["outcome"] == "failed"
        assert "assertion" in action["result_preview"]
        assert source.raw_event(action["event_ids"][-1])["payload"]["result"]["exit_code"] == 1
        actor_log.record_event(
            "agent_state", {"state": "waiting_llm"}, call_label="model-1", flush=True
        )
        source.refresh(force=True)
        assert source.model.actions["worker:model:model-1"]["outcome"] == "running"
        actor_log.record_llm_exchange(
            "model-1",
            exchange_type="llm_block",
            prompt_chain=[
                ("prompt", {"role": "user", "type": "text", "text": "Check the result"}),
                (
                    "hidden-prompt",
                    {"role": "assistant", "type": "thinking", "text": "hidden reasoning"},
                ),
            ],
            response_chain=[
                ("response", {"type": "text", "text": "Visible reply"}),
                ("hidden", {"type": "thinking", "text": "hidden reasoning"}),
                (
                    "usage",
                    {"type": "metadata", "usage": {"prompt_tokens": 17, "completion_tokens": 4}},
                ),
            ],
        )
        source.refresh(force=True)
        invocation = source.model.actions["worker:model:model-1"]
        assert invocation["result"] == "Visible reply"
        assert invocation["tokens"] == 17
        assert not invocation["context"]["comparison_available"]
        assert "hidden reasoning" not in json.dumps(source.model.snapshot())
        assert "hidden reasoning" not in json.dumps(source.raw_event(invocation["event_ids"][-1]))
    finally:
        actor_log.stop()
        global_log.stop()


def test_optional_suggestion_preserves_original_label_and_primary_membership(monkeypatch):
    from agency.observability.agwebui.trajectory_semantics import SemanticRefiner

    trajectory = model()
    check(trajectory, 1, "a", {"exit_code": 0, "stdout": "1 passed"})
    trajectory.apply(event(3, "tool_call", call_id="edit", tool="edit"))
    episode = trajectory.episodes[trajectory.actions["worker:a"]["episode"]]
    future = Future()
    future.set_result(
        validate_suggestion(
            {"label": "Checking retry behavior", "summary": "1 passed", "evidence": ["worker:a"]},
            ["worker:a"],
        )
    )
    refiner = SemanticRefiner("not-configured.json")
    monkeypatch.setattr(refiner.executor, "submit", lambda *args: future)
    try:
        refiner.update(trajectory)
        refiner.update(trajectory)
        assert episode["annotation"]["label"] == "Checking retry behavior"
        assert episode["title"] == "Checking the behavior"
        assert episode["actions"] == ["worker:a"]
    finally:
        refiner.close()


def test_skill_failure_and_cancellation_are_explicit_and_never_task_success():
    for status in ("error", "cancelled"):
        trajectory = model()
        trajectory.apply(event(1, "skill_start", request_id="r", skill="check"))
        trajectory.apply(
            event(
                2, f"skill_{status}", request_id="r", error="failed" if status == "error" else None
            )
        )
        action = trajectory.actions["worker:skill:r"]
        assert action["outcome"] == ("failed" if status == "error" else "cancelled")
        if status == "error":
            trajectory.apply(event(3, "skill_call", request_id="r", output={"error": "failed"}))
            assert action["outcome"] == "failed"
        assert trajectory.run["resolved"] is None


def test_review_keeps_run_relative_timing_and_tool_tokens_unattributed():
    trajectory = Trajectory("review", "Review", mode="review")
    trajectory.apply(event(1, "tool_call", ts=12, call_id="a", tool="Bash"))
    trajectory.apply(event(2, "tool_result", ts=20, call_id="a", result="returned", tokens=999))
    assert trajectory.actions["worker:a"]["start"] == 12
    assert trajectory.actions["worker:a"]["duration"] == 8
    assert trajectory.actions["worker:a"]["tokens"] is None


def test_scale_fixture_retains_five_actor_streams_and_bounded_episode_membership():
    from agency.observability.agwebui.server import _scenario_trajectory

    run = _scenario_trajectory("trajectory-scale")
    assert run["source"] == "synthetic"
    assert len(run["actions"]) == 5000
    assert run["coverage"]["events"] == 10002
    assert run["duration"] == 50
    assignments = [action for episode in run["episodes"] for action in episode["actions"]]
    assert len(assignments) == len(set(assignments)) == 5000
    assert max(len(episode["actions"]) for episode in run["episodes"]) == 8


def test_signals_update_incrementally_without_resending_history_on_clock_ticks():
    trajectory = model()
    trajectory.apply(event(1, "input_required", message="Choose a limit"))
    patch = trajectory.patch()
    assert patch["signals"][0]["active"]
    assert "signals" not in patch["meta"]
    trajectory.tick(10)
    assert trajectory.patch()["signals"] == []
    trajectory.apply(event(2, "input_resolved"))
    assert not trajectory.patch()["signals"][0]["active"]


def test_late_end_recovers_missing_evidence_without_rewriting_final_execution_state():
    trajectory = model()
    trajectory.apply(event(1, "tool_call", call_id="a", tool="Bash"))
    trajectory.apply(event(3, "done", status="cancelled"))
    assert trajectory.signals["unfinished"]["active"]
    trajectory.apply(event(2, "tool_result", call_id="a", result="late retained output"))
    assert trajectory.run["status"] == "cancelled"
    assert trajectory.actions["worker:a"]["result"] == "late retained output"
    assert not trajectory.signals["unfinished"]["active"]


def test_pause_resume_projects_authoritative_state_and_freezes_running_calls():
    trajectory = model()
    trajectory.apply(event(1, "agent_state", state="waiting_llm", call_label="unused"))
    trajectory.apply(event(2, "tool_call", call_id="open", tool="Bash"))
    trajectory.apply(event(3, "tool_call", actor="other", call_id="other", tool="Bash"))
    action = trajectory.actions["worker:open"]
    episode = trajectory.episodes[action["episode"]]
    trajectory.patch()
    trajectory.apply(event(10, "agent_paused"))
    patch = trajectory.patch()
    assert patch["agents"][0]["status"] == "paused"
    assert trajectory.snapshot()["agents"][0]["status"] == "paused"
    assert action["duration"] == action["execution_duration"] == 8
    frozen_end = episode["end"]
    trajectory.apply(event(11, "agent_paused"))  # Repeated controls do not double-count.
    trajectory.apply(event(12, "agent_state", state="executing_tool"))
    trajectory.tick(20)
    assert trajectory.agents["worker"]["status"] == "paused"
    assert action["duration"] == action["execution_duration"] == 8
    assert episode["end"] == frozen_end
    assert action["outcome"] == episode["status"] == "running"
    assert trajectory.actions["other:other"]["duration"] == 17
    trajectory.apply(event(21, "agent_resumed"))
    assert trajectory.agents["worker"]["status"] == "executing_tool"
    assert trajectory.patch()["agents"][0]["status"] == "executing_tool"
    trajectory.tick(25)
    assert action["duration"] == 23  # Trace axis retains wall-clock boundaries.
    assert action["execution_duration"] == 12
    trajectory.apply(event(26, "agent_paused"))
    trajectory.tick(30)
    assert action["execution_duration"] == 13
    trajectory.apply(event(31, "agent_resumed"))
    trajectory.apply(event(35, "tool_result", call_id="open", result="done"))
    assert action["execution_duration"] == 17
    assert action["started_ts"] == 2
    assert action["end_ts"] == 35
    assert action["duration"] == 33


def test_pause_state_ignores_stale_controls_and_finishes_cleanly():
    trajectory = model()
    trajectory.apply(event(1, "agent_state", state="waiting_llm"))
    trajectory.apply(event(10, "agent_paused"))
    trajectory.apply(event(5, "agent_resumed"))
    assert trajectory.agents["worker"]["status"] == "paused"
    trajectory.apply(event(11, "agent_resumed"))
    assert trajectory.agents["worker"]["status"] == "waiting_llm"
    trajectory.apply(event(12, "agent_paused"))
    trajectory.apply(event(13, "done", status="completed"))
    assert trajectory.agents["worker"]["status"] == "completed"
    trajectory.apply(event(14, "agent_paused"))
    assert trajectory.agents["worker"]["status"] == "completed"


def test_live_source_reads_pause_resume_into_snapshots_and_patches(tmp_path):
    source, actor = registered_source(tmp_path)
    write_rows(actor, [event(3, "agent_paused")])
    source.refresh(force=True)
    snapshot = source.message()
    agent = snapshot["run"]["agents"][0]
    assert agent["status"] == "paused"
    assert snapshot["run"]["actions"][0]["execution_duration"] == 1
    write_rows(actor, [event(4, "agent_resumed")])
    source.refresh(force=True)
    update = source.message(snapshot["cursor"], snapshot["epoch"])
    assert update["patches"][-1]["agents"][0]["status"] != "paused"
    assert update["patches"][-1]["agents"][0]["control_ts"] == 4


def test_live_profiler_commands_use_existing_dispatch_bridge(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agency.agent import agent
    from agency.observability.agwebui import _dispatch_command, server

    source, _actor = registered_source(tmp_path)
    source.refresh(force=True)
    source.model.apply(event(3, "agent_registered", actor="other"))
    command_dir = tmp_path / "ui_commands"
    command_dir.mkdir()
    (command_dir / ".heartbeat").touch()
    monkeypatch.setattr(server, "_run_dir", tmp_path)
    monkeypatch.setattr(server, "_command_dir", command_dir)
    monkeypatch.setattr(server, "_trajectory_source", lambda: source)
    called = []
    agents = [
        SimpleNamespace(
            agname=name,
            pause=lambda n=name: called.append((n, "pause")),
            resume=lambda n=name: called.append((n, "resume")),
        )
        for name in ("worker", "other")
    ]
    monkeypatch.setattr(agent, "all", classmethod(lambda cls: agents))
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/trajectory") as websocket:
            assert websocket.receive_json()["execution_controls"]["available"]
            for command in ("pause", "resume", "pause_all", "resume_all"):
                websocket.send_json(
                    {
                        "type": "execution_command",
                        "command": command,
                        "agname": "worker",
                        "id": command,
                    }
                )
                while (result := websocket.receive_json())["type"] != "execution_command_result":
                    pass
                assert result == {"type": "execution_command_result", "id": command, "queued": True}
                files = list(command_dir.glob("*.json"))
                assert len(files) == 1
                _dispatch_command(json.loads(files[0].read_text()))
                files[0].unlink()
            # The trajectory socket is still streaming after execution controls.
            assert websocket.receive_json()["type"] == "updates"
    assert called == [
        ("worker", "pause"),
        ("worker", "resume"),
        ("worker", "pause"),
        ("other", "pause"),
        ("worker", "resume"),
        ("other", "resume"),
    ]


def test_live_execution_command_failures_do_not_queue_commands(tmp_path, monkeypatch):
    from agency.observability.agwebui import server

    source, _ = registered_source(tmp_path)
    source.refresh(force=True)
    commands = tmp_path / "ui_commands"
    commands.mkdir()
    monkeypatch.setattr(server, "_command_dir", commands)
    message = {"id": "one", "type": "execution_command", "command": "pause", "agname": "worker"}
    assert "error" in server._trajectory_execution_command(source, message)
    heartbeat = commands / ".heartbeat"
    heartbeat.touch()
    assert "error" in server._trajectory_execution_command(source, {**message, "agname": "missing"})
    assert "error" in server._trajectory_execution_command(source, {**message, "command": "play"})

    def fail_write(_command):
        raise OSError("disk unavailable")

    monkeypatch.setattr(server, "_queue_command", fail_write)
    assert "Could not deliver" in server._trajectory_execution_command(source, message)["error"]
    source.model.apply(event(4, "done", status="completed"))
    assert "error" in server._trajectory_execution_command(source, message)
    assert not list(commands.glob("*.json"))


def test_replay_pause_and_execution_commands_never_reach_live_bridge(tmp_path, monkeypatch):
    from agency.observability.agwebui import server

    monkeypatch.setattr(server, "_run_dir", tmp_path)
    queued = []
    monkeypatch.setattr(server, "_queue_command", queued.append)
    with TestClient(server.app) as client:
        with client.websocket_connect(
            "/ws/trajectory?mode=replay&run=trajectory-scenarios"
        ) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "play", "speed": 0.1})
            assert websocket.receive_json()["replay"]["playing"]
            websocket.send_json({"type": "pause"})
            while (update := websocket.receive_json())["replay"]["playing"]:
                pass
            assert not update["replay"]["playing"]
            websocket.send_json({"type": "execution_command", "command": "pause_all", "id": "bad"})
            while (response := websocket.receive_json())["type"] != "execution_command_result":
                pass
            assert "during replay" in response["error"]
            assert queued == []
