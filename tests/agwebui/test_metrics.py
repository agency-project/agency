import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from agency.observability.agwebui.metrics import ExecutionCounters


def test_counters_replace_call_contribution_on_completion_and_replay():
    counters = ExecutionCounters()
    call = dict(id="a", kind="tool", outcome="running")
    assert counters.update([call])["tools"] == 1
    call["outcome"] = "failed"
    totals = counters.update([call, call])
    assert totals["tools"] == totals["completed_tools"] == totals["tool_errors"] == 1
    call["outcome"] = "success"
    assert counters.update([call])["tool_errors"] == 0


def test_syscalls_never_count_as_model_tool_invocations_or_errors():
    counters = ExecutionCounters()
    totals = counters.update([
        {"id": "send", "kind": "syscall", "outcome": "running", "agent": "a"},
        {"id": "exec", "kind": "syscall", "outcome": "failed", "agent": "a"},
        {"id": "tool", "kind": "tool", "outcome": "success", "agent": "a"},
    ])
    assert totals["tools"] == totals["completed_tools"] == 1
    assert totals["tool_errors"] == 0
    assert counters.by_agent["a"]["tools"] == 1


def test_partial_usage_and_wait_intervals():
    counters = ExecutionCounters()
    totals = counters.update(
        [
            dict(id="a", kind="model", outcome="success", tokens=0),
            dict(id="b", kind="queue", outcome="success"),
        ]
    )
    assert totals["models"] == totals["input_samples"] == 1
    assert totals["output_samples"] == totals["tools"] == 0


def test_wall_clock_waits_for_agents_and_includes_pauses_until_completion():
    from agency.observability.agwebui.trajectory import Trajectory

    trajectory = Trajectory("live", "Clock regression")

    def apply(index, kind, ts, **payload):
        trajectory.apply(dict(id=str(index), type=kind, ts=ts, actor="worker", payload=payload))

    apply(1, "agent_created", 100)
    apply(2, "workload_started", 110)
    apply(3, "resource_sample", 150, counters={"workload:memory_mb": 12})
    trajectory.tick(200)
    assert trajectory.run["execution_wall_time"] == 0
    assert trajectory.run["execution_started_at"] is None
    apply(4, "request_submitted", 205, request_id="r")
    assert trajectory.run["execution_wall_time"] == 0
    apply(5, "request_started", 210, request_id="r")
    trajectory.tick(220)
    assert trajectory.run["execution_wall_time"] == 10
    apply(6, "agent_paused", 220)
    trajectory.tick(240)
    assert trajectory.run["execution_wall_time"] == 30
    apply(7, "agent_resumed", 245)
    apply(8, "done", 250, status="completed")
    trajectory.tick(300)
    assert trajectory.run["execution_wall_time"] == 40
    assert trajectory.run["duration"] == 150
    meta = trajectory.patch()["meta"]
    assert meta["execution_started_at"] == 210
    assert meta["execution_wall_time"] == 40


def test_wall_clock_without_agent_execution_stays_zero_after_cancel():
    from agency.observability.agwebui.trajectory import Trajectory

    trajectory = Trajectory("live", "Never launched")
    for index, kind in enumerate(["workload_started", "done"]):
        trajectory.apply(
            dict(
                id=str(index),
                type=kind,
                ts=index * 100,
                actor="webui",
                payload={"status": "cancelled"},
            )
        )
    assert trajectory.snapshot()["execution_wall_time"] == 0


def _clock(title="Measurement"):
    from agency.observability.agwebui.trajectory import Trajectory

    trajectory = Trajectory("live", title)

    def apply(index, kind, ts, actor="worker", **payload):
        trajectory.apply(dict(id=str(index), type=kind, ts=ts, actor=actor, payload=payload))

    return trajectory, apply


def _user(identifier, kind, **fields):
    return dict(id=identifier, type=kind, recorded_at=0, **fields)


def test_wall_clock_ends_at_workload_completion_not_profiler_finalization():
    trajectory, apply = _clock()
    apply(1, "workload_started", 200, actor="webui")
    apply(2, "request_started", 210, request_id="r")
    apply(3, "workload_completed", 250, actor="webui", status="completed")
    trajectory.tick(256)  # Trace export and summaries are still being written.
    assert trajectory.run["execution_wall_time"] == 40
    assert trajectory.run["status"] == "running"
    apply(4, "done", 257, actor="webui", status="completed", finalization_seconds=7)
    trajectory.tick(300)
    measurement = trajectory.patch()["meta"]["measurement"]
    assert trajectory.run["execution_wall_time"] == 40
    assert trajectory.run["duration"] == 57  # The trace axis still covers finalization.
    assert measurement["state"] == "ended" and measurement["policy"] == "automatic"
    assert measurement["automatic_end_at"] == measurement["effective_end_at"] == 250
    assert measurement["automatic_end_source"] == "workload_completed"
    assert measurement["end_source"] == "workload_done" and measurement["end_offset"] == 50
    assert "webui" not in trajectory.agents


def test_done_alone_recovers_the_callback_boundary_or_falls_back_for_legacy_runs():
    # workload_completed was lost (or profiler cleanup failed before it flushed).
    trajectory, apply = _clock()
    apply(1, "request_started", 210, request_id="r")
    apply(2, "done", 257, status="failed", finalization_seconds=7)
    assert trajectory.run["execution_wall_time"] == 40
    assert trajectory.run["measurement"]["automatic_end_source"] == "done"
    for invalid in (None, -1, float("nan"), "7", True):
        legacy, apply = _clock()
        apply(1, "request_started", 210, request_id="r")
        apply(2, "done", 257, status="completed", finalization_seconds=invalid)
        legacy.tick(900)
        assert legacy.run["execution_wall_time"] == 47
        assert legacy.run["measurement"]["state"] == "ended"


def test_failed_cancelled_and_crashed_work_still_produce_one_completion_boundary():
    for status in ("failed", "cancelled"):
        trajectory, apply = _clock()
        apply(1, "request_started", 210, request_id="r")
        apply(2, "request_failed", 220, request_id="r", error="agent crashed")
        trajectory.tick(230)
        # An agent failing is not the workload returning.
        assert trajectory.run["measurement"]["state"] == "running"
        assert trajectory.run["execution_wall_time"] == 20
        apply(3, "workload_completed", 240, actor="webui", status=status)
        apply(4, "done", 241, actor="webui", status=status, finalization_seconds=1)
        assert trajectory.run["execution_wall_time"] == 30
        assert trajectory.run["status"] == status


def test_idle_gaps_and_concurrent_agents_do_not_end_the_automatic_measurement():
    trajectory, apply = _clock()
    apply(1, "request_started", 100, actor="first", request_id="a")
    apply(2, "request_started", 101, actor="second", request_id="b")
    apply(3, "request_completed", 110, actor="first", request_id="a")
    apply(4, "request_completed", 112, actor="second", request_id="b")
    trajectory.tick(150)  # Every agent is idle between sequential skill requests.
    assert trajectory.run["measurement"]["state"] == "running"
    assert trajectory.run["measurement"]["automatic_end_at"] is None
    assert trajectory.run["execution_wall_time"] == 50
    apply(5, "request_started", 160, actor="third", request_id="c")
    apply(6, "request_completed", 170, actor="third", request_id="c")
    apply(7, "workload_completed", 175, actor="webui", status="completed")
    assert trajectory.run["execution_started_at"] == 100
    assert trajectory.run["execution_wall_time"] == 75


def test_manual_end_leaves_execution_running_and_reset_restores_the_automatic_boundary():
    trajectory, apply = _clock()
    apply(1, "request_started", 100, request_id="r")
    apply(2, "tool_call", 105, call_id="open", tool="Bash")
    assert trajectory.apply_measurement(_user("end-1", "end", at=120, source="user"))
    assert not trajectory.apply_measurement(_user("end-1", "end", at=999))  # Repeat is inert.
    trajectory.tick(130)
    measurement = trajectory.run["measurement"]
    assert trajectory.run["execution_wall_time"] == 20
    assert measurement["state"] == "ended" and measurement["policy"] == "manual"
    assert measurement["end_source"] == "user" and measurement["automatic_end_at"] is None
    # Agents and the trace are untouched by the measurement boundary.
    assert trajectory.finished_at is None and trajectory.run["status"] == "running"
    assert trajectory.actions["worker:open"]["outcome"] == "running"
    apply(3, "tool_result", 140, call_id="open", result="ok")
    assert trajectory.actions["worker:open"]["outcome"] == "success"
    assert trajectory.actions["worker:open"]["duration"] == 35
    apply(4, "workload_completed", 150, actor="webui", status="completed")
    assert measurement["automatic_end_at"] == 150 and measurement["effective_end_at"] == 120
    assert trajectory.run["execution_wall_time"] == 20
    trajectory.apply_measurement(_user("reset-1", "reset"))
    assert measurement["policy"] == "automatic" and measurement["end_source"] == "workload_done"
    assert trajectory.run["execution_wall_time"] == 50


def test_keep_running_counts_continuously_past_completion_until_ended_or_reset():
    trajectory, apply = _clock()
    apply(1, "request_started", 100, request_id="r")
    apply(2, "workload_completed", 120, actor="webui", status="completed")
    apply(3, "done", 121, actor="webui", status="completed")
    trajectory.tick(200)
    assert trajectory.run["execution_wall_time"] == 20
    trajectory.apply_measurement(_user("hold-1", "hold"))
    # The interval since automatic completion is included, not skipped.
    assert trajectory.run["execution_wall_time"] == 100
    trajectory.tick(260)
    measurement = trajectory.run["measurement"]
    assert trajectory.run["execution_wall_time"] == 160
    assert measurement["state"] == "running" and measurement["held"]
    assert measurement["policy"] == "manual" and measurement["effective_end_at"] is None
    assert measurement["automatic_end_at"] == 120  # The original boundary is kept.
    assert trajectory.run["duration"] == 21 and trajectory.run["status"] == "completed"
    # A later boundary chosen after completion, without restarting anything.
    trajectory.apply_measurement(_user("end-2", "end", at=250, source="user"))
    trajectory.tick(400)
    assert trajectory.run["execution_wall_time"] == 150 and measurement["end_offset"] == 150
    assert measurement["state"] == "ended" and not measurement["held"]
    trajectory.apply_measurement(_user("reset-2", "reset"))
    assert trajectory.run["execution_wall_time"] == 20


def test_a_replay_prefix_has_not_reached_a_later_recorded_boundary():
    from agency.observability.agwebui.trajectory import Trajectory

    replay = Trajectory("live", "Replay", mode="replay")
    replay.apply(dict(id="1", type="request_started", ts=10, actor="worker", payload={}))
    replay.apply_measurement(_user("end-1", "end", at=40, source="user"))
    replay.tick(25)
    assert replay.run["execution_wall_time"] == 15
    assert replay.run["measurement"]["state"] == "running"
    replay.tick(90)
    assert replay.run["execution_wall_time"] == 30
    assert replay.run["measurement"]["state"] == "ended"


@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled", "finalization_failed"])
def test_webui_records_the_boundary_at_callback_return_before_finalization(
    monkeypatch, tmp_path, outcome
):
    """Slow or failing profiler finalization must not move or lose the boundary."""
    import agency.observability.agwebui as ui
    import agency.orchestrator as orchestrator
    from agency.observability.agwebui.trajectory import Trajectory

    recorded, finalization = [], 0.25

    class Logger:
        def record_event(self, kind, payload, **_):
            recorded.append((kind, dict(payload), time.time()))

        def flush(self):
            pass

    fake = SimpleNamespace(agresource_pool=None, data_logger=Logger())
    monkeypatch.setattr(orchestrator, "get_orchestrator", lambda **_: fake)

    @contextmanager
    def workload(*, out_dir):
        try:
            yield
        finally:
            time.sleep(finalization)  # Trace writing, summaries, telemetry flush.
            if outcome == "finalization_failed":
                raise RuntimeError("trace export failed")

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def setsockopt(self, *args):
            pass

        def connect_ex(self, address):
            return 1

    process = SimpleNamespace(poll=lambda: None, terminate=lambda: None, wait=lambda timeout: 0)
    monkeypatch.setattr(ui.agprof, "workload", workload)
    monkeypatch.setattr(ui.agprof, "active_output_dir", lambda: None)
    monkeypatch.setattr(ui.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(ui.urllib.request, "urlopen", lambda *args, **kwargs: None)
    monkeypatch.setattr("socket.socket", lambda *args, **kwargs: Socket())
    monkeypatch.setattr(ui.atexit, "register", lambda fn: None)
    returned = []

    def fn():
        recorded.append(("request_started", {"request_id": "r"}, time.time()))
        time.sleep(0.05)
        returned.append(time.time())
        if outcome == "failed":
            raise ValueError("task raised")
        if outcome == "cancelled":
            raise KeyboardInterrupt

    def run():
        ui.agwebui.run(fn, run_dir=tmp_path / "logs", port=17861, linger=False)

    if outcome == "cancelled":
        with pytest.raises(KeyboardInterrupt):
            run()
    else:
        run()

    kinds = [kind for kind, _, _ in recorded]
    assert kinds == ["workload_started", "request_started", "workload_completed", "done"]
    completed, done = recorded[2], recorded[3]
    callback = "completed" if outcome == "finalization_failed" else outcome
    assert completed[1] == {"status": callback}
    assert 0 <= completed[2] - returned[0] < 0.1
    assert done[2] - completed[2] >= finalization
    assert done[1]["finalization_seconds"] == pytest.approx(done[2] - completed[2], abs=0.05)
    # Lifecycle status is unchanged: a cleanup failure is still reported there.
    assert done[1]["status"] == ("failed" if outcome == "finalization_failed" else outcome)

    for events in (recorded, [row for row in recorded if row[0] != "workload_completed"]):
        trajectory = Trajectory("live", "Recorded")
        for index, (kind, payload, ts) in enumerate(events):
            trajectory.apply(dict(id=str(index), type=kind, ts=ts, actor="webui", payload=payload))
        wall = trajectory.run["execution_wall_time"]
        assert wall == pytest.approx(0.05, abs=0.04)
        assert wall < finalization


def test_counters_keep_additive_totals_per_agent_and_follow_call_updates():
    counters = ExecutionCounters()
    first = dict(id="m1", agent="a", kind="model", outcome="success", tokens=100, output_tokens=20)
    counters.update(
        [
            first,
            dict(id="t1", agent="a", kind="tool", outcome="failed"),
            dict(id="m2", agent="b", kind="model", outcome="success", tokens=7),
            dict(id="q", agent="b", kind="queue", outcome="success"),
        ]
    )
    assert counters.by_agent["a"]["input"] == 100 and counters.by_agent["a"]["output"] == 20
    assert counters.by_agent["a"]["tools"] == counters.by_agent["a"]["tool_errors"] == 1
    assert counters.by_agent["b"]["input"] == 7 and counters.by_agent["b"]["output_samples"] == 0
    # A replayed or completed call replaces its own contribution, per agent too.
    first["tokens"] = 150
    totals = counters.update([first, first])
    assert counters.by_agent["a"]["input"] == 150 and counters.by_agent["a"]["models"] == 1
    for key, value in totals.items():
        assert value == sum(agent[key] for agent in counters.by_agent.values())


def test_patches_carry_per_agent_totals_and_the_recorded_sandbox_of_each_agent():
    from agency.observability.agwebui.trajectory import Trajectory

    trajectory = Trajectory("live", "Scope regression")

    def apply(index, kind, actor, **payload):
        trajectory.apply(
            dict(id=str(index), type=kind, ts=100 + index, actor=actor, payload=payload)
        )

    apply(1, "agent_created", "a")
    apply(2, "agent_created", "b")
    apply(3, "agent_state", "a", state="running_harness", sandbox="sandbox_a_0000")
    apply(4, "agent_state", "b", state="running_harness")
    apply(5, "tool_call", "a", call_id="one", name="bash")
    apply(6, "tool_call", "b", call_id="two", name="bash")
    apply(7, "tool_call", "b", call_id="three", name="bash")
    patch = trajectory.patch()
    assert patch["meta"]["metrics_totals"]["tools"] == 3
    assert {k: v["tools"] for k, v in patch["meta"]["metrics_by_agent"].items()} == {"a": 1, "b": 2}
    agents = {agent["id"]: agent for agent in trajectory.run["agents"]}
    assert agents["a"]["sandbox"] == "sandbox_a_0000"
    # No link is invented from names when the engine did not record one.
    assert "sandbox" not in agents["b"]
