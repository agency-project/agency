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
