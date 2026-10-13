"""Failure injection: the runner must fail safely and keep its diagnostics."""

import json
import threading
import time

import pytest

from benchmarks.terminalbench_contention import analysis, topology
from benchmarks.terminalbench_contention import runner as R
from benchmarks.terminalbench_contention.api_budget import verify_ledger
from benchmarks.terminalbench_contention.util import read_json, read_json_or, read_jsonl

from .conftest import Recorder, make_runner


def by_slot(summary):
    return {task["slot"]: task for task in summary["tasks"]}


def test_worker_termination_is_recorded_and_the_run_continues(tmp_path):
    run, recorder = make_runner(
        tmp_path, agents=3, tasks=6, behaviour=dict(by_slot={"2": dict(die="running")})
    )
    assert run.execute() == R.EXIT_DEGRADED
    summary = read_json(run.run_dir / "summary.json")
    tasks = by_slot(summary)
    assert tasks[2]["status"] == "worker_died" and tasks[2]["censored"]
    assert tasks[2]["worker_returncode"] == 9 and not tasks[2]["collection_ok"]
    assert summary["counts"]["completed"] == 5 and summary["counts"]["infrastructure_failures"] == 1
    assert summary["infrastructure_failed_tasks"] == [tasks[2]["task_id"]]
    # No retry and no rescheduling: the dead task ran once and stays failed.
    launches = [
        e["slot"] for e in read_jsonl(run.run_dir / "events.jsonl") if e["kind"] == "launch"
    ]
    assert launches == [1, 2, 3, 4, 5, 6]
    assert 2 not in recorder.collected and len(recorder.graded) == 5
    assert summary["status"] == "completed" and summary["exit_code"] == R.EXIT_DEGRADED


def test_setup_failure_in_the_first_wave_stops_before_anything_is_measured(tmp_path):
    run, recorder = make_runner(
        tmp_path, agents=3, tasks=6, behaviour=dict(by_slot={"1": dict(fail_setup=True)})
    )
    assert run.execute() == R.EXIT_PRECONDITION
    record = read_json(run.run_dir / "run.json")
    assert record["status"] == "not_started" and record["measured_start_ns"] is None
    assert "first-wave setup failed" in record["error"]
    summary = read_json(run.run_dir / "summary.json")
    statuses = {t["slot"]: t["status"] for t in summary["tasks"]}
    assert statuses[1] == "setup_failed"
    assert all(statuses[s] == "not_started" for s in (4, 5, 6))  # never launched
    assert summary["counts"]["completed"] == 0 and recorder.collected == []
    assert all(slot.handle is None or slot.handle.poll() is not None for slot in run.slots)
    assert verify_ledger(run.run_dir / "api-metrics" / "ledger")["admitted"] == 0


def test_setup_failure_during_refill_is_an_infrastructure_failure_not_an_abort(tmp_path):
    run, _ = make_runner(
        tmp_path, agents=2, tasks=5, behaviour=dict(by_slot={"4": dict(fail_setup=True)})
    )
    assert run.execute() == R.EXIT_DEGRADED
    summary = read_json(run.run_dir / "summary.json")
    tasks = by_slot(summary)
    assert tasks[4]["status"] == "setup_failed" and tasks[4]["start_ns"] is None
    assert summary["counts"]["completed"] == 4 and tasks[5]["status"] == "completed"
    assert summary["counts"]["by_status"] == dict(completed=4, setup_failed=1)


def test_worker_dying_during_setup_is_detected(tmp_path):
    run, _ = make_runner(
        tmp_path, agents=2, tasks=4, behaviour=dict(by_slot={"3": dict(die="setup")})
    )
    assert run.execute() == R.EXIT_DEGRADED
    task = by_slot(read_json(run.run_dir / "summary.json"))[3]
    assert task["status"] == "setup_failed" and task["worker_returncode"] == 9


def test_task_timeout_is_censored_not_dropped_and_still_graded(tmp_path):
    run, recorder = make_runner(
        tmp_path,
        agents=2,
        tasks=3,
        assignments=["timeouts.task_seconds=0.6"],
        behaviour=dict(by_slot={"1": dict(duration=30)}),
    )
    assert run.execute() == R.EXIT_OK
    summary = read_json(run.run_dir / "summary.json")
    tasks = by_slot(summary)
    assert tasks[1]["status"] == "timeout" and tasks[1]["censored"] and not tasks[1]["completed"]
    assert tasks[1]["wall_seconds"] == pytest.approx(0.6, abs=0.3)
    assert tasks[1]["collection_ok"] and tasks[1]["verdict"]["ok"]  # partial work is graded
    assert summary["counts"]["censored"] == 1 and summary["counts"]["completed"] == 2
    assert summary["counts"]["tasks"] == 3 and 1 in recorder.collected
    assert summary["rewards"]["passed_censored"] == 1
    assert summary["rewards"]["passed_completed"] == 2


def test_worker_that_ignores_its_deadline_is_terminated_by_the_runner(tmp_path):
    run, _ = make_runner(
        tmp_path,
        agents=2,
        tasks=2,
        assignments=["timeouts.task_seconds=0.4"],
        behaviour=dict(by_slot={"1": dict(ignore_deadline=True)}),
    )
    code = run.execute()
    summary = read_json(run.run_dir / "summary.json")
    task = by_slot(summary)[1]
    assert task["status"] == "cancelled" and task["censored"]
    assert any("SIGTERM sent by the runner" in note for note in task["notes"])
    assert code == R.EXIT_OK and by_slot(summary)[2]["status"] == "completed"


def test_worker_that_ignores_sigterm_is_killed_and_reported(tmp_path):
    run, _ = make_runner(
        tmp_path,
        agents=1,
        tasks=1,
        assignments=["timeouts.task_seconds=0.4"],
        behaviour=dict(ignore_deadline=True, ignore_sigterm=True),
    )
    assert run.execute() == R.EXIT_DEGRADED
    task = read_json(run.run_dir / "summary.json")["tasks"][0]
    assert task["status"] == "worker_died" and task["censored"]
    assert any("killed by the runner" in note for note in task["notes"])
    assert all(slot.handle.poll() is not None for slot in run.slots)


def test_operator_stop_cancels_agents_and_leaves_the_rest_not_started(tmp_path):
    stop = tmp_path / "STOP"
    run, recorder = make_runner(
        tmp_path, agents=2, tasks=6, stop_file=stop, behaviour=dict(duration=20)
    )
    timer = threading.Timer(1.5, stop.write_text, args=("stop",))
    timer.start()
    try:
        code = run.execute()
    finally:
        timer.cancel()
    assert code == R.EXIT_INVALID
    record = read_json(run.run_dir / "run.json")
    assert record["status"] == "aborted" and "STOP file" in record["error"]
    summary = read_json(run.run_dir / "summary.json")
    statuses = [t["status"] for t in summary["tasks"]]
    assert statuses.count("cancelled") == 2 and statuses.count("not_started") == 4
    assert all(t["censored"] for t in summary["tasks"])
    assert summary["counts"]["not_started"] == 4 and summary["status"] == "aborted"
    # Cancelled agents were still handed off, collected and graded; nothing leaked.
    assert sorted(recorder.collected) == [1, 2]
    assert all(slot.handle is None or slot.handle.poll() is not None for slot in run.slots)
    assert verify_ledger(run.run_dir / "api-metrics" / "ledger")["ok"]


def test_signal_to_the_runner_stops_the_run_safely(tmp_path):
    import os
    import signal

    run, _ = make_runner(tmp_path, agents=2, tasks=4, behaviour=dict(duration=20))
    timer = threading.Timer(1.5, os.kill, args=(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        code = run.execute()
    finally:
        timer.cancel()
    assert code == R.EXIT_INTERRUPTED
    record = read_json(run.run_dir / "run.json")
    assert record["status"] == "aborted" and "signal" in record["error"]
    assert signal.getsignal(signal.SIGTERM) is not run._on_signal  # handlers restored


def test_run_time_limit_censors_and_stops(tmp_path):
    run, _ = make_runner(
        tmp_path,
        agents=2,
        tasks=4,
        assignments=["timeouts.run_seconds=1.0", "timeouts.task_seconds=30"],
        behaviour=dict(duration=20),
    )
    assert run.execute() == R.EXIT_INVALID
    summary = read_json(run.run_dir / "summary.json")
    assert "run time limit" in summary["error"]
    assert summary["counts"]["censored"] == 4 and summary["counts"]["completed"] == 0


def test_collection_failure_is_reported_without_losing_the_run(tmp_path):
    run, recorder = make_runner(tmp_path, agents=2, tasks=3, recorder=Recorder(fail_collect={2}))
    assert run.execute() == R.EXIT_OK
    summary = read_json(run.run_dir / "summary.json")
    task = by_slot(summary)[2]
    assert task["status"] == "completed" and not task["collection_ok"]
    assert "verdict" not in task and summary["counts"]["collected"] == 2
    assert (
        read_json(run.run_dir / "agents" / task["directory"] / "collected.json")["error"]
        == "injected collection failure"
    )
    assert len(recorder.graded) == 2


@pytest.mark.parametrize("mode", ["fail_grade", "raise_grade"])
def test_grading_failure_is_distinct_from_a_failed_task(tmp_path, mode):
    run, _ = make_runner(tmp_path, agents=2, tasks=3, recorder=Recorder(**{mode: {3}}))
    assert run.execute() == R.EXIT_DEGRADED
    summary = read_json(run.run_dir / "summary.json")
    task = by_slot(summary)[3]
    assert task["status"] == "completed" and task["verdict"]["ok"] is False
    assert task["verdict"]["reward"] is None  # not silently scored as zero
    assert summary["counts"]["grading_failures"] == 1 and summary["counts"]["graded"] == 2
    assert summary["grading_failed_tasks"] == [task["task_id"]]
    assert summary["rewards"]["mean_reward_graded"] == 1.0
    assert summary["rewards"]["mean_reward_corpus"] == pytest.approx(2 / 3)
    assert read_json(run.run_dir / "grading" / "verdicts.json")[task["directory"]]["error"]


def test_corrupted_admission_ledger_fails_closed(tmp_path):
    """Corrupt the shared state mid-run: agents stop being admitted, nothing is reset."""
    run, _ = make_runner(
        tmp_path, agents=2, tasks=4, behaviour=dict(requests=40, model_seconds=0.02, duration=1.0)
    )
    state = run.run_dir / "api-metrics" / "ledger" / "state.json"

    def corrupt():
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            ledger = state.parent / "ledger.jsonl"
            if ledger.exists() and ledger.read_text().count("\n") >= 6:
                state.write_text("{corrupted")
                return
            time.sleep(0.01)

    thread = threading.Thread(target=corrupt)
    thread.start()
    code = run.execute()
    thread.join()
    assert code == R.EXIT_DEGRADED
    summary = read_json(run.run_dir / "summary.json")
    assert summary["ledger_ok"] is False and summary["ledger_problems"]
    assert not read_json(run.run_dir / "api-metrics" / "verification.json")["ok"]
    assert all(
        t["status"] == "agent_error" and "LedgerCorrupted" in t["error"] for t in summary["tasks"]
    )
    assert list(state.parent.glob("state.corrupt-*.json"))  # evidence preserved
    events = read_jsonl(state.parent / "ledger.jsonl")
    assert 6 <= len(events) < 4 * 40 * 2  # admission stopped; the ledger was not rebuilt


def test_partial_artifact_writes_do_not_break_analysis(tmp_path):
    run, _ = make_runner(tmp_path, agents=2, tasks=3)
    assert run.execute() == R.EXIT_OK
    record = read_json(run.run_dir / "run.json")
    agents = run.run_dir / "agents"
    first, second, third = [agents / t["directory"] for t in record["tasks"]]
    # A torn JSON file, a torn JSONL tail and a missing file.
    (first / "task.json").write_text('{"status": "comp')
    with (second / "api-requests.jsonl").open("a") as stream:
        stream.write('{"kind": "request", "ready_ns": 12')
    (third / "api-requests.jsonl").unlink()
    with (run.run_dir / "resource-metrics" / "samples.jsonl").open("a") as stream:
        stream.write('{"t": 99')
    assert read_json_or(first / "task.json") is None
    summary = analysis.finalize(run.run_dir, record, {}, run.config)
    assert summary["counts"]["tasks"] == 3 and len(summary["tasks"]) == 3
    tasks = by_slot(summary)
    assert tasks[2]["api"]["requests"] == 2  # the complete lines are still used
    assert tasks[3]["api"]["requests"] == 0
    json.dumps(summary)


def test_missing_final_record_is_reported_as_a_dead_worker(tmp_path):
    run, _ = make_runner(tmp_path, agents=1, tasks=1)
    run.prepare_directories()
    slot = run.slots[0]
    slot.directory.mkdir(parents=True)
    (slot.directory / "task.json").write_text('{"status": "running", "start_ns": 5')
    slot.state, slot.released_ns, slot.returncode = "exited", 5, -9
    [record] = run.collect_task_records()
    assert record["status"] == "worker_died" and record["censored"]
    assert record["completed"] is False and record["worker_returncode"] == -9


def test_insufficient_cpu_resources_block_the_run(tmp_path):
    rows = [dict(cpu=i, core=i, package=0, node=0, siblings=[i], max_khz=None) for i in range(4)]
    with pytest.raises(topology.InsufficientCpuError, match="4 online physical cores"):
        topology.build_layout(rows)


def test_shared_writable_state_between_agents_is_flagged(tmp_path):
    shared = tmp_path / "shared-cache"
    run, _ = make_runner(tmp_path, agents=2, tasks=2, behaviour=dict(write_outside=str(shared)))
    run.execute()
    summary = read_json(run.run_dir / "summary.json")
    assert str(shared) in summary["isolation"]["cross_agent_writable_mounts"]
    assert not summary["isolation"]["all_verified"]


def test_only_one_run_may_hold_the_run_lock(tmp_path):
    with R.RunLock(tmp_path / "run.lock"):
        with pytest.raises(R.PreconditionError, match="run lock"):
            with R.RunLock(tmp_path / "run.lock"):
                pass
    with R.RunLock(tmp_path / "run.lock"):
        pass


def test_low_disk_space_stops_before_measurement(tmp_path):
    run, _ = make_runner(
        tmp_path, agents=2, tasks=2, assignments=["memory.min_free_disk_gb=1000000"]
    )
    assert run.execute() == R.EXIT_PRECONDITION
    assert "free disk space" in read_json(run.run_dir / "run.json")["error"]
