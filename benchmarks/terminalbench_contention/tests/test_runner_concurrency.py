"""Concurrency validation with mocked workers: 6, 12 and 24 agents.

Real scheduling loop, real worker processes, real shared ledger and real
artifact pipeline; no container and no model.
"""

import json

import pytest

from benchmarks.terminalbench_contention import config as C
from benchmarks.terminalbench_contention import runner as R
from benchmarks.terminalbench_contention.api_budget import read_ledger, verify_ledger
from benchmarks.terminalbench_contention.util import (
    REPOSITORY,
    load_env_file,
    read_json,
    read_jsonl,
)

from .conftest import make_runner, synthetic_manifest


def concurrency_profile(events):
    """Maximum simultaneously released-and-not-exited agents, from the event log."""
    active = peak = 0
    for event in events:
        if event["kind"] == "release":
            active += 1
            peak = max(peak, active)
        elif event["kind"] == "exit" and event["from_state"] not in ("setup", "ready"):
            active -= 1
    return peak


@pytest.mark.parametrize("agents", [6, 12, 24])
def test_condition_runs_every_task_once_within_its_concurrency_limit(tmp_path, agents):
    run, recorder = make_runner(tmp_path, agents=agents)
    assert run.execute() == R.EXIT_OK
    summary = read_json(run.run_dir / "summary.json")
    record = read_json(run.run_dir / "run.json")
    # Fixed corpus: all 24 tasks, each exactly once.
    task_ids = [t["task_id"] for t in summary["tasks"]]
    assert sorted(task_ids) == sorted(t["task_id"] for t in synthetic_manifest()["tasks"])
    assert len(set(task_ids)) == 24 and summary["counts"]["completed"] == 24
    assert sorted(recorder.collected) == list(range(1, 25))
    assert len(recorder.graded) == len(set(recorder.graded)) == 24
    # Launch order is the planned, seeded order.
    assert [t["task_id"] for t in summary["tasks"]] == run.entry["launch_order"]
    launches = [
        e["task"] for e in read_jsonl(run.run_dir / "events.jsonl") if e["kind"] == "launch"
    ]
    assert launches == run.entry["launch_order"]
    # Maximum concurrency is enforced and reached.
    events = read_jsonl(run.run_dir / "events.jsonl")
    assert concurrency_profile(events) == agents
    assert summary["concurrency"] == dict(
        maximum_allowed=agents, peak_in_flight=agents, peak_executing=agents
    )
    assert record["peak_in_flight"] <= agents
    # The first wave is released together, after all of it is ready.
    releases = [e for e in events if e["kind"] == "release"]
    readies = [e for e in events if e["kind"] == "ready"][:agents]
    assert len({e["t"] // 10**7 for e in releases[:agents]}) <= 2
    assert max(e["t"] for e in readies) <= releases[0]["t"]
    assert record["measured_start_ns"] >= max(e["t"] for e in readies)
    # CPU pool consistency: every worker saw exactly the experimental pool.
    pool = sorted(run.layout.pool_cpus)
    assert not set(pool) & set(run.layout.reserved_cpus)
    for task in summary["tasks"]:
        state = read_json(run.run_dir / "agents" / task["directory"] / "task.json")
        assert state["host_affinity"] == pool
    # Shared budget consistency across the separate worker processes.
    ledger = verify_ledger(run.run_dir / "api-metrics" / "ledger")
    assert ledger["ok"] and ledger["admitted"] == 48 and not ledger["unsettled"]
    pids = {e["pid"] for e in read_ledger(run.run_dir / "api-metrics" / "ledger")}
    assert len(pids) == 24
    assert summary["ledger_ok"] and summary["api"]["admissions"] == 48
    # Isolation and cleanup.
    assert summary["isolation"]["cross_agent_writable_mounts"] == {}
    assert summary["isolation"]["all_verified"]
    assert summary["isolation"]["leaked_containers"] == []
    assert all(slot.handle.poll() is not None for slot in run.slots)
    owners = {
        (run.run_dir / "agents" / t["directory"] / "workspace" / "owner.txt").read_text()
        for t in summary["tasks"]
    }
    assert len(owners) == 24  # each agent wrote only its own workspace
    assert summary["makespan_seconds"] > 0 and summary["status"] == "completed"


def test_all_conditions_use_one_corpus_one_order_and_one_policy(tmp_path):
    runs = {}
    for agents in (6, 12, 24):
        run, _ = make_runner(tmp_path, agents=agents, run_name=f"c{agents}", sample=False)
        assert run.execute() == R.EXIT_OK
        runs[agents] = run
    orders = {a: read_json(r.run_dir / "plan.json")["launch_order"] for a, r in runs.items()}
    assert orders[6] == orders[12] == orders[24]
    metadata = {a: read_json(r.run_dir / "metadata.json") for a, r in runs.items()}
    assert len({m["order_seed"] for m in metadata.values()}) == 1
    assert len({m["api_policy_fingerprint"] for m in metadata.values()}) == 1
    assert len({json.dumps(m["cpu"]["layout"]["pool_cpus"]) for m in metadata.values()}) == 1
    assert len({json.dumps(m["timeouts"], sort_keys=True) for m in metadata.values()}) == 1
    assert len({json.dumps(m["cache_control"], sort_keys=True) for m in metadata.values()}) == 1
    assert {m["agents"] for m in metadata.values()} == {6, 12, 24}


def test_repetitions_are_isolated_and_use_recorded_seeds(tmp_path):
    first, _ = make_runner(tmp_path, agents=6, repetition=1, run_name="r1", sample=False)
    second, _ = make_runner(tmp_path, agents=6, repetition=2, run_name="r2", sample=False)
    assert first.execute() == R.EXIT_OK and second.execute() == R.EXIT_OK
    a, b = read_json(first.run_dir / "metadata.json"), read_json(second.run_dir / "metadata.json")
    assert a["order_seed"] != b["order_seed"] and a["launch_order"] != b["launch_order"]
    assert a["order_seed"] == C.repetition_seed(first.config, 1)
    assert b["order_seed"] == C.repetition_seed(second.config, 2)
    # Fresh ledger, fresh agent directories, nothing shared between runs.
    ledgers = [read_ledger(r.run_dir / "api-metrics" / "ledger") for r in (first, second)]
    assert all(events[0]["seq"] == 1 for events in ledgers)
    assert not {e["request_id"] + str(e["pid"]) for e in ledgers[0]} & {
        e["request_id"] + str(e["pid"]) for e in ledgers[1]
    }
    assert first.run_dir != second.run_dir
    assert (
        not set(p.name for p in (first.run_dir / "agents").iterdir())
        & set(p.name for p in (second.run_dir / "agents").iterdir())
        or a["launch_order"] != b["launch_order"]
    )
    for run in (first, second):
        assert (
            read_json(run.run_dir / "api-metrics" / "ledger" / "state.json")["run_id"] == run.run_id
        )


def test_a_run_directory_is_never_reused(tmp_path):
    run, _ = make_runner(tmp_path, agents=6, tasks=6, sample=False)
    assert run.execute() == R.EXIT_OK
    again, _ = make_runner(tmp_path, agents=6, tasks=6, sample=False)
    with pytest.raises(FileExistsError):
        again.execute()
    assert R.next_attempt(tmp_path / "label").name == "attempt-01"
    (tmp_path / "label" / "attempt-01").mkdir(parents=True)
    assert R.next_attempt(tmp_path / "label").name == "attempt-02"


def test_results_are_complete_and_machine_readable(tmp_path):
    run, _ = make_runner(tmp_path, agents=6, tasks=8)
    assert run.execute() == R.EXIT_OK
    for relative in (
        "config.json",
        "metadata.json",
        "plan.json",
        "status.json",
        "run.json",
        "summary.json",
        "events.jsonl",
        "api-metrics/ledger/ledger.jsonl",
        "api-metrics/ledger/state.json",
        "api-metrics/verification.json",
        "api-metrics/summary.json",
        "resource-metrics/samples.jsonl",
        "resource-metrics/capabilities.json",
        "resource-metrics/summary.json",
        "resource-metrics/sampler.json",
        "traces/trajectories.jsonl",
        "traces/timeline.jsonl",
        "traces/cross-resource.json",
        "traces/run.trace.json",
        "traces/index.json",
        "grading/verdicts.json",
    ):
        path = run.run_dir / relative
        assert path.is_file() and path.stat().st_size > 0, relative
        if path.suffix == ".json":
            read_json(path)
        else:
            assert read_jsonl(path, strict=True)
    metadata = read_json(run.run_dir / "metadata.json")
    for key in (
        "git",
        "dataset",
        "corpus_sha256",
        "task_digests",
        "harbor_version",
        "model",
        "cpu",
        "memory",
        "api_budget",
        "cache_control",
        "order_seed",
        "launch_order",
        "timeouts",
        "agent",
        "clock",
        "dependency_locks_sha256",
        "images",
    ):
        assert metadata[key] is not None, key
    # Credentials: only the names of the environment variables are stored.
    text = json.dumps(metadata) + (run.run_dir / "config.json").read_text()
    assert "api_key" not in metadata["model"] and "base_url" not in metadata["model"]
    assert metadata["credential_sources"]["api_key_env"] == "LITELLM_API_KEY"
    secrets = [v for v in load_env_file(REPOSITORY / ".env").values() if len(v) >= 8]
    assert not any(secret in text for secret in secrets)
    summary = read_json(run.run_dir / "summary.json")
    assert summary["counts"]["tasks"] == 8 and summary["denominators"]
    for task in summary["tasks"]:
        assert set(task["state_seconds"]) == set(
            (
                "tool",
                "model_stream",
                "model_wait",
                "api_wait",
                "checkpoint",
                "harness_startup",
                "other",
            )
        )
        total = sum(task["state_seconds"].values())
        assert total == pytest.approx(task["wall_seconds"], abs=1e-6)
        assert task["verdict"]["reward"] == 1.0 and task["api"]["requests"] == 2
    trajectory = read_jsonl(run.run_dir / "traces" / "trajectories.jsonl")
    assert {row["slot"] for row in trajectory} == set(range(1, 9))
    assert all(row["source"].startswith(("measured", "inferred")) for row in trajectory)
    assert read_json(run.run_dir / "status.json")["status"] == "completed"


def test_shared_budget_contention_is_visible_across_worker_processes(tmp_path):
    """A tight window forces admission waits that span separate processes."""
    run, _ = make_runner(
        tmp_path,
        agents=6,
        tasks=6,
        assignments=[
            "api.tokens_per_window=700",
            "api.window_seconds=0.5",
            "api.poll_interval_seconds=0.01",
        ],
        behaviour=dict(requests=3, request_tokens=150),
    )
    assert run.execute() == R.EXIT_OK
    report = verify_ledger(run.run_dir / "api-metrics" / "ledger")
    assert report["ok"] and report["peak_window_tokens"] <= 700 and report["admitted"] == 18
    summary = read_json(run.run_dir / "summary.json")
    waits = summary["api"]["admission_wait_seconds"]
    assert waits["waited"] > 0 and waits["total"] > 0
    assert summary["api"]["blocked_by"].get("token_budget", 0) > 0
    assert sum(t["state_seconds"]["api_wait"] for t in summary["tasks"]) > 0
    assert sum(t["waiting"]["api_admission_seconds"] for t in summary["tasks"]) == pytest.approx(
        waits["total"], rel=0.05, abs=0.05
    )
