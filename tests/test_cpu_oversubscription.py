"""Pure policy tests; no subprocesses, Docker, models, or host affinity changes."""

import json
import signal
from types import SimpleNamespace

import pytest

from benchmarks.cpu_oversubscription.config import assignments, load_config, validate_manifest
from benchmarks.cpu_oversubscription.linux import layout_from_rows
from benchmarks.cpu_oversubscription.metrics import reconcile, summarize


def manifest():
    # Synthetic IDs are test fixtures only, never an executable workload.
    return {
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "dataset_revision": "a" * 40,
        "agents": [
            [
                {
                    "instance_id": f"fixture-{agent}-{index}",
                    "repo": "fixture/repo",
                    "version": "1",
                    "base_commit": "a" * 40,
                    "problem_statement": "fixture",
                    "image": "sha256:" + "b" * 64,
                }
                for index in range(5)
            ]
            for agent in range(6)
        ],
    }


def test_deadline_keeps_all_assignments_and_censors_only_started_tasks():
    rows = assignments(manifest())
    events = [
        {"attempt_id": "a0-t0", "kind": "started", "time": 110},
        {"attempt_id": "a0-t0", "kind": "finished", "time": 120, "returncode": 0},
        {"attempt_id": "a1-t0", "kind": "started", "time": 130},
    ]
    result = reconcile(rows, events, 100, 1900, "deadline")
    result[0]["success"] = True
    assert result[5]["status"] == "timed_out"
    assert result[5]["duration_seconds"] == 1770
    assert result[5]["censored"] is True
    assert result[1]["status"] == "unattempted"
    assert result[1]["reason"] == "deadline"
    assert result[1]["duration_seconds"] is None
    metrics = summarize(result, 1810, 1800)
    assert metrics["pass_rate"] == 1 / 30
    assert metrics["average_task_wall_seconds"] == 890
    assert metrics["throughput_attempts_per_hour"] == 2
    assert metrics["timed_out_tasks"] == 29


@pytest.mark.parametrize("finish", [1900, 1900.001, 2000])
def test_cleanup_cannot_turn_a_deadline_interruption_into_completion(finish):
    rows = reconcile(
        assignments(manifest()),
        [
            {"attempt_id": "a0-t0", "kind": "started", "time": 100},
            {"attempt_id": "a0-t0", "kind": "finished", "time": finish, "returncode": 0},
        ],
        100,
        1900,
        "deadline",
    )
    assert rows[0]["censored"]
    assert rows[0]["duration_seconds"] == 1800
    assert summarize(rows, 1800, 1800)["throughput_attempts_per_hour"] == 0


def test_start_at_deadline_remains_unattempted():
    rows = reconcile(
        assignments(manifest()),
        [{"attempt_id": "a0-t0", "kind": "started", "time": 1900}],
        100,
        1900,
        "deadline",
    )
    assert rows[0]["status"] == "unattempted"
    assert summarize(rows, 1800, 1800)["average_task_wall_seconds"] is None


def test_nonzero_exit_is_a_completed_attempt_but_grading_stays_pending():
    rows = reconcile(
        assignments(manifest()),
        [
            {"attempt_id": "a0-t0", "kind": "started", "time": 10},
            {"attempt_id": "a0-t0", "kind": "finished", "time": 20, "returncode": 1},
        ],
        10,
        30,
        "worker_failure",
    )
    metrics = summarize(rows, 20, 1800)
    assert metrics["completed_attempts"] == 1
    assert metrics["pass_rate"] is None
    assert metrics["throughput_attempts_per_hour"] == 180


def test_each_condition_uses_its_own_start_time():
    def condition(start):
        return reconcile(
            assignments(manifest()),
            [{"attempt_id": "a0-t0", "kind": "started", "time": start + 100}],
            start,
            start + 1800,
            "deadline",
        )

    assert condition(100)[0]["duration_seconds"] == condition(10000)[0]["duration_seconds"] == 1700


def test_topology_uses_physical_core_identity_and_noncontiguous_cpu_ids():
    rows = [
        {"socket": 0, "core": core, "cpu": 2 * core + 3, "node": 0} for core in reversed(range(8))
    ]
    dcb = layout_from_rows(rows, "DCB")
    lsb = layout_from_rows(rows, "LSB")
    assert dcb["reserved"] == [3, 5]
    assert dcb["assignments"] == [7, 9, 11, 13, 15, 17]
    assert lsb["assignments"] == [7] * 6
    assert lsb["reserved"] == dcb["reserved"]


def test_smt_siblings_are_rejected():
    rows = [{"socket": 0, "core": core, "cpu": core, "node": 0} for core in range(8)]
    rows.append({"socket": 0, "core": 0, "cpu": 8, "node": 0})
    with pytest.raises(ValueError, match="SMT"):
        layout_from_rows(rows, "DCB")


def test_duplicate_tasks_require_explicit_replication():
    data = manifest()
    data["agents"] = [data["agents"][0]] * 6
    with pytest.raises(ValueError, match="distinct"):
        validate_manifest(data)
    data["replicated"] = True
    validate_manifest(data)


def test_manifest_rejects_gold_fields():
    data = manifest()
    data["agents"][0][0]["test_patch"] = "gold"
    with pytest.raises(ValueError, match="gold"):
        validate_manifest(data)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_condition_budget_is_rejected(tmp_path, timeout):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"condition": "DCB", "condition_timeout_seconds": timeout}))
    with pytest.raises(ValueError, match="positive"):
        load_config(path)


def test_default_condition_budget_is_1800_seconds(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "condition": "DCB",
                "manifest": "manifest.json",
                "dataset_snapshot": "/tmp/verified.json",
                "codex_binary": "/opt/codex",
                "codex_sha256": "a" * 64,
                "codex_version": "fixture",
            }
        )
    )
    assert load_config(path)["condition_timeout_seconds"] == 1800


def test_missing_assignments_cannot_shrink_denominator():
    with pytest.raises(ValueError, match="30"):
        summarize(assignments(manifest())[:29], 1800, 1800)


def test_termination_signals_all_descendants_then_forces_after_grace(monkeypatch, tmp_path):
    from benchmarks.cpu_oversubscription import linux

    elapsed = [0.0]
    signals = []
    kill_file = tmp_path / "cgroup.kill"
    kill_file.write_text("")
    monkeypatch.setattr(linux, "require_linux", lambda: None)
    monkeypatch.setattr(linux, "CGROUP", tmp_path)
    monkeypatch.setattr(
        linux, "processes", lambda: {11, 12} if not kill_file.read_text() else set()
    )
    monkeypatch.setattr(linux.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(
        linux,
        "time",
        SimpleNamespace(
            monotonic=lambda: elapsed[0],
            sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
        ),
    )
    linux.terminate_tree(5)
    assert set(signals) == {(11, signal.SIGTERM), (12, signal.SIGTERM)}
    assert 5 <= elapsed[0] < 5.03
    assert kill_file.read_text() == "1"


def test_watchdog_persists_deadline_before_termination(monkeypatch, tmp_path):
    from benchmarks.cpu_oversubscription import runner

    (tmp_path / "plan.json").write_text(json.dumps({"config": {"termination_grace_seconds": 5}}))
    (tmp_path / "start.json").write_text(json.dumps({"start": 0, "deadline": 1800}))
    monkeypatch.setattr(runner, "require_linux", lambda: None)
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: 1800.01))
    observed = []

    def terminate(grace):
        observed.append(json.loads((tmp_path / "end.json").read_text()))
        assert grace == 5

    monkeypatch.setattr(runner, "terminate_tree", terminate)
    runner.watchdog(tmp_path, 123)
    assert observed[0]["cutoff"] == 1800
    assert observed[0]["reason"] == "deadline"
    assert observed[0]["cleanup_complete"] is False
    assert json.loads((tmp_path / "end.json").read_text())["cleanup_complete"] is True
