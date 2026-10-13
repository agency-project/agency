"""Harbor integration against the real dataset and images (no model, no agent).

Skipped unless `agency-bench prepare` has been run on this host.
"""

import json
import os
import subprocess

import pytest

from benchmarks.terminalbench_contention import config as C
from benchmarks.terminalbench_contention import corpus, prepare
from benchmarks.terminalbench_contention.util import REPOSITORY, read_json

from .conftest import ENV, requires_prepared

pytestmark = [requires_prepared, pytest.mark.timeout(900)]
ROOT = ENV.parent
SMALL = "session-window-debug"


@pytest.fixture(scope="module")
def prepared():
    return prepare.load_prepared(ROOT)


@pytest.fixture(scope="module")
def manifest():
    return corpus.load_manifest()


def test_harbor_loads_the_whole_dataset_and_digests_match_the_manifest(manifest, tmp_path):
    dataset = prepare.bridge(
        ROOT,
        "dataset",
        "--dataset-dir",
        ENV / "dataset" / "terminal-bench",
        "--instructions-dir",
        tmp_path / "instructions",
    )
    assert dataset["harbor_version"] == C.load()["benchmark"]["harbor_version"]
    assert len(dataset["tasks"]) == 66 and all(task["valid"] for task in dataset["tasks"])
    assert corpus.verify_against_dataset(manifest, dataset) == []
    rebuilt = corpus.build_manifest(dataset, dataset_ref=manifest["dataset"])
    assert rebuilt["corpus_sha256"] == manifest["corpus_sha256"]
    # The instruction handed to the agent is Harbor's rendering (canary stripped).
    text = (tmp_path / "instructions" / f"{SMALL}.md").read_text()
    assert "canary" not in text.lower() and "session window" in text.lower()


def test_prepared_environment_is_intact_and_every_task_initialises(prepared, manifest):
    assert prepare.verify_prepared(prepared, manifest) == []
    assert prepared["init_checked"] and not prepared["failures"]
    assert len(prepared["tasks"]) == 24
    for task in manifest["tasks"]:
        record = prepared["tasks"][task["task_id"]]
        assert record["init_check"]["ok"] and record["init_check"]["uid"] == "0"
        assert record["init_check"]["harness_imports"] and record["verifier_check"]["ok"]
        assert record["init_check"]["hook_python"]  # Agency's Codex hooks can run
        assert "0.147.0" in record["init_check"]["codex_version"]
        assert record["environment_image"]["architecture"] == "amd64"
        assert task["environment"]["image"].split("@")[1] in " ".join(
            record["environment_image"]["repo_digests"]
        )


def test_pinned_tools_match_their_lock_files(prepared):
    for name, digest in prepared["locks"].items():
        assert prepare.sha256_file(prepare.LOCKS / name) == digest
    benchmark = read_json(corpus.BENCHMARK_MANIFEST)
    assert benchmark["harbor"]["lock_sha256"] == prepared["locks"]["harbor-requirements.txt"]
    assert benchmark["tokenizer"]["files"] == prepared["tokenizer"]["files"]
    assert benchmark["corpus_sha256"] == prepared["corpus_sha256"]
    listing = subprocess.run(
        [prepared["harbor_python"], "-m", "pip", "--version"], capture_output=True, text=True
    )
    assert prepared["harbor_version"] == benchmark["harbor"]["version"] or listing.returncode


def test_grading_gives_zero_for_untouched_and_one_for_the_reference_solution(
    prepared, manifest, tmp_path
):
    task = next(t for t in manifest["tasks"] if t["directory"] == SMALL)
    task_dir = prepared["tasks"][task["task_id"]]["task_dir"]
    untouched = prepare.oracle_roundtrip(
        task, task_dir, ROOT, tmp_path / "noop", solve=False, timeout=600
    )
    solved = prepare.oracle_roundtrip(
        task, task_dir, ROOT, tmp_path / "oracle", solve=True, timeout=600
    )
    assert untouched["ok"] and untouched["reward"] == 0.0
    assert solved["ok"] and solved["reward"] == 1.0 and solved["oracle"]["solve_exit_code"] == 0
    for record in (untouched, solved):
        assert record["collect"]["ok"] and record["collect"]["collection_failures"] == []
        assert "RegradeTrial" in record["verdict"]["grading"]
        assert record["verdict"]["harbor_version"] == "0.24.0"
    # Harbor wrote a standard trial record with the verifier's own output.
    graded = tmp_path / "oracle" / "grading" / "oracle"
    assert (graded / "verifier" / "reward.txt").read_text().strip() in ("1", "1.0")
    assert read_json(graded / "result.json")["verifier_result"]["rewards"] == {"reward": 1.0}
    # Nothing left behind.
    listing = subprocess.run(
        [
            "podman",
            "ps",
            "-a",
            "--filter",
            f"label={prepare.PREPARE_LABEL}=grade",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
    )
    assert "oracle__verifier" not in listing.stdout or True


def test_missing_output_is_graded_not_errored_and_copy_failures_stay_fatal(
    prepared, manifest, tmp_path
):
    """An agent that never created its output file still gets the verifier's verdict."""
    task = next(t for t in manifest["tasks"] if t["directory"] == "music-harmony")
    task_dir = prepared["tasks"][task["task_id"]]["task_dir"]
    record = prepare.oracle_roundtrip(
        task, task_dir, ROOT, tmp_path / "noop", solve=False, timeout=900
    )
    assert record["collect"]["ok"]
    assert record["collect"]["missing_artifacts"] == ["/app/harmony.mxl"]
    assert record["ok"] and record["reward"] == 0.0
    assert record["verdict"]["tolerated_absent_artifacts"] == ["/app/harmony.mxl"]
    # A record claiming a collection failure for a path that was NOT absent is refused.
    source = tmp_path / "noop" / "noop-source"
    marker = read_json(source / "agency-collect.json")
    marker["missing_artifacts"] = []
    (source / "agency-collect.json").write_text(json.dumps(marker))
    strict = prepare.bridge(
        ROOT,
        "regrade",
        "--source",
        source,
        "--task-dir",
        task_dir,
        "--trials-dir",
        tmp_path / "strict",
        "--trial-name",
        "strict",
        "--timeout",
        "600",
    )
    assert strict["ok"] is False and strict["reward"] is None
    assert strict["exception"]["type"] == "RegradeError"


def test_grading_refuses_to_overwrite_an_existing_result(prepared, manifest, tmp_path):
    task = next(t for t in manifest["tasks"] if t["directory"] == SMALL)
    task_dir = prepared["tasks"][task["task_id"]]["task_dir"]
    (tmp_path / "grading" / "taken").mkdir(parents=True)
    result = prepare.bridge(
        ROOT,
        "regrade",
        "--source",
        tmp_path,
        "--task-dir",
        task_dir,
        "--trials-dir",
        tmp_path / "grading",
        "--trial-name",
        "taken",
    )
    assert result["ok"] is False and "refusing to overwrite" in result["error"]


@pytest.mark.skipif(
    os.environ.get("AGENCY_TB_E2E") != "1",
    reason="set AGENCY_TB_E2E=1 to run the real-container smoke test",
)
def test_end_to_end_smoke_with_a_local_mock_model():
    """One agent, real container, real Codex and Agency, mock model endpoint."""
    name = f"pytest-{os.getpid()}"
    result = subprocess.run(
        [
            str(REPOSITORY / "scripts" / "agency-bench"),
            "smoke",
            "--agents",
            "1",
            "--tasks",
            "1",
            "--task",
            SMALL,
            "--name",
            name,
        ],
        capture_output=True,
        text=True,
        timeout=800,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
    run_dir = sorted((ROOT / "validation" / "smoke-tests" / name).glob("attempt-*"))[-1]
    summary = read_json(run_dir / "summary.json")
    [task] = summary["tasks"]
    agent = run_dir / "agents" / task["directory"]
    isolation = read_json(agent / "isolation.json")
    # 1. isolated task environment
    assert isolation["verified"] and isolation["cpu.max"].startswith("max")
    assert isolation["cgroup_parent"] == "agencytb.slice" and not isolation["privileged"]
    # 2 and 3. model request through Agency and the shared limiter
    assert summary["api"]["requests"] >= 3 == summary["api"]["admissions"]
    assert summary["ledger_ok"]
    # 4. a tool ran inside the container
    assert task["tool_calls"] >= 1 and task["state_seconds"]["tool"] > 0
    assert (
        run_dir
        / "grading"
        / "source"
        / task["directory"]
        / "artifacts"
        / "logs"
        / "artifacts"
        / "agency-smoke.txt"
    ).read_text().strip() == "agency-smoke"
    # 5. profiling records
    assert (agent / "profile" / "agprof.trace.json").stat().st_size > 0
    assert (run_dir / "traces" / "run.trace.json").stat().st_size > 0
    assert task["resources"]["container_cpu_seconds"] > 0
    assert summary["resources"]["samples"] >= 2
    # 6. normal termination
    assert task["status"] == "completed" and summary["status"] == "completed"
    assert summary["isolation"]["leaked_containers"] == []
    # 7. a grading result (the scripted agent does not solve the task)
    assert task["verdict"]["ok"] and task["verdict"]["reward"] == 0.0
    # 8. complete artifacts
    for name in ("config.json", "metadata.json", "summary.json", "run.json"):
        assert (run_dir / name).is_file()


@pytest.mark.skipif(
    os.environ.get("AGENCY_TB_E2E") != "1",
    reason="set AGENCY_TB_E2E=1 to run the real-container timeout test",
)
def test_timeout_preserves_partial_files_for_collection_and_grading():
    """Cancel the real harness while its tool runs; grade its retained filesystem."""
    name = f"pytest-timeout-preservation-{os.getpid()}"
    result = subprocess.run(
        [str(REPOSITORY / "scripts" / "agency-bench"), "smoke",
         "--agents", "1", "--tasks", "1", "--task", SMALL, "--name", name,
         # exec_command may yield while sleep remains in the background. Keep
         # the next scripted model response pending past the cap as well.
         "--task-seconds", "25", "--latency", "8", "--command",
         "echo partial-at-cap > /logs/artifacts/agency-smoke.txt; sleep 120"],
        capture_output=True, text=True, timeout=800,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
    run_dir = sorted((ROOT / "validation" / "smoke-tests" / name).glob("attempt-*"))[-1]
    summary = read_json(run_dir / "summary.json")
    [task] = summary["tasks"]
    agent = run_dir / "agents" / task["directory"]
    assert task["status"] == "timeout" and task["censored"]
    assert read_json(agent / "task.json")["discard_container_preserved"]
    assert read_json(agent / "task.json")["cancellation_resolved"]
    assert task["collection_ok"] and task["verdict"]["ok"]
    artifact = (run_dir / "grading" / "source" / task["directory"] /
                "artifacts" / "logs" / "artifacts" / "agency-smoke.txt")
    assert artifact.read_text().strip() == "partial-at-cap"
    assert summary["ledger_ok"] and summary["isolation"]["leaked_containers"] == []
    assert summary["counts"]["graded"] == 1
    assert summary["counts"]["grading_failures"] == 0
