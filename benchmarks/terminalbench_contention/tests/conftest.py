"""Shared fixtures. Nothing here contacts a model or the network."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from benchmarks.terminalbench_contention import config as config_module
from benchmarks.terminalbench_contention import runner as runner_module
from benchmarks.terminalbench_contention import topology
from benchmarks.terminalbench_contention.placement import ProcessLauncher
from benchmarks.terminalbench_contention.util import REPOSITORY

ENV = REPOSITORY / "artifacts" / "terminalbench-contention" / "env"
TIKTOKEN_CACHE = ENV / "tiktoken-cache"
if TIKTOKEN_CACHE.is_dir() and "TIKTOKEN_CACHE_DIR" not in os.environ:
    os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_CACHE)

MOCK_WORKER = "benchmarks.terminalbench_contention.mock_worker"


def prepared_environment() -> bool:
    return (ENV / "prepared.json").exists() and shutil.which("podman") is not None


requires_prepared = pytest.mark.skipif(
    not prepared_environment(), reason="needs `agency-bench prepare` and podman"
)


class Clock:
    """Deterministic clock whose sleep advances time."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def read(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return Clock()


def synthetic_manifest(count: int = 24) -> dict:
    tasks = [
        dict(
            task_id=f"mock/task-{i:02d}",
            directory=f"task-{i:02d}",
            digest=f"sha256:{i:064d}",
            environment=dict(image=f"registry.invalid/env@sha256:{i:064d}", memory_mb=1024),
            verifier=dict(image=f"registry.invalid/ver@sha256:{i:064d}"),
        )
        for i in range(count)
    ]
    return dict(tasks=tasks, corpus_sha256="synthetic", task_count=count)


def mock_config(*assignments, tasks: int = 24) -> dict:
    return config_module.load(
        smoke=True,
        assignments=[
            "timeouts.task_seconds=15",
            "timeouts.setup_seconds=20",
            "timeouts.stop_grace_seconds=1",
            "timeouts.collect_seconds=10",
            "profiling.host_sample_seconds=0.2",
            "memory.min_free_disk_gb=0.05",
            "memory.host_min_available_mb=1",
            "api.minimum_request_spacing_seconds=0",
            f"benchmark.tasks={tasks}",
            *assignments,
        ],
    )


class Recorder:
    """Collector and grader doubles that remember what they were asked to do."""

    def __init__(self, *, fail_collect=(), fail_grade=(), raise_grade=()):
        self.collected, self.graded = [], []
        self.fail_collect, self.fail_grade = set(fail_collect), set(fail_grade)
        self.raise_grade = set(raise_grade)

    def collect(self, slot, container, run_dir):
        self.collected.append(slot.index)
        if slot.index in self.fail_collect:
            return dict(ok=False, error="injected collection failure")
        return dict(ok=True, container=container, artifacts=[])

    def grade(self, name, task, source, run_dir):
        self.graded.append(name)
        index = int(name.split("-", 1)[0])
        if index in self.raise_grade:
            raise RuntimeError("injected grader crash")
        if index in self.fail_grade:
            return dict(ok=False, error="injected grading failure")
        return dict(ok=True, reward=1.0, rewards={"reward": 1.0})


def make_runner(
    tmp_path: Path,
    *,
    agents: int = 6,
    tasks: int = 24,
    repetition: int = 1,
    behaviour=None,
    assignments=(),
    recorder=None,
    run_name="run",
    sample=True,
    stop_file=None,
    layout=None,
    worker_module=MOCK_WORKER,
):
    """A Runner over mocked workers: real scheduling, real ledger, no containers."""
    config = mock_config(*assignments, tasks=tasks)
    manifest = synthetic_manifest(tasks)
    identifiers = [t["task_id"] for t in manifest["tasks"]]
    condition = next((n for n, a in config["conditions"].items() if a == agents), None)
    if condition is None:
        config["conditions"] = {f"TEST-{agents}": agents}
        condition = f"TEST-{agents}"
    entry = config_module.plan_entry(config, identifiers, condition, repetition)
    layout = layout or topology.detect_layout(config["cpu"])
    recorder = recorder or Recorder()
    prepared = dict(
        harness_python="/nonexistent",
        tiktoken_cache=os.environ.get("TIKTOKEN_CACHE_DIR"),
        tasks={t: dict(instruction_file="/dev/null", task_dir="/dev/null") for t in identifiers},
    )
    launcher = ProcessLauncher(
        pool_cpus=layout.pool_cpus, working_directory=REPOSITORY, pythonpath=str(REPOSITORY)
    )
    run = runner_module.Runner(
        config,
        manifest,
        entry,
        prepared=prepared,
        layout=layout,
        run_dir=tmp_path / run_name,
        launcher=launcher,
        collector=recorder.collect,
        grader=recorder.grade,
        worker_module=worker_module,
        poll_seconds=0.02,
        freeze_source=False,
        sample=sample,
        stop_file=stop_file,
        job_extra=dict(
            mock_worker={
                **dict(duration=0.15, requests=2, model_seconds=0.01, tool_seconds=0.01),
                **(behaviour or {}),
            }
        ),
    )
    return run, recorder
