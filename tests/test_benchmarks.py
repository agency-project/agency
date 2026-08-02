import json

import pytest
from agency.agconfig import agConfig

from benchmarks.backends import agency as agency_backend_module
from benchmarks.base import (
    BenchmarkResult,
    BenchmarkTask,
    ExecutionEnvironment,
    PreparedEnvironment,
)
from benchmarks.cli import ENVIRONMENTS, _select_environment
from benchmarks.environments import ContainerImageEnvironment, HostWorkspaceEnvironment
from benchmarks.providers.swe_bench import SWEBenchProvider

# ---------------------------------------------------------------------------
# Environment selection (benchmarks/cli.py)
# ---------------------------------------------------------------------------


def test_select_environment_swe_bench_uses_host_workspace():
    assert isinstance(_select_environment("swe-bench"), HostWorkspaceEnvironment)


def test_select_environment_terminal_bench_uses_container_image():
    assert isinstance(_select_environment("terminal-bench"), ContainerImageEnvironment)


def test_select_environment_unknown_provider_raises():
    with pytest.raises(ValueError):
        _select_environment("no-such-benchmark")


def test_environments_registry_covers_every_provider_choice():
    # argparse's --provider choices are sorted(ENVIRONMENTS) in cli.py -- if a
    # provider is ever added to one without the other, this catches it.
    assert set(ENVIRONMENTS) == {"swe-bench", "terminal-bench"}


# ---------------------------------------------------------------------------
# AgencyBackend <-> ExecutionEnvironment contract: the backend must run the
# agent the same way regardless of which environment it's given, and must
# call the environment's hooks in a fixed order relative to sandbox destroy.
# ---------------------------------------------------------------------------


class _FakeSandbox:
    def __init__(self):
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


class _FakeResult:
    def wait(self):
        pass

    def to_dict(self):
        return {"status": "ok", "summary": "done"}


class _FakeAgent:
    def __init__(self, agconfig):
        self.agconfig = agconfig
        self.sandbox = _FakeSandbox()

    def run(self, skill, data):
        return _FakeResult()


class _RecordingEnvironment(ExecutionEnvironment):
    def __init__(self):
        self.calls = []

    def prepare(self, cfg, task):
        self.calls.append(("prepare",))
        return PreparedEnvironment(prompt_hint="fake hint", context={"marker": "x"})

    def collect_artifacts(self, prepared, sandbox):
        assert prepared.context == {"marker": "x"}
        self.calls.append(("collect_artifacts", sandbox.destroyed))
        return {"patch": "diff --git a b", "metadata": {"env": "fake"}}

    def cleanup(self, prepared):
        self.calls.append(("cleanup",))


def test_agency_backend_runs_agent_independent_of_environment(monkeypatch):
    created_agents = []

    def fake_agent(agconfig):
        ag = _FakeAgent(agconfig)
        created_agents.append(ag)
        return ag

    captured_skill_kwargs = {}

    def fake_agskill(**kwargs):
        captured_skill_kwargs.update(kwargs)
        return kwargs

    monkeypatch.setattr(agency_backend_module, "agent", fake_agent)
    monkeypatch.setattr(agency_backend_module, "agskill", fake_agskill)
    monkeypatch.setattr(agency_backend_module, "agdata", lambda **kw: kw)

    environment = _RecordingEnvironment()
    backend = agency_backend_module.AgencyBackend(agconfig=agConfig(), environment=environment)
    task = BenchmarkTask(task_id="t1", benchmark="fake-bench", instructions="do the thing")

    result = backend.run_task(task)

    assert result.completed is True
    assert result.patch == "diff --git a b"
    assert result.metadata == {"status": "ok", "env": "fake"}
    assert "fake hint" in captured_skill_kwargs["system_prompt"]

    # collect_artifacts() must run with a still-live sandbox (destroyed=False
    # at the moment it's called), before destroy(); cleanup() runs last.
    assert environment.calls == [
        ("prepare",),
        ("collect_artifacts", False),
        ("cleanup",),
    ]
    assert created_agents[0].sandbox.destroyed is True


# ---------------------------------------------------------------------------
# SWE-bench prediction export (official-harness-format predictions.jsonl)
# ---------------------------------------------------------------------------


def test_to_prediction_defaults_missing_patch_to_empty_string():
    prediction = SWEBenchProvider.to_prediction("t1", None, "agency:claude-sonnet-5")
    assert prediction == {
        "instance_id": "t1",
        "model_name_or_path": "agency:claude-sonnet-5",
        "model_patch": "",
    }


def test_write_predictions_writes_official_harness_jsonl(tmp_path):
    records = [
        {"task_id": "repo__1", "patch": "diff --git a/x b/x\n+1\n-2\n"},
        {"task_id": "repo__2", "patch": None},
    ]
    path = tmp_path / "out" / "predictions.jsonl"

    written = SWEBenchProvider.write_predictions(records, path, "agency:claude-sonnet-5")

    assert written == path
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {
        "instance_id": "repo__1",
        "model_name_or_path": "agency:claude-sonnet-5",
        "model_patch": "diff --git a/x b/x\n+1\n-2\n",
    }
    assert json.loads(lines[1])["model_patch"] == ""


def test_evaluate_never_reports_a_boolean_passed(tmp_path):
    provider = SWEBenchProvider(tasks_path=tmp_path / "unused.jsonl", cache_dir=tmp_path / "cache")
    task = BenchmarkTask(task_id="t", benchmark="swe-bench", instructions="")
    result = BenchmarkResult(
        task_id="t", completed=True, summary="", patch="diff --git a b\n+x\n-y\n"
    )

    metrics = provider.evaluate(task, result)

    assert metrics["passed"] is None
    assert metrics["patch_generated"] is True
    assert metrics["lines_changed"] == 2
