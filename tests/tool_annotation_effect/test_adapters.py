import asyncio
import json
import sys
from types import SimpleNamespace, ModuleType

from benchmarks.tool_annotation_effect.adapters import (
    SweBenchAdapter,
    TerminalBenchAdapter,
    directory_hash,
)
from benchmarks.tool_annotation_effect.common import native
from benchmarks.tool_annotation_effect.planning import TERMINAL_RELEASE


def test_swebench_prediction_format_and_unique_evaluation_ids(tmp_path):
    adapter = SweBenchAdapter()
    task = {"id": "repo__issue-1"}
    prediction = tmp_path / "prediction.jsonl"
    exported = adapter.export_prediction(task, "diff --git a/a b/a", "m", prediction)
    assert json.loads(prediction.read_text()) == exported
    assert set(exported) == {"instance_id", "model_name_or_path", "model_patch"}
    commands = []

    def saved_evaluator(command, **kwargs):
        commands.append(command)
        run_id = command[command.index("--run_id") + 1]
        path = tmp_path / "logs/run_evaluation" / run_id / "m" / task["id"] / "report.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({task["id"]: {"resolved": True}}))
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    first = adapter.evaluate(task, prediction, tmp_path, command_runner=saved_evaluator)
    second = adapter.evaluate(task, prediction, tmp_path, command_runner=saved_evaluator)
    assert first["success"] is True
    assert first["evaluation_run_id"] != second["evaluation_run_id"]
    assert commands[0][0:3] == [sys.executable, "-m", "swebench.harness.run_evaluation"]


def test_swebench_setup_keeps_reference_patch_and_tests_outside_workspace(tmp_path, monkeypatch):
    adapter = SweBenchAdapter()
    monkeypatch.setattr(
        "benchmarks.tool_annotation_effect.adapters.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=b"base-archive"),
    )
    writes, commands = [], []
    environment = SimpleNamespace(
        write_file_bytes=lambda path, data: writes.append((path, data)),
        exec=lambda command: (commands.append(command) or "", 0),
    )
    prompt = adapter.setup(
        {
            "repository_path": str(tmp_path),
            "base_commit": "abc",
            "problem_statement": "Fix",
            "patch": "REFERENCE SECRET",
            "test_patch": "TEST SECRET",
        },
        environment,
    )
    assert prompt == "Fix"
    assert writes == [("/tmp/agency-base.tar", b"base-archive")]
    assert not any("SECRET" in command for command in commands)
    assert adapter.extract_patch(environment) == ""


def test_official_saved_verifier_formats_are_imported():
    swe = SweBenchAdapter()
    assert swe.import_report({"id": "x"}, {"resolved_ids": ["x"]})["success"] is True
    assert swe.import_report({"id": "x"}, {"unresolved_ids": ["x"]})["success"] is False
    assert swe.import_report({"id": "x"}, {"error_ids": ["x"]})["success"] is None
    harbor = TerminalBenchAdapter()
    assert harbor.import_report({"verifier_result": {"rewards": {"reward": 1}}})["success"] is True
    assert harbor.import_report({"verifier_result": {"rewards": {"reward": 0}}})["success"] is False
    assert (
        harbor.import_report({"exception_info": {"exception_type": "Timeout"}})["failure"]
        == "infrastructure"
    )
    for exception in (
        {"exception_type": "AgentTimeoutError"},
        {"exception_type": "RuntimeError", "exception_message": "exceeded max_steps=50"},
    ):
        verdict = harbor.import_report({"exception_info": exception})
        assert verdict["failure"] == "budget"
        assert verdict["success"] is False
    passing_after_timeout = harbor.import_report(
        {
            "exception_info": {"exception_type": "AgentTimeoutError"},
            "verifier_result": {"rewards": {"reward": 1}},
        }
    )
    assert passing_after_timeout["success"] is True


def test_harbor_job_uses_agency_native_agent_and_one_owned_environment(tmp_path):
    (tmp_path / "instruction.md").write_text("task")
    task = {
        "path": str(tmp_path),
        "id": "task",
        "release": TERMINAL_RELEASE,
        "content_hash": directory_hash(tmp_path),
    }
    job = TerminalBenchAdapter().job_config(
        task, {"trial_id": "trial", "arm": "purpose"}, {"model": "fake"}, {}, tmp_path / "output"
    )
    assert job["agents"][0]["import_path"].endswith("harbor_agent:AgencyNativeAgent")
    assert job["n_attempts"] == 1
    assert job["retry"]["max_retries"] == 0
    assert job["tasks"] == [{"path": str(tmp_path)}]
    assert job["environment"]["override_cpus"] == 2
    assert job["environment"]["override_memory_mb"] == 4096


def test_harbor_agent_runs_native_loop_and_remote_tools_without_nested_sandbox(
    tmp_path, monkeypatch
):
    base_module = ModuleType("harbor.agents.base")

    class Base:
        def __init__(self, logs_dir, model_name, **kwargs):
            self.logs_dir, self.model_name = logs_dir, model_name

    base_module.BaseAgent = Base
    monkeypatch.setitem(sys.modules, "harbor.agents.base", base_module)
    installed_module = ModuleType("harbor.agents.installed.base")
    installed_module.NonZeroAgentExitCodeError = type(
        "NonZeroAgentExitCodeError", (RuntimeError,), {}
    )
    monkeypatch.setitem(sys.modules, "harbor.agents.installed.base", installed_module)
    sys.modules.pop("benchmarks.tool_annotation_effect.harbor_agent", None)
    from benchmarks.tool_annotation_effect.harbor_agent import AgencyNativeAgent

    requests = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            assert kwargs["retry_rate_limits"] is True
            assert kwargs["max_attempts"] == 13
            assert kwargs["model_settings"]["stream_options"] == {"include_usage": True}

        def dispatch(self, model, messages, tools):
            requests.append((messages.copy(), tools))
            if len(requests) == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "m1",
                                "function": {
                                    "name": "bash",
                                    "arguments": json.dumps(
                                        {
                                            "command": "echo hi",
                                            "_agency": {"purpose": "Check shell"},
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            return {"message": {"role": "assistant", "content": "done"}}

        def close(self):
            pass

    monkeypatch.setattr(native("llm_client"), "LLMClient", FakeClient)
    monkeypatch.setenv("TEST_URL", "http://fake")
    monkeypatch.setenv("TEST_KEY", "not-a-real-key")
    executions, uploads = [], []

    class Environment:
        async def exec(self, command, **kwargs):
            executions.append(command)
            return SimpleNamespace(return_code=0, stdout='{"result":"hi"}', stderr="")

        async def upload_file(self, source, target):
            uploads.append(str(target))

    environment = Environment()
    agent = AgencyNativeAgent(
        logs_dir=tmp_path,
        model_name="fake",
        arm="purpose",
        model_config={"base_url_env": "TEST_URL", "api_key_env": "TEST_KEY"},
        experiment_config={
            "budgets": {"max_steps": 2, "timeout_s": 5},
            "context_limit": None,
            "tool_python_path": "/opt/agency-python/bin/python3",
        },
    )
    context = SimpleNamespace()

    async def execute():
        await agent.setup(environment)
        await agent.run("instruction", environment, context)

    asyncio.run(execute())
    assert uploads == ["/tmp/agency_native_tools.py"]
    assert "agency_native_tools" in executions[-1]
    assert executions[0].startswith("command -v /opt/agency-python/bin/python3")
    assert executions[-1].startswith("/opt/agency-python/bin/python3 -c ")
    assert len(requests) == 2
    assistant = next(message for message in requests[1][0] if message.get("tool_calls"))
    assert "_agency" in assistant["tool_calls"][0]["function"]["arguments"]
    assert context.n_input_tokens is None
    assert (tmp_path / "native-result.json").exists()

    # A step-limited episode uses Harbor's supported nonzero-agent exit type,
    # so SingleStepTrial proceeds to its official verifier.
    result = SimpleNamespace(status="error", final_text="", message="exceeded max_steps=2")
    monkeypatch.setattr(native("react_loop"), "run_react_loop", lambda *args, **kwargs: result)
    import pytest

    with pytest.raises(installed_module.NonZeroAgentExitCodeError, match="max_steps"):
        asyncio.run(agent.run("instruction", environment, context))


def test_migration_compilation_alone_is_not_success(tmp_path, monkeypatch):
    from benchmarks.tool_annotation_effect.execution import verify_migration
    from benchmarks.tool_annotation_effect.fixtures import local_tasks

    monkeypatch.setattr("shutil.which", lambda name: "/fake/" + name)
    (tmp_path / "solution.rs").write_text("fake Rust artifact")
    calls = []

    def fake_compiler_or_program(command, **kwargs):
        calls.append(command)
        if command[0] in ("cc", "rustc"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        value = int(kwargs["input"])
        result = value if command[0].endswith("/reference") else value + 1
        return SimpleNamespace(returncode=0, stdout=str(result) + "\n", stderr="")

    verdict = verify_migration(
        local_tasks("migration")[0], tmp_path, command_runner=fake_compiler_or_program
    )
    assert verdict["rust_compilation"] is True
    assert verdict["behavioral"] is False
    assert verdict["success"] is False
    assert len(verdict["mismatches"]) == verdict["cases"]


def test_migration_rust_compilation_failure_is_a_task_failure(tmp_path, monkeypatch):
    from benchmarks.tool_annotation_effect.execution import verify_migration
    from benchmarks.tool_annotation_effect.fixtures import local_tasks

    monkeypatch.setattr("shutil.which", lambda name: "/fake/" + name)

    def fake_compiler(command, **kwargs):
        return SimpleNamespace(
            returncode=1 if command[0] == "rustc" else 0, stdout="", stderr="syntax error"
        )

    verdict = verify_migration(local_tasks("migration")[0], tmp_path, command_runner=fake_compiler)
    assert verdict["rust_compilation"] is False
    assert verdict["behavioral"] is None
    assert verdict["success"] is False


def test_benchmark_runtime_mount_excludes_gold_manifests_and_credentials(tmp_path, monkeypatch):
    from pathlib import Path
    from benchmarks.tool_annotation_effect import execution

    observed = {}

    def fake_execution(task, trial, model, config, directory):
        runtime = Path(config["runtime_source"])
        observed["path"] = runtime
        assert (runtime / "agency/native_harness/react_loop.py").exists()
        assert not (runtime / "benchmarks").exists()
        assert not (runtime / ".env").exists()
        assert not (runtime / "artifacts").exists()
        return {"success": True}

    monkeypatch.setattr(execution, "_execute", fake_execution)
    assert execution.execute({}, {"suite": "rag"}, {}, {}, tmp_path) == {"success": True}
    assert not observed["path"].exists()


def test_terminal_execution_retains_native_duration_and_final_answer(tmp_path, monkeypatch):
    from benchmarks.tool_annotation_effect.execution import execute

    def launch(*args):
        logs = tmp_path / "harbor" / "trial" / "agent"
        logs.mkdir(parents=True)
        (logs / "native-result.json").write_text(
            json.dumps({"final_text": "Finished artifact", "agent_seconds": 3.25})
        )
        return {"verifier_result": {"rewards": {"reward": 1}}}

    monkeypatch.setattr(TerminalBenchAdapter, "launch", launch)
    result = execute({}, {"suite": "terminalbench"}, {}, {}, tmp_path)
    assert result["final_text"] == "Finished artifact"
    assert result["agent_seconds"] == 3.25


def test_agent_config_uses_explicit_container_backend(tmp_path, monkeypatch):
    from benchmarks.tool_annotation_effect.execution import agent_config

    monkeypatch.setenv("TEST_URL", "http://fake")
    monkeypatch.setenv("TEST_KEY", "not-a-real-key")
    config = agent_config(
        {"sandbox_backend": "docker", "context_limit": 196000, "budgets": {"max_steps": 50}},
        {
            "provider": "openai",
            "model": "fake",
            "base_url_env": "TEST_URL",
            "api_key_env": "TEST_KEY",
        },
        {"suite": "swebench", "arm": "baseline", "trial_id": "trial"},
        tmp_path,
        "agent",
    )
    assert config.sandbox.backend == "docker"


def test_failed_benchmark_episode_preserves_its_working_files(monkeypatch):
    import threading

    from agency.agdata import agerror
    from agency.engine.engine import AgentEngine
    from benchmarks.tool_annotation_effect.sandbox import EpisodeSandbox
    from tests.engine.test_engine import _FakeAgent

    sandbox = EpisodeSandbox.__new__(EpisodeSandbox)
    sandbox._destroyed = True
    sandbox._lock = threading.RLock()
    files, snapshot = {}, {}
    sandbox.commit = lambda: snapshot.update(files)
    sandbox.stop = lambda: None
    engine = AgentEngine(_FakeAgent())

    def exhausted(*args, **kwargs):
        files["solution.py"] = "attempted fix"
        return agerror("exceeded max_steps=50")

    monkeypatch.setattr(engine, "_execute_harness", exhausted)
    result = engine.execute(None, None, None, None, sandbox)
    assert isinstance(result, agerror)
    assert snapshot["solution.py"] == "attempted fix"
