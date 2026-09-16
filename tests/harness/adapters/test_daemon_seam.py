from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agency.configs.agconfig import agconfig
from agency.harness.adapters.base import AdapterRuntime, HarnessAdapter
from agency.harness.adapters.claude_code import ClaudeCodeAdapter
from agency.harness.adapters.codex import CodexAdapter
from agency.harness.adapters.grok import GrokAdapter
from agency.harness.adapters.native import NativeAdapter
from agency.harness.adapters.opencode import OpenCodeAdapter


def _runtime(*, sandbox=None) -> AdapterRuntime:
    return AdapterRuntime(
        agconfig=agconfig(),
        model="test-model",
        engine_name="test-agent",
        harness_base_url="http://127.0.0.1:8766",
        token="test-token",
        syscall_policy=object(),
        sandbox=sandbox,
    )


@pytest.mark.parametrize("name", ["native", "claude_code", "codex", "grok", "opencode"])
def test_every_engine_implements_daemon_attempt_seam(name):
    adapter = HarnessAdapter.for_config(name, agconfig())
    assert type(adapter).run_daemon_attempt is not HarnessAdapter.run_daemon_attempt


@pytest.mark.parametrize("backend_cls", [CodexAdapter, GrokAdapter, OpenCodeAdapter])
def test_external_adapters_use_the_shared_pty_runner(monkeypatch, backend_cls):
    from agency.harness.adapters.base import AttemptResult
    from agency.harness.adapters.pty.execution import PtyExecution

    captured = {}

    def run(execution, prompt):
        captured.update(runtime=execution.runtime, prompt=prompt, driver=execution.driver)
        from agency.harness.agharness import cleanup_config_home

        cleanup_config_home(execution.driver.root)
        return AttemptResult(ok=True, final_text="native-final")

    monkeypatch.setattr(PtyExecution, "run", run)
    runtime = _runtime()
    result = backend_cls(agconfig()).run_daemon_attempt(
        runtime,
        prompt="do the thing",
        resume_session_id=None,
        prior_session_blob=None,
        max_steps=4,
    )
    assert result.ok and result.final_text == "native-final"
    assert captured["runtime"] is runtime
    assert captured["prompt"] == "do the thing"
    assert captured["driver"].cwd == "/workspace"
    assert not {"exec", "run", "--json", "--prompt-file", "--format"} & set(captured["driver"].argv)


@pytest.mark.parametrize("sandbox", [None, object()])
def test_claude_uses_staged_path_and_local_session_files(monkeypatch, tmp_path, sandbox):
    from agency.configs.agconfig import harnessadapterconfig
    from agency.harness.adapters.claude_code import _session_path

    binary = tmp_path / "mounted-cache" / "claude"
    binary.parent.mkdir()
    binary.write_bytes(b"staged executable")
    binary.chmod(0o755)
    config = agconfig(harnessadapterconfig(binary_path=str(binary)))
    config_home = tmp_path / "config"
    config_home.mkdir()
    monkeypatch.setattr(
        "agency.harness.agharness.materialize_config_home", lambda *args: config_home
    )
    session = Path(_session_path(str(config_home), "session-one"))

    argv, _env = ClaudeCodeAdapter(config).prepare_pty(
        replace(_runtime(sandbox=sandbox), agconfig=config),
        config_home,
        resume_session_id="session-one",
        prior_session_blob=b"prior transcript",
    )
    assert argv[0] == str(binary)
    assert argv[-2:] == ["--resume", "session-one"]
    assert session.read_bytes() == b"prior transcript"
    assert (config_home / "agpolicy_hook.py").is_file()
    assert "-p" not in argv


class _NativeSandbox:
    """Only backs materialize/cleanup_config_home_in_container and session
    blob I/O now -- native's harness launch itself goes through the same
    agProxyPtrace.launch() mock every other adapter's test uses (see
    test_native_adapter_launches_through_typed_runtime), not sandbox.exec()."""

    class _Backend:
        IMAGE_KIND = "container"

        def ingest_ptrace_pids(self, **_changes):
            return None

    def __init__(self):
        self._backend = self._Backend()
        self.commands = []
        self.files = {}

    def exec(self, cmd, workdir="/workspace", timeout=600):
        self.commands.append(cmd)
        return "", 0

    def read_file_bytes(self, path):
        return self.files[path].encode()

    def write_file_bytes(self, path, data):
        self.files[path] = data.decode()


def _native_launch(captured: dict, stdout: str):
    def launch(_self, argv, envp, *, cwd, policy, ag, stdin_data=None):
        import sys

        assert argv[0] == sys.executable
        assert cwd == "/workspace"
        assert stdin_data is None
        captured["argv"] = argv
        captured["envp"] = envp
        handle = MagicMock()
        handle.wait.return_value = (stdout, "", 0)
        return handle

    return launch


@pytest.mark.parametrize("outcome", ["timeout", "wait_error", "registration_error"])
def test_native_reaps_process_before_removing_scratch_files(monkeypatch, outcome):
    sandbox = _NativeSandbox()
    handle = MagicMock()
    handle.wait.return_value = ("", "", -1)
    if outcome == "wait_error":
        handle.wait.side_effect = RuntimeError("wait failed")
    register = MagicMock()
    if outcome == "registration_error":
        register.side_effect = RuntimeError("registration failed")
    monkeypatch.setattr(
        "agency.utils.agutil.ensure_python_packages_locally", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "agency.harness.ptrace.supervisor.agProxyPtrace.launch", lambda *args, **kwargs: handle
    )

    def cleanup(*args):
        handle.close.assert_called_once_with()

    monkeypatch.setattr("agency.harness.agharness.cleanup_config_home_in_container", cleanup)
    runtime = replace(_runtime(sandbox=sandbox), register_control_handle=register)
    kwargs = dict(prompt="work", resume_session_id=None, prior_session_blob=None, max_steps=1)
    if outcome == "timeout":
        assert not NativeAdapter(agconfig()).run_daemon_attempt(runtime, **kwargs).ok
    else:
        with pytest.raises(RuntimeError, match="failed"):
            NativeAdapter(agconfig()).run_daemon_attempt(runtime, **kwargs)


def test_native_adapter_launches_through_typed_runtime(monkeypatch):
    sandbox = _NativeSandbox()
    captured: dict = {}
    monkeypatch.setattr(
        "agency.utils.agutil.ensure_python_packages_locally", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "agency.harness.ptrace.supervisor.agProxyPtrace.launch",
        _native_launch(captured, json.dumps({"result": "native-ok", "usage": {}})),
    )

    result = NativeAdapter(agconfig()).run_daemon_attempt(
        _runtime(sandbox=sandbox),
        prompt="do the thing",
        resume_session_id=None,
        prior_session_blob=None,
        max_steps=4,
    )

    assert result.ok
    assert result.final_text == "native-ok"
    argv = captured["argv"]
    assert argv[argv.index("--max-steps") + 1] == "4"


def test_native_adapter_uses_existing_default_when_max_steps_is_none(monkeypatch):
    sandbox = _NativeSandbox()
    captured: dict = {}
    monkeypatch.setattr(
        "agency.utils.agutil.ensure_python_packages_locally", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "agency.harness.ptrace.supervisor.agProxyPtrace.launch",
        _native_launch(captured, json.dumps({"result": "native-ok", "usage": {}})),
    )

    result = NativeAdapter(agconfig()).run_daemon_attempt(
        _runtime(sandbox=sandbox),
        prompt="do the thing",
        resume_session_id=None,
        prior_session_blob=None,
        max_steps=None,
    )

    assert result.ok
    argv = captured["argv"]
    assert argv[argv.index("--max-steps") + 1] == "20"


@pytest.mark.parametrize("backend_cls", [NativeAdapter, ClaudeCodeAdapter])
@pytest.mark.parametrize("has_sandbox_tools", [False, True])
def test_mcp_adapters_include_separate_sandbox_config(monkeypatch, backend_cls, has_sandbox_tools):
    captured = {}
    sandbox = _NativeSandbox() if backend_cls is NativeAdapter else None

    def launch(_self, argv, envp, **kwargs):
        captured["argv"] = argv
        handle = MagicMock()
        handle.wait.return_value = ('{"result": "done"}', "", 0)
        return handle

    monkeypatch.setattr("shutil.which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr("agency.harness.ptrace.supervisor.agProxyPtrace.launch", launch)
    monkeypatch.setattr(
        "agency.utils.agutil.ensure_python_packages_locally", lambda *args, **kwargs: None
    )
    runtime = replace(_runtime(sandbox=sandbox), has_sandbox_mcp_tools=has_sandbox_tools)
    if backend_cls is ClaudeCodeAdapter:
        from agency.harness.agharness import cleanup_config_home, materialize_config_home

        config_home = materialize_config_home(runtime.engine_name)
        argv, _ = backend_cls(agconfig()).prepare_pty(runtime, config_home)
        cleanup_config_home(config_home)
    else:
        result = backend_cls(agconfig()).run_daemon_attempt(
            runtime, prompt="test", resume_session_id=None, prior_session_blob=None, max_steps=2
        )
        assert result.ok
        argv = captured["argv"]
    servers = json.loads(argv[argv.index("--mcp-config") + 1])["mcpServers"]
    assert set(servers) == ({"agency", "agency-sandbox"} if has_sandbox_tools else {"agency"})
    assert servers["agency"]["url"] == f"{runtime.harness_base_url}/mcp"
    if has_sandbox_tools:
        assert servers["agency-sandbox"] == {
            "type": "http",
            "url": f"{runtime.harness_base_url}/sandbox/mcp",
            "headers": {"Authorization": f"Bearer {runtime.token}"},
        }
