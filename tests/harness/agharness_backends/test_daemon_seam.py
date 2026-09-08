from __future__ import annotations

import json
import re
import shlex
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agency.configs.agconfig import agconfig
from agency.harness.adapters.agharness_backend import AdapterRuntime, agharness_backend
from agency.harness.adapters.claude_code import _ClaudeCodeBackend
from agency.harness.adapters.codex import _CodexBackend
from agency.harness.adapters.grok import _GrokBackend
from agency.harness.adapters.native import _NativeBackend
from agency.harness.adapters.opencode import _OpencodeBackend


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
    adapter = agharness_backend.for_config(name, agconfig())
    assert type(adapter).run_daemon_attempt is not agharness_backend.run_daemon_attempt


@pytest.mark.parametrize(
    ("backend_cls", "stdout", "expected", "config_name"),
    [
        (
            _ClaudeCodeBackend,
            json.dumps({"result": "claude-ok", "usage": {"input_tokens": 2}}),
            "claude-ok",
            None,
        ),
        (
            _CodexBackend,
            json.dumps(
                {"type": "item.completed", "item": {"type": "agent_message", "text": "codex-ok"}}
            ),
            "codex-ok",
            "config.toml",
        ),
        (_GrokBackend, json.dumps({"text": "grok-ok"}), "grok-ok", "config.toml"),
        (_OpencodeBackend, json.dumps({"text": "opencode-ok"}), "opencode-ok", "opencode.json"),
    ],
)
def test_external_cli_adapters_launch_through_typed_runtime(
    monkeypatch, backend_cls, stdout, expected, config_name
):
    handle = MagicMock()
    handle.wait.return_value = (stdout, "", 0)
    captured = {}

    def launch(_self, argv, envp, *, cwd, policy, ag, stdin_data=None):
        captured.update(
            argv=argv,
            envp=envp,
            cwd=cwd,
            policy=policy,
            ag=ag,
            stdin_data=stdin_data,
        )
        if config_name is not None:
            if "OPENCODE_CONFIG" in envp:
                config_path = Path(envp["OPENCODE_CONFIG"])
            else:
                config_home = Path(envp.get("CODEX_HOME") or envp["GROK_HOME"])
                config_path = config_home / config_name
            captured["config"] = config_path.read_text()
        if backend_cls is _GrokBackend:
            prompt_path = Path(argv[argv.index("--prompt-file") + 1])
            captured["prompt"] = prompt_path.read_text(encoding="utf-8")
        return handle

    monkeypatch.setattr("shutil.which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr("agency.harness.ptrace.supervisor.agProxyPtrace.launch", launch)
    runtime = _runtime()

    result = backend_cls(agconfig()).run_daemon_attempt(
        runtime,
        prompt="do the thing",
        resume_session_id=None,
        prior_session_blob=None,
        max_steps=4,
    )

    assert result.ok
    assert result.final_text == expected
    assert captured["policy"] is runtime.syscall_policy
    assert captured["ag"] is None
    if backend_cls is _GrokBackend:
        assert "do the thing" not in captured["argv"]
        assert captured["prompt"] == "do the thing"
        assert captured["stdin_data"] is None
    else:
        assert "do the thing" not in captured["argv"]
        assert captured["stdin_data"] == b"do the thing"


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

    def launch(_self, argv, envp, *, cwd, stdin_data, **kwargs):
        assert argv[0] == str(binary)
        assert argv[argv.index("--resume") + 1] == "session-one"
        assert "continue" not in argv
        assert stdin_data == b"continue"
        assert session.read_bytes() == b"prior transcript"
        assert Path(cwd, "agpolicy_hook.py").is_file()
        session.write_bytes(b"updated transcript")
        handle = MagicMock()
        handle.wait.return_value = ('{"result":"done","session_id":"session-one"}', "", 0)
        return handle

    monkeypatch.setattr("agency.harness.ptrace.supervisor.agProxyPtrace.launch", launch)
    result = _ClaudeCodeBackend(config).run_daemon_attempt(
        replace(_runtime(sandbox=sandbox), agconfig=config),
        prompt="continue",
        resume_session_id="session-one",
        prior_session_blob=b"prior transcript",
        max_steps=4,
    )
    assert result.ok
    assert result.session_blob == b"updated transcript"
    assert not config_home.exists()


class _NativeSandbox:
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
        if "native_harness.cli" in cmd:
            match = re.search(r"> ([^ ]+) 2> ([^ ]+)$", cmd)
            assert match
            self.files[match.group(1)] = json.dumps({"result": "native-ok", "usage": {}})
            self.files[match.group(2)] = ""
        return "", 0

    def read_file(self, path):
        return self.files[path]

    def read_file_bytes(self, path):
        return self.files[path].encode()

    def write_file_bytes(self, path, data):
        self.files[path] = data.decode()


def test_native_adapter_launches_through_typed_runtime(monkeypatch):
    sandbox = _NativeSandbox()
    monkeypatch.setattr(
        "agency.utils.agutil.ensure_python_packages_in_container", lambda *args, **kwargs: None
    )

    result = _NativeBackend(agconfig()).run_daemon_attempt(
        _runtime(sandbox=sandbox),
        prompt="do the thing",
        resume_session_id=None,
        prior_session_blob=None,
        max_steps=4,
    )

    assert result.ok
    assert result.final_text == "native-ok"
    assert any("native_harness.cli" in command for command in sandbox.commands)


@pytest.mark.parametrize("backend_cls", [_NativeBackend, _ClaudeCodeBackend])
@pytest.mark.parametrize("has_sandbox_tools", [False, True])
def test_mcp_adapters_include_separate_sandbox_config(monkeypatch, backend_cls, has_sandbox_tools):
    captured = {}
    sandbox = _NativeSandbox() if backend_cls is _NativeBackend else None

    def launch(_self, argv, envp, **kwargs):
        captured["argv"] = argv
        handle = MagicMock()
        handle.wait.return_value = ('{"result": "done"}', "", 0)
        return handle

    monkeypatch.setattr("shutil.which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr("agency.harness.ptrace.supervisor.agProxyPtrace.launch", launch)
    monkeypatch.setattr(
        "agency.utils.agutil.ensure_python_packages_in_container", lambda *args, **kwargs: None
    )
    runtime = replace(_runtime(sandbox=sandbox), has_sandbox_mcp_tools=has_sandbox_tools)
    result = backend_cls(agconfig()).run_daemon_attempt(
        runtime, prompt="test", resume_session_id=None, prior_session_blob=None, max_steps=2
    )
    assert result.ok
    argv = (
        shlex.split(next(cmd for cmd in sandbox.commands if "native_harness.cli" in cmd))
        if sandbox is not None
        else captured["argv"]
    )
    servers = json.loads(argv[argv.index("--mcp-config") + 1])["mcpServers"]
    assert set(servers) == ({"agency", "agency-sandbox"} if has_sandbox_tools else {"agency"})
    assert servers["agency"]["url"] == f"{runtime.harness_base_url}/mcp"
    if has_sandbox_tools:
        assert servers["agency-sandbox"] == {
            "type": "http",
            "url": f"{runtime.harness_base_url}/sandbox/mcp",
            "headers": {"Authorization": f"Bearer {runtime.token}"},
        }
