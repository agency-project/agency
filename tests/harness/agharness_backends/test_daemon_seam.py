from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agency.agconfig import agConfig
from agency.harness.adapters.base import AdapterRuntime, agharness_backend
from agency.harness.adapters.claude_code import _ClaudeCodeBackend
from agency.harness.adapters.codex import _CodexBackend
from agency.harness.adapters.grok import _GrokBackend
from agency.harness.adapters.native import _NativeBackend
from agency.harness.adapters.opencode import _OpencodeBackend


def _runtime(*, sandbox=None) -> AdapterRuntime:
    return AdapterRuntime(
        agconfig=agConfig(),
        model="test-model",
        engine_name="test-agent",
        harness_base_url="http://127.0.0.1:8766",
        token="test-token",
        syscall_policy=object(),
        sandbox=sandbox,
    )


@pytest.mark.parametrize("name", ["native", "claude_code", "codex", "grok", "opencode"])
def test_every_engine_implements_daemon_attempt_seam(name):
    adapter = agharness_backend.for_config(name, agConfig())
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

    def launch(_self, argv, envp, *, cwd, policy, ag):
        captured.update(argv=argv, envp=envp, cwd=cwd, policy=policy, ag=ag)
        if config_name is not None:
            captured["config"] = (Path(cwd) / config_name).read_text()
        return handle

    monkeypatch.setattr("shutil.which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr("agency.harness.ptrace.supervisor.agProxyPtrace.launch", launch)
    runtime = _runtime()

    result = backend_cls(agConfig()).run_daemon_attempt(
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
    assert "do the thing" in captured["argv"]


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
        "agency.agutil.ensure_python_packages_in_container", lambda *args, **kwargs: None
    )

    result = _NativeBackend(agConfig()).run_daemon_attempt(
        _runtime(sandbox=sandbox),
        prompt="do the thing",
        resume_session_id=None,
        prior_session_blob=None,
        max_steps=4,
    )

    assert result.ok
    assert result.final_text == "native-ok"
    assert any("native_harness.cli" in command for command in sandbox.commands)
