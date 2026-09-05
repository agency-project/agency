"""Tests for the Codex CLI adapter's availability and output parser."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agency.configs.agconfig import agconfig
from agency.harness.adapters.agharness_backend import AdapterRuntime
from agency.harness.adapters.codex import _CodexBackend, codex_available


def _make_agent():
    ag = MagicMock()
    ag.agname = "test-agent"
    ag.agconfig = agconfig()
    ag.model = "test-model"
    ag.sandbox = None
    return ag


def _make_handle(stdout="", stderr="", rc=0):
    handle = MagicMock()
    handle.wait.return_value = (stdout, stderr, rc)
    return handle


def test_codex_available_reflects_real_which():
    import shutil

    assert codex_available() == (shutil.which("codex") is not None)


def test_run_attempt_writes_and_registers_admission_and_completion_hooks(monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: f"/usr/bin/{name}")
    backend = _CodexBackend(agconfig())
    ag = _make_agent()
    handle = _make_handle(stdout='{"result": "ok"}')
    captured = {}

    def fake_launch(argv, envp, *, cwd, policy, ag):
        from pathlib import Path

        config_home = Path(cwd)
        captured["argv"] = argv
        captured["envp"] = envp
        captured["cwd"] = cwd
        captured["config_home_hooks"] = (config_home / "hooks.json").read_text()
        captured["config_home_hook_script"] = (config_home / "agpolicy_hook.py").exists()
        return handle

    runtime = AdapterRuntime(
        agconfig=ag.agconfig,
        model=ag.model,
        engine_name=ag.agname,
        harness_base_url="http://harness.local",
        token="tok-1",
        syscall_policy=MagicMock(),
        sandbox=ag.sandbox,
    )
    with patch("agency.harness.ptrace.supervisor.agProxyPtrace") as ptrace_cls:
        ptrace_cls.return_value.launch.side_effect = fake_launch
        backend.run_daemon_attempt(
            runtime, prompt="go", resume_session_id=None, prior_session_blob=None, max_steps=None
        )

    hooks = json.loads(captured["config_home_hooks"])
    assert set(hooks["hooks"]) == {"PreToolUse", "PostToolUse"}
    assert captured["config_home_hook_script"] is True
    assert captured["envp"]["AGPOLICY_BASE_URL"] == "http://harness.local"
    assert captured["envp"]["AGPOLICY_TOKEN"] == "tok-1"
    assert captured["envp"]["AGPOLICY_STATE_DIR"] == captured["cwd"]


def test_parse_output_events_ignores_non_agent_message_items():
    stdout = "\n".join(
        [
            json.dumps(
                {"type": "item.completed", "item": {"type": "reasoning", "text": "thinking"}}
            ),
            json.dumps(
                {"type": "item.completed", "item": {"type": "agent_message", "text": "final"}}
            ),
        ]
    )
    assert _CodexBackend._parse_output_events(stdout) == "final"


def test_parse_output_events_falls_back_to_raw_text():
    assert _CodexBackend._parse_output_events("plain text") == "plain text"
