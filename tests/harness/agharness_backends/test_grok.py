"""Tests for the Grok CLI adapter's availability and result parser."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agency.configs.agconfig import agconfig
from agency.harness.adapters.agharness_backend import AdapterRuntime
from agency.harness.adapters.grok import _GrokBackend, grok_available


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


def test_grok_available_reflects_real_which():
    import shutil

    assert grok_available() == (shutil.which("grok") is not None)


def test_run_attempt_writes_and_registers_admission_and_completion_hooks(monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: f"/usr/bin/{name}")
    backend = _GrokBackend(agconfig())
    ag = _make_agent()
    handle = _make_handle(stdout='{"text": "ok"}')
    captured = {}

    def fake_launch(argv, envp, *, cwd, policy, ag):
        from pathlib import Path

        config_home = Path(cwd)
        captured["envp"] = envp
        captured["cwd"] = cwd
        captured["hooks_json"] = (config_home / "hooks" / "agpolicy.json").read_text()
        captured["hook_script_exists"] = (config_home / "hooks" / "agpolicy_hook.py").exists()
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

    hooks = json.loads(captured["hooks_json"])
    assert set(hooks["hooks"]) == {"PreToolUse", "PostToolUse"}
    assert captured["hook_script_exists"] is True
    assert captured["envp"]["AGPOLICY_BASE_URL"] == "http://harness.local"
    assert captured["envp"]["AGPOLICY_TOKEN"] == "tok-1"
    assert captured["envp"]["AGPOLICY_STATE_DIR"] == captured["cwd"]


def test_parse_result_json_extracts_text_usage_and_session():
    payload = json.dumps(
        {"text": "abc", "usage": {"input_tokens": 1, "output_tokens": 2}, "sessionId": "s1"}
    )
    text, usage, session_id = _GrokBackend._parse_result_json(payload)
    assert text == "abc"
    assert usage == {"input_tokens": 1, "output_tokens": 2}
    assert session_id == "s1"


def test_parse_result_json_falls_back_on_malformed_json():
    text, usage, session_id = _GrokBackend._parse_result_json("not json")
    assert text == "not json"
    assert usage == {}
    assert session_id is None
