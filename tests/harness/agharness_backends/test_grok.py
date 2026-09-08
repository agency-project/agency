"""Tests for the Grok CLI adapter's availability and result parser."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

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


def test_run_attempt_writes_and_registers_admission_and_completion_hooks(monkeypatch, tmp_path):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: f"/usr/bin/{name}")
    backend = _GrokBackend(agconfig())
    ag = _make_agent()
    handle = _make_handle(stdout='{"text": "ok"}')
    captured = {}
    config_home = tmp_path / "grok-config"
    config_home.mkdir()
    prompt = "first line\nUnicode: Grök 🚀\nlast line"
    monkeypatch.setattr(
        "agency.harness.agharness.materialize_config_home", lambda *args: config_home
    )

    def fake_launch(argv, envp, *, cwd, policy, ag):
        from pathlib import Path

        launched_config_home = Path(envp["GROK_HOME"])
        prompt_path = Path(argv[argv.index("--prompt-file") + 1])
        captured["argv"] = argv
        captured["envp"] = envp
        captured["cwd"] = cwd
        captured["config_toml"] = (launched_config_home / "config.toml").read_text()
        captured["prompt_path"] = prompt_path
        captured["prompt_exists_at_launch"] = prompt_path.exists()
        captured["prompt_text"] = prompt_path.read_text(encoding="utf-8")
        captured["hooks_json"] = (launched_config_home / "hooks" / "agpolicy.json").read_text()
        captured["hook_script_exists"] = (
            launched_config_home / "hooks" / "agpolicy_hook.py"
        ).exists()
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
        result = backend.run_daemon_attempt(
            runtime, prompt=prompt, resume_session_id=None, prior_session_blob=None, max_steps=None
        )

    hooks = json.loads(captured["hooks_json"])
    assert set(hooks["hooks"]) == {"PreToolUse", "PostToolUse"}
    assert captured["hook_script_exists"] is True
    assert captured["envp"]["AGPOLICY_BASE_URL"] == "http://harness.local"
    assert captured["envp"]["AGPOLICY_TOKEN"] == "tok-1"
    assert captured["cwd"] == "/workspace"
    assert captured["envp"]["GROK_HOME"] == str(config_home)
    assert captured["envp"]["AGPOLICY_STATE_DIR"] == str(config_home)
    assert captured["config_toml"] == (
        "[models]\n"
        'default = "agency-proxy"\n\n'
        "[model.agency-proxy]\n"
        'model = "test-model"\n'
        'base_url = "http://harness.local/v1"\n'
        'api_key = "tok-1"\n'
        'api_backend = "chat_completions"\n'
    )
    assert "--yolo" in captured["argv"]
    assert "--prompt-file" in captured["argv"]
    assert prompt not in captured["argv"]
    assert "-p" not in captured["argv"]
    assert "--single" not in captured["argv"]
    assert captured["prompt_exists_at_launch"] is True
    assert captured["prompt_text"] == prompt
    assert captured["prompt_path"].parent == config_home
    assert not config_home.exists()
    assert result.ok is True
    assert result.final_text == "ok"


def test_run_attempt_cleans_config_home_when_launch_fails(monkeypatch, tmp_path):
    backend = _GrokBackend(agconfig())
    config_home = tmp_path / "grok-config"
    config_home.mkdir()
    monkeypatch.setattr(
        "agency.harness.agharness.materialize_config_home", lambda *args: config_home
    )

    runtime = AdapterRuntime(
        agconfig=agconfig(),
        model="test-model",
        engine_name="test-agent",
        harness_base_url="http://harness.local",
        token="tok-1",
        syscall_policy=MagicMock(),
        sandbox=None,
    )
    with patch("agency.harness.ptrace.supervisor.agProxyPtrace") as ptrace_cls:
        ptrace_cls.return_value.launch.side_effect = RuntimeError("launch failed")
        with pytest.raises(RuntimeError, match="launch failed"):
            backend.run_daemon_attempt(
                runtime,
                prompt="cleanup me",
                resume_session_id=None,
                prior_session_blob=None,
                max_steps=None,
            )

    assert not config_home.exists()


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
