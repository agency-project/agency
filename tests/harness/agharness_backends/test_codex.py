"""Tests for the Codex CLI adapter's availability and output parser."""

from __future__ import annotations

import json
import tomllib
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


def test_run_attempt_uses_isolated_generated_config(monkeypatch, tmp_path):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: f"/usr/bin/{name}")
    backend = _CodexBackend(agconfig())
    ag = _make_agent()
    handle = _make_handle(stdout='{"result": "ok"}')
    captured = {}
    config_home = tmp_path / "codex-config"
    config_home.mkdir()
    host_config_home = tmp_path / "host-codex-config"
    host_config_home.mkdir()
    (host_config_home / "config.toml").write_text('model = "host-model"\n')
    monkeypatch.setenv("CODEX_HOME", str(host_config_home))
    monkeypatch.setattr(
        "agency.harness.agharness.materialize_config_home", lambda *args: config_home
    )

    def fake_launch(argv, envp, *, cwd, stdin_data, policy, ag):
        from pathlib import Path

        launched_config_home = Path(envp["CODEX_HOME"])
        captured["argv"] = argv
        captured["envp"] = envp
        captured["cwd"] = cwd
        captured["stdin_data"] = stdin_data
        captured["config_path"] = launched_config_home / "config.toml"
        captured["config"] = tomllib.loads(captured["config_path"].read_text())
        captured["config_home_hooks"] = (launched_config_home / "hooks.json").read_text()
        captured["config_home_hook_script"] = (launched_config_home / "agpolicy_hook.py").exists()
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
    assert captured["argv"] == ["codex", "exec", "--skip-git-repo-check", "--json", "-"]
    assert "go" not in captured["argv"]
    assert captured["stdin_data"] == b"go"
    assert "--ignore-user-config" not in captured["argv"]
    assert captured["config_path"] == config_home / "config.toml"
    assert captured["config"]["model"] == "test-model"
    assert captured["config"]["model_provider"] == "agency-proxy"
    provider = captured["config"]["model_providers"]["agency-proxy"]
    assert provider["base_url"] == "http://harness.local/v1"
    assert provider["env_key"] == "AGENCY_PROXY_API_KEY"
    assert provider["wire_api"] == "responses"
    assert captured["envp"]["AGENCY_PROXY_API_KEY"] == "tok-1"
    assert captured["envp"]["AGPOLICY_BASE_URL"] == "http://harness.local"
    assert captured["envp"]["AGPOLICY_TOKEN"] == "tok-1"
    assert captured["cwd"] == "/workspace"
    assert captured["envp"]["CODEX_HOME"] == str(config_home)
    assert captured["envp"]["AGPOLICY_STATE_DIR"] == str(config_home)
    assert not config_home.exists()
    assert (host_config_home / "config.toml").read_text() == 'model = "host-model"\n'


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
