"""Tests for the OpenCode CLI adapter's availability and output parser."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agency.configs.agconfig import agconfig
from agency.harness.adapters.agharness_backend import AdapterRuntime
from agency.harness.adapters.opencode import _OpencodeBackend, opencode_available


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


def test_opencode_available_reflects_real_which():
    import shutil

    assert opencode_available() == (shutil.which("opencode") is not None)


def test_run_attempt_writes_and_registers_the_agpolicy_plugin(monkeypatch, tmp_path):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: f"/usr/bin/{name}")
    backend = _OpencodeBackend(agconfig())
    ag = _make_agent()
    handle = _make_handle(
        stdout=json.dumps({"type": "text", "part": {"type": "text", "text": "hello from opencode"}})
    )
    captured = {}
    config_home = tmp_path / "opencode-config"
    config_home.mkdir()
    monkeypatch.setattr(
        "agency.harness.agharness.materialize_config_home", lambda *args: config_home
    )

    def fake_launch(argv, envp, *, cwd, stdin_data, policy, ag):
        from pathlib import Path

        launched_config_home = Path(envp["HOME"])
        captured["envp"] = envp
        captured["cwd"] = cwd
        captured["argv"] = argv
        captured["stdin_data"] = stdin_data
        captured["config"] = json.loads((launched_config_home / "opencode.json").read_text())
        captured["plugin_exists"] = (
            launched_config_home / "plugin" / "agpolicy_plugin.js"
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
            runtime, prompt="go", resume_session_id=None, prior_session_blob=None, max_steps=None
        )

    assert result.final_text == "hello from opencode"
    assert captured["plugin_exists"] is True
    assert len(captured["config"]["plugin"]) == 1
    assert captured["config"]["plugin"][0].startswith("file://")
    assert captured["config"]["plugin"][0].endswith("agpolicy_plugin.js")
    assert "agent" not in captured["config"]
    assert captured["envp"]["AGPOLICY_BASE_URL"] == "http://harness.local"
    assert captured["envp"]["AGPOLICY_TOKEN"] == "tok-1"
    assert captured["argv"] == ["opencode", "run", "--format", "json"]
    assert "go" not in captured["argv"]
    assert captured["stdin_data"] == b"go"
    assert captured["cwd"] == "/workspace"
    assert captured["envp"]["HOME"] == str(config_home)
    assert captured["envp"]["OPENCODE_CONFIG"] == str(config_home / "opencode.json")
    assert not config_home.exists()


def test_parse_output_events_extracts_last_text_from_ndjson():
    stdout = "\n".join(
        [
            json.dumps({"type": "turn.started"}),
            json.dumps({"type": "item.completed", "text": "partial"}),
            json.dumps({"type": "item.completed", "text": "final answer"}),
        ]
    )
    assert _OpencodeBackend._parse_output_events(stdout) == "final answer"


def test_parse_output_events_extracts_nested_text_part():
    stdout = json.dumps({"type": "text", "part": {"type": "text", "text": "hello from opencode"}})
    assert _OpencodeBackend._parse_output_events(stdout) == "hello from opencode"


def test_parse_output_events_ignores_non_text_events_and_malformed_lines():
    stdout = "\n".join(
        [
            json.dumps({"type": "status", "part": {"type": "status", "text": "working"}}),
            "malformed json",
            json.dumps({"type": "tool", "part": {"type": "tool", "text": "tool output"}}),
            json.dumps({"type": "text", "part": {"type": "text", "text": "final answer"}}),
        ]
    )
    assert _OpencodeBackend._parse_output_events(stdout) == "final answer"


def test_parse_output_events_joins_nested_text_parts_in_order():
    stdout = "\n".join(
        [
            json.dumps({"type": "text", "part": {"type": "text", "text": "hello "}}),
            json.dumps({"type": "text", "part": {"type": "text", "text": "from "}}),
            json.dumps({"type": "text", "part": {"type": "text", "text": "opencode"}}),
        ]
    )
    assert _OpencodeBackend._parse_output_events(stdout) == "hello from opencode"


def test_parse_output_events_falls_back_to_raw_text():
    assert _OpencodeBackend._parse_output_events("just plain text, no JSON") == (
        "just plain text, no JSON"
    )
