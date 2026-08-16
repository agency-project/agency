"""Tests for the Grok Build (xAI) agharness_backend. No real `grok` binary
is exercised here (see grok_available(), expected False -- installing it
requires running xAI's curl|bash install script, deliberately not done
without being asked) -- `agProxyPtrace.launch` is mocked throughout,
matching test_opencode.py's Tier 1 strategy (grok also routes through
`agproxy_llm`'s passthrough gateway, unlike Claude Code/Codex)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agency.agconfig import agConfig
from agency.agdata import agdata, agerror
from agency.agcontext import agcontext
from agency.agharness_internal.agharness_backends.grok import _GrokBackend, grok_available
from agency.agskill import agskill


def _make_agent(with_sandbox=True):
    ag = MagicMock()
    ag.agconfig = agConfig()
    ag.llm.backend.model = "test-model"
    ag.sandbox = MagicMock() if with_sandbox else None
    return ag


def _make_handle(stdout="", stderr="", rc=0):
    handle = MagicMock()
    handle.wait.return_value = (stdout, stderr, rc)
    return handle


@pytest.fixture(autouse=True)
def _patch_which_finds_grok(monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: f"/usr/bin/{name}")


def _patched_gateway_and_ptrace(handle):
    mock_gateway = MagicMock()
    mock_gateway.base_url = "http://127.0.0.1:1"

    def apply(mock_gateway_getter, mock_px_cls):
        mock_gateway_getter.return_value = mock_gateway
        mock_px_cls.return_value.launch.return_value = handle

    return mock_gateway, apply


def test_grok_available_reflects_real_which():
    import shutil

    assert grok_available() == (shutil.which("grok") is not None)


def test_execute_returns_agerror_when_binary_missing(monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: None)
    backend = _GrokBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    result, ctx, delta = backend.execute(ag, agcontext(), agdata(x=1), None, skill=skill)
    assert isinstance(result, agerror)
    assert "not found on PATH" in result.error


def test_execute_parses_json_result_and_routes_through_gateway():
    backend = _GrokBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    prev_ctx = agcontext()

    payload = json.dumps(
        {
            "text": "Hi there!",
            "usage": {"input_tokens": 5, "output_tokens": 2},
            "sessionId": "sess-1",
        }
    )
    handle = _make_handle(stdout=payload)
    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as mock_gateway_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox") as mock_wire,
    ):
        mock_gateway, apply = _patched_gateway_and_ptrace(handle)
        apply(mock_gateway_getter, mock_px_cls)

        result, ctx, delta = backend.execute(ag, prev_ctx, agdata(task="go"), None, skill=skill)

    assert not isinstance(result, agerror)
    assert result.result == "Hi there!"
    assert ctx is prev_ctx
    assert ctx.total_input_tokens == 5
    assert ctx.total_output_tokens == 2
    assert backend.session_resume_id == "sess-1"
    mock_wire.assert_called_once_with(handle, ag.sandbox)
    mock_gateway.register.assert_called_once()
    mock_gateway.unregister.assert_called_once()


def test_execute_uses_grok_home_for_config_isolation():
    backend = _GrokBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()

    handle = _make_handle(stdout='{"text": "ok"}')
    captured = {}

    def fake_launch(argv, envp, *, cwd, policy, ag, sandbox=None, stdin=None):
        captured["argv"] = argv
        captured["envp"] = envp
        captured["cwd"] = cwd
        return handle

    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as mock_gateway_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox"),
    ):
        mock_gateway, apply = _patched_gateway_and_ptrace(handle)
        mock_gateway_getter.return_value = mock_gateway
        mock_px_cls.return_value.launch.side_effect = fake_launch

        backend.execute(ag, agcontext(), agdata(task="go"), None, skill=skill)

    assert "GROK_HOME" in captured["envp"]
    assert captured["cwd"] == "/workspace"
    assert captured["argv"][0] == "/usr/bin/grok"
    assert "-p" in captured["argv"]
    assert "task.txt" in captured["argv"][captured["argv"].index("-p") + 1]
    assert "--output-format" in captured["argv"] and "json" in captured["argv"]


def test_execute_writes_chat_completions_config_toml():
    backend = _GrokBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()

    handle = _make_handle(stdout='{"text": "ok"}')
    written_config = {}

    def fake_launch(argv, envp, *, cwd, policy, ag, sandbox=None, stdin=None):
        written_config["toml"] = (cwd, envp)
        from pathlib import Path

        written_config["content"] = (Path(envp["GROK_HOME"]) / "config.toml").read_text()
        return handle

    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as mock_gateway_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox"),
    ):
        mock_gateway, _ = _patched_gateway_and_ptrace(handle)
        mock_gateway_getter.return_value = mock_gateway
        mock_px_cls.return_value.launch.side_effect = fake_launch

        backend.execute(ag, agcontext(), agdata(task="go"), None, skill=skill)

    content = written_config["content"]
    assert 'api_backend = "chat_completions"' in content
    assert "test-model" in content
    assert mock_gateway.base_url in content


def test_execute_skips_wire_to_sandbox_when_no_sandbox():
    backend = _GrokBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent(with_sandbox=False)

    handle = _make_handle(stdout='{"text": "ok"}')
    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as mock_gateway_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox") as mock_wire,
    ):
        mock_gateway, apply = _patched_gateway_and_ptrace(handle)
        apply(mock_gateway_getter, mock_px_cls)

        backend.execute(ag, agcontext(), agdata(task="go"), None, skill=skill)

    mock_wire.assert_not_called()


def test_execute_nonzero_exit_returns_agerror():
    backend = _GrokBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    prev_ctx = agcontext()

    handle = _make_handle(stdout="", stderr="auth error", rc=1)
    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as mock_gateway_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox"),
    ):
        mock_gateway, apply = _patched_gateway_and_ptrace(handle)
        apply(mock_gateway_getter, mock_px_cls)

        result, ctx, delta = backend.execute(ag, prev_ctx, agdata(task="go"), None, skill=skill)

    assert isinstance(result, agerror)
    assert "auth error" in result.error
    assert ctx is prev_ctx


def test_execute_recovers_structured_output_schema():
    backend = _GrokBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing", output_schema=agdata(answer=str))
    ag = _make_agent()
    prev_ctx = agcontext()

    handle = _make_handle(stdout=json.dumps({"text": '{"answer": "42"}'}))
    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as mock_gateway_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox"),
    ):
        mock_gateway, apply = _patched_gateway_and_ptrace(handle)
        apply(mock_gateway_getter, mock_px_cls)

        result, ctx, delta = backend.execute(ag, prev_ctx, agdata(task="go"), None, skill=skill)

    assert not isinstance(result, agerror)
    assert result.answer == "42"


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
