"""Tests for the Codex CLI agharness_backend -- structural only, per this
backend's own module docstring: no `codex` binary was installable in this
environment, so `agProxyPtrace.launch` is mocked throughout and there is no
Tier 2 real-CLI test here (unlike test_claude_code.py's `real_claude`
tier)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agency.agconfig import agConfig
from agency.agdata import agdata, agerror
from agency.agcontext import agcontext
from agency.agharness_internal.agharness_backends.codex import _CodexBackend, codex_available
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


def _patched_gateway_and_ptrace(handle):
    mock_gateway = MagicMock()
    mock_gateway.base_url = "http://127.0.0.1:1"

    def apply(mock_gateway_getter, mock_px_cls):
        mock_gateway_getter.return_value = mock_gateway
        mock_px_cls.return_value.launch.return_value = handle

    return mock_gateway, apply


@pytest.fixture(autouse=True)
def _patch_which_finds_codex(monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: f"/usr/bin/{name}")


def test_codex_available_reflects_real_which():
    import shutil

    assert codex_available() == (shutil.which("codex") is not None)


def test_execute_returns_agerror_when_binary_missing(monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: None)
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    result, ctx, delta = backend.execute(ag, agcontext(), agdata(x=1), None, skill=skill)
    assert isinstance(result, agerror)
    assert "not found on PATH" in result.error


def test_execute_parses_ndjson_agent_message():
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    prev_ctx = agcontext()

    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "t1"}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}}),
        ]
    )
    handle = _make_handle(stdout=stdout)
    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as mock_gateway_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox") as mock_wire,
    ):
        mock_gateway, apply = _patched_gateway_and_ptrace(handle)
        apply(mock_gateway_getter, mock_px_cls)
        result, ctx, delta = backend.execute(ag, prev_ctx, agdata(task="go"), None, skill=skill)

    assert not isinstance(result, agerror)
    assert result.result == "hi"
    assert ctx is prev_ctx
    mock_wire.assert_called_once_with(handle, ag.sandbox)
    mock_gateway.register.assert_called_once()
    mock_gateway.unregister.assert_called_once()


def test_execute_nonzero_exit_returns_agerror():
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    prev_ctx = agcontext()

    handle = _make_handle(stdout="", stderr="boom", rc=1)
    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as mock_gateway_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox"),
    ):
        mock_gateway, apply = _patched_gateway_and_ptrace(handle)
        apply(mock_gateway_getter, mock_px_cls)
        result, ctx, delta = backend.execute(ag, prev_ctx, agdata(task="go"), None, skill=skill)

    assert isinstance(result, agerror)
    assert "boom" in result.error


def test_execute_recovers_structured_output_schema():
    backend = _CodexBackend(agConfig())
    skill = agskill(
        name="s", system_prompt="do the thing", output_schema=agdata(answer=str)
    )
    ag = _make_agent()
    prev_ctx = agcontext()

    stdout = json.dumps(
        {"type": "item.completed", "item": {"type": "agent_message", "text": '{"answer": "42"}'}}
    )
    handle = _make_handle(stdout=stdout)
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


def test_execute_writes_model_providers_config_toml():
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()

    handle = _make_handle(stdout='{"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}}')
    written = {}

    def fake_launch(argv, envp, *, cwd, policy, ag):
        from pathlib import Path

        written["envp"] = envp
        written["content"] = (Path(cwd) / "config.toml").read_text()
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

    content = written["content"]
    assert 'model_provider = "agency-proxy"' in content
    assert 'wire_api = "responses"' in content
    assert mock_gateway.base_url in content
    assert "test-model" in content
    # the gateway token is passed via the env var config.toml's env_key names,
    # not embedded directly in the file.
    assert written["envp"]["AGENCY_PROXY_API_KEY"]
    assert written["envp"]["AGENCY_PROXY_API_KEY"] not in content


def test_parse_output_events_ignores_non_agent_message_items():
    stdout = "\n".join(
        [
            json.dumps({"type": "item.completed", "item": {"type": "reasoning", "text": "thinking"}}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "final"}}),
        ]
    )
    assert _CodexBackend._parse_output_events(stdout) == "final"


def test_parse_output_events_falls_back_to_raw_text():
    assert _CodexBackend._parse_output_events("plain text") == "plain text"
