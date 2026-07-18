"""Tests for the opencode agharness_backend -- the reference concrete
implementation. No real `opencode` binary is exercised here (see
opencode_available(), which will be False in most environments) --
`agProxyPtrace.launch` is mocked throughout, matching the plan's Tier 1
strategy. Function-local imports inside `_OpencodeBackend.execute()` mean
these must be patched at their *source* module, not as attributes of
`agency.agharness_internal.agharness_backends.opencode`.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agency.agconfig import agConfig
from agency.agdata import agdata, agerror
from agency.agcontext import agcontext
from agency.agharness_internal.agharness_backends.opencode import _OpencodeBackend, opencode_available
from agency.agskill import agskill


def _make_agent(with_sandbox=True):
    ag = MagicMock()
    ag.agconfig = agConfig()
    ag.llm.backend.model = "test-model"
    ag.sandbox = MagicMock() if with_sandbox else None
    return ag


def _make_handle(stdout="hello world", stderr="", rc=0):
    handle = MagicMock()
    handle.wait.return_value = (stdout, stderr, rc)
    return handle


@pytest.fixture(autouse=True)
def _patch_which_finds_opencode(monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: f"/usr/bin/{name}")


def test_opencode_available_reflects_real_which():
    # Not mocked here -- exercises the real shutil.which, expected False in
    # this development environment (no node/bun to install opencode).
    import shutil

    assert opencode_available() == (shutil.which("opencode") is not None)


def test_execute_returns_agerror_when_binary_missing(monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: None)
    backend = _OpencodeBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    result, ctx, delta = backend.execute(ag, agcontext(), agdata(x=1), None, skill=skill)
    assert isinstance(result, agerror)
    assert "not found on PATH" in result.error


def test_execute_launches_via_agproxy_ptrace_and_returns_raw_text():
    backend = _OpencodeBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    prev_ctx = agcontext()

    handle = _make_handle(stdout="the final answer")
    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as mock_gateway_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox") as mock_wire,
    ):
        mock_gateway = MagicMock()
        mock_gateway.base_url = "http://127.0.0.1:1"
        mock_gateway_getter.return_value = mock_gateway
        mock_px_cls.return_value.launch.return_value = handle

        result, ctx, delta = backend.execute(ag, prev_ctx, agdata(task="go"), None, skill=skill)

    assert not isinstance(result, agerror)
    assert result.result == "the final answer"
    assert ctx is prev_ctx  # SAME object, mutated in place -- not a new one
    assert len(delta) == 3  # [system, user, assistant]
    mock_wire.assert_called_once_with(handle, ag.sandbox)
    mock_gateway.register.assert_called_once()
    mock_gateway.unregister.assert_called_once()


def test_execute_skips_wire_to_sandbox_when_no_sandbox():
    backend = _OpencodeBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent(with_sandbox=False)
    prev_ctx = agcontext()

    handle = _make_handle(stdout="ok")
    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as mock_gateway_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox") as mock_wire,
    ):
        mock_gateway = MagicMock()
        mock_gateway.base_url = "http://127.0.0.1:1"
        mock_gateway_getter.return_value = mock_gateway
        mock_px_cls.return_value.launch.return_value = handle

        backend.execute(ag, prev_ctx, agdata(task="go"), None, skill=skill)

    mock_wire.assert_not_called()


def test_execute_nonzero_exit_returns_agerror():
    backend = _OpencodeBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    prev_ctx = agcontext()

    handle = _make_handle(stdout="", stderr="boom", rc=1)
    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as mock_gateway_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox"),
    ):
        mock_gateway = MagicMock()
        mock_gateway.base_url = "http://127.0.0.1:1"
        mock_gateway_getter.return_value = mock_gateway
        mock_px_cls.return_value.launch.return_value = handle

        result, ctx, delta = backend.execute(ag, prev_ctx, agdata(task="go"), None, skill=skill)

    assert isinstance(result, agerror)
    assert "boom" in result.error
    assert ctx is prev_ctx


def test_execute_recovers_structured_output_schema():
    backend = _OpencodeBackend(agConfig())
    skill = agskill(
        name="s", system_prompt="do the thing", output_schema=agdata(answer=str)
    )
    ag = _make_agent()
    prev_ctx = agcontext()

    handle = _make_handle(stdout='{"answer": "42"}')
    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as mock_gateway_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox"),
    ):
        mock_gateway = MagicMock()
        mock_gateway.base_url = "http://127.0.0.1:1"
        mock_gateway_getter.return_value = mock_gateway
        mock_px_cls.return_value.launch.return_value = handle

        result, ctx, delta = backend.execute(ag, prev_ctx, agdata(task="go"), None, skill=skill)

    assert not isinstance(result, agerror)
    assert result.answer == "42"


def test_parse_output_events_extracts_last_text_from_ndjson():
    stdout = "\n".join(
        [
            json.dumps({"type": "turn.started"}),
            json.dumps({"type": "item.completed", "text": "partial"}),
            json.dumps({"type": "item.completed", "text": "final answer"}),
        ]
    )
    assert _OpencodeBackend._parse_output_events(stdout) == "final answer"


def test_parse_output_events_falls_back_to_raw_text():
    assert _OpencodeBackend._parse_output_events("just plain text, no JSON") == (
        "just plain text, no JSON"
    )
