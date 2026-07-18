"""Tests for the Claude Code agharness_backend.

Tier 1 (mocked agProxyPtrace.launch) covers the orchestration logic
identically to test_opencode.py. Tier 2 (marked `real_claude`) runs the
actual installed `claude` CLI end-to-end -- verified working (v2.1.212)
during development, both raw-text and structured-output-schema paths; kept
here as a regression check, skipped when the binary/auth isn't available
so this suite doesn't require real API access to run.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agency.agconfig import agConfig
from agency.agsandbox_backends import agSandboxBackendConfig
from agency.agdata import agdata, agerror
from agency.agcontext import agcontext
from agency.agent import agent
from agency.agharness_internal.agharness_backends.claude_code import _ClaudeCodeBackend, claude_code_available
from agency.agskill import agskill


def _make_agent(with_sandbox=True):
    ag = MagicMock()
    ag.agconfig = agConfig()
    ag.sandbox = MagicMock() if with_sandbox else None
    return ag


def _make_handle(stdout="", stderr="", rc=0):
    handle = MagicMock()
    handle.wait.return_value = (stdout, stderr, rc)
    return handle


@pytest.fixture
def _patch_which_finds_claude(monkeypatch):
    """Explicitly requested (NOT autouse) -- the real_claude-marked tests
    below must see the genuine shutil.which("claude") result, not a fake
    path, or execve() fails with FileNotFoundError (hit during development:
    an earlier autouse version of this fixture broke the real-CLI tests by
    patching `which` out from under them too)."""
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: f"/usr/bin/{name}")


def test_execute_returns_agerror_when_binary_missing(monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: None)
    backend = _ClaudeCodeBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    result, ctx, delta = backend.execute(ag, agcontext(), agdata(x=1), None, skill=skill)
    assert isinstance(result, agerror)
    assert "not found on PATH" in result.error


def test_execute_parses_json_result_field(_patch_which_finds_claude):
    backend = _ClaudeCodeBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    prev_ctx = agcontext()

    payload = json.dumps({"result": "Hi there!", "usage": {"input_tokens": 5, "output_tokens": 2}})
    handle = _make_handle(stdout=payload)
    with (
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox") as mock_wire,
    ):
        mock_px_cls.return_value.launch.return_value = handle
        result, ctx, delta = backend.execute(ag, prev_ctx, agdata(task="go"), None, skill=skill)

    assert not isinstance(result, agerror)
    assert result.result == "Hi there!"
    assert ctx is prev_ctx
    assert ctx.total_input_tokens == 5
    assert ctx.total_output_tokens == 2
    mock_wire.assert_called_once_with(handle, ag.sandbox)


def test_execute_does_not_override_home(monkeypatch, _patch_which_finds_claude):
    """Regression test for a real bug hit during development: overriding
    HOME cut Claude Code off from its own ~/.claude/.credentials.json,
    forcing "Not logged in" on every run."""
    monkeypatch.setenv("HOME", "/real/home")
    backend = _ClaudeCodeBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()

    handle = _make_handle(stdout='{"result": "ok"}')
    captured_envp = {}

    def fake_launch(argv, envp, *, cwd, policy, ag):
        captured_envp.update(envp)
        return handle

    with (
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox"),
    ):
        mock_px_cls.return_value.launch.side_effect = fake_launch
        backend.execute(ag, agcontext(), agdata(task="go"), None, skill=skill)

    assert captured_envp.get("HOME") == "/real/home"


def test_execute_nonzero_exit_returns_agerror(_patch_which_finds_claude):
    backend = _ClaudeCodeBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    prev_ctx = agcontext()

    handle = _make_handle(stdout="", stderr="auth error", rc=1)
    with (
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox"),
    ):
        mock_px_cls.return_value.launch.return_value = handle
        result, ctx, delta = backend.execute(ag, prev_ctx, agdata(task="go"), None, skill=skill)

    assert isinstance(result, agerror)
    assert "auth error" in result.error


def test_execute_recovers_structured_output_schema(_patch_which_finds_claude):
    backend = _ClaudeCodeBackend(agConfig())
    skill = agskill(
        name="s", system_prompt="do the thing", output_schema=agdata(greeting=str, word_count=int)
    )
    ag = _make_agent()
    prev_ctx = agcontext()

    payload = json.dumps({"result": '{"greeting": "hi there friend", "word_count": 3}'})
    handle = _make_handle(stdout=payload)
    with (
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox"),
    ):
        mock_px_cls.return_value.launch.return_value = handle
        result, ctx, delta = backend.execute(ag, prev_ctx, agdata(task="go"), None, skill=skill)

    assert not isinstance(result, agerror)
    assert result.greeting == "hi there friend"
    assert result.word_count == 3


def test_parse_result_json_extracts_result_and_usage():
    payload = json.dumps({"result": "abc", "usage": {"input_tokens": 1, "output_tokens": 2}})
    text, usage = _ClaudeCodeBackend._parse_result_json(payload)
    assert text == "abc"
    assert usage == {"input_tokens": 1, "output_tokens": 2}


def test_parse_result_json_falls_back_on_malformed_json():
    text, usage = _ClaudeCodeBackend._parse_result_json("not json")
    assert text == "not json"
    assert usage == {}


# ---------------------------------------------------------------------------
# Tier 2: real `claude` CLI (skipped unless the binary + auth are present)
# ---------------------------------------------------------------------------

real_claude = pytest.mark.skipif(
    not claude_code_available(), reason="claude CLI not installed on this host"
)


@real_claude
def test_real_claude_raw_text_end_to_end():
    cfg = agConfig(
        agSandboxBackendConfig(backend="docker"),
        {"agllm_backend": {"api_key": "unused", "model": "unused"}},
    )
    ag = agent(agconfig=cfg, engine="claude_code")
    skill = agskill(
        name="two_word_greeting_test",
        system_prompt="Respond with exactly the two words requested, nothing else.",
    )
    result = ag.run(skill, agdata(instruction="Say hi in exactly two words."))
    result.wait()
    raw = result.to_dict()
    assert "error" not in raw, raw
    assert isinstance(raw.get("result"), str) and raw["result"]


@real_claude
def test_real_claude_structured_output_end_to_end():
    cfg = agConfig(
        agSandboxBackendConfig(backend="docker"),
        {"agllm_backend": {"api_key": "unused", "model": "unused"}},
    )
    ag = agent(agconfig=cfg, engine="claude_code")
    skill = agskill(
        name="structured_greeting_test",
        system_prompt="You produce a structured greeting.",
        output_schema=agdata(greeting=str, word_count=int),
    )
    result = ag.run(
        skill, agdata(instruction="Greet the user with exactly 3 words, then report the count.")
    )
    result.wait()
    raw = result.to_dict()
    assert "error" not in raw, raw
    assert isinstance(raw.get("greeting"), str)
    assert isinstance(raw.get("word_count"), int)
