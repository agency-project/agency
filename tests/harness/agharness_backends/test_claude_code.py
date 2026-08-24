"""Tests for the Claude Code agharness_backend.

Tier 1 (mocked `agProxyPtrace.launch`) covers `_ClaudeCodeBackend.
_run_attempt()` -- the one thing genuinely specific to this engine (binary
resolution, argv/env construction, hook enabling, output parsing). The
retry loop, structured-output collection, session capture, and transcript
building it runs inside are all SHARED logic now (`agharness_backends/
base.py`'s `execute()` template method) and are tested once, generically,
in test_base.py's `TestSharedExecuteTemplate` instead of being re-tested
per engine here.

Tier 2 (marked `real_claude`) runs the actual installed `claude` CLI end-to-
end against a REAL backend (Amazon Bedrock, via AWS_BEARER_TOKEN_BEDROCK)
-- verified working (v2.1.212/v2.1.220) during development: raw-text,
structured-output-schema, and cross-container session-continuity paths,
with the request genuinely translated and routed through agmanager_harness
rather than Claude Code using its own host credentials. Kept here as a
regression check, skipped when the binary/auth isn't available so this
suite doesn't require real API access to run.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agency.agconfig import agConfig
from agency.sandbox import agSandboxBackendConfig
from agency.llm import agBedrockBackendConfig
from agency.agdata import agdata
from agency.agent import agent
from agency.harness.adapters.claude_code import (
    _ClaudeCodeBackend,
    claude_code_available,
)
from agency.agskill import agskill


def _make_agent(with_sandbox=True):
    ag = MagicMock()
    ag.agname = "test-agent"
    ag.agconfig = agConfig()
    ag.llm.backend.model = "test-model"
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


def _run_attempt(
    backend,
    ag,
    *,
    harness_base_url="http://harness.local",
    token="tok-1",
    prompt="go",
    resume_session_id=None,
    prior_session_blob=None,
):
    skill = agskill(name="s", system_prompt="do the thing")
    return backend._run_attempt(
        ag,
        None,
        harness_base_url,
        SimpleNamespace(token=token),
        skill,
        prompt=prompt,
        resume_session_id=resume_session_id,
        prior_session_blob=prior_session_blob,
        max_steps=None,
    )


def test_run_attempt_returns_error_when_binary_missing(monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: None)
    backend = _ClaudeCodeBackend(agConfig())
    ag = _make_agent(with_sandbox=False)

    attempt = _run_attempt(backend, ag)

    assert not attempt.ok
    assert "not found" in attempt.error_message


def test_run_attempt_parses_json_result_field(_patch_which_finds_claude):
    backend = _ClaudeCodeBackend(agConfig())
    ag = _make_agent(with_sandbox=False)

    payload = json.dumps({"result": "Hi there!", "usage": {"input_tokens": 5, "output_tokens": 2}})
    handle = _make_handle(stdout=payload)
    with patch("agency.harness.ptrace.supervisor.agProxyPtrace") as mock_px_cls:
        mock_px_cls.return_value.launch.return_value = handle
        attempt = _run_attempt(backend, ag)

    assert attempt.ok
    assert attempt.final_text == "Hi there!"
    assert attempt.input_tokens == 5
    assert attempt.output_tokens == 2


def test_run_attempt_does_not_override_home(monkeypatch, _patch_which_finds_claude):
    """Regression test for a real bug hit during development: overriding
    HOME cut Claude Code off from its own ~/.claude/.credentials.json,
    forcing "Not logged in" on every run."""
    monkeypatch.setenv("HOME", "/real/home")
    backend = _ClaudeCodeBackend(agConfig())
    ag = _make_agent(with_sandbox=False)

    handle = _make_handle(stdout='{"result": "ok"}')
    captured_envp = {}
    captured_argv = []

    def fake_launch(argv, envp, *, cwd, policy, ag):
        captured_argv.extend(argv)
        captured_envp.update(envp)
        return handle

    with patch("agency.harness.ptrace.supervisor.agProxyPtrace") as mock_px_cls:
        mock_px_cls.return_value.launch.side_effect = fake_launch
        _run_attempt(backend, ag, harness_base_url="http://harness.local", token="tok-1")

    assert captured_envp.get("HOME") == "/real/home"
    # LLM traffic is routed through agmanager_harness, not the host's own creds.
    assert captured_envp.get("ANTHROPIC_BASE_URL") == "http://harness.local"
    assert captured_envp.get("ANTHROPIC_AUTH_TOKEN") == "tok-1"
    assert "ANTHROPIC_API_KEY" not in captured_envp
    settings = json.loads(captured_argv[captured_argv.index("--settings") + 1])
    assert set(settings["hooks"]) == {"PreToolUse"}
    assert "AGPROF_BASE_URL" not in captured_envp


def test_run_attempt_enables_exact_claude_hooks_only_while_profiling(_patch_which_finds_claude):
    backend = _ClaudeCodeBackend(agConfig())
    ag = _make_agent(with_sandbox=False)
    handle = _make_handle(stdout='{"result": "ok"}')
    captured = {}

    def fake_launch(argv, envp, *, cwd, policy, ag):
        captured["argv"] = argv
        captured["envp"] = envp
        return handle

    with (
        patch("agency.harness.ptrace.supervisor.agProxyPtrace") as ptrace_cls,
        patch("agency.profiler.agprof.enabled", return_value=True),
    ):
        ptrace_cls.return_value.launch.side_effect = fake_launch
        _run_attempt(backend, ag, harness_base_url="http://harness.local", token="tok-1")

    settings = json.loads(captured["argv"][captured["argv"].index("--settings") + 1])
    assert set(settings["hooks"]) == {"PreToolUse", "PostToolUse", "PostToolUseFailure"}
    assert captured["envp"]["AGPROF_BASE_URL"] == "http://harness.local"
    assert captured["envp"]["AGPROF_TOKEN"] == "tok-1"


def test_run_attempt_nonzero_exit_returns_error(_patch_which_finds_claude):
    backend = _ClaudeCodeBackend(agConfig())
    ag = _make_agent(with_sandbox=False)

    handle = _make_handle(stdout="", stderr="auth error", rc=1)
    with patch("agency.harness.ptrace.supervisor.agProxyPtrace") as mock_px_cls:
        mock_px_cls.return_value.launch.return_value = handle
        attempt = _run_attempt(backend, ag)

    assert not attempt.ok
    assert "auth error" in attempt.error_message


def test_run_attempt_threads_resume_session_id_into_argv(_patch_which_finds_claude):
    """`resume_session_id` is decided by the SHARED retry loop in base.py
    (see test_base.py's TestSharedExecuteTemplate) and simply threaded
    through into `--resume <id>` here -- this only tests that threading,
    not the retry loop itself."""
    backend = _ClaudeCodeBackend(agConfig())
    ag = _make_agent(with_sandbox=False)
    handle = _make_handle(stdout=json.dumps({"result": "done", "session_id": "sess-abc"}))
    captured = {}

    def fake_launch(argv, envp, *, cwd, policy, ag):
        captured["argv"] = argv
        return handle

    with patch("agency.harness.ptrace.supervisor.agProxyPtrace") as mock_px_cls:
        mock_px_cls.return_value.launch.side_effect = fake_launch
        attempt = _run_attempt(backend, ag, resume_session_id="sess-abc")

    assert attempt.ok
    assert "--resume" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--resume") + 1] == "sess-abc"


def test_run_attempt_omits_resume_flag_for_a_fresh_session(_patch_which_finds_claude):
    backend = _ClaudeCodeBackend(agConfig())
    ag = _make_agent(with_sandbox=False)
    handle = _make_handle(stdout='{"result": "ok"}')
    captured = {}

    def fake_launch(argv, envp, *, cwd, policy, ag):
        captured["argv"] = argv
        return handle

    with patch("agency.harness.ptrace.supervisor.agProxyPtrace") as mock_px_cls:
        mock_px_cls.return_value.launch.side_effect = fake_launch
        _run_attempt(backend, ag, resume_session_id=None)

    assert "--resume" not in captured["argv"]


def test_parse_result_json_extracts_result_and_usage():
    payload = json.dumps(
        {
            "result": "abc",
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "session_id": "session-123",
        }
    )
    text, usage, session_id = _ClaudeCodeBackend._parse_result_json(payload)
    assert text == "abc"
    assert usage == {"input_tokens": 1, "output_tokens": 2}
    assert session_id == "session-123"


def test_parse_result_json_falls_back_on_malformed_json():
    text, usage, session_id = _ClaudeCodeBackend._parse_result_json("not json")
    assert text == "not json"
    assert usage == {}
    assert session_id is None


# ---------------------------------------------------------------------------
# Tier 2: real `claude` CLI (skipped unless the binary + auth are present)
# ---------------------------------------------------------------------------

real_claude = pytest.mark.skipif(
    not claude_code_available(), reason="claude CLI not installed on this host"
)


@real_claude
def test_real_claude_raw_text_end_to_end():
    # A genuinely-working backend, not a placeholder -- this backend routes
    # claude's LLM traffic through agmanager_harness's `/v1/messages` route
    # to this agent's own agmanager_host, so the real credentialed dispatch
    # must actually happen, not just accept the connection. Uses the same
    # Bedrock bearer-token credential (AWS_BEARER_TOKEN_BEDROCK) this dev
    # environment already has.
    cfg = agConfig(
        agSandboxBackendConfig(backend="docker"),
        agBedrockBackendConfig(model="us.anthropic.claude-sonnet-5"),
    )

    ag = agent(agconfig=cfg, harness="claude_code")
    skill = agskill(
        name="two_word_greeting_test",
        system_prompt="Respond with exactly the two words requested, nothing else.",
    )
    result = ag.run(skill, agdata(instruction="Say hi in exactly two words."))
    result.wait()
    raw = result.to_dict()

    # The correct answer alone doesn't prove the real backend was actually
    # used -- HOME is deliberately left untouched (see this backend's
    # docstring), so a real OAuth-logged-in `claude` on this host could in
    # principle answer correctly via its OWN credentials if
    # ANTHROPIC_BASE_URL/AUTH_TOKEN were somehow ignored. Assert directly
    # against this agent's own agmanager_host request log -- the one place
    # a real credentialed dispatch is ever recorded -- instead of inferring
    # "it must have gone through" from the result looking right.
    request_log = ag._host_agent_manager.request_log
    assert len(request_log) >= 1, "claude's request never reached agmanager_host"
    assert all(e["model"] == "us.anthropic.claude-sonnet-5" for e in request_log)
    assert "error" not in raw, raw
    assert isinstance(raw.get("result"), str) and raw["result"]


@real_claude
def test_real_claude_tool_call_history_is_not_flattened():
    """Phase 5: a real run that uses a tool must produce a `ctx.messages`
    with the actual tool-call/tool-result turns in it -- not the old
    2-message [user, final-assistant-text] collapse, which would silently
    discard exactly this kind of turn."""
    cfg = agConfig(
        agSandboxBackendConfig(backend="docker"),
        agBedrockBackendConfig(model="us.anthropic.claude-sonnet-5"),
    )
    ag = agent(agconfig=cfg, harness="claude_code")
    skill = agskill(
        name="claude_tool_history_test",
        system_prompt=(
            "You have a bash tool. Use it to run the exact command the user "
            "gives you, then report its output back in one short sentence."
        ),
    )
    result = ag.run(skill, agdata(instruction="Run: echo agency-history-marker"))
    result.wait()
    raw = result.to_dict()

    assert "error" not in raw, raw
    assert "agency-history-marker" in raw.get("result", "")
    # ag.ctx is a future-backed placeholder until resolved -- reading
    # .messages directly would just see the unresolved default `[]`.
    messages = ag.ctx.get_resolved_messages()
    assert any(m.get("role") == "tool" for m in messages), (
        "no tool-role message in ctx.messages -- history fell back to the "
        f"flattened 2-message shape instead of the real transcript: {messages}"
    )
    assert len(messages) > 2, "transcript should have more than [user, assistant]"


@real_claude
def test_real_claude_structured_output_end_to_end():
    # A genuinely-working backend, not a placeholder -- see
    # test_real_claude_raw_text_end_to_end's comment for why the request
    # log check is against this agent's own agmanager_host.
    cfg = agConfig(
        agSandboxBackendConfig(backend="docker"),
        agBedrockBackendConfig(model="us.anthropic.claude-sonnet-5"),
    )

    ag = agent(agconfig=cfg, harness="claude_code")
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

    request_log = ag._host_agent_manager.request_log
    assert len(request_log) >= 1, "claude's request never reached agmanager_host"
    assert all(e["model"] == "us.anthropic.claude-sonnet-5" for e in request_log)
    assert "error" not in raw, raw
    assert isinstance(raw.get("greeting"), str)
    assert isinstance(raw.get("word_count"), int)


@real_claude
def test_real_claude_history_continues_across_a_fresh_sandbox():
    """Session continuity (docs/Design_harness_history.md) travels with the
    AGENT (`ag._harness_sessions`), not with any particular container: call 1
    tells the agent a fact, then `ag.sandbox` is swapped for a brand-new
    sandbox (a different container instance) before call 2 asks the agent to
    recall that fact via `--resume` against the captured session blob."""
    cfg = agConfig(
        agSandboxBackendConfig(backend="docker"),
        agBedrockBackendConfig(model="us.anthropic.claude-sonnet-5"),
    )
    skill = agskill(name="continuity_test_skill", system_prompt="You are a test assistant.")

    from agency.sandbox.agsandbox import agSandbox
    import uuid

    ag = agent(
        agconfig=cfg, sandbox=agSandbox(str(uuid.uuid4()), agconfig=cfg), harness="claude_code"
    )
    try:
        r1 = ag.run(
            skill,
            agdata(
                instruction="My project's deployment codename is PURPLE-42-NARWHAL. Just say OK."
            ),
        )
        r1.wait()
        raw1 = r1.to_dict()
        assert "error" not in raw1, raw1

        stored = ag._harness_sessions.get("claude_code")
        assert stored and stored.get("session_id"), "no session captured after call 1"
    finally:
        ag.sandbox.rm_container()

    # Fresh container for call 2 -- proves continuity doesn't depend on
    # reusing the first container.
    ag.sandbox = agSandbox(str(uuid.uuid4()), agconfig=cfg)
    try:
        r2 = ag.run(
            skill,
            agdata(
                instruction="What's my project's deployment codename? Reply with just the codename."
            ),
        )
        r2.wait()
        raw2 = r2.to_dict()

        assert "error" not in raw2, raw2
        assert "PURPLE-42-NARWHAL" in raw2.get("result", ""), (
            f"continuity failed -- agent didn't recall the code: {raw2!r}"
        )
    finally:
        ag.sandbox.rm_container()
