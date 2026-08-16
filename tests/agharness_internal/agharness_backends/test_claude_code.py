"""Tests for the Claude Code agharness_backend.

Tier 1 (mocked agProxyPtrace.launch and agproxy_llm.get_shared_gateway)
covers the orchestration logic identically to test_opencode.py. Tier 2
(marked `real_claude`) runs the actual installed `claude` CLI end-to-end
against a REAL backend (Amazon Bedrock, via AWS_BEARER_TOKEN_BEDROCK) --
verified working (v2.1.212) during development, both raw-text and
structured-output-schema paths, with the request genuinely translated and
routed through agproxy_llm's `/v1/messages` route rather than Claude Code
using its own host credentials. Kept here as a regression check, skipped
when the binary/auth isn't available so this suite doesn't require real API
access to run.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agency.agconfig import agConfig
from agency.agsandbox_backends import agSandboxBackendConfig
from agency.agllm_backends import agBedrockBackendConfig
from agency.agdata import agdata, agerror
from agency.agcontext import agcontext
from agency.agent import agent
from agency.agharness_internal.agharness_backends.claude_code import (
    _ClaudeCodeBackend,
    claude_code_available,
)
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
    mock_wire.assert_called_once_with(handle, ag.sandbox)
    mock_gateway.register.assert_called_once()
    mock_gateway.unregister.assert_called_once()


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
    captured_argv = []
    captured_stdin = []

    def fake_launch(argv, envp, *, cwd, policy, ag, sandbox=None, stdin=None):
        captured_argv.extend(argv)
        captured_envp.update(envp)
        captured_stdin.append(stdin)
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

    assert captured_envp.get("HOME") == "/real/home"
    # LLM traffic is routed through the gateway, not the host's own creds.
    assert captured_envp.get("ANTHROPIC_BASE_URL") == mock_gateway.base_url
    assert "ANTHROPIC_API_KEY" not in captured_envp
    assert captured_stdin and "[SYSTEM INSTRUCTIONS]\ndo the thing" in captured_stdin[0]
    assert all("New Skill Input" not in arg for arg in captured_argv)
    settings = json.loads(captured_argv[captured_argv.index("--settings") + 1])
    assert set(settings["hooks"]) == {"PreToolUse"}
    assert "AGPROF_BASE_URL" not in captured_envp


def test_execute_enables_exact_claude_hooks_only_while_profiling(
    _patch_which_finds_claude,
):
    backend = _ClaudeCodeBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    handle = _make_handle(stdout='{"result": "ok"}')
    captured = {}

    def fake_launch(argv, envp, *, cwd, policy, ag, sandbox=None, stdin=None):
        captured["argv"] = argv
        captured["envp"] = envp
        return handle

    profiler_ingest = MagicMock()
    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as gateway_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as ptrace_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox"),
        patch(
            "agency.agharness_internal.agprof_ingest.get_shared_profiler_ingest",
            return_value=profiler_ingest,
        ),
        patch("agency.profiler.agprof.enabled", return_value=True),
    ):
        gateway, _ = _patched_gateway_and_ptrace(handle)
        gateway_getter.return_value = gateway
        ptrace_cls.return_value.launch.side_effect = fake_launch
        backend.execute(ag, agcontext(), agdata(task="go"), None, skill=skill)

    settings = json.loads(captured["argv"][captured["argv"].index("--settings") + 1])
    assert set(settings["hooks"]) == {"PreToolUse", "PostToolUse", "PostToolUseFailure"}
    assert captured["envp"]["AGPROF_BASE_URL"] == gateway.base_url
    assert captured["envp"]["AGPROF_TOKEN"]
    profiler_ingest.register.assert_called_once()
    assert profiler_ingest.register.call_args.kwargs == {"exact_tool_events": True}


def test_execute_nonzero_exit_returns_agerror(_patch_which_finds_claude):
    backend = _ClaudeCodeBackend(agConfig())
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


def _patched_mcp_server(collected=None):
    """A fake agmcp_server -- structured output no longer comes from
    parsing the harness's own final text (see agharness.py's
    build_output_format_instruction and claude_code.py's execute()): it's
    whatever `submit_output` calls landed against the real server during
    the run, read back via `collected_output(token)`. Faking that return
    value here is the mocked-tier equivalent of a real submit_output call
    having happened -- test_real_claude_structured_output_end_to_end (Tier
    2) is what proves the real MCP round trip itself works."""
    mock_server = MagicMock()
    mock_server.start.return_value = "http://127.0.0.1:1"
    mock_server.ensure_uds_started.return_value = "/tmp/fake.sock"
    mock_server.collected_output.return_value = collected or {}
    return mock_server


def test_execute_recovers_structured_output_schema(_patch_which_finds_claude):
    backend = _ClaudeCodeBackend(agConfig())
    skill = agskill(
        name="s", system_prompt="do the thing", output_schema=agdata(greeting=str, word_count=int)
    )
    ag = _make_agent()
    prev_ctx = agcontext()

    handle = _make_handle(stdout=json.dumps({"result": "hi there friend"}))
    mock_mcp_server = _patched_mcp_server({"greeting": "hi there friend", "word_count": 3})
    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as mock_gateway_getter,
        patch("agency.agharness_internal.agmcp_server.get_shared_mcp_server") as mock_mcp_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox"),
    ):
        mock_gateway, apply = _patched_gateway_and_ptrace(handle)
        apply(mock_gateway_getter, mock_px_cls)
        mock_mcp_getter.return_value = mock_mcp_server
        result, ctx, delta = backend.execute(ag, prev_ctx, agdata(task="go"), None, skill=skill)

    assert not isinstance(result, agerror)
    assert result.greeting == "hi there friend"
    assert result.word_count == 3
    mock_mcp_server.register.assert_called_once()
    mock_mcp_server.unregister.assert_called_once()


def test_execute_reports_missing_fields_when_submit_output_never_called(_patch_which_finds_claude):
    """A harness that never calls submit_output (or misses a field), even
    after exhausting its retries, must surface a clear agerror, not
    silently return a partial/empty agdata."""
    backend = _ClaudeCodeBackend(agConfig())
    skill = agskill(
        name="s",
        system_prompt="do the thing",
        output_schema=agdata(greeting=str, word_count=int),
        max_output_schema_retries=2,
    )
    ag = _make_agent()
    prev_ctx = agcontext()

    handle = _make_handle(stdout=json.dumps({"result": "I forgot to submit"}))
    mock_mcp_server = _patched_mcp_server({"greeting": "hi"})
    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as mock_gateway_getter,
        patch("agency.agharness_internal.agmcp_server.get_shared_mcp_server") as mock_mcp_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox"),
    ):
        mock_gateway, apply = _patched_gateway_and_ptrace(handle)
        apply(mock_gateway_getter, mock_px_cls)
        mock_mcp_getter.return_value = mock_mcp_server
        result, ctx, delta = backend.execute(ag, prev_ctx, agdata(task="go"), None, skill=skill)

    assert isinstance(result, agerror)
    assert "word_count" in result.error
    # 1 initial attempt + 2 retries -- never gives up silently, never loops
    # unbounded either.
    assert mock_px_cls.return_value.launch.call_count == 3


def test_execute_retries_and_recovers_when_submit_output_arrives_on_retry(
    _patch_which_finds_claude,
):
    """The Phase 6 'outer' retry: a first attempt that's missing a field
    must trigger a reprompted relaunch (via --resume, not a fresh session),
    and a subsequent attempt that completes the fields must succeed --
    proving this isn't just a give-up-immediately path."""
    backend = _ClaudeCodeBackend(agConfig())
    skill = agskill(
        name="s",
        system_prompt="do the thing",
        output_schema=agdata(greeting=str, word_count=int),
        max_output_schema_retries=3,
    )
    ag = _make_agent()
    prev_ctx = agcontext()

    handle1 = _make_handle(stdout=json.dumps({"result": "partial", "session_id": "sess-abc"}))
    handle2 = _make_handle(stdout=json.dumps({"result": "done", "session_id": "sess-abc"}))
    mock_mcp_server = _patched_mcp_server()
    # First call: only "greeting" landed. Second call (after the reprompt
    # relaunch): both fields present -- simulates the model completing the
    # missing field once reminded.
    mock_mcp_server.collected_output.side_effect = [
        {"greeting": "hi"},
        {"greeting": "hi", "word_count": 2},
    ]
    captured_argvs = []

    captured_stdins = []

    def fake_launch(argv, envp, *, cwd, policy, ag, sandbox=None, stdin=None):
        captured_argvs.append(argv)
        captured_stdins.append(stdin)
        return handle1 if len(captured_argvs) == 1 else handle2

    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as mock_gateway_getter,
        patch("agency.agharness_internal.agmcp_server.get_shared_mcp_server") as mock_mcp_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox"),
    ):
        mock_gateway, _ = _patched_gateway_and_ptrace(handle1)
        mock_gateway_getter.return_value = mock_gateway
        mock_px_cls.return_value.launch.side_effect = fake_launch
        mock_mcp_getter.return_value = mock_mcp_server
        result, ctx, delta = backend.execute(ag, prev_ctx, agdata(task="go"), None, skill=skill)

    assert not isinstance(result, agerror), result
    assert result.greeting == "hi"
    assert result.word_count == 2
    assert len(captured_argvs) == 2
    # First attempt is a fresh session (no --resume yet).
    assert "--resume" not in captured_argvs[0]
    # Second attempt resumes the first's session and carries a reprompt,
    # not the original task prompt again.
    assert "--resume" in captured_argvs[1]
    assert captured_argvs[1][captured_argvs[1].index("--resume") + 1] == "sess-abc"
    assert "still missing" in captured_stdins[1].lower()


def test_execute_uses_terminus_transcript_for_history_when_available(_patch_which_finds_claude):
    """Phase 5 (history unification): once agllm_terminus has recorded a
    real transcript for this token, prev_ctx.messages/the returned delta
    must come from THAT (tool calls/results included), not the old
    flattened [user_msg, assistant_msg] collapse."""
    backend = _ClaudeCodeBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    prev_ctx = agcontext()

    handle = _make_handle(stdout=json.dumps({"result": "the command printed ok"}))
    recorded_transcript = [
        {"role": "system", "content": "claude code's own system prompt"},
        {"role": "user", "content": "run echo ok"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "ok"},
        {"role": "assistant", "content": "the command printed ok"},
    ]
    mock_terminus = MagicMock()
    mock_terminus.transcript_for_token.return_value = recorded_transcript
    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as mock_gateway_getter,
        patch(
            "agency.agharness_internal.agllm_terminus.get_shared_terminus"
        ) as mock_terminus_getter,
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as mock_px_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox"),
    ):
        mock_gateway, apply = _patched_gateway_and_ptrace(handle)
        apply(mock_gateway_getter, mock_px_cls)
        mock_terminus_getter.return_value = mock_terminus
        result, ctx, delta = backend.execute(ag, prev_ctx, agdata(task="go"), None, skill=skill)

    assert not isinstance(result, agerror)
    # System message dropped from ctx.messages (matching agskill.py's own
    # native-loop convention), but present in the returned delta.
    assert ctx.messages == recorded_transcript[1:]
    assert any(m.get("role") == "tool" for m in ctx.messages), "tool turn was flattened away"
    assert delta[0]["role"] == "system"
    assert delta[1:] == recorded_transcript[1:]


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
    # A genuinely-working backend, not a placeholder -- now that this
    # backend routes claude's LLM traffic through an in-container
    # agproxy_llm instance (Phase 2b-ii) to the host-side agllm_terminus,
    # the real credentialed dispatch must actually happen, not just accept
    # the connection. Uses the same Bedrock bearer-token credential
    # (AWS_BEARER_TOKEN_BEDROCK) this dev environment already has.
    from agency.agharness_internal.agllm_terminus import get_shared_terminus

    cfg = agConfig(
        agSandboxBackendConfig(backend="docker"),
        agBedrockBackendConfig(model="us.anthropic.claude-sonnet-5"),
    )
    terminus = get_shared_terminus(cfg)
    log_before = len(terminus.request_log)

    ag = agent(agconfig=cfg, engine="claude_code")
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
    # against the terminus's own request log -- the one place a real
    # credentialed dispatch is ever recorded, regardless of which process
    # (host-side gateway, or now an in-container agproxy_llm) routed the
    # request here -- instead of inferring "it must have gone through" from
    # the result looking right.
    new_entries = terminus.request_log[log_before:]
    assert len(new_entries) >= 1, "claude's request never reached agllm_terminus"
    assert all(e["model"] == "us.anthropic.claude-sonnet-5" for e in new_entries)
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
    ag = agent(agconfig=cfg, engine="claude_code")
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
    # log check is against agllm_terminus, not agproxy_llm's own (no longer
    # host-side-readable once its routing layer runs in-container).
    from agency.agharness_internal.agllm_terminus import get_shared_terminus

    cfg = agConfig(
        agSandboxBackendConfig(backend="docker"),
        agBedrockBackendConfig(model="us.anthropic.claude-sonnet-5"),
    )
    terminus = get_shared_terminus(cfg)
    log_before = len(terminus.request_log)

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

    new_entries = terminus.request_log[log_before:]
    assert len(new_entries) >= 1, "claude's request never reached agllm_terminus"
    assert all(e["model"] == "us.anthropic.claude-sonnet-5" for e in new_entries)
    assert "error" not in raw, raw
    assert isinstance(raw.get("greeting"), str)
    assert isinstance(raw.get("word_count"), int)
