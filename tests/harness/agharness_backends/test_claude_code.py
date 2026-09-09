"""Claude native launch configuration and opt-in provider integration tests."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest

from agency.configs.agconfig import agconfig, llmconfig, sandboxconfig
from agency.agdata import agdata
from agency.agent import agent
from agency.harness.adapters.claude_code import _ClaudeCodeBackend, claude_code_available
from agency.harness.adapters.agharness_backend import AdapterRuntime
from agency.agskill import agskill


def test_native_launch_uses_attempt_credential_and_lifecycle_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr("agency.harness.agharness.materialize_config_home", lambda *a: tmp_path)
    config = agconfig()
    runtime = AdapterRuntime(
        config, "test-model", "agent", "http://daemon", "attempt-key", MagicMock()
    )
    argv, env, config_home = _ClaudeCodeBackend(config).prepare_pty(runtime)
    assert argv[argv.index("--permission-mode") + 1] == "bypassPermissions"
    assert json.loads((config_home / ".claude.json").read_text())["bypassPermissionsModeAccepted"]
    assert "-p" not in argv
    assert "--output-format" not in argv
    assert env["ANTHROPIC_AUTH_TOKEN"] == env["AGPOLICY_TOKEN"] == "attempt-key"
    assert env["CLAUDE_CONFIG_DIR"] == str(config_home)
    assert env["IS_SANDBOX"] == "1"
    hooks = json.loads(argv[argv.index("--settings") + 1])["hooks"]
    assert set(hooks) >= {
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "SessionStart",
        "UserPromptSubmit",
        "Stop",
        "StopFailure",
        "SessionEnd",
    }
    commands = {entry["hooks"][0]["command"] for entries in hooks.values() for entry in entries}
    assert commands == {
        f"python3 {config_home}/agpolicy_hook.py",
        f"python3 {config_home}/claude_pty_hook.py",
    }
    assert not (config_home / "agency_lifecycle.py").exists()
    assert (config_home / "agency-turn.json").exists()


def test_native_launch_restores_conversation_before_resuming(tmp_path, monkeypatch):
    monkeypatch.setattr("agency.harness.agharness.materialize_config_home", lambda *a: tmp_path)
    from agency.harness.adapters.claude_code import _session_path

    config = agconfig()
    runtime = AdapterRuntime(
        config, "test-model", "agent", "http://daemon", "attempt-key", MagicMock()
    )
    argv, _, _ = _ClaudeCodeBackend(config).prepare_pty(
        runtime, resume_session_id="native-session", prior_session_blob=b'{"type":"user"}\n'
    )
    assert argv[-2:] == ["--resume", "native-session"]
    from pathlib import Path

    assert Path(_session_path(str(tmp_path), "native-session")).read_bytes() == b'{"type":"user"}\n'


real_claude = pytest.mark.skipif(
    not claude_code_available() or os.environ.get("AGENCY_TEST_REAL_CLAUDE") != "1",
    reason="explicit paid-provider integration opt-in required",
)


@real_claude
def test_real_claude_raw_text_end_to_end():
    # A genuinely-working backend, not a placeholder -- this backend routes
    # claude's LLM traffic through agmanager_harness's `/v1/messages` route
    # to this agent's own agmanager_host, so the real credentialed dispatch
    # must actually happen, not just accept the connection. Uses the same
    # Bedrock bearer-token credential (AWS_BEARER_TOKEN_BEDROCK) this dev
    # environment already has.
    cfg = agconfig(
        sandboxconfig(backend="docker"),
        llmconfig(provider="bedrock", model="us.anthropic.claude-sonnet-5"),
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
    """Phase 5: a real run that uses a tool must produce a `context.messages`
    with the actual tool-call/tool-result turns in it -- not the old
    2-message [user, final-assistant-text] collapse, which would silently
    discard exactly this kind of turn."""
    cfg = agconfig(
        sandboxconfig(backend="docker"),
        llmconfig(provider="bedrock", model="us.anthropic.claude-sonnet-5"),
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
    # ag.context is a future-backed placeholder until resolved -- reading
    # .messages directly would just see the unresolved default `[]`.
    messages = ag.context.get_resolved_messages()
    assert any(m.get("role") == "tool" for m in messages), (
        "no tool-role message in context.messages -- history fell back to the "
        f"flattened 2-message shape instead of the real transcript: {messages}"
    )
    assert len(messages) > 2, "transcript should have more than [user, assistant]"


@real_claude
def test_real_claude_structured_output_end_to_end():
    # A genuinely-working backend, not a placeholder -- see
    # test_real_claude_raw_text_end_to_end's comment for why the request
    # log check is against this agent's own agmanager_host.
    cfg = agconfig(
        sandboxconfig(backend="docker"),
        llmconfig(provider="bedrock", model="us.anthropic.claude-sonnet-5"),
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
    """Session continuity travels with the AGENT
    (`ag.context.harness_sessions`), not with any particular container:
    call 1 tells the agent a fact, then `ag.sandbox` is swapped for a
    brand-new sandbox (a different container instance) before call 2 asks
    the agent to recall that fact via `--resume` against the captured
    session blob."""
    cfg = agconfig(
        sandboxconfig(backend="docker"),
        llmconfig(provider="bedrock", model="us.anthropic.claude-sonnet-5"),
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

        stored = ag.context.harness_sessions.get("claude_code")
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
