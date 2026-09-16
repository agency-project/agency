"""Claude native launch configuration and opt-in provider integration tests."""

from __future__ import annotations

import json
import os
import sqlite3
from unittest.mock import MagicMock

import pytest

from agency.configs.agconfig import agconfig, harnessadapterconfig, llmconfig, sandboxconfig
from agency.agdata import agdata
from agency.agent import agent
from agency.harness.adapters.claude_code import ClaudeCodeAdapter, claude_code_available
from agency.harness.adapters.base import AdapterRuntime
from agency.agskill import agskill


def test_native_launch_uses_attempt_credential_and_lifecycle_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr("agency.harness.agharness.materialize_config_home", lambda *a: tmp_path)
    config = agconfig()
    runtime = AdapterRuntime(
        config, "test-model", "agent", "http://daemon", "attempt-key", MagicMock()
    )
    config_home = tmp_path
    argv, env = ClaudeCodeAdapter(config).prepare_pty(runtime, config_home)
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
        f"python3 {config_home}/claude_lifecycle_hook.py",
    }
    assert not (config_home / "agency_lifecycle.py").exists()
    assert (config_home / "agency-turn.json").exists()


def test_native_launch_disallows_subagents_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr("agency.harness.agharness.materialize_config_home", lambda *a: tmp_path)
    config = agconfig()
    runtime = AdapterRuntime(
        config, "test-model", "agent", "http://daemon", "attempt-key", MagicMock()
    )
    argv, _ = ClaudeCodeAdapter(config).prepare_pty(runtime, tmp_path)
    idx = argv.index("--disallowedTools")
    assert argv[idx + 1] == "Agent"


def test_native_launch_omits_disallowed_tools_when_subagents_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr("agency.harness.agharness.materialize_config_home", lambda *a: tmp_path)
    config = agconfig(harnessadapterconfig(allow_subagents=True))
    runtime = AdapterRuntime(
        config, "test-model", "agent", "http://daemon", "attempt-key", MagicMock()
    )
    argv, _ = ClaudeCodeAdapter(config).prepare_pty(runtime, tmp_path)
    assert "--disallowedTools" not in argv


def test_native_launch_restores_conversation_before_resuming(tmp_path, monkeypatch):
    monkeypatch.setattr("agency.harness.agharness.materialize_config_home", lambda *a: tmp_path)
    from agency.harness.adapters.claude_code import _session_path

    config = agconfig()
    runtime = AdapterRuntime(
        config, "test-model", "agent", "http://daemon", "attempt-key", MagicMock()
    )
    argv, _ = ClaudeCodeAdapter(config).prepare_pty(
        runtime,
        tmp_path,
        resume_session_id="native-session",
        prior_session_blob=b'{"type":"user"}\n',
    )
    assert argv[-2:] == ["--resume", "native-session"]
    from pathlib import Path

    assert Path(_session_path(str(tmp_path), "native-session")).read_bytes() == b'{"type":"user"}\n'


pytestmark = pytest.mark.timeout(240)


real_claude = pytest.mark.skipif(
    not claude_code_available() or os.environ.get("AGENCY_TEST_REAL_CLAUDE") != "1",
    reason="explicit paid-provider integration opt-in required",
)


@pytest.fixture
def real_agent():
    """Use a configured paid model through the real Claude harness and gateway.

    Set AGENCY_TEST_LLM_PROVIDER/MODEL to select a different provider; OpenAI
    uses OPENAI_API_KEY, while Bedrock uses AWS_BEARER_TOKEN_BEDROCK.
    """
    provider = os.environ.get("AGENCY_TEST_LLM_PROVIDER", "bedrock")
    model = os.environ.get("AGENCY_TEST_LLM_MODEL", "us.anthropic.claude-sonnet-5")
    kwargs = {"provider": provider, "model": model}
    if provider == "openai":
        kwargs.update(api_key=os.environ["OPENAI_API_KEY"], reasoning_effort="none")
    config = agconfig(
        sandboxconfig(
            backend="docker",
            base_image=os.environ.get(
                "AGENCY_TEST_HARNESS_IMAGE", "docker.io/library/python:3.12-slim"
            ),
        ),
        llmconfig(**kwargs),
    )
    owner = agent(agconfig=config, harness="claude_code")
    try:
        yield owner
    finally:
        if owner.sandbox is not None:
            owner.sandbox.destroy()


def _run_live(owner, skill, value):
    result = owner.run(skill, value)
    try:
        return result.wait(timeout=90).to_dict()
    finally:
        if result.is_pending():
            owner.cancel(result)
            result.wait(timeout=30)


def _assert_host_llm_exchange(owner):
    # Successful, request-tagged usage in the host logger proves the CLI used
    # Agency's model gateway instead of answering through its own credentials.
    owner.data_logger.flush()
    with sqlite3.connect(owner.data_logger.db_path) as connection:
        blocks = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT payload FROM events WHERE type = 'llm_block' AND name = ?",
                (str(owner.agname),),
            )
        ]
    assert any(
        block.get("type") == "metadata"
        and block.get("request_id")
        and (block.get("usage") or {}).get("prompt_tokens", 0) > 0
        for block in blocks
    ), "no successful credentialed exchange recorded by the host gateway"


@real_claude
def test_real_claude_raw_text_end_to_end(real_agent):
    skill = agskill(
        name="two_word_greeting_test",
        prompt="Respond with exactly the two words requested, nothing else.",
    )
    raw = _run_live(real_agent, skill, agdata(instruction="Say hi in exactly two words."))
    assert "error" not in raw, raw
    assert isinstance(raw.get("result"), str) and raw["result"]
    _assert_host_llm_exchange(real_agent)


@real_claude
def test_real_claude_tool_call_history_is_not_flattened(real_agent):
    """The resolved transcript retains actual tool calls and their results."""
    skill = agskill(
        name="claude_tool_history_test",
        prompt=(
            "You have a bash tool. Use it to run the exact command the user "
            "gives you, then report its output back in one short sentence."
        ),
    )
    raw = _run_live(real_agent, skill, agdata(instruction="Run: echo agency-history-marker"))
    assert "error" not in raw, raw
    assert "agency-history-marker" in raw.get("result", "")
    messages = real_agent.context.get_resolved_transcript()
    assert any(m.get("role") == "tool" for m in messages), messages
    assert len(messages) > 2, "transcript should include tool execution"
    _assert_host_llm_exchange(real_agent)


@real_claude
def test_real_claude_structured_output_end_to_end(real_agent):
    skill = agskill(
        name="structured_greeting_test",
        prompt="You produce a structured greeting.",
        output_schema=agdata(greeting=str, word_count=int),
    )
    raw = _run_live(
        real_agent,
        skill,
        agdata(instruction="Greet the user with exactly 3 words, then report the count."),
    )
    assert "error" not in raw, raw
    assert isinstance(raw.get("greeting"), str)
    assert raw["word_count"] == len(raw["greeting"].split()) == 3
    _assert_host_llm_exchange(real_agent)


@real_claude
def test_real_claude_history_continues_across_a_fresh_sandbox(real_agent):
    """The native session survives replacement of its original container."""
    owner = real_agent
    skill = agskill(name="continuity_test_skill", prompt="You are a test assistant.")
    first = _run_live(
        owner,
        skill,
        agdata(instruction="My project's deployment codename is PURPLE-42-NARWHAL. Just say OK."),
    )
    assert "error" not in first, first
    owner.context.resolve_prev_dependencies()
    stored = owner.context.harness_sessions.get("claude_code")
    assert stored and stored.get("session_id"), "no session captured after call 1"
    previous = owner.sandbox
    previous.destroy()
    owner.sandbox = None

    second = _run_live(
        owner,
        skill,
        agdata(
            instruction="What's my project's deployment codename? Reply with just the codename."
        ),
    )
    assert owner.sandbox is not previous
    assert "error" not in second, second
    assert "PURPLE-42-NARWHAL" in second.get("result", ""), second
    _assert_host_llm_exchange(owner)
