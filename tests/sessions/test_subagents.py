"""Subagents become their own trajectory agents, linked to the CLI that spawned them.

Payload shapes are the ones the real CLIs sent (Claude Code 2.1.299, Codex
0.162.1): hooks fired inside a subagent carry its agent_id; Claude's model
telemetry names only the subagent's type; a Codex subagent's model telemetry
carries its agent_id as the conversation id.
"""

import json
import time

from agency.sessions.codex import CodexSession

from .helpers import FIXTURES, hook, otlp, trajectory

CLAUDE_SUBAGENT = {"agent_id": "a668abf921237672f", "agent_type": "general-purpose"}
CODEX_SUBAGENT = {"agent_id": "01a13913-0cc5-7902-8a95-cb3c2056b485", "agent_type": "default"}


def actions_of(model, agent):
    return [a for a in model.actions.values() if a["agent"] == agent]


def subagent_name(model):
    [name] = [name for name, agent in model.agents.items() if agent.get("parent")]
    return name


def spawn_claude_subagent(session):
    hook(session, "UserPromptSubmit", prompt="delegate", prompt_id="p1")
    hook(
        session,
        "PreToolUse",
        tool_name="Agent",
        tool_use_id="toolu_agent",
        tool_input={
            "description": "Run echo command",
            "prompt": "Run echo hello",
            "subagent_type": "general-purpose",
        },
    )
    hook(session, "SubagentStart", **CLAUDE_SUBAGENT)


def test_claude_subagent_is_its_own_agent_with_a_labelled_delegation(make_session):
    session = make_session()
    spawn_claude_subagent(session)
    hook(
        session,
        "PreToolUse",
        tool_name="Bash",
        tool_use_id="toolu_bash",
        tool_input={"command": "echo hello"},
        **CLAUDE_SUBAGENT,
    )
    hook(
        session,
        "PostToolUse",
        tool_name="Bash",
        tool_use_id="toolu_bash",
        tool_response={"stdout": "hello"},
        **CLAUDE_SUBAGENT,
    )
    hook(session, "SubagentStop", **CLAUDE_SUBAGENT)
    hook(session, "PostToolUse", tool_name="Agent", tool_use_id="toolu_agent", tool_response="ok")

    model = trajectory(session)
    child = subagent_name(model)
    assert child == "claude/general-purpose-1237672f"  # the last 8 of its agent_id
    assert model.agents[child]["parent"] == "claude"
    assert model.agents[child]["harness"] == "claude_code"
    assert model.agents[child]["status"] == "completed"
    [edge] = model.run["edges"]
    assert (edge["from"], edge["to"], edge["kind"]) == ("claude", child, "delegation")
    assert edge["label"] == "Run echo command"
    [bash] = actions_of(model, child)
    assert (bash["name"], bash["command"], bash["outcome"]) == ("Bash", "echo hello", "success")
    [spawn] = actions_of(model, "claude")
    assert (spawn["name"], spawn["outcome"]) == ("Agent", "success")


def test_claude_subagent_model_request_is_attributed_by_type_while_it_ran(make_session):
    session = make_session()
    spawn_claude_subagent(session)
    otlp(
        session,
        "api_request",
        time.time(),
        model="claude-sonnet-5-5",
        duration_ms=900,
        input_tokens=10,
        output_tokens=5,
        query_source="agent:builtin:general-purpose",
        **{"agent.name": "general-purpose"},
    )
    otlp(session, "api_request", time.time(), model="claude-sonnet-5-5", duration_ms=700)
    model = trajectory(session)
    child = subagent_name(model)
    [child_request] = [a for a in actions_of(model, child) if a["kind"] == "model"]
    assert child_request["metadata"]["attribution"] == "subagent_type_and_time"
    assert [a["kind"] for a in actions_of(model, "claude")].count("model") == 1


def test_model_request_stays_on_the_parent_when_two_subagents_share_a_type(make_session):
    session = make_session()
    spawn_claude_subagent(session)
    hook(session, "SubagentStart", agent_id="b000000000000000b", agent_type="general-purpose")
    otlp(
        session,
        "api_request",
        time.time(),
        model="claude-sonnet-5-5",
        duration_ms=500,
        **{"agent.name": "general-purpose"},
    )
    model = trajectory(session)
    [request] = [a for a in model.actions.values() if a["kind"] == "model"]
    assert request["agent"] == "claude"
    assert request["metadata"]["attribution"] == "subagent_ambiguous_kept_on_parent"


def test_codex_subagent_gets_its_tools_and_model_requests_by_id(make_session):
    session = make_session(CodexSession, FIXTURES / "fake_codex_session.py")
    hook(session, "UserPromptSubmit", prompt="delegate", turn_id="t1")
    hook(
        session,
        "PreToolUse",
        tool_name="collaboration.spawn_agent",
        tool_use_id="call_spawn",
        tool_input={"task_name": "five_noops", "message": "<encrypted>"},
    )
    hook(session, "PostToolUse", tool_name="collaboration.spawn_agent", tool_use_id="call_spawn")
    hook(session, "SubagentStart", **CODEX_SUBAGENT)
    for number in range(5):
        tool = {"tool_name": "Bash", "tool_use_id": f"exec-{number}", **CODEX_SUBAGENT}
        hook(session, "PreToolUse", tool_input={"command": ': "$RANDOM"'}, **tool)
        hook(session, "PostToolUse", tool_response="", **tool)
    started = time.time()
    for name, attributes in (
        ("api_request", {"duration_ms": 400, "http.response.status_code": 200}),
        ("sse_event", {"event.kind": "response.completed", "input_token_count": 99}),
    ):
        otlp(
            session,
            f"codex.{name}",
            started,
            time_in_attribute=True,
            model="gpt-6.1-sol",
            **{"conversation.id": CODEX_SUBAGENT["agent_id"]},
            **attributes,
        )
    hook(session, "SubagentStop", **CODEX_SUBAGENT)

    model = trajectory(session)
    child = subagent_name(model)
    assert child.startswith("codex/default-")
    assert model.run["edges"][0]["label"] == "five_noops"
    child_actions = actions_of(model, child)
    assert sorted(a["kind"] == "model" for a in child_actions) == [False] * 5 + [True]
    assert all(a["outcome"] == "success" for a in child_actions)


def test_interrupting_the_turn_cancels_a_running_subagent(make_session):
    session = make_session()
    spawn_claude_subagent(session)
    hook(session, "Interrupt", turn_id="t1")
    model = trajectory(session)
    assert model.agents[subagent_name(model)]["status"] == "cancelled"


def test_a_subagent_cannot_be_paused_on_its_own(make_session, tmp_path, monkeypatch):
    from agency.agent import agent
    from agency.observability.agwebui.relay import ProfilerRelay

    session = make_session()
    spawn_claude_subagent(session)
    child = subagent_name(trajectory(session))
    monkeypatch.setattr(agent, "all", classmethod(lambda cls: []))
    relay = ProfilerRelay(tmp_path / "relay")
    monkeypatch.setattr(relay, "_event", lambda *_args: None)
    try:
        relay.publish()
        capabilities = json.loads((tmp_path / "relay" / ".session").read_text())["capabilities"]
        assert child in capabilities and "claude" in capabilities
        receipt = relay.dispatch(
            {"id": "c1", "type": "pause", "targets": [child], "relay_id": relay.identifier}
        )
        [result] = receipt["targets"]
        assert result["state"] == "failed"
        assert "pause or stop claude" in result["detail"]
    finally:
        relay.close()
