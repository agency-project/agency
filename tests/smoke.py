"""Smoke check: end-to-end agent + agskill + tools with mocked LLM."""

import json
from unittest.mock import patch
from agency.agdata import agdata
from agency.agskill import agskill
from agency.agent import agent
from agency.agtool import agtool
from agency.agconfig import agConfig

LLM_CONFIG = {"api_key": "dummy", "model": ""}
LLM_AGCONFIG = agConfig({"agllm_backend": LLM_CONFIG})


# ---------------------------------------------------------------------------
# Streaming mock helpers (agskill uses stream=True)
# ---------------------------------------------------------------------------


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.model_extra = {}
        self.reasoning_content = None


class _Choice:
    def __init__(self, delta):
        self.delta = delta


class _Usage:
    prompt_tokens = 5


class _Chunk:
    def __init__(self, content=None, tool_calls=None, usage=None):
        self.usage = usage
        self.choices = (
            [_Choice(_Delta(content, tool_calls))] if (content is not None or tool_calls) else []
        )


class _TCDelta:
    def __init__(self, name, args_json, call_id):
        self.id = call_id
        self.index = 0
        self.function = type("F", (), {"name": name, "arguments": args_json})()


def _direct(content: str) -> list:
    return [_Chunk(content=content), _Chunk(usage=_Usage())]


def _tool_call(name: str, args: dict, call_id: str = "c1") -> list:
    tc = _TCDelta(name, json.dumps(args), call_id)
    return [_Chunk(tool_calls=[tc]), _Chunk(usage=_Usage())]


# ---------------------------------------------------------------------------
# Module-level tool functions (must be picklable — no lambdas, no closures)
# ---------------------------------------------------------------------------


def _skill_tool_fn(arg: agdata) -> agdata:
    return agdata(r=1)


def _agent_tool_fn(arg: agdata) -> agdata:
    return agdata()


def smoke_write_read_cycle():
    """Skill uses write then read inside the sandbox container, then answers."""
    file_skill = agskill(
        name="file_ops",
        system_prompt="You manage files. Use write and read tools.",
    )
    # No tools= → uses default sandboxed tool list; files live in container
    ag = agent(agconfig=LLM_AGCONFIG)

    responses = [
        _tool_call("write", {"file_path": "/workspace/greeting.txt", "content": "Hello, World!"}),
        _tool_call("read", {"file_path": "/workspace/greeting.txt"}),
        _direct('{"result": "File written and read successfully"}'),
    ]

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result = ag.run(file_skill, agdata(task="write then read a greeting file"))

    assert result.result == "File written and read successfully"
    # Verify the file is in the container, not on the host
    content = ag.sandbox.read_file("/workspace/greeting.txt")
    assert content == "Hello, World!"
    print(f"  history: {len(ag.history.messages)} messages")
    return True


def smoke_history_shared_across_skills():
    """Two different skills share and accumulate history."""
    skill_a = agskill(name="a", system_prompt="Skill A")
    skill_b = agskill(name="b", system_prompt="Skill B")
    ag = agent(agconfig=LLM_AGCONFIG)

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"turn": 1}')
        ag.run(skill_a, agdata(msg="first"))

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"turn": 2}')
        ag.run(skill_b, agdata(msg="second"))

    assert len(ag.history.messages) >= 4
    return True


def smoke_skill_own_tools():
    """add_tools on a skill extends the default sandboxed tools."""
    skill_t = agtool(name="skill_tool", description="", fn=_skill_tool_fn)

    skill = agskill(name="s", system_prompt="", add_tools=[skill_t])
    ag = agent(agconfig=LLM_AGCONFIG)

    responses = [_tool_call("skill_tool", {}), _direct("{}")]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        ag.run(skill, agdata())

    tool_msgs = [m for m in ag.history.messages if m.get("role") == "tool"]
    assert any(json.loads(m["content"]) == {"r": 1} for m in tool_msgs), "skill_tool result missing"
    return True


if __name__ == "__main__":
    tests = [
        ("write/read cycle (sandboxed)", smoke_write_read_cycle),
        ("history shared across skills", smoke_history_shared_across_skills),
        ("skill owns tools", smoke_skill_own_tools),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            import traceback

            print(f"  FAIL: {name} — {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
