"""Smoke check: end-to-end agent + agskill + tools with mocked LLM."""
import json
from unittest.mock import patch, MagicMock
from src.agdata import agdata
from src.agskill import agskill
from src.agent import agent

LLM_CONFIG = {"api_key": "dummy", "model": "gpt-4o"}


def _direct(content: str):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def _tool_call(name: str, args: dict, call_id: str = "c1"):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tc]
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def smoke_write_read_cycle():
    """Skill uses write then read inside the sandbox container, then answers."""
    file_skill = agskill(
        name="file_ops",
        system_prompt="You manage files. Use write and read tools.",
    )
    # No tools= → uses default sandboxed tool list; files live in container
    ag = agent(llm_config=LLM_CONFIG, agskills=[file_skill])

    responses = [
        _tool_call("write", {"filePath": "/workspace/greeting.txt", "content": "Hello, World!"}),
        _tool_call("read", {"filePath": "/workspace/greeting.txt"}),
        _direct('{"result": "File written and read successfully"}'),
    ]

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result = ag.run("file_ops", agdata(task="write then read a greeting file"))

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
    ag = agent(llm_config=LLM_CONFIG, agskills=[skill_a, skill_b])

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"turn": 1}')
        ag.run("a", agdata(msg="first"))

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"turn": 2}')
        ag.run("b", agdata(msg="second"))

    assert len(ag.history.messages) >= 4
    return True


def smoke_skill_own_tools():
    """A skill with its own tools list ignores agent-level tools."""
    from src.agtool import agtool

    agent_tool_called = []
    skill_tool_called = []

    agent_t = agtool(name="agent_tool", description="", fn=lambda a: (agent_tool_called.append(1) or agdata()))
    skill_t = agtool(name="skill_tool", description="", fn=lambda a: (skill_tool_called.append(1) or agdata(r=1)))

    skill = agskill(name="s", system_prompt="", tools=[skill_t])
    ag = agent(llm_config=LLM_CONFIG, agskills=[skill], tools=[agent_t])

    responses = [_tool_call("skill_tool", {}), _direct("{}")]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        ag.run("s", agdata())

    assert skill_tool_called
    assert not agent_tool_called
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
