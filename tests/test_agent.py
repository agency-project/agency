import json
import pytest
from unittest.mock import MagicMock, patch
from src.agdata import agdata
from src.agskill import agskill
from src.tool import tool
from src.agent import agent


def _direct(content: str):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def _tool_resp(name: str, args: dict, call_id: str = "c1"):
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


def make_agent(**kwargs) -> agent:
    return agent(llm_config={"api_key": "k", "model": "gpt-4o"}, **kwargs)


# ---------------------------------------------------------------------------
# Basic routing
# ---------------------------------------------------------------------------

def test_run_calls_named_agskill():
    called = []
    def fake_run(llm_cfg, inp, hist, tools, max_steps):
        called.append(inp.to_dict())
        return agdata(done=True), agdata(messages=[])

    skill = agskill(name="dowork", system_prompt="")
    skill.run = fake_run

    ag = make_agent(agskills=[skill])
    result = ag.run("dowork", agdata(task="go"))
    assert called == [{"task": "go"}]
    assert result.done is True


def test_run_unknown_agskill_returns_error():
    ag = make_agent()
    result = ag.run("nonexistent", agdata(x=1))
    assert result.error is not None
    assert "nonexistent" in result.error


def test_repr():
    ag = make_agent(agskills=[agskill("a", ""), agskill("b", "")])
    assert "a" in repr(ag) and "b" in repr(ag)


# ---------------------------------------------------------------------------
# History is shared and updated across calls
# ---------------------------------------------------------------------------

def test_history_updated_after_run():
    skill = agskill(name="s", system_prompt="")
    def fake_run(llm_cfg, inp, hist, tools, ms):
        # Real agskill.run includes existing history + new turns
        existing = list(hist._data.get("messages", []))
        new_hist = agdata(messages=existing + [
            {"role": "user", "content": inp.to_json()},
            {"role": "assistant", "content": "{}"},
        ])
        return agdata(ok=True), new_hist

    skill.run = fake_run
    ag = make_agent(agskills=[skill])

    ag.run("s", agdata(turn=1))
    assert len(ag.history.messages) == 2

    ag.run("s", agdata(turn=2))
    assert len(ag.history.messages) == 4  # each run appended 2 messages


def test_history_passed_to_agskill():
    ag = make_agent(agskills=[agskill("s", "")])
    ag.history = agdata(messages=[{"role": "user", "content": "prior"}])

    received = {}
    def fake_run(llm_cfg, inp, hist, tools, ms):
        received["hist"] = hist
        return agdata(), agdata(messages=[])
    ag.agskills[0].run = fake_run

    ag.run("s", agdata(x=1))
    assert received["hist"].messages[0]["content"] == "prior"


# ---------------------------------------------------------------------------
# Tools are passed to agskill
# ---------------------------------------------------------------------------

def test_agent_tools_passed_to_agskill():
    t = tool(name="t1", description="", fn=lambda a: agdata())
    ag = make_agent(agskills=[agskill("s", "")], tools=[t])

    received = {}
    def fake_run(llm_cfg, inp, hist, tools, ms):
        received["tools"] = tools
        return agdata(), agdata(messages=[])
    ag.agskills[0].run = fake_run

    ag.run("s", agdata())
    assert received["tools"] == [t]


# ---------------------------------------------------------------------------
# End-to-end with mocked OpenAI
# ---------------------------------------------------------------------------

def test_end_to_end_direct_answer():
    skill = agskill(name="qa", system_prompt="Answer questions.")
    ag = make_agent(agskills=[skill])

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"answer": "Paris"}')
        result = ag.run("qa", agdata(question="Capital of France?"))

    assert result.answer == "Paris"


def test_end_to_end_with_tool():
    called = []
    def calc_fn(arg: agdata) -> agdata:
        called.append(arg.to_dict())
        return agdata(result=arg.a + arg.b)  # type: ignore[operator]

    calc = tool(
        name="add",
        description="Add two numbers.",
        fn=calc_fn,
        params={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
    )
    skill = agskill(name="math", system_prompt="You are a calculator.")
    ag = make_agent(agskills=[skill], tools=[calc])

    responses = [_tool_resp("add", {"a": 3, "b": 4}), _direct('{"result": 7}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result = ag.run("math", agdata(task="add 3 and 4"))

    assert called == [{"a": 3, "b": 4}]
    assert result.result == 7


def test_multiple_agskills_coexist():
    ag = make_agent(agskills=[agskill("a", ""), agskill("b", "")])
    results = {}
    for af in ag.agskills:
        def fake(llm, inp, hist, tools, ms, _name=af.name):
            return agdata(from_skill=_name), agdata(messages=[])
        af.run = fake

    results["a"] = ag.run("a", agdata())
    results["b"] = ag.run("b", agdata())
    assert results["a"].from_skill == "a"
    assert results["b"].from_skill == "b"
