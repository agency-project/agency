"""Tests for agskill as a self-contained ReAct skill."""
import json
import pytest
from unittest.mock import patch, MagicMock
from agency.agdata import agdata
from agency.agskill import agskill
from agency.agtool import agtool

LLM_CONFIG = {"api_key": "test", "model": "gpt-4o"}


def _noop(arg: agdata) -> agdata:
    return agdata()


def _noop_r1(arg: agdata) -> agdata:
    return agdata(r=1)

# ---------------------------------------------------------------------------
# Streaming mock helpers
# agskill uses stream=True; the mock must return a list of chunk objects.
# Using a list (not iter()) lets the same return_value be re-iterated across
# multiple calls (e.g. retry tests).
# ---------------------------------------------------------------------------

class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.model_extra = {}
        self.reasoning_content = None

class _Choice:
    def __init__(self, delta): self.delta = delta

class _Usage:
    prompt_tokens = 5

class _Chunk:
    def __init__(self, content=None, tool_calls=None, usage=None):
        self.usage = usage
        self.choices = [_Choice(_Delta(content, tool_calls))] if (content is not None or tool_calls) else []

class _TCDelta:
    def __init__(self, name, args_json, call_id):
        self.id = call_id
        self.index = 0
        self.function = _TCFnDelta(name, args_json)

class _TCFnDelta:
    def __init__(self, name, args): self.name = name; self.arguments = args


def _direct(content: str) -> list:
    """Streaming response list for a plain-text or JSON reply."""
    return [_Chunk(content=content), _Chunk(usage=_Usage())]


def _tool_call(name: str, args: dict, call_id: str = "c1") -> list:
    """Streaming response list for a tool-call reply."""
    tc = _TCDelta(name, json.dumps(args), call_id)
    return [_Chunk(tool_calls=[tc]), _Chunk(usage=_Usage())]


def make_skill(name="summarise", add_tools=None, replace_tools=None) -> agskill:
    return agskill(
        name=name,
        system_prompt="You are a summarisation assistant.",
        add_tools=add_tools,
        replace_tools=replace_tools,
    )


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

def test_name_and_repr():
    s = make_skill()
    assert s.name == "summarise"
    assert "summarise" in repr(s)


def test_run_returns_agdata_and_history():
    s = make_skill()
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"summary": "ok"}')
        result, hist, delta = s.run(LLM_CONFIG, agdata(text="hello"), agdata(messages=[]), sandbox=None)
    assert isinstance(result, agdata)
    assert isinstance(hist, agdata)
    assert isinstance(delta, list)


def test_run_direct_json_response():
    s = make_skill()
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"answer": "42"}')
        result, _, _ = s.run(LLM_CONFIG, agdata(q="6*7"), agdata(messages=[]), sandbox=None)
    assert result.answer == "42"


def test_run_plain_text_fallback():
    s = make_skill()
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct("hello world")
        result, _, _ = s.run(LLM_CONFIG, agdata(q="hi"), agdata(messages=[]), sandbox=None)
    assert result.result == "hello world"


# ---------------------------------------------------------------------------
# System prompt is sent but NOT stored in history
# ---------------------------------------------------------------------------

def test_system_prompt_prepended_to_llm_call():
    s = make_skill()
    captured = {}
    def capture(*args, **kwargs):
        captured["messages"] = kwargs.get("messages", [])
        return _direct("{}")
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = capture
        s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][0]["content"] == "You are a summarisation assistant."


def test_system_prompt_not_in_returned_history():
    s = make_skill()
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct("{}")
        _, hist, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    roles = [m["role"] for m in hist.messages]
    assert "system" not in roles


def test_existing_history_included_in_call():
    s = make_skill()
    prior = agdata(messages=[{"role": "user", "content": "prior"}, {"role": "assistant", "content": "ok"}])
    captured = {}
    def capture(*args, **kwargs):
        captured["messages"] = list(kwargs.get("messages", []))  # snapshot before list is mutated
        return _direct("{}")
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = capture
        s.run(LLM_CONFIG, agdata(x=1), prior, sandbox=None)
    # system at [0], prior messages at [1] and [2], new user at [-1]
    assert captured["messages"][1]["content"] == "prior"
    assert captured["messages"][-1]["role"] == "user"


# ---------------------------------------------------------------------------
# Tool call path
# ---------------------------------------------------------------------------

def test_tool_call_executes_and_continues():
    def fn(arg: agdata) -> agdata:
        return agdata(val=arg.x * 10)

    t = agtool(name="calc", description="", fn=fn,
             params={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]})

    s = make_skill(replace_tools=[t])
    responses = [_tool_call("calc", {"x": 7}), _direct('{"result": 70}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, hist, delta = s.run(LLM_CONFIG, agdata(task="calc"), agdata(messages=[]), sandbox=None)

    # Verify the tool ran with the right args and its output reached the LLM
    tool_msgs = [m for m in hist.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert json.loads(tool_msgs[0]["content"]) == {"val": 70}
    assert result.result == 70


def test_unknown_tool_error_in_history():
    s = make_skill()
    responses = [_tool_call("ghost", {}), _direct("{}")]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _, hist, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    tool_msgs = [m for m in hist.messages if m.get("role") == "tool"]
    assert any("unknown tool" in m["content"] for m in tool_msgs)


def test_max_steps_exceeded():
    s = make_skill()
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = lambda **kw: _tool_call("x", {})
        result, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None, max_steps=3)
    assert result.error == "max_steps exceeded"


# ---------------------------------------------------------------------------
# replace_tools / add_tools
# ---------------------------------------------------------------------------

def test_replace_tools_overrides_defaults():
    """replace_tools replaces the tool list entirely; no sandbox tools included."""
    my_tool = agtool(name="mt", description="my tool", fn=_noop_r1)
    s = agskill(name="s", system_prompt="", replace_tools=[my_tool])
    captured = {}
    def capture(**kwargs):
        captured["tools"] = kwargs.get("tools")
        return _direct("{}")
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = capture
        s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    assert captured["tools"] is not None
    assert len(captured["tools"]) == 1
    assert captured["tools"][0]["function"]["name"] == "mt"


def test_replace_tools_empty_list_gives_no_tools():
    """replace_tools=[] means no tools at all."""
    s = agskill(name="s", system_prompt="", replace_tools=[])
    captured = {}
    def capture(**kwargs):
        captured["tools"] = kwargs.get("tools")
        return _direct("{}")
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = capture
        s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    assert captured["tools"] is None


def test_add_tools_extends_sandbox_defaults():
    """add_tools appends to whatever make_sandboxed_tools returns."""
    extra = agtool(name="extra", description="extra", fn=_noop_r1)
    s = agskill(name="s", system_prompt="", add_tools=[extra])
    captured = {}
    fake_default = agtool(name="bash", description="", fn=_noop_r1)
    def fake_make_sandboxed(sandbox, pool):
        return [fake_default]
    import agency.tools as _tools_mod
    with patch("openai.OpenAI") as MockClient, \
         patch.object(_tools_mod, "make_sandboxed_tools", side_effect=fake_make_sandboxed):
        def capture(**kwargs):
            captured["tools"] = kwargs.get("tools")
            return _direct("{}")
        MockClient.return_value.chat.completions.create.side_effect = capture
        s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=object())
    names = [t["function"]["name"] for t in (captured.get("tools") or [])]
    assert "bash" in names
    assert "extra" in names


# ---------------------------------------------------------------------------
# input_schema and output_schema
# ---------------------------------------------------------------------------

def test_input_schema_missing_field_returns_error():
    s = agskill(
        name="s", system_prompt="",
        input_schema=agdata(question=str, context=str),
    )
    result, _, _ = s.run(LLM_CONFIG, agdata(question="hi"), agdata(messages=[]), sandbox=None)
    assert result.error is not None
    assert "context" in result.error


def test_input_schema_type_error_returns_error():
    s = agskill(
        name="s", system_prompt="",
        input_schema=agdata(count=int),
    )
    result, _, _ = s.run(LLM_CONFIG, agdata(count="not-an-int"), agdata(messages=[]), sandbox=None)
    assert result.error is not None
    assert "count" in result.error


def test_input_schema_valid_proceeds():
    s = agskill(
        name="s", system_prompt="",
        input_schema=agdata(text=str),
    )
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"ok": true}')
        result, _, _ = s.run(LLM_CONFIG, agdata(text="hello"), agdata(messages=[]), sandbox=None)
    assert getattr(result, "error", None) is None


def test_input_schema_description_value_only_checks_presence():
    """Non-type-name values (descriptions) only trigger a missing-key error."""
    s = agskill(
        name="s", system_prompt="",
        input_schema=agdata(query="the search query"),
    )
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct("{}")
        result, _, _ = s.run(LLM_CONFIG, agdata(query=42), agdata(messages=[]), sandbox=None)
    assert getattr(result, "error", None) is None  # 42 is not type-checked


def test_output_schema_missing_field_triggers_retry():
    """LLM gives bad output first, correct output on retry."""
    s = agskill(
        name="s", system_prompt="",
        output_schema=agdata(summary=str),
        max_retries=2,
    )
    responses = [
        _direct('{"wrong_key": "oops"}'),   # fails validation → retry injected
        _direct('{"summary": "good"}'),     # passes
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _ = s.run(LLM_CONFIG, agdata(text="hi"), agdata(messages=[]), sandbox=None)
    assert result.summary == "good"


def test_output_schema_retry_exhausted_returns_error():
    s = agskill(
        name="s", system_prompt="",
        output_schema=agdata(answer=str),
        max_retries=2,
    )
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"wrong": 1}')
        result, _, _ = s.run(LLM_CONFIG, agdata(q="hi"), agdata(messages=[]), sandbox=None, max_steps=10)
    assert result.error is not None
    assert "output schema error" in result.error


def test_output_schema_type_mismatch_triggers_retry():
    s = agskill(
        name="s", system_prompt="",
        output_schema=agdata(count=int),
        max_retries=1,
    )
    responses = [
        _direct('{"count": "should-be-int"}'),   # type mismatch
        _direct('{"count": 5}'),                  # correct
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    assert result.count == 5


def test_correction_message_appended_on_retry():
    """The retry message is appended to the conversation before the next LLM call."""
    s = agskill(
        name="s", system_prompt="",
        output_schema=agdata(answer=str),
        max_retries=1,
    )
    call_messages: list[list[dict]] = []
    def capture(**kwargs):
        call_messages.append(list(kwargs.get("messages", [])))
        return _direct('{"answer": "fixed"}')

    # First call will get the bad output injected; second call should see the correction.
    # We simulate: first response bad (no 'answer'), second response good.
    first_call = True
    def side_effect(**kwargs):
        nonlocal first_call
        call_messages.append(list(kwargs.get("messages", [])))
        if first_call:
            first_call = False
            return _direct('{"wrong": 1}')
        return _direct('{"answer": "fixed"}')

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = side_effect
        result, _, _ = s.run(LLM_CONFIG, agdata(q="hi"), agdata(messages=[]), sandbox=None)

    assert result.answer == "fixed"
    # Second call should have a correction user message near the end
    assert len(call_messages) == 2
    last_msgs = call_messages[1]
    assert any("Output schema errors" in m.get("content", "") for m in last_msgs if m["role"] == "user")


def test_schemas_appended_to_system_prompt():
    s = agskill(
        name="s",
        system_prompt="Be helpful.",
        input_schema=agdata(text=str),
        output_schema=agdata(summary=str),
    )
    prompt = s._build_system_prompt()
    assert "Be helpful." in prompt
    assert "Input JSON format" in prompt
    assert '"text"' in prompt
    assert "Output JSON format" in prompt
    assert '"summary"' in prompt


def test_no_schemas_system_prompt_unchanged():
    s = agskill(name="s", system_prompt="Be helpful.")
    assert s._build_system_prompt() == "Be helpful."
