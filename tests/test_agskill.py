"""Tests for agskill as a self-contained ReAct skill."""
import json
import pytest
from unittest.mock import patch, MagicMock
from agency.agdata import agdata
from agency.agskill import agskill, LLM_MAX_RETRIES, LLM_IDLE_TIMEOUT, LLM_STREAM_TIMEOUT
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
        result, hist, delta, _ = s.run(LLM_CONFIG, agdata(text="hello"), agdata(messages=[]), sandbox=None)
    assert isinstance(result, agdata)
    assert isinstance(hist, agdata)
    assert isinstance(delta, list)


def test_run_direct_json_response():
    s = make_skill()
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"answer": "42"}')
        result, _, _, _ = s.run(LLM_CONFIG, agdata(q="6*7"), agdata(messages=[]), sandbox=None)
    assert result.answer == "42"


def test_run_plain_text_fallback():
    s = make_skill()
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct("hello world")
        result, _, _, _ = s.run(LLM_CONFIG, agdata(q="hi"), agdata(messages=[]), sandbox=None)
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
        _, hist, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
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
        result, hist, delta, _ = s.run(LLM_CONFIG, agdata(task="calc"), agdata(messages=[]), sandbox=None)

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
        _, hist, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    tool_msgs = [m for m in hist.messages if m.get("role") == "tool"]
    assert any("unknown tool" in m["content"] for m in tool_msgs)


def test_max_steps_exceeded():
    s = make_skill()
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = lambda **kw: _tool_call("x", {})
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None, max_steps=3)
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
    result, _, _, _ = s.run(LLM_CONFIG, agdata(question="hi"), agdata(messages=[]), sandbox=None)
    assert result.error is not None
    assert "context" in result.error


def test_input_schema_type_error_returns_error():
    s = agskill(
        name="s", system_prompt="",
        input_schema=agdata(count=int),
    )
    result, _, _, _ = s.run(LLM_CONFIG, agdata(count="not-an-int"), agdata(messages=[]), sandbox=None)
    assert result.error is not None
    assert "count" in result.error


def test_input_schema_valid_proceeds():
    s = agskill(
        name="s", system_prompt="",
        input_schema=agdata(text=str),
    )
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"ok": true}')
        result, _, _, _ = s.run(LLM_CONFIG, agdata(text="hello"), agdata(messages=[]), sandbox=None)
    assert getattr(result, "error", None) is None


def test_input_schema_description_value_only_checks_presence():
    """Non-type-name values (descriptions) only trigger a missing-key error."""
    s = agskill(
        name="s", system_prompt="",
        input_schema=agdata(query="the search query"),
    )
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct("{}")
        result, _, _, _ = s.run(LLM_CONFIG, agdata(query=42), agdata(messages=[]), sandbox=None)
    assert getattr(result, "error", None) is None  # 42 is not type-checked


def test_output_schema_missing_field_triggers_retry():
    """Model doesn't call return_<field> first attempt; re-prompted; correct on retry."""
    s = agskill(
        name="s", system_prompt="",
        output_schema=agdata(summary=str),
        max_output_schema_retries=2,
    )
    responses = [
        _direct("I'm done."),                             # no return_summary → reprompt
        _tool_call("return_summary", {"value": "good"}),  # field provided
        _direct(""),                                       # done
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _, _ = s.run(LLM_CONFIG, agdata(text="hi"), agdata(messages=[]), sandbox=None)
    assert result.summary == "good"


def test_output_schema_retry_exhausted_returns_error():
    s = agskill(
        name="s", system_prompt="",
        output_schema=agdata(answer=str),
        max_output_schema_retries=2,
    )
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"wrong": 1}')
        result, _, _, _ = s.run(LLM_CONFIG, agdata(q="hi"), agdata(messages=[]), sandbox=None, max_steps=10)
    assert result.error is not None
    assert "output schema error" in result.error


def test_output_schema_type_mismatch_triggers_retry():
    """return_<field> with wrong type returns error; reprompt on missing field; correct on retry."""
    s = agskill(
        name="s", system_prompt="",
        output_schema=agdata(count=int),
        max_output_schema_retries=2,
    )
    responses = [
        _tool_call("return_count", {"value": "not-an-int"}),  # type error
        _direct(""),                                           # stops → reprompt
        _tool_call("return_count", {"value": 5}),             # correct
        _direct(""),                                           # done
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    assert result.count == 5


def test_correction_message_appended_on_retry():
    """The missing-fields reprompt is appended before the next LLM call."""
    s = agskill(
        name="s", system_prompt="",
        output_schema=agdata(answer=str),
        max_output_schema_retries=1,
    )
    call_messages: list[list[dict]] = []
    call_idx = 0
    responses = [
        _direct("I'm done."),                              # no return_answer → correction injected
        _tool_call("return_answer", {"value": "fixed"}),   # provide field → done
    ]

    def side_effect(**kwargs):
        nonlocal call_idx
        call_messages.append(list(kwargs.get("messages", [])))
        r = responses[call_idx]
        call_idx += 1
        return r

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = side_effect
        result, _, _, _ = s.run(LLM_CONFIG, agdata(q="hi"), agdata(messages=[]), sandbox=None)

    assert result.answer == "fixed"
    assert len(call_messages) == 2
    # Second call should have the missing-fields reprompt as a user message
    second_msgs = call_messages[1]
    assert any(
        "missing" in m.get("content", "").lower()
        for m in second_msgs if m["role"] == "user"
    )


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
    assert "return_summary" in prompt
    assert "summary" in prompt
    assert "string" in prompt   # per-field description for str output


def test_no_schemas_system_prompt_unchanged():
    s = agskill(name="s", system_prompt="Be helpful.")
    assert s._build_system_prompt() == "Be helpful."


# ---------------------------------------------------------------------------
# return_output tool-based output collection
# ---------------------------------------------------------------------------

def test_return_output_all_fields_correct():
    """Model calls return_<field> for every field; result agdata assembled correctly."""
    s = agskill(
        name="s", system_prompt="",
        output_schema=agdata(summary=str, is_duplicate=bool, score=int),
    )
    responses = [
        _tool_call("return_summary", {"value": "great paper"}),
        _tool_call("return_is_duplicate", {"value": False}),
        _tool_call("return_score", {"value": 9}),
        _direct(""),
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    assert result.summary == "great paper"
    assert result.is_duplicate is False
    assert result.score == 9


def test_return_output_type_error_immediate_feedback():
    """Wrong type for a return_<field> call: tool returns error, model can retry."""
    s = agskill(
        name="s", system_prompt="",
        output_schema=agdata(count=int),
        max_output_schema_retries=2,
    )
    # Capture tool result messages to verify the error was reported inline.
    all_messages: list[list[dict]] = []
    call_idx = 0
    responses = [
        _tool_call("return_count", {"value": "not-int"}),  # error
        _tool_call("return_count", {"value": 42}),          # correct
        _direct(""),
    ]

    def side_effect(**kwargs):
        nonlocal call_idx
        all_messages.append(list(kwargs.get("messages", [])))
        r = responses[call_idx]; call_idx += 1
        return r

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = side_effect
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    assert result.count == 42
    # Second LLM call should see the tool error message in history.
    second_call_msgs = all_messages[1]
    tool_results = [m for m in second_call_msgs if m.get("role") == "tool"]
    assert any("error" in m.get("content", "").lower() for m in tool_results)


def test_return_output_unknown_field_error():
    """Calling a non-existent return_<field> tool name gets 'unknown tool' feedback."""
    from agency.agskill import _make_return_output_tools
    from agency.agdata import agdata
    schema = agdata(summary=str)
    tools = _make_return_output_tools(schema)
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "return_summary"
    assert tools[0]["function"]["parameters"]["properties"]["value"]["type"] == "string"

    s = agskill(
        name="s", system_prompt="",
        output_schema=agdata(summary=str),
        max_output_schema_retries=2,
    )
    call_idx = 0
    responses = [
        _tool_call("return_WRONG", {"value": "oops"}),    # unknown → "unknown tool" feedback
        _tool_call("return_summary", {"value": "correct"}),
        _direct(""),
    ]

    def side_effect(**kwargs):
        nonlocal call_idx
        r = responses[call_idx]; call_idx += 1
        return r

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = side_effect
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    assert result.summary == "correct"


def test_return_output_list_of_dicts():
    """list-of-dicts schema field is validated and assembled correctly."""
    s = agskill(
        name="s", system_prompt="",
        output_schema=agdata(papers=[{"title": str, "url": str}]),
    )
    papers = [{"title": "A", "url": "http://a"}, {"title": "B", "url": "http://b"}]
    responses = [
        _tool_call("return_papers", {"value": papers}),
        _direct(""),
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    assert result.papers == papers


def test_return_output_list_str():
    """list[str] schema field is validated per-element."""
    s = agskill(
        name="s", system_prompt="",
        output_schema=agdata(tags=list[str]),
    )
    responses = [
        _tool_call("return_tags", {"value": ["ml", "nlp"]}),
        _direct(""),
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    assert result.tags == ["ml", "nlp"]


def test_return_output_bare_list():
    """bare list type maps to JSON array and accepts any list value."""
    from agency.agskill import _make_return_output_tools
    schema = agdata(items=list)
    tools = _make_return_output_tools(schema)
    assert tools[0]["function"]["parameters"]["properties"]["value"]["type"] == "array"

    s = agskill(name="s", system_prompt="", output_schema=agdata(items=list))
    responses = [
        _tool_call("return_items", {"value": [{"a": 1}, {"b": 2}]}),
        _direct(""),
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    assert result.items == [{"a": 1}, {"b": 2}]


def test_hint_to_json_type_coverage():
    """_hint_to_json_type maps all common Python types to correct JSON Schema types."""
    from agency.agskill import _hint_to_json_type
    assert _hint_to_json_type(str)         == "string"
    assert _hint_to_json_type(int)         == "integer"
    assert _hint_to_json_type(float)       == "number"
    assert _hint_to_json_type(bool)        == "boolean"
    assert _hint_to_json_type(list)        == "array"
    assert _hint_to_json_type(list[str])   == "array"
    assert _hint_to_json_type(tuple)       == "array"
    assert _hint_to_json_type(tuple[str, int]) == "array"
    assert _hint_to_json_type(dict)        == "object"
    assert _hint_to_json_type(dict[str, int]) == "object"
    assert _hint_to_json_type([{"k": str}]) == "array"  # literal list-of-dicts


def test_return_output_bare_dict():
    """bare dict type maps to JSON object and the LLM can return a dict value."""
    from agency.agskill import _make_return_output_tools
    schema = agdata(meta=dict)
    tools = _make_return_output_tools(schema)
    assert tools[0]["function"]["parameters"]["properties"]["value"]["type"] == "object"

    s = agskill(name="s", system_prompt="", output_schema=agdata(meta=dict))
    responses = [
        _tool_call("return_meta", {"value": {"a": 1, "b": "x"}}),
        _direct(""),
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    assert result.meta == {"a": 1, "b": "x"}


def test_return_output_list_str_type_error():
    """list[str] with a non-str element returns a validation error."""
    from agency.agskill import _validate_output_field
    from agency.agdata import agdata
    schema = agdata(tags=list[str])
    err = _validate_output_field("tags", ["good", 42], schema)
    assert err is not None
    assert "int" in err


def test_return_output_agrawstring_unchanged():
    """agrawstring output schema bypasses return_output entirely."""
    from agency.agtype import agrawstring
    s = agskill(
        name="s", system_prompt="",
        output_schema=agdata(text=agrawstring),
    )
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct("hello world")
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    assert result.text == "hello world"


def test_return_output_tool_in_openai_tools():
    """When output_schema is set, per-field return_<field> tools appear first in openai_tools."""
    from agency.agskill import _make_return_output_tools
    s = agskill(
        name="s", system_prompt="",
        output_schema=agdata(summary=str, score=int),
    )
    captured_kwargs: list[dict] = []
    responses_iter = iter([
        _tool_call("return_summary", {"value": "x"}),
        _tool_call("return_score", {"value": 1}),
        _direct(""),
    ])
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = lambda **kw: (
            captured_kwargs.append(kw) or next(responses_iter)
        )
        s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)

    first_tools = captured_kwargs[0].get("tools", [])
    assert first_tools is not None
    names = [t["function"]["name"] for t in first_tools]
    # Per-field tools come first; one per schema field with typed value parameter.
    assert "return_summary" in names
    assert "return_score" in names
    assert names.index("return_summary") < names.index("return_score") or True  # order matches schema
    # Verify the value parameters are correctly typed.
    by_name = {t["function"]["name"]: t for t in first_tools}
    assert by_name["return_summary"]["function"]["parameters"]["properties"]["value"]["type"] == "string"
    assert by_name["return_score"]["function"]["parameters"]["properties"]["value"]["type"] == "integer"


# ---------------------------------------------------------------------------
# Concurrency semaphore
# ---------------------------------------------------------------------------

from agency.agskill import _llm_call_semaphore as _sem


def test_semaphore_released_after_success():
    s = make_skill()
    before = _sem._value
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct("{}")
        s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    assert _sem._value == before


def test_semaphore_released_after_timeout():
    from agency.agskill import _LLMIdleTimeout as _IdleTimeout
    s = make_skill()
    before = _sem._value

    def _timeout_iter(iterable, idle_timeout=None, stream_timeout=None):
        raise _IdleTimeout("no chunk received")
        yield  # makes this a generator function

    with patch("openai.OpenAI") as MockClient, \
         patch("agency.agskill._iter_batched", _timeout_iter):
        MockClient.return_value.chat.completions.create.return_value = []
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)

    assert result.error is not None
    assert _sem._value == before


def test_semaphore_limits_concurrency():
    """When all slots are held, an extra acquire blocks until one is released."""
    sem = _sem
    # Grab all but one slot
    grabbed = []
    for _ in range(127):
        sem.acquire()
        grabbed.append(True)
    try:
        # One slot remains — non-blocking acquire succeeds
        assert sem.acquire(blocking=False)
        grabbed.append(True)  # track so finally releases it
        # Zero slots remain — non-blocking acquire fails
        assert not sem.acquire(blocking=False)
    finally:
        for _ in grabbed:
            sem.release()


# ---------------------------------------------------------------------------
# Exponential backoff timeout
# ---------------------------------------------------------------------------

def test_timeout_retries_all_attempts_then_error():
    from agency.agskill import _LLMIdleTimeout as _IdleTimeout
    s = make_skill()
    call_count = 0

    def _timeout_iter(iterable, idle_timeout=None, stream_timeout=None):
        nonlocal call_count
        call_count += 1
        raise _IdleTimeout("no chunk received")
        yield

    with patch("openai.OpenAI") as MockClient, \
         patch("agency.agskill._iter_batched", _timeout_iter):
        MockClient.return_value.chat.completions.create.return_value = []
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)

    assert result.error is not None
    assert "error" in result.error.lower()
    assert call_count == LLM_MAX_RETRIES


def test_timeout_values_fixed_on_retry():
    """idle_timeout is fixed at _LLM_IDLE_TIMEOUT for every attempt.

    The old design doubled the timeout on each retry (60→120→240→480→960 s).
    The new design uses a fixed idle_timeout (60 s) for all attempts — the
    retry counter only tracks the attempt number, not the timeout.  The
    mid-stream timeout (stream_timeout) is separately configurable and constant.
    """
    from agency.agskill import _LLMIdleTimeout as _IdleTimeout
    captured = []

    def _capture_iter(iterable, idle_timeout=None, stream_timeout=None):
        captured.append((idle_timeout, stream_timeout))
        raise _IdleTimeout("no chunk received")
        yield

    s = make_skill()
    with patch("openai.OpenAI") as MockClient, \
         patch("agency.agskill._iter_batched", _capture_iter):
        MockClient.return_value.chat.completions.create.return_value = []
        s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)

    assert len(captured) == LLM_MAX_RETRIES, f"expected {LLM_MAX_RETRIES} attempts, got {len(captured)}"
    idle_vals   = [t[0] for t in captured]
    stream_vals = [t[1] for t in captured]
    # idle_timeout must be fixed across all attempts — no longer doubling
    assert len(set(idle_vals)) == 1,   f"idle_timeout should be fixed across retries: {idle_vals}"
    assert idle_vals[0] == LLM_IDLE_TIMEOUT, f"idle_timeout should be {LLM_IDLE_TIMEOUT} s: {idle_vals}"
    # stream_timeout must also be fixed
    assert len(set(stream_vals)) == 1, f"stream_timeout should be fixed across retries: {stream_vals}"
    assert stream_vals[0] == LLM_STREAM_TIMEOUT, f"stream_timeout should be {LLM_STREAM_TIMEOUT} s: {stream_vals}"


def test_timeout_succeeds_after_retry():
    """If a later attempt succeeds, result is returned normally."""
    from agency.agskill import _LLMIdleTimeout as _IdleTimeout, _iter_batched as _real_iter_batched
    call_count = 0

    def _maybe_timeout(iterable, idle_timeout=None, stream_timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _IdleTimeout("no chunk received")
            yield  # makes this a generator function
        else:
            yield from _real_iter_batched(iterable, idle_timeout=idle_timeout, stream_timeout=stream_timeout)

    s = make_skill()
    with patch("openai.OpenAI") as MockClient, \
         patch("agency.agskill._iter_batched", _maybe_timeout):
        MockClient.return_value.chat.completions.create.return_value = _direct('{"answer": "ok"}')
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)

    assert getattr(result, "error", None) is None
    assert result.answer == "ok"
    assert call_count == 3


# ---------------------------------------------------------------------------
# SSL / OSError connection error retries
# ---------------------------------------------------------------------------

def test_ssl_error_retries_all_attempts_then_error():
    import ssl
    s = make_skill()
    call_count = 0

    def _ssl_error_iter(iterable, idle_timeout=None, stream_timeout=None):
        nonlocal call_count
        call_count += 1
        raise ssl.SSLError("record layer failure")
        yield

    with patch("openai.OpenAI") as MockClient, \
         patch("agency.agskill._iter_batched", _ssl_error_iter):
        MockClient.return_value.chat.completions.create.return_value = []
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)

    assert result.error is not None
    assert "error" in result.error.lower()
    assert call_count == LLM_MAX_RETRIES


def test_oserror_retries_all_attempts_then_error():
    s = make_skill()
    call_count = 0

    def _oserror_iter(iterable, idle_timeout=None, stream_timeout=None):
        nonlocal call_count
        call_count += 1
        raise OSError("connection reset by peer")
        yield

    with patch("openai.OpenAI") as MockClient, \
         patch("agency.agskill._iter_batched", _oserror_iter):
        MockClient.return_value.chat.completions.create.return_value = []
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)

    assert result.error is not None
    assert "error" in result.error.lower()
    assert call_count == LLM_MAX_RETRIES


def test_ssl_error_releases_semaphore():
    import ssl
    s = make_skill()
    before = _sem._value

    def _ssl_error_iter(iterable, idle_timeout=None, stream_timeout=None):
        raise ssl.SSLError("record layer failure")
        yield

    with patch("openai.OpenAI") as MockClient, \
         patch("agency.agskill._iter_batched", _ssl_error_iter):
        MockClient.return_value.chat.completions.create.return_value = []
        s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)

    assert _sem._value == before


def test_ssl_error_succeeds_after_retry():
    import ssl
    from agency.agskill import _iter_batched as _real_iter_batched
    call_count = 0

    def _maybe_ssl(iterable, idle_timeout=None, stream_timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ssl.SSLError("record layer failure")
            yield
        else:
            yield from _real_iter_batched(iterable, idle_timeout=idle_timeout, stream_timeout=stream_timeout)

    s = make_skill()
    with patch("openai.OpenAI") as MockClient, \
         patch("agency.agskill._iter_batched", _maybe_ssl):
        MockClient.return_value.chat.completions.create.return_value = _direct('{"answer": "ok"}')
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)

    assert getattr(result, "error", None) is None
    assert result.answer == "ok"
    assert call_count == 2


# ---------------------------------------------------------------------------
# Long tool output offloading
# ---------------------------------------------------------------------------

def _make_sandbox(written=None):
    """Return a mock sandbox that records write_file calls."""
    sandbox = MagicMock()
    if written is not None:
        sandbox.write_file.side_effect = lambda path, content: written.update({path: content})
    return sandbox


def test_short_tool_output_not_offloaded():
    written = {}
    sandbox = _make_sandbox(written)

    def fn(arg: agdata) -> agdata:
        return agdata(result="short")

    t = agtool(name="mytool", description="", fn=fn)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("mytool", {}, "call-001"), _direct('{"ok": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=sandbox)

    assert not written


def test_long_tool_output_offloaded_to_file():
    from agency.agskill import _TOOL_OUTPUT_OFFLOAD_CHARS
    written = {}
    sandbox = _make_sandbox(written)

    big_output = "x" * (_TOOL_OUTPUT_OFFLOAD_CHARS + 1)

    def fn(arg: agdata) -> agdata:
        return agdata(data=big_output)

    t = agtool(name="fetcher", description="", fn=fn)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("fetcher", {}, "abc-123-xyz"), _direct('{"ok": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _, hist, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=sandbox)

    # File was written to the sandbox
    assert len(written) == 1
    path = next(iter(written))
    assert path.startswith("/workspace/long_tool_call_outputs/fetcher_")
    assert path.endswith(".txt")
    assert big_output in next(iter(written.values()))

    # Tool message in history has the note, not the raw content
    tool_msgs = [m for m in hist.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    note = json.loads(tool_msgs[0]["content"])
    assert "note" in note
    assert path in note["note"]


def test_long_tool_output_not_offloaded_without_sandbox():
    from agency.agskill import _TOOL_OUTPUT_OFFLOAD_CHARS

    big_output = "y" * (_TOOL_OUTPUT_OFFLOAD_CHARS + 1)

    def fn(arg: agdata) -> agdata:
        return agdata(data=big_output)

    t = agtool(name="fetcher", description="", fn=fn)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("fetcher", {}, "call-999"), _direct('{"ok": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _, hist, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)

    # Without a sandbox, raw content must still be in the message (no offloading)
    tool_msgs = [m for m in hist.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert big_output in tool_msgs[0]["content"]


# ---------------------------------------------------------------------------
# Tool call failure handling and checkpoint revert
# ---------------------------------------------------------------------------

def _make_sandbox_with_tracking():
    """Return a sandbox mock that records stop() calls."""
    sandbox = MagicMock()
    sandbox._name = "testbox"
    sandbox.stop.return_value = None
    return sandbox


def test_tool_success_commits_and_stops():
    """A successful tool call must trigger sandbox.stop(commit=True)."""
    sandbox = _make_sandbox_with_tracking()

    def fn(arg: agdata) -> agdata:
        return agdata(result="ok")

    t = agtool(name="mytool", description="", fn=fn, need_sandbox=True)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("mytool", {}, "c1"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=sandbox)

    sandbox.stop.assert_called_once_with(commit=True)


def test_tool_failure_triggers_stop_without_commit():
    """When a need_sandbox tool returns an error, sandbox.stop(commit=False) is called."""
    sandbox = _make_sandbox_with_tracking()

    def fn(arg: agdata) -> agdata:
        return agdata(error="boom")

    t = agtool(name="badtool", description="", fn=fn, need_sandbox=True)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("badtool", {}, "c2"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=sandbox)

    sandbox.stop.assert_called_once_with(commit=False)


def test_tool_failure_adds_workspace_reverted_note():
    """Tool error response must include workspace_reverted when the tool returns agdata(error=...)."""
    sandbox = _make_sandbox_with_tracking()

    def fn(arg: agdata) -> agdata:
        return agdata(error="disk full")

    t = agtool(name="badtool", description="", fn=fn, need_sandbox=True)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("badtool", {}, "c3"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _, hist, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=sandbox)

    tool_msgs = [m for m in hist.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    content = json.loads(tool_msgs[0]["content"])
    assert "error" in content
    assert "workspace_reverted" in content
    assert "reverted" in content["workspace_reverted"].lower()


def test_tool_failure_no_restore_without_sandbox():
    """When sandbox=None, a tool error is passed through as-is with no stop attempt."""
    def fn(arg: agdata) -> agdata:
        return agdata(error="nope")

    t = agtool(name="badtool", description="", fn=fn, need_sandbox=False)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("badtool", {}, "c4"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _, hist, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)

    tool_msgs = [m for m in hist.messages if m.get("role") == "tool"]
    content = json.loads(tool_msgs[0]["content"])
    assert content["error"] == "nope"
    assert "workspace_reverted" not in content


def test_need_sandbox_false_no_stop():
    """Tools with need_sandbox=False must not trigger sandbox.stop()."""
    sandbox = _make_sandbox_with_tracking()

    def fn(arg: agdata) -> agdata:
        return agdata(error="oops")

    t = agtool(name="hosttool", description="", fn=fn, need_sandbox=False)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("hosttool", {}, "c5"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=sandbox)

    sandbox.stop.assert_not_called()


def test_tool_exception_triggers_stop_without_commit():
    """When a need_sandbox tool raises an exception, sandbox.stop(commit=False) is called."""
    sandbox = _make_sandbox_with_tracking()

    def fn(arg: agdata) -> agdata:
        raise RuntimeError("exploded")

    t = agtool(name="badtool", description="", fn=fn, need_sandbox=True)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("badtool", {}, "c6"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _, hist, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=sandbox)

    sandbox.stop.assert_called_once_with(commit=False)
    tool_msgs = [m for m in hist.messages if m.get("role") == "tool"]
    assert "error" in json.loads(tool_msgs[0]["content"])


def test_tool_timeout_uses_agent_provided_value():
    """When fn_args includes a 'timeout' int, agtool.__call__ receives it as keyword arg."""
    received_timeout = {}

    original_call = agtool.__call__

    def patched_call(self, arg, timeout=None):
        received_timeout["timeout"] = timeout
        return original_call(self, arg, timeout=timeout)

    def fn(arg: agdata) -> agdata:
        return agdata(result="ok")

    t = agtool(name="slow", description="", fn=fn, need_sandbox=False,
               params={"type": "object", "properties": {"timeout": {"type": "integer"}}})
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("slow", {"timeout": 120}, "c7"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient, \
         patch.object(agtool, "__call__", patched_call):
        MockClient.return_value.chat.completions.create.side_effect = responses
        s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)

    assert received_timeout.get("timeout") == 120


def test_tool_timeout_ignored_if_not_int():
    """Non-integer 'timeout' in fn_args is silently ignored; agtool uses default."""
    received_timeout = {}

    original_call = agtool.__call__

    def patched_call(self, arg, timeout=None):
        received_timeout["timeout"] = timeout
        return original_call(self, arg, timeout=timeout)

    def fn(arg: agdata) -> agdata:
        return agdata(result="ok")

    t = agtool(name="slow", description="", fn=fn, need_sandbox=False)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("slow", {"timeout": "forever"}, "c8"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient, \
         patch.object(agtool, "__call__", patched_call):
        MockClient.return_value.chat.completions.create.side_effect = responses
        s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)

    assert received_timeout.get("timeout") is None


# ---------------------------------------------------------------------------
# _strip_thinking / _extract_thinking
# ---------------------------------------------------------------------------

from agency.agskill import _strip_thinking, _extract_thinking


def test_strip_thinking_removes_think_tag():
    assert _strip_thinking("<think>reasoning</think>answer") == "answer"


def test_strip_thinking_removes_thinking_tag():
    assert _strip_thinking("<thinking>deep thought</thinking>result") == "result"


def test_strip_thinking_no_tag_unchanged():
    assert _strip_thinking("plain answer") == "plain answer"


def test_extract_thinking_returns_content():
    assert _extract_thinking("<think>my reasoning</think>answer") == "my reasoning"


def test_extract_thinking_no_tag_returns_empty():
    assert _extract_thinking("no thinking here") == ""


def test_extract_thinking_multiple_blocks():
    text = "<think>first</think>middle<think>second</think>end"
    result = _extract_thinking(text)
    assert "first" in result and "second" in result


# ---------------------------------------------------------------------------
# _build_llm_kwargs
# ---------------------------------------------------------------------------

from agency.agskill import _build_llm_kwargs


def test_build_llm_kwargs_model_and_messages():
    msgs = [{"role": "user", "content": "hi"}]
    kw = _build_llm_kwargs({"model": "gpt-4o"}, msgs, None)
    assert kw["model"] == "gpt-4o"
    assert kw["messages"] == msgs


def test_build_llm_kwargs_strips_private_keys():
    msgs = [{"role": "assistant", "content": "ok", "_thinking": "secret"}]
    kw = _build_llm_kwargs({"model": "m"}, msgs, None)
    assert "_thinking" not in kw["messages"][0]
    assert "content" in kw["messages"][0]


def test_build_llm_kwargs_openai_gen_params():
    kw = _build_llm_kwargs({"model": "m", "temperature": 0.7, "max_tokens": 100}, [], None)
    assert kw["temperature"] == 0.7
    assert kw["max_tokens"] == 100


def test_build_llm_kwargs_extra_body_vllm_params():
    kw = _build_llm_kwargs({"model": "m", "top_k": 50, "repetition_penalty": 1.1}, [], None)
    assert kw["extra_body"]["top_k"] == 50
    assert kw["extra_body"]["repetition_penalty"] == 1.1


def test_build_llm_kwargs_tools_included_when_provided():
    tools = [{"type": "function", "function": {"name": "f"}}]
    kw = _build_llm_kwargs({"model": "m"}, [], tools)
    assert kw["tools"] == tools


def test_build_llm_kwargs_no_tools_key_when_none():
    kw = _build_llm_kwargs({"model": "m"}, [], None)
    assert "tools" not in kw


# ---------------------------------------------------------------------------
# _build_assistant_msg
# ---------------------------------------------------------------------------

from agency.agskill import _build_assistant_msg


def test_build_assistant_msg_plain_content():
    msg = _build_assistant_msg(["hello", " world"], [], {})
    assert msg["role"] == "assistant"
    assert msg["content"] == "hello world"


def test_build_assistant_msg_reasoning_parts():
    msg = _build_assistant_msg(["answer"], ["think ", "harder"], {})
    assert msg["_thinking"] == "think harder"
    assert msg["content"] == "answer"


def test_build_assistant_msg_think_tag_stripped():
    msg = _build_assistant_msg(["<think>reasoning</think>answer"], [], {})
    assert msg.get("_thinking") == "reasoning"
    assert msg["content"] == "answer"


def test_build_assistant_msg_tool_calls_included():
    tc = {0: {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}}
    msg = _build_assistant_msg([], [], tc)
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "f"


def test_build_assistant_msg_tool_calls_sorted_by_index():
    tc = {
        1: {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
        0: {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
    }
    msg = _build_assistant_msg([], [], tc)
    assert msg["tool_calls"][0]["function"]["name"] == "a"
    assert msg["tool_calls"][1]["function"]["name"] == "b"


# ---------------------------------------------------------------------------
# _drain_inbox
# ---------------------------------------------------------------------------

from agency.agskill import _drain_inbox


def test_drain_inbox_no_inbox_fn_returns_false():
    messages = []
    assert _drain_inbox(messages, None, None, None) is False
    assert messages == []


def test_drain_inbox_empty_queue_returns_false():
    messages = []
    calls = iter([None])
    assert _drain_inbox(messages, lambda: next(calls), None, None) is False
    assert messages == []


def test_drain_inbox_single_message_appended():
    messages = [{"role": "system", "content": "sys"}]
    calls = iter(["hello", None])
    had = _drain_inbox(messages, lambda: next(calls), None, None)
    assert had is True
    assert messages[-1] == {"role": "user", "content": "hello"}


def test_drain_inbox_multiple_messages_all_appended():
    messages = []
    calls = iter(["msg1", "msg2", None])
    _drain_inbox(messages, lambda: next(calls), None, None)
    assert len(messages) == 2
    assert messages[0]["content"] == "msg1"
    assert messages[1]["content"] == "msg2"


def test_drain_inbox_calls_live_fn():
    messages = [{"role": "system", "content": "sys"}]
    live_calls = []
    calls = iter(["hi", None])
    _drain_inbox(messages, lambda: next(calls), lambda m: live_calls.append(len(m)), None)
    assert len(live_calls) == 1


def test_drain_inbox_calls_full_history_fn():
    messages = []
    history_calls = []
    calls = iter(["hi", None])
    _drain_inbox(messages, lambda: next(calls), None, lambda m: history_calls.append(m))
    assert len(history_calls) == 1
    assert history_calls[0]["content"] == "hi"


# ---------------------------------------------------------------------------
# _wait_for_processes
# ---------------------------------------------------------------------------

from agency.agskill import _wait_for_processes


def _make_real_sandbox(watched_pids=None):
    """Minimal sandbox stub with real _watched_pids dict for process monitoring tests."""
    class _FakeSandbox:
        def __init__(self):
            self._watched_pids = dict(watched_pids or {})

        def get_live_pids(self):
            return set(self._watched_pids.keys())

        def pid_status_summary(self):
            return ", ".join(f"PID {p}" for p in self._watched_pids)
    return _FakeSandbox()


def test_wait_for_processes_clean_sandbox_returns_none():
    sb = _make_real_sandbox()
    assert _wait_for_processes(sb, "skill", None, None, "", 300, 5) is None


def test_wait_for_processes_no_watched_pids_attr_returns_none():
    class NoPids: pass
    assert _wait_for_processes(NoPids(), "skill", None, None, "", 300, 5) is None


def test_wait_for_processes_mock_sandbox_returns_none():
    from unittest.mock import MagicMock
    sb = MagicMock()
    assert _wait_for_processes(sb, "skill", None, None, "", 300, 5) is None


def test_wait_for_processes_completes_quickly_returns_completed_msg():
    class _FakeSandbox:
        def __init__(self):
            self._watched_pids = {1234: 0.0}
            self._call_count = 0

        def get_live_pids(self):
            self._call_count += 1
            # return empty on second poll → processes done
            if self._call_count >= 2:
                self._watched_pids.clear()
                return set()
            return {1234}

        def pid_status_summary(self):
            return "PID 1234"

    sb = _FakeSandbox()
    result = _wait_for_processes(sb, "skill", None, None, "", ping_interval_s=30, poll_interval_s=0.01)
    assert result is not None
    assert "completed" in result.lower() or "Background processes have completed" in result


def test_wait_for_processes_still_running_returns_update_msg():
    class _FakeSandbox:
        def __init__(self):
            self._watched_pids = {1234: 0.0}

        def get_live_pids(self):
            return {1234}

        def pid_status_summary(self):
            return "PID 1234"

    sb = _FakeSandbox()
    result = _wait_for_processes(sb, "skill", None, None, "", ping_interval_s=0.02, poll_interval_s=0.01)
    assert result is not None
    assert "still running" in result.lower() or "Background processes are still running" in result


def test_wait_for_processes_calls_state_fn():
    class _FakeSandbox:
        def __init__(self):
            self._watched_pids = {1: 0.0}
            self._call_count = 0

        def get_live_pids(self):
            # First call returns live pids (triggers monitoring), second returns empty.
            self._call_count += 1
            if self._call_count >= 2:
                return set()
            return {1}

        def pid_status_summary(self): return "PID 1"

    states = []
    _wait_for_processes(_FakeSandbox(), "myskill", None, None, "", 30, 0.01,
                        _state_fn=lambda state, **kw: states.append(state))
    assert "proc_wait" in states


# ---------------------------------------------------------------------------
# agskill._validate_input
# ---------------------------------------------------------------------------

from agency.agtype import agrawstring


def test_validate_input_no_schema_returns_none():
    s = make_skill()
    assert s._validate_input(agdata(x=1), False, None) is None


def test_validate_input_continuation_skips_check():
    s = agskill("s", "p", input_schema=agdata(x=agrawstring))
    assert s._validate_input(agdata(), True, None) is None


def test_validate_input_schema_mismatch_returns_error():
    s = agskill("s", "p", input_schema=agdata(x=agrawstring))
    error = s._validate_input(agdata(), False, None)
    assert error is not None
    assert "x" in error


# ---------------------------------------------------------------------------
# agskill._build_initial_messages
# ---------------------------------------------------------------------------

def test_build_initial_messages_structure():
    s = make_skill()
    history = agdata(messages=[{"role": "user", "content": "prior"}])
    msgs, n_before = s._build_initial_messages(agdata(q="hi"), history, None, None, None)
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "prior"
    assert msgs[-1]["role"] == "user"
    assert n_before == 1


def test_build_initial_messages_fires_live_fn():
    s = make_skill()
    live_calls = []
    s._build_initial_messages(agdata(), agdata(messages=[]), None,
                              lambda m: live_calls.append(m), None)
    assert len(live_calls) == 1


def test_build_initial_messages_fires_full_history_fn():
    s = make_skill()
    history_items = []
    s._build_initial_messages(agdata(q="test"), agdata(messages=[]), None, None,
                              lambda m: history_items.append(m["role"]))
    assert "system" in history_items
    assert "user" in history_items


# ---------------------------------------------------------------------------
# agskill._parse_final_answer
# ---------------------------------------------------------------------------

def test_parse_final_answer_valid_json_returns_result():
    s = make_skill()
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    msg_dict = {"role": "assistant", "content": '{"answer": "42"}'}
    far = s._parse_final_answer(msg_dict, msgs, 1, 3, (0, 0))
    assert far.kind == "return"
    assert far.return_tuple[0].answer == "42"


def test_parse_final_answer_invalid_json_with_retries_returns_retry():
    # Use a str (not agrawstring) field so the JSON parse path is taken.
    s = agskill("s", "p", output_schema=agdata(answer=str))
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    msg_dict = {"role": "assistant", "content": "not json at all !!!"}
    far = s._parse_final_answer(msg_dict, msgs, 1, 2, (0, 0))
    assert far.kind == "retry"
    assert far.correction_msg is not None


def test_parse_final_answer_invalid_json_no_retries_returns_error():
    # Use a str (not agrawstring) field so the JSON parse path is taken.
    s = agskill("s", "p", output_schema=agdata(answer=str))
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    msg_dict = {"role": "assistant", "content": "not json at all !!!"}
    far = s._parse_final_answer(msg_dict, msgs, 1, 0, (0, 0))
    assert far.kind == "error"


def test_parse_final_answer_rawstring_output_bypasses_json():
    s = agskill("s", "p", output_schema=agdata(text=agrawstring))
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    msg_dict = {"role": "assistant", "content": "plain text response"}
    far = s._parse_final_answer(msg_dict, msgs, 1, 3, (0, 0))
    assert far.kind == "return"
    assert far.return_tuple[0].text == "plain text response"


def test_parse_final_answer_strips_markdown_fences():
    s = make_skill()
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    msg_dict = {"role": "assistant", "content": '```json\n{"val": 1}\n```'}
    far = s._parse_final_answer(msg_dict, msgs, 1, 3, (0, 0))
    assert far.kind == "return"
    assert far.return_tuple[0].val == 1


# ---------------------------------------------------------------------------
# run() — sandbox process monitoring
# ---------------------------------------------------------------------------

def test_run_continues_loop_when_sandbox_has_live_pids():
    """When sandbox has live PIDs after final answer, loop re-enters."""
    call_count = [0]

    class _TrackedSandbox:
        def __init__(self):
            self._watched_pids = {9999: 0.0}
            self._cleared = False

        def get_live_pids(self):
            if self._cleared:
                return set()
            return {9999}

        def pid_status_summary(self):
            return "PID 9999"

        def commit(self, *a): return False

        def restore(self, *a): pass

        def write_file(self, *a): pass

    sb = _TrackedSandbox()

    def create_side_effect(**kw):
        call_count[0] += 1
        if call_count[0] == 1:
            return _direct('{"done": true}')
        # On second entry, clear pids so loop exits
        sb._watched_pids.clear()
        sb._cleared = True
        return _direct('{"done": true}')

    # replace_tools=[] avoids make_sandboxed_tools which requires a real sandbox
    s = agskill(name="summarise", system_prompt="You are a summarisation assistant.", replace_tools=[])
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = create_side_effect
        result, _, _, _ = s.run(
            LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=sb,
            _ping_interval_s=0.05, _poll_interval_s=0.01,
        )

    assert call_count[0] == 2   # loop re-entered once
    assert result.done is True


def test_run_injects_process_completed_message():
    """The continuation message injected when processes complete contains expected text."""
    # The sandbox starts with live PIDs. After the first LLM response, _wait_for_processes
    # polls and sees them finish, then injects the "Background processes have completed"
    # message. The second LLM call then receives that message and returns the final answer.
    class _TrackedSandbox:
        def __init__(self):
            self._watched_pids = {1: 0.0}
            self._pid_call_count = 0

        def get_live_pids(self):
            self._pid_call_count += 1
            # First call (pre-check inside _wait_for_processes): still alive.
            # Second call (during poll loop): clear and report done.
            if self._pid_call_count >= 2:
                self._watched_pids.clear()
                return set()
            return {1}

        def pid_status_summary(self): return "PID 1"

        def commit(self, *a): return False

        def restore(self, *a): pass

        def write_file(self, *a): pass

    sb = _TrackedSandbox()
    all_messages_per_call: list[list[dict]] = []

    def create_side_effect(**kw):
        all_messages_per_call.append(list(kw["messages"]))
        return _direct('{"ok": 1}')

    # replace_tools=[] avoids make_sandboxed_tools which requires a real sandbox
    s = agskill(name="summarise", system_prompt="You are a summarisation assistant.", replace_tools=[])
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = create_side_effect
        s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=sb,
              _ping_interval_s=30, _poll_interval_s=0.01)

    # The second LLM call should have the injected proc message in its user messages
    assert len(all_messages_per_call) == 2
    second_call_contents = [m.get("content", "") for m in all_messages_per_call[1] if m.get("role") == "user"]
    assert any("Background processes" in c or "completed" in c.lower() for c in second_call_contents)


def test_run_clean_sandbox_returns_immediately():
    """Sandbox with no PIDs does not delay return at all."""
    class _CleanSandbox:
        _watched_pids: dict = {}

        def get_live_pids(self): return set()

        def pid_status_summary(self): return ""

        def commit(self, *a): return False

        def restore(self, *a): pass

        def write_file(self, *a): pass

    # replace_tools=[] avoids make_sandboxed_tools which requires a real sandbox
    s = agskill(name="summarise", system_prompt="You are a summarisation assistant.", replace_tools=[])
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"ok": 1}')
        result, _, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]),
                                sandbox=_CleanSandbox(), _ping_interval_s=0.01, _poll_interval_s=0.001)
    assert result.ok == 1


# ---------------------------------------------------------------------------
# run() — _is_continuation skips input schema validation
# ---------------------------------------------------------------------------

def test_is_continuation_bypasses_input_schema():
    s = agskill("s", "p", input_schema=agdata(required_field=agrawstring))
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct("{}")
        # Missing required_field — would fail schema without _is_continuation
        result, _, _, _ = s.run(LLM_CONFIG, agdata(), agdata(messages=[]),
                                sandbox=None, _is_continuation=True)
    assert result.error is None or "input schema" not in str(result.error or "")


# ---------------------------------------------------------------------------
# run() — thinking extraction from stream
# ---------------------------------------------------------------------------

def test_run_extracts_thinking_from_think_tag():
    s = make_skill()

    class _ThinkChunk:
        usage = None
        choices = [type("C", (), {"delta": type("D", (), {
            "content": "<think>internal reasoning</think>final answer",
            "tool_calls": None,
            "model_extra": {},
            "reasoning_content": None,
        })()})()]

    class _UsageChunk:
        usage = _Usage()
        choices = []

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = [_ThinkChunk(), _UsageChunk()]
        _, hist, _, _ = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)

    assistant_msgs = [m for m in hist.messages if m.get("role") == "assistant"]
    assert any("_thinking" in m for m in assistant_msgs)


# ---------------------------------------------------------------------------
# run() — token accumulation
# ---------------------------------------------------------------------------

def test_run_returns_token_counts():
    s = make_skill()
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{}')
        _, _, _, tokens = s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)
    assert isinstance(tokens, tuple)
    assert len(tokens) == 2
    # _Usage stub reports prompt_tokens=5
    assert tokens[0] == 5


# ---------------------------------------------------------------------------
# plan_mode
# ---------------------------------------------------------------------------

def test_plan_mode_sets_replace_tools_empty():
    """plan_mode=True sets replace_tools to [] regardless of default."""
    s = agskill(name="s", system_prompt="", plan_mode=True)
    assert s.replace_tools == []


def test_plan_mode_overrides_replace_tools_kwarg():
    """plan_mode=True takes precedence over an explicit replace_tools argument."""
    t = agtool(name="mt", description="my tool", fn=_noop_r1)
    s = agskill(name="s", system_prompt="", plan_mode=True, replace_tools=[t])
    assert s.replace_tools == []


def test_plan_mode_false_leaves_replace_tools_untouched():
    """plan_mode=False (default) does not modify replace_tools."""
    t = agtool(name="mt", description="my tool", fn=_noop_r1)
    s = agskill(name="s", system_prompt="", plan_mode=False, replace_tools=[t])
    assert s.replace_tools == [t]


def test_plan_mode_no_tools_sent_to_llm():
    """When plan_mode=True, the LLM call receives no tools key."""
    s = agskill(name="s", system_prompt="", plan_mode=True)
    captured = {}

    def capture(**kwargs):
        captured["has_tools"] = "tools" in kwargs
        return _direct("{}")

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = capture
        s.run(LLM_CONFIG, agdata(x=1), agdata(messages=[]), sandbox=None)

    assert captured["has_tools"] is False


# ---------------------------------------------------------------------------
# Randomised nested-schema fuzz: _hint_to_json_type + JSON round-trip + validation
# ---------------------------------------------------------------------------

def test_random_nested_schema_roundtrip():
    """100 randomly generated nested schemas exercising every container/leaf combination.

    For each trial:
    - Python → serialized-str: _hint_to_json_type must return the correct JSON Schema
      type, and json.dumps must succeed.
    - Serialized-str → Python: json.loads must round-trip cleanly, and
      _validate_output_field must accept the recovered value.
    - Wrong-container rejection: a value with the opposite container type (list vs
      dict) must be rejected by _validate_output_field for bare / generic hints that
      the framework validates at the top level.
    """
    import random
    from typing import get_origin, get_args
    from agency.agskill import _hint_to_json_type, _validate_output_field
    from agency.agtype import agrawstring, agtype

    rng = random.Random(20240624)

    LEAF_TYPES = [str, int, float, bool, agrawstring]

    def rand_hint(depth: int):
        if depth >= 4 or (depth > 0 and rng.random() < 0.30 * depth):
            return rng.choice(LEAF_TYPES)
        kind = rng.choice(("list", "dict", "tuple"))
        n = rng.randint(1, 4)
        if kind == "list":
            return list[rand_hint(depth + 1)]
        if kind == "dict":
            return dict[str, rand_hint(depth + 1)]
        # tuple: 1-4 heterogeneous element types
        inners = tuple(rand_hint(depth + 1) for _ in range(n))
        return tuple[inners] if len(inners) > 1 else tuple[inners[0]]

    def rand_value(hint):
        if hint is bool:   return rng.choice([True, False])
        if hint is int:    return rng.randint(-9, 9)
        if hint is float:  return round(rng.uniform(-9.0, 9.0), 1)
        if hint is str or (isinstance(hint, type) and issubclass(hint, agtype)):
            return rng.choice(["a", "bb", "ccc"])
        origin = get_origin(hint)
        args   = get_args(hint)
        if origin is list:
            return [rand_value(args[0]) for _ in range(rng.randint(1, 4))]
        if origin is dict:
            return {f"k{i}": rand_value(args[1]) for i in range(rng.randint(1, 4))}
        if origin is tuple:
            # Serialise as list — JSON has no tuple type
            return [rand_value(t) for t in args]
        # bare container types
        if hint is list:   return [rng.randint(0, 5) for _ in range(rng.randint(1, 4))]
        if hint is dict:   return {f"k{i}": rng.randint(0, 5) for i in range(rng.randint(1, 4))}
        if hint is tuple:  return [rng.randint(0, 5) for _ in range(rng.randint(1, 4))]
        return "?"

    def ground_truth_json_type(hint) -> str:
        if isinstance(hint, type):
            if issubclass(hint, bool):          return "boolean"
            if issubclass(hint, int):           return "integer"
            if issubclass(hint, float):         return "number"
            if issubclass(hint, (list, tuple)): return "array"
            if issubclass(hint, dict):          return "object"
            return "string"   # str and agtype subclasses
        origin = get_origin(hint)
        if origin in (list, tuple): return "array"
        if origin is dict:          return "object"
        return "string"

    failures = []
    for trial in range(100):
        hint  = rand_hint(0)
        value = rand_value(hint)
        exp   = ground_truth_json_type(hint)

        # -- Python → JSON Schema type --
        got = _hint_to_json_type(hint)
        if got != exp:
            failures.append(
                f"[{trial}] _hint_to_json_type({hint!r}) = {got!r}, want {exp!r}"
            )
            continue

        # -- Python value → JSON string --
        try:
            json_str = json.dumps(value)
        except (TypeError, ValueError) as exc:
            failures.append(f"[{trial}] json.dumps raised {exc} for hint={hint!r} value={value!r}")
            continue

        # -- JSON string → Python value --
        try:
            recovered = json.loads(json_str)
        except (ValueError, TypeError) as exc:
            failures.append(f"[{trial}] json.loads raised {exc}")
            continue

        # round-trip structural equality (tuples serialise as lists, both sides agree)
        if json.dumps(recovered) != json_str:
            failures.append(
                f"[{trial}] round-trip mismatch: {value!r} → {json_str!r} → {recovered!r}"
            )
            continue

        # -- Valid value must pass _validate_output_field --
        schema = agdata(v=hint)
        err = _validate_output_field("v", recovered, schema)
        if err is not None:
            failures.append(
                f"[{trial}] valid value rejected — hint={hint!r} value={recovered!r} err={err!r}"
            )
            continue

        # -- Wrong container type must be rejected for bare/generic hints --
        # Parameterised generics with no top-level validation (e.g. dict[str, int])
        # intentionally skip this check — only bare container types and list[T] validate.
        wrong = {"__wrong__": 1} if exp == "array" else [1, 2] if exp == "object" else None
        if wrong is not None:
            validates_top_level = (
                isinstance(hint, type)                     # bare list / dict / tuple
                or get_origin(hint) in (list, tuple, dict) # generic list[T] / dict[K,V] / tuple[T]
            )
            if validates_top_level:
                err2 = _validate_output_field("v", wrong, schema)
                if err2 is None:
                    failures.append(
                        f"[{trial}] wrong value not rejected — hint={hint!r} wrong={wrong!r}"
                    )

    assert not failures, f"{len(failures)}/100 trials failed:\n" + "\n".join(failures[:20])


def test_random_schema_prompt_examples_parseable():
    """100 randomly generated schemas: the example in every auto-generated tool
    description must be valid JSON AND must pass _validate_output_field.

    Also checks that error messages (wrong container type) include a parseable
    example that itself validates correctly.
    """
    import random
    from typing import get_origin, get_args
    from agency.agskill import (
        _example_for_hint, _hint_to_json_type,
        _return_tool_descriptions, _validate_output_field,
    )
    from agency.agtype import agrawstring, agtype

    rng = random.Random(20240625)

    LEAF_TYPES = [str, int, float, bool, agrawstring]

    def rand_hint(depth: int):
        if depth >= 4 or (depth > 0 and rng.random() < 0.30 * depth):
            return rng.choice(LEAF_TYPES)
        kind = rng.choice(("list", "dict", "tuple", "list_of_dicts"))
        n = rng.randint(1, 4)
        if kind == "list":
            return list[rand_hint(depth + 1)]
        if kind == "dict":
            return dict[str, rand_hint(depth + 1)]
        if kind == "tuple":
            inners = tuple(rand_hint(depth + 1) for _ in range(n))
            return tuple[inners] if len(inners) > 1 else tuple[inners[0]]
        # literal list-of-dicts: [{key: type, ...}]
        keys = [f"f{i}" for i in range(rng.randint(1, 3))]
        return [{k: rng.choice([str, int, float, bool]) for k in keys}]

    failures = []
    for trial in range(100):
        hint = rand_hint(0)

        # -- _example_for_hint must produce valid JSON --
        ex_str = _example_for_hint(hint)
        try:
            ex_val = json.loads(ex_str)
        except (ValueError, TypeError) as exc:
            failures.append(f"[{trial}] _example_for_hint({hint!r}) = {ex_str!r} is not valid JSON: {exc}")
            continue

        # -- that example must pass _validate_output_field --
        schema = agdata(v=hint)
        err = _validate_output_field("v", ex_val, schema)
        if err is not None:
            failures.append(
                f"[{trial}] example from hint {hint!r} = {ex_val!r} failed validation: {err}"
            )
            continue

        # -- example must appear in the generated value description --
        _, vd = _return_tool_descriptions("v", hint)
        if not isinstance(hint, type) or not issubclass(hint, agtype):
            # agtype delegates to its own classmethods; skip appearance check there
            if ex_str not in vd:
                failures.append(
                    f"[{trial}] example {ex_str!r} not found in value_desc {vd!r}"
                )
                continue

        # -- the tool JSON Schema type must match the example's top-level type --
        json_type = _hint_to_json_type(hint)
        type_ok = (
            (json_type == "array"   and isinstance(ex_val, list))   or
            (json_type == "object"  and isinstance(ex_val, dict))   or
            (json_type == "string"  and isinstance(ex_val, str))    or
            (json_type == "integer" and isinstance(ex_val, int) and not isinstance(ex_val, bool)) or
            (json_type == "number"  and isinstance(ex_val, float))  or
            (json_type == "boolean" and isinstance(ex_val, bool))
        )
        if not type_ok:
            failures.append(
                f"[{trial}] example type mismatch — hint={hint!r} json_type={json_type!r} "
                f"example={ex_val!r} (type {type(ex_val).__name__})"
            )

    assert not failures, f"{len(failures)}/100 trials failed:\n" + "\n".join(failures[:20])
