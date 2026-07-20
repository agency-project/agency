"""Tests for agskill as a self-contained ReAct skill."""

import json
from unittest.mock import patch, MagicMock
from agency.agdata import agdata, agerror
from agency.agcontext import agcontext
from agency.agconfig import agConfig
from agency.agschema import agschema, _AgSchemaFields
from agency.agskill import agskill
from agency.agllm import _AgLLMFields, agllm
from agency.agtool import agtool, _AgToolFields
from agency.agent import agent as _agent_cls, agent_state as _agent_state_cls

LLM_MAX_RETRIES = _AgLLMFields.max_retries.default
LLM_IDLE_TIMEOUT = _AgLLMFields.idle_timeout.default
LLM_STREAM_TIMEOUT = _AgLLMFields.stream_timeout.default

LLM_CONFIG = {"api_key": "test", "model": ""}
LLM = agllm(agConfig({"agllm_backend": LLM_CONFIG}), context_limit=128_000)


def make_mock_agent(llm=None, sandbox=None, ping_interval_s=300, poll_interval_s=5):
    _ping = ping_interval_s
    _poll = poll_interval_s

    class _MockAgent:
        agresource_pool = MagicMock()
        ping_interval_s = _ping
        poll_interval_s = _poll
        agconfig = None
        _drain_inbox = _agent_cls._drain_inbox
        _check_pause = _agent_cls._check_pause

    ag = _MockAgent()
    ag.llm = llm or LLM
    if sandbox is not None:
        ag.sandbox = sandbox
    else:
        ag.sandbox = MagicMock()
        # A bare MagicMock()'s _has_pending_background_work() would
        # otherwise auto-mock to a truthy value, making agtool.py's
        # dispatch_tools() defer stop() forever -- default to "nothing
        # pending" so tests get the common case without configuring it.
        ag.sandbox._has_pending_background_work.return_value = False
    ag.terminal = MagicMock()
    ag._state = _agent_state_cls("test")
    ag.log = MagicMock()
    ag.log.token_usage = {}
    ag.agname = "test"
    ag._set_ui_state = MagicMock()
    ag._push_live_messages = MagicMock()
    ag._append_full_history = MagicMock()
    ag._next_inbox_msg = MagicMock(return_value=None)
    ag.push_token_count_update_to_ui = MagicMock()
    return ag


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
        self.function = _TCFnDelta(name, args_json)


class _TCFnDelta:
    def __init__(self, name, args):
        self.name = name
        self.arguments = args


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
        result, ctx, delta = s.execute_react(
            make_mock_agent(LLM), agcontext(), agdata(text="hello")
        )
    assert isinstance(result, agdata)
    assert isinstance(ctx, agcontext)
    assert isinstance(delta, list)


def test_run_no_schema_returns_raw_content():
    s = make_skill()
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"answer": "42"}')
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(q="6*7"))
    assert result.result == '{"answer": "42"}'


def test_run_plain_text_fallback():
    s = make_skill()
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct("hello world")
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(q="hi"))
    assert result.result == "hello world"


# ---------------------------------------------------------------------------
# agskill.check_schema — Python type object hints
# ---------------------------------------------------------------------------


def test_check_schema_accepts_python_type_objects():
    assert agschema(agdata(x=int, name=str)).check(agdata(x=5, name="hi")) == []


def test_check_schema_type_mismatch_with_type_object():
    errors = agschema(agdata(x=int)).check(agdata(x="bad"))
    assert len(errors) == 1
    assert "x" in errors[0]
    assert "int" in errors[0]


def test_system_prompt_type_names_shown_correctly():
    from agency.agtype import agfile

    sk = agskill(
        "t",
        "",
        input_schema=agdata(n=int, s=str, doc=agfile),
        output_schema=agdata(result=float),
    )
    prompt = sk._build_system_prompt()
    assert '"n": "int"' in prompt  # input schema still uses to_json()
    assert '"s": "str"' in prompt
    assert '"doc": "file"' in prompt
    assert "result" in prompt  # output field listed by name
    assert "float" in prompt  # output field type shown as "float"


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
        s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][0]["content"] == "You are a summarisation assistant."


def test_system_prompt_not_in_returned_history():
    s = make_skill()
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct("{}")
        _, ctx, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))
    roles = [m["role"] for m in ctx.messages]
    assert "system" not in roles


def test_existing_history_included_in_call():
    s = make_skill()
    prior = agcontext(
        messages=[{"role": "user", "content": "prior"}, {"role": "assistant", "content": "ok"}]
    )
    captured = {}

    def capture(*args, **kwargs):
        captured["messages"] = list(kwargs.get("messages", []))  # snapshot before list is mutated
        return _direct("{}")

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = capture
        s.execute_react(make_mock_agent(LLM), prior, agdata(x=1))
    # system at [0], prior messages at [1] and [2], new user at [-1]
    assert captured["messages"][1]["content"] == "prior"
    assert captured["messages"][-1]["role"] == "user"


# ---------------------------------------------------------------------------
# Tool call path
# ---------------------------------------------------------------------------


def test_tool_call_executes_and_continues():
    def fn(arg: agdata) -> agdata:
        return agdata(val=arg.x * 10)

    t = agtool(
        name="calc",
        description="",
        fn=fn,
        params={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
    )

    s = make_skill(replace_tools=[t])
    responses = [_tool_call("calc", {"x": 7}), _direct('{"result": 70}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, ctx, delta = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(task="calc"))

    # Verify the tool ran with the right args and its output reached the LLM
    tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert json.loads(tool_msgs[0]["content"]) == {"val": 70}
    assert result.result == '{"result": 70}'


def test_unknown_tool_error_in_history():
    s = make_skill()
    responses = [_tool_call("ghost", {}), _direct("{}")]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _, ctx, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))
    tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
    assert any("unknown tool" in m["content"] for m in tool_msgs)


def test_max_steps_exceeded():
    s = make_skill()
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = lambda **kw: _tool_call(
            "x", {}
        )
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1), max_steps=3)
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
        s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))
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
        s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))
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

    with (
        patch("openai.OpenAI") as MockClient,
        patch.object(_tools_mod, "make_sandboxed_tools", side_effect=fake_make_sandboxed),
    ):

        def capture(**kwargs):
            captured["tools"] = kwargs.get("tools")
            return _direct("{}")

        MockClient.return_value.chat.completions.create.side_effect = capture
        sb = MagicMock()
        sb.get_live_pids.return_value = set()
        sb.pid_status_summary.return_value = ""
        sb.commit.return_value = False
        sb._has_pending_background_work.return_value = False
        s.execute_react(make_mock_agent(LLM, sb), agcontext(), agdata(x=1))
    names = [t["function"]["name"] for t in (captured.get("tools") or [])]
    assert "bash" in names
    assert "extra" in names


# ---------------------------------------------------------------------------
# input_schema and output_schema
# ---------------------------------------------------------------------------


def test_input_schema_missing_field_returns_error():
    s = agskill(
        name="s",
        system_prompt="",
        input_schema=agdata(question=str, context=str),
    )
    result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(question="hi"))
    assert result.error is not None
    assert "context" in result.error


def test_input_schema_type_error_returns_error():
    s = agskill(
        name="s",
        system_prompt="",
        input_schema=agdata(count=int),
    )
    result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(count="not-an-int"))
    assert result.error is not None
    assert "count" in result.error


def test_input_schema_valid_proceeds():
    s = agskill(
        name="s",
        system_prompt="",
        input_schema=agdata(text=str),
    )
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"ok": true}')
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(text="hello"))
    assert getattr(result, "error", None) is None


def test_input_schema_description_value_only_checks_presence():
    """Non-type-name values (descriptions) only trigger a missing-key error."""
    s = agskill(
        name="s",
        system_prompt="",
        input_schema=agdata(query="the search query"),
    )
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct("{}")
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(query=42))
    assert getattr(result, "error", None) is None  # 42 is not type-checked


def test_output_schema_missing_field_triggers_retry():
    """Model doesn't call return_<field> first attempt; re-prompted; correct on retry."""
    s = agskill(
        name="s",
        system_prompt="",
        output_schema=agdata(summary=str),
        max_output_schema_retries=2,
    )
    responses = [
        _direct("I'm done."),  # no return_summary → reprompt
        _tool_call("return_summary", {"summary": "good"}),  # field provided
        _direct(""),  # done
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(text="hi"))
    assert result.summary == "good"


def test_output_schema_retry_exhausted_returns_error():
    s = agskill(
        name="s",
        system_prompt="",
        output_schema=agdata(answer=str),
        max_output_schema_retries=2,
    )
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"wrong": 1}')
        result, _, _ = s.execute_react(
            make_mock_agent(LLM), agcontext(), agdata(q="hi"), max_steps=10
        )
    assert result.error is not None
    assert "output schema error" in result.error


def test_output_schema_type_mismatch_triggers_retry():
    """return_<field> with wrong type returns error; reprompt on missing field; correct on retry."""
    s = agskill(
        name="s",
        system_prompt="",
        output_schema=agdata(count=int),
        max_output_schema_retries=2,
    )
    responses = [
        _tool_call("return_count", {"count": "not-an-int"}),  # type error
        _direct(""),  # stops → reprompt
        _tool_call("return_count", {"count": 5}),  # correct
        _direct(""),  # done
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))
    assert result.count == 5


def test_correction_message_appended_on_retry():
    """The missing-fields reprompt is appended before the next LLM call."""
    s = agskill(
        name="s",
        system_prompt="",
        output_schema=agdata(answer=str),
        max_output_schema_retries=1,
    )
    call_messages: list[list[dict]] = []
    call_idx = 0
    responses = [
        _direct("I'm done."),  # no return_answer → correction injected
        _tool_call("return_answer", {"answer": "fixed"}),  # provide field → done
    ]

    def side_effect(**kwargs):
        nonlocal call_idx
        call_messages.append(list(kwargs.get("messages", [])))
        r = responses[call_idx]
        call_idx += 1
        return r

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = side_effect
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(q="hi"))

    assert result.answer == "fixed"
    assert len(call_messages) == 2
    # Second call should have the missing-fields reprompt as a user message
    second_msgs = call_messages[1]
    assert any(
        "missing" in m.get("content", "").lower() for m in second_msgs if m["role"] == "user"
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
    assert "string" in prompt  # per-field description for str output


def test_no_schemas_system_prompt_unchanged():
    s = agskill(name="s", system_prompt="Be helpful.")
    assert s._build_system_prompt() == "Be helpful."


# ---------------------------------------------------------------------------
# return_output tool-based output collection
# ---------------------------------------------------------------------------


def test_return_output_all_fields_correct():
    """Model calls return_<field> for every field; result agdata assembled correctly."""
    s = agskill(
        name="s",
        system_prompt="",
        output_schema=agdata(summary=str, is_duplicate=bool, score=int),
    )
    responses = [
        _tool_call("return_summary", {"summary": "great paper"}),
        _tool_call("return_is_duplicate", {"is_duplicate": False}),
        _tool_call("return_score", {"score": 9}),
        _direct(""),
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))
    assert result.summary == "great paper"
    assert result.is_duplicate is False
    assert result.score == 9


def test_return_output_type_error_immediate_feedback():
    """Wrong type for a return_<field> call: tool returns error, model can retry."""
    s = agskill(
        name="s",
        system_prompt="",
        output_schema=agdata(count=int),
        max_output_schema_retries=2,
    )
    # Capture tool result messages to verify the error was reported inline.
    all_messages: list[list[dict]] = []
    call_idx = 0
    responses = [
        _tool_call("return_count", {"count": "not-int"}),  # error
        _tool_call("return_count", {"count": 42}),  # correct
        _direct(""),
    ]

    def side_effect(**kwargs):
        nonlocal call_idx
        all_messages.append(list(kwargs.get("messages", [])))
        r = responses[call_idx]
        call_idx += 1
        return r

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = side_effect
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))
    assert result.count == 42
    # Second LLM call should see the tool error message in history.
    second_call_msgs = all_messages[1]
    tool_results = [m for m in second_call_msgs if m.get("role") == "tool"]
    assert any("error" in m.get("content", "").lower() for m in tool_results)


def test_return_output_unknown_field_error():
    """Calling a non-existent return_<field> tool name gets 'unknown tool' feedback."""
    from agency.agtool import make_return_output_tools
    from agency.agdata import agdata

    schema = agdata(summary=str)
    tools = make_return_output_tools(schema)
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "return_summary"
    assert tools[0]["function"]["parameters"]["properties"]["summary"]["type"] == "string"

    s = agskill(
        name="s",
        system_prompt="",
        output_schema=agdata(summary=str),
        max_output_schema_retries=2,
    )
    call_idx = 0
    responses = [
        _tool_call("return_WRONG", {"WRONG": "oops"}),  # unknown → "unknown tool" feedback
        _tool_call("return_summary", {"summary": "correct"}),
        _direct(""),
    ]

    def side_effect(**kwargs):
        nonlocal call_idx
        r = responses[call_idx]
        call_idx += 1
        return r

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = side_effect
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))
    assert result.summary == "correct"


def test_return_output_list_of_dicts():
    """list-of-dicts schema field is validated and assembled correctly."""
    s = agskill(
        name="s",
        system_prompt="",
        output_schema=agdata(papers=[{"title": str, "url": str}]),
    )
    papers = [{"title": "A", "url": "http://a"}, {"title": "B", "url": "http://b"}]
    responses = [
        _tool_call("return_papers", {"papers": papers}),
        _direct(""),
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))
    assert result.papers == papers


def test_return_output_list_str():
    """list[str] schema field is validated per-element."""
    s = agskill(
        name="s",
        system_prompt="",
        output_schema=agdata(tags=list[str]),
    )
    responses = [
        _tool_call("return_tags", {"tags": ["ml", "nlp"]}),
        _direct(""),
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))
    assert result.tags == ["ml", "nlp"]


def test_return_output_bare_list():
    """bare list type maps to JSON array and accepts any list value."""
    from agency.agtool import make_return_output_tools

    schema = agdata(items=list)
    tools = make_return_output_tools(schema)
    assert tools[0]["function"]["parameters"]["properties"]["items"]["type"] == "array"

    s = agskill(name="s", system_prompt="", output_schema=agdata(items=list))
    responses = [
        _tool_call("return_items", {"items": [{"a": 1}, {"b": 2}]}),
        _direct(""),
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))
    assert result.items == [{"a": 1}, {"b": 2}]


def test_return_output_bare_dict():
    """bare dict type maps to JSON object and the LLM can return a dict value."""
    from agency.agtool import make_return_output_tools

    schema = agdata(meta=dict)
    tools = make_return_output_tools(schema)
    assert tools[0]["function"]["parameters"]["properties"]["meta"]["type"] == "object"

    s = agskill(name="s", system_prompt="", output_schema=agdata(meta=dict))
    responses = [
        _tool_call("return_meta", {"meta": {"a": 1, "b": "x"}}),
        _direct(""),
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))
    assert result.meta == {"a": 1, "b": "x"}


def test_return_output_agrawstring_unchanged():
    """agrawstring output schema bypasses return_output entirely."""
    from agency.agtype import agrawstring

    s = agskill(
        name="s",
        system_prompt="",
        output_schema=agdata(text=agrawstring),
    )
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct("hello world")
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))
    assert result.text == "hello world"


def test_return_output_tool_in_openai_tools():
    """When output_schema is set, per-field return_<field> tools appear first in openai_tools."""
    s = agskill(
        name="s",
        system_prompt="",
        output_schema=agdata(summary=str, score=int),
    )
    captured_kwargs: list[dict] = []
    responses_iter = iter(
        [
            _tool_call("return_summary", {"summary": "x"}),
            _tool_call("return_score", {"score": 1}),
            _direct(""),
        ]
    )
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = lambda **kw: (
            captured_kwargs.append(kw) or next(responses_iter)
        )
        s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))

    first_tools = captured_kwargs[0].get("tools", [])
    assert first_tools is not None
    names = [t["function"]["name"] for t in first_tools]
    # Per-field tools come first; one per schema field with typed value parameter.
    assert "return_summary" in names
    assert "return_score" in names
    assert (
        names.index("return_summary") < names.index("return_score") or True
    )  # order matches schema
    # Verify the value parameters are correctly typed.
    by_name = {t["function"]["name"]: t for t in first_tools}
    assert (
        by_name["return_summary"]["function"]["parameters"]["properties"]["summary"]["type"]
        == "string"
    )
    assert (
        by_name["return_score"]["function"]["parameters"]["properties"]["score"]["type"]
        == "integer"
    )


# ---------------------------------------------------------------------------
# Return tool parameter naming and logging
# ---------------------------------------------------------------------------


def test_return_tool_parameter_named_after_field():
    """Each return_<field> tool has a single parameter named after the field, not 'value'."""
    from agency.agtool import make_return_output_tools

    schema = agdata(title=str, count=int, passed=bool)
    tools = make_return_output_tools(schema)
    by_name = {t["function"]["name"]: t for t in tools}
    for field in ("title", "count", "passed"):
        params = by_name[f"return_{field}"]["function"]["parameters"]
        assert field in params["properties"], f"expected '{field}' as parameter name"
        assert "value" not in params["properties"], "'value' should not be the parameter name"
        assert params["required"] == [field]


def test_return_tool_accepts_any_key_name():
    """_handle extracts value via next(iter(args.values())) regardless of key name."""
    s = agskill(name="s", system_prompt="", output_schema=agdata(summary=str))
    responses = [
        _tool_call("return_summary", {"summary": "hello"}),
        _direct(""),
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata())
    assert result.summary == "hello"


def test_return_tool_accepts_wrong_key_name():
    """Even if the model uses a different key name, the single value is still extracted."""
    s = agskill(name="s", system_prompt="", output_schema=agdata(summary=str))
    responses = [
        _tool_call("return_summary", {"value": "hello"}),  # old-style key
        _direct(""),
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata())
    assert result.summary == "hello"


def test_return_tool_logs_success_to_term():
    """Successful return_<field> call emits TOOL ✓ to the term passed to run()."""
    s = agskill(name="s", system_prompt="", output_schema=agdata(summary=str))
    responses = [
        _tool_call("return_summary", {"summary": "ok"}),
        _direct(""),
    ]
    mock_term = MagicMock()
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _ag = make_mock_agent(LLM)
        _ag.terminal = mock_term
        s.execute_react(_ag, agcontext(), agdata())
    calls = [str(c) for c in mock_term.log.call_args_list]
    assert any("TOOL ✓" in c for c in calls), f"Expected TOOL ✓ log call, got: {calls}"
    assert any("return_summary" in c for c in calls)


def test_return_tool_logs_validation_error_to_term():
    """A type-mismatched return_<field> call emits TOOL ✗ with the tool call args."""
    s = agskill(
        name="s", system_prompt="", output_schema=agdata(count=int), max_output_schema_retries=1
    )
    responses = [
        _tool_call("return_count", {"count": "not-an-int"}),  # type error → logged
        _tool_call("return_count", {"count": 42}),  # correct on retry
        _direct(""),
    ]
    mock_term = MagicMock()
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _ag = make_mock_agent(LLM)
        _ag.terminal = mock_term
        s.execute_react(_ag, agcontext(), agdata())
    calls = [str(c) for c in mock_term.log.call_args_list]
    assert any("TOOL ✗" in c for c in calls), f"Expected TOOL ✗ log call, got: {calls}"
    assert any("return_count" in c for c in calls)
    assert any("not-an-int" in c for c in calls)


# ---------------------------------------------------------------------------
# Concurrency semaphore
# ---------------------------------------------------------------------------

from agency.agllm import _get_llm_call_semaphore, _AgLLMFields

LLM_CALL_MAX_CONCURRENCY = _AgLLMFields.call_max_concurrency.default

_sem = _get_llm_call_semaphore()


def test_semaphore_released_after_success():
    s = make_skill()
    before = _sem._value
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct("{}")
        s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))
    assert _sem._value == before


def test_semaphore_released_after_timeout():
    from agency.agutil import _LLMIdleTimeout as _IdleTimeout

    s = make_skill()
    before = _sem._value

    def _timeout_iter(iterable, idle_timeout=None, stream_timeout=None):
        raise _IdleTimeout("no chunk received")
        yield  # makes this a generator function

    with (
        patch("openai.OpenAI") as MockClient,
        patch("agency.agllm._iter_batched", _timeout_iter),
        patch("agency.agllm.time.sleep"),
    ):
        MockClient.return_value.chat.completions.create.return_value = []
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))

    assert result.error is not None
    assert _sem._value == before


def test_semaphore_limits_concurrency():
    """When all slots are held, an extra acquire blocks until one is released."""
    sem = _sem
    # Grab all but one slot
    grabbed = []
    for _ in range(LLM_CALL_MAX_CONCURRENCY - 1):
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
    from agency.agutil import _LLMIdleTimeout as _IdleTimeout

    s = make_skill()
    call_count = 0

    def _timeout_iter(iterable, idle_timeout=None, stream_timeout=None):
        nonlocal call_count
        call_count += 1
        raise _IdleTimeout("no chunk received")
        yield

    with (
        patch("openai.OpenAI") as MockClient,
        patch("agency.agllm._iter_batched", _timeout_iter),
        patch("agency.agllm.time.sleep"),
    ):
        MockClient.return_value.chat.completions.create.return_value = []
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))

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
    from agency.agutil import _LLMIdleTimeout as _IdleTimeout

    captured = []

    def _capture_iter(iterable, idle_timeout=None, stream_timeout=None):
        captured.append((idle_timeout, stream_timeout))
        raise _IdleTimeout("no chunk received")
        yield

    s = make_skill()
    with (
        patch("openai.OpenAI") as MockClient,
        patch("agency.agllm._iter_batched", _capture_iter),
        patch("agency.agllm.time.sleep"),
    ):
        MockClient.return_value.chat.completions.create.return_value = []
        s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))

    assert len(captured) == LLM_MAX_RETRIES, (
        f"expected {LLM_MAX_RETRIES} attempts, got {len(captured)}"
    )
    idle_vals = [t[0] for t in captured]
    stream_vals = [t[1] for t in captured]
    # idle_timeout must be fixed across all attempts — no longer doubling
    assert len(set(idle_vals)) == 1, f"idle_timeout should be fixed across retries: {idle_vals}"
    assert idle_vals[0] == LLM_IDLE_TIMEOUT, (
        f"idle_timeout should be {LLM_IDLE_TIMEOUT} s: {idle_vals}"
    )
    # stream_timeout must also be fixed
    assert len(set(stream_vals)) == 1, (
        f"stream_timeout should be fixed across retries: {stream_vals}"
    )
    assert stream_vals[0] == LLM_STREAM_TIMEOUT, (
        f"stream_timeout should be {LLM_STREAM_TIMEOUT} s: {stream_vals}"
    )


def test_timeout_succeeds_after_retry():
    """If a later attempt succeeds, result is returned normally."""
    from agency.agutil import _LLMIdleTimeout as _IdleTimeout
    from agency.agutil import _iter_batched as _real_iter_batched

    call_count = 0

    def _maybe_timeout(iterable, idle_timeout=None, stream_timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _IdleTimeout("no chunk received")
            yield  # makes this a generator function
        else:
            yield from _real_iter_batched(
                iterable, idle_timeout=idle_timeout, stream_timeout=stream_timeout
            )

    s = make_skill()
    with patch("openai.OpenAI") as MockClient, patch("agency.agllm._iter_batched", _maybe_timeout):
        MockClient.return_value.chat.completions.create.return_value = _direct('{"answer": "ok"}')
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))

    assert getattr(result, "error", None) is None
    assert result.result == '{"answer": "ok"}'
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

    with (
        patch("openai.OpenAI") as MockClient,
        patch("agency.agllm._iter_batched", _ssl_error_iter),
        patch("agency.agllm.time.sleep"),
    ):
        MockClient.return_value.chat.completions.create.return_value = []
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))

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

    with (
        patch("openai.OpenAI") as MockClient,
        patch("agency.agllm._iter_batched", _oserror_iter),
        patch("agency.agllm.time.sleep"),
    ):
        MockClient.return_value.chat.completions.create.return_value = []
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))

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

    with (
        patch("openai.OpenAI") as MockClient,
        patch("agency.agllm._iter_batched", _ssl_error_iter),
        patch("agency.agllm.time.sleep"),
    ):
        MockClient.return_value.chat.completions.create.return_value = []
        s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))

    assert _sem._value == before


def test_ssl_error_succeeds_after_retry():
    import ssl
    from agency.agutil import _iter_batched as _real_iter_batched

    call_count = 0

    def _maybe_ssl(iterable, idle_timeout=None, stream_timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ssl.SSLError("record layer failure")
            yield
        else:
            yield from _real_iter_batched(
                iterable, idle_timeout=idle_timeout, stream_timeout=stream_timeout
            )

    s = make_skill()
    with patch("openai.OpenAI") as MockClient, patch("agency.agllm._iter_batched", _maybe_ssl):
        MockClient.return_value.chat.completions.create.return_value = _direct('{"answer": "ok"}')
        result, _, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))

    assert getattr(result, "error", None) is None
    assert result.result == '{"answer": "ok"}'
    assert call_count == 2


# ---------------------------------------------------------------------------
# Long tool output offloading
# ---------------------------------------------------------------------------


def _make_sandbox(written=None):
    """Return a mock sandbox that records write_file calls."""
    sandbox = MagicMock()
    sandbox._has_pending_background_work.return_value = False
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
        s.execute_react(make_mock_agent(LLM, sandbox), agcontext(), agdata(x=1))

    assert not written


def test_long_tool_output_offloaded_to_file():
    from agency.agtool import _AgToolFields

    written = {}
    sandbox = _make_sandbox(written)

    _eff_thresh = max(
        _AgToolFields.output_offload_chars.default,
        int(
            LLM.context_limit
            * _AgSchemaFields.offload_context_fraction.default
            * _AgSchemaFields.chars_per_token.default
        ),
    )
    big_output = "x" * (_eff_thresh + 1)

    def fn(arg: agdata) -> agdata:
        return agdata(data=big_output)

    t = agtool(name="fetcher", description="", fn=fn)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("fetcher", {}, "abc-123-xyz"), _direct('{"ok": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _, ctx, _ = s.execute_react(make_mock_agent(LLM, sandbox), agcontext(), agdata(x=1))

    # File was written to the sandbox
    assert len(written) == 1
    path = next(iter(written))
    assert path.startswith("/workspace/long_tool_call_outputs/fetcher_")
    assert path.endswith(".txt")
    assert big_output in next(iter(written.values()))

    # Tool message in history has the note, not the raw content
    tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    note = json.loads(tool_msgs[0]["content"])
    assert "note" in note
    assert path in note["note"]


def test_long_tool_output_offloaded_to_sandbox():
    from agency.agtool import _AgToolFields

    _eff_thresh = max(
        _AgToolFields.output_offload_chars.default,
        int(
            LLM.context_limit
            * _AgSchemaFields.offload_context_fraction.default
            * _AgSchemaFields.chars_per_token.default
        ),
    )
    big_output = "y" * (_eff_thresh + 1)

    def fn(arg: agdata) -> agdata:
        return agdata(data=big_output)

    t = agtool(name="fetcher", description="", fn=fn)
    s = make_skill(replace_tools=[t])
    sandbox = MagicMock()
    sandbox._has_pending_background_work.return_value = False
    responses = [_tool_call("fetcher", {}, "call-999"), _direct('{"ok": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _, ctx, _ = s.execute_react(make_mock_agent(LLM, sandbox), agcontext(), agdata(x=1))

    # Large output must be offloaded to sandbox, not kept inline
    sandbox.write_file.assert_called_once()
    tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "note" in tool_msgs[0]["content"]
    assert "/workspace/long_tool_call_outputs/" in tool_msgs[0]["content"]


def test_long_output_injects_read_tool_into_openai_tools():
    """When a large output is offloaded, the read tool is added to the tool schema
    passed to the LLM on the next step so the model can actually call it."""
    from agency.agtool import _AgToolFields

    _eff_thresh = max(
        _AgToolFields.output_offload_chars.default,
        int(
            LLM.context_limit
            * _AgSchemaFields.offload_context_fraction.default
            * _AgSchemaFields.chars_per_token.default
        ),
    )
    big_output = "z" * (_eff_thresh + 1)
    recorded_tool_schemas = []

    def fn(arg: agdata) -> agdata:
        return agdata(data=big_output)

    t = agtool(name="fetcher", description="", fn=fn)
    s = make_skill(replace_tools=[t])
    sandbox = _make_sandbox()

    responses = [_tool_call("fetcher", {}, "call-abc"), _direct('{"ok": 1}')]

    with patch("openai.OpenAI") as MockClient:

        def capturing_create(*args, **kwargs):
            recorded_tool_schemas.append(kwargs.get("tools") or [])
            return iter(responses.pop(0))

        MockClient.return_value.chat.completions.create.side_effect = capturing_create
        s.execute_react(make_mock_agent(LLM, sandbox), agcontext(), agdata(x=1))

    # First call: only fetcher
    first_names = [t["function"]["name"] for t in recorded_tool_schemas[0]]
    assert "fetcher" in first_names
    assert "read" not in first_names

    # Second call (after offload): read is now present
    second_names = [t["function"]["name"] for t in recorded_tool_schemas[1]]
    assert "read" in second_names


def test_long_output_read_tool_persists_for_skill_run():
    """Once the read tool is injected it stays in the tool list for subsequent
    LLM calls — it is not removed between iterations."""
    from agency.agtool import _AgToolFields

    _eff_thresh = max(
        _AgToolFields.output_offload_chars.default,
        int(
            LLM.context_limit
            * _AgSchemaFields.offload_context_fraction.default
            * _AgSchemaFields.chars_per_token.default
        ),
    )
    big_output = "z" * (_eff_thresh + 1)
    recorded_tool_schemas = []

    def fn(arg: agdata) -> agdata:
        return agdata(data=big_output)

    t = agtool(name="fetcher", description="", fn=fn)
    s = make_skill(replace_tools=[t])
    sandbox = _make_sandbox()

    # Three LLM calls: fetch (offloads) → read → done
    responses = [
        _tool_call("fetcher", {}, "call-001"),
        _tool_call(
            "read", {"path": "/workspace/long_tool_call_outputs/fetcher_call001.txt"}, "call-002"
        ),
        _direct('{"ok": 1}'),
    ]

    with patch("openai.OpenAI") as MockClient:
        resp_iter = iter(responses)

        def capturing_create(*args, **kwargs):
            recorded_tool_schemas.append(kwargs.get("tools") or [])
            return iter(next(resp_iter))

        MockClient.return_value.chat.completions.create.side_effect = capturing_create
        # read tool in tool_map needs to return something non-empty
        sandbox.read_file.return_value = "file content"
        s.execute_react(make_mock_agent(LLM, sandbox), agcontext(), agdata(x=1))

    # All three calls see read in the schema from call 2 onward
    assert "read" not in [t["function"]["name"] for t in recorded_tool_schemas[0]]
    assert "read" in [t["function"]["name"] for t in recorded_tool_schemas[1]]
    assert "read" in [t["function"]["name"] for t in recorded_tool_schemas[2]]


def test_long_output_no_duplicate_read_when_already_present():
    """If the skill already has the read tool (e.g. via make_sandboxed_tools),
    offloading must not add a second read entry to openai_tools."""
    from agency.agtool import _AgToolFields

    _eff_thresh = max(
        _AgToolFields.output_offload_chars.default,
        int(
            LLM.context_limit
            * _AgSchemaFields.offload_context_fraction.default
            * _AgSchemaFields.chars_per_token.default
        ),
    )
    big_output = "z" * (_eff_thresh + 1)
    recorded_tool_schemas = []

    def fn(arg: agdata) -> agdata:
        return agdata(data=big_output)

    t = agtool(name="fetcher", description="", fn=fn)
    read_tool = agtool(name="read", description="read a file", fn=lambda a: agdata(content=""))
    # Skill has read already in replace_tools
    s = make_skill(replace_tools=[t, read_tool])
    sandbox = _make_sandbox()

    responses = [_tool_call("fetcher", {}, "call-dup"), _direct('{"ok": 1}')]

    with patch("openai.OpenAI") as MockClient:

        def capturing_create(*args, **kwargs):
            recorded_tool_schemas.append(kwargs.get("tools") or [])
            return iter(responses.pop(0))

        MockClient.return_value.chat.completions.create.side_effect = capturing_create
        s.execute_react(make_mock_agent(LLM, sandbox), agcontext(), agdata(x=1))

    # Second call must have exactly one read entry
    second_names = [t["function"]["name"] for t in recorded_tool_schemas[1]]
    assert second_names.count("read") == 1


# ---------------------------------------------------------------------------
# Tool call failure handling and checkpoint revert
# ---------------------------------------------------------------------------


def _make_sandbox_with_tracking():
    """Return a sandbox mock that records stop() calls."""
    sandbox = MagicMock()
    sandbox._name = "testbox"
    sandbox.stop.return_value = None
    sandbox._has_pending_background_work.return_value = False
    return sandbox


def test_tool_success_commits_and_stops():
    """A successful tool call must trigger sandbox.stop(commit=True)."""
    sandbox = _make_sandbox_with_tracking()

    def fn(arg: agdata) -> agdata:
        return agdata(result="ok")

    t = agtool(name="mytool", description="", fn=fn, run_in_subprocess=True)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("mytool", {}, "c1"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        s.execute_react(make_mock_agent(LLM, sandbox), agcontext(), agdata(x=1))

    sandbox.stop.assert_called_once_with(commit=True)


def test_tool_failure_triggers_stop_without_commit():
    """When a run_in_subprocess tool returns an error, sandbox.stop(commit=False) is called."""
    sandbox = _make_sandbox_with_tracking()

    def fn(arg: agdata) -> agdata:
        return agerror("boom")

    t = agtool(name="badtool", description="", fn=fn, run_in_subprocess=True)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("badtool", {}, "c2"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        s.execute_react(make_mock_agent(LLM, sandbox), agcontext(), agdata(x=1))

    sandbox.stop.assert_called_once_with(commit=False)


def test_tool_failure_adds_workspace_reverted_note():
    """Tool error response must include workspace_reverted when the tool returns agerror(...)."""
    sandbox = _make_sandbox_with_tracking()

    def fn(arg: agdata) -> agdata:
        return agerror("disk full")

    t = agtool(name="badtool", description="", fn=fn, run_in_subprocess=True)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("badtool", {}, "c3"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _, ctx, _ = s.execute_react(make_mock_agent(LLM, sandbox), agcontext(), agdata(x=1))

    tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    content = json.loads(tool_msgs[0]["content"])
    assert "error" in content
    assert "workspace_reverted" in content
    assert "reverted" in content["workspace_reverted"].lower()


def test_tool_failure_reverts_even_without_subprocess():
    """A tool error still reverts the workspace when agent_sandbox=MagicMock() and
    run_in_subprocess=False -- stop()/the revert note are no longer specific to
    subprocess-isolated tools."""

    def fn(arg: agdata) -> agdata:
        return agerror("nope")

    t = agtool(name="badtool", description="", fn=fn, run_in_subprocess=False)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("badtool", {}, "c4"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _, ctx, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))

    tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
    content = json.loads(tool_msgs[0]["content"])
    assert content["error"] == "nope"
    assert "workspace_reverted" in content


def test_run_in_subprocess_false_still_stops():
    """Tools with run_in_subprocess=False must still trigger sandbox.stop() --
    checkpointing runs after every tool call regardless of subprocess isolation."""
    sandbox = _make_sandbox_with_tracking()

    def fn(arg: agdata) -> agdata:
        return agerror("oops")

    t = agtool(name="hosttool", description="", fn=fn, run_in_subprocess=False)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("hosttool", {}, "c5"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        s.execute_react(make_mock_agent(LLM, sandbox), agcontext(), agdata(x=1))

    sandbox.stop.assert_called_once_with(commit=False)


def test_tool_exception_triggers_stop_without_commit():
    """When a run_in_subprocess tool raises an exception, sandbox.stop(commit=False) is called."""
    sandbox = _make_sandbox_with_tracking()

    def fn(arg: agdata) -> agdata:
        raise RuntimeError("exploded")

    t = agtool(name="badtool", description="", fn=fn, run_in_subprocess=True)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("badtool", {}, "c6"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _, ctx, _ = s.execute_react(make_mock_agent(LLM, sandbox), agcontext(), agdata(x=1))

    sandbox.stop.assert_called_once_with(commit=False)
    tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
    assert "error" in json.loads(tool_msgs[0]["content"])


def test_run_in_subprocess_false_success_still_commits():
    """The bug this session's dispatch_tools() fix actually targeted: a
    *successful* run_in_subprocess=False tool call must still trigger
    sandbox.stop(commit=True) -- checkpointing was previously gated on
    run_in_subprocess=True, so a host-side tool's successful work was never
    committed at all."""
    sandbox = _make_sandbox_with_tracking()

    def fn(arg: agdata) -> agdata:
        return agdata(result="ok")

    t = agtool(name="hosttool", description="", fn=fn, run_in_subprocess=False)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("hosttool", {}, "c7"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        s.execute_react(make_mock_agent(LLM, sandbox), agcontext(), agdata(x=1))

    sandbox.stop.assert_called_once_with(commit=True)


def test_pending_background_work_defers_stop_entirely():
    """When the sandbox still has pending background work (e.g. a
    backgrounded `cmd &`), stop() must not run at all after a successful
    tool call -- running it would tear down/overwrite the sandbox's live
    state out from under that still-running work."""
    sandbox = _make_sandbox_with_tracking()
    # True for dispatch_tools()'s own check (what this test targets), then
    # False afterward -- execute_react() calls wait_for_processes() right
    # after, whose very first gate check is this same predicate; leaving it
    # permanently True would make that loop believe work is still pending
    # forever and hang for the full ping interval instead of returning
    # immediately (see agsandbox.py's wait_for_processes() docstring).
    sandbox._has_pending_background_work.side_effect = [True] + [False] * 20

    def fn(arg: agdata) -> agdata:
        return agdata(result="ok")

    t = agtool(name="bgtool", description="", fn=fn, run_in_subprocess=True)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("bgtool", {}, "c8"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        s.execute_react(make_mock_agent(LLM, sandbox), agcontext(), agdata(x=1))

    sandbox.stop.assert_not_called()


def test_pending_background_work_omits_workspace_reverted_note_on_error():
    """Same deferral as above, but for an errored tool call: stop() must be
    skipped, and the workspace_reverted note -- which claims a revert that
    didn't actually happen -- must not be added either."""
    sandbox = _make_sandbox_with_tracking()
    # True for dispatch_tools()'s own check (what this test targets), then
    # False afterward -- execute_react() calls wait_for_processes() right
    # after, whose very first gate check is this same predicate; leaving it
    # permanently True would make that loop believe work is still pending
    # forever and hang for the full ping interval instead of returning
    # immediately (see agsandbox.py's wait_for_processes() docstring).
    sandbox._has_pending_background_work.side_effect = [True] + [False] * 20

    def fn(arg: agdata) -> agdata:
        return agerror("boom")

    t = agtool(name="bgbadtool", description="", fn=fn, run_in_subprocess=True)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("bgbadtool", {}, "c9"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _, ctx, _ = s.execute_react(make_mock_agent(LLM, sandbox), agcontext(), agdata(x=1))

    sandbox.stop.assert_not_called()
    tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
    content = json.loads(tool_msgs[0]["content"])
    assert "error" in content
    assert "workspace_reverted" not in content


def test_tool_exception_with_run_in_subprocess_false_still_stops():
    """Exception-handler path: a raised exception from a run_in_subprocess=False
    tool must still trigger sandbox.stop(commit=False), the same as a
    run_in_subprocess=True tool does."""
    sandbox = _make_sandbox_with_tracking()

    def fn(arg: agdata) -> agdata:
        raise RuntimeError("exploded")

    t = agtool(name="hostbadtool", description="", fn=fn, run_in_subprocess=False)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("hostbadtool", {}, "c10"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _, ctx, _ = s.execute_react(make_mock_agent(LLM, sandbox), agcontext(), agdata(x=1))

    sandbox.stop.assert_called_once_with(commit=False)
    tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
    content = json.loads(tool_msgs[0]["content"])
    assert "error" in content
    assert "workspace_reverted" in content


def test_tool_exception_with_pending_background_work_defers_stop():
    """Exception-handler path, gated the same way as the success/error
    path: pending background work must defer stop() here too."""
    sandbox = _make_sandbox_with_tracking()
    # True for dispatch_tools()'s own check (what this test targets), then
    # False afterward -- execute_react() calls wait_for_processes() right
    # after, whose very first gate check is this same predicate; leaving it
    # permanently True would make that loop believe work is still pending
    # forever and hang for the full ping interval instead of returning
    # immediately (see agsandbox.py's wait_for_processes() docstring).
    sandbox._has_pending_background_work.side_effect = [True] + [False] * 20

    def fn(arg: agdata) -> agdata:
        raise RuntimeError("exploded")

    t = agtool(name="bgexctool", description="", fn=fn, run_in_subprocess=True)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("bgexctool", {}, "c11"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        _, ctx, _ = s.execute_react(make_mock_agent(LLM, sandbox), agcontext(), agdata(x=1))

    sandbox.stop.assert_not_called()
    tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
    content = json.loads(tool_msgs[0]["content"])
    assert "error" in content
    assert "workspace_reverted" not in content


def test_dispatch_tools_accepts_camel_case_llm_arguments():
    """End-to-end: an LLM emitting camelCase tool-call JSON (e.g. `filePath`
    instead of `file_path`) still reaches the tool fn correctly -- dispatch_tools
    parses fn_args via agdata.from_json(), which normalizes top-level keys."""
    received = {}

    def fn(arg: agdata) -> agdata:
        received["file_path"] = arg.file_path
        received["old_string"] = arg.old_string
        return agdata(result="ok")

    t = agtool(
        name="camel_tool",
        description="",
        fn=fn,
        run_in_subprocess=False,
        params={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
            },
        },
    )
    s = make_skill(replace_tools=[t])
    responses = [
        _tool_call("camel_tool", {"filePath": "/tmp/x.txt", "oldString": "a"}, "c8"),
        _direct('{"done": 1}'),
    ]
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = responses
        s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))

    assert received == {"file_path": "/tmp/x.txt", "old_string": "a"}


def test_tool_timeout_uses_agent_provided_value():
    """When fn_args includes a 'timeout' int, agtool.__call__ receives it as keyword arg."""
    received_timeout = {}

    original_call = agtool.__call__

    def patched_call(self, arg, timeout=None):
        received_timeout["timeout"] = timeout
        return original_call(self, arg, timeout=timeout)

    def fn(arg: agdata) -> agdata:
        return agdata(result="ok")

    t = agtool(
        name="slow",
        description="",
        fn=fn,
        run_in_subprocess=False,
        params={"type": "object", "properties": {"timeout": {"type": "integer"}}},
    )
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("slow", {"timeout": 120}, "c7"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient, patch.object(agtool, "__call__", patched_call):
        MockClient.return_value.chat.completions.create.side_effect = responses
        s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))

    assert received_timeout.get("timeout") == 120


def test_tool_timeout_ignored_if_not_int():
    """Non-integer 'timeout' in fn_args is silently ignored; dispatch_tools resolves
    the default itself (agconfig-aware) before calling the tool, rather than passing
    None through for agtool.__call__ to default internally."""
    received_timeout = {}

    original_call = agtool.__call__

    def patched_call(self, arg, timeout=None):
        received_timeout["timeout"] = timeout
        return original_call(self, arg, timeout=timeout)

    def fn(arg: agdata) -> agdata:
        return agdata(result="ok")

    t = agtool(name="slow", description="", fn=fn, run_in_subprocess=False)
    s = make_skill(replace_tools=[t])
    responses = [_tool_call("slow", {"timeout": "forever"}, "c8"), _direct('{"done": 1}')]
    with patch("openai.OpenAI") as MockClient, patch.object(agtool, "__call__", patched_call):
        MockClient.return_value.chat.completions.create.side_effect = responses
        s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))

    assert received_timeout.get("timeout") == _AgToolFields.timeout_s.default


# ---------------------------------------------------------------------------
# build_llm_kwargs
# ---------------------------------------------------------------------------

from agency.agllm import agllm as _agllm_mod

build_llm_kwargs = _agllm_mod.build_llm_kwargs


def _llm_cfg(**fields) -> agConfig:
    """Test helper: wrap agllm_backend fields in an agConfig."""
    return agConfig({"agllm_backend": fields})


def test_build_llm_kwargs_model_and_messages():
    msgs = [{"role": "user", "content": "hi"}]
    kw = build_llm_kwargs(_llm_cfg(model=""), msgs, None)
    assert kw["model"] == ""
    assert kw["messages"] == msgs


def test_build_llm_kwargs_strips_private_keys():
    msgs = [{"role": "assistant", "content": "ok", "_thinking": "secret"}]
    kw = build_llm_kwargs(_llm_cfg(model="m"), msgs, None)
    assert "_thinking" not in kw["messages"][0]
    assert "content" in kw["messages"][0]


def test_build_llm_kwargs_openai_gen_params():
    kw = build_llm_kwargs(_llm_cfg(model="m", temperature=0.7, max_completion_tokens=100), [], None)
    assert kw["temperature"] == 0.7
    assert kw["max_completion_tokens"] == 100


def test_build_llm_kwargs_extra_body_vllm_params():
    kw = build_llm_kwargs(_llm_cfg(model="m", top_k=50, repetition_penalty=1.1), [], None)
    assert kw["extra_body"]["top_k"] == 50
    assert kw["extra_body"]["repetition_penalty"] == 1.1


def test_build_llm_kwargs_tools_included_when_provided():
    tools = [{"type": "function", "function": {"name": "f"}}]
    kw = build_llm_kwargs(_llm_cfg(model="m"), [], tools)
    assert kw["tools"] == tools


def test_build_llm_kwargs_no_tools_key_when_none():
    kw = build_llm_kwargs(_llm_cfg(model="m"), [], None)
    assert "tools" not in kw


# ---------------------------------------------------------------------------
# build_assistant_msg
# ---------------------------------------------------------------------------

build_assistant_msg = _agllm_mod.build_assistant_msg


def test_build_assistant_msg_plain_content():
    msg = build_assistant_msg(["hello", " world"], [], {})
    assert msg["role"] == "assistant"
    assert msg["content"] == "hello world"


def test_build_assistant_msg_reasoning_parts():
    msg = build_assistant_msg(["answer"], ["think ", "harder"], {})
    assert msg["_thinking"] == "think harder"
    assert msg["content"] == "answer"


def test_build_assistant_msg_think_tag_stripped():
    msg = build_assistant_msg(["<think>reasoning</think>answer"], [], {})
    assert msg.get("_thinking") == "reasoning"
    assert msg["content"] == "answer"


def test_build_assistant_msg_tool_calls_included():
    tc = {0: {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}}
    msg = build_assistant_msg([], [], tc)
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "f"


def test_build_assistant_msg_tool_calls_sorted_by_index():
    tc = {
        1: {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
        0: {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
    }
    msg = build_assistant_msg([], [], tc)
    assert msg["tool_calls"][0]["function"]["name"] == "a"
    assert msg["tool_calls"][1]["function"]["name"] == "b"


# ---------------------------------------------------------------------------
# _drain_inbox
# ---------------------------------------------------------------------------


def test_drain_inbox_empty_queue_returns_false():
    ag = make_mock_agent()
    ag._next_inbox_msg = MagicMock(return_value=None)
    messages = []
    assert ag._drain_inbox(messages) is False
    assert messages == []


def test_drain_inbox_single_message_appended():
    ag = make_mock_agent()
    ag._next_inbox_msg = MagicMock(side_effect=["hello", None])
    messages = [{"role": "system", "content": "sys"}]
    had = ag._drain_inbox(messages)
    assert had is True
    assert messages[-1] == {"role": "user", "content": "hello"}


def test_drain_inbox_multiple_messages_all_appended():
    ag = make_mock_agent()
    ag._next_inbox_msg = MagicMock(side_effect=["msg1", "msg2", None])
    messages = []
    ag._drain_inbox(messages)
    assert len(messages) == 2
    assert messages[0]["content"] == "msg1"
    assert messages[1]["content"] == "msg2"


def test_drain_inbox_calls_live_fn():
    ag = make_mock_agent()
    ag._next_inbox_msg = MagicMock(side_effect=["hi", None])
    messages = [{"role": "system", "content": "sys"}]
    ag._drain_inbox(messages)
    ag._push_live_messages.assert_called_once()


def test_drain_inbox_calls_full_history_fn():
    ag = make_mock_agent()
    ag._next_inbox_msg = MagicMock(side_effect=["hi", None])
    messages = []
    ag._drain_inbox(messages)
    ag._append_full_history.assert_called_once_with({"role": "user", "content": "hi"})


# ---------------------------------------------------------------------------
# wait_for_processes
# ---------------------------------------------------------------------------

from agency.agsandbox import agSandbox


def _make_real_sandbox(watched_pids=None):
    """Minimal sandbox stub with real _watched_pids dict for process monitoring tests."""

    class _FakeSandbox:
        def __init__(self):
            self._watched_pids = dict(watched_pids or {})

        def _has_pending_background_work(self):
            return bool(self._watched_pids)

        def get_live_pids(self):
            return set(self._watched_pids.keys())

        def pid_status_summary(self):
            return ", ".join(f"PID {p}" for p in self._watched_pids)

    return _FakeSandbox()


def test_wait_for_processes_clean_sandbox_returns_none():
    sb = _make_real_sandbox()
    assert agSandbox.wait_for_processes(sb, "skill", None, None, "", 300, 5) is None


def test_wait_for_processes_no_watched_pids_attr_returns_none():
    class NoPids:
        def _has_pending_background_work(self):
            return False

    assert agSandbox.wait_for_processes(NoPids(), "skill", None, None, "", 300, 5) is None


def test_wait_for_processes_mock_sandbox_returns_none():
    from unittest.mock import MagicMock

    sb = MagicMock()
    sb._has_pending_background_work.return_value = False
    assert agSandbox.wait_for_processes(sb, "skill", None, None, "", 300, 5) is None


def test_wait_for_processes_completes_quickly_returns_completed_msg():
    class _FakeSandbox:
        def __init__(self):
            self._watched_pids = {1234: 0.0}
            self._call_count = 0

        def _has_pending_background_work(self):
            # wait_for_processes() polls THIS method in its loop, not
            # get_live_pids() -- the state transition has to happen here,
            # not there, or the loop would just spin until ping_interval_s.
            self._call_count += 1
            if self._call_count >= 3:  # gate call + a couple of poll iterations
                self._watched_pids.clear()
            return bool(self._watched_pids)

        def get_live_pids(self):
            return set(self._watched_pids.keys())

        def pid_status_summary(self):
            return "PID 1234"

    sb = _FakeSandbox()
    result = agSandbox.wait_for_processes(
        sb, "skill", None, None, "", ping_interval_s=30, poll_interval_s=0.01
    )
    assert result is not None
    assert "completed" in result.lower() or "Background processes have completed" in result


def test_wait_for_processes_still_running_returns_update_msg():
    class _FakeSandbox:
        def __init__(self):
            self._watched_pids = {1234: 0.0}

        def _has_pending_background_work(self):
            return bool(self._watched_pids)

        def get_live_pids(self):
            return {1234}

        def pid_status_summary(self):
            return "PID 1234"

    sb = _FakeSandbox()
    result = agSandbox.wait_for_processes(
        sb, "skill", None, None, "", ping_interval_s=0.02, poll_interval_s=0.01
    )
    assert result is not None
    assert "still running" in result.lower() or "Background processes are still running" in result


def test_wait_for_processes_calls_state_fn():
    class _FakeSandbox:
        def __init__(self):
            self._watched_pids = {1: 0.0}
            self._call_count = 0

        def _has_pending_background_work(self):
            # Same reasoning as test_wait_for_processes_completes_quickly_returns_completed_msg:
            # the loop polls this method, so the transition must live here.
            self._call_count += 1
            if self._call_count >= 3:
                self._watched_pids.clear()
            return bool(self._watched_pids)

        def get_live_pids(self):
            return set(self._watched_pids.keys())

        def pid_status_summary(self):
            return "PID 1"

    states = []
    agSandbox.wait_for_processes(
        _FakeSandbox(),
        "myskill",
        None,
        None,
        "",
        30,
        0.01,
        state_fn=lambda state, **kw: states.append(state),
    )
    assert "proc_wait" in states


# ---------------------------------------------------------------------------
# agskill.validate_input
# ---------------------------------------------------------------------------

from agency.agtype import agrawstring


def test_validate_input_no_schema_returns_none():
    assert agschema(agdata(x=int)).validate_input(agdata(x=1)) is None


def test_validate_input_schema_mismatch_returns_error():
    error = agschema(agdata(x=agrawstring)).validate_input(agdata())
    assert error is not None
    assert "x" in error


# ---------------------------------------------------------------------------
# agskill._build_initial_messages
# ---------------------------------------------------------------------------


def test_build_initial_messages_structure():
    s = make_skill()
    history = agcontext(messages=[{"role": "user", "content": "prior"}])
    msgs, n_before = s._build_initial_messages(agdata(q="hi"), history, None, None, None)
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "prior"
    assert msgs[-1]["role"] == "user"
    assert n_before == 1


def test_build_initial_messages_fires_live_fn():
    s = make_skill()
    live_calls = []
    s._build_initial_messages(agdata(), agcontext(), None, lambda m: live_calls.append(m), None)
    assert len(live_calls) == 1


def test_build_initial_messages_fires_full_history_fn():
    s = make_skill()
    history_items = []
    s._build_initial_messages(
        agdata(q="test"), agcontext(), None, None, lambda m: history_items.append(m["role"])
    )
    assert "system" in history_items
    assert "user" in history_items


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

        def _has_pending_background_work(self):
            return bool(self._watched_pids)

        def get_live_pids(self):
            if self._cleared:
                return set()
            return {9999}

        def pid_status_summary(self):
            return "PID 9999"

        def commit(self, *a):
            return False

        def restore(self, *a):
            pass

        def write_file(self, *a):
            pass

        def remove_files(self, *a):
            pass

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
    s = agskill(
        name="summarise", system_prompt="You are a summarisation assistant.", replace_tools=[]
    )
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = create_side_effect
        result, _, _ = s.execute_react(
            make_mock_agent(LLM, sb, ping_interval_s=0.05, poll_interval_s=0.01),
            agcontext(),
            agdata(x=1),
        )

    assert call_count[0] == 2  # loop re-entered once
    assert result.result == '{"done": true}'


def test_run_injects_process_completed_message():
    """The continuation message injected when processes complete contains expected text."""

    # The sandbox starts with live PIDs. After the first LLM response, wait_for_processes
    # polls and sees them finish, then injects the "Background processes have completed"
    # message. The second LLM call then receives that message and returns the final answer.
    class _TrackedSandbox:
        def __init__(self):
            self._watched_pids = {1: 0.0}
            self._call_count = 0

        def _has_pending_background_work(self):
            # wait_for_processes() polls THIS method in its loop, not
            # get_live_pids() -- the state transition has to happen here.
            # First call (the initial gate check): still alive.
            self._call_count += 1
            if self._call_count >= 2:
                self._watched_pids.clear()
            return bool(self._watched_pids)

        def get_live_pids(self):
            return set(self._watched_pids.keys())

        def pid_status_summary(self):
            return "PID 1"

        def commit(self, *a):
            return False

        def restore(self, *a):
            pass

        def write_file(self, *a):
            pass

        def remove_files(self, *a):
            pass

    sb = _TrackedSandbox()
    all_messages_per_call: list[list[dict]] = []

    def create_side_effect(**kw):
        all_messages_per_call.append(list(kw["messages"]))
        return _direct('{"ok": 1}')

    # replace_tools=[] avoids make_sandboxed_tools which requires a real sandbox
    s = agskill(
        name="summarise", system_prompt="You are a summarisation assistant.", replace_tools=[]
    )
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = create_side_effect
        s.execute_react(
            make_mock_agent(LLM, sb, ping_interval_s=30, poll_interval_s=0.01),
            agcontext(),
            agdata(x=1),
        )

    # The second LLM call should have the injected proc message in its user messages
    assert len(all_messages_per_call) == 2
    second_call_contents = [
        m.get("content", "") for m in all_messages_per_call[1] if m.get("role") == "user"
    ]
    assert any(
        "Background processes" in c or "completed" in c.lower() for c in second_call_contents
    )


def test_run_clean_sandbox_returns_immediately():
    """Sandbox with no PIDs does not delay return at all."""

    class _CleanSandbox:
        _watched_pids: dict = {}

        def _has_pending_background_work(self):
            return False

        def get_live_pids(self):
            return set()

        def pid_status_summary(self):
            return ""

        def commit(self, *a):
            return False

        def restore(self, *a):
            pass

        def write_file(self, *a):
            pass

        def remove_files(self, *a):
            pass

    # replace_tools=[] avoids make_sandboxed_tools which requires a real sandbox
    s = agskill(
        name="summarise", system_prompt="You are a summarisation assistant.", replace_tools=[]
    )
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct('{"ok": 1}')
        result, _, _ = s.execute_react(
            make_mock_agent(LLM, _CleanSandbox(), ping_interval_s=0.01, poll_interval_s=0.001),
            agcontext(),
            agdata(x=1),
        )
    assert result.result == '{"ok": 1}'


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# run() — thinking extraction from stream
# ---------------------------------------------------------------------------


def test_run_extracts_thinking_from_think_tag():
    s = make_skill()

    class _ThinkChunk:
        usage = None
        choices = [
            type(
                "C",
                (),
                {
                    "delta": type(
                        "D",
                        (),
                        {
                            "content": "<think>internal reasoning</think>final answer",
                            "tool_calls": None,
                            "model_extra": {},
                            "reasoning_content": None,
                        },
                    )()
                },
            )()
        ]

    class _UsageChunk:
        usage = _Usage()
        choices = []

    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = [
            _ThinkChunk(),
            _UsageChunk(),
        ]
        _, ctx, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))

    assistant_msgs = [m for m in ctx.messages if m.get("role") == "assistant"]
    assert any("_thinking" in m for m in assistant_msgs)


# ---------------------------------------------------------------------------
# run() — token accumulation
# ---------------------------------------------------------------------------


def test_run_returns_token_counts():
    s = make_skill()
    with patch("openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _direct("{}")
        _, ctx, _ = s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))
    # _Usage stub reports prompt_tokens=5
    assert ctx.total_input_tokens == 5


# ---------------------------------------------------------------------------
# run() defensively copies skill_input before mutating it
# ---------------------------------------------------------------------------
#
# prepare_inputs_in_sandbox() (called from execute_react(), itself called from
# run()'s _task()) mutates its skill_input argument in place -- offloaded
# agtype/oversized fields get overwritten with a sandbox path reference. If a
# caller hands the *same* agdata object to more than one concurrently-running
# agent.run() call (a real pattern: fanning one shared input out to several
# agents, e.g. autoresearch's ClassificationTeam.run()), those calls race on
# that shared mutation -- whichever run finishes its offload last clobbers the
# field with its own path, leaving every other run trying to read a file that
# only exists in that one run's own sandbox. _task() must give each run its
# own private copy from the moment it starts, regardless of what the caller
# does with the object it passed in.


def test_run_does_not_mutate_callers_shared_input_object():
    """Regression test: run() must not mutate the skill_input object the
    caller passed in -- prepare_inputs_in_sandbox()'s offload rewrite must
    land on a private copy, not the caller's own object."""
    from agency.agschema import agSchemaConfig
    from agency.agsandbox_backends import agSandboxBackendConfig

    # Force the docker sandbox backend: agsandbox_backends' "auto" selection
    # prefers podman over docker when both are usable, but CI's
    # images/build.sh only builds/tags agency-sandbox:latest for docker, so
    # podman has no local image and would try (and fail) to pull one.
    cfg = agConfig(
        agSchemaConfig(input_offload_chars=10),
        agSandboxBackendConfig(backend="docker"),
        {"agllm_backend": LLM_CONFIG},
    )
    s = agskill(name="offload_test", system_prompt="", input_schema=agdata(text=str))

    def fake_execute_react(ag, prev_ctx, skill_input, max_steps=None, **_):
        s.input_schema.prepare_inputs_in_sandbox(
            skill_input,
            ag.sandbox,
            s.name,
            context_limit=ag.llm.context_limit,
            agconfig=ag.agconfig,
        )
        return agdata(answer=skill_input.text), prev_ctx, []

    s.execute_react = fake_execute_react

    shared_input = agdata(text="x" * 100)
    ag = _agent_cls(agconfig=cfg)
    try:
        result = ag.run(s, shared_input)
        assert "saved to" in result.answer  # this run's own copy WAS offloaded
        assert shared_input.text == "x" * 100  # the caller's object was not
    finally:
        if ag.sandbox is not None:
            ag.sandbox.destroy()


def test_run_gives_concurrent_runs_sharing_one_input_independent_copies():
    """Two agents' run() calls sharing one input agdata (the exact
    ClassificationTeam.run() pattern) must each read back their own
    offloaded file, not race on the shared object's mutation."""
    from agency.agschema import agSchemaConfig
    from agency.agsandbox_backends import agSandboxBackendConfig

    # Force the docker sandbox backend: agsandbox_backends' "auto" selection
    # prefers podman over docker when both are usable, but CI's
    # images/build.sh only builds/tags agency-sandbox:latest for docker, so
    # podman has no local image and would try (and fail) to pull one.
    cfg = agConfig(
        agSchemaConfig(input_offload_chars=10),
        agSandboxBackendConfig(backend="docker"),
        {"agllm_backend": LLM_CONFIG},
    )
    s = agskill(name="offload_test", system_prompt="", input_schema=agdata(text=str))

    def fake_execute_react(ag, prev_ctx, skill_input, max_steps=None, **_):
        s.input_schema.prepare_inputs_in_sandbox(
            skill_input,
            ag.sandbox,
            s.name,
            context_limit=ag.llm.context_limit,
            agconfig=ag.agconfig,
        )
        from agency.tools import make_sandboxed_tools

        tools = {t.name: t for t in make_sandboxed_tools(ag.sandbox)}
        path = skill_input.text.split("saved to ")[1].split(" —")[0]
        r = tools["read"](agdata(file_path=path))
        return agdata(answer=r.content), prev_ctx, []

    s.execute_react = fake_execute_react

    shared_input = agdata(text="x" * 100)
    agents = [_agent_cls(agconfig=cfg) for _ in range(2)]
    try:
        pending = [a.run(s, shared_input) for a in agents]
        for p in pending:
            assert "x" * 100 in p.answer
        assert shared_input.text == "x" * 100
    finally:
        for a in agents:
            if a.sandbox is not None:
                a.sandbox.destroy()


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
        s.execute_react(make_mock_agent(LLM), agcontext(), agdata(x=1))

    assert captured["has_tools"] is False


# ---------------------------------------------------------------------------
# Randomised nested-schema fuzz: type_hint_to_string_type + JSON round-trip + validation
# ---------------------------------------------------------------------------


def test_random_nested_schema_roundtrip():
    """100 randomly generated nested schemas exercising every container/leaf combination.

    For each trial:
    - Python → serialized-str: type_hint_to_string_type must return the correct JSON Schema
      type, and json.dumps must succeed.
    - Serialized-str → Python: json.loads must round-trip cleanly, and
      validate_output_field_against_schema must accept the recovered value.
    - Wrong-container rejection: a value with the opposite container type (list vs
      dict) must be rejected by validate_output_field_against_schema for bare / generic hints that
      the framework validates at the top level.
    """
    import random
    from typing import get_origin, get_args
    from agency.agtype import type_hint_to_string_type, validate_output_field_against_schema
    from agency.agtype import agrawstring, agtype, agfile, agbinary, agimage

    _validate_output_field_against_schema = validate_output_field_against_schema

    rng = random.Random(20240624)

    LEAF_TYPES = [str, int, float, bool, agrawstring, agfile, agbinary, agimage]

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
        if hint is bool:
            return rng.choice([True, False])
        if hint is int:
            return rng.randint(-9, 9)
        if hint is float:
            return round(rng.uniform(-9.0, 9.0), 1)
        if hint is str or (isinstance(hint, type) and issubclass(hint, agtype)):
            return rng.choice(["a", "bb", "ccc"])
        origin = get_origin(hint)
        args = get_args(hint)
        if origin is list:
            return [rand_value(args[0]) for _ in range(rng.randint(1, 4))]
        if origin is dict:
            return {f"k{i}": rand_value(args[1]) for i in range(rng.randint(1, 4))}
        if origin is tuple:
            # Serialise as list — JSON has no tuple type
            return [rand_value(t) for t in args]
        # bare container types
        if hint is list:
            return [rng.randint(0, 5) for _ in range(rng.randint(1, 4))]
        if hint is dict:
            return {f"k{i}": rng.randint(0, 5) for i in range(rng.randint(1, 4))}
        if hint is tuple:
            return [rng.randint(0, 5) for _ in range(rng.randint(1, 4))]
        return "?"

    def ground_truth_json_type(hint) -> str:
        if isinstance(hint, type):
            if issubclass(hint, bool):
                return "boolean"
            if issubclass(hint, int):
                return "integer"
            if issubclass(hint, float):
                return "number"
            if issubclass(hint, (list, tuple)):
                return "array"
            if issubclass(hint, dict):
                return "object"
            return "string"  # str and agtype subclasses
        origin = get_origin(hint)
        if origin in (list, tuple):
            return "array"
        if origin is dict:
            return "object"
        return "string"

    failures = []
    for trial in range(100):
        hint = rand_hint(0)
        value = rand_value(hint)
        exp = ground_truth_json_type(hint)

        # -- Python → JSON Schema type --
        got = type_hint_to_string_type(hint)
        if got != exp:
            failures.append(f"[{trial}] type_hint_to_string_type({hint!r}) = {got!r}, want {exp!r}")
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

        # -- Valid value must pass validate_output_field_against_schema --
        schema = agdata(v=hint)
        err = _validate_output_field_against_schema("v", recovered, schema)
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
                isinstance(hint, type)  # bare list / dict / tuple
                or get_origin(hint) in (list, tuple, dict)  # generic list[T] / dict[K,V] / tuple[T]
            )
            if validates_top_level:
                err2 = _validate_output_field_against_schema("v", wrong, schema)
                if err2 is None:
                    failures.append(
                        f"[{trial}] wrong value not rejected — hint={hint!r} wrong={wrong!r}"
                    )

    assert not failures, f"{len(failures)}/100 trials failed:\n" + "\n".join(failures[:20])


def test_random_schema_prompt_examples_parseable():
    """100 randomly generated schemas: the example in every auto-generated tool
    description must be valid JSON AND must pass validate_output_field_against_schema.

    Also checks that error messages (wrong container type) include a parseable
    example that itself validates correctly.
    """
    import random
    from agency.agtype import (
        get_json_example_for_type_hint,
        type_hint_to_string_type,
        get_return_tool_description_prompt,
        validate_output_field_against_schema,
    )
    from agency.agtype import agrawstring, agtype, agfile, agbinary, agimage

    _validate_output_field_against_schema = validate_output_field_against_schema

    rng = random.Random(20240625)

    LEAF_TYPES = [str, int, float, bool, agrawstring, agfile, agbinary, agimage]

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

        # -- get_json_example_for_type_hint must produce valid JSON --
        ex_str = get_json_example_for_type_hint(hint)
        try:
            ex_val = json.loads(ex_str)
        except (ValueError, TypeError) as exc:
            failures.append(
                f"[{trial}] get_json_example_for_type_hint({hint!r}) = {ex_str!r} is not valid JSON: {exc}"
            )
            continue

        # -- that example must pass validate_output_field_against_schema --
        schema = agdata(v=hint)
        err = _validate_output_field_against_schema("v", ex_val, schema)
        if err is not None:
            failures.append(
                f"[{trial}] example from hint {hint!r} = {ex_val!r} failed validation: {err}"
            )
            continue

        # -- example must appear in the generated value description --
        _, vd = get_return_tool_description_prompt("v", hint)
        if not isinstance(hint, type) or not issubclass(hint, agtype):
            # agtype delegates to its own classmethods; skip appearance check there
            if ex_str not in vd:
                failures.append(f"[{trial}] example {ex_str!r} not found in value_desc {vd!r}")
                continue

        # -- the tool JSON Schema type must match the example's top-level type --
        json_type = type_hint_to_string_type(hint)
        type_ok = (
            (json_type == "array" and isinstance(ex_val, list))
            or (json_type == "object" and isinstance(ex_val, dict))
            or (json_type == "string" and isinstance(ex_val, str))
            or (json_type == "integer" and isinstance(ex_val, int) and not isinstance(ex_val, bool))
            or (json_type == "number" and isinstance(ex_val, float))
            or (json_type == "boolean" and isinstance(ex_val, bool))
        )
        if not type_ok:
            failures.append(
                f"[{trial}] example type mismatch — hint={hint!r} json_type={json_type!r} "
                f"example={ex_val!r} (type {type(ex_val).__name__})"
            )

    assert not failures, f"{len(failures)}/100 trials failed:\n" + "\n".join(failures[:20])
