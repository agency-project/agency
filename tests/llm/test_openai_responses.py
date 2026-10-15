"""Tests for the OpenAI Responses API backend (agency.llm.openai_responses):
provider selection, config validation, agency <-> Responses translation in
both directions, streaming, and a round trip through the host's stream
accumulator. No network access."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agency.llm.message_store import with_tool_schema
from agency.configs.agconfig import agconfig, llmconfig
from agency.engine.host_servers.llm_handler_server import LlmHandlerServer
from agency.llm.agllm import agllm
from agency.llm.openai import _OpenAICompatibleBackend
from agency.llm.anthropic import _AnthropicBackend
from agency.llm.openai_responses import ENCRYPTED_REASONING_TAG, _OpenAIResponsesBackend
from agency.llm.usage_tracker import LlmUsageTracker


def _cfg(**fields) -> agconfig:
    fields.setdefault("provider", "openai_responses")
    fields.setdefault("base_url", "https://api.openai.com/v1")
    fields.setdefault("model", "gpt-6-astra")
    return agconfig(llmconfig(**fields))


def _backend(**fields) -> _OpenAIResponsesBackend:
    return agllm.for_config(_cfg(**fields))


def _text_msg(role, text):
    return {"role": role, "blocks": [{"type": "text", "index": 0, "text": text}]}


def _tagged(encrypted_content: str) -> str:
    """The signature this backend stores for its own encrypted reasoning."""
    return f"{ENCRYPTED_REASONING_TAG}{encrypted_content}"


class _SdkObj(SimpleNamespace):
    """Stands in for an openai SDK pydantic object: attribute access plus
    model_dump()."""

    def model_dump(self):
        return {k: _dump(v) for k, v in self.__dict__.items()}


def _dump(value):
    if isinstance(value, _SdkObj):
        return value.model_dump()
    if isinstance(value, list):
        return [_dump(v) for v in value]
    return value


def _usage(input_tokens=10, output_tokens=5):
    return _SdkObj(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=None)


def _response(output, *, status="completed", usage=None, incomplete_reason=None):
    return _SdkObj(
        status=status,
        output=output,
        usage=usage if usage is not None else _usage(),
        incomplete_details=_SdkObj(reason=incomplete_reason) if incomplete_reason else None,
    )


def _reasoning_item(encrypted="enc-1", summary=()):
    return _SdkObj(
        type="reasoning",
        id="rs_real",
        summary=[_SdkObj(type="summary_text", text=t) for t in summary],
        encrypted_content=encrypted,
    )


def _function_call_item(call_id="call_1", name="get_secret_word", arguments="{}"):
    return _SdkObj(type="function_call", id="fc_1", call_id=call_id, name=name, arguments=arguments)


def _message_item(*parts):
    return _SdkObj(type="message", role="assistant", content=list(parts))


def _output_text(text):
    return _SdkObj(type="output_text", text=text, annotations=[])


# ---------------------------------------------------------------------------
# Selection and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna"])
def test_openai_responses_provider_selects_responses_backend_and_keeps_model_id(model):
    backend = _backend(model=model)
    assert type(backend) is _OpenAIResponsesBackend
    assert backend.model == model


def test_openai_provider_still_selects_chat_completions_backend():
    backend = agllm.for_config(_cfg(provider="openai"))
    assert type(backend) is _OpenAICompatibleBackend


def test_construction_without_base_url_raises():
    with pytest.raises(ValueError, match="base_url"):
        agllm.for_config(agconfig(llmconfig(provider="openai_responses", model="gpt-6-astra")))


@pytest.mark.parametrize("field,value", [("seed", 1), ("stop", ["x"]), ("frequency_penalty", 0.5)])
def test_chat_only_generation_params_are_rejected_not_silently_dropped(field, value):
    with pytest.raises(ValueError, match=field):
        _backend(**{field: value})


# ---------------------------------------------------------------------------
# Agency -> Responses request
# ---------------------------------------------------------------------------


def test_request_defaults_are_stateless_with_encrypted_reasoning():
    kwargs = _backend()._format_context_agency_to_backend({"messages": [_text_msg("user", "hi")]})
    assert kwargs == {
        "model": "gpt-6-astra",
        "input": [{"role": "user", "content": "hi"}],
        "max_output_tokens": 32000,
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }


def test_generation_params_map_to_responses_names():
    kwargs = _backend(
        reasoning_effort="high", max_completion_tokens=4096, temperature=1.0, top_p=0.5
    )._format_context_agency_to_backend({"messages": [_text_msg("user", "hi")]})
    assert kwargs["reasoning"] == {"effort": "high"}
    assert kwargs["max_output_tokens"] == 4096
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 0.5
    assert "max_completion_tokens" not in kwargs
    assert "reasoning_effort" not in kwargs


def test_deprecated_max_tokens_alias_still_sets_max_output_tokens():
    kwargs = _backend(max_tokens=1234)._format_context_agency_to_backend(
        {"messages": [_text_msg("user", "hi")]}
    )
    assert kwargs["max_output_tokens"] == 1234


def test_system_and_developer_messages_stay_in_place():
    messages = [
        _text_msg("system", "be terse"),
        _text_msg("developer", "answer plainly"),
        _text_msg("user", "hi"),
        _text_msg("system", "late operator note"),
    ]
    kwargs = _backend()._format_context_agency_to_backend({"messages": messages})
    assert kwargs["input"] == [
        {"role": "system", "content": "be terse"},
        {"role": "developer", "content": "answer plainly"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "late operator note"},
    ]
    assert "instructions" not in kwargs


def test_tool_loop_history_translates_to_responses_items():
    messages = [
        _text_msg("user", "call the tool"),
        {
            "role": "assistant",
            "blocks": [
                {
                    "type": "thinking",
                    "index": 0,
                    "text": "",
                    "signature": _tagged("enc-1"),
                    "id": "rs_x",
                },
                {"type": "text", "index": 1, "text": "checking"},
                {"type": "tool_use", "index": 2, "id": "call_1", "name": "f", "arguments": ""},
            ],
        },
        {
            "role": "tool",
            "blocks": [{"type": "tool_result", "index": 0, "tool_call_id": "call_1", "text": "42"}],
        },
    ]
    kwargs = _backend()._format_context_agency_to_backend({"messages": messages})
    assert kwargs["input"] == [
        {"role": "user", "content": "call the tool"},
        # No "id", even though the block carries one: a mismatched id is a
        # 400 under store=False, and omitting it is accepted.
        {"type": "reasoning", "summary": [], "encrypted_content": "enc-1"},
        {"role": "assistant", "content": "checking"},
        {"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "42"},
    ]


def test_reasoning_summary_text_is_replayed_as_summary():
    messages = [
        {
            "role": "assistant",
            "blocks": [
                {"type": "thinking", "index": 0, "text": "plan", "signature": _tagged("enc")}
            ],
        }
    ]
    kwargs = _backend()._format_context_agency_to_backend({"messages": messages})
    assert kwargs["input"] == [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "plan"}],
            "encrypted_content": "enc",
        }
    ]


def test_thinking_without_encrypted_content_and_foreign_blocks_are_dropped():
    messages = [
        {
            "role": "assistant",
            "blocks": [
                {"type": "thinking", "index": 0, "text": "vllm trace", "signature": ""},
                {"type": "anthropic_server_tool_use", "index": 1, "data": {"id": "x"}},
                {"type": "openai_chatcompletions_refusal", "index": 2, "data": "no"},
                {"type": "text", "index": 3, "text": "done"},
            ],
        }
    ]
    kwargs = _backend()._format_context_agency_to_backend({"messages": messages})
    assert kwargs["input"] == [{"role": "assistant", "content": "done"}]


def test_chat_completions_tools_are_flattened_and_strict_only_when_given():
    tools = [
        {
            "type": "function",
            "function": {"name": "a", "description": "A", "parameters": {"type": "object"}},
        },
        {"type": "function", "function": {"name": "b", "strict": True}},
    ]
    kwargs = _backend()._format_context_agency_to_backend(
        {"messages": with_tool_schema([_text_msg("user", "hi")], tools)}
    )
    assert kwargs["tools"] == [
        {"type": "function", "name": "a", "description": "A", "parameters": {"type": "object"}},
        {
            "type": "function",
            "name": "b",
            "description": "",
            "parameters": {"type": "object", "properties": {}},
            "strict": True,
        },
    ]


@pytest.mark.parametrize(
    "tool_choice,expected",
    [
        ("auto", "auto"),
        ("required", "required"),
        ("none", "none"),
        ({"type": "function", "function": {"name": "f"}}, {"type": "function", "name": "f"}),
    ],
)
def test_tool_choice_translated_when_tools_present(tool_choice, expected):
    kwargs = _backend()._format_context_agency_to_backend(
        {
            "messages": with_tool_schema(
                [_text_msg("user", "hi")],
                [{"type": "function", "function": {"name": "f"}}],
                tool_choice,
            )
        }
    )
    assert kwargs["tool_choice"] == expected


def test_tool_choice_dropped_without_tools_or_when_foreign():
    backend = _backend()
    no_tools = backend._format_context_agency_to_backend(
        {"messages": with_tool_schema([_text_msg("user", "hi")], tool_choice="auto")}
    )
    assert "tool_choice" not in no_tools
    foreign = backend._format_context_agency_to_backend(
        {
            "messages": with_tool_schema(
                [_text_msg("user", "hi")],
                [{"type": "function", "function": {"name": "f"}}],
                {"type": "openai_responses_web_search", "data": {}},
            )
        }
    )
    assert "tool_choice" not in foreign


# ---------------------------------------------------------------------------
# Responses -> Agency (non-streaming)
# ---------------------------------------------------------------------------


def test_response_output_items_become_agency_blocks():
    raw = _response(
        [
            _reasoning_item(encrypted="enc-1", summary=("step one", "step two")),
            _message_item(_output_text("hello"), _SdkObj(type="refusal", refusal="no")),
            _function_call_item(arguments='{"q": 1}'),
            _SdkObj(type="web_search_call", id="ws_1", status="completed"),
        ]
    )
    result = _backend()._format_context_backend_to_agency(raw)
    blocks = result["message"]["blocks"]
    assert blocks[:5] == [
        {
            "type": "thinking",
            "index": 0,
            "text": "step one\n\nstep two",
            "signature": _tagged("enc-1"),
        },
        {"type": "text", "index": 1, "text": "hello"},
        {
            "type": "openai_responses_refusal",
            "index": 2,
            "data": {"type": "refusal", "refusal": "no"},
        },
        {
            "type": "tool_use",
            "index": 3,
            "id": "call_1",
            "name": "get_secret_word",
            "arguments": '{"q": 1}',
        },
        {
            "type": "openai_responses_web_search_call",
            "index": 4,
            "data": {"type": "web_search_call", "id": "ws_1", "status": "completed"},
        },
    ]
    assert blocks[5]["type"] == "metadata"
    assert result["stop_reason"] == "tool_calls"
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


@pytest.mark.parametrize(
    "output,status,reason,expected",
    [
        ([_message_item(_output_text("hi"))], "completed", None, "stop"),
        ([_function_call_item()], "completed", None, "tool_calls"),
        ([], "incomplete", "max_output_tokens", "length"),
        ([], "incomplete", "content_filter", "content_filter"),
    ],
)
def test_stop_reason_uses_chat_completions_vocabulary(output, status, reason, expected):
    raw = _response(output, status=status, incomplete_reason=reason)
    assert _backend()._format_context_backend_to_agency(raw)["stop_reason"] == expected


def test_call_backend_uses_responses_create_and_closes_client():
    calls = []
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kw: calls.append(kw) or "raw"),
        closed=False,
    )
    client.close = lambda: setattr(client, "closed", True)
    backend = _backend()
    backend.make_client = lambda timeout: client
    assert backend._call_backend({"model": "m", "input": []}) == "raw"
    assert calls == [{"model": "m", "input": []}]
    assert client.closed is True


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _ev(event_type, **fields):
    return SimpleNamespace(type=event_type, **fields)


def _tool_turn_events():
    """What a GPT-6 tool-calling turn streams: reasoning (summary optional),
    then a function call, then the completed response."""
    reasoning = _reasoning_item(encrypted="enc-1", summary=("a", "b"))
    call = _function_call_item(arguments='{"city": "Paris"}')
    return [
        _ev("response.created", response=None),
        _ev("response.output_item.added", output_index=0, item=reasoning),
        _ev("response.reasoning_summary_part.added", output_index=0, summary_index=0),
        _ev("response.reasoning_summary_text.delta", output_index=0, summary_index=0, delta="a"),
        _ev("response.reasoning_summary_part.added", output_index=0, summary_index=1),
        _ev("response.reasoning_summary_text.delta", output_index=0, summary_index=1, delta="b"),
        _ev("response.output_item.done", output_index=0, item=reasoning),
        _ev("response.output_item.added", output_index=1, item=call),
        _ev("response.function_call_arguments.delta", output_index=1, delta='{"city"'),
        _ev("response.output_item.done", output_index=1, item=call),
        _ev("response.completed", response=_response([reasoning, call], usage=_usage(20, 7))),
    ]


def test_stream_emits_thinking_signature_and_one_complete_tool_call():
    deltas = list(_backend()._format_stream_to_agency(iter(_tool_turn_events())))
    content = [d for d in deltas if d["type"] == "block_delta" and d["block_type"] != "metadata"]
    assert content == [
        {"type": "block_delta", "index": 0, "block_type": "thinking", "text": "a"},
        {"type": "block_delta", "index": 0, "block_type": "thinking", "text": "\n\n"},
        {"type": "block_delta", "index": 0, "block_type": "thinking", "text": "b"},
        {
            "type": "block_delta",
            "index": 0,
            "block_type": "thinking",
            "signature": _tagged("enc-1"),
        },
        {
            "type": "block_delta",
            "index": 1,
            "block_type": "tool_use",
            "id": "call_1",
            "name": "get_secret_word",
            "arguments": '{"city": "Paris"}',
        },
    ]
    assert deltas[-1] == {
        "type": "usage",
        "usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
        "stop_reason": "tool_calls",
    }


def test_stream_text_and_refusal_parts_get_separate_ordered_indexes():
    message = _message_item(_output_text("hel lo"), _SdkObj(type="refusal", refusal="no"))
    events = [
        _ev("response.output_text.delta", output_index=0, content_index=0, delta="hel"),
        _ev("response.output_text.delta", output_index=0, content_index=0, delta=" lo"),
        _ev("response.output_item.done", output_index=0, item=message),
        _ev("response.completed", response=_response([message])),
    ]
    deltas = list(_backend()._format_stream_to_agency(iter(events)))
    content = [d for d in deltas if d["type"] == "block_delta" and d["block_type"] != "metadata"]
    assert [(d["index"], d["block_type"]) for d in content] == [
        (0, "text"),
        (0, "text"),
        (1, "openai_responses_refusal"),
    ]
    assert deltas[-1]["stop_reason"] == "stop"


def test_stream_incomplete_response_reports_length():
    events = [
        _ev("response.output_text.delta", output_index=0, content_index=0, delta="cut"),
        _ev(
            "response.incomplete",
            response=_response([], status="incomplete", incomplete_reason="max_output_tokens"),
        ),
    ]
    deltas = list(_backend()._format_stream_to_agency(iter(events)))
    assert deltas[-1]["stop_reason"] == "length"


@pytest.mark.parametrize(
    "event",
    [
        _ev("response.failed", response=SimpleNamespace(error=SimpleNamespace(message="boom"))),
        _ev("error", message="boom"),
    ],
)
def test_stream_failure_events_raise(event):
    with pytest.raises(RuntimeError, match="boom"):
        list(_backend()._format_stream_to_agency(iter([event])))


class _NullDataLogger:
    def record_event(self, *args, **kwargs):
        pass

    def record_stream_delta(self, *args, **kwargs):
        pass

    def record_llm_exchange(self, *args, **kwargs):
        pass


def test_streamed_turn_accumulates_on_host_and_replays_as_next_request():
    """The host's generic stream accumulator must rebuild a thinking block
    that carries the encrypted reasoning, and a complete tool call -- and
    that assembled message must translate back into valid next-turn input."""
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return iter(_tool_turn_events())

    client = SimpleNamespace(responses=SimpleNamespace(create=create), close=lambda: None)
    server = LlmHandlerServer(_cfg(), _NullDataLogger(), LlmUsageTracker())
    server._backend.make_client = lambda timeout: client

    handle = server.start_stream({"messages": [_text_msg("user", "weather?")]})
    item = handle.first()
    while item["type"] not in ("done", "error"):
        item = handle.first()
    handle._thread.join(timeout=2.0)
    assert item["type"] == "done", item
    assert requests[0]["stream"] is True
    assistant = item["message"]
    thinking, tool_use = [b for b in assistant["blocks"] if b["type"] != "metadata"]
    assert (thinking["type"], thinking["text"], thinking["signature"]) == (
        "thinking",
        "a\n\nb",
        _tagged("enc-1"),
    )
    assert (tool_use["id"], tool_use["name"], tool_use["arguments"]) == (
        "call_1",
        "get_secret_word",
        '{"city": "Paris"}',
    )

    tool_result = {
        "role": "tool",
        "blocks": [{"type": "tool_result", "index": 0, "tool_call_id": "call_1", "text": "Sunny"}],
    }
    next_request = server._backend._format_context_agency_to_backend(
        {"messages": [_text_msg("user", "weather?"), assistant, tool_result]}
    )
    assert next_request["input"] == [
        {"role": "user", "content": "weather?"},
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "a\n\nb"}],
            "encrypted_content": "enc-1",
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_secret_word",
            "arguments": '{"city": "Paris"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "Sunny"},
    ]


# ---------------------------------------------------------------------------
# Cross-provider history: opaque reasoning only goes back to its producer
# ---------------------------------------------------------------------------

_ANTHROPIC_SIGNATURE = "EqQBCkgIBxABGAIiQJ+anthropic/base64=="


def _mixed_provider_history() -> "list[dict]":
    """A conversation whose first tool turn ran on Claude and second on
    provider="openai_responses", as after an agent's change_config()."""
    return [
        _text_msg("user", "look up a and b"),
        {
            "role": "assistant",
            "blocks": [
                {"type": "thinking", "index": 0, "text": "", "signature": _ANTHROPIC_SIGNATURE},
                {"type": "text", "index": 1, "text": "claude text"},
                {"type": "tool_use", "index": 2, "id": "toolu_1", "name": "f", "arguments": "{}"},
            ],
        },
        {
            "role": "tool",
            "blocks": [{"type": "tool_result", "index": 0, "tool_call_id": "toolu_1", "text": "A"}],
        },
        {
            "role": "assistant",
            "blocks": [
                {"type": "thinking", "index": 0, "text": "", "signature": _tagged("enc-2")},
                {"type": "text", "index": 1, "text": "gpt text"},
                {"type": "tool_use", "index": 2, "id": "call_2", "name": "f", "arguments": "{}"},
            ],
        },
        {
            "role": "tool",
            "blocks": [{"type": "tool_result", "index": 0, "tool_call_id": "call_2", "text": "B"}],
        },
    ]


def test_anthropic_signature_never_becomes_openai_encrypted_content():
    kwargs = _backend()._format_context_agency_to_backend({"messages": _mixed_provider_history()})
    reasoning_items = [i for i in kwargs["input"] if i.get("type") == "reasoning"]
    assert reasoning_items == [{"type": "reasoning", "summary": [], "encrypted_content": "enc-2"}]
    # Everything that isn't opaque reasoning state survives the switch.
    assert kwargs["input"][1:] == [
        {"role": "assistant", "content": "claude text"},
        {"type": "function_call", "call_id": "toolu_1", "name": "f", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "toolu_1", "output": "A"},
        {"type": "reasoning", "summary": [], "encrypted_content": "enc-2"},
        {"role": "assistant", "content": "gpt text"},
        {"type": "function_call", "call_id": "call_2", "name": "f", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_2", "output": "B"},
    ]


def test_openai_encrypted_reasoning_never_becomes_anthropic_signature():
    backend = _AnthropicBackend(agconfig(llmconfig(provider="anthropic", model="claude-opus-5-5")))
    kwargs = backend._format_context_agency_to_backend({"messages": _mixed_provider_history()})
    first_turn, second_turn = kwargs["messages"][1]["content"], kwargs["messages"][3]["content"]
    # Claude's own signature is still replayed to Claude...
    assert first_turn[0] == {"type": "thinking", "thinking": "", "signature": _ANTHROPIC_SIGNATURE}
    # ...but the OpenAI turn keeps only its text and tool call.
    assert [block["type"] for block in second_turn] == ["text", "tool_use"]
    assert all(ENCRYPTED_REASONING_TAG not in str(message) for message in kwargs["messages"])
    tool_results = [
        block["tool_use_id"]
        for message in kwargs["messages"]
        if isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert tool_results == ["toolu_1", "call_2"]


# ---------------------------------------------------------------------------
# Responses that are not a finished turn, and state that must survive a turn.
#
# Shapes are those recorded from GPT-6.1 Sol through LiteLLM 1.104.1 to
# Bedrock's /openai/v1/responses (analysis/sol_integration/data).
# ---------------------------------------------------------------------------

from agency.llm.agllm import ProviderResponseError, StreamInterruptedError  # noqa: E402

CUT_ARGUMENTS = '{"cmd":"cat <<\'EOF\' > notes.txt\\n1. Linux scheduling determines which'


def test_stream_that_ends_without_a_terminal_event_is_interrupted():
    events = [
        _ev("response.created", response=_response([], status="in_progress")),
        _ev("response.output_text.delta", output_index=0, content_index=0, delta="par"),
    ]
    with pytest.raises(StreamInterruptedError, match="without a terminal event after 2"):
        list(_backend()._format_stream_to_agency(iter(events)))
    with pytest.raises(StreamInterruptedError, match="after 0 event"):
        list(_backend()._format_stream_to_agency(iter([])))


def test_interrupted_stream_check_can_be_turned_off():
    events = [_ev("response.output_text.delta", output_index=0, content_index=0, delta="par")]
    deltas = list(_backend(strict_stream_termination=False)._format_stream_to_agency(iter(events)))
    assert [d["text"] for d in deltas] == ["par"]


def test_truncated_function_call_arrives_as_a_tool_call_with_stop_reason_length():
    """Bedrock sends output_item.done for the cut-off call and only then
    response.incomplete. The backend must carry both facts: the fragment
    (so it is on record) and the stop reason (so nothing runs it)."""
    call = _SdkObj(
        type="function_call",
        id="fc_1",
        call_id="call_cut",
        name="exec_command",
        arguments=CUT_ARGUMENTS,
        status="incomplete",
    )
    events = [
        _ev("response.output_item.added", output_index=0, item=call),
        _ev("response.function_call_arguments.delta", output_index=0, delta=CUT_ARGUMENTS),
        _ev("response.output_item.done", output_index=0, item=call),
        _ev(
            "response.incomplete",
            response=_response(
                [call],
                status="incomplete",
                incomplete_reason="max_output_tokens",
                usage=_usage(101, 64),
            ),
        ),
    ]
    deltas = list(_backend()._format_stream_to_agency(iter(events)))
    tool = next(d for d in deltas if d.get("block_type") == "tool_use")
    assert (tool["id"], tool["arguments"]) == ("call_cut", CUT_ARGUMENTS)
    assert deltas[-1]["stop_reason"] == "length"
    assert deltas[-1]["usage"]["completion_tokens"] == 64


@pytest.mark.parametrize(
    "code,retryable,transient,status",
    [
        ("server_error", True, True, 502),
        (None, True, True, 502),
        ("validation_error", False, False, 400),
        ("context_length_exceeded", False, False, 400),
        ("rate_limit_exceeded", False, True, 429),
    ],
)
def test_failed_response_keeps_the_providers_code_and_says_whether_to_ask_again(
    code, retryable, transient, status
):
    failed = SimpleNamespace(id="resp_1", error=SimpleNamespace(code=code, message="boom"))
    with pytest.raises(ProviderResponseError, match="boom") as raised:
        list(_backend()._format_stream_to_agency(iter([_ev("response.failed", response=failed)])))
    error = raised.value
    assert (error.retryable, error.transient, error.status_code) == (retryable, transient, status)
    assert error.details["code"] == code and error.details["response_id"] == "resp_1"


def test_non_streaming_failed_response_raises_instead_of_returning_an_empty_turn():
    failed = _SdkObj(
        status="failed",
        output=[],
        usage=None,
        id="resp_2",
        error=_SdkObj(code="server_error", message="upstream fell over"),
    )
    with pytest.raises(ProviderResponseError, match="upstream fell over"):
        _backend()._format_context_backend_to_agency(failed)


def test_usage_reports_cached_cache_write_and_reasoning_tokens():
    usage = _SdkObj(
        input_tokens=2344,
        output_tokens=50,
        total_tokens=2394,
        input_tokens_details=_SdkObj(cached_tokens=2342, cache_write_tokens=0),
        output_tokens_details=_SdkObj(reasoning_tokens=45),
    )
    result = _backend()._format_context_backend_to_agency(_response([], usage=usage))
    assert result["usage"] == {
        "prompt_tokens": 2344,
        "completion_tokens": 50,
        "total_tokens": 2394,
        "cached_tokens": 2342,
        "cache_write_tokens": 0,
        "reasoning_tokens": 45,
    }


def test_usage_without_details_gains_no_invented_zeroes():
    result = _backend()._format_context_backend_to_agency(_response([], usage=_usage(10, 5)))
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_message_phase_survives_response_and_replay():
    message = _message_item(_output_text("Checking."))
    message.phase = "commentary"
    blocks = _backend()._format_context_backend_to_agency(_response([message]))["message"]["blocks"]
    assert blocks[0] == {"type": "text", "index": 0, "text": "Checking.", "phase": "commentary"}

    events = [
        _ev("response.output_text.delta", output_index=0, content_index=0, delta="Checking."),
        _ev("response.output_item.done", output_index=0, item=message),
        _ev("response.completed", response=_response([message])),
    ]
    deltas = list(_backend()._format_stream_to_agency(iter(events)))
    assert {
        "type": "block_delta",
        "index": 0,
        "block_type": "text",
        "phase": "commentary",
    } in deltas

    request = _backend()._format_context_agency_to_backend(
        {"messages": [{"role": "assistant", "blocks": [blocks[0]]}]}
    )
    assert request["input"] == [
        {
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "Checking."}],
        }
    ]


def test_reasoning_then_message_then_tool_call_replay_in_the_models_own_order():
    """A reasoning item is only valid directly before what it led to, and its
    encrypted content is sent back without an id (a mismatched id is a 400)."""
    history = [
        _text_msg("user", "how many cores?"),
        {
            "role": "assistant",
            "blocks": [
                {"type": "thinking", "index": 0, "text": "", "signature": _tagged("rsn_1")},
                {"type": "text", "index": 1, "text": "Checking.", "phase": "commentary"},
                {
                    "type": "tool_use",
                    "index": 2,
                    "id": "call_abc",
                    "name": "exec_command",
                    "arguments": '{"cmd":"nproc"}',
                },
            ],
        },
        {
            "role": "tool",
            "blocks": [
                {"type": "tool_result", "index": 0, "tool_call_id": "call_abc", "text": "6\n"}
            ],
        },
    ]
    items = _backend()._format_context_agency_to_backend({"messages": history})["input"]
    assert [item.get("type", "message") for item in items] == [
        "message",
        "reasoning",
        "message",
        "function_call",
        "function_call_output",
    ]
    assert items[1] == {"type": "reasoning", "summary": [], "encrypted_content": "rsn_1"}
    assert items[3]["call_id"] == items[4]["call_id"] == "call_abc"
    assert items[4]["output"] == "6\n"


def test_reasoning_effort_and_summary_are_sent_as_configured_for_the_model():
    request = _backend(model="gpt-6.1-sol", reasoning_effort="medium")
    assert request._format_context_agency_to_backend({"messages": []})["reasoning"] == {
        "effort": "medium"
    }
    request = _backend(model="gpt-6.1-sol", reasoning_effort="medium", reasoning_summary="auto")
    assert request._format_context_agency_to_backend({"messages": []})["reasoning"] == {
        "effort": "medium",
        "summary": "auto",
    }
    assert "reasoning" not in _backend()._format_context_agency_to_backend({"messages": []})


def test_dropping_another_providers_reasoning_state_is_said_out_loud(capsys):
    from agency.llm.agllm import _WARNED_ONCE

    _WARNED_ONCE.discard(("responses_drops_foreign_reasoning",))
    history = [
        {
            "role": "assistant",
            "blocks": [
                {"type": "thinking", "index": 0, "text": "", "signature": "anthropic-base64=="},
                {"type": "text", "index": 1, "text": "ok"},
            ],
        }
    ]
    items = _backend()._format_context_agency_to_backend({"messages": history})["input"]
    assert items == [{"role": "assistant", "content": "ok"}]
    assert "cannot be replayed on /v1/responses" in capsys.readouterr().out


@pytest.mark.parametrize("reasoning_text", ["", "summary"])
def test_chat_completions_backend_says_when_it_cannot_send_reasoning_state(capsys, reasoning_text):
    from agency.llm.agllm import _WARNED_ONCE

    _WARNED_ONCE.discard(("chat_drops_reasoning_state", "openai"))
    backend = agllm.for_config(
        agconfig(llmconfig(provider="openai", base_url="http://x/v1", model="gpt-6-luna"))
    )
    history = [
        {
            "role": "assistant",
            "blocks": [
                {
                    "type": "thinking",
                    "index": 0,
                    "text": reasoning_text,
                    "signature": _tagged("rsn_1"),
                },
                {"type": "text", "index": 1, "text": "ok"},
            ],
        }
    ]
    wire = backend.build_kwargs(history)["messages"]
    assert wire == [{"role": "assistant", "content": "ok"}]
    assert "provider='openai_responses'" in capsys.readouterr().out


def test_stream_carries_upstream_call_ids_and_attaches_them_to_a_failure():
    class Stream:
        response = SimpleNamespace(
            headers={
                "x-litellm-call-id": "call-7",
                "llm_provider-x-amzn-requestid": "amzn-1",
                "authorization": "Bearer secret",
                "x-litellm-key-spend": "17.0",
            }
        )

        def __init__(self, events):
            self._events = events

        def __iter__(self):
            return iter(self._events)

    message = _message_item(_output_text("hi"))
    good = [
        _ev("response.output_text.delta", output_index=0, content_index=0, delta="hi"),
        _ev("response.completed", response=_response([message])),
    ]
    backend = _backend()
    client = SimpleNamespace(close=lambda: None)
    backend._call_backend_stream = lambda request, on_client=None: (Stream(good), client)
    items = list(backend.dispatch_stream({"messages": []}))
    ids = {"x-litellm-call-id": "call-7", "llm_provider-x-amzn-requestid": "amzn-1"}
    # First item, so the last metadata fragment is still the one with usage.
    assert items[0]["block_type"] == "metadata" and items[0]["data"] == {"response_ids": ids}
    assert "secret" not in repr(items)

    backend._call_backend_stream = lambda request, on_client=None: (Stream(good[:1]), client)
    with pytest.raises(StreamInterruptedError) as raised:
        list(backend.dispatch_stream({"messages": []}))
    assert raised.value.response_ids == ids

    backend._call_backend_stream = lambda request, on_client=None: (Stream([]), client)
    with pytest.raises(StreamInterruptedError) as raised:
        list(backend.dispatch_stream({"messages": []}))
    assert raised.value.response_ids == ids
