"""Tests for agproxy_llm_adapters.py -- the reverse-direction wire
conversion functions (Anthropic Messages API / OpenAI Responses API <->
OpenAI chat.completions kwargs/response shapes). Pure functions, no FastAPI/
network involved -- fake OpenAI-shaped response/chunk objects stand in for
what a real `client.chat.completions.create()` call returns.
"""

from __future__ import annotations

import json

from agency.harness.agproxy_llm_adapters import (
    anthropic_messages_to_openai,
    anthropic_tools_to_openai,
    openai_response_to_anthropic_message,
    openai_chunks_to_anthropic_sse,
    responses_request_to_openai,
    responses_tools_to_openai,
    openai_response_to_responses_api,
    openai_chunks_to_responses_sse,
)


class _Fn:
    def __init__(self, name="", arguments=""):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id="", name="", arguments="", index=None):
        self.id = id
        self.function = _Fn(name, arguments)
        if index is not None:
            self.index = index


class _Message:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message=None, delta=None, finish_reason=None):
        self.message = message
        self.delta = delta
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Response:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Chunk:
    def __init__(self, choices=(), usage=None):
        self.choices = list(choices)
        self.usage = usage


# ---------------------------------------------------------------------------
# Anthropic Messages -> OpenAI kwargs
# ---------------------------------------------------------------------------


def test_anthropic_messages_to_openai_basic_text():
    body = {
        "model": "claude-x",
        "system": "be helpful",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "hello"}],
    }
    kwargs = anthropic_messages_to_openai(body)
    assert kwargs["model"] == "claude-x"
    assert kwargs["max_completion_tokens"] == 512
    assert "max_tokens" not in kwargs
    assert kwargs["messages"][0] == {"role": "system", "content": "be helpful"}
    assert kwargs["messages"][1] == {"role": "user", "content": "hello"}
    assert kwargs["stream"] is False


def test_anthropic_messages_to_openai_system_as_block_list():
    body = {
        "model": "m",
        "system": [{"type": "text", "text": "part one"}, {"type": "text", "text": " part two"}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    kwargs = anthropic_messages_to_openai(body)
    assert kwargs["messages"][0] == {"role": "system", "content": "part one part two"}


def test_anthropic_messages_to_openai_assistant_tool_use():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "let me check"},
                    {
                        "type": "tool_use",
                        "id": "tu1",
                        "name": "get_weather",
                        "input": {"city": "SF"},
                    },
                ],
            }
        ],
    }
    kwargs = anthropic_messages_to_openai(body)
    msg = kwargs["messages"][0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "let me check"
    assert msg["tool_calls"] == [
        {
            "id": "tu1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "SF"})},
        }
    ]


def test_anthropic_messages_to_openai_tool_result_becomes_tool_role():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "72F and sunny"}
                ],
            }
        ],
    }
    kwargs = anthropic_messages_to_openai(body)
    assert kwargs["messages"][0] == {
        "role": "tool",
        "tool_call_id": "tu1",
        "content": "72F and sunny",
    }


def test_anthropic_messages_to_openai_tool_result_with_block_list_content():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": [{"type": "text", "text": "result text"}],
                    }
                ],
            }
        ],
    }
    kwargs = anthropic_messages_to_openai(body)
    assert kwargs["messages"][0]["content"] == "result text"


def test_anthropic_messages_to_openai_tools_and_tool_choice():
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "get weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }
    kwargs = anthropic_messages_to_openai(body)
    assert kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_anthropic_tools_to_openai_empty():
    assert anthropic_tools_to_openai(None) is None
    assert anthropic_tools_to_openai([]) is None


# ---------------------------------------------------------------------------
# OpenAI response -> Anthropic Messages response
# ---------------------------------------------------------------------------


def test_openai_response_to_anthropic_message_text_only():
    resp = _Response(
        [_Choice(message=_Message(content="hi there"), finish_reason="stop")], usage=_Usage(5, 3)
    )
    out = openai_response_to_anthropic_message(resp, "claude-x", request_id="msg_1")
    assert out["id"] == "msg_1"
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["content"] == [{"type": "text", "text": "hi there"}]
    assert out["stop_reason"] == "end_turn"
    assert out["usage"] == {"input_tokens": 5, "output_tokens": 3}


def test_openai_response_to_anthropic_message_with_tool_call():
    tc = _ToolCall(id="call1", name="get_weather", arguments=json.dumps({"city": "SF"}))
    resp = _Response(
        [_Choice(message=_Message(content=None, tool_calls=[tc]), finish_reason="tool_calls")],
        usage=_Usage(10, 2),
    )
    out = openai_response_to_anthropic_message(resp, "claude-x")
    assert out["content"] == [
        {"type": "tool_use", "id": "call1", "name": "get_weather", "input": {"city": "SF"}}
    ]
    assert out["stop_reason"] == "tool_use"


def test_openai_response_to_anthropic_message_length_finish_reason():
    resp = _Response(
        [_Choice(message=_Message(content="cut off"), finish_reason="length")], usage=_Usage(1, 1)
    )
    out = openai_response_to_anthropic_message(resp, "m")
    assert out["stop_reason"] == "max_tokens"


# ---------------------------------------------------------------------------
# OpenAI streaming chunks -> Anthropic SSE
# ---------------------------------------------------------------------------


def _parse_sse(text):
    events = []
    for frame in text.split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        lines = frame.split("\n")
        event_line = next(l for l in lines if l.startswith("event: "))
        data_line = next(l for l in lines if l.startswith("data: "))
        events.append((event_line[len("event: ") :], json.loads(data_line[len("data: ") :])))
    return events


def test_openai_chunks_to_anthropic_sse_text_stream():
    chunks = [
        _Chunk(choices=[_Choice(delta=_Delta(content="Hel"))]),
        _Chunk(choices=[_Choice(delta=_Delta(content="lo"))]),
        _Chunk(choices=[_Choice(delta=_Delta(), finish_reason="stop")]),
        _Chunk(usage=_Usage(prompt_tokens=7, completion_tokens=2)),
    ]
    frames = "".join(openai_chunks_to_anthropic_sse(chunks, "claude-x", request_id="msg_1"))
    events = _parse_sse(frames)
    types = [t for t, _ in events]
    assert types[0] == "message_start"
    assert types[1] == "content_block_start"
    assert types[2] == "content_block_delta"
    assert types[3] == "content_block_delta"
    assert "content_block_stop" in types
    assert types[-2] == "message_delta"
    assert types[-1] == "message_stop"

    text_deltas = [d for t, d in events if t == "content_block_delta"]
    assert "".join(d["delta"]["text"] for d in text_deltas) == "Hello"

    message_delta = next(d for t, d in events if t == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "end_turn"
    assert message_delta["usage"]["output_tokens"] == 2


def test_openai_chunks_to_anthropic_sse_tool_call_stream():
    chunks = [
        _Chunk(choices=[_Choice(delta=_Delta(content="checking..."))]),
        _Chunk(
            choices=[
                _Choice(
                    delta=_Delta(
                        tool_calls=[
                            _ToolCall(id="call1", name="get_weather", arguments='{"ci', index=0)
                        ]
                    )
                )
            ]
        ),
        _Chunk(
            choices=[
                _Choice(
                    delta=_Delta(tool_calls=[_ToolCall(arguments='ty": "SF"}', index=0)]),
                    finish_reason="tool_calls",
                )
            ]
        ),
    ]
    frames = "".join(openai_chunks_to_anthropic_sse(chunks, "claude-x"))
    events = _parse_sse(frames)
    types = [t for t, _ in events]
    # text block must close before the tool_use block opens
    text_start_idx = types.index("content_block_start")
    text_stop_idx = types.index("content_block_stop")
    tool_start_idx = types.index("content_block_start", text_start_idx + 1)
    assert text_stop_idx < tool_start_idx

    tool_start_event = next(
        d
        for t, d in events
        if t == "content_block_start" and d["content_block"]["type"] == "tool_use"
    )
    assert tool_start_event["content_block"]["id"] == "call1"
    assert tool_start_event["content_block"]["name"] == "get_weather"

    json_deltas = [
        d["delta"]["partial_json"]
        for t, d in events
        if t == "content_block_delta" and d["delta"]["type"] == "input_json_delta"
    ]
    assert "".join(json_deltas) == '{"city": "SF"}'

    message_delta = next(d for t, d in events if t == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"


# ---------------------------------------------------------------------------
# Responses API request -> OpenAI kwargs
# ---------------------------------------------------------------------------


def test_responses_request_to_openai_plain_string_input():
    body = {"model": "m", "instructions": "be terse", "input": "what is 2+2?"}
    kwargs = responses_request_to_openai(body)
    assert kwargs["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "what is 2+2?"},
    ]


def test_responses_request_to_openai_structured_input_with_function_call_roundtrip():
    body = {
        "model": "m",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "what's the weather?"}]},
            {
                "type": "function_call",
                "call_id": "call1",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            },
            {"type": "function_call_output", "call_id": "call1", "output": "72F and sunny"},
        ],
    }
    kwargs = responses_request_to_openai(body)
    assert kwargs["messages"][0] == {"role": "user", "content": "what's the weather?"}
    assert kwargs["messages"][1]["role"] == "assistant"
    assert kwargs["messages"][1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert kwargs["messages"][2] == {
        "role": "tool",
        "tool_call_id": "call1",
        "content": "72F and sunny",
    }


def test_responses_request_to_openai_max_output_tokens_maps_to_max_completion_tokens():
    body = {"model": "m", "input": "hi", "max_output_tokens": 256}
    kwargs = responses_request_to_openai(body)
    assert kwargs["max_completion_tokens"] == 256
    assert "max_tokens" not in kwargs


def test_responses_tools_to_openai_flattens_to_nested():
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "d",
            "parameters": {"type": "object"},
        }
    ]
    converted = responses_tools_to_openai(tools)
    assert converted == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "d",
                "parameters": {"type": "object"},
            },
        }
    ]


# ---------------------------------------------------------------------------
# OpenAI response -> Responses API response
# ---------------------------------------------------------------------------


def test_openai_response_to_responses_api_text_only():
    resp = _Response([_Choice(message=_Message(content="the answer is 4"))], usage=_Usage(3, 4))
    out = openai_response_to_responses_api(resp, "m", request_id="resp_1")
    assert out["id"] == "resp_1"
    assert out["status"] == "completed"
    assert out["output"][0]["type"] == "message"
    assert out["output"][0]["content"] == [
        {"type": "output_text", "text": "the answer is 4", "annotations": []}
    ]
    assert out["usage"] == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}


def test_openai_response_to_responses_api_function_call():
    tc = _ToolCall(id="call1", name="get_weather", arguments='{"city": "SF"}')
    resp = _Response([_Choice(message=_Message(content=None, tool_calls=[tc]))], usage=_Usage(1, 1))
    out = openai_response_to_responses_api(resp, "m")
    assert out["output"][0]["type"] == "function_call"
    assert out["output"][0]["call_id"] == "call1"
    assert out["output"][0]["name"] == "get_weather"
    assert out["output"][0]["arguments"] == '{"city": "SF"}'


# ---------------------------------------------------------------------------
# OpenAI streaming chunks -> Responses API SSE
# ---------------------------------------------------------------------------


def test_openai_chunks_to_responses_sse_text_stream():
    chunks = [
        _Chunk(choices=[_Choice(delta=_Delta(content="Hi"))]),
        _Chunk(choices=[_Choice(delta=_Delta(), finish_reason="stop")]),
        _Chunk(usage=_Usage(prompt_tokens=2, completion_tokens=1)),
    ]
    frames = "".join(openai_chunks_to_responses_sse(chunks, "m", request_id="resp_1"))
    events = _parse_sse(frames)
    types = [t for t, _ in events]
    assert types[0] == "response.created"
    assert "response.output_item.added" in types
    assert "response.output_text.delta" in types
    assert types[-1] == "response.completed"

    completed = next(d for t, d in events if t == "response.completed")
    assert completed["response"]["status"] == "completed"
    assert completed["response"]["usage"]["input_tokens"] == 2
    assert completed["response"]["usage"]["output_tokens"] == 1


def test_openai_chunks_to_responses_sse_function_call_stream():
    chunks = [
        _Chunk(
            choices=[
                _Choice(
                    delta=_Delta(
                        tool_calls=[
                            _ToolCall(id="call1", name="get_weather", arguments='{"city"', index=0)
                        ]
                    )
                )
            ]
        ),
        _Chunk(
            choices=[
                _Choice(
                    delta=_Delta(tool_calls=[_ToolCall(arguments=': "SF"}', index=0)]),
                    finish_reason="tool_calls",
                )
            ]
        ),
    ]
    frames = "".join(openai_chunks_to_responses_sse(chunks, "m"))
    events = _parse_sse(frames)
    done_items = [d["item"] for t, d in events if t == "response.output_item.done"]
    fc_item = next(i for i in done_items if i["type"] == "function_call")
    assert fc_item["name"] == "get_weather"
    assert fc_item["arguments"] == '{"city": "SF"}'
    assert fc_item["call_id"] == "call1"
