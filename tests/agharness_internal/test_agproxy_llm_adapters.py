"""Tests for agproxy_llm_adapters.py -- the reverse-direction wire
conversion functions (Anthropic Messages API / OpenAI Responses API <->
OpenAI chat.completions kwargs/response shapes). Pure functions, no FastAPI/
network involved -- fake OpenAI-shaped response/chunk objects stand in for
what a real `client.chat.completions.create()` call returns.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agency.agharness_internal.agproxy_llm_adapters import (
    UnsupportedResponsesRequest,
    anthropic_messages_to_openai,
    anthropic_tools_to_openai,
    openai_response_to_anthropic_message,
    openai_chunks_to_anthropic_sse,
    responses_request_to_openai,
    responses_tools_to_openai,
    openai_response_to_responses_api,
    openai_chunks_to_responses_sse,
)


_FIXTURES = Path(__file__).parent / "fixtures" / "codex"


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
    def __init__(
        self, prompt_tokens=0, completion_tokens=0, cached_tokens=None, reasoning_tokens=None
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        if cached_tokens is not None:
            self.prompt_tokens_details = {"cached_tokens": cached_tokens}
        if reasoning_tokens is not None:
            self.completion_tokens_details = {"reasoning_tokens": reasoning_tokens}


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


def test_responses_request_to_openai_coalesces_parallel_function_calls():
    body = {
        "model": "m",
        "input": [
            {"role": "user", "content": "inspect both files"},
            {
                "type": "function_call",
                "call_id": "call1",
                "name": "read_file",
                "arguments": '{"path":"a.py"}',
            },
            {
                "type": "function_call",
                "call_id": "call2",
                "name": "read_file",
                "arguments": '{"path":"b.py"}',
            },
            {"type": "function_call_output", "call_id": "call1", "output": "a contents"},
            {"type": "function_call_output", "call_id": "call2", "output": "b contents"},
        ],
    }

    kwargs = responses_request_to_openai(body)

    assert kwargs["messages"] == [
        {"role": "user", "content": "inspect both files"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                },
                {
                    "id": "call2",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"b.py"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call1", "content": "a contents"},
        {"role": "tool", "tool_call_id": "call2", "content": "b contents"},
    ]


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


def test_responses_request_to_openai_matches_captured_codex_0_140_function_subset():
    body = json.loads((_FIXTURES / "responses_request_function_tools_v0_140.json").read_text())
    translation_warnings = []

    kwargs = responses_request_to_openai(body, warning_handler=translation_warnings.append)

    assert kwargs["messages"] == [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "system", "content": "Honor the workspace permissions."},
        {"role": "user", "content": "Run pwd once, then reply done."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-redacted",
                    "type": "function",
                    "function": {"name": "exec_command", "arguments": '{"cmd":"pwd"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-redacted", "content": "/workspace\n"},
    ]
    assert kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "description": "Runs a command and returns its output.",
                "parameters": body["tools"][0]["parameters"],
                "strict": False,
            },
        }
    ]
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is True
    assert kwargs["reasoning_effort"] == "medium"
    assert kwargs["verbosity"] == "low"
    assert kwargs["store"] is False
    assert kwargs["prompt_cache_key"] == "thread-redacted"
    assert len(translation_warnings) == 1
    assert "opaque-reasoning continuity is unavailable" in translation_warnings[0]
    assert kwargs["stream"] is True


def test_responses_request_to_openai_loads_tool_search_namespace_results():
    tool_name_map = {}
    loaded_namespace = {
        "type": "namespace",
        "name": "mcp__agency",
        "description": "Tools in the mcp__agency namespace.",
        "tools": [
            {
                "type": "function",
                "name": "submit_output",
                "description": "Submit one output field.",
                "strict": False,
                "defer_loading": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["field", "value"],
                },
            }
        ],
    }
    body = {
        "model": "m",
        "input": [
            {"type": "message", "role": "user", "content": "Return structured output."},
            {
                "type": "tool_search_call",
                "call_id": "search-1",
                "execution": "client",
                "arguments": {"query": "submit_output", "limit": 1},
            },
            {
                "type": "tool_search_output",
                "call_id": "search-1",
                "status": "completed",
                "execution": "client",
                "tools": [loaded_namespace],
            },
            {
                "type": "function_call",
                "call_id": "submit-1",
                "namespace": "mcp__agency",
                "name": "submit_output",
                "arguments": '{"field":"status","value":"done"}',
            },
            {
                "type": "function_call_output",
                "call_id": "submit-1",
                "output": "accepted",
            },
        ],
        "tools": [
            {
                "type": "tool_search",
                "execution": "client",
                "description": "Search available tools.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
        "tool_choice": "auto",
    }

    kwargs = responses_request_to_openai(body, tool_name_map=tool_name_map)

    assert [tool["function"]["name"] for tool in kwargs["tools"]] == [
        "tool_search",
        "mcp__agency__submit_output",
    ]
    assert kwargs["messages"][1]["tool_calls"][0]["function"] == {
        "name": "tool_search",
        "arguments": '{"query":"submit_output","limit":1}',
    }
    assert json.loads(kwargs["messages"][2]["content"])["tools"] == [loaded_namespace]
    assert kwargs["messages"][3]["tool_calls"][0]["function"] == {
        "name": "mcp__agency__submit_output",
        "arguments": '{"field":"status","value":"done"}',
    }
    assert tool_name_map["tool_search"]["response_type"] == "tool_search_call"
    assert tool_name_map["mcp__agency__submit_output"]["namespace"] == "mcp__agency"


def test_responses_request_to_openai_uses_latest_refreshed_tool_search_definition():
    def search_output(call_id, description, schema_type):
        return {
            "type": "tool_search_output",
            "call_id": call_id,
            "status": "completed",
            "execution": "client",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__agency",
                    "tools": [
                        {
                            "type": "function",
                            "name": "submit_output",
                            "description": description,
                            "defer_loading": True,
                            "parameters": {
                                "type": "object",
                                "properties": {"value": {"type": schema_type}},
                            },
                        }
                    ],
                }
            ],
        }

    body = {
        "model": "m",
        "input": [
            {"type": "message", "role": "user", "content": "Submit output."},
            {
                "type": "tool_search_call",
                "call_id": "search-old",
                "execution": "client",
                "arguments": {"query": "submit"},
            },
            search_output("search-old", "Old description", "string"),
            {
                "type": "tool_search_call",
                "call_id": "search-new",
                "execution": "client",
                "arguments": {"query": "submit"},
            },
            search_output("search-new", "New description", "number"),
        ],
        "tools": [
            {
                "type": "tool_search",
                "execution": "client",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    }

    kwargs = responses_request_to_openai(body)

    names = [tool["function"]["name"] for tool in kwargs["tools"]]
    assert names == ["tool_search", "mcp__agency__submit_output"]
    refreshed = kwargs["tools"][1]["function"]
    assert refreshed["description"] == "New description"
    assert refreshed["parameters"]["properties"]["value"]["type"] == "number"


@pytest.mark.parametrize("tool_type", ["custom", "web_search"])
def test_responses_tools_to_openai_warns_and_omits_nonfunction_tools(tool_type):
    warnings = []

    converted = responses_tools_to_openai(
        [{"type": tool_type, "name": "unsupported"}], warning_handler=warnings.append
    )

    assert converted is None
    assert len(warnings) == 1
    assert repr(tool_type) in warnings[0]


def test_responses_tools_to_openai_flattens_namespace_functions_reversibly():
    tool_name_map = {}
    converted = responses_tools_to_openai(
        [
            {
                "type": "namespace",
                "name": "mcp__agency",
                "description": "Agency harness tools.",
                "tools": [
                    {
                        "type": "function",
                        "name": "submit_output",
                        "description": "Submit one output field.",
                        "strict": False,
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            }
        ],
        tool_name_map=tool_name_map,
    )

    function = converted[0]["function"]
    assert function["name"] == "mcp__agency__submit_output"
    assert function["description"] == "Agency harness tools.\n\nSubmit one output field."
    assert tool_name_map["mcp__agency__submit_output"]["namespace"] == "mcp__agency"
    assert tool_name_map["mcp__agency__submit_output"]["name"] == "submit_output"


def test_responses_tools_to_openai_warns_and_omits_namespace_custom_child():
    translation_warnings = []

    converted = responses_tools_to_openai(
        [
            {
                "type": "namespace",
                "name": "mcp__agency",
                "tools": [{"type": "custom", "name": "freeform"}],
            }
        ],
        warning_handler=translation_warnings.append,
    )

    assert converted is None
    assert len(translation_warnings) == 1
    assert "namespace 'mcp__agency' child tool type 'custom'" in translation_warnings[0]


def test_responses_tools_to_openai_maps_client_tool_search():
    tool_name_map = {}
    converted = responses_tools_to_openai(
        [
            {
                "type": "tool_search",
                "execution": "client",
                "description": "Search available tools.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
        tool_name_map=tool_name_map,
    )

    assert converted[0]["function"]["name"] == "tool_search"
    assert converted[0]["function"]["parameters"]["required"] == ["query"]
    assert tool_name_map["tool_search"]["response_type"] == "tool_search_call"


def test_responses_tools_to_openai_rejects_flattened_name_collision():
    with pytest.raises(UnsupportedResponsesRequest, match="collide"):
        responses_tools_to_openai(
            [
                {
                    "type": "function",
                    "name": "mcp__agency__submit_output",
                    "parameters": {"type": "object"},
                },
                {
                    "type": "namespace",
                    "name": "mcp__agency",
                    "tools": [
                        {
                            "type": "function",
                            "name": "submit_output",
                            "parameters": {"type": "object"},
                        }
                    ],
                },
            ]
        )


def test_responses_tools_to_openai_keeps_functions_when_custom_tool_is_omitted():
    warnings = []
    tools = [
        {
            "type": "function",
            "name": "exec_command",
            "description": "Run a command",
            "parameters": {"type": "object"},
            "strict": False,
        },
        {
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch",
            "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
        },
    ]

    converted = responses_tools_to_openai(tools, warning_handler=warnings.append)

    assert [tool["function"]["name"] for tool in converted] == ["exec_command"]
    assert len(warnings) == 1
    assert "custom" in warnings[0]


def test_responses_tools_to_openai_rejects_deferred_function_tool():
    with pytest.raises(UnsupportedResponsesRequest, match="defer_loading"):
        responses_tools_to_openai(
            [
                {
                    "type": "function",
                    "name": "later",
                    "parameters": {"type": "object"},
                    "defer_loading": True,
                }
            ]
        )


@pytest.mark.parametrize("content_type", ["input_image", "input_file", "refusal"])
def test_responses_request_to_openai_rejects_nontext_message_content(content_type):
    body = {
        "model": "m",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": content_type, "image_url": "data:image/png;base64,AA=="}],
            }
        ],
    }

    with pytest.raises(UnsupportedResponsesRequest, match=repr(content_type)):
        responses_request_to_openai(body)


def test_responses_request_to_openai_rejects_unsupported_input_item_type():
    body = {"model": "m", "input": [{"type": "reasoning", "id": "reasoning-1"}]}

    with pytest.raises(UnsupportedResponsesRequest, match="reasoning"):
        responses_request_to_openai(body)


def test_responses_request_to_openai_rejects_native_json_schema_format():
    body = {
        "model": "m",
        "input": "hi",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {"type": "object"},
                "strict": True,
            }
        },
    }

    with pytest.raises(UnsupportedResponsesRequest, match=r"text\.format"):
        responses_request_to_openai(body)


def test_responses_request_to_openai_maps_named_function_tool_choice():
    body = {
        "model": "m",
        "input": "hi",
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object"},
            }
        ],
        "tool_choice": {"type": "function", "name": "get_weather"},
    }

    kwargs = responses_request_to_openai(body)

    assert kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_weather"},
    }


def test_responses_request_to_openai_rejects_nonstring_tool_choice_namespace():
    body = {
        "model": "m",
        "input": "hi",
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object"},
            }
        ],
        "tool_choice": {"type": "function", "namespace": 42, "name": "get_weather"},
    }

    with pytest.raises(UnsupportedResponsesRequest, match="namespace must be a string or null"):
        responses_request_to_openai(body)


def test_responses_request_to_openai_rejects_required_choice_without_function_tools():
    body = {
        "model": "m",
        "input": "hi",
        "tools": [{"type": "custom", "name": "apply_patch"}],
        "tool_choice": "required",
    }

    with pytest.warns(RuntimeWarning, match="custom"):
        with pytest.raises(UnsupportedResponsesRequest, match="no translatable function"):
            responses_request_to_openai(body)


# ---------------------------------------------------------------------------
# OpenAI response -> Responses API response
# ---------------------------------------------------------------------------


def test_openai_response_to_responses_api_text_only():
    resp = _Response(
        [_Choice(message=_Message(content="the answer is 4"), finish_reason="stop")],
        usage=_Usage(3, 4),
    )
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
    resp = _Response(
        [_Choice(message=_Message(content=None, tool_calls=[tc]), finish_reason="tool_calls")],
        usage=_Usage(1, 1),
    )
    out = openai_response_to_responses_api(resp, "m")
    assert out["output"][0]["type"] == "function_call"
    assert out["output"][0]["call_id"] == "call1"
    assert out["output"][0]["name"] == "get_weather"
    assert out["output"][0]["arguments"] == '{"city": "SF"}'


def test_openai_response_to_responses_api_restores_namespace_function_call():
    tc = _ToolCall(
        id="call1",
        name="mcp__agency__submit_output",
        arguments='{"field":"status","value":"done"}',
    )
    resp = _Response(
        [_Choice(message=_Message(content=None, tool_calls=[tc]), finish_reason="tool_calls")],
        usage=_Usage(1, 1),
    )
    tool_name_map = {
        "mcp__agency__submit_output": {
            "response_type": "function_call",
            "namespace": "mcp__agency",
            "name": "submit_output",
        }
    }

    out = openai_response_to_responses_api(resp, "m", tool_name_map=tool_name_map)

    assert out["output"][0]["type"] == "function_call"
    assert out["output"][0]["namespace"] == "mcp__agency"
    assert out["output"][0]["name"] == "submit_output"


def test_openai_response_to_responses_api_restores_tool_search_call():
    tc = _ToolCall(
        id="search-1", name="tool_search", arguments='{"query":"submit_output","limit":1}'
    )
    resp = _Response(
        [_Choice(message=_Message(content=None, tool_calls=[tc]), finish_reason="tool_calls")],
        usage=_Usage(1, 1),
    )
    tool_name_map = {
        "tool_search": {
            "response_type": "tool_search_call",
            "namespace": None,
            "name": "tool_search",
            "execution": "client",
        }
    }

    out = openai_response_to_responses_api(resp, "m", tool_name_map=tool_name_map)

    assert out["output"][0] == {
        "type": "tool_search_call",
        "id": out["output"][0]["id"],
        "call_id": "search-1",
        "execution": "client",
        "arguments": {"query": "submit_output", "limit": 1},
        "status": "completed",
    }


def test_openai_response_to_responses_api_preserves_cached_and_reasoning_usage():
    usage = _Usage(11, 7, cached_tokens=4, reasoning_tokens=3)
    resp = _Response([_Choice(message=_Message(content="done"), finish_reason="stop")], usage=usage)

    out = openai_response_to_responses_api(resp, "m")

    assert out["usage"] == {
        "input_tokens": 11,
        "input_tokens_details": {"cached_tokens": 4},
        "output_tokens": 7,
        "output_tokens_details": {"reasoning_tokens": 3},
        "total_tokens": 18,
    }


def test_openai_response_to_responses_api_rejects_missing_finish_reason():
    resp = _Response([_Choice(message=_Message(content="partial"))], usage=_Usage(1, 1))

    out = openai_response_to_responses_api(resp, "m")

    assert out["status"] == "failed"
    assert out["error"]["code"] == "missing_finish_reason"
    assert out["output"][0]["status"] == "incomplete"


@pytest.mark.parametrize(
    ("finish_reason", "expected_reason"),
    [("length", "max_output_tokens"), ("content_filter", "content_filter")],
)
def test_openai_response_to_responses_api_does_not_complete_truncated_output(
    finish_reason, expected_reason
):
    resp = _Response(
        [_Choice(message=_Message(content="partial"), finish_reason=finish_reason)],
        usage=_Usage(3, 4),
    )

    out = openai_response_to_responses_api(resp, "m")

    assert out["status"] == "incomplete"
    assert out["incomplete_details"] == {"reason": expected_reason}
    assert out["output"][0]["status"] == "incomplete"


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


def test_openai_chunks_to_responses_sse_preserves_cached_and_reasoning_usage():
    chunks = [
        _Chunk(choices=[_Choice(delta=_Delta(content="done"))]),
        _Chunk(choices=[_Choice(delta=_Delta(), finish_reason="stop")]),
        _Chunk(
            usage=_Usage(
                prompt_tokens=11,
                completion_tokens=7,
                cached_tokens=4,
                reasoning_tokens=3,
            )
        ),
    ]

    frames = "".join(openai_chunks_to_responses_sse(chunks, "m"))
    completed = next(data for event, data in _parse_sse(frames) if event == "response.completed")

    assert completed["response"]["usage"] == {
        "input_tokens": 11,
        "input_tokens_details": {"cached_tokens": 4},
        "output_tokens": 7,
        "output_tokens_details": {"reasoning_tokens": 3},
        "total_tokens": 18,
    }


def test_openai_chunks_to_responses_sse_rejects_empty_stream():
    events = _parse_sse("".join(openai_chunks_to_responses_sse([], "m")))

    assert events[-1][0] == "response.failed"
    assert events[-1][1]["response"]["status"] == "failed"
    assert events[-1][1]["response"]["error"]["code"] == "missing_finish_reason"


def test_openai_chunks_to_responses_sse_rejects_unterminated_partial_stream():
    chunks = [_Chunk(choices=[_Choice(delta=_Delta(content="partial"))])]

    events = _parse_sse("".join(openai_chunks_to_responses_sse(chunks, "m")))

    assert events[-1][0] == "response.failed"
    assert events[-1][1]["response"]["error"]["code"] == "missing_finish_reason"
    done = next(data for event, data in events if event == "response.output_item.done")
    assert done["item"]["status"] == "incomplete"


def test_openai_chunks_to_responses_sse_emits_failure_after_provider_stream_error():
    def broken_chunks():
        yield _Chunk(choices=[_Choice(delta=_Delta(content="partial"))])
        raise RuntimeError("provider stream disconnected")

    events = _parse_sse("".join(openai_chunks_to_responses_sse(broken_chunks(), "m")))

    assert events[-1][0] == "response.failed"
    response = events[-1][1]["response"]
    assert response["status"] == "failed"
    assert response["error"]["code"] == "upstream_stream_error"
    assert "provider stream disconnected" in response["error"]["message"]
    assert not any(event == "response.completed" for event, _data in events)


def test_openai_chunks_to_responses_sse_does_not_complete_length_truncation():
    chunks = [
        _Chunk(choices=[_Choice(delta=_Delta(content="partial"))]),
        _Chunk(choices=[_Choice(delta=_Delta(), finish_reason="length")]),
    ]

    frames = "".join(openai_chunks_to_responses_sse(chunks, "m"))
    events = _parse_sse(frames)

    assert events[-1][0] == "response.incomplete"
    assert events[-1][1]["response"]["status"] == "incomplete"
    assert events[-1][1]["response"]["incomplete_details"] == {"reason": "max_output_tokens"}
    done = next(data for event, data in events if event == "response.output_item.done")
    assert done["item"]["status"] == "incomplete"


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
    assert fc_item["call_id"] == "call1"
    assert fc_item["name"] == "get_weather"
    assert fc_item["arguments"] == '{"city": "SF"}'


def test_openai_chunks_to_responses_sse_restores_namespace_function_call():
    chunks = [
        _Chunk(
            choices=[
                _Choice(
                    delta=_Delta(
                        tool_calls=[
                            _ToolCall(
                                id="submit-1",
                                name="mcp__agency__submit_output",
                                arguments='{"field":"status"}',
                                index=0,
                            )
                        ]
                    ),
                    finish_reason="tool_calls",
                )
            ]
        )
    ]
    tool_name_map = {
        "mcp__agency__submit_output": {
            "response_type": "function_call",
            "namespace": "mcp__agency",
            "name": "submit_output",
        }
    }

    events = _parse_sse(
        "".join(openai_chunks_to_responses_sse(chunks, "m", tool_name_map=tool_name_map))
    )
    done = next(data["item"] for event, data in events if event == "response.output_item.done")

    assert done["type"] == "function_call"
    assert done["namespace"] == "mcp__agency"
    assert done["name"] == "submit_output"


def test_openai_chunks_to_responses_sse_restores_tool_search_call():
    chunks = [
        _Chunk(
            choices=[
                _Choice(
                    delta=_Delta(
                        tool_calls=[
                            _ToolCall(
                                id="search-1",
                                name="tool_search",
                                arguments='{"query":"submit_output","limit":1}',
                                index=0,
                            )
                        ]
                    ),
                    finish_reason="tool_calls",
                )
            ]
        )
    ]
    tool_name_map = {
        "tool_search": {
            "response_type": "tool_search_call",
            "namespace": None,
            "name": "tool_search",
            "execution": "client",
        }
    }

    events = _parse_sse(
        "".join(openai_chunks_to_responses_sse(chunks, "m", tool_name_map=tool_name_map))
    )
    done = next(data["item"] for event, data in events if event == "response.output_item.done")

    assert done["type"] == "tool_search_call"
    assert done["execution"] == "client"
    assert done["arguments"] == {"query": "submit_output", "limit": 1}


def test_openai_chunks_to_responses_sse_keeps_fragmented_tool_search_identity_consistent():
    chunks = [
        _Chunk(
            choices=[
                _Choice(
                    delta=_Delta(
                        tool_calls=[_ToolCall(id="search-1", name="tool_", arguments="{", index=0)]
                    )
                )
            ]
        ),
        _Chunk(
            choices=[
                _Choice(
                    delta=_Delta(
                        tool_calls=[
                            _ToolCall(name="search", arguments='"query":"submit_output"}', index=0)
                        ]
                    ),
                    finish_reason="tool_calls",
                )
            ]
        ),
    ]
    tool_name_map = {
        "tool_search": {
            "response_type": "tool_search_call",
            "namespace": None,
            "name": "tool_search",
            "execution": "client",
        }
    }

    events = _parse_sse(
        "".join(openai_chunks_to_responses_sse(chunks, "m", tool_name_map=tool_name_map))
    )
    added = next(data["item"] for event, data in events if event == "response.output_item.added")
    done = next(data["item"] for event, data in events if event == "response.output_item.done")

    assert added["id"] == done["id"]
    assert added["type"] == done["type"] == "tool_search_call"
    assert added["execution"] == done["execution"] == "client"


def test_openai_chunks_to_responses_sse_keeps_fragmented_namespace_identity_consistent():
    chunks = [
        _Chunk(
            choices=[
                _Choice(
                    delta=_Delta(
                        tool_calls=[
                            _ToolCall(
                                id="submit-1",
                                name="mcp__agency__submit_",
                                arguments="{",
                                index=0,
                            )
                        ]
                    )
                )
            ]
        ),
        _Chunk(
            choices=[
                _Choice(
                    delta=_Delta(tool_calls=[_ToolCall(name="output", arguments="}", index=0)]),
                    finish_reason="tool_calls",
                )
            ]
        ),
    ]
    tool_name_map = {
        "mcp__agency__submit_output": {
            "response_type": "function_call",
            "namespace": "mcp__agency",
            "name": "submit_output",
        }
    }

    events = _parse_sse(
        "".join(openai_chunks_to_responses_sse(chunks, "m", tool_name_map=tool_name_map))
    )
    added = next(data["item"] for event, data in events if event == "response.output_item.added")
    done = next(data["item"] for event, data in events if event == "response.output_item.done")

    assert added["id"] == done["id"]
    assert added["namespace"] == done["namespace"] == "mcp__agency"
    assert added["name"] == done["name"] == "submit_output"


def test_openai_chunks_to_responses_sse_accepts_fragmented_tool_metadata():
    chunks = [
        _Chunk(
            choices=[
                _Choice(
                    delta=_Delta(
                        tool_calls=[_ToolCall(id="", name="exec_", arguments="{", index=0)]
                    )
                )
            ]
        ),
        _Chunk(
            choices=[
                _Choice(
                    delta=_Delta(
                        tool_calls=[
                            _ToolCall(
                                id="call-late",
                                name="command",
                                arguments='"cmd":"pwd"}',
                                index=0,
                            )
                        ]
                    ),
                    finish_reason="tool_calls",
                )
            ]
        ),
    ]

    frames = "".join(openai_chunks_to_responses_sse(chunks, "m"))
    events = _parse_sse(frames)
    done_items = [data["item"] for event, data in events if event == "response.output_item.done"]
    function_call = next(item for item in done_items if item["type"] == "function_call")

    assert function_call["call_id"] == "call-late"
    assert function_call["name"] == "exec_command"
    assert function_call["arguments"] == '{"cmd":"pwd"}'
