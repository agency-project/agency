"""Tests for _CodexBackend's LLM wire-format translation -- OpenAI Responses
API <-> agency format, both directions, both streaming and non-streaming."""

from __future__ import annotations

import json

from agency.agconfig import agConfig
from agency.harness.adapters.codex import _CodexBackend


def _backend() -> _CodexBackend:
    return _CodexBackend(agConfig())


def _text_block(text, index=0):
    return {"type": "text", "index": index, "text": text}


def test_harness_to_agency_instructions_and_string_input():
    body = {"model": "m", "instructions": "be helpful", "input": "hello"}
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"] == [
        {"role": "system", "blocks": [_text_block("be helpful")]},
        {"role": "user", "blocks": [_text_block("hello")]},
    ]


def test_harness_to_agency_message_item_list_input():
    body = {
        "model": "m",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi there"}]}],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"] == [{"role": "user", "blocks": [_text_block("hi there")]}]


def test_harness_to_agency_function_call_item():
    body = {
        "model": "m",
        "input": [
            {
                "type": "function_call",
                "call_id": "call1",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            }
        ],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"] == [
        {
            "role": "assistant",
            "blocks": [
                {
                    "type": "tool_use",
                    "index": 0,
                    "id": "call1",
                    "name": "get_weather",
                    "arguments": '{"city": "SF"}',
                }
            ],
        }
    ]


def test_harness_to_agency_function_call_output_item():
    body = {
        "model": "m",
        "input": [{"type": "function_call_output", "call_id": "call1", "output": "72F and sunny"}],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"] == [
        {
            "role": "tool",
            "blocks": [
                {
                    "type": "tool_result",
                    "index": 0,
                    "tool_call_id": "call1",
                    "text": "72F and sunny",
                }
            ],
        }
    ]


def test_harness_to_agency_function_call_output_with_block_list():
    body = {
        "model": "m",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call1",
                "output": [{"type": "output_text", "text": "result text"}],
            }
        ],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"][0]["blocks"][0]["text"] == "result text"


def test_harness_to_agency_tools_and_tool_choice():
    body = {
        "model": "m",
        "input": "hi",
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        "tool_choice": {"type": "function", "name": "get_weather"},
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    assert agency["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_harness_to_agency_unrecognized_content_item_preserved():
    body = {
        "model": "m",
        "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "http://x"}]}],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"][0]["blocks"] == [
        {
            "type": "openai_responses_input_image",
            "index": 0,
            "data": {"type": "input_image", "image_url": "http://x"},
        }
    ]


def test_harness_to_agency_reasoning_as_input_item_becomes_thinking():
    body = {
        "model": "m",
        "input": [
            {
                "type": "reasoning",
                "id": "rs1",
                "summary": [{"type": "summary_text", "text": "pondering"}],
                "encrypted_content": "sig-abc",
            }
        ],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"] == [
        {
            "role": "assistant",
            "blocks": [
                {"type": "thinking", "index": 0, "text": "pondering", "signature": "sig-abc"}
            ],
        }
    ]


def test_harness_to_agency_local_shell_call_item_preserved():
    body = {
        "model": "m",
        "input": [{"type": "local_shell_call", "call_id": "c1", "action": {"command": ["ls"]}}],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    block = agency["messages"][0]["blocks"][0]
    assert block["type"] == "openai_responses_local_shell_call"
    assert block["data"]["call_id"] == "c1"


def test_harness_to_agency_tool_choice_auto_passthrough():
    body = {"model": "m", "input": "hi", "tool_choice": "auto"}
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["tool_choice"] == "auto"


def test_harness_to_agency_hosted_tool_choice_preserved_with_named_type():
    body = {"model": "m", "input": "hi", "tool_choice": {"type": "file_search"}}
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["tool_choice"] == {
        "type": "openai_responses_file_search",
        "data": {"type": "file_search"},
    }


def test_agency_to_harness_text_only():
    agency_response = {
        "message": {"role": "assistant", "blocks": [_text_block("hi there")]},
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "stop_reason": "stop",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "codex-x")
    assert out["object"] == "response"
    assert out["output"] == [
        {
            "type": "message",
            "id": out["output"][0]["id"],
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi there", "annotations": []}],
        }
    ]
    assert out["usage"] == {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}


def test_agency_to_harness_tool_use():
    agency_response = {
        "message": {
            "role": "assistant",
            "blocks": [
                {
                    "type": "tool_use",
                    "index": 0,
                    "id": "call1",
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "SF"}),
                }
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "tool_calls",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "codex-x")
    fc = out["output"][0]
    assert fc["type"] == "function_call"
    assert fc["call_id"] == "call1"
    assert fc["name"] == "get_weather"
    assert fc["arguments"] == json.dumps({"city": "SF"})


def test_agency_to_harness_thinking_becomes_reasoning_item():
    agency_response = {
        "message": {
            "role": "assistant",
            "blocks": [{"type": "thinking", "index": 0, "text": "pondering"}],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "stop",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "m")
    reasoning_item = out["output"][0]
    assert reasoning_item["type"] == "reasoning"
    assert reasoning_item["summary"] == [{"type": "summary_text", "text": "pondering"}]


def test_agency_to_harness_unknown_block_reconstructed():
    agency_response = {
        "message": {
            "role": "assistant",
            "blocks": [
                {
                    "type": "openai_responses_local_shell_call",
                    "index": 0,
                    "data": {"call_id": "c1", "action": {"command": ["ls"]}},
                }
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "stop",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "m")
    assert out["output"][0] == {
        "type": "local_shell_call",
        "call_id": "c1",
        "action": {"command": ["ls"]},
    }


def test_agency_to_harness_foreign_origin_unknown_block_dropped():
    agency_response = {
        "message": {
            "role": "assistant",
            "blocks": [
                {
                    "type": "anthropic_redacted_thinking",
                    "index": 0,
                    "data": {"data": "encrypted-blob"},
                }
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "stop",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "m")
    assert out["output"] == []


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


def test_agency_stream_to_harness_text_stream():
    stream = [
        {"type": "delta", "content": "Hel"},
        {"type": "delta", "content": "lo"},
        {
            "type": "done",
            "message": {"role": "assistant", "blocks": [_text_block("Hello")]},
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            "stop_reason": "stop",
        },
    ]
    frames = "".join(_backend()._format_agency_stream_to_harness(iter(stream), "codex-x"))
    events = _parse_sse(frames)
    types = [t for t, _ in events]
    assert types[0] == "response.created"
    assert "response.output_item.added" in types
    delta_events = [d for t, d in events if t == "response.output_text.delta"]
    assert "".join(d["delta"] for d in delta_events) == "Hello"
    assert types[-1] == "response.completed"
    completed = events[-1][1]
    assert completed["response"]["usage"]["output_tokens"] == 2


def test_agency_stream_to_harness_tool_use_after_text():
    stream = [
        {"type": "delta", "content": "checking..."},
        {
            "type": "done",
            "message": {
                "role": "assistant",
                "blocks": [
                    _text_block("checking..."),
                    {
                        "type": "tool_use",
                        "index": 1,
                        "id": "call1",
                        "name": "get_weather",
                        "arguments": '{"city": "SF"}',
                    },
                ],
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "stop_reason": "tool_calls",
        },
    ]
    frames = "".join(_backend()._format_agency_stream_to_harness(iter(stream), "codex-x"))
    events = _parse_sse(frames)
    function_call_items = [
        d["item"]
        for t, d in events
        if t == "response.output_item.done" and d["item"]["type"] == "function_call"
    ]
    assert function_call_items[0]["call_id"] == "call1"
    assert function_call_items[0]["name"] == "get_weather"
    assert function_call_items[0]["arguments"] == '{"city": "SF"}'


def test_agency_stream_to_harness_no_text_emits_only_final_items():
    stream = [
        {
            "type": "done",
            "message": {
                "role": "assistant",
                "blocks": [
                    {"type": "tool_use", "index": 0, "id": "c1", "name": "f", "arguments": "{}"}
                ],
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "stop_reason": "tool_calls",
        },
    ]
    frames = "".join(_backend()._format_agency_stream_to_harness(iter(stream), "m"))
    events = _parse_sse(frames)
    types = [t for t, _ in events]
    assert "response.output_text.delta" not in types
    assert types == [
        "response.created",
        "response.output_item.added",
        "response.output_item.done",
        "response.completed",
    ]


def test_agency_stream_to_harness_unknown_block_reconstructed():
    stream = [
        {
            "type": "done",
            "message": {
                "role": "assistant",
                "blocks": [
                    {
                        "type": "openai_responses_local_shell_call",
                        "index": 0,
                        "data": {"call_id": "c1", "action": {"command": ["ls"]}},
                    }
                ],
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "stop_reason": "stop",
        },
    ]
    frames = "".join(_backend()._format_agency_stream_to_harness(iter(stream), "m"))
    events = _parse_sse(frames)
    done_item = next(
        d["item"]
        for t, d in events
        if t == "response.output_item.done" and d["item"]["type"] == "local_shell_call"
    )
    assert done_item["call_id"] == "c1"
