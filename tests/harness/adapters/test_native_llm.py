"""Tests for NativeAdapter's LLM wire-format translation -- plain OpenAI
chat-completions <-> agency format, both directions, both streaming and
non-streaming."""

from __future__ import annotations

import json

from agency.configs.agconfig import agconfig
from agency.harness.adapters.native import NativeAdapter


def _backend() -> NativeAdapter:
    return NativeAdapter(agconfig())


def test_harness_to_agency_plain_messages():
    body = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
        ],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"] == [
        {"role": "system", "blocks": [{"type": "text", "index": 0, "text": "be helpful"}]},
        {"role": "user", "blocks": [{"type": "text", "index": 0, "text": "hello"}]},
    ]


def test_harness_to_agency_assistant_with_tool_calls():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                    }
                ],
            }
        ],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"][0]["blocks"] == [
        {"type": "text", "index": 0, "text": "checking"},
        {
            "type": "tool_use",
            "index": 1,
            "id": "call1",
            "name": "get_weather",
            "arguments": '{"city": "SF"}',
        },
    ]


def test_harness_to_agency_tool_role_message():
    body = {
        "model": "m",
        "messages": [{"role": "tool", "tool_call_id": "call1", "content": "72F and sunny"}],
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


def test_harness_to_agency_multimodal_content_array():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "http://x"}},
                ],
            }
        ],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"][0]["blocks"] == [
        {"type": "text", "index": 0, "text": "what is this?"},
        {
            "type": "openai_chatcompletions_image_url",
            "index": 1,
            "data": {"type": "image_url", "image_url": {"url": "http://x"}},
        },
    ]


def test_harness_to_agency_legacy_function_call_becomes_tool_use():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "get_weather", "arguments": "{}"},
            }
        ],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"][0]["blocks"] == [
        {"type": "tool_use", "index": 0, "id": "", "name": "get_weather", "arguments": "{}"}
    ]


def test_harness_to_agency_refusal_field_preserved():
    body = {
        "model": "m",
        "messages": [{"role": "assistant", "content": None, "refusal": "I can't help with that"}],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"][0]["blocks"] == [
        {"type": "openai_chatcompletions_refusal", "index": 0, "data": "I can't help with that"}
    ]


def test_harness_to_agency_tools_and_tool_choice_pass_through_unchanged():
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "description": "d", "parameters": {}},
        }
    ]
    tool_choice = {"type": "function", "function": {"name": "get_weather"}}
    body = {"model": "m", "messages": [], "tools": tools, "tool_choice": tool_choice}
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["tools"] == tools
    assert agency["tool_choice"] == tool_choice


def test_agency_to_harness_text_only():
    agency_response = {
        "message": {
            "role": "assistant",
            "blocks": [{"type": "text", "index": 0, "text": "hi there"}],
        },
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "stop_reason": "stop",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "m")
    assert out["object"] == "chat.completion"
    choice = out["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "hi there"}
    assert choice["finish_reason"] == "stop"
    assert out["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


def test_agency_to_harness_tool_use_and_anthropic_stop_reason_normalized():
    agency_response = {
        "message": {
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
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "tool_use",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "m")
    choice = out["choices"][0]
    assert choice["message"]["tool_calls"] == [
        {
            "id": "call1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
        }
    ]
    assert choice["message"]["content"] is None
    assert choice["finish_reason"] == "tool_calls"


def test_agency_to_harness_thinking_becomes_reasoning_content():
    agency_response = {
        "message": {
            "role": "assistant",
            "blocks": [{"type": "thinking", "index": 0, "text": "pondering"}],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "end_turn",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "m")
    assert out["choices"][0]["message"]["reasoning_content"] == "pondering"
    assert out["choices"][0]["finish_reason"] == "stop"


def test_agency_to_harness_unknown_block_reconstructed_as_message_field():
    agency_response = {
        "message": {
            "role": "assistant",
            "blocks": [
                {
                    "type": "openai_chatcompletions_refusal",
                    "index": 0,
                    "data": "I can't help with that",
                }
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "stop",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "m")
    assert out["choices"][0]["message"]["refusal"] == "I can't help with that"


def test_agency_to_harness_foreign_origin_unknown_block_dropped():
    agency_response = {
        "message": {
            "role": "assistant",
            "blocks": [
                {"type": "anthropic_redacted_thinking", "index": 0, "data": {"data": "blob"}}
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "stop",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "m")
    assert "refusal" not in out["choices"][0]["message"]
    assert out["choices"][0]["message"] == {"role": "assistant", "content": None}


def test_agency_to_harness_unknown_block_flattens_streaming_origin_fragments():
    agency_response = {
        "message": {
            "role": "assistant",
            "blocks": [
                {
                    "type": "openai_chatcompletions_refusal",
                    "index": 0,
                    "data": ["par", "tial"],
                }
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "stop",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "m")
    assert out["choices"][0]["message"]["refusal"] == "partial"


def _parse_sse(text):
    events = []
    for frame in text.strip().split("\n\n"):
        if not frame:
            continue
        assert frame.startswith("data: ")
        events.append(json.loads(frame[len("data: ") :]) if frame != "data: [DONE]" else "[DONE]")
    return events


def test_agency_stream_to_harness_emits_committed_text_and_ends_with_done():
    stream = [
        {"type": "delta", "content": "discarded draft"},
        {
            "type": "done",
            "message": {
                "role": "assistant",
                "blocks": [{"type": "text", "index": 0, "text": "Hello"}],
            },
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            "stop_reason": "stop",
        },
    ]
    frames = "".join(_backend()._format_agency_stream_to_harness(iter(stream), "m"))
    events = _parse_sse(frames)
    assert events[-1] == "[DONE]"
    content_deltas = [
        e["choices"][0]["delta"]["content"]
        for e in events
        if isinstance(e, dict) and e["choices"] and "content" in e["choices"][0]["delta"]
    ]
    assert content_deltas == ["Hello"]
    finish_chunk = next(
        e
        for e in events
        if isinstance(e, dict) and e["choices"] and e["choices"][0].get("finish_reason")
    )
    assert finish_chunk["choices"][0]["finish_reason"] == "stop"
    usage_chunk = next(e for e in events if isinstance(e, dict) and e["choices"] == [])
    assert usage_chunk["usage"]["completion_tokens"] == 2


def test_agency_stream_to_harness_tool_use_emits_full_arguments_at_done():
    stream = [
        {
            "type": "done",
            "message": {
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
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "stop_reason": "tool_use",
        },
    ]
    frames = "".join(_backend()._format_agency_stream_to_harness(iter(stream), "m"))
    events = _parse_sse(frames)
    tool_call_chunk = next(
        e
        for e in events
        if isinstance(e, dict) and e["choices"] and e["choices"][0]["delta"].get("tool_calls")
    )
    tc = tool_call_chunk["choices"][0]["delta"]["tool_calls"][0]
    assert tc == {
        "index": 0,
        "id": "call1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
    }
    finish_chunk = next(
        e
        for e in events
        if isinstance(e, dict) and e["choices"] and e["choices"][0].get("finish_reason")
    )
    assert finish_chunk["choices"][0]["finish_reason"] == "tool_calls"


def test_agency_stream_to_harness_unknown_block_reconstructed():
    stream = [
        {
            "type": "done",
            "message": {
                "role": "assistant",
                "blocks": [{"type": "openai_chatcompletions_refusal", "index": 0, "data": "no"}],
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "stop_reason": "stop",
        },
    ]
    frames = "".join(_backend()._format_agency_stream_to_harness(iter(stream), "m"))
    events = _parse_sse(frames)
    unknown_chunk = next(
        e
        for e in events
        if isinstance(e, dict) and e["choices"] and e["choices"][0]["delta"].get("refusal")
    )
    assert unknown_chunk["choices"][0]["delta"]["refusal"] == "no"


def test_harness_to_agency_reasoning_fields_become_thinking_block():
    raw = {
        "messages": [
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "plan",
                "reasoning_signature": "sig",
            }
        ]
    }
    blocks = _backend()._format_context_harness_to_agency(raw)["messages"][0]["blocks"]
    assert blocks == [
        {"type": "thinking", "index": 0, "text": "plan", "signature": "sig"},
        {"type": "text", "index": 1, "text": "answer"},
    ]


def test_harness_to_agency_without_reasoning_fields_has_no_thinking_block():
    raw = {"messages": [{"role": "assistant", "content": "answer"}]}
    blocks = _backend()._format_context_harness_to_agency(raw)["messages"][0]["blocks"]
    assert blocks == [{"type": "text", "index": 0, "text": "answer"}]


def test_agency_to_harness_thinking_signature_becomes_reasoning_signature():
    agency_response = {
        "message": {
            "role": "assistant",
            "blocks": [{"type": "thinking", "index": 0, "text": "", "signature": "sig"}],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "end_turn",
    }
    message = _backend()._format_context_agency_to_harness(agency_response, "m")["choices"][0][
        "message"
    ]
    assert message["reasoning_signature"] == "sig"


def test_agency_stream_to_harness_emits_reasoning_signature():
    stream = [
        {
            "type": "done",
            "message": {
                "role": "assistant",
                "blocks": [
                    {"type": "thinking", "index": 0, "text": "plan", "signature": "sig"},
                    {"type": "text", "index": 1, "text": "answer"},
                ],
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "stop_reason": "end_turn",
        }
    ]
    events = _parse_sse("".join(_backend()._format_agency_stream_to_harness(iter(stream), "m")))
    deltas = [e["choices"][0]["delta"] for e in events if e != "[DONE]" and e["choices"]]
    assert {"reasoning_content": "plan"} in deltas
    assert {"reasoning_signature": "sig"} in deltas


def test_reasoning_round_trips_agency_to_harness_and_back():
    agency_response = {
        "message": {
            "role": "assistant",
            "blocks": [
                {"type": "thinking", "index": 0, "text": "plan", "signature": "sig"},
                {"type": "text", "index": 1, "text": "answer"},
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "end_turn",
    }
    backend = _backend()
    harness_message = backend._format_context_agency_to_harness(agency_response, "m")["choices"][0][
        "message"
    ]
    blocks = backend._format_context_harness_to_agency({"messages": [harness_message]})["messages"][
        0
    ]["blocks"]
    assert blocks == agency_response["message"]["blocks"]
