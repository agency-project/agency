"""Tests for _ClaudeCodeBackend's LLM wire-format translation -- Anthropic
Messages API <-> agency format, both directions, both streaming and
non-streaming."""

from __future__ import annotations

import json

from agency.agconfig import agConfig
from agency.harness.adapters.claude_code import _ClaudeCodeBackend


def _backend() -> _ClaudeCodeBackend:
    return _ClaudeCodeBackend(agConfig())


def _text_block(text, index=0):
    return {"type": "text", "index": index, "text": text}


def test_harness_to_agency_basic_text():
    body = {
        "model": "claude-x",
        "system": "be helpful",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "hello"}],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"][0] == {"role": "system", "blocks": [_text_block("be helpful")]}
    assert agency["messages"][1] == {"role": "user", "blocks": [_text_block("hello")]}


def test_harness_to_agency_text_citations_preserved():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "see source", "citations": [{"url": "http://x"}]}
                ],
            }
        ],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"][0]["blocks"] == [
        {"type": "text", "index": 0, "text": "see source", "citations": [{"url": "http://x"}]}
    ]


def test_harness_to_agency_system_as_block_list():
    body = {
        "model": "m",
        "system": [{"type": "text", "text": "part one"}, {"type": "text", "text": " part two"}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"][0] == {"role": "system", "blocks": [_text_block("part one part two")]}


def test_harness_to_agency_assistant_tool_use():
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
    agency = _backend()._format_context_harness_to_agency(body)
    msg = agency["messages"][0]
    assert msg["role"] == "assistant"
    assert msg["blocks"] == [
        _text_block("let me check"),
        {
            "type": "tool_use",
            "index": 1,
            "id": "tu1",
            "name": "get_weather",
            "arguments": json.dumps({"city": "SF"}),
        },
    ]


def test_harness_to_agency_assistant_thinking_preserved_with_signature():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "pondering", "signature": "sig123"}],
            }
        ],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"][0]["blocks"] == [
        {"type": "thinking", "index": 0, "text": "pondering", "signature": "sig123"}
    ]


def test_harness_to_agency_tool_result_becomes_tool_role():
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
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"][0] == {
        "role": "tool",
        "blocks": [
            {"type": "tool_result", "index": 0, "tool_call_id": "tu1", "text": "72F and sunny"}
        ],
    }


def test_harness_to_agency_tool_result_with_block_list_content():
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
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"][0]["blocks"][0]["text"] == "result text"


def test_harness_to_agency_text_flushed_before_tool_result():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "here you go"},
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "ok"},
                ],
            }
        ],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"][0] == {"role": "user", "blocks": [_text_block("here you go")]}
    assert agency["messages"][1]["role"] == "tool"


def test_harness_to_agency_tools_and_tool_choice():
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


def test_harness_to_agency_unrecognized_user_content_block_preserved():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "source": {"type": "base64", "data": "abc"}}],
            }
        ],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"][0]["blocks"] == [
        {
            "type": "anthropic_image",
            "index": 0,
            "data": {"type": "image", "source": {"type": "base64", "data": "abc"}},
        }
    ]


def test_harness_to_agency_unrecognized_assistant_content_block_preserved():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "redacted_thinking", "data": "encrypted-blob"}],
            }
        ],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"][0]["blocks"] == [
        {
            "type": "anthropic_redacted_thinking",
            "index": 0,
            "data": {"type": "redacted_thinking", "data": "encrypted-blob"},
        }
    ]


def test_harness_to_agency_tool_result_with_image_content_preserves_raw():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": [{"type": "image", "source": {"type": "base64", "data": "abc"}}],
                    }
                ],
            }
        ],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    tool_result_block = agency["messages"][0]["blocks"][0]
    assert tool_result_block["text"] == ""
    assert tool_result_block["raw_content"] == [
        {"type": "image", "source": {"type": "base64", "data": "abc"}}
    ]


def test_harness_to_agency_tool_choice_none():
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": {"type": "none"},
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["tool_choice"] == "none"


def test_harness_to_agency_mid_array_system_message_folded():
    body = {
        "model": "m",
        "system": "top level",
        "messages": [
            {"role": "system", "content": "mid array"},
            {"role": "user", "content": "hi"},
        ],
    }
    agency = _backend()._format_context_harness_to_agency(body)
    assert agency["messages"][0] == {
        "role": "system",
        "blocks": [_text_block("top level\n\nmid array")],
    }


def test_agency_to_harness_text_citations_reconstructed():
    agency_response = {
        "message": {
            "role": "assistant",
            "blocks": [
                {
                    "type": "text",
                    "index": 0,
                    "text": "see source",
                    "citations": [{"url": "http://x"}],
                }
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "stop",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "m")
    assert out["content"] == [
        {"type": "text", "text": "see source", "citations": [{"url": "http://x"}]}
    ]


def test_agency_to_harness_text_only():
    agency_response = {
        "message": {"role": "assistant", "blocks": [_text_block("hi there")]},
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "stop_reason": "stop",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "claude-x")
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["content"] == [{"type": "text", "text": "hi there"}]
    assert out["stop_reason"] == "end_turn"
    assert out["usage"] == {"input_tokens": 5, "output_tokens": 3}


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
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        "stop_reason": "tool_calls",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "claude-x")
    assert out["content"] == [
        {"type": "tool_use", "id": "call1", "name": "get_weather", "input": {"city": "SF"}}
    ]
    assert out["stop_reason"] == "tool_use"


def test_agency_to_harness_thinking_with_signature():
    agency_response = {
        "message": {
            "role": "assistant",
            "blocks": [
                {"type": "thinking", "index": 0, "text": "pondering", "signature": "sig123"}
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "end_turn",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "m")
    assert out["content"] == [{"type": "thinking", "thinking": "pondering", "signature": "sig123"}]


def test_agency_to_harness_length_finish_reason():
    agency_response = {
        "message": {"role": "assistant", "blocks": [_text_block("cut off")]},
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "length",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "m")
    assert out["stop_reason"] == "max_tokens"


def test_agency_to_harness_unknown_block_reconstructed():
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
    assert out["content"] == [{"type": "redacted_thinking", "data": "encrypted-blob"}]


def test_agency_to_harness_foreign_origin_unknown_block_dropped():
    agency_response = {
        "message": {
            "role": "assistant",
            "blocks": [
                {
                    "type": "openai_responses_local_shell_call",
                    "index": 0,
                    "data": {"call_id": "c1"},
                }
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "stop",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "m")
    assert out["content"] == []


def test_agency_to_harness_anthropic_native_stop_reason_passes_through():
    agency_response = {
        "message": {"role": "assistant", "blocks": []},
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "pause_turn",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "m")
    assert out["stop_reason"] == "pause_turn"


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


def test_agency_stream_to_harness_text_citations_emitted_before_stop():
    stream = [
        {"type": "delta", "content": "see source"},
        {
            "type": "done",
            "message": {
                "role": "assistant",
                "blocks": [
                    {
                        "type": "text",
                        "index": 0,
                        "text": "see source",
                        "citations": [{"url": "http://x"}],
                    }
                ],
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "stop_reason": "stop",
        },
    ]
    frames = "".join(_backend()._format_agency_stream_to_harness(iter(stream), "m"))
    events = _parse_sse(frames)
    types = [t for t, _ in events]
    citation_idx = next(
        i
        for i, (t, d) in enumerate(events)
        if t == "content_block_delta" and d["delta"]["type"] == "citations_delta"
    )
    stop_idx = types.index("content_block_stop")
    assert citation_idx < stop_idx
    assert events[citation_idx][1]["delta"]["citation"] == {"url": "http://x"}


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
    frames = "".join(_backend()._format_agency_stream_to_harness(iter(stream), "claude-x"))
    events = _parse_sse(frames)
    types = [t for t, _ in events]
    assert types[0] == "message_start"
    assert types[1] == "content_block_start"
    assert types[2] == "content_block_delta"
    assert types[3] == "content_block_stop"
    assert "content_block_stop" in types
    assert types[-2] == "message_delta"
    assert types[-1] == "message_stop"

    text_deltas = [d for t, d in events if t == "content_block_delta"]
    assert "".join(d["delta"]["text"] for d in text_deltas) == "Hello"

    message_delta = next(d for t, d in events if t == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "end_turn"
    assert message_delta["usage"]["output_tokens"] == 2


def test_agency_stream_discards_draft_text_replaced_by_redirect():
    def stream():
        yield {"type": "delta", "content": "STALE-FINAL-ANSWER"}
        yield {
            "type": "done",
            "message": {"role": "assistant", "blocks": [_text_block("GENERATION-REDIRECTED")]},
            "stop_reason": "stop",
        }

    frames = "".join(_backend()._format_agency_stream_to_harness(stream(), "claude-x"))
    assert "STALE-FINAL-ANSWER" not in frames
    events = _parse_sse(frames)
    text = "".join(
        data["delta"]["text"]
        for kind, data in events
        if kind == "content_block_delta" and data["delta"]["type"] == "text_delta"
    )
    assert text == "GENERATION-REDIRECTED"


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
    frames = "".join(_backend()._format_agency_stream_to_harness(iter(stream), "claude-x"))
    events = _parse_sse(frames)
    types = [t for t, _ in events]
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

    json_delta = next(
        d["delta"]["partial_json"]
        for t, d in events
        if t == "content_block_delta" and d["delta"]["type"] == "input_json_delta"
    )
    assert json_delta == '{"city": "SF"}'

    message_delta = next(d for t, d in events if t == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"


def test_agency_stream_to_harness_thinking_block():
    stream = [
        {
            "type": "done",
            "message": {
                "role": "assistant",
                "blocks": [
                    {"type": "thinking", "index": 0, "text": "pondering", "signature": "sig-1"}
                ],
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "stop_reason": "end_turn",
        },
    ]
    frames = "".join(_backend()._format_agency_stream_to_harness(iter(stream), "m"))
    events = _parse_sse(frames)
    thinking_deltas = [
        d
        for t, d in events
        if t == "content_block_delta" and d["delta"]["type"] == "thinking_delta"
    ]
    assert thinking_deltas[0]["delta"]["thinking"] == "pondering"
    signature_deltas = [
        d
        for t, d in events
        if t == "content_block_delta" and d["delta"]["type"] == "signature_delta"
    ]
    assert signature_deltas[0]["delta"]["signature"] == "sig-1"


def test_agency_stream_to_harness_unknown_block_reconstructed():
    stream = [
        {
            "type": "done",
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
            "stop_reason": "end_turn",
        },
    ]
    frames = "".join(_backend()._format_agency_stream_to_harness(iter(stream), "m"))
    events = _parse_sse(frames)
    start_event = next(d for t, d in events if t == "content_block_start")
    assert start_event["content_block"] == {"type": "redacted_thinking", "data": "encrypted-blob"}
