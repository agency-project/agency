"""Tests for CodexAdapter's LLM wire-format translation -- OpenAI Responses
API <-> agency format, both directions, both streaming and non-streaming."""

from __future__ import annotations

import json
import pytest

from agency.llm.message_store import split_tool_schema
from agency.configs.agconfig import agconfig
from agency.harness.adapters.codex import CodexAdapter


def _backend() -> CodexAdapter:
    return CodexAdapter(agconfig())


def _text_block(text, index=0):
    return {"type": "text", "index": index, "text": text}


def test_parallel_calls_are_one_assistant_message_before_replies():
    body = {
        "input": [
            {"type": "function_call", "name": "a", "call_id": "a", "arguments": "{}"},
            {"type": "message", "role": "assistant", "content": "commentary"},
            {"type": "custom_tool_call", "name": "b", "call_id": "b", "input": "text(1)"},
            {"type": "function_call_output", "call_id": "a", "output": "A"},
            {"type": "custom_tool_call_output", "call_id": "b", "output": "B"},
        ]
    }
    messages = _backend()._format_context_harness_to_agency(body)["messages"]
    assert [m["role"] for m in messages] == ["assistant", "tool", "tool"]
    assert [b["index"] for b in messages[0]["blocks"]] == [0, 1, 2]
    assert [b["id"] for b in messages[0]["blocks"] if b["type"] == "tool_use"] == ["a", "b"]
    assert [m["blocks"][0]["tool_call_id"] for m in messages[1:]] == ["a", "b"]


@pytest.mark.parametrize("status,transient", [(400, False), (503, True)])
def test_stream_preserves_upstream_error_status(status, transient):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from agency.harness.clients.host_services_client import HostDispatchError

    class Router:
        def validate_token(self, token):
            return True

        def resolve_model(self, token):
            return "m"

        async def dispatch_stream_async(self, token, request):
            raise HostDispatchError(
                {"message": "synthetic error", "status_code": status, "transient": transient}
            )
            yield

    app = FastAPI()
    _backend().register(app, Router())
    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer fake"},
            json={"stream": True, "input": "hello"},
        )
    assert response.status_code == status
    assert response.json()["error"]["transient"] is transient


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
    assert split_tool_schema(agency["messages"])[1] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    assert split_tool_schema(agency["messages"])[2] == {
        "type": "function",
        "function": {"name": "get_weather"},
    }


def test_additional_tools_are_definitions_not_assistant_messages():
    body = {
        "input": [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [{"type": "function", "name": "pwd", "parameters": {"type": "object"}}],
            },
            {"role": "user", "content": "Run pwd"},
        ]
    }
    converted = _backend()._format_context_harness_to_agency(body)
    chat_messages, tools, _tool_choice, _extra = split_tool_schema(converted["messages"])
    assert chat_messages == [{"role": "user", "blocks": [_text_block("Run pwd")]}]
    assert tools[0]["function"]["name"] == "pwd"


def test_namespaced_custom_tools_round_trip_without_shared_adapter_state():
    from agency.harness.adapters.codex import _responses_tool_routes

    body = {
        "input": [
            {
                "type": "additional_tools",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "functions",
                        "tools": [
                            {
                                "type": "custom",
                                "name": "exec",
                                "description": "Run code",
                                "format": {"type": "text"},
                            },
                            {"type": "function", "name": "wait", "parameters": {"type": "object"}},
                        ],
                    }
                ],
            }
        ]
    }
    backend = _backend()
    converted = backend._format_context_harness_to_agency(body)
    tools = split_tool_schema(converted["messages"])[1]
    assert len(tools) == 2
    custom_name = tools[0]["function"]["name"]
    assert len(custom_name) <= 64
    assert tools[0]["function"]["parameters"]["properties"]["input"]["type"] == "string"
    code = 'text(await tools.exec_command({cmd: "pwd"}));'
    response = {
        "message": {
            "blocks": [
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": custom_name,
                    "arguments": json.dumps({"input": code}),
                }
            ]
        }
    }
    routes = _responses_tool_routes(body)
    output = backend._format_context_agency_to_harness(response, "m", tool_routes=routes)["output"][
        0
    ]
    assert {k: output[k] for k in ("type", "call_id", "name", "namespace", "input")} == {
        "type": "custom_tool_call",
        "call_id": "c1",
        "name": "exec",
        "namespace": "functions",
        "input": code,
    }
    history = backend._format_context_harness_to_agency(
        {"input": [output, {"type": "custom_tool_call_output", "call_id": "c1", "output": "done"}]}
    )
    assert history["messages"][0]["blocks"][0]["name"] == custom_name
    assert json.loads(history["messages"][0]["blocks"][0]["arguments"]) == {"input": code}
    assert history["messages"][1]["blocks"][0]["text"] == "done"
    frames = backend._format_agency_stream_to_harness(
        [{"type": "done", **response}], "m", tool_routes=routes
    )
    events = _parse_sse("".join(frames))
    item = next(d["item"] for t, d in events if t == "response.output_item.done")
    assert item["type"] == "custom_tool_call" and item["input"] == code
    # A different request cannot inherit another session's tool route table.
    other = backend._format_context_agency_to_harness(response, "m")["output"][0]
    assert other["type"] == "function_call"


def test_namespace_function_identity_survives_history_and_forced_choice():
    from agency.harness.adapters.codex import _responses_tool_routes

    body = {
        "tools": [
            {
                "type": "namespace",
                "name": namespace,
                "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}],
            }
            for namespace in ("files", "database")
        ],
        "tool_choice": {"type": "function", "namespace": "files", "name": "read"},
    }
    backend = _backend()
    converted = backend._format_context_harness_to_agency(body)
    names = [tool["function"]["name"] for tool in split_tool_schema(converted["messages"])[1]]
    assert names[0] != names[1]
    assert split_tool_schema(converted["messages"])[2]["function"]["name"] == names[0]
    response = {
        "message": {
            "blocks": [{"type": "tool_use", "id": "c", "name": names[0], "arguments": "{}"}]
        }
    }
    item = backend._format_context_agency_to_harness(
        response, "m", tool_routes=_responses_tool_routes(body)
    )["output"][0]
    assert item["name"] == "read" and item["namespace"] == "files"
    history = backend._format_context_harness_to_agency({"input": [item]})
    assert history["messages"][0]["blocks"][0]["name"] == names[0]


@pytest.mark.parametrize("stream", [False, True])
def test_registered_route_restores_custom_calls_for_each_request(stream):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    class Router:
        def validate_token(self, token):
            return token == "synthetic-token"

        def resolve_model(self, token):
            return "m"

        def dispatch(self, token, request):
            return {
                "message": {
                    "blocks": [
                        {
                            "type": "tool_use",
                            "id": "c",
                            "name": split_tool_schema(request["messages"])[1][0]["function"][
                                "name"
                            ],
                            "arguments": json.dumps({"input": "text(42)"}),
                        }
                    ]
                }
            }

        async def dispatch_stream_async(self, token, request):
            yield {"type": "done", **self.dispatch(token, request)}

    app = FastAPI()
    _backend().register(app, Router())
    with TestClient(app) as client:
        for namespace in ("first", "second"):
            response = client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer synthetic-token"},
                json={
                    "stream": stream,
                    "input": [
                        {
                            "type": "additional_tools",
                            "tools": [
                                {
                                    "type": "namespace",
                                    "name": namespace,
                                    "tools": [{"type": "custom", "name": "exec"}],
                                }
                            ],
                        }
                    ],
                },
            )
            assert response.status_code == 200
            if stream:
                item = next(
                    data["item"]
                    for kind, data in _parse_sse(response.text)
                    if kind == "response.output_item.done"
                )
            else:
                item = response.json()["output"][0]
            assert item["type"] == "custom_tool_call"
            assert item["namespace"] == namespace
            assert item["input"] == "text(42)"


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
    assert split_tool_schema(agency["messages"])[2] == "auto"


def test_harness_to_agency_hosted_tool_choice_preserved_with_named_type():
    body = {"model": "m", "input": "hi", "tool_choice": {"type": "file_search"}}
    agency = _backend()._format_context_harness_to_agency(body)
    assert split_tool_schema(agency["messages"])[2] == {
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
    assert out["usage"]["input_tokens"] == 5
    assert out["usage"]["output_tokens"] == 3
    assert out["usage"]["total_tokens"] == 8
    assert "prompt_tokens" not in out["usage"]
    assert "completion_tokens" not in out["usage"]


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
            "blocks": [
                {"type": "thinking", "index": 0, "text": "pondering"},
                _text_block("done", 1),
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "stop",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "m")
    reasoning_item = out["output"][0]
    assert reasoning_item["type"] == "reasoning"
    assert reasoning_item["summary"] == [{"type": "summary_text", "text": "pondering"}]
    # The model's own order is kept: reasoning comes before what it led to.
    assert [item["type"] for item in out["output"]] == ["reasoning", "message"]


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
                },
                _text_block("done", 1),
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "stop",
    }
    out = _backend()._format_context_agency_to_harness(agency_response, "m")
    assert [item["type"] for item in out["output"]] == ["message"]


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


# ---------------------------------------------------------------------------
# _tool_wire_name
# ---------------------------------------------------------------------------


def test_tool_wire_name_is_readable_like_claude_code_mcp_names():
    """Was a bare sha256 hash (agency_<48 hex chars>) -- unreadable in a
    transcript or log. Now mirrors Claude Code's own MCP tool naming scheme
    (`mcp__<server>__<tool>`), flattening a dotted/nested namespace into
    "__"-joined segments instead."""
    from agency.harness.adapters.codex import _tool_wire_name

    assert _tool_wire_name("function", "read") == "read"
    assert _tool_wire_name("function", "exec", "files") == "mcp__files__exec"
    assert _tool_wire_name("function", "exec", "files.local") == "mcp__files__local__exec"
    # A non-function kind gets its own trailing segment so it can't collide
    # on the wire with a same-named, same-namespace "function" tool.
    assert _tool_wire_name("custom", "exec", "files") == "mcp__files__exec__custom"
    assert _tool_wire_name("custom", "exec") == "mcp__exec__custom"


def test_tool_wire_name_truncates_from_the_front_when_over_64_chars():
    from agency.harness.adapters.codex import _tool_wire_name

    name = _tool_wire_name("function", "exec", "a." * 40 + "deep")
    assert len(name) == 64
    # The tail -- the actual tool name -- survives the truncation intact.
    assert name.endswith("__exec")


# ---------------------------------------------------------------------------
# Incomplete, empty and malformed model responses
#
# Regressions for defects found in recorded Terminal-Bench runs: 30 responses
# stopped at the output-token limit (29 with a tool call whose JSON arguments
# were cut off) were sent to Codex as completed function calls, and 41 empty
# responses were sent as completed turns, which Codex took as final answers.
# ---------------------------------------------------------------------------

TRUNCATED_ARGUMENTS = '{"cmd":"cat <<\'EOF\' > notes.txt\\n1. Linux scheduling determines'


def _done(blocks, stop_reason, usage=None):
    return {
        "type": "done",
        "message": {"role": "assistant", "blocks": blocks},
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 4096, "total_tokens": 4106},
        "stop_reason": stop_reason,
    }


def _tool_block(arguments, *, name="exec_command", call_id="call_1", index=0, **extra):
    return {
        "type": "tool_use",
        "index": index,
        "id": call_id,
        "name": name,
        "arguments": arguments,
        **extra,
    }


def _stream_events(item, **kwargs):
    text = "".join(_backend()._format_agency_stream_to_harness([item], "m", **kwargs))
    return _parse_sse(text)


def test_truncated_tool_call_is_reported_incomplete_and_never_emitted():
    events = _stream_events(_done([_tool_block(TRUNCATED_ARGUMENTS)], "length"))
    assert [name for name, _ in events] == ["response.created", "response.incomplete"]
    response = events[-1][1]["response"]
    assert response["status"] == "incomplete"
    assert response["incomplete_details"] == {"reason": "max_output_tokens"}
    assert response["usage"]["output_tokens"] == 4096
    # Codex starts a tool the moment its output item arrives, so not even a
    # well-formed call from a cut-off turn may be sent.
    assert not any("function_call" in json.dumps(data) for _, data in events)


def test_truncated_turn_emits_no_items_even_when_one_tool_call_is_whole():
    blocks = [
        _tool_block('{"cmd":"ls"}', call_id="whole"),
        _tool_block(TRUNCATED_ARGUMENTS, call_id="cut", index=1, incomplete=True),
    ]
    events = _stream_events(_done(blocks, "length"))
    assert [name for name, _ in events] == ["response.created", "response.incomplete"]


def test_truncated_text_only_turn_is_incomplete_not_a_final_answer():
    events = _stream_events(_done([_text_block("The answer is")], "length"))
    assert [name for name, _ in events] == ["response.created", "response.incomplete"]


def test_non_stream_truncated_turn_is_incomplete_with_no_output():
    out = _backend()._format_context_agency_to_harness(
        {
            "message": {"role": "assistant", "blocks": [_tool_block(TRUNCATED_ARGUMENTS)]},
            "usage": {"prompt_tokens": 1, "completion_tokens": 4096},
            "stop_reason": "length",
        },
        "m",
    )
    assert out["status"] == "incomplete"
    assert out["incomplete_details"] == {"reason": "max_output_tokens"}
    assert out["output"] == []


def test_content_filter_stop_is_incomplete():
    events = _stream_events(_done([_text_block("partial")], "content_filter"))
    assert events[-1][0] == "response.incomplete"
    assert events[-1][1]["response"]["incomplete_details"] == {"reason": "content_filter"}


@pytest.mark.parametrize(
    "blocks",
    [
        [],
        [{"type": "metadata", "index": 0, "usage": None, "stop_reason": "stop"}],
        [{"type": "thinking", "index": 0, "text": "", "signature": "openai_responses:abc"}],
        [_text_block("")],
    ],
    ids=["no-blocks", "metadata-only", "reasoning-only", "empty-text"],
)
def test_empty_response_fails_before_any_frame_is_sent(blocks):
    """The generator must raise on its first step: that is the only point at
    which the route can still answer with an HTTP error instead of a 200."""
    frames = _backend()._format_agency_stream_to_harness([_done(blocks, "stop")], "m")
    with pytest.raises(ValueError, match="no message and no tool call"):
        next(frames)


@pytest.mark.parametrize("arguments", [TRUNCATED_ARGUMENTS, "not json", "[1, 2]"])
def test_malformed_tool_call_with_a_normal_stop_reason_is_never_emitted(arguments):
    frames = _backend()._format_agency_stream_to_harness(
        [_done([_tool_block(arguments)], "tool_calls")], "m"
    )
    with pytest.raises(ValueError, match="incomplete arguments"):
        next(frames)


def test_truncated_custom_tool_call_fails_cleanly_instead_of_breaking_the_stream():
    """A custom tool's arguments are unwrapped with json.loads. A cut-off one
    used to raise after `response.created` had been sent, severing the SSE
    stream Codex was reading."""
    body = {"tools": [{"type": "custom", "name": "apply_patch", "description": "patch"}]}
    from agency.harness.adapters.codex import _responses_tool_routes

    routes = _responses_tool_routes(body)
    wire_name = next(iter(routes))
    cut = _tool_block('{"input":"*** Begin Patch\\n*** Add File: a.py', name=wire_name)
    assert [n for n, _ in _stream_events(_done([cut], "length"), tool_routes=routes)] == [
        "response.created",
        "response.incomplete",
    ]
    frames = _backend()._format_agency_stream_to_harness(
        [_done([cut], "tool_calls")], "m", tool_routes=routes
    )
    with pytest.raises(ValueError):
        next(frames)


def test_empty_text_with_a_valid_tool_call_is_a_normal_completed_turn():
    """Empty content next to a tool call is the ordinary shape of a tool
    turn (8,242 of 8,438 recorded responses), not an empty response."""
    events = _stream_events(_done([_text_block(""), _tool_block('{"cmd":"ls"}', index=1)], "stop"))
    assert events[-1][0] == "response.completed"
    items = [d["item"] for n, d in events if n == "response.output_item.done"]
    assert [i["type"] for i in items] == ["function_call"]
    assert items[0]["call_id"] == "call_1" and items[0]["status"] == "completed"
    assert json.loads(items[0]["arguments"]) == {"cmd": "ls"}


def test_tool_call_with_no_arguments_is_complete():
    events = _stream_events(_done([_tool_block("")], "tool_calls"))
    item = next(d["item"] for n, d in events if n == "response.output_item.done")
    assert item["arguments"] == "{}"


def test_reasoning_state_tool_call_ids_and_phase_round_trip_through_codex():
    """What Codex is sent, it sends back; translating that back must give the
    backend the same reasoning state, call ids and message phase."""
    signature = "openai_responses:rsn_opaque"
    blocks = [
        {"type": "thinking", "index": 0, "text": "", "signature": signature},
        {"type": "text", "index": 1, "text": "Checking the CPU count.", "phase": "commentary"},
        _tool_block('{"cmd":"nproc"}', call_id="call_abc", index=2),
    ]
    events = _stream_events(_done(blocks, "tool_calls"))
    items = [d["item"] for n, d in events if n == "response.output_item.done"]
    assert [i["type"] for i in items] == ["reasoning", "message", "function_call"]
    assert items[0]["encrypted_content"] == signature and items[0]["summary"] == []
    assert items[1]["phase"] == "commentary"
    assert items[2]["call_id"] == "call_abc"

    replay = [*items, {"type": "function_call_output", "call_id": "call_abc", "output": "6\n"}]
    messages = _backend()._format_context_harness_to_agency({"input": replay})["messages"]
    assert [m["role"] for m in messages] == ["assistant", "tool"]
    thinking, text, tool = messages[0]["blocks"]
    assert (thinking["type"], thinking["signature"]) == ("thinking", signature)
    assert (text["text"], text["phase"]) == ("Checking the CPU count.", "commentary")
    assert (tool["id"], tool["arguments"]) == ("call_abc", '{"cmd":"nproc"}')
    assert messages[1]["blocks"][0] == {
        "type": "tool_result",
        "index": 0,
        "tool_call_id": "call_abc",
        "text": "6\n",
    }


def test_usage_details_reach_codex_when_the_backend_reports_them():
    usage = {
        "prompt_tokens": 2344,
        "completion_tokens": 50,
        "total_tokens": 2394,
        "cached_tokens": 2342,
        "reasoning_tokens": 45,
    }
    events = _stream_events(_done([_text_block("ok")], "stop", usage))
    assert events[-1][1]["response"]["usage"] == {
        "input_tokens": 2344,
        "output_tokens": 50,
        "total_tokens": 2394,
        "input_tokens_details": {"cached_tokens": 2342},
        "output_tokens_details": {"reasoning_tokens": 45},
    }


def _route_client(done_item=None, error=None):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    warnings = []

    class Router:
        def validate_token(self, token):
            return True

        def resolve_model(self, token):
            return "m"

        def log_warning(self, token, message):
            warnings.append(message)

        async def dispatch_stream_async(self, token, request):
            if error is not None:
                raise error
            yield done_item

    app = FastAPI()
    _backend().register(app, Router())
    return TestClient(app), warnings


def test_route_answers_an_empty_response_with_a_retryable_http_error_not_a_200():
    client, _ = _route_client(_done([], "stop"))
    with client:
        response = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer t"},
            json={"input": "hi", "stream": True},
        )
    assert response.status_code == 502
    assert response.json()["error"]["transient"] is True
    assert "no message and no tool call" in response.json()["error"]["message"]


def test_route_streams_response_incomplete_for_a_truncated_turn():
    client, _ = _route_client(_done([_tool_block(TRUNCATED_ARGUMENTS)], "length"))
    with client:
        response = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer t"},
            json={"input": "hi", "stream": True},
        )
    assert response.status_code == 200
    assert [name for name, _ in _parse_sse(response.text)] == [
        "response.created",
        "response.incomplete",
    ]


def test_route_reports_untranslated_request_fields_once():
    client, warnings = _route_client(_done([_text_block("ok")], "stop"))
    body = {
        "input": "hi",
        "stream": True,
        "reasoning": {"effort": "high"},
        "prompt_cache_key": "conversation-1",
        "include": [],
    }
    with client:
        for _ in range(2):
            assert (
                client.post(
                    "/v1/responses", headers={"Authorization": "Bearer t"}, json=body
                ).status_code
                == 200
            )
    assert len(warnings) == 1
    assert "prompt_cache_key" in warnings[0] and "reasoning" in warnings[0]
    assert "include" not in warnings[0]  # empty values are not worth a warning
