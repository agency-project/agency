"""Tests for _CodexBackend's LLM wire-format translation -- OpenAI Responses
API <-> agency format, both directions, both streaming and non-streaming."""

from __future__ import annotations

import json
import pytest

from agency.configs.agconfig import agconfig
from agency.harness.adapters.codex import _CodexBackend


def _backend() -> _CodexBackend:
    return _CodexBackend(agconfig())


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
    assert converted["messages"] == [{"role": "user", "blocks": [_text_block("Run pwd")]}]
    assert converted["tools"][0]["function"]["name"] == "pwd"


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
    tools = converted["tools"]
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
    names = [tool["function"]["name"] for tool in converted["tools"]]
    assert names[0] != names[1]
    assert converted["tool_choice"]["function"]["name"] == names[0]
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
                            "name": request["tools"][0]["function"]["name"],
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
