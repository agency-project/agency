"""Tests for NativeAdapter's LLM wire-format translation -- plain OpenAI
chat-completions <-> agency format, both directions, both streaming and
non-streaming."""

from __future__ import annotations

import json

import pytest

from agency.configs.agconfig import agconfig
from agency.harness.adapters.native import NativeAdapter


def _backend() -> NativeAdapter:
    return NativeAdapter(agconfig())


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


def _stream_client(router):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    _backend().register(app, router)
    return TestClient(app)


class _StreamRouter:
    def __init__(self, items, exc=None):
        self.items, self.exc = items, exc

    def validate_token(self, token):
        return True

    def resolve_model(self, token):
        return "m"

    async def dispatch_stream_async(self, token, request):
        for item in self.items:
            yield item
        if self.exc:
            raise self.exc


def _post_stream(client):
    return client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer fake"},
        json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )


def _done(text="hello"):
    return {
        "type": "done",
        "message": {"blocks": [{"type": "text", "index": 0, "text": text}]},
        "stop_reason": "end_turn",
        "usage": {},
    }


@pytest.mark.parametrize("status,transient", [(400, False), (503, True)])
def test_stream_upstream_error_before_first_event_preserves_status_and_transient(status, transient):
    from agency.harness.clients.host_services_client import HostDispatchError

    exc = HostDispatchError(
        {"message": "invalid_request: bad param", "status_code": status, "transient": transient}
    )
    with _stream_client(_StreamRouter([], exc)) as client:
        response = _post_stream(client)
    assert response.status_code == status
    error = response.json()["error"]
    assert error["message"] == "invalid_request: bad param"
    assert error["type"] == "upstream_error"
    assert error["transient"] is transient


def test_stream_ending_without_events_is_502():
    with _stream_client(_StreamRouter([])) as client:
        response = _post_stream(client)
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"


def test_stream_tool_use_and_usage_pass_through_route():
    done = {
        "type": "done",
        "message": {
            "blocks": [{"type": "tool_use", "index": 0, "id": "c1", "name": "f", "arguments": "{}"}]
        },
        "stop_reason": "tool_use",
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }
    with _stream_client(_StreamRouter([{"type": "delta"}, done])) as client:
        response = _post_stream(client)
    payloads = [
        json.loads(f[len("data: ") :])
        for f in response.text.split("\n\n")
        if f.startswith("data: {")
    ]
    assert payloads[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "c1"
    assert payloads[1]["choices"][0]["finish_reason"] == "tool_calls"
    assert payloads[-1]["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}


def test_stream_exception_before_first_event_is_structured_error():
    with _stream_client(_StreamRouter([], RuntimeError("upstream exploded"))) as client:
        response = _post_stream(client)
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "upstream exploded"


def test_stream_exception_after_delta_before_done_still_surfaces_error():
    # Native only publishes the finalized message, so deltas commit nothing.
    items = [{"type": "delta", "delta": {"text": "par"}}]
    with _stream_client(_StreamRouter(items, RuntimeError("mid-stream"))) as client:
        response = _post_stream(client)
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "mid-stream"


def test_stream_success_is_incremental_sse():
    with _stream_client(_StreamRouter([_done()])) as client:
        response = _post_stream(client)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [f for f in response.text.split("\n\n") if f]
    assert len(frames) > 2 and frames[-1] == "data: [DONE]"
    assert json.loads(frames[0][len("data: ") :])["choices"][0]["delta"] == {"content": "hello"}


def test_stream_error_after_first_frame_keeps_earlier_frames_intact():
    import asyncio

    from agency.harness.adapters.streaming import stream_response

    router = _StreamRouter([_done()], RuntimeError("late"))
    frames = []

    async def collect():
        with pytest.raises(RuntimeError, match="late"):
            async for frame in stream_response(
                router, "t", {}, "m", _backend()._format_agency_stream_to_harness
            ):
                frames.append(frame)

    asyncio.run(collect())
    assert json.loads(frames[0][len("data: ") :])["choices"][0]["delta"] == {"content": "hello"}
    assert frames[-1] == "data: [DONE]\n\n"


def test_harness_to_agency_keeps_unsigned_reasoning_content_as_unsigned_thinking():
    """Unsigned reasoning stays in history for backends that accept it; the
    Anthropic serializer drops it (see tests/llm/test_anthropic.py)."""
    context = _backend()._format_context_harness_to_agency(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello", "reasoning_content": "thoughts"},
            ]
        }
    )
    assert context["messages"][1]["blocks"] == [
        {"type": "thinking", "index": 0, "text": "thoughts"},
        {"type": "text", "index": 1, "text": "hello"},
    ]


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


def test_stream_error_after_keepalive_is_an_in_stream_error_event():
    import asyncio

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from agency.configs.agconfig import harnessadapterconfig
    from agency.harness.clients.host_services_client import HostDispatchError

    class SlowFailingRouter(_StreamRouter):
        async def dispatch_stream_async(self, token, request):
            await asyncio.sleep(0.15)
            raise HostDispatchError(
                {"message": "overloaded", "status_code": 503, "transient": True}
            )
            yield

    app = FastAPI()
    NativeAdapter(agconfig(harnessadapterconfig(stream_keepalive_s=0.05))).register(
        app, SlowFailingRouter([])
    )
    with TestClient(app) as client:
        response = _post_stream(client)
    assert response.status_code == 200
    frames = [f for f in response.text.split("\n\n") if f]
    assert frames[0] == ": keepalive"
    assert json.loads(frames[-1][len("data: ") :]) == {
        "error": {"message": "overloaded", "type": "upstream_error", "transient": True}
    }
