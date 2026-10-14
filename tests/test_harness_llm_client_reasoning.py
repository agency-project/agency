"""native_harness LLMClient keeps reasoning deltas on the
assistant message they return, so the next request sends them back."""

from __future__ import annotations

import json

import httpx
import pytest

from agency.native_harness.llm_client import LLMClient as NativeLLMClient


def _sse(deltas):
    lines = [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": d, "finish_reason": None}]})
        for d in deltas
    ]
    return "\n\n".join(lines + ["data: [DONE]"]) + "\n\n"


def _client(cls, body):
    client = cls("http://bridge", "token")
    client._client = httpx.Client(
        base_url="http://bridge",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body)),
    )
    return client


@pytest.mark.parametrize("cls", [NativeLLMClient])
def test_reasoning_deltas_kept_on_message(cls):
    body = _sse(
        [
            {"reasoning_content": "pl"},
            {"reasoning_content": "an"},
            {"reasoning_signature": "sig"},
            {"content": "answer"},
        ]
    )
    message = _client(cls, body).dispatch("m", [{"role": "user", "content": "q"}])["message"]
    assert message == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "plan",
        "reasoning_signature": "sig",
    }


@pytest.mark.parametrize("cls", [NativeLLMClient])
def test_no_reasoning_deltas_leaves_message_unchanged(cls):
    message = _client(cls, _sse([{"content": "answer"}])).dispatch(
        "m", [{"role": "user", "content": "q"}]
    )["message"]
    assert message == {"role": "assistant", "content": "answer"}


def _error_chunk(transient):
    error = {"message": "overloaded", "type": "upstream_error", "transient": transient}
    return ": keepalive\n\n" + "data: " + json.dumps({"error": error}) + "\n\n"


def _sequenced_client(bodies, monkeypatch):
    monkeypatch.setattr("agency.native_harness.llm_client.time.sleep", lambda _s: None)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, text=bodies[min(len(calls), len(bodies)) - 1])

    client = NativeLLMClient("http://bridge", "token")
    client._client = httpx.Client(base_url="http://bridge", transport=httpx.MockTransport(handler))
    return client, calls


def test_in_stream_error_event_is_returned_as_an_error(monkeypatch):
    client, calls = _sequenced_client([_error_chunk(transient=False)], monkeypatch)
    result = client.dispatch("m", [{"role": "user", "content": "hi"}])
    assert result == {"error": "dispatch failed: overloaded"}
    assert len(calls) == 1


def test_transient_in_stream_error_before_content_is_retried(monkeypatch):
    client, calls = _sequenced_client(
        [_error_chunk(transient=True), _sse([{"content": "answer"}])], monkeypatch
    )
    result = client.dispatch("m", [{"role": "user", "content": "hi"}])
    assert result["message"]["content"] == "answer"
    assert len(calls) == 2


def test_transient_in_stream_error_after_content_is_not_retried(monkeypatch):
    body = _sse([{"content": "par"}]).replace("data: [DONE]\n\n", "") + _error_chunk(True)
    client, calls = _sequenced_client([body, _sse([{"content": "answer"}])], monkeypatch)
    result = client.dispatch("m", [{"role": "user", "content": "hi"}])
    assert "error" in result
    assert len(calls) == 1
