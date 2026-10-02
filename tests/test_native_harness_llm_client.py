import json

import httpx
import pytest

from agency.native_harness import llm_client


def client_with_responses(monkeypatch, responses, **kwargs):
    calls, delays, events = [], [], []

    def reply(request):
        calls.append(json.loads(request.content))
        return responses.pop(0)

    client = llm_client.LLMClient(
        "https://example.invalid",
        "unused",
        observer=lambda k, p: events.append({"kind": k, **p}),
        **kwargs,
    )
    client._client.close()
    client._client = httpx.Client(
        base_url="https://example.invalid", transport=httpx.MockTransport(reply)
    )
    monkeypatch.setattr(llm_client.time, "sleep", delays.append)
    monkeypatch.setattr(llm_client.random, "uniform", lambda low, high: low)
    return client, calls, delays, events


def success():
    chunk = {
        "choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1},
    }
    return httpx.Response(
        200, content=("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()
    )


def limited(code="rate_limit_exceeded", hint="2.5"):
    return httpx.Response(429, json={"error": {"code": code}}, headers={"retry-after": hint})


def test_rate_limit_retry_respects_server_hint_and_records_attempts(monkeypatch):
    client, calls, delays, events = client_with_responses(
        monkeypatch, [limited(), success()], retry_rate_limits=True
    )
    result = client.dispatch("fake", [], [])
    assert result["message"]["content"] == "done"
    assert result["usage"]["prompt_tokens"] == 10
    assert delays == [2.5]
    assert calls[0] == calls[1]
    assert len([e for e in events if e["kind"] == "model_attempt"]) == 2
    assert len([e for e in events if e["kind"] == "model_dispatch_complete"]) == 1
    client.close()


@pytest.mark.parametrize(
    "kwargs,code",
    [({}, "rate_limit_exceeded"), ({"retry_rate_limits": True}, "insufficient_quota")],
)
def test_default_and_quota_errors_are_not_retried(monkeypatch, kwargs, code):
    client, calls, delays, _ = client_with_responses(monkeypatch, [limited(code)], **kwargs)
    assert "429" in client.dispatch("fake", [])["error"]
    assert len(calls) == 1 and delays == []
    client.close()


def test_valid_hint_beyond_deadline_is_never_shortened(monkeypatch):
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: 100)
    client, calls, delays, _ = client_with_responses(
        monkeypatch, [limited(hint="1000")], retry_rate_limits=True, deadline=105
    )
    assert "deadline exhausted" in client.dispatch("fake", [])["error"]
    assert len(calls) == 1 and delays == []
    client.close()


def test_retry_attempts_are_bounded(monkeypatch):
    client, calls, delays, _ = client_with_responses(
        monkeypatch, [limited(), limited()], retry_rate_limits=True, max_attempts=2
    )
    assert "429" in client.dispatch("fake", [])["error"]
    assert len(calls) == 2 and delays == [2.5]
    client.close()


def test_retry_hint_milliseconds_and_http_date(monkeypatch):
    assert llm_client._server_retry_delay({"retry-after-ms": "900"}) == 0.9
    monkeypatch.setattr(llm_client.time, "time", lambda: 0)
    assert llm_client._server_retry_delay({"retry-after": "Thu, 01 Jan 1970 00:00:03 GMT"}) == 3
    assert llm_client._server_retry_delay({"retry-after": "nan"}) is None


def test_stream_failure_after_a_chunk_is_not_retried(monkeypatch):
    class Partial(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise httpx.ReadTimeout("stream interrupted")

    client, calls, delays, _ = client_with_responses(
        monkeypatch, [httpx.Response(200, stream=Partial())], retry_rate_limits=True
    )
    assert "stream interrupted" in client.dispatch("fake", [])["error"]
    assert len(calls) == 1 and delays == []
    client.close()


def test_direct_provider_does_not_receive_gateway_only_compaction_field(monkeypatch):
    client, calls, _, events = client_with_responses(
        monkeypatch, [success()], send_internal_kind=False
    )
    client.dispatch("fake", [], internal_kind="compaction")
    assert "agency_internal_kind" not in calls[0]
    assert events[-1]["internal_kind"] == "compaction"
    client.close()
