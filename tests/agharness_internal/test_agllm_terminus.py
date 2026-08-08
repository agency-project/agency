"""Tests for agllm_terminus -- the host-side credential-holding dispatch
service agproxy_llm's routing/translation layer forwards to instead of
constructing a real backend client itself.

All tests use fastapi.testclient.TestClient against the app directly (no
real port bind, no real LLM backend) -- mirrors test_agproxy_llm.py.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
from openai.types.completion_usage import CompletionUsage

from agency.agharness_internal.agllm_terminus import agLLMTerminus
from agency.profiler import agprof


def _completion(content=None, finish_reason="stop", id="chatcmpl-test", model="m", usage=None):
    """A real, valid ChatCompletion -- agllm_terminus's dispatch route
    serializes these via duck-typed attribute access
    (_serialize_result/_serialize_chunk), NOT `.model_dump()`/
    `.model_dump_json()` (some backends' response objects don't implement
    those at all -- confirmed for Anthropic/Bedrock's lightweight
    __slots__-based compatibility objects during development), so a
    hand-rolled fake needs the real attribute surface, not just those two
    methods."""
    message = ChatCompletionMessage(role="assistant", content=content)
    choice = Choice(index=0, finish_reason=finish_reason, message=message)
    return ChatCompletion(
        id=id, object="chat.completion", created=0, model=model, choices=[choice], usage=usage
    )


def _chunk(content=None, finish_reason=None, id="chatcmpl-test", model="m", usage=None):
    delta = ChoiceDelta(content=content)
    choice = ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)
    return ChatCompletionChunk(
        id=id, object="chat.completion.chunk", created=0, model=model, choices=[choice], usage=usage
    )


class _RecordingSpan:
    def __init__(self):
        self.metadata = {}
        self.exit_info = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.exit_info = exc_info

    def annotate(self, **metadata):
        self.metadata.update(metadata)


def _make_terminus_with_agent(token="tok", stream_result=None, single_result=None):
    term = agLLMTerminus()
    fake_client = MagicMock()
    if stream_result is not None:
        fake_client.chat.completions.create.return_value = stream_result
    else:
        fake_client.chat.completions.create.return_value = single_result or _completion(
            content="hi"
        )
    fake_ag = MagicMock()
    fake_ag.llm.backend.make_client.return_value = fake_client
    term.register(token, fake_ag)
    return term, fake_ag, fake_client


def _client_for(term):
    from fastapi.testclient import TestClient

    return TestClient(term._app)


def test_missing_token_returns_401():
    term, _, _ = _make_terminus_with_agent()
    client = _client_for(term)
    resp = client.post("/internal/dispatch", json={"kwargs": {"model": "m", "messages": []}})
    assert resp.status_code == 401


def test_wrong_token_returns_401():
    term, _, _ = _make_terminus_with_agent(token="right-token")
    client = _client_for(term)
    resp = client.post(
        "/internal/dispatch",
        json={"token": "wrong-token", "kwargs": {"model": "m", "messages": []}},
    )
    assert resp.status_code == 401


def test_valid_token_non_streaming_dispatch():
    term, ag, fake_client = _make_terminus_with_agent(
        token="tok", single_result=_completion(content="hi", id="abc")
    )
    client = _client_for(term)
    kwargs = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False}
    resp = client.post("/internal/dispatch", json={"token": "tok", "kwargs": kwargs})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["id"] == "abc"
    assert payload["choices"][0]["message"]["content"] == "hi"
    # kwargs forwarded verbatim to the real backend client -- the terminus
    # itself does no reshaping, that's agproxy_llm's job.
    fake_client.chat.completions.create.assert_called_once_with(**kwargs)
    # The real client was constructed from THIS agent's own backend -- the
    # one place credentials are actually touched.
    ag.llm.backend.make_client.assert_called_once()


def test_dispatch_records_request_log_entry():
    """The one place a caller can prove a real credentialed dispatch
    happened, regardless of which process/engine routed the request here
    -- unlike agProxyLLM's own request_log, which stops being host-side
    readable once that routing layer runs inside the container."""
    term, _, _ = _make_terminus_with_agent(token="tok", single_result=_completion(content="hi"))
    client = _client_for(term)
    assert term.request_log == []
    kwargs = {"model": "claude-x", "messages": [], "stream": False}
    client.post("/internal/dispatch", json={"token": "tok", "kwargs": kwargs})
    assert len(term.request_log) == 1
    assert term.request_log[0] == {"token": "tok", "model": "claude-x"}


def test_dispatch_does_not_record_unauthenticated_calls():
    term, _, _ = _make_terminus_with_agent(token="tok")
    client = _client_for(term)
    client.post(
        "/internal/dispatch", json={"token": "wrong", "kwargs": {"model": "m", "messages": []}}
    )
    assert term.request_log == []


def test_valid_token_streaming_dispatch():
    chunks = [_chunk(content="a"), _chunk(content="b", finish_reason="stop")]
    term, ag, fake_client = _make_terminus_with_agent(token="tok", stream_result=chunks)
    client = _client_for(term)
    kwargs = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    resp = client.post("/internal/dispatch", json={"token": "tok", "kwargs": kwargs})
    assert resp.status_code == 200
    lines = [l for l in resp.text.split("\n\n") if l.strip()]
    assert json.loads(lines[0][len("data: ") :])["choices"][0]["delta"]["content"] == "a"
    assert json.loads(lines[1][len("data: ") :])["choices"][0]["delta"]["content"] == "b"
    assert lines[2] == "data: [DONE]"


def test_streaming_dispatch_profiles_ttft_final_usage_and_backend(monkeypatch):
    usage = CompletionUsage(prompt_tokens=11, completion_tokens=4, total_tokens=15)
    chunks = [_chunk(content="a"), _chunk(content=None, finish_reason="stop", usage=usage)]
    term, ag, fake_client = _make_terminus_with_agent(token="tok", stream_result=chunks)
    client = _client_for(term)
    span = _RecordingSpan()
    monkeypatch.setattr(
        "agency.agharness_internal.agllm_terminus.agprof.span",
        lambda name: span if name == "llm:attempt[0]" else None,
    )

    kwargs = {"model": "profiled-model", "messages": [], "stream": True}
    resp = client.post("/internal/dispatch", json={"token": "tok", "kwargs": kwargs})

    assert resp.status_code == 200
    assert span.exit_info == (None, None, None)
    assert span.metadata["outcome"] == "success"
    assert span.metadata["model"] == "profiled-model"
    assert span.metadata["provider"] == type(ag.llm.backend).__name__
    assert span.metadata["input_tokens"] == 11
    assert span.metadata["output_tokens"] == 4
    assert span.metadata["ttft_ms"] >= 0
    assert span.metadata["generation_ms"] >= 0
    forwarded = fake_client.chat.completions.create.call_args.kwargs
    assert forwarded["stream_options"] == {"include_usage": True}


def test_streaming_dispatch_populates_llm_summary(monkeypatch, tmp_path):
    usage = CompletionUsage(prompt_tokens=13, completion_tokens=5, total_tokens=18)
    chunks = [_chunk(content="a"), _chunk(content=None, finish_reason="stop", usage=usage)]
    term, _, _ = _make_terminus_with_agent(token="tok", stream_result=chunks)
    client = _client_for(term)
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        resp = client.post(
            "/internal/dispatch",
            json={
                "token": "tok",
                "kwargs": {"model": "summary-model", "messages": [], "stream": True},
            },
        )
        assert resp.status_code == 200

    summary = agprof.summary_metrics()
    assert summary["llm_metrics"]["calls"] == 1
    assert summary["llm_metrics"]["successful_calls"] == 1
    assert summary["llm_metrics"]["input_tokens"] == 13
    assert summary["llm_metrics"]["output_tokens"] == 5
    assert summary["llm_metrics"]["ttft"]["mean_ms"] is not None


def test_non_streaming_dispatch_profiles_usage_and_backend(monkeypatch):
    usage = CompletionUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10)
    term, ag, _ = _make_terminus_with_agent(
        token="tok", single_result=_completion(content="hi", usage=usage)
    )
    client = _client_for(term)
    span = _RecordingSpan()
    monkeypatch.setattr(
        "agency.agharness_internal.agllm_terminus.agprof.span",
        lambda name: span if name == "llm:attempt[0]" else None,
    )

    resp = client.post(
        "/internal/dispatch",
        json={"token": "tok", "kwargs": {"model": "m", "messages": [], "stream": False}},
    )

    assert resp.status_code == 200
    assert span.exit_info == (None, None, None)
    assert span.metadata == {
        "model": "m",
        "provider": type(ag.llm.backend).__name__,
        "outcome": "success",
        "input_tokens": 7,
        "output_tokens": 3,
    }


# ---------------------------------------------------------------------------
# Retry-policy classification -- see agllm_terminus.py's own module-level
# comment for why this terminus does exactly one attempt (no sleep, no
# loop) rather than retrying itself: retrying here, on top of whatever an
# external harness's own CLI already does, risks duplicate real provider
# calls with no coordination between the two layers. What it DOES do is
# force the first chunk before committing to any HTTP response, so a
# caller (native.py's own retry loop) gets an honest, immediate signal for
# whether the failure happened before any data was sent (retriable) or not.
# ---------------------------------------------------------------------------


def test_streaming_dispatch_transient_error_before_first_chunk_returns_503():
    import httpx
    import openai

    term, _, fake_client = _make_terminus_with_agent(token="tok")
    fake_client.chat.completions.create.side_effect = openai.APIConnectionError(
        request=httpx.Request("POST", "http://x")
    )
    client = _client_for(term)
    kwargs = {"model": "m", "messages": [], "stream": True}
    resp = client.post("/internal/dispatch", json={"token": "tok", "kwargs": kwargs})
    assert resp.status_code == 503
    payload = resp.json()
    assert payload["error"]["transient"] is True


def test_streaming_dispatch_bad_request_before_first_chunk_returns_400():
    import openai
    from unittest.mock import MagicMock

    term, _, fake_client = _make_terminus_with_agent(token="tok")
    fake_client.chat.completions.create.side_effect = openai.BadRequestError(
        message="invalid request", response=MagicMock(status_code=400), body=None
    )
    client = _client_for(term)
    kwargs = {"model": "m", "messages": [], "stream": True}
    resp = client.post("/internal/dispatch", json={"token": "tok", "kwargs": kwargs})
    assert resp.status_code == 400
    payload = resp.json()
    assert payload["error"]["transient"] is False


def test_streaming_dispatch_profiles_classified_failure(monkeypatch):
    import openai

    term, ag, fake_client = _make_terminus_with_agent(token="tok")
    fake_client.chat.completions.create.side_effect = openai.BadRequestError(
        message="invalid request", response=MagicMock(status_code=400), body=None
    )
    client = _client_for(term)
    span = _RecordingSpan()
    monkeypatch.setattr(
        "agency.agharness_internal.agllm_terminus.agprof.span",
        lambda name: span if name == "llm:attempt[0]" else None,
    )

    resp = client.post(
        "/internal/dispatch",
        json={"token": "tok", "kwargs": {"model": "m", "messages": [], "stream": True}},
    )

    assert resp.status_code == 400
    assert span.exit_info[0] is openai.BadRequestError
    assert span.metadata == {
        "model": "m",
        "provider": type(ag.llm.backend).__name__,
        "outcome": "failure",
        "error_type": "BadRequestError",
        "status_code": 400,
        "transient": False,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def test_streaming_dispatch_rate_limit_before_first_chunk_returns_503():
    import openai
    from unittest.mock import MagicMock

    term, _, fake_client = _make_terminus_with_agent(token="tok")
    fake_client.chat.completions.create.side_effect = openai.RateLimitError(
        message="rate limited", response=MagicMock(status_code=429), body=None
    )
    client = _client_for(term)
    kwargs = {"model": "m", "messages": [], "stream": True}
    resp = client.post("/internal/dispatch", json={"token": "tok", "kwargs": kwargs})
    assert resp.status_code == 503


def test_streaming_dispatch_empty_stream_returns_clean_done_not_an_error():
    term, _, fake_client = _make_terminus_with_agent(token="tok", stream_result=[])
    client = _client_for(term)
    kwargs = {"model": "m", "messages": [], "stream": True}
    resp = client.post("/internal/dispatch", json={"token": "tok", "kwargs": kwargs})
    assert resp.status_code == 200
    assert resp.text.strip() == "data: [DONE]"


def test_unregister_removes_token():
    term, _, _ = _make_terminus_with_agent(token="tok")
    term.unregister("tok")
    client = _client_for(term)
    resp = client.post(
        "/internal/dispatch", json={"token": "tok", "kwargs": {"model": "m", "messages": []}}
    )
    assert resp.status_code == 401


def test_validate_token_true_for_registered_token():
    term, _, _ = _make_terminus_with_agent(token="tok")
    client = _client_for(term)
    resp = client.post("/internal/validate_token", json={"token": "tok"})
    assert resp.status_code == 200
    assert resp.json() == {"valid": True}


def test_validate_token_false_for_unknown_token():
    term, _, _ = _make_terminus_with_agent(token="tok")
    client = _client_for(term)
    resp = client.post("/internal/validate_token", json={"token": "nope"})
    assert resp.status_code == 200
    assert resp.json() == {"valid": False}


def test_validate_token_false_after_unregister():
    term, _, _ = _make_terminus_with_agent(token="tok")
    term.unregister("tok")
    client = _client_for(term)
    resp = client.post("/internal/validate_token", json={"token": "tok"})
    assert resp.json() == {"valid": False}


# ---------------------------------------------------------------------------
# resolve_model / log_warning / check_tool_policy -- let agproxy_llm.py's
# routing/translation layer act on an agent's model/logging/policy without
# holding a live `ag` reference itself, the prerequisite for that layer to
# run somewhere `ag` isn't reachable at all (e.g. inside the container).
# ---------------------------------------------------------------------------


def test_resolve_model_returns_the_agents_configured_model():
    term, ag, _ = _make_terminus_with_agent(token="tok")
    ag.llm.backend.model = "configured-model"
    client = _client_for(term)
    resp = client.post("/internal/resolve_model", json={"token": "tok"})
    assert resp.status_code == 200
    assert resp.json()["model"] == "configured-model"


def test_resolve_model_unknown_token_returns_401():
    term, _, _ = _make_terminus_with_agent()
    client = _client_for(term)
    resp = client.post("/internal/resolve_model", json={"token": "nope"})
    assert resp.status_code == 401


def test_context_limit_returns_the_fetched_value():
    """Lets native.py's entrypoint (which can't import agllm.py at all)
    learn its model's context window for its own compaction -- reuses
    agllm.fetch_context_limit's real lookup, not a second guess."""
    term, ag, _ = _make_terminus_with_agent(token="tok")
    client = _client_for(term)
    with patch("agency.agllm.agllm.fetch_context_limit", return_value=128_000) as mock_fetch:
        resp = client.post("/internal/context_limit", json={"token": "tok"})
    assert resp.status_code == 200
    assert resp.json()["context_limit"] == 128_000
    mock_fetch.assert_called_once_with(ag.llm.backend)


def test_context_limit_unknown_token_returns_401():
    term, _, _ = _make_terminus_with_agent()
    client = _client_for(term)
    resp = client.post("/internal/context_limit", json={"token": "nope"})
    assert resp.status_code == 401


def test_log_warning_calls_the_agents_terminal():
    term, ag, _ = _make_terminus_with_agent(token="tok")
    client = _client_for(term)
    resp = client.post("/internal/log_warning", json={"token": "tok", "message": "careful"})
    assert resp.status_code == 200
    ag.terminal.log.assert_called_once_with("WARNING  ", "careful")


def test_log_warning_unknown_token_returns_401():
    term, _, _ = _make_terminus_with_agent()
    client = _client_for(term)
    resp = client.post("/internal/log_warning", json={"token": "nope", "message": "x"})
    assert resp.status_code == 401


def test_check_tool_policy_delegates_to_agharness_default_policy():
    from agency.agpolicy import agdecision

    term, ag, _ = _make_terminus_with_agent(token="tok")
    client = _client_for(term)
    with patch("agency.agharness.default_policy") as mock_default_policy:
        mock_policy = MagicMock()
        mock_policy.check_tool.return_value = agdecision.deny("blocked for testing")
        mock_default_policy.return_value = mock_policy

        resp = client.post(
            "/internal/check_tool_policy",
            json={"token": "tok", "tool_name": "bash", "tool_input": {"command": "rm -rf /"}},
        )
        assert resp.status_code == 200
        assert resp.json() == {"decision": "deny", "reason": "blocked for testing"}
        mock_policy.check_tool.assert_called_once_with(ag, "bash", {"command": "rm -rf /"})


def test_check_tool_policy_unknown_token_returns_401():
    term, _, _ = _make_terminus_with_agent()
    client = _client_for(term)
    resp = client.post(
        "/internal/check_tool_policy",
        json={"token": "nope", "tool_name": "bash", "tool_input": {}},
    )
    assert resp.status_code == 401
    assert resp.json()["decision"] == "deny"


# ---------------------------------------------------------------------------
# Duck-typed serialization -- must work for backends whose response objects
# don't implement .model_dump()/.model_dump_json() at all (confirmed real
# gap: Anthropic/Bedrock's lightweight __slots__-based compatibility
# objects, agllm_backends/anthropic.py's _AnthropicNonStreamResponse/
# _FakeChunk family).
# ---------------------------------------------------------------------------


class _SlotsMessage:
    __slots__ = ("content", "tool_calls")

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _SlotsChoice:
    __slots__ = ("message",)

    def __init__(self, message):
        self.message = message
        # Deliberately no .index/.finish_reason -- some backends' fakes
        # don't set these either; _serialize_result must tolerate that via
        # getattr defaults, not crash.


class _SlotsResult:
    """No .model_dump() at all -- mirrors _AnthropicNonStreamResponse's
    real shape exactly (a plain object exposing only .choices)."""

    __slots__ = ("choices",)

    def __init__(self, choices):
        self.choices = choices


def test_non_streaming_dispatch_works_without_model_dump():
    result = _SlotsResult([_SlotsChoice(_SlotsMessage(content="no pydantic here"))])
    term, ag, fake_client = _make_terminus_with_agent(token="tok", single_result=result)
    client = _client_for(term)
    kwargs = {"model": "m", "messages": [], "stream": False}
    resp = client.post("/internal/dispatch", json={"token": "tok", "kwargs": kwargs})
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "no pydantic here"


class _SlotsDelta:
    __slots__ = ("content", "tool_calls")

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _SlotsChunkChoice:
    __slots__ = ("delta",)

    def __init__(self, delta):
        self.delta = delta


class _SlotsChunk:
    """No .model_dump_json() at all -- mirrors the Anthropic backend's
    real streaming _FakeChunk shape."""

    __slots__ = ("choices", "usage")

    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


def test_streaming_dispatch_works_without_model_dump_json():
    chunks = [_SlotsChunk([_SlotsChunkChoice(_SlotsDelta(content="no pydantic"))])]
    term, ag, fake_client = _make_terminus_with_agent(token="tok", stream_result=chunks)
    client = _client_for(term)
    kwargs = {"model": "m", "messages": [], "stream": True}
    resp = client.post("/internal/dispatch", json={"token": "tok", "kwargs": kwargs})
    assert resp.status_code == 200
    lines = [l for l in resp.text.split("\n\n") if l.strip()]
    assert json.loads(lines[0][len("data: ") :])["choices"][0]["delta"]["content"] == "no pydantic"
    assert lines[-1] == "data: [DONE]"


# ---------------------------------------------------------------------------
# Real socket tests -- proves genuine over-the-wire reachability (TCP and
# UDS), not just in-process ASGI dispatch via TestClient above.
# ---------------------------------------------------------------------------


def test_real_tcp_roundtrip():
    import httpx

    term, ag, fake_client = _make_terminus_with_agent(
        token="tok", single_result=_completion(content="hi", id="real")
    )
    try:
        base_url = term.start()
        assert base_url.startswith("http://127.0.0.1:")
        resp = httpx.post(
            f"{base_url}/internal/dispatch",
            json={"token": "tok", "kwargs": {"model": "m", "messages": [], "stream": False}},
            timeout=10,
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["id"] == "real"
        assert payload["choices"][0]["message"]["content"] == "hi"
    finally:
        term.stop()


def test_real_uds_roundtrip():
    import httpx

    term, ag, fake_client = _make_terminus_with_agent(
        token="tok", single_result=_completion(content="hi", id="real-uds")
    )
    try:
        sock_path = term.ensure_uds_started()
        transport = httpx.HTTPTransport(uds=sock_path)
        with httpx.Client(transport=transport, base_url="http://uds") as client:
            resp = client.post(
                "/internal/dispatch",
                json={"token": "tok", "kwargs": {"model": "m", "messages": [], "stream": False}},
                timeout=10,
            )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["id"] == "real-uds"
        assert payload["choices"][0]["message"]["content"] == "hi"
    finally:
        term.stop_uds()
