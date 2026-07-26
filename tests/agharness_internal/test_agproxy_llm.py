"""Tests for agproxy_llm -- the local HTTP gateway routing a harness's LLM
traffic to the launching agent's own agllm backend.

All tests use fastapi.testclient.TestClient against the app directly (no
real port bind, no real LLM backend) -- see docs/agproxy_llm.md.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from agency.agharness_internal.agproxy_llm import agProxyLLM, agProxyLLMConfig


class _FakeChunk:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload

    def model_dump_json(self):
        return json.dumps(self._payload)


def _make_gateway_with_agent(token="tok", stream_result=None, single_result=None):
    px = agProxyLLM()
    fake_client = MagicMock()
    if stream_result is not None:
        fake_client.chat.completions.create.return_value = stream_result
    else:
        fake_client.chat.completions.create.return_value = single_result or _FakeChunk(
            {"id": "x", "choices": [{"message": {"content": "hi"}}]}
        )
    fake_ag = MagicMock()
    fake_ag.llm.backend.make_client.return_value = fake_client
    fake_ag.llm.backend.model = "configured-model"
    px.register(token, fake_ag)
    return px, fake_ag, fake_client


def _client_for(px):
    from fastapi.testclient import TestClient

    return TestClient(px._app)


def test_missing_token_returns_401():
    px, _, _ = _make_gateway_with_agent()
    client = _client_for(px)
    resp = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert resp.status_code == 401


def test_wrong_token_returns_401():
    px, _, _ = _make_gateway_with_agent(token="right-token")
    client = _client_for(px)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": []},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_valid_token_non_streaming_passthrough():
    px, ag, fake_client = _make_gateway_with_agent(
        token="tok", single_result=_FakeChunk({"id": "abc", "choices": []})
    )
    client = _client_for(px)
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False}
    resp = client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": "Bearer tok"}
    )
    assert resp.status_code == 200
    assert resp.json() == {"id": "abc", "choices": []}
    # The request body is forwarded verbatim -- genuine passthrough, no reshaping.
    fake_client.chat.completions.create.assert_called_once_with(**body)


def test_valid_token_streaming_passthrough():
    chunks = [_FakeChunk({"n": 1}), _FakeChunk({"n": 2})]
    px, ag, fake_client = _make_gateway_with_agent(token="tok", stream_result=chunks)
    client = _client_for(px)
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    resp = client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": "Bearer tok"}
    )
    assert resp.status_code == 200
    lines = [l for l in resp.text.split("\n\n") if l.strip()]
    assert lines[0] == 'data: {"n": 1}'
    assert lines[1] == 'data: {"n": 2}'
    assert lines[2] == "data: [DONE]"


def test_unregister_removes_agent_access():
    px, ag, _ = _make_gateway_with_agent(token="tok")
    px.unregister("tok")
    client = _client_for(px)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": []},
        headers={"Authorization": "Bearer tok"},
    )
    assert resp.status_code == 401


def test_x_api_key_header_also_accepted():
    px, ag, fake_client = _make_gateway_with_agent(token="tok")
    client = _client_for(px)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [], "stream": False},
        headers={"x-api-key": "tok"},
    )
    assert resp.status_code == 200


def test_start_returns_real_bound_url_and_stop_is_idempotent():
    px = agProxyLLM()
    url = px.start()
    assert url.startswith("http://127.0.0.1:")
    assert url == px.start()  # idempotent
    px.stop()
    px.stop()  # idempotent, must not raise
    assert px.base_url is None


class _Fn:
    def __init__(self, name="", arguments=""):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id="", name="", arguments="", index=0):
        self.id = id
        self.function = _Fn(name, arguments)
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
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


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


def test_anthropic_messages_route_missing_token_returns_401():
    px, _, _ = _make_gateway_with_agent()
    client = _client_for(px)
    resp = client.post("/v1/messages", json={"model": "m", "messages": []})
    assert resp.status_code == 401
    assert resp.json()["type"] == "error"


def test_anthropic_messages_route_non_streaming_translates_request_and_response():
    px, ag, fake_client = _make_gateway_with_agent(token="tok")
    fake_client.chat.completions.create.return_value = _Response(
        [_Choice(message=_Message(content="hi there"), finish_reason="stop")], usage=_Usage(5, 3)
    )
    client = _client_for(px)
    body = {
        "model": "claude-x",
        "system": "be helpful",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    resp = client.post("/v1/messages", json=body, headers={"Authorization": "Bearer tok"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["type"] == "message"
    assert payload["content"] == [{"type": "text", "text": "hi there"}]
    assert payload["stop_reason"] == "end_turn"
    assert payload["usage"] == {"input_tokens": 5, "output_tokens": 3}

    # the underlying client was called with reshaped OpenAI kwargs, not the
    # raw Anthropic body -- genuine translation, not passthrough.
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"][0] == {"role": "system", "content": "be helpful"}
    assert call_kwargs["messages"][1] == {"role": "user", "content": "hello"}
    assert call_kwargs["max_tokens"] == 100
    # The agent's OWN configured model is used, never the harness's own
    # request body model -- regression test for a real bug hit against the
    # live `claude` CLI: Claude Code's default model id has no reason to
    # match this agent's configured backend model (e.g. a Bedrock
    # inference-profile id), so trusting the harness's choice 400s against
    # the real backend instead of routing through agency's own config.
    assert call_kwargs["model"] == "configured-model"
    assert call_kwargs["model"] != body["model"]


def test_anthropic_messages_route_streaming_returns_anthropic_sse():
    px, ag, fake_client = _make_gateway_with_agent(token="tok")
    fake_client.chat.completions.create.return_value = [
        _Chunk(choices=[_Choice(delta=_Delta(content="hi"))]),
        _Chunk(choices=[_Choice(delta=_Delta(), finish_reason="stop")]),
        _Chunk(usage=_Usage(prompt_tokens=1, completion_tokens=1)),
    ]
    client = _client_for(px)
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    resp = client.post("/v1/messages", json=body, headers={"Authorization": "Bearer tok"})
    assert resp.status_code == 200
    assert "event: message_start" in resp.text
    assert "event: content_block_delta" in resp.text
    assert "event: message_stop" in resp.text


def test_anthropic_count_tokens_route_returns_estimate():
    px, _, _ = _make_gateway_with_agent(token="tok")
    client = _client_for(px)
    body = {"system": "abcd", "messages": [{"role": "user", "content": "abcdefgh"}]}
    resp = client.post(
        "/v1/messages/count_tokens", json=body, headers={"Authorization": "Bearer tok"}
    )
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] >= 1


def test_anthropic_count_tokens_route_missing_token_returns_401():
    px, _, _ = _make_gateway_with_agent()
    client = _client_for(px)
    resp = client.post("/v1/messages/count_tokens", json={"messages": []})
    assert resp.status_code == 401


def test_openai_responses_route_missing_token_returns_401():
    px, _, _ = _make_gateway_with_agent()
    client = _client_for(px)
    resp = client.post("/v1/responses", json={"model": "m", "input": "hi"})
    assert resp.status_code == 401


def test_openai_responses_route_non_streaming_translates_request_and_response():
    px, ag, fake_client = _make_gateway_with_agent(token="tok")
    fake_client.chat.completions.create.return_value = _Response(
        [_Choice(message=_Message(content="the answer is 4"))], usage=_Usage(3, 4)
    )
    client = _client_for(px)
    body = {"model": "m", "instructions": "be terse", "input": "what is 2+2?", "stream": False}
    resp = client.post("/v1/responses", json=body, headers={"Authorization": "Bearer tok"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "completed"
    assert payload["output"][0]["content"] == [
        {"type": "output_text", "text": "the answer is 4", "annotations": []}
    ]

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "what is 2+2?"},
    ]
    assert call_kwargs["model"] == "configured-model"


def test_openai_responses_route_streaming_returns_responses_sse():
    px, ag, fake_client = _make_gateway_with_agent(token="tok")
    fake_client.chat.completions.create.return_value = [
        _Chunk(choices=[_Choice(delta=_Delta(content="hi"))]),
        _Chunk(choices=[_Choice(delta=_Delta(), finish_reason="stop")]),
    ]
    client = _client_for(px)
    body = {"model": "m", "input": "hi", "stream": True}
    resp = client.post("/v1/responses", json=body, headers={"Authorization": "Bearer tok"})
    assert resp.status_code == 200
    assert "event: response.created" in resp.text
    assert "event: response.completed" in resp.text


def test_request_log_records_authenticated_chat_completions_calls():
    px, _, _ = _make_gateway_with_agent(token="tok")
    client = _client_for(px)
    assert px.request_log == []
    client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [], "stream": False},
        headers={"Authorization": "Bearer tok"},
    )
    assert len(px.request_log) == 1
    assert px.request_log[0]["route"] == "/v1/chat/completions"
    assert px.request_log[0]["token"] == "tok"


def test_request_log_does_not_record_unauthenticated_calls():
    px, _, _ = _make_gateway_with_agent(token="tok")
    client = _client_for(px)
    client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert px.request_log == []


def test_request_log_records_anthropic_messages_calls():
    px, ag, fake_client = _make_gateway_with_agent(token="tok")
    fake_client.chat.completions.create.return_value = _Response(
        [_Choice(message=_Message(content="hi"), finish_reason="stop")], usage=_Usage(1, 1)
    )
    client = _client_for(px)
    client.post(
        "/v1/messages",
        json={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        headers={"Authorization": "Bearer tok"},
    )
    assert len(px.request_log) == 1
    assert px.request_log[0]["route"] == "/v1/messages"
    assert px.request_log[0]["model"] == "configured-model"  # overridden, not "claude-x"


def test_request_log_records_responses_calls():
    px, ag, fake_client = _make_gateway_with_agent(token="tok")
    fake_client.chat.completions.create.return_value = _Response(
        [_Choice(message=_Message(content="hi"))], usage=_Usage(1, 1)
    )
    client = _client_for(px)
    client.post(
        "/v1/responses",
        json={"model": "m", "input": "hi", "stream": False},
        headers={"Authorization": "Bearer tok"},
    )
    assert len(px.request_log) == 1
    assert px.request_log[0]["route"] == "/v1/responses"


def test_agproxy_llm_config_view():
    # port is a DynamicConfigParam (per-instance, freely settable/re-settable);
    # bind_host/request_timeout_s are tier-1 GlobalConfigParams like other
    # process-wide timeout knobs in the codebase (e.g. agllm_backend's
    # model_listing_timeout_seconds) -- NOT exercised here with a real
    # framework owner, since other tests in this file already read
    # request_timeout_s via the route handler, permanently locking it
    # process-wide (see agconfig.md's tier-1 "write-once" semantics); a
    # config-view-vs-registry interaction test belongs in test_agconfig.py
    # against a uniquely-prefixed test-only owner, not here.
    from agency.agconfig import agConfig

    cfg = agConfig(agProxyLLMConfig(port=12345))
    assert cfg.get("agproxy_llm", "port") == 12345
