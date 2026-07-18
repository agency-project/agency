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
