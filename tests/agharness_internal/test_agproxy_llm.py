"""Tests for agproxy_llm -- the local HTTP gateway routing a harness's LLM
traffic to the launching agent's own agllm backend.

Most tests use fastapi.testclient.TestClient against the app directly (no
real port bind) -- see docs/agproxy_llm.md. Real backend credentials are
never touched here at all, though: every dispatch is forwarded to
agllm_terminus.agLLMTerminus over a genuine HTTP round-trip (started
lazily on first use, via `agProxyLLM._terminus_http_client()`) -- see the
"credential separation" section at the end of this file for tests that
verify this boundary directly, not just its externally-visible effect.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, call, patch

from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
from openai.types.completion_usage import CompletionUsage

from agency.agharness_internal.agproxy_llm import agProxyLLM, agProxyLLMConfig
from agency.agharness_internal import agproxy_llm


def _completion(content=None, finish_reason="stop", usage=None, id="chatcmpl-test", model="m"):
    """A real, valid `ChatCompletion` -- these now cross a genuine HTTP
    boundary (agproxy_llm -> agllm_terminus and back) and are reconstructed
    via `ChatCompletion.model_validate()` on the way back, so a hand-rolled
    fake lacking real pydantic fields/methods no longer round-trips. See
    agproxy_llm.py's `_dispatch()`."""
    message = ChatCompletionMessage(role="assistant", content=content)
    choice = Choice(index=0, finish_reason=finish_reason, message=message)
    return ChatCompletion(
        id=id,
        object="chat.completion",
        created=0,
        model=model,
        choices=[choice],
        usage=CompletionUsage(
            prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=sum(usage)
        )
        if usage
        else None,
    )


def _chunk(
    content=None, finish_reason=None, usage=None, id="chatcmpl-test", model="m", has_choice=True
):
    """A real, valid `ChatCompletionChunk` -- same reconstruction reasoning
    as `_completion` above."""
    choices = []
    if has_choice:
        delta = ChoiceDelta(content=content)
        choices = [ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)]
    return ChatCompletionChunk(
        id=id,
        object="chat.completion.chunk",
        created=0,
        model=model,
        choices=choices,
        usage=CompletionUsage(
            prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=sum(usage)
        )
        if usage
        else None,
    )


def _make_gateway_with_agent(token="tok", stream_result=None, single_result=None):
    px = agProxyLLM()
    fake_client = MagicMock()
    if stream_result is not None:
        fake_client.chat.completions.create.return_value = stream_result
    else:
        fake_client.chat.completions.create.return_value = single_result or _completion(
            content="hi"
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
        token="tok", single_result=_completion(content="hi", id="abc", finish_reason="stop")
    )
    client = _client_for(px)
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False}
    resp = client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer tok"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["id"] == "abc"
    assert payload["choices"][0]["message"]["content"] == "hi"
    # The request body is forwarded verbatim -- genuine passthrough, no reshaping.
    fake_client.chat.completions.create.assert_called_once_with(**body)


def test_valid_token_streaming_passthrough():
    chunks = [_chunk(content="a"), _chunk(content="b", finish_reason="stop")]
    px, ag, fake_client = _make_gateway_with_agent(token="tok", stream_result=chunks)
    client = _client_for(px)
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    resp = client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer tok"})
    assert resp.status_code == 200
    lines = [l for l in resp.text.split("\n\n") if l.strip()]
    assert json.loads(lines[0][len("data: ") :])["choices"][0]["delta"]["content"] == "a"
    assert json.loads(lines[1][len("data: ") :])["choices"][0]["delta"]["content"] == "b"
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


def test_anthropic_messages_route_missing_token_returns_401():
    px, _, _ = _make_gateway_with_agent()
    client = _client_for(px)
    resp = client.post("/v1/messages", json={"model": "m", "messages": []})
    assert resp.status_code == 401
    assert resp.json()["type"] == "error"


def test_anthropic_messages_route_non_streaming_translates_request_and_response():
    px, ag, fake_client = _make_gateway_with_agent(token="tok")
    fake_client.chat.completions.create.return_value = _completion(
        content="hi there", finish_reason="stop", usage=(5, 3)
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
    assert call_kwargs["max_completion_tokens"] == 100
    assert "max_tokens" not in call_kwargs
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
        _chunk(content="hi"),
        _chunk(finish_reason="stop"),
        _chunk(usage=(1, 1), has_choice=False),
    ]
    client = _client_for(px)
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    resp = client.post("/v1/messages", json=body, headers={"Authorization": "Bearer tok"})
    assert resp.status_code == 200
    assert "event: message_start" in resp.text
    assert "event: content_block_delta" in resp.text
    assert "event: message_stop" in resp.text


def test_anthropic_messages_route_warns_on_mid_array_system_message_via_terminus():
    """The warning must reach the real agent's terminal through the
    terminus's /internal/log_warning route, not a direct ag.terminal.log
    call -- proving _warn_mid_array_system_messages no longer needs a live
    `ag` reference in this process at all."""
    px, ag, fake_client = _make_gateway_with_agent(token="tok")
    fake_client.chat.completions.create.return_value = _completion(content="hi")
    client = _client_for(px)
    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "unexpected mid-array system message"},
        ],
        "stream": False,
    }
    resp = client.post("/v1/messages", json=body, headers={"Authorization": "Bearer tok"})
    assert resp.status_code == 200
    ag.terminal.log.assert_called_once()
    args, _ = ag.terminal.log.call_args
    assert args[0] == "WARNING  "
    assert "mid-conversation system-role message" in args[1]


def test_agpolicy_check_tool_missing_token_returns_401():
    px, _, _ = _make_gateway_with_agent()
    client = _client_for(px)
    resp = client.post(
        "/agpolicy/check_tool", json={"tool_name": "bash", "tool_input": {"command": "ls"}}
    )
    assert resp.status_code == 401
    assert resp.json()["decision"] == "deny"


def test_agpolicy_check_tool_delegates_through_terminus_to_default_policy():
    from unittest.mock import patch

    from agency.agpolicy import agdecision

    px, ag, _ = _make_gateway_with_agent(token="tok")
    client = _client_for(px)
    with patch("agency.agharness.default_policy") as mock_default_policy:
        mock_policy = MagicMock()
        mock_policy.check_tool.return_value = agdecision.deny("blocked for testing")
        mock_default_policy.return_value = mock_policy

        resp = client.post(
            "/agpolicy/check_tool",
            json={"tool_name": "bash", "tool_input": {"command": "rm -rf /"}},
            headers={"Authorization": "Bearer tok"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"decision": "deny", "reason": "blocked for testing"}
        mock_policy.check_tool.assert_called_once_with(ag, "bash", {"command": "rm -rf /"})


def test_agprof_hook_authenticates_then_delegates_to_profiler_bridge():
    px, _, _ = _make_gateway_with_agent(token="profiler-secret")

    def forward_to_ingest(token, _event):
        if token != "profiler-secret":
            return {"ok": False, "error": "unknown or missing token"}
        return {"ok": True}

    forward = MagicMock(side_effect=forward_to_ingest)
    px._forward_profiler_hook = forward
    client = _client_for(px)
    event = {
        "token": "body-token-must-not-authenticate",
        "hook_event_name": "PreToolUse",
        "perf_ns": time.perf_counter_ns(),
        "wall_ns": time.time_ns(),
        "payload": {"tool_use_id": "tool-bridge", "tool_name": "Read", "tool_input": {}},
    }

    assert client.post("/agprof/hook", json=event).status_code == 401
    response = client.post(
        "/agprof/hook",
        json=event,
        headers={"Authorization": "Bearer profiler-secret"},
    )

    assert response.json() == {"ok": True}
    assert (
        client.post(
            "/agprof/hook",
            json=event,
            headers={"Authorization": "Bearer wrong-token"},
        ).status_code
        == 401
    )
    assert forward.call_args_list == [
        call("profiler-secret", event),
        call("wrong-token", event),
    ]

    oversized = client.post(
        "/agprof/hook",
        content=b"{" + b"x" * (256 * 1024),
        headers={"Authorization": "Bearer profiler-secret"},
    )
    assert oversized.status_code == 413
    assert oversized.json()["error"] == "profiler hook body too large"
    assert forward.call_count == 2


def test_agprof_hook_saturation_rejects_before_queuing_auth_or_forward():
    px, _, _ = _make_gateway_with_agent(token="profiler-secret")
    px._token_valid = MagicMock(return_value=True)
    px._forward_profiler_hook = MagicMock(return_value={"ok": True})
    px._profiler_forward_slots = MagicMock()
    px._profiler_forward_slots.acquire.return_value = False
    client = _client_for(px)

    response = client.post(
        "/agprof/hook",
        json={"hook_event_name": "PreToolUse"},
        headers={"Authorization": "Bearer profiler-secret"},
    )

    assert response.status_code == 429
    assert response.json()["error"] == "profiler hook bridge saturated"
    px._token_valid.assert_not_called()
    px._forward_profiler_hook.assert_not_called()
    px._profiler_forward_slots.release.assert_not_called()


def test_agprof_hook_rate_limit_rejects_valid_token_before_body_forward():
    px, _, _ = _make_gateway_with_agent(token="profiler-secret")
    px._token_valid = MagicMock(return_value=True)
    px._profiler_hook_rate_allowed = MagicMock(return_value=False)
    px._forward_profiler_hook = MagicMock(return_value={"ok": True})
    client = _client_for(px)

    response = client.post(
        "/agprof/hook",
        json={"hook_event_name": "PreToolUse"},
        headers={"Authorization": "Bearer profiler-secret"},
    )

    assert response.status_code == 429
    assert response.json()["error"] == "profiler hook rate limit exceeded"
    px._profiler_hook_rate_allowed.assert_called_once_with("profiler-secret")
    px._forward_profiler_hook.assert_not_called()


def test_profiler_hook_rate_window_and_token_state_are_bounded():
    px = agProxyLLM(terminus=MagicMock())
    with (
        patch.object(agproxy_llm, "_MAX_PROFILER_HOOK_EVENTS_PER_TOKEN_PER_SECOND", 2),
        patch.object(agproxy_llm, "_MAX_PROFILER_RATE_TOKENS", 2),
    ):
        assert px._profiler_hook_rate_allowed("a", now=10.0)
        assert px._profiler_hook_rate_allowed("a", now=10.1)
        assert not px._profiler_hook_rate_allowed("a", now=10.2)
        assert px._profiler_hook_rate_allowed("a", now=11.0)
        assert px._profiler_hook_rate_allowed("b", now=11.0)
        assert px._profiler_hook_rate_allowed("c", now=11.0)

    assert len(px._profiler_rate_windows) == 2
    assert set(px._profiler_rate_windows) == {"b", "c"}


def test_profiler_bridge_uses_separate_framed_uds_and_overrides_body_token():
    px = agProxyLLM(terminus=MagicMock(), profiler_uds_path="/bridge/agprof-ingest.sock")
    sock = MagicMock()
    responses = [
        {"ok": True, "host_wall_ns": time.time_ns(), "host_perf_ns": time.perf_counter_ns()},
        {"ok": True},
        {"ok": True},
    ]
    event = {
        "token": "untrusted-body-token",
        "hook_event_name": "PreToolUse",
        "wall_ns": time.time_ns(),
        "perf_ns": time.perf_counter_ns(),
        "payload": {"tool_use_id": "tool-1", "tool_name": "Bash", "tool_input": {}},
    }
    with (
        patch("agency.agharness_internal.agproxy_llm.socket.socket", return_value=sock),
        patch("agency.profiler.agprof_emit._send_framed") as send,
        patch("agency.profiler.agprof_emit._recv_framed", side_effect=responses),
    ):
        assert px._forward_profiler_hook("header-secret", event) == {"ok": True}

    sock.connect.assert_called_once_with("/bridge/agprof-ingest.sock")
    sock.settimeout.assert_called_once_with(agproxy_llm._PROFILER_FORWARD_TIMEOUT_S)
    forwarded = send.call_args_list[-1].args[1]
    assert forwarded["ev"] == "hook"
    assert forwarded["token"] == "header-secret"
    assert "untrusted-body-token" not in json.dumps(forwarded)


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
    fake_client.chat.completions.create.return_value = _completion(
        content="the answer is 4", usage=(3, 4)
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
        _chunk(content="hi"),
        _chunk(finish_reason="stop"),
    ]
    client = _client_for(px)
    body = {"model": "m", "input": "hi", "stream": True}
    resp = client.post("/v1/responses", json=body, headers={"Authorization": "Bearer tok"})
    assert resp.status_code == 200
    assert "event: response.created" in resp.text
    assert "event: response.completed" in resp.text


def test_openai_responses_route_warns_and_omits_unsupported_tool_type():
    px, ag, fake_client = _make_gateway_with_agent(token="tok")
    client = _client_for(px)
    body = {
        "model": "m",
        "input": "hi",
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "description": "Run a command",
                "parameters": {"type": "object"},
                "strict": False,
            },
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "description": "Unsupported Responses namespace tool",
                "tools": [],
            },
        ],
        "stream": False,
    }

    resp = client.post("/v1/responses", json=body, headers={"Authorization": "Bearer tok"})

    assert resp.status_code == 200
    forwarded_tools = fake_client.chat.completions.create.call_args.kwargs["tools"]
    assert [tool["function"]["name"] for tool in forwarded_tools] == ["exec_command"]
    ag.terminal.log.assert_called_once()
    warning_prefix, warning_message = ag.terminal.log.call_args.args
    assert warning_prefix == "WARNING  "
    assert "namespace" in warning_message
    assert "omitted" in warning_message


def test_openai_responses_route_rejects_unsupported_input_content_before_dispatch():
    px, _, fake_client = _make_gateway_with_agent(token="tok")
    client = _client_for(px)
    body = {
        "model": "m",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AA==",
                        "detail": "auto",
                    }
                ],
            }
        ],
        "stream": True,
    }

    resp = client.post("/v1/responses", json=body, headers={"Authorization": "Bearer tok"})

    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
    assert resp.json()["error"]["code"] == "unsupported_responses_translation"
    assert "input_image" in resp.json()["error"]["message"]
    fake_client.chat.completions.create.assert_not_called()


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
    fake_client.chat.completions.create.return_value = _completion(
        content="hi", finish_reason="stop", usage=(1, 1)
    )
    client = _client_for(px)
    client.post(
        "/v1/messages",
        json={
            "model": "claude-x",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
        headers={"Authorization": "Bearer tok"},
    )
    assert len(px.request_log) == 1
    assert px.request_log[0]["route"] == "/v1/messages"
    assert px.request_log[0]["model"] == "configured-model"  # overridden, not "claude-x"


def test_request_log_records_responses_calls():
    px, ag, fake_client = _make_gateway_with_agent(token="tok")
    fake_client.chat.completions.create.return_value = _completion(content="hi", usage=(1, 1))
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


# ---------------------------------------------------------------------------
# Credential separation -- agProxyLLM must never construct a real backend
# client itself; every dispatch has to genuinely resolve through
# agllm_terminus.agLLMTerminus. See that module and agproxy_llm.py's
# `_dispatch()`/class docstring.
# ---------------------------------------------------------------------------


def test_register_propagates_to_injected_terminus():
    from agency.agharness_internal.agllm_terminus import agLLMTerminus

    terminus = agLLMTerminus()
    px = agProxyLLM(terminus=terminus)
    fake_ag = MagicMock()
    px.register("tok", fake_ag)
    assert terminus._agent_for_token("tok") is fake_ag


def test_unregister_propagates_to_injected_terminus():
    from agency.agharness_internal.agllm_terminus import agLLMTerminus

    terminus = agLLMTerminus()
    px = agProxyLLM(terminus=terminus)
    fake_ag = MagicMock()
    px.register("tok", fake_ag)
    px.unregister("tok")
    assert terminus._agent_for_token("tok") is None


def test_dispatch_resolves_agent_via_the_terminus_not_agproxy_llms_own_registry():
    """Direct proof of the credential-separation property: mutating the
    TERMINUS's registry entry, independently of agProxyLLM's own copy, must
    change which backend client actually gets called for real dispatch --
    if `_dispatch()` were secretly still calling
    `ag.llm.backend.make_client()` against agProxyLLM's own
    `_agents_by_token` (used only for the 401 check / model routing), this
    would keep using the original agent instead."""
    from agency.agharness_internal.agllm_terminus import agLLMTerminus

    terminus = agLLMTerminus()
    px = agProxyLLM(terminus=terminus)

    original_client = MagicMock()
    original_client.chat.completions.create.return_value = _completion(content="original")
    original_ag = MagicMock()
    original_ag.llm.backend.make_client.return_value = original_client
    original_ag.llm.backend.model = "m"
    px.register("tok", original_ag)  # propagates to `terminus` too

    replacement_client = MagicMock()
    replacement_client.chat.completions.create.return_value = _completion(content="replacement")
    replacement_ag = MagicMock()
    replacement_ag.llm.backend.make_client.return_value = replacement_client
    # Mutate the terminus's registry DIRECTLY -- agProxyLLM's own registry
    # (px._agents_by_token, used for the 401 check above) still has
    # original_ag and is never touched again.
    terminus.register("tok", replacement_ag)

    client = _client_for(px)
    body = {"model": "m", "messages": [], "stream": False}
    resp = client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer tok"})

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "replacement"
    original_client.chat.completions.create.assert_not_called()
    replacement_client.chat.completions.create.assert_called_once()


def test_real_end_to_end_over_two_real_http_servers():
    """Genuine proof of over-the-wire separation: a real client hits
    agProxyLLM's own real (TCP-bound) server, which in turn reaches
    agLLMTerminus's real (TCP-bound) server over an actual socket -- two
    independently-bound uvicorn servers, not one ASGI app and not an
    in-process shortcut."""
    import httpx

    from agency.agharness_internal.agllm_terminus import agLLMTerminus

    terminus = agLLMTerminus()
    px = agProxyLLM(terminus=terminus)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _completion(content="over the wire")
    fake_ag = MagicMock()
    fake_ag.llm.backend.make_client.return_value = fake_client
    px.register("tok", fake_ag)

    try:
        base_url = px.start()
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={"model": "m", "messages": [], "stream": False},
            headers={"Authorization": "Bearer tok"},
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "over the wire"
        # A genuine second, independently-bound TCP port for the terminus --
        # not the same server, not an ASGI-in-memory shortcut.
        assert terminus.base_url is not None
        assert terminus.base_url != base_url
    finally:
        px.stop()


def test_terminus_uds_path_construction_with_no_local_registry():
    """The construction mode agproxy_llm actually needs once it runs
    inside the container: no live `terminus` object at all, no
    register()/unregister() ever called on THIS instance (the caller
    registers directly on the real terminus before this process even
    starts, exactly as claude_code.py's execute() does for a
    container-backed launch) -- every auth check and every piece of
    per-agent state must still resolve correctly purely over the
    bind-mounted UDS socket."""
    from agency.agharness_internal.agllm_terminus import agLLMTerminus

    terminus = agLLMTerminus()
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _completion(content="via uds only")
    fake_ag = MagicMock()
    fake_ag.llm.backend.make_client.return_value = fake_client
    fake_ag.llm.backend.model = "configured-model"
    # Registered directly on the terminus -- never on the agProxyLLM
    # instance below, which (in this construction mode) has no register()
    # effect at all.
    terminus.register("tok", fake_ag)

    try:
        uds_path = terminus.ensure_uds_started()
        px = agProxyLLM(terminus_uds_path=uds_path)

        # register()/unregister() on THIS instance must be harmless no-ops,
        # not errors -- nothing to mirror onto since there's no live
        # terminus object here.
        px.register("unused-token", MagicMock())
        px.unregister("unused-token")

        client = _client_for(px)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [], "stream": False},
            headers={"Authorization": "Bearer tok"},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "via uds only"

        # /v1/messages' model resolution also goes through the terminus
        # purely over the UDS path.
        assert px._resolve_model("tok") == "configured-model"

        # An unknown token must still be rejected correctly.
        resp2 = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": []},
            headers={"Authorization": "Bearer nope"},
        )
        assert resp2.status_code == 401
    finally:
        terminus.stop_uds()
