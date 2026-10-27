"""Provider control metadata survives the actual SDK, host and Claude adapter."""

import json

import anthropic
from anthropic import _base_client
import pytest

from agency.configs.agconfig import agconfig, llmconfig
from agency.engine.host_servers.llm_handler_server import LlmHandlerServer
from agency.harness.adapters.claude_code import ClaudeCodeAdapter
from agency.llm.usage_tracker import LlmUsageTracker
from tests.engine.host_servers_tests.test_llm_handler_server import _FakeDataLogger, _drain


@pytest.mark.parametrize("streamed", [False, True])
def test_native_response_control_fields_survive_sdk_and_host(streamed):
    details = {"type": "refusal", "category": "cyber", "explanation": "fixture"}
    context = {"applied_edits": [{"type": "clear_thinking_20251015"}]}
    usage = {
        "input_tokens": 2,
        "output_tokens": 0,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 20,
    }
    message = {
        "id": "msg_provider_fixture",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": usage,
        "container": {"id": "container_fixture"},
        "context_management": None,
        "stop_details": None,
        "future_message_field": {"fixture": True},
    }
    delta = {
        "type": "message_delta",
        "delta": {
            "stop_reason": "refusal",
            "stop_sequence": "fixture-stop",
            "stop_details": details,
            "container": None,
            "future_delta_field": {"fixture": True},
        },
        "usage": usage,
        "context_management": context,
        "future_event_field": ["fixture"],
    }
    start = {"type": "message_start", "message": message, "future_start_field": "fixture"}
    full = {
        **message,
        "stop_reason": "refusal",
        "stop_sequence": "fixture-stop",
        "stop_details": details,
        "context_management": context,
    }
    events = [start, delta, {"type": "message_stop"}]
    transport_http = getattr(_base_client, "httpx", None) or _base_client.httpx2

    def handle(request):
        if streamed:
            text = "".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
            )
            return transport_http.Response(
                200, headers={"content-type": "text/event-stream"}, text=text
            )
        return transport_http.Response(200, json=full)

    cfg = agconfig(llmconfig(provider="anthropic", model="claude-sonnet-5-5"))
    adapter = ClaudeCodeAdapter(cfg)
    request = adapter._format_context_harness_to_agency(
        {
            "model": "claude-sonnet-5-5",
            "max_tokens": 1024,
            "stream": streamed,
            "messages": [{"role": "user", "content": "fixture"}],
        }
    )
    with anthropic.Anthropic(
        api_key="fixture",
        base_url="http://fixture",
        http_client=transport_http.Client(transport=transport_http.MockTransport(handle)),
    ) as client:
        server = LlmHandlerServer(cfg, _FakeDataLogger(), usage_tracker=LlmUsageTracker())
        server._backend.make_client = lambda timeout: client
        if streamed:
            stream = server.start_stream(request)
            items = _drain(stream)
            assert items[-1]["type"] == "done", items
            encoded = "".join(
                adapter._format_agency_stream_to_harness(iter(items), "logical-model")
            )
            decoded = [
                json.loads(line[6:]) for line in encoded.splitlines() if line.startswith("data: ")
            ]
            actual_start = next(event for event in decoded if event["type"] == "message_start")
            actual_delta = next(event for event in decoded if event["type"] == "message_delta")
            assert actual_start["future_start_field"] == "fixture"
            assert actual_start["message"]["future_message_field"] == {"fixture": True}
            assert actual_start["message"]["id"] == message["id"]
            assert actual_start["message"]["model"] == message["model"]
            assert actual_start["message"]["container"] == message["container"]
            for key, value in delta.items():
                if key == "delta":
                    for field, contents in value.items():
                        assert actual_delta["delta"][field] == contents
                else:
                    assert actual_delta[key] == value
            stream._thread.join(timeout=2)
        else:
            response = adapter._format_context_agency_to_harness(
                server.dispatch(request), "logical-model"
            )
            for key in (
                "id",
                "model",
                "stop_reason",
                "stop_sequence",
                "stop_details",
                "context_management",
                "container",
                "future_message_field",
            ):
                assert response[key] == full[key]
