"""Native Claude requests retain their wire semantics through history patches."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agency.configs.agconfig import agconfig, llmconfig
from agency.harness.adapters.claude_code import ClaudeCodeAdapter
from agency.llm.anthropic import _AnthropicBackend
from agency.llm.message_store import MessageBlock, MessageStore


@pytest.fixture
def native_request():
    return {
        "model": "claude-sonnet-5-5",
        "max_tokens": 1024,
        "stream": True,
        "thinking": {"type": "adaptive", "display": "updates"},
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        "output_config": {"effort": "medium"},
        "metadata": {"user_id": "fixture-user"},
        "system": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second", "cache_control": {"type": "ephemeral"}},
        ],
        "tools": [
            {
                "name": "shell",
                "description": "Run a command",
                "strict": False,
                "input_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {},
                },
            }
        ],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        "messages": [
            {"role": "user", "content": "task"},
            {
                "role": "system",
                "content": [{"type": "text", "text": "environment"}],
                "output_config": {"effort": "medium"},
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "fixture reasoning",
                        "signature": "fixture-signature",
                    },
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "shell",
                        "input": {"command": "false"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "is_error": True,
                        "content": [
                            {
                                "type": "text",
                                "text": "failed",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {"type": "text", "text": "after"},
                ],
            },
        ],
    }


def config(**kwargs):
    return agconfig(llmconfig(provider="anthropic", **kwargs))


def wire_body(kwargs):
    body = copy.deepcopy(kwargs)
    headers = body.pop("extra_headers", {})
    body.update(body.pop("extra_body", {}))
    return body, headers


def roundtrip(request, cfg=None, patch=None, headers=None):
    cfg = cfg or config()
    translated = ClaudeCodeAdapter(cfg)._format_context_harness_to_agency(request, headers)
    store = MessageStore()
    hashes = store.add_messages(translated["messages"])
    if patch:
        old, replacements = patch(translated["messages"])
        store.add_patch(old.hash, [store.add_block(block) for block in replacements])
    translated["messages"] = store.materialize(hashes)
    return wire_body(_AnthropicBackend(cfg)._format_context_agency_to_backend(translated))


def test_native_request_roundtrip_preserves_fields_and_only_safe_headers(native_request):
    original = copy.deepcopy(native_request)
    actual, headers = roundtrip(
        native_request,
        headers={
            "Anthropic-Beta": "fixture-beta",
            "Anthropic-Version": "2023-06-01",
            "Authorization": "Bearer secret",
            "X-Api-Key": "secret",
            "Host": "untrusted",
        },
    )
    expected = copy.deepcopy(original)
    expected.pop("stream")
    assert actual == expected
    assert headers == {"anthropic-beta": "fixture-beta", "anthropic-version": "2023-06-01"}
    assert native_request == original


@pytest.mark.parametrize(
    "role,text", [("user", "task"), ("system", "environment"), ("system", "second")]
)
def test_history_patch_changes_only_target_text(native_request, role, text):
    def patch(messages):
        message = next(
            m
            for m in messages
            if m["role"] == role and any(b.get("text") == text for b in m["blocks"])
        )
        block = next(b for b in message["blocks"] if b.get("text") == text)
        replacement = {**block, "text": "replacement"}
        return MessageBlock(role, block), [MessageBlock(role, replacement)]

    actual, _ = roundtrip(native_request, patch=patch)
    expected = copy.deepcopy(native_request)
    expected.pop("stream")
    if text == "task":
        expected["messages"][0]["content"] = "replacement"
    elif text == "environment":
        expected["messages"][1]["content"][0]["text"] = "replacement"
    else:
        expected["system"][1]["text"] = "replacement"
    assert actual == expected


def test_history_patch_changes_structured_tool_result_without_losing_error_flag(native_request):
    def patch(messages):
        block = next(m["blocks"][0] for m in messages if m["role"] == "tool")
        replacement = copy.deepcopy(block)
        replacement["raw_content"][0]["text"] = "patched failure"
        return MessageBlock("tool", block), [MessageBlock("tool", replacement)]

    actual, _ = roundtrip(native_request, patch=patch)
    expected = copy.deepcopy(native_request)
    expected.pop("stream")
    expected["messages"][3]["content"][1]["content"][0]["text"] = "patched failure"
    assert actual == expected


def test_history_patch_removal_does_not_restore_original_from_native_metadata(native_request):
    def patch(messages):
        block = next(m["blocks"][0] for m in messages if m["role"] == "tool")
        return MessageBlock("tool", block), []

    actual, _ = roundtrip(native_request, patch=patch)
    expected = copy.deepcopy(native_request)
    expected.pop("stream")
    expected["messages"][3]["content"].pop(1)
    assert actual == expected


def test_explicit_overrides_preserve_other_native_options(native_request):
    native_request["output_config"]["format"] = {
        "type": "json_schema",
        "schema": {"type": "object"},
    }
    actual, _ = roundtrip(
        native_request,
        config(model="other-claude", reasoning_effort="high", max_completion_tokens=2048),
    )
    expected = copy.deepcopy(native_request)
    expected.pop("stream")
    expected.update(max_tokens=2048)
    expected["output_config"]["effort"] = "high"
    assert actual == expected


def test_native_fallback_model_is_not_remapped_to_the_initial_model(native_request):
    native_request["model"] = "claude-sonnet-5"
    actual, _ = roundtrip(native_request, config(model="claude-sonnet-5-5"))
    assert actual["model"] == "claude-sonnet-5"


@pytest.mark.parametrize(
    "content",
    [
        "plain result",
        [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}],
        [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "fixture"},
            }
        ],
    ],
)
@pytest.mark.parametrize("is_error", [True, False])
def test_tool_result_shapes_and_error_flags_survive(native_request, content, is_error):
    native_request["messages"][3]["content"][1].update(content=content, is_error=is_error)
    actual, _ = roundtrip(native_request)
    expected = copy.deepcopy(native_request)
    expected.pop("stream")
    assert actual == expected


def test_unknown_content_and_native_builtin_tool_survive(native_request):
    native_request["tools"].append({"type": "web_search_20250305", "name": "web_search"})
    native_request["messages"][0]["content"] = [{"type": "future_block", "value": [1, 2]}]
    actual, _ = roundtrip(native_request)
    expected = copy.deepcopy(native_request)
    expected.pop("stream")
    assert actual == expected


def test_tool_result_with_omitted_content_stays_omitted(native_request):
    native_request["messages"][3]["content"][1].pop("content")
    actual, _ = roundtrip(native_request)
    expected = copy.deepcopy(native_request)
    expected.pop("stream")
    assert actual == expected


@pytest.mark.parametrize("system", ["instructions", [], [{"type": "text", "text": ""}]])
def test_system_shapes_survive(native_request, system):
    native_request["system"] = system
    actual, _ = roundtrip(native_request)
    expected = copy.deepcopy(native_request)
    expected.pop("stream")
    assert actual == expected


def test_configured_endpoint_is_passed_to_sdk():
    backend = _AnthropicBackend(config(base_url="http://127.0.0.1:12345"))
    assert backend._client_kwargs(httpx.Timeout(1))["base_url"] == "http://127.0.0.1:12345"


def test_daemon_bootstrap_selects_native_request_path_without_host_credentials(
    monkeypatch, native_request
):
    from agency.engine.harness_daemon_launcher import _daemon_config
    from agency.harness import daemon

    host_config = config(api_key="host-secret", base_url="https://host-only")
    payload = json.loads(json.dumps(_daemon_config(host_config)))
    captured = []
    monkeypatch.setattr(
        daemon,
        "_parse_args",
        lambda _argv: SimpleNamespace(
            config_json=json.dumps(payload),
            host_uds="/host",
            sandbox_uds="/sandbox",
            engine_name="fixture",
            harness="claude_code",
            harness_api_port=1234,
        ),
    )
    monkeypatch.setattr(daemon, "_ping_daemon_lifecycle", lambda *_args: None)

    def manager(*args, **kwargs):
        captured.append(kwargs["agconfig"])
        raise RuntimeError("stop after config transport")

    monkeypatch.setattr(daemon, "HarnessManager", manager)
    with pytest.raises(RuntimeError, match="stop after config transport"):
        daemon.main([])
    daemon_config = captured[0]
    assert daemon_config.llm.provider == "anthropic"
    assert daemon_config.llm.api_key is None
    assert daemon_config.llm.base_url is None
    request = ClaudeCodeAdapter(daemon_config)._format_context_harness_to_agency(native_request)
    actual, _ = wire_body(_AnthropicBackend(config())._format_context_agency_to_backend(request))
    expected = copy.deepcopy(native_request)
    expected.pop("stream")
    assert actual == expected


def test_sdk_sends_preserved_native_fields_on_the_wire(native_request):
    import anthropic
    from anthropic import _base_client

    transport_http = getattr(_base_client, "httpx", None) or _base_client.httpx2
    captured = []

    def handle(request):
        captured.append(request)
        return transport_http.Response(
            200,
            json={
                "id": "msg_fixture",
                "type": "message",
                "role": "assistant",
                "model": native_request["model"],
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    cfg = config()
    context = ClaudeCodeAdapter(cfg)._format_context_harness_to_agency(
        native_request, {"anthropic-beta": "fixture-beta"}, {"beta": "true"}
    )
    kwargs = _AnthropicBackend(cfg)._format_context_agency_to_backend(context)
    with anthropic.Anthropic(
        api_key="fixture",
        base_url="http://fixture",
        http_client=transport_http.Client(transport=transport_http.MockTransport(handle)),
    ) as client:
        client.messages.create(**kwargs)
    expected = copy.deepcopy(native_request)
    expected.pop("stream")
    assert json.loads(captured[0].content) == expected
    assert captured[0].headers["anthropic-beta"] == "fixture-beta"
    assert "beta=true" in str(captured[0].url)


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("status", [429, 500, 503, 524])
@pytest.mark.parametrize("native", [False, True])
def test_native_request_does_not_add_sdk_retries(
    native_request, monkeypatch, streamed, status, native
):
    import anthropic
    from anthropic import _base_client

    transport_http = getattr(_base_client, "httpx", None) or _base_client.httpx2
    captured = []

    def handle(request):
        captured.append(request)
        return transport_http.Response(
            status,
            headers={"retry-after-ms": "1"},
            json={"type": "error", "error": {"type": "api_error", "message": "fixture"}},
        )

    cfg = config()
    backend = _AnthropicBackend(cfg)
    context = ClaudeCodeAdapter(cfg)._format_context_harness_to_agency(native_request)
    kwargs = backend._format_context_agency_to_backend(context)
    if not native:
        kwargs = dict(kwargs)
    with anthropic.Anthropic(
        api_key="fixture",
        base_url="http://fixture",
        http_client=transport_http.Client(transport=transport_http.MockTransport(handle)),
    ) as client:
        monkeypatch.setattr(backend, "make_client", lambda timeout: client)
        with pytest.raises(anthropic.APIStatusError) as caught:
            if streamed:
                backend._call_backend_stream(kwargs)
            else:
                backend._call_backend(kwargs)
        assert caught.value.status_code == status
        assert len(captured) == (1 if native else 3)
        expected = copy.deepcopy(native_request)
        if not streamed:
            expected.pop("stream")
        assert json.loads(captured[0].content) == expected


def test_actual_captured_native_request_roundtrips():
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments/harness_fidelity_smoke_20261023/results/standalone/provider/002.request.json"
    )
    if not path.exists():
        pytest.skip("The optional local smoke capture is not installed")
    request = json.loads(path.read_text())
    actual, _ = roundtrip(request)
    request.pop("stream")
    assert actual == request
