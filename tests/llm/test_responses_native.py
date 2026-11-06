"""Native Codex uses the actual SDK transport without reconstructing wire state."""

import copy
import json
import httpx
import openai
import pytest
from agency.configs.agconfig import agconfig, llmconfig
from agency.llm.agllm import agllm
from agency.llm.responses_native import context_from_request
from agency.harness.adapters.codex import CodexAdapter


def setup_backend(monkeypatch, handler):
    config = agconfig(
        llmconfig(
            provider="openai_responses",
            model="gpt-6.1-sol",
            base_url="https://mock/v1",
            api_key="fake",
        )
    )
    backend = agllm.for_config(config)
    client = openai.OpenAI(
        base_url="https://mock/v1",
        api_key="fake",
        max_retries=4,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(backend, "make_client", lambda *_: client)
    return backend, CodexAdapter(config)


@pytest.mark.parametrize(
    "input_value",
    [
        "",
        "hello",
        [],
        [
            {
                "type": "reasoning",
                "id": "rs-original",
                "summary": [],
                "encrypted_content": "opaque",
            },
            {
                "type": "message",
                "id": "m-original",
                "role": "assistant",
                "phase": "commentary",
                "content": [
                    {"type": "output_text", "text": "hi", "annotations": [{"future": True}]}
                ],
            },
            {
                "type": "custom_tool_call",
                "id": "ct-original",
                "call_id": "call",
                "name": "apply_patch",
                "input": "*** patch",
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call",
                "output": [{"type": "input_image", "image_url": "data:image/png;base64,AA"}],
            },
            {"type": "future_item", "future": {"x": 1}},
        ],
    ],
)
def test_native_request_exact_sdk_json_and_patchable_history(monkeypatch, input_value):
    body = {
        "model": "native-fallback",
        "input": input_value,
        "instructions": None,
        "stream": False,
        "tools": [
            {
                "type": "custom",
                "name": "apply_patch",
                "format": {"type": "grammar", "syntax": "lark", "definition": 'start: "x"'},
            }
        ],
        "reasoning": {"effort": "medium", "summary": "auto"},
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "tool_choice": {
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [{"type": "custom", "name": "apply_patch"}],
        },
        "parallel_tool_calls": True,
        "prompt_cache_key": "key",
        "future_option": {"x": [1]},
        "truncation": "disabled",
    }
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        assert req.headers["openai-beta"] == "test-beta"
        assert req.url.params["native"] == "yes"
        return httpx.Response(
            200,
            json={
                "id": "resp-real",
                "object": "response",
                "status": "completed",
                "output": [],
                "created_at": 123,
                "text": {
                    "format": {"type": "json_schema", "name": "x", "schema": {"type": "object"}}
                },
            },
        )

    backend, adapter = setup_backend(monkeypatch, handler)
    context = adapter._format_context_harness_to_agency(
        body, {"openai-beta": "test-beta", "authorization": "secret"}, {"native": "yes"}
    )
    raw = backend._call_backend(backend._format_context_agency_to_backend(context))
    assert seen == [{k: v for k, v in body.items() if k != "stream"}]
    response = backend._format_context_backend_to_agency(raw)
    assert adapter._format_context_agency_to_harness(response, "wrong-model") == {
        "id": "resp-real",
        "object": "response",
        "status": "completed",
        "output": [],
        "created_at": 123,
        "text": {"format": {"type": "json_schema", "name": "x", "schema": {"type": "object"}}},
    }
    # Patches alter the materialized block rather than an unpatchable saved body.
    copied = copy.deepcopy(context)
    if isinstance(input_value, str):
        copied["messages"][-1]["blocks"][0]["text"] = "patched"
        kwargs = backend._format_context_agency_to_backend(copied)
        assert kwargs["extra_body"]["input"] == "patched"
    assert "secret" not in json.dumps(context)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status", [400, 429, 503])
def test_native_sdk_no_hidden_retries(monkeypatch, stream, status):
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(
            status,
            json={"error": {"message": "synthetic", "type": "rate_limit_error"}},
            headers={"retry-after": "3"},
        )

    backend, _ = setup_backend(monkeypatch, handler)
    kwargs = backend._format_context_agency_to_backend(
        context_from_request({"model": "gpt-6.1-sol", "input": "hi"})
    )
    with pytest.raises(openai.APIStatusError) as error:
        (backend._call_backend_stream if stream else backend._call_backend)(kwargs)
    assert error.value.status_code == status and len(seen) == 1
    assert error.value.response.headers["retry-after"] == "3"


@pytest.mark.parametrize(
    "terminal", ["response.completed", "response.incomplete", "response.failed", "error"]
)
def test_native_all_stream_events_exact(monkeypatch, terminal):
    result = {
        "id": "resp-real",
        "object": "response",
        "model": "native-model",
        "status": terminal.split(".")[-1],
        "created_at": 123,
        "text": {"format": {"type": "json_schema", "name": "x", "schema": {"type": "object"}}},
        "output": [
            {"type": "reasoning", "id": "rs-real", "summary": [], "encrypted_content": "opaque"},
            {
                "type": "custom_tool_call",
                "id": "ct-real",
                "call_id": "c",
                "name": "apply_patch",
                "input": "patch",
            },
        ],
        "future_control": {"x": True},
        "error": {"code": "refusal", "message": "no"} if terminal == "response.failed" else None,
        "usage": {
            "input_tokens": 3,
            "output_tokens": 4,
            "total_tokens": 7,
            "input_tokens_details": {"cached_tokens": 1},
        },
    }
    events = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": "resp-real",
                "object": "response",
                "status": "in_progress",
                "future_field": 7,
            },
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": 0,
            "item": result["output"][0],
        },
        {"type": terminal, "sequence_number": 2, "response": result}
        if terminal != "error"
        else {
            "type": "error",
            "sequence_number": 2,
            "code": "synthetic",
            "message": "failed",
            "param": None,
            "future": True,
        },
    ]
    payload = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)

    def handler(req):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=payload)

    backend, adapter = setup_backend(monkeypatch, handler)
    kwargs = backend._format_context_agency_to_backend(
        context_from_request({"model": "gpt-6.1-sol", "input": "hi"})
    )
    stream, client = backend._call_backend_stream(kwargs)
    try:
        deltas = list(backend._format_stream_to_agency(stream))
    finally:
        client.close()
    metadata = next(d for d in deltas if d.get("block_type") == "metadata")
    message = {"blocks": [{"type": "metadata", "index": 0, "data": [metadata["data"]]}]}
    wire = "".join(
        adapter._format_agency_stream_to_harness([{"type": "done", "message": message}], "wrong")
    )
    actual = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: ")]
    assert actual == events


@pytest.mark.parametrize("shape", ["partial", "untyped_error", "empty"])
def test_native_nonterminal_stream_is_replayed_without_host_error_or_retry(monkeypatch, shape):
    events = (
        []
        if shape == "empty"
        else [{"type": "response.created", "response": {"id": "r", "status": "in_progress"}}]
    )
    if shape == "untyped_error":
        events.append(
            {
                "error": {
                    "type": "server_error",
                    "code": "internal_server_error",
                    "message": "upstream failed",
                },
                "model": "gpt-6.1-sol",
            }
        )
    calls = []

    def handler(request):
        calls.append(request)
        payload = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=payload + "data: [DONE]\n\n"
        )

    backend, adapter = setup_backend(monkeypatch, handler)
    kwargs = backend._format_context_agency_to_backend(
        context_from_request({"model": "gpt-6.1-sol", "input": "hi"})
    )
    stream, client = backend._call_backend_stream(kwargs)
    try:
        deltas = list(backend._format_stream_to_agency(stream))
    finally:
        stream.close()
        client.close()
    block = next(value for value in deltas if value.get("block_type") == "metadata")
    message = {"blocks": [{"type": "metadata", "index": 0, "data": [block["data"]]}]}
    wire = "".join(
        adapter._format_agency_stream_to_harness([{"type": "done", "message": message}], "wrong")
    )
    assert [
        json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: ")
    ] == events
    assert len(calls) == 1
    assert deltas[-1]["stop_reason"] is None  # No fabricated successful completion.
