from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agency.configs.agconfig import agconfig
from agency.harness.adapters.claude_code import _ClaudeCodeBackend
from agency.harness.adapters.codex import _CodexBackend
from agency.harness.adapters.grok import _GrokBackend
from agency.harness.adapters.native import _NativeBackend
from agency.harness.adapters.opencode import _OpencodeBackend


@pytest.mark.parametrize(
    "backend_cls", [_ClaudeCodeBackend, _CodexBackend, _GrokBackend, _OpencodeBackend]
)
def test_external_stream_only_emits_authoritative_text_after_redirect(backend_cls):
    stream = [
        {"type": "delta", "content": "superseded draft"},
        {
            "type": "done",
            "message": {
                "role": "assistant",
                "blocks": [{"type": "text", "index": 0, "text": "authoritative final"}],
            },
            "stop_reason": "stop",
            "usage": {},
        },
    ]
    frames = "".join(backend_cls(agconfig())._format_agency_stream_to_harness(iter(stream), "m"))
    assert "superseded draft" not in frames
    assert "authoritative final" in frames


@pytest.mark.parametrize(
    ("backend_cls", "path"),
    [
        (_ClaudeCodeBackend, "/v1/messages"),
        (_CodexBackend, "/v1/responses"),
        (_GrokBackend, "/v1/chat/completions"),
        (_NativeBackend, "/v1/chat/completions"),
        (_OpencodeBackend, "/v1/chat/completions"),
    ],
)
def test_registered_adapter_routes_inject_fastapi_request(backend_cls, path):
    app = FastAPI()
    bridge = SimpleNamespace(validate_token=lambda _token: False)
    backend_cls(agconfig()).register(app, bridge)

    with TestClient(app) as client:
        response = client.post(path, json={})

    assert response.status_code == 401
    assert "unknown or missing bearer token" in response.text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("title_format", ["legacy", "session_tags"])
def test_claude_session_title_request_is_answered_without_model_dispatch(stream, title_format):
    app = FastAPI()
    bridge = SimpleNamespace(
        validate_token=lambda _token: True,
        resolve_model=lambda _token: "test-model",
        dispatch=MagicMock(side_effect=AssertionError("title request reached the model")),
        dispatch_stream=MagicMock(side_effect=AssertionError("title request reached the model")),
        log_warning=MagicMock(),
    )
    _ClaudeCodeBackend(agconfig()).register(app, bridge)
    body = {
        "stream": stream,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Generate a concise, sentence-case title (3-7 words) "
                    "that captures the main topic of this coding session."
                ),
            }
        ],
    }
    if title_format == "session_tags":
        # Claude Code 2.1.251 moved the title instructions into the system
        # prompt and wraps the real invocation input as data, not a model turn.
        body["system"] = (
            "The session content is provided inside <session> tags. "
            'Return JSON with a single "title" field. Capitalize the first letter of the title.'
        )
        body["messages"] = [{"role": "user", "content": "<session>Fix a bug</session>"}]

    with TestClient(app) as client:
        response = client.post("/v1/messages", headers={"x-api-key": "token"}, json=body)

    assert response.status_code == 200
    assert "Agency session" in response.text
    bridge.dispatch.assert_not_called()
    bridge.dispatch_stream.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("auxiliary", ["title", "dashboard"])
def test_grok_auxiliary_request_is_answered_without_model_dispatch(stream, auxiliary):
    app = FastAPI()
    bridge = SimpleNamespace(
        validate_token=lambda _token: True,
        resolve_model=lambda _token: "test-model",
        dispatch=MagicMock(side_effect=AssertionError("title request reached the model")),
        dispatch_stream=MagicMock(side_effect=AssertionError("title request reached the model")),
    )
    _GrokBackend(agconfig()).register(app, bridge)
    body = {
        "stream": stream,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are tasked with generating the session title.\n"
                    "Just generate the session_title and nothing else"
                ),
            },
            {"role": "user", "content": "<user_query>Fix a bug</user_query>"},
        ],
    }
    if auxiliary == "dashboard":
        body["messages"] = [
            {"role": "assistant", "content": "The actual answer"},
            {
                "role": "user",
                "content": (
                    "<system-reminder>Write an ultra-short dashboard line that captures "
                    "the AGENT'S REPLY for the last turn only — a summary.</system-reminder>"
                ),
            },
        ]
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer token"}, json=body
        )
    assert response.status_code == 200
    assert ("Agency session" if auxiliary == "title" else "Agency turn") in response.text
    bridge.dispatch.assert_not_called()
    bridge.dispatch_stream.assert_not_called()
