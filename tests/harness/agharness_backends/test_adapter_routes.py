from __future__ import annotations

from types import SimpleNamespace

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
