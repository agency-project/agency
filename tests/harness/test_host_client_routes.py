from __future__ import annotations

import asyncio
import json
import threading

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agency.harness._syscall_event import agsyscallevent
from agency.harness.clients.host_services_client import HostServicesClient
from agency.harness.interaction_router import build_router as build_interaction_router
from agency.harness.mcp_proxy import build_router as build_mcp_router
from agency.harness.protocol import ATTEMPT_TOKEN_HEADER


def _bridge(handler, token: str = "token"):
    bridge = HostServicesClient.__new__(HostServicesClient)
    bridge.client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://agency-host"
    )
    bridge._uds_path = "/unused/test-host.sock"
    bridge._timeout_s = 300
    bridge._new_mcp_async_client = lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://agency-host",
    )
    bridge._attempt_token_lock = threading.Lock()
    bridge._active_attempt_token = None
    bridge.register_attempt_token(token)
    return bridge


def test_bridge_uses_stable_host_service_routes_and_request_shapes():
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers[ATTEMPT_TOKEN_HEADER] == "token"
        responses = {
            ("GET", "/llm/resolve_model"): {"model": "gpt-test"},
            ("GET", "/llm/context_limit"): {"context_limit": 128_000},
            ("POST", "/interaction/record_event"): {"ok": True},
            ("POST", "/interaction/check_tool"): {"allowed": False, "reason": "blocked"},
            ("POST", "/interaction/check_syscall"): {"allowed": True, "reason": None},
            ("POST", "/interaction/record_span"): {"ok": True},
            ("POST", "/interaction/record_samples"): {"ok": True, "rejected": 0},
            ("POST", "/interaction/profile_settings"): {"enabled": True, "automatic": None},
        }
        return httpx.Response(200, json=responses[(request.method, request.url.path)])

    bridge = _bridge(handler)
    try:
        assert bridge.resolve_model("token") == "gpt-test"
        assert bridge.context_limit("token") == 128_000
        bridge.log_warning("token", "bad shape")
        assert bridge.check_tool_policy("token", "bash", {"cmd": "x"}) == {
            "decision": "deny",
            "reason": "blocked",
            "call_id": None,
        }
        syscall = agsyscallevent(
            syscall="execve",
            pid=12,
            tid=12,
            argv=["/bin/true"],
            envp=None,
            path="/bin/true",
            timestamp=1.0,
        )
        assert bridge.check_syscall_policy("token", syscall) == (True, None, None)
        assert bridge.record_profiler_span(
            "token", {"name": "turn0", "span_id": "s1", "attributes": {}}
        ) == {"ok": True}
        assert bridge.record_profiler_samples("token", [{"name": "f"}]) == {
            "ok": True,
            "rejected": 0,
        }
        assert bridge.profiler_settings("token") == {"enabled": True, "automatic": None}
    finally:
        bridge.close()

    bodies = [json.loads(request.content) if request.content else None for request in requests]
    assert bodies == [
        None,
        None,
        {"type": "warning", "payload": {"message": "bad shape"}},
        {"tool_name": "bash", "tool_input": {"cmd": "x"}},
        {
            "syscall": "execve",
            "pid": 12,
            "tid": 12,
            "argv": ["/bin/true"],
            "envp": None,
            "path": "/bin/true",
            "timestamp": 1.0,
            "tool_name": None,
            "tool_args": None,
            "program": None,
            "address": None,
            "port": None,
        },
        {"name": "turn0", "span_id": "s1", "attributes": {}},
        {"samples": [{"name": "f"}]},
        {},
    ]


def test_bridge_complete_tool_and_syscall_policy_post_expected_routes():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"ok": True})

    bridge = _bridge(handler)
    try:
        bridge.complete_tool_policy("token", "call-1", result={"stdout": "ok"})
        bridge.complete_syscall_policy("token", "call-2", return_value=3)
    finally:
        bridge.close()

    assert seen == [
        (
            "POST",
            "/interaction/complete_tool",
            {"call_id": "call-1", "result": {"stdout": "ok"}, "error": None},
        ),
        (
            "POST",
            "/interaction/complete_syscall",
            {"call_id": "call-2", "return_value": 3, "error": None},
        ),
    ]


def test_bridge_dispatch_uses_llm_route_and_preserves_agency_response():
    seen = []
    agency_response = {
        "message": {"role": "assistant", "blocks": [{"type": "text", "index": 0, "text": "hello"}]},
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        "stop_reason": "stop",
    }

    def handler(request):
        assert request.headers[ATTEMPT_TOKEN_HEADER] == "token"
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=agency_response)

    bridge = _bridge(handler)
    try:
        result = bridge.dispatch("token", {"model": "gpt-test", "messages": []})
    finally:
        bridge.close()

    assert result == agency_response
    assert seen == [("POST", "/llm/dispatch", {"model": "gpt-test", "messages": []})]


def test_bridge_stream_dispatch_yields_wire_items_and_stops_at_done():
    def handler(request):
        assert request.url.path == "/llm/dispatch"
        assert request.headers[ATTEMPT_TOKEN_HEADER] == "token"
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200,
            content=(
                b'{"type":"delta","content":"hel"}\n'
                b'{"type":"delta","content":"lo"}\n'
                b'{"type":"done","message":{"role":"assistant","blocks":'
                b'[{"type":"text","index":0,"text":"hello"}]},'
                b'"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3},'
                b'"stop_reason":"stop"}\n'
            ),
            headers={"content-type": "application/x-ndjson"},
        )

    bridge = _bridge(handler)
    try:
        items = list(bridge.dispatch_stream("token", {"model": "gpt-test", "messages": []}))
    finally:
        bridge.close()

    assert items[0] == {"type": "delta", "content": "hel"}
    assert items[1] == {"type": "delta", "content": "lo"}
    assert items[2]["type"] == "done"
    assert items[2]["stop_reason"] == "stop"
    assert items[2]["message"]["blocks"] == [{"type": "text", "index": 0, "text": "hello"}]


def test_bridge_rejects_stale_token_before_host_request():
    requests = []
    bridge = _bridge(lambda request: requests.append(request) or httpx.Response(200, json={}))
    try:
        assert bridge.validate_token("non-ascii-é") is False
        assert bridge.validate_token("malformed-\ud800") is False
        assert bridge.clear_attempt_token("token") is True
        bridge.register_attempt_token("successor")
        assert bridge.clear_attempt_token("token") is False
        assert bridge.validate_token("successor") is True
        try:
            bridge.dispatch("token", {"messages": []})
        except RuntimeError as exc:
            assert str(exc) == "unknown or missing bearer token"
        else:
            raise AssertionError("stale token was accepted")
    finally:
        bridge.close()

    assert requests == []


def test_profiler_and_context_routes_reject_unknown_tokens():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"enabled": False, "automatic": None})

    bridge = _bridge(handler)
    app = FastAPI()
    app.include_router(build_interaction_router(bridge))
    try:
        with TestClient(app) as client:
            assert (
                client.post(
                    "/agprof/span",
                    headers={"Authorization": "Bearer stale"},
                    json={"name": "span"},
                ).status_code
                == 401
            )
            assert (
                client.post(
                    "/agprof/samples",
                    headers={"Authorization": "Bearer stale"},
                    json={"samples": []},
                ).status_code
                == 401
            )
            assert (
                client.get(
                    "/agprof/status",
                    headers={"Authorization": "Bearer stale"},
                ).status_code
                == 401
            )
            status = client.get(
                "/agprof/status",
                headers={"Authorization": "Bearer token"},
            )
            assert status.status_code == 200
            assert status.json() == {"enabled": False, "automatic": None}
            assert seen == [("POST", "/interaction/profile_settings")]
            assert (
                client.post(
                    "/internal/context_limit",
                    json={"token": "stale"},
                ).status_code
                == 401
            )
    finally:
        bridge.close()


def test_mcp_proxy_requires_active_token_replaces_spoofed_header_and_preserves_response_headers():
    seen = []

    class JsonResponseStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"jsonrpc":"2.0","result":{}}'

    def handler(request):
        seen.append(request)
        assert request.headers[ATTEMPT_TOKEN_HEADER] == "token"
        return httpx.Response(
            200,
            stream=JsonResponseStream(),
            headers={
                "content-type": "application/json",
                "mcp-session-id": "session-1",
            },
        )

    bridge = _bridge(handler)
    app = FastAPI()
    app.include_router(build_mcp_router(bridge))
    try:
        with TestClient(app) as client:
            assert client.post("/mcp", json={}).status_code == 401
            response = client.post(
                "/mcp?mode=test",
                headers={
                    "Authorization": "Bearer token",
                    ATTEMPT_TOKEN_HEADER: "spoofed",
                },
                content=b'{"jsonrpc":"2.0"}',
            )
    finally:
        bridge.close()

    assert response.status_code == 200
    assert response.headers["mcp-session-id"] == "session-1"
    assert response.json() == {"jsonrpc": "2.0", "result": {}}
    assert len(seen) == 1
    assert dict(seen[0].url.params) == {"mode": "test"}


def test_mcp_proxy_streams_and_closes_long_lived_upstream_on_client_disconnect():
    class LongLivedStream(httpx.AsyncByteStream):
        def __init__(self):
            self.closed = asyncio.Event()
            self.keep_open = asyncio.Event()

        async def __aiter__(self):
            yield b"event: endpoint\ndata: /messages\n\n"
            await self.keep_open.wait()

        async def aclose(self):
            self.closed.set()

    async def scenario():
        stream = LongLivedStream()
        seen = []

        async def handler(request):
            seen.append(request)
            return httpx.Response(
                200,
                stream=stream,
                headers={"content-type": "text/event-stream"},
            )

        bridge = _bridge(handler)
        app = FastAPI()
        app.include_router(build_mcp_router(bridge))
        first_chunk_sent = asyncio.Event()
        request_delivered = False
        sent = []

        async def receive():
            nonlocal request_delivered
            if not request_delivered:
                request_delivered = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await first_chunk_sent.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body" and message.get("body"):
                first_chunk_sent.set()

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/mcp",
                "raw_path": b"/mcp",
                "query_string": b"",
                "root_path": "",
                "headers": [(b"authorization", b"Bearer token")],
                "client": ("test", 1),
                "server": ("test", 80),
            },
            receive,
            send,
        )
        bridge.close()
        stream.keep_open.set()

        assert len(seen) == 1
        assert seen[0].headers[ATTEMPT_TOKEN_HEADER] == "token"
        assert sent[0]["type"] == "http.response.start"
        assert sent[0]["status"] == 200
        assert any(
            message["type"] == "http.response.body"
            and message.get("body") == b"event: endpoint\ndata: /messages\n\n"
            for message in sent
        )
        assert stream.closed.is_set()

    asyncio.run(scenario())
