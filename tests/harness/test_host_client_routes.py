from __future__ import annotations

import json

import httpx

from agency.harness._syscall_event import agsyscallevent
from agency.harness.clients.host_services_client import HostServicesClient


def _bridge(handler):
    bridge = HostServicesClient.__new__(HostServicesClient)
    bridge.client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://agency-host"
    )
    bridge.profiler_uds_path = None
    bridge._profiler_synced_tokens = set()
    return bridge


def test_bridge_uses_stable_host_service_routes_and_request_shapes():
    requests = []

    def handler(request):
        requests.append(request)
        responses = {
            ("GET", "/llm/resolve_model"): {"model": "gpt-test"},
            ("GET", "/llm/context_limit"): {"context_limit": 128_000},
            ("POST", "/interaction/record_event"): {"ok": True},
            ("POST", "/interaction/check_tool"): {"allowed": False, "reason": "blocked"},
            ("POST", "/interaction/check_syscall"): {"allowed": True, "reason": None},
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
        assert bridge.check_syscall_policy(syscall) is True
    finally:
        bridge.client.close()

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
        },
    ]


def test_bridge_dispatch_uses_llm_route_and_preserves_agency_response():
    seen = []
    agency_response = {
        "message": {"role": "assistant", "blocks": [{"type": "text", "index": 0, "text": "hello"}]},
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        "stop_reason": "stop",
    }

    def handler(request):
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=agency_response)

    bridge = _bridge(handler)
    try:
        result = bridge.dispatch("token", {"model": "gpt-test", "messages": []})
    finally:
        bridge.client.close()

    assert result == agency_response
    assert seen == [("POST", "/llm/dispatch", {"model": "gpt-test", "messages": []})]


def test_bridge_stream_dispatch_yields_wire_items_and_stops_at_done():
    def handler(request):
        assert request.url.path == "/llm/dispatch"
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
        bridge.client.close()

    assert items[0] == {"type": "delta", "content": "hel"}
    assert items[1] == {"type": "delta", "content": "lo"}
    assert items[2]["type"] == "done"
    assert items[2]["stop_reason"] == "stop"
    assert items[2]["message"]["blocks"] == [{"type": "text", "index": 0, "text": "hello"}]
