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


def test_bridge_dispatch_uses_llm_route_and_preserves_completion_contract():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "hello"},
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                "stop_reason": "stop",
            },
        )

    bridge = _bridge(handler)
    try:
        result = bridge.dispatch("token", {"model": "gpt-test", "messages": []})
    finally:
        bridge.client.close()

    assert result.choices[0].message.content == "hello"
    assert seen == [("POST", "/llm/dispatch", {"model": "gpt-test", "messages": []})]


def test_bridge_stream_dispatch_translates_host_ndjson_to_completion_chunks():
    def handler(request):
        assert request.url.path == "/llm/dispatch"
        return httpx.Response(
            200,
            content=(
                b'{"type":"delta","content":"hel"}\n'
                b'{"type":"delta","content":"lo"}\n'
                b'{"type":"done","message":{"role":"assistant","content":"hello"},'
                b'"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n'
            ),
            headers={"content-type": "application/x-ndjson"},
        )

    bridge = _bridge(handler)
    try:
        chunks = list(
            bridge.dispatch("token", {"model": "gpt-test", "messages": [], "stream": True})
        )
    finally:
        bridge.client.close()

    assert [chunk.choices[0].delta.content for chunk in chunks] == ["hel", "lo", None]
    assert chunks[-1].choices[0].finish_reason == "stop"
