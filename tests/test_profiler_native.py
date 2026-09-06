"""Native telemetry transport, hierarchy, capture, and validation regressions."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from agency.agpolicy import agpolicy
from agency.engine.host_servers.host_interaction_server import HostInteractionServer
from agency.native_harness.profiling import NativeProfiler
from agency.observability.profiler import agprof


class Logger:
    def record_event(self, *args, **kwargs):
        pass

    def record_span(self, *args, **kwargs):
        pass


def test_native_turn_tool_hierarchy_and_measured_duration(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    with agprof.session(tmp_path, sample_hz=0, auto_functions=False):
        with agprof.span("run0:task:agent"):
            server = HostInteractionServer(
                SimpleNamespace(policy=agpolicy()),
                Logger(),
                parent_context=agprof.current_span_context(),
            )
            app = TestClient(server.build_app())

            def transport(request):
                route = request.url.path.replace("/agprof/", "/profile/")
                response = app.post(route, json=json.loads(request.content))
                return httpx.Response(response.status_code, json=response.json())

            bridge = SimpleNamespace(
                token="secret",
                _client=httpx.Client(
                    transport=httpx.MockTransport(transport), base_url="http://bridge"
                ),
            )
            with NativeProfiler(bridge) as native:
                with native.span("turn0"):
                    call_id = server.admit_tool_call("read", {})["call_id"]
                    started = time.perf_counter_ns()
                    sum(range(100))
                    duration = time.perf_counter_ns() - started
                    server.complete_tool_call(
                        call_id, result="ok", duration_ns=duration, started_perf_ns=started
                    )
                # No subsequent LLM dispatch is needed to record the final tool.
    records = {r[1]: r for r in agprof.profile_records()}
    assert records["tool:read"][8] == records["turn0"][7]
    assert records["turn0"][8] == records["run0:task:agent"][7]
    assert records["tool:read"][3] == duration
    assert records["tool:read"][6]["provenance"] == "container_asserted"
    assert "secret" not in (tmp_path / "agprof.trace.json").read_text()


def test_profile_ingest_rejects_stale_sessions_and_unknown_parents(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    with agprof.session(tmp_path, sample_hz=0, auto_functions=False):
        server = HostInteractionServer(SimpleNamespace(policy=agpolicy()), Logger())
        config = server.profile_config()
        assert not server.profile_events({"session_id": "stale"})["ok"]
        result = server.profile_events(
            {
                "session_id": config["session_id"],
                "events": [
                    {
                        "kind": "start",
                        "id": "child",
                        "parent_id": "missing",
                        "name": "turn0",
                        "perf_ns": time.perf_counter_ns(),
                    },
                    {"kind": "automatic", "name": "bad", "perf_ns": -1, "duration_ns": 5},
                ],
            }
        )
        assert result == {"ok": False, "rejected": 2}
        assert not server._remote_spans
    assert agprof.summary_metrics()["sampling"]["telemetry_errors"]["remote_events_rejected"] == 2


def test_standalone_native_collector_captures_dependencies_without_host_imports(tmp_path):
    # A real separate interpreter verifies sys.monitoring ownership, rather
    # than mocking away the central mechanism under test.
    script = """
import json, sys, time
from types import SimpleNamespace
from native_harness.profiling import NativeProfiler
class Response:
    def __init__(self, data): self.data = data
    def raise_for_status(self): pass
    def json(self): return self.data
class Client:
    def post(self, path, json, **kwargs):
        if path.endswith("config"):
            return Response({"enabled": True, "session_id": "test", "host_perf_ns": time.perf_counter_ns(),
                "automatic": {"min_duration_ms": 0, "max_depth": 32, "max_events": 1000, "include_dependencies": True}})
        events.extend(json["events"])
        return Response({"ok": True})
events = []
with NativeProfiler(SimpleNamespace(token="test", _client=Client())):
    from pathlib import PurePosixPath
    PurePosixPath("/one/two").as_posix()
assert "agency" not in sys.modules
assert "opentelemetry" not in sys.modules
print(json.dumps(events))
"""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "agency")}
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    events = json.loads(result.stdout)
    automatic = [e for e in events if e["kind"] == "automatic"]
    assert automatic
    assert any("pathlib" in e["name"] for e in automatic)


@pytest.mark.parametrize("engine", ["native", "claude_code", "codex", "opencode", "grok"])
def test_common_harness_profile_contract_matches_golden(engine, monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    with agprof.session(tmp_path, sample_hz=0, auto_functions=False):
        agprof.register_engine(engine)
        server = HostInteractionServer(
            SimpleNamespace(policy=agpolicy()), Logger(), profile_attributes={"harness": engine}
        )
        client = TestClient(server.build_app())
        admitted = client.post("/check_tool", json={"tool_name": "read", "tool_input": {}}).json()
        client.post("/complete_tool", json={"call_id": admitted["call_id"], "result": "ok"})
        client.post("/check_tool", json={"tool_name": "unfinished", "tool_input": {}})
        server.finalize_profile()
    summary = agprof.summary_metrics()
    shape = {
        key: summary["tool_metrics"][key]
        for key in ("started", "completed", "succeeded", "failed", "unknown", "interrupted")
    }
    golden = json.loads((Path(__file__).parent / "fixtures/profiler_contract.json").read_text())
    assert shape == golden
    assert engine in summary["coverage"]["engines"]
    assert summary["tool_metrics"]["latency"]["p95_ms"] is None


@pytest.mark.parametrize("engine", ["native", "claude_code", "codex", "opencode", "grok"])
def test_adapter_http_dispatch_reaches_real_profiler(engine, monkeypatch, tmp_path):
    import importlib
    from fastapi import FastAPI
    from agency.configs.agconfig import agconfig, llmconfig
    from agency.engine.host_servers.llm_handler_server import LlmHandlerServer
    from agency.llm.usage_tracker import LlmUsageTracker

    class EventLogger(Logger):
        def finalize_stream(self, *args, **kwargs):
            pass

    classes = {
        "native": "_NativeBackend",
        "claude_code": "_ClaudeCodeBackend",
        "codex": "_CodexBackend",
        "opencode": "_OpencodeBackend",
        "grok": "_GrokBackend",
    }
    module = importlib.import_module("agency.harness.adapters." + engine)
    config = agconfig(llmconfig(model="fixture-model"))
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    with agprof.session(tmp_path, sample_hz=0, auto_functions=False):
        with agprof.span("run0:fixture:agent"):
            handler = LlmHandlerServer(
                config,
                EventLogger(),
                LlmUsageTracker(),
                parent_context=agprof.current_span_context(),
            )
            handler._backend.dispatch = lambda request: {
                "message": {
                    "role": "assistant",
                    "blocks": [{"type": "text", "index": 0, "text": "done"}],
                },
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                "stop_reason": "end_turn",
            }
            bridge = SimpleNamespace(
                validate_token=lambda token: token == "fixture-token",
                resolve_model=lambda token: "fixture-model",
                dispatch=lambda token, request: handler.dispatch(request),
            )
            app = FastAPI()
            getattr(module, classes[engine])(config).register(app, bridge)
            path = (
                "/v1/responses"
                if engine == "codex"
                else "/v1/messages"
                if engine == "claude_code"
                else "/v1/chat/completions"
            )
            body = (
                {"input": "hello"}
                if engine == "codex"
                else {"messages": [{"role": "user", "content": "hello"}]}
            )
            with TestClient(app) as client:
                response = client.post(
                    path,
                    json=body,
                    headers={"Authorization": "Bearer fixture-token", "x-api-key": "fixture-token"},
                )
            assert response.status_code == 200, response.text
            assert "done" in response.text
            handler.stop()
    records = {r[1]: r for r in agprof.profile_records()}
    assert records["llm:attempt[0]"][8] == records["run0:fixture:agent"][7]
    result = agprof.summary_metrics()["llm_metrics"]
    assert {
        k: result[k]
        for k in ("calls", "successful_calls", "input_tokens", "output_tokens", "retries")
    } == {
        "calls": 1,
        "successful_calls": 1,
        "input_tokens": 12,
        "output_tokens": 3,
        "retries": None,
    }


def test_profiler_gateway_authenticates_and_forwards_attempt_header():
    from fastapi import FastAPI
    from agency.harness.clients.host_services_client import HostServicesClient
    from agency.harness.interaction_router import build_router
    from agency.harness.protocol import ATTEMPT_TOKEN_HEADER

    requests = []

    def receive(request):
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    bridge = HostServicesClient("/unused", None)
    bridge.client.close()
    bridge.client = httpx.Client(base_url="http://host", transport=httpx.MockTransport(receive))
    bridge.register_attempt_token("current")
    app = FastAPI()
    app.include_router(build_router(bridge))
    with TestClient(app) as client:
        assert client.post("/agprof/events", json={}).status_code == 401
        assert not requests
        response = client.post(
            "/agprof/events", json={"events": []}, headers={"Authorization": "Bearer current"}
        )
        assert response.json() == {"ok": True}
        assert requests[0].url.path == "/interaction/profile/events"
        assert requests[0].headers[ATTEMPT_TOKEN_HEADER] == "current"
        bridge.clear_attempt_token("current")
        assert (
            client.post(
                "/agprof/events", json={}, headers={"Authorization": "Bearer current"}
            ).status_code
            == 401
        )
        assert len(requests) == 1
    bridge.client.close()
