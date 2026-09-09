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


def _bridge_for(app: TestClient) -> SimpleNamespace:
    """A minimal bridge exposing exactly the methods NativeProfiler calls,
    routed straight to a HostInteractionServer's own TestClient -- the same
    canonical routes any harness's real bridge would hit."""
    return SimpleNamespace(
        profiler_settings=lambda: app.post("/profile_settings").json(),
        record_profiler_span=lambda payload: app.post("/record_span", json=payload).json(),
        record_profiler_samples=lambda samples: app.post(
            "/record_samples", json={"samples": samples}
        ).json(),
    )


def test_native_turn_tool_hierarchy_and_measured_duration(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    with agprof.session(tmp_path, sample_hz=0, auto_functions=False):
        with agprof.span("run0:task:agent"):
            server = HostInteractionServer(
                SimpleNamespace(policy=agpolicy()),
                Logger(),
                "test-agent",
                parent_context=agprof.current_span_context(),
            )
            bridge = _bridge_for(TestClient(server.build_app()))
            with NativeProfiler(bridge) as native:
                with native.span("turn0"):
                    call_id = server.admit_tool_call("read", {})["call_id"]
                    started = time.perf_counter_ns()
                    started_wall_ns = time.time_ns()
                    sum(range(100))
                    duration = time.perf_counter_ns() - started
                    server.complete_tool_call(
                        call_id,
                        result="ok",
                        duration_ns=duration,
                        started_wall_ns=started_wall_ns,
                    )
                # No subsequent LLM dispatch is needed to record the final tool.
    records = {r[1]: r for r in agprof.profile_records()}
    assert records["tool:read"][8] == records["turn0"][7]
    assert records["turn0"][8] == records["run0:task:agent"][7]
    assert records["tool:read"][3] == duration
    assert records["tool:read"][6]["provenance"] == "container_asserted"


def test_record_span_opens_closes_and_nests_without_any_harness_specific_state(
    monkeypatch, tmp_path
):
    """The canonical mechanism new spans go through -- open now (no end_ts),
    close later by repeating the same span_id, and a child naming an open
    span_id as its parent nests under it. Nothing here is native-specific;
    HostInteractionServer holds no state named after any one harness."""
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    with agprof.session(tmp_path, sample_hz=0, auto_functions=False):
        with agprof.span("run0:task:agent"):
            server = HostInteractionServer(
                SimpleNamespace(policy=agpolicy()),
                Logger(),
                "test-agent",
                parent_context=agprof.current_span_context(),
            )
            server.record_span("turn0", time.time(), None, {}, span_id="t0")
            assert "t0" in server._open_spans
            server.record_span("tool:read", time.time(), time.time(), {}, span_id="c0", parent="t0")
            assert "c0" not in server._open_spans  # reported complete in one call
            server.record_span("turn0", None, time.time(), {}, span_id="t0")
            assert not server._open_spans
    records = {r[1]: r for r in agprof.profile_records()}
    assert records["tool:read"][8] == records["turn0"][7]
    assert records["turn0"][8] == records["run0:task:agent"][7]


def test_record_samples_rejects_invalid_entries(monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    with agprof.session(tmp_path, sample_hz=0, auto_functions=True):
        server = HostInteractionServer(SimpleNamespace(policy=agpolicy()), Logger(), "test-agent")
        result = server.record_samples([{"name": "bad", "perf_ns": -1, "duration_ns": 5, "tid": 1}])
        assert result == {"ok": False, "rejected": 1}
    assert agprof.summary_metrics()["sampling"]["telemetry_errors"]["remote_events_rejected"] == 1


def test_standalone_native_collector_captures_dependencies_without_host_imports(tmp_path):
    # A real separate interpreter verifies sys.monitoring ownership, rather
    # than mocking away the central mechanism under test.
    script = """
import json, sys
from native_harness.profiling import NativeProfiler

samples = []

class Bridge:
    token = "test"

    def profiler_settings(self):
        return {
            "enabled": True,
            "automatic": {
                "min_duration_ms": 0,
                "max_depth": 32,
                "max_events": 1000,
                "include_dependencies": True,
            },
        }

    def record_profiler_span(self, payload):
        return {"ok": True}

    def record_profiler_samples(self, batch):
        samples.extend(batch)
        return {"ok": True, "rejected": 0}

with NativeProfiler(Bridge()):
    from pathlib import PurePosixPath
    PurePosixPath("/one/two").as_posix()
assert "agency" not in sys.modules
assert "opentelemetry" not in sys.modules
print(json.dumps(samples))
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
    samples = json.loads(result.stdout)
    assert samples
    assert any("pathlib" in s["name"] for s in samples)


@pytest.mark.parametrize("engine", ["native", "claude_code", "codex", "opencode", "grok"])
def test_common_harness_profile_contract_matches_golden(engine, monkeypatch, tmp_path):
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)
    with agprof.session(tmp_path, sample_hz=0, auto_functions=False):
        agprof.register_engine(engine)
        server = HostInteractionServer(
            SimpleNamespace(policy=agpolicy()),
            Logger(),
            "test-agent",
            profile_attributes={"harness": engine},
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

    bridge = HostServicesClient("/unused")
    bridge.client.close()
    bridge.client = httpx.Client(base_url="http://host", transport=httpx.MockTransport(receive))
    bridge.register_attempt_token("current")
    app = FastAPI()
    app.include_router(build_router(bridge))
    with TestClient(app) as client:
        assert client.post("/agprof/span", json={"name": "s"}).status_code == 401
        assert not requests
        response = client.post(
            "/agprof/span",
            json={"name": "s", "start_ts": 1.0, "end_ts": 2.0, "attributes": {}},
            headers={"Authorization": "Bearer current"},
        )
        assert response.json() == {"ok": True}
        assert requests[0].url.path == "/interaction/record_span"
        assert requests[0].headers[ATTEMPT_TOKEN_HEADER] == "current"
        bridge.clear_attempt_token("current")
        assert (
            client.post(
                "/agprof/span",
                json={"name": "s"},
                headers={"Authorization": "Bearer current"},
            ).status_code
            == 401
        )
        assert len(requests) == 1
    bridge.client.close()
