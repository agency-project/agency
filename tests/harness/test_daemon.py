from __future__ import annotations

import base64
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agency.agconfig import agConfig
from agency.engine.clients import SandboxInteractionClient
from agency.harness import daemon
from agency.harness.adapters.base import AdapterRuntime, AttemptResult, agharness_backend
from agency.harness.daemon import HarnessManager
from agency.harness.llm_router import LlmRequestBudget, build_router
from agency.harness.protocol import HarnessAttemptRequest, HarnessAttemptResult, PromptPayload


def test_adapter_session_blob_crosses_daemon_protocol(monkeypatch):
    seen = {}

    class FakeAdapter(agharness_backend):
        engine_key = "fake"

        def run_daemon_attempt(self, runtime, **kwargs):
            seen["runtime"] = runtime
            seen.update(kwargs)
            return AttemptResult(
                ok=True,
                final_text="done",
                session_id="session-2",
                session_blob=b"updated session state",
            )

    monkeypatch.setattr(
        agharness_backend,
        "for_config",
        classmethod(lambda cls, name, config: FakeAdapter(config)),
    )
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system", "user"),
        harness="fake",
        resume_session_id="session-1",
        prior_session_blob_b64=base64.b64encode(b"prior session state").decode("ascii"),
    )

    result = daemon._run_adapter_attempt(
        request,
        agConfig(),
        "http://127.0.0.1:8766",
        "model",
        "agent-1",
        object(),
    )

    assert seen["resume_session_id"] == "session-1"
    assert seen["prior_session_blob"] == b"prior session state"
    assert isinstance(seen["runtime"], AdapterRuntime)
    assert seen["runtime"].engine_name == "agent-1"
    assert seen["runtime"].model == "model"
    assert result.session_id == "session-2"
    assert base64.b64decode(result.session_blob_b64) == b"updated session state"


def test_daemon_dispatch_selects_adapter_from_request(monkeypatch):
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system", "user"),
        harness="claude_code",
        max_steps=4,
    )
    expected = HarnessAttemptResult(ok=True, final_text="done")
    seen = []
    budget_resets = []
    manager = HarnessManager.__new__(HarnessManager)
    manager._agconfig = agConfig()
    manager._engine_name = "agent-1"
    manager._harness_api = type(
        "HarnessApi",
        (),
        {
            "base_url": "http://127.0.0.1:8766",
            "resolve_model": lambda self: "model",
            "syscall_policy": object(),
            "request_budget": type(
                "Budget", (), {"reset": lambda self, limit: budget_resets.append(limit)}
            )(),
        },
    )()

    def run_adapter(got_request, config, base_url, model, engine_name, syscall_policy):
        seen.append((got_request, config, base_url, model, engine_name, syscall_policy))
        return expected

    monkeypatch.setattr(daemon, "_run_adapter_attempt", run_adapter)

    assert manager._dispatch_attempt(request) is expected
    assert seen == [
        (
            request,
            manager._agconfig,
            "http://127.0.0.1:8766",
            "model",
            "agent-1",
            manager._harness_api.syscall_policy,
        )
    ]
    assert budget_resets == [4, None]


def test_llm_request_budget_rejects_calls_after_max_steps():
    budget = LlmRequestBudget()
    budget.reset(2)
    assert budget.claim()
    assert budget.claim()
    assert not budget.claim()
    budget.reset(None)
    assert budget.claim()


def test_llm_router_returns_non_retryable_error_when_budget_is_exhausted():
    class Bridge:
        @staticmethod
        def validate_token(token):
            return bool(token)

        @staticmethod
        def dispatch(*_args, **_kwargs):
            raise AssertionError("exhausted requests must not reach the model")

    budget = LlmRequestBudget()
    budget.reset(0)
    app = FastAPI()
    app.include_router(build_router(Bridge(), budget))

    response = TestClient(app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer benchmark"},
        json={"messages": [{"role": "user", "content": "inspect"}]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Agency harness max_steps exhausted"


def test_harness_manager_returns_attempt_result_on_original_rpc():
    socket_dir = Path(f"/tmp/agency-daemon-{uuid.uuid4().hex[:8]}")
    socket_dir.mkdir()
    socket_path = socket_dir / "sandbox.sock"
    request = HarnessAttemptRequest(
        prompt=PromptPayload("system", "user", "output"),
        harness="claude_code",
        max_steps=8,
        suppress_builtin_tools=True,
    )
    expected = HarnessAttemptResult(
        ok=True,
        final_text="mock daemon result",
        input_tokens=5,
        output_tokens=2,
        session_id="session-1",
        session_blob_b64="c2Vzc2lvbiBzdGF0ZQ==",
    )
    seen = []
    manager = HarnessManager(
        str(socket_path),
        str(socket_dir / "host.sock"),
        "agent-1",
        attempt_handler=lambda got_request: seen.append(got_request) or expected,
        harness_api_port=0,
    )

    try:
        manager.start()
        with SandboxInteractionClient(str(socket_path), timeout_s=2.0) as client:
            assert client.is_ready()
            result = client.run_harness_attempt(request)
    finally:
        manager.stop()
        socket_path.unlink(missing_ok=True)
        socket_dir.rmdir()

    assert seen == [request]
    assert result == expected
