# Tests for host_server_manager.py -- the composition root owning the sub-servers.

from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agency.configs.agconfig import agconfig, dataloggerconfig, hostserverconfig, llmconfig
from agency.observability.agdatalogger import agDataLogger
from agency.agpolicy import agpolicy
from agency.agskill import agskill
from agency.engine.host_servers import host_server_manager as manager_mod
from agency.engine.host_servers.host_interaction_server import HostInteractionServer
from agency.engine.host_servers.host_server_manager import (
    HostServerManager,
    _AttemptFenceMiddleware,
)
from agency.harness.protocol import ATTEMPT_TOKEN_HEADER
from agency.llm.usage_tracker import LlmUsageTracker
from agency.observability.profiler import agprof


def _make_manager(tmp_path, policy=None, is_cancelled=None, harness="claude_code"):
    cfg = agconfig(
        llmconfig(model="test-model"),
        hostserverconfig(uds_path=str(tmp_path / "host.sock")),
        dataloggerconfig(db_path=str(tmp_path / "agent.db")),
    )
    agent = SimpleNamespace(
        agname="agent-1",
        agconfig=cfg,
        harness=harness,
        data_logger=agDataLogger(cfg),
        llm_usage_tracker=LlmUsageTracker(),
    )
    sandbox = SimpleNamespace()
    skill = SimpleNamespace(
        name="test-skill",
        policy=policy if policy is not None else agpolicy(),
        host_mcp_tools=[],
        output_schema=None,
    )
    resource_pool = SimpleNamespace()
    return (
        HostServerManager(
            agent,
            sandbox,
            skill,
            resource_pool,
            is_cancelled=is_cancelled,
        ),
        agent,
        skill,
    )


def test_construction_uses_the_agents_own_data_logger(tmp_path):
    manager, agent, _ = _make_manager(tmp_path)

    assert manager._configs.host_server.uds_path.endswith(".sock")
    assert manager._data_logger is agent.data_logger


def test_construction_wires_skill_into_interaction_server(tmp_path):
    policy = agpolicy()
    manager, _, _ = _make_manager(tmp_path, policy=policy)
    assert isinstance(manager._interaction_server, HostInteractionServer)
    assert manager._interaction_server._policy is policy


def test_construction_captures_run_context_for_llm_requests(tmp_path, monkeypatch):
    parent_context = object()
    monkeypatch.setattr(agprof, "current_span_context", lambda: parent_context)

    manager, _, _ = _make_manager(tmp_path)

    assert manager._llm_handler_server._parent_context is parent_context


def test_construction_binds_cancel_check_to_the_interaction_server(tmp_path):
    """llm_handler_server no longer takes is_cancelled: the harness's own
    OS process is now killed directly on cancel (agent.cancel() ->
    cancel_harness()), so the host-side LLM handler has nothing left to
    cooperatively re-check. host_interaction_server (tool calls) still
    does."""
    is_cancelled = lambda: False  # noqa: E731
    manager, _, _ = _make_manager(tmp_path, is_cancelled=is_cancelled)

    assert not hasattr(manager._llm_handler_server, "_is_cancelled")
    assert manager._interaction_server._is_cancelled is is_cancelled


def test_attempt_token_gate_accepts_only_the_exact_active_token(tmp_path):
    manager, _, _ = _make_manager(tmp_path)

    assert manager._allows_attempt_token(None) is False
    assert manager._allows_attempt_token("") is False
    assert manager._allows_attempt_token("old") is False
    with pytest.raises(ValueError, match="non-empty"):
        manager.bind_attempt_token("")

    manager.bind_attempt_token("current")
    assert manager._allows_attempt_token("current") is True
    assert manager._allows_attempt_token("old") is False
    assert manager._allows_attempt_token("non-ascii-\u00e9") is False
    with pytest.raises(RuntimeError, match="already bound"):
        manager.bind_attempt_token("replacement")

    assert manager.clear_attempt_token("old") is False
    assert manager._allows_attempt_token("current") is True
    assert manager.clear_attempt_token("current") is True
    assert manager._allows_attempt_token("current") is False


def test_stop_revokes_the_active_attempt_token(tmp_path):
    manager, _, _ = _make_manager(tmp_path)
    manager.bind_attempt_token("current")

    manager.stop()

    assert manager._allows_attempt_token("current") is False


def test_stop_closes_attempt_admission_before_teardown_can_race_a_new_bind(tmp_path):
    manager, _, _ = _make_manager(tmp_path)
    stop_entered = threading.Event()
    release_stop = threading.Event()
    stop_errors = []

    def block_llm_stop():
        stop_entered.set()
        assert release_stop.wait(timeout=2.0)

    manager._llm_handler_server = SimpleNamespace(stop=block_llm_stop)

    def stop_manager():
        try:
            manager.stop()
        except BaseException as exc:
            stop_errors.append(exc)

    worker = threading.Thread(target=stop_manager)
    worker.start()
    assert stop_entered.wait(timeout=2.0)
    try:
        with pytest.raises(RuntimeError, match="admission is closed"):
            manager.bind_attempt_token("late-attempt")
    finally:
        release_stop.set()
        worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert stop_errors == []
    assert manager._allows_attempt_token("late-attempt") is False


def test_stop_cancels_llm_streams_before_joining_the_server(tmp_path):
    manager, _, _ = _make_manager(tmp_path)
    events = []

    manager._llm_handler_server = SimpleNamespace(stop=lambda: events.append("llm-stop"))
    manager._server = SimpleNamespace(should_exit=False, force_exit=False)

    class StoppedThread:
        @staticmethod
        def join(timeout):
            del timeout
            events.append("server-join")

        @staticmethod
        def is_alive():
            return False

    manager._server_thread = StoppedThread()

    manager.stop()

    assert events == ["llm-stop", "server-join"]
    assert manager._server is None
    assert manager._server_thread is None


def test_stop_retains_a_live_server_thread_for_retry(tmp_path):
    manager, _, _ = _make_manager(tmp_path)
    manager._configs.host_server.shutdown_timeout_s = 0
    manager._llm_handler_server = SimpleNamespace(stop=lambda: None)
    server = SimpleNamespace(should_exit=False, force_exit=False)

    class RetryableThread:
        alive = True

        @classmethod
        def join(cls, timeout):
            del timeout

        @classmethod
        def is_alive(cls):
            return cls.alive

    thread = RetryableThread()
    manager._server = server
    manager._server_thread = thread

    with pytest.raises(RuntimeError, match="worker did not stop"):
        manager.stop()

    assert server.should_exit is True
    assert server.force_exit is True
    assert manager._server is server
    assert manager._server_thread is thread

    RetryableThread.alive = False
    manager.stop()
    assert manager._server is None
    assert manager._server_thread is None


def test_failed_start_clears_dead_worker_state_and_can_retry(tmp_path, monkeypatch):
    manager, _, _ = _make_manager(tmp_path)
    manager._configs.host_server.startup_timeout_s = 0
    manager._configs.host_server.shutdown_timeout_s = 0
    servers = []

    class FailedServer:
        started = False
        should_exit = False
        force_exit = False

        def __init__(self, _config):
            servers.append(self)

        @staticmethod
        def run():
            return None

    class DeadThread:
        def __init__(self, *, target, daemon, name):
            del daemon, name
            self._target = target
            self._alive = False

        def start(self):
            self._target()

        @staticmethod
        def join(timeout):
            del timeout

        def is_alive(self):
            return self._alive

    monkeypatch.setattr(manager_mod.uvicorn, "Server", FailedServer)
    monkeypatch.setattr(manager_mod.threading, "Thread", DeadThread)

    with pytest.raises(RuntimeError, match="did not start"):
        manager.start()
    assert manager._server is None
    assert manager._server_thread is None

    with pytest.raises(RuntimeError, match="did not start"):
        manager.start()
    assert len(servers) == 2


def test_thread_start_failure_clears_server_refs_and_remains_stoppable(tmp_path, monkeypatch):
    manager, _, _ = _make_manager(tmp_path)

    class StartFailureThread:
        def __init__(self, *, target, daemon, name):
            del target, daemon, name

        @staticmethod
        def start():
            raise OSError("thread capacity exhausted")

    monkeypatch.setattr(manager_mod.threading, "Thread", StartFailureThread)

    with pytest.raises(OSError, match="thread capacity exhausted"):
        manager.start()

    assert manager._server is None
    assert manager._server_thread is None
    with pytest.raises(RuntimeError, match="admission is closed"):
        manager.bind_attempt_token("should-not-bind")
    manager.stop()


def test_clear_waits_for_a_streaming_response_to_release_its_lease(tmp_path):
    manager, _, _ = _make_manager(tmp_path)
    manager.bind_attempt_token("current")
    body_started = threading.Event()
    finish_body = threading.Event()
    request_done = threading.Event()
    clear_done = threading.Event()
    clear_results = []
    request_errors = []
    sent = []

    async def streaming_app(_scope, _receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"first", "more_body": True})
        body_started.set()
        assert await asyncio.to_thread(finish_body.wait, 2.0)
        await send({"type": "http.response.body", "body": b"last", "more_body": False})

    middleware = _AttemptFenceMiddleware(streaming_app, manager=manager)
    scope = {
        "type": "http",
        "headers": [(ATTEMPT_TOKEN_HEADER.lower().encode(), b"current")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    def run_request():
        try:
            asyncio.run(middleware(scope, receive, send))
        except BaseException as exc:  # surfaced in the main test thread below
            request_errors.append(exc)
        finally:
            request_done.set()

    request_thread = threading.Thread(target=run_request)
    request_thread.start()
    assert body_started.wait(timeout=2.0)

    def clear_token():
        clear_results.append(manager.clear_attempt_token("current"))
        clear_done.set()

    clear_thread = threading.Thread(target=clear_token)
    clear_thread.start()
    with manager._attempt_token_condition:
        assert manager._attempt_token_condition.wait_for(
            lambda: manager._active_attempt_token is None,
            timeout=2.0,
        )

    assert clear_done.is_set() is False
    assert manager._acquire_attempt_lease("current") is False
    with pytest.raises(RuntimeError, match="already bound"):
        manager.bind_attempt_token("next")

    finish_body.set()
    assert request_done.wait(timeout=2.0)
    assert clear_done.wait(timeout=2.0)
    request_thread.join(timeout=2.0)
    clear_thread.join(timeout=2.0)

    assert request_errors == []
    assert clear_results == [True]
    assert sent[-1] == {
        "type": "http.response.body",
        "body": b"last",
        "more_body": False,
    }
    manager.bind_attempt_token("next")
    assert manager._allows_attempt_token("next") is True


def test_interaction_server_property_returns_the_same_instance(tmp_path):
    manager, _, _ = _make_manager(tmp_path)
    assert manager.interaction_server is manager._interaction_server


def test_host_mcp_server_property_returns_the_same_instance(tmp_path):
    manager, _, _ = _make_manager(tmp_path)
    assert manager.host_mcp_server is manager._host_mcp_server


def test_start_serves_the_mounted_mcp_server_without_a_lifespan_error():
    suffix = uuid.uuid4().hex[:8]
    uds_path = f"/tmp/hsm_test_{suffix}.sock"
    db_path = f"/tmp/hsm_test_{suffix}.db"
    cfg = agconfig(
        llmconfig(model="test-model"),
        hostserverconfig(uds_path=uds_path),
        dataloggerconfig(db_path=db_path),
    )
    agent = SimpleNamespace(
        agname="agent-1",
        agconfig=cfg,
        data_logger=agDataLogger(cfg),
        llm_usage_tracker=LlmUsageTracker(),
    )
    sandbox = SimpleNamespace()
    skill = agskill(name="s", prompt="p", policy=agpolicy())
    resource_pool = SimpleNamespace()
    manager = HostServerManager(agent, sandbox, skill, resource_pool)
    try:
        manager.start()

        with httpx2.Client(
            transport=httpx2.HTTPTransport(uds=uds_path), base_url="http://localhost"
        ) as client:
            assert client.get("/llm/resolve_model").status_code == 401
            assert (
                client.post(
                    "/interaction/check_tool",
                    headers={ATTEMPT_TOKEN_HEADER: "unknown"},
                    json={"tool_name": "unknown", "tool_input": {}},
                ).status_code
                == 401
            )
            assert client.post("/mcp", content=b"{}").status_code == 401

        attempt_token = "current-attempt"
        manager.bind_attempt_token(attempt_token)

        async def _list_tools():
            async with httpx2.AsyncClient(
                transport=httpx2.AsyncHTTPTransport(uds=uds_path),
                base_url="http://localhost",
                headers={ATTEMPT_TOKEN_HEADER: attempt_token},
            ) as client:
                async with streamable_http_client(
                    "http://localhost/mcp", http_client=client
                ) as streams:
                    read, write = streams[0], streams[1]
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        return await session.list_tools()

        tools = asyncio.run(_list_tools())
        assert {t.name for t in tools.tools} == {
            "reserve_resource",
            "release_resource",
            "get_current_resources",
            "daemon_release",
            "submit_output",
            "submitted_output",
        }

        with httpx2.Client(
            transport=httpx2.HTTPTransport(uds=uds_path), base_url="http://localhost"
        ) as client:
            headers = {ATTEMPT_TOKEN_HEADER: attempt_token}
            assert client.get("/llm/resolve_model", headers=headers).json() == {
                "model": "test-model"
            }
            check_tool_body = client.post(
                "/interaction/check_tool",
                headers=headers,
                json={"tool_name": "unknown", "tool_input": {}},
            ).json()
            assert check_tool_body["allowed"] is True
            assert check_tool_body["reason"] is None
            assert client.get("/LlmHandlerServer/resolve_model", headers=headers).status_code == 404

            assert manager.clear_attempt_token(attempt_token) is True
            assert client.get("/llm/resolve_model", headers=headers).status_code == 401
            manager.bind_attempt_token("next-attempt")
            assert client.get("/llm/resolve_model", headers=headers).status_code == 401
            assert (
                client.get(
                    "/llm/resolve_model",
                    headers={ATTEMPT_TOKEN_HEADER: "next-attempt"},
                ).status_code
                == 200
            )
    finally:
        manager.stop()
        Path(uds_path).unlink(missing_ok=True)
        Path(db_path).unlink(missing_ok=True)
