# Tests for host_server_manager.py -- the composition root owning the sub-servers.

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agency.agpolicy import agpolicy
from agency.agskill import agskill
from agency.engine.agDataCollector import agDataCollectorConfigs
from agency.engine.host_servers.harness_interaction_server import HarnessInteractionServer
from agency.engine.host_servers.host_server_manager import (
    HostServerManager,
    HostServerManagerConfigs,
)


def _make_manager(tmp_path, policy=None):
    configs = HostServerManagerConfigs(uds_path=str(tmp_path / "host.sock"))
    data_collector_configs = agDataCollectorConfigs(db_path=str(tmp_path / "agent.db"))
    agconfig = SimpleNamespace(
        HostServerManagerConfigs=configs, agDataCollectorConfigs=data_collector_configs
    )
    agent = SimpleNamespace(agconfig=agconfig, inbox=object())
    sandbox = SimpleNamespace()
    skill = SimpleNamespace(policy=policy if policy is not None else agpolicy())
    resource_pool = SimpleNamespace()
    return HostServerManager(agent, sandbox, skill, resource_pool), agent, skill


class _LifecycleComponent:
    def __init__(self, name, events, *, start_error=None, stop_error=None):
        self.name = name
        self.events = events
        self.start_error = start_error
        self.stop_error = stop_error

    def start(self):
        self.events.append(f"{self.name}.start")
        if self.start_error is not None:
            raise self.start_error

    def stop(self):
        self.events.append(f"{self.name}.stop")
        if self.stop_error is not None:
            raise self.stop_error


class _NeverStopsThread:
    def __init__(self):
        self.join_timeouts = []

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)

    def is_alive(self):
        return True


def test_construction_wires_agent_and_skill_into_harness_interaction_server(tmp_path):
    policy = agpolicy()
    manager, agent, skill = _make_manager(tmp_path, policy=policy)
    assert isinstance(manager._harness_interaction_server, HarnessInteractionServer)
    assert manager._harness_interaction_server._agent is agent
    assert manager._harness_interaction_server._policy is policy


def test_does_not_expose_a_public_harness_interaction_server_property(tmp_path):
    manager, _, _ = _make_manager(tmp_path)
    assert not hasattr(manager, "harness_interaction_server")


def test_start_serves_the_mounted_mcp_server_without_a_lifespan_error():
    token = uuid.uuid4().hex[:8]
    uds_path = f"/tmp/hsm_test_{token}.sock"
    db_path = f"/tmp/hsm_test_{token}.db"
    configs = HostServerManagerConfigs(uds_path=uds_path)
    data_collector_configs = agDataCollectorConfigs(db_path=db_path)
    agconfig = SimpleNamespace(
        HostServerManagerConfigs=configs, agDataCollectorConfigs=data_collector_configs
    )
    agent = SimpleNamespace(agconfig=agconfig, inbox=object())
    sandbox = SimpleNamespace()
    skill = agskill(name="s", system_prompt="p", policy=agpolicy())
    resource_pool = SimpleNamespace()
    manager = HostServerManager(agent, sandbox, skill, resource_pool)
    try:
        assert manager.start() == uds_path

        async def _list_tools():
            client = httpx2.AsyncClient(
                transport=httpx2.AsyncHTTPTransport(uds=uds_path), base_url="http://localhost"
            )
            async with streamable_http_client(
                "http://localhost/HostMcpServer/mcp", http_client=client
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
    finally:
        manager.stop()
        Path(uds_path).unlink(missing_ok=True)
        Path(db_path).unlink(missing_ok=True)


def test_partial_start_cleanup_attempts_every_possibly_started_component(tmp_path):
    manager, _, _ = _make_manager(tmp_path)
    events = []
    first_stop_error = RuntimeError("first child stop failed")
    second_stop_error = RuntimeError("second child stop failed")
    collector_stop_error = RuntimeError("collector stop failed")
    collector = _LifecycleComponent("collector", events, stop_error=collector_stop_error)
    first = _LifecycleComponent("first", events, stop_error=first_stop_error)
    second = _LifecycleComponent(
        "second",
        events,
        start_error=RuntimeError("second child start failed"),
        stop_error=second_stop_error,
    )
    never_started = _LifecycleComponent("never", events)
    manager._data_collector = collector
    manager._server_instances = [first, second, never_started]

    with pytest.raises(RuntimeError, match="second child start failed"):
        manager.start()

    with pytest.raises(RuntimeError, match="first child stop failed") as raised:
        manager.stop()

    assert raised.value is first_stop_error
    assert events == [
        "collector.start",
        "first.start",
        "second.start",
        "first.stop",
        "second.stop",
        "collector.stop",
    ]
    notes = getattr(raised.value, "__notes__", [])
    assert any("second child stop failed" in note for note in notes)
    assert any("collector stop failed" in note for note in notes)


def test_shutdown_timeout_is_surfaced_after_attempting_remaining_cleanup(tmp_path):
    manager, _, _ = _make_manager(tmp_path)
    events = []
    child_error = RuntimeError("child cleanup failed")
    child = _LifecycleComponent("child", events, stop_error=child_error)
    collector = _LifecycleComponent("collector", events)
    thread = _NeverStopsThread()
    server = SimpleNamespace(should_exit=False)
    uds_path = Path(manager._configs.uds_path)
    uds_path.write_text("placeholder")

    manager._server = server
    manager._server_thread = thread
    manager._server_instances_to_stop = [child]
    manager._data_collector = collector
    manager._data_collector_needs_stop = True

    with pytest.raises(TimeoutError, match="did not stop") as raised:
        manager.stop()

    assert server.should_exit is True
    assert thread.join_timeouts == [manager._configs.shutdown_timeout_s]
    assert events == ["child.stop", "collector.stop"]
    assert not uds_path.exists()
    assert manager._server is server
    assert manager._server_thread is thread
    assert any("child cleanup failed" in note for note in getattr(raised.value, "__notes__", []))


def test_successful_stop_is_idempotent_for_child_services(tmp_path):
    manager, _, _ = _make_manager(tmp_path)
    events = []
    child = _LifecycleComponent("child", events)
    collector = _LifecycleComponent("collector", events)
    manager._server_instances_to_stop = [child]
    manager._data_collector = collector
    manager._data_collector_needs_stop = True

    manager.stop()
    manager.stop()

    assert events == ["child.stop", "collector.stop"]
