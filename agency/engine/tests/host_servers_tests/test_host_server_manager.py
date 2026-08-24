# Tests for host_server_manager.py -- the composition root owning the sub-servers.

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agency.agconfig import agConfig
from agency.agpolicy import agpolicy
from agency.agskill import agskill
from agency.engine.agDataCollector import agDataCollectorConfigs
from agency.engine.host_servers.interaction_server import HarnessInteractionServer
from agency.engine.host_servers.host_server_manager import (
    HostServerManager,
    HostServerManagerConfigs,
)


def _make_manager(tmp_path, policy=None):
    configs = HostServerManagerConfigs(uds_path=str(tmp_path / "host.sock"))
    data_collector_configs = agDataCollectorConfigs(db_path=str(tmp_path / "agent.db"))
    agconfig = agConfig({"agllm_backend": {"model": "test-model"}})
    agconfig.HostServerManagerConfigs = configs
    agconfig.agDataCollectorConfigs = data_collector_configs
    agent = SimpleNamespace(agconfig=agconfig, inbox=object())
    sandbox = SimpleNamespace()
    skill = SimpleNamespace(policy=policy if policy is not None else agpolicy())
    resource_pool = SimpleNamespace()
    return HostServerManager(agent, sandbox, skill, resource_pool), agent, skill


def test_construction_wires_agent_and_skill_into_harness_interaction_server(tmp_path):
    policy = agpolicy()
    manager, agent, skill = _make_manager(tmp_path, policy=policy)
    assert isinstance(manager._harness_interaction_server, HarnessInteractionServer)
    assert manager._harness_interaction_server._agent is agent
    assert manager._harness_interaction_server._policy is policy


def test_harness_interaction_server_property_returns_the_same_instance(tmp_path):
    manager, _, _ = _make_manager(tmp_path)
    assert manager.harness_interaction_server is manager._harness_interaction_server


def test_host_mcp_server_property_returns_the_same_instance(tmp_path):
    manager, _, _ = _make_manager(tmp_path)
    assert manager.host_mcp_server is manager._host_mcp_server


def test_start_serves_the_mounted_mcp_server_without_a_lifespan_error():
    token = uuid.uuid4().hex[:8]
    uds_path = f"/tmp/hsm_test_{token}.sock"
    db_path = f"/tmp/hsm_test_{token}.db"
    configs = HostServerManagerConfigs(uds_path=uds_path)
    data_collector_configs = agDataCollectorConfigs(db_path=db_path)
    agconfig = agConfig({"agllm_backend": {"model": "test-model"}})
    agconfig.HostServerManagerConfigs = configs
    agconfig.agDataCollectorConfigs = data_collector_configs
    agent = SimpleNamespace(agconfig=agconfig, inbox=object())
    sandbox = SimpleNamespace()
    skill = agskill(name="s", system_prompt="p", policy=agpolicy())
    resource_pool = SimpleNamespace()
    manager = HostServerManager(agent, sandbox, skill, resource_pool)
    try:
        manager.start()

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
