# Tests for host_server_manager.py -- the composition root owning the sub-servers.

from __future__ import annotations

from types import SimpleNamespace

from agency.agpolicy import agpolicy
from agency.engine.host_servers.harness_interaction_server import HarnessInteractionServer
from agency.engine.host_servers.host_server_manager import (
    HostServerManager,
    HostServerManagerConfigs,
)


def _make_manager(tmp_path, policy=None):
    configs = HostServerManagerConfigs(uds_path=str(tmp_path / "host.sock"))
    agconfig = SimpleNamespace(HostServerManagerConfigs=configs)
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


def test_does_not_expose_a_public_harness_interaction_server_property(tmp_path):
    manager, _, _ = _make_manager(tmp_path)
    assert not hasattr(manager, "harness_interaction_server")
