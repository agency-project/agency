# Tests for harness_interaction_server.py -- the tool/syscall mediation point.

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agency.agpolicy import agpolicy
from agency.engine.host_servers.harness_interaction_server import HarnessInteractionServer
from agency.harness._syscall_event import agsyscallevent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(agconfig=None):
    return SimpleNamespace(
        agconfig=agconfig if agconfig is not None else SimpleNamespace(), inbox=object()
    )


def _make_skill(policy=None):
    return SimpleNamespace(policy=policy if policy is not None else agpolicy())


def _make_server(policy=None, agconfig=None):
    agent = _make_agent(agconfig)
    skill = _make_skill(policy)
    return HarnessInteractionServer(agent, skill), agent


def _make_syscall(
    syscall="openat", path=None, argv=None, envp=None, tool_name=None, tool_args=None
):
    return agsyscallevent(
        syscall=syscall,
        pid=1,
        tid=1,
        argv=argv,
        envp=envp,
        path=path,
        timestamp=0.0,
        tool_name=tool_name,
        tool_args=tool_args,
    )


# ---------------------------------------------------------------------------
# __init__ / set_config
# ---------------------------------------------------------------------------


def test_init_stores_agent_and_skills_policy():
    policy = agpolicy()
    server, agent = _make_server(policy=policy)
    assert server._agent is agent
    assert server._policy is policy


def test_init_calls_set_config_with_agents_agconfig():
    agconfig = SimpleNamespace(marker="the-config")
    server, agent = _make_server(agconfig=agconfig)
    assert server._agconfig is agconfig


def test_set_config_replaces_the_stored_agconfig():
    server, _ = _make_server()
    new_agconfig = SimpleNamespace(marker="new")
    server.set_config(new_agconfig)
    assert server._agconfig is new_agconfig


# ---------------------------------------------------------------------------
# check_tool
# ---------------------------------------------------------------------------


def test_check_tool_allows_by_default_when_no_hooks_and_not_default_to_deny():
    server, _ = _make_server(policy=agpolicy())
    assert server.check_tool("bash", {"cmd": "ls"}) == (True, None)


def test_check_tool_denies_by_default_when_default_to_deny_set():
    server, _ = _make_server(policy=agpolicy(default_to_deny=True))
    assert server.check_tool("bash", {"cmd": "ls"}) == (False, None)


def test_check_tool_uses_bool_returning_hook():
    def hook(tool_input):
        return tool_input["cmd"] != "rm -rf /"

    server, _ = _make_server(policy=agpolicy(tool_hooks={"bash": hook}))
    assert server.check_tool("bash", {"cmd": "ls"}) == (True, None)
    assert server.check_tool("bash", {"cmd": "rm -rf /"}) == (False, None)


def test_check_tool_uses_tuple_returning_hook():
    def hook(tool_input):
        return (False, "blocked by policy")

    server, _ = _make_server(policy=agpolicy(tool_hooks={"bash": hook}))
    assert server.check_tool("bash", {"cmd": "ls"}) == (False, "blocked by policy")


def test_check_tool_passes_through_tool_input_unmodified():
    seen = {}

    def hook(tool_input):
        seen["tool_input"] = tool_input
        return True

    server, _ = _make_server(policy=agpolicy(tool_hooks={"bash": hook}))
    tool_input = {"cmd": "ls", "cwd": "/tmp"}
    server.check_tool("bash", tool_input)
    assert seen["tool_input"] is tool_input


def test_check_tool_falls_back_to_default_for_unregistered_tool_name():
    def hook(tool_input):
        return False

    server, _ = _make_server(policy=agpolicy(tool_hooks={"bash": hook}, default_to_deny=False))
    assert server.check_tool("other_tool", {}) == (True, None)


def test_check_tool_empty_hooks_dict_falls_back_to_default():
    server, _ = _make_server(policy=agpolicy(tool_hooks={}, default_to_deny=True))
    assert server.check_tool("bash", {}) == (False, None)


# ---------------------------------------------------------------------------
# check_syscall
# ---------------------------------------------------------------------------


def test_check_syscall_allows_by_default_when_no_hooks_and_not_default_to_deny():
    server, _ = _make_server(policy=agpolicy())
    assert server.check_syscall(_make_syscall()) == (True, None)


def test_check_syscall_denies_by_default_when_default_to_deny_set():
    server, _ = _make_server(policy=agpolicy(default_to_deny=True))
    assert server.check_syscall(_make_syscall()) == (False, None)


def test_check_syscall_uses_bool_returning_hook():
    def hook(syscall):
        return syscall.path != "/etc/passwd"

    server, _ = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}))
    assert server.check_syscall(_make_syscall(syscall="openat", path="/tmp/x")) == (True, None)
    assert server.check_syscall(_make_syscall(syscall="openat", path="/etc/passwd")) == (
        False,
        None,
    )


def test_check_syscall_uses_tuple_returning_hook():
    def hook(syscall):
        return (False, "sensitive path")

    server, _ = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}))
    result = server.check_syscall(_make_syscall(syscall="openat", path="/etc/passwd"))
    assert result == (False, "sensitive path")


def test_check_syscall_passes_the_full_event_object_to_the_hook():
    seen = {}

    def hook(syscall):
        seen["syscall"] = syscall
        return True

    server, _ = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}))
    event = _make_syscall(syscall="openat", path="/tmp/x")
    server.check_syscall(event)
    assert seen["syscall"] is event


def test_check_syscall_falls_back_to_default_for_unregistered_syscall_name():
    def hook(syscall):
        return False

    server, _ = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}, default_to_deny=False))
    assert server.check_syscall(_make_syscall(syscall="execve")) == (True, None)


# ---------------------------------------------------------------------------
# check_inbox (unimplemented stub)
# ---------------------------------------------------------------------------


def test_check_inbox_raises_not_implemented():
    server, _ = _make_server()
    with pytest.raises(NotImplementedError):
        server.check_inbox()


# ---------------------------------------------------------------------------
# build_app / HTTP routes
# ---------------------------------------------------------------------------


def test_build_app_check_tool_route_allows():
    server, _ = _make_server(policy=agpolicy())
    client = TestClient(server.build_app())
    response = client.post("/check_tool", json={"tool_name": "bash", "tool_input": {"cmd": "ls"}})
    assert response.status_code == 200
    assert response.json() == {"allowed": True, "reason": None}


def test_build_app_check_tool_route_denies_with_reason():
    def hook(tool_input):
        return (False, "nope")

    server, _ = _make_server(policy=agpolicy(tool_hooks={"bash": hook}))
    client = TestClient(server.build_app())
    response = client.post("/check_tool", json={"tool_name": "bash", "tool_input": {"cmd": "ls"}})
    assert response.status_code == 200
    assert response.json() == {"allowed": False, "reason": "nope"}


def test_build_app_check_inbox_route_surfaces_the_not_implemented_stub():
    server, _ = _make_server()
    client = TestClient(server.build_app(), raise_server_exceptions=False)
    response = client.post("/check_inbox")
    assert response.status_code == 500


def test_build_app_has_no_route_for_check_syscall():
    server, _ = _make_server()
    app = server.build_app()
    paths = {route.path for route in app.routes}
    assert "/check_syscall" not in paths
