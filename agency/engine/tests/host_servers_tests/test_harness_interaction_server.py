# Tests for harness_interaction_server.py -- the tool/syscall mediation point.

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from agency.agpolicy import agpolicy
from agency.engine.host_servers.harness_interaction_server import HarnessInteractionServer
from agency.harness._syscall_event import agsyscallevent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeDataCollector:
    def __init__(self):
        self.events = []
        self.spans = []

    def record_event(self, type, payload, call_label=None, do_update=False):
        self.events.append((type, payload, call_label, do_update))

    def record_span(
        self,
        name,
        start_ts,
        end_ts,
        attributes,
        cpu_ms=None,
        runqueue_ms=None,
        blocked_ms=None,
        parent=None,
        call_label=None,
    ):
        self.spans.append(
            (
                name,
                start_ts,
                end_ts,
                attributes,
                cpu_ms,
                runqueue_ms,
                blocked_ms,
                parent,
                call_label,
            )
        )


def _make_agent(agconfig=None, drain_inbox=None):
    agent = SimpleNamespace(
        agconfig=agconfig if agconfig is not None else SimpleNamespace(),
        inbox=object(),
        _state=SimpleNamespace(update_state=lambda *a, **kw: None),
    )
    agent._drain_inbox = drain_inbox if drain_inbox is not None else (lambda messages: False)
    return agent


def _make_skill(policy=None):
    return SimpleNamespace(policy=policy if policy is not None else agpolicy())


def _make_server(policy=None, agconfig=None, drain_inbox=None, data_collector=None):
    agent = _make_agent(agconfig, drain_inbox)
    skill = _make_skill(policy)
    data_collector = data_collector if data_collector is not None else _FakeDataCollector()
    return HarnessInteractionServer(agent, skill, data_collector), agent


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


def test_check_tool_denies_with_reason_when_hook_raises():
    def hook(tool_input):
        raise ValueError("boom")

    server, _ = _make_server(policy=agpolicy(tool_hooks={"bash": hook}))
    allowed, reason = server.check_tool("bash", {"cmd": "ls"})
    assert allowed is False
    assert "boom" in reason


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


def test_check_syscall_denies_with_reason_when_hook_raises():
    def hook(syscall):
        raise ValueError("boom")

    server, _ = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}))
    allowed, reason = server.check_syscall(_make_syscall(syscall="openat"))
    assert allowed is False
    assert "boom" in reason


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
# check_inbox
# ---------------------------------------------------------------------------


def test_check_inbox_returns_empty_list_when_nothing_pending():
    server, _ = _make_server()
    assert server.check_inbox() == []


def test_check_inbox_returns_messages_drained_from_the_agent():
    def drain_inbox(messages):
        messages.append({"role": "user", "content": "hello"})
        messages.append({"role": "user", "content": "world"})
        return True

    server, _ = _make_server(drain_inbox=drain_inbox)
    assert server.check_inbox() == [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "world"},
    ]


def test_check_inbox_passes_a_fresh_list_to_agent_drain_inbox_each_call():
    seen_lists = []

    def drain_inbox(messages):
        seen_lists.append(messages)
        messages.append({"role": "user", "content": "x"})
        return True

    server, _ = _make_server(drain_inbox=drain_inbox)
    first = server.check_inbox()
    second = server.check_inbox()
    assert first == [{"role": "user", "content": "x"}]
    assert second == [{"role": "user", "content": "x"}]
    assert seen_lists[0] is not seen_lists[1]


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


def test_build_app_check_tool_route_denies_when_hook_raises():
    def hook(tool_input):
        raise ValueError("boom")

    server, _ = _make_server(policy=agpolicy(tool_hooks={"bash": hook}))
    client = TestClient(server.build_app())
    response = client.post("/check_tool", json={"tool_name": "bash", "tool_input": {"cmd": "ls"}})
    assert response.status_code == 200
    body = response.json()
    assert body["allowed"] is False
    assert "boom" in body["reason"]


def test_build_app_check_inbox_route_returns_drained_messages():
    def drain_inbox(messages):
        messages.append({"role": "user", "content": "hello"})
        return True

    server, _ = _make_server(drain_inbox=drain_inbox)
    client = TestClient(server.build_app())
    response = client.post("/check_inbox")
    assert response.status_code == 200
    assert response.json() == {"messages": [{"role": "user", "content": "hello"}]}


def test_build_app_check_inbox_route_returns_empty_list_when_nothing_pending():
    server, _ = _make_server()
    client = TestClient(server.build_app())
    response = client.post("/check_inbox")
    assert response.status_code == 200
    assert response.json() == {"messages": []}


def test_build_app_check_syscall_route_allows():
    server, _ = _make_server(policy=agpolicy())
    client = TestClient(server.build_app())
    response = client.post(
        "/check_syscall",
        json={
            "syscall": "openat",
            "pid": 1,
            "tid": 1,
            "argv": None,
            "envp": None,
            "path": "/tmp/x",
            "timestamp": 0.0,
        },
    )
    assert response.status_code == 200
    assert response.json() == {"allowed": True, "reason": None}


def test_build_app_check_syscall_route_denies_with_reason():
    def hook(syscall):
        return (False, "sensitive path")

    server, _ = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}))
    client = TestClient(server.build_app())
    response = client.post(
        "/check_syscall",
        json={
            "syscall": "openat",
            "pid": 1,
            "tid": 1,
            "argv": None,
            "envp": None,
            "path": "/etc/passwd",
            "timestamp": 0.0,
        },
    )
    assert response.status_code == 200
    assert response.json() == {"allowed": False, "reason": "sensitive path"}


def test_build_app_check_syscall_route_denies_when_hook_raises():
    def hook(syscall):
        raise ValueError("boom")

    server, _ = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}))
    client = TestClient(server.build_app())
    response = client.post(
        "/check_syscall",
        json={
            "syscall": "openat",
            "pid": 1,
            "tid": 1,
            "argv": None,
            "envp": None,
            "path": "/tmp/x",
            "timestamp": 0.0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["allowed"] is False
    assert "boom" in body["reason"]


# ---------------------------------------------------------------------------
# update_state
# ---------------------------------------------------------------------------


def test_update_state_calls_agent_state_update_state():
    seen = []
    server, agent = _make_server()
    agent._state.update_state = lambda *a, **kw: seen.append((a, kw))
    server.update_state("skill", skill="s", tool="bash")
    assert seen == [(("skill", "s", "bash"), {})]


def test_build_app_update_state_route_calls_agent_state():
    seen = []
    server, agent = _make_server()
    agent._state.update_state = lambda *a, **kw: seen.append((a, kw))
    client = TestClient(server.build_app())
    response = client.post("/update_state", json={"state": "paused", "skill": "s", "tool": None})
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert seen == [(("paused", "s", None), {})]


# ---------------------------------------------------------------------------
# record_event / record_span
# ---------------------------------------------------------------------------


def test_record_event_delegates_to_data_collector():
    collector = _FakeDataCollector()
    server, _ = _make_server(data_collector=collector)
    server.record_event("warning", {"message": "bad shape"}, call_label="dispatch")
    assert collector.events == [("warning", {"message": "bad shape"}, "dispatch", False)]


def test_record_span_delegates_to_data_collector():
    collector = _FakeDataCollector()
    server, _ = _make_server(data_collector=collector)
    server.record_span("llm:attempt", 0.0, 1.0, {"model": "x"}, cpu_ms=5.0)
    assert collector.spans == [
        ("llm:attempt", 0.0, 1.0, {"model": "x"}, 5.0, None, None, None, None)
    ]


def test_build_app_record_event_route_delegates_to_data_collector():
    collector = _FakeDataCollector()
    server, _ = _make_server(data_collector=collector)
    client = TestClient(server.build_app())
    response = client.post(
        "/record_event", json={"type": "warning", "payload": {"message": "bad shape"}}
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert collector.events == [("warning", {"message": "bad shape"}, None, False)]


def test_build_app_record_span_route_delegates_to_data_collector():
    collector = _FakeDataCollector()
    server, _ = _make_server(data_collector=collector)
    client = TestClient(server.build_app())
    response = client.post(
        "/record_span",
        json={"name": "llm:attempt", "start_ts": 0.0, "end_ts": 1.0, "attributes": {"model": "x"}},
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert collector.spans == [
        ("llm:attempt", 0.0, 1.0, {"model": "x"}, None, None, None, None, None)
    ]
