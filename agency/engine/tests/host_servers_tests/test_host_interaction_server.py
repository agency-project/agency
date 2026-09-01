# Tests for host_interaction_server.py -- the tool/syscall mediation point.

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from agency.agpolicy import agpolicy
from agency.engine.host_servers.host_interaction_server import HostInteractionServer
from agency.harness._syscall_event import agsyscallevent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeDataCollector:
    def __init__(self):
        self.events = []
        self.spans = []

    def record_event(
        self, type, payload, call_label=None, overwrite=False, term_message=None, flush=False
    ):
        self.events.append((type, payload, call_label, overwrite, term_message, flush))

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


def _make_skill(policy=None):
    return SimpleNamespace(policy=policy if policy is not None else agpolicy())


def _make_server(policy=None, data_collector=None):
    skill = _make_skill(policy)
    data_collector = data_collector if data_collector is not None else _FakeDataCollector()
    return HostInteractionServer(skill, data_collector)


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
# __init__
# ---------------------------------------------------------------------------


def test_init_stores_skills_policy():
    policy = agpolicy()
    server = _make_server(policy=policy)
    assert server._policy is policy


# ---------------------------------------------------------------------------
# check_tool
# ---------------------------------------------------------------------------


def test_check_tool_allows_by_default_when_no_hooks_and_not_default_to_deny():
    server = _make_server(policy=agpolicy())
    assert server.check_tool("bash", {"cmd": "ls"}) == (True, None)


def test_check_tool_denies_by_default_when_default_to_deny_set():
    server = _make_server(policy=agpolicy(default_to_deny=True))
    assert server.check_tool("bash", {"cmd": "ls"}) == (False, None)


def test_check_tool_uses_bool_returning_hook():
    def hook(tool_input):
        return tool_input["cmd"] != "rm -rf /"

    server = _make_server(policy=agpolicy(tool_hooks={"bash": hook}))
    assert server.check_tool("bash", {"cmd": "ls"}) == (True, None)
    assert server.check_tool("bash", {"cmd": "rm -rf /"}) == (False, None)


def test_check_tool_uses_tuple_returning_hook():
    def hook(tool_input):
        return (False, "blocked by policy")

    server = _make_server(policy=agpolicy(tool_hooks={"bash": hook}))
    assert server.check_tool("bash", {"cmd": "ls"}) == (False, "blocked by policy")


def test_check_tool_passes_through_tool_input_unmodified():
    seen = {}

    def hook(tool_input):
        seen["tool_input"] = tool_input
        return True

    server = _make_server(policy=agpolicy(tool_hooks={"bash": hook}))
    tool_input = {"cmd": "ls", "cwd": "/tmp"}
    server.check_tool("bash", tool_input)
    assert seen["tool_input"] is tool_input


def test_check_tool_denies_with_reason_when_hook_raises():
    def hook(tool_input):
        raise ValueError("boom")

    server = _make_server(policy=agpolicy(tool_hooks={"bash": hook}))
    allowed, reason = server.check_tool("bash", {"cmd": "ls"})
    assert allowed is False
    assert "boom" in reason


def test_check_tool_falls_back_to_default_for_unregistered_tool_name():
    def hook(tool_input):
        return False

    server = _make_server(policy=agpolicy(tool_hooks={"bash": hook}, default_to_deny=False))
    assert server.check_tool("other_tool", {}) == (True, None)


def test_check_tool_empty_hooks_dict_falls_back_to_default():
    server = _make_server(policy=agpolicy(tool_hooks={}, default_to_deny=True))
    assert server.check_tool("bash", {}) == (False, None)


# ---------------------------------------------------------------------------
# check_syscall
# ---------------------------------------------------------------------------


def test_check_syscall_allows_by_default_when_no_hooks_and_not_default_to_deny():
    server = _make_server(policy=agpolicy())
    assert server.check_syscall(_make_syscall()) == (True, None)


def test_check_syscall_denies_by_default_when_default_to_deny_set():
    server = _make_server(policy=agpolicy(default_to_deny=True))
    assert server.check_syscall(_make_syscall()) == (False, None)


def test_check_syscall_uses_bool_returning_hook():
    def hook(syscall):
        return syscall.path != "/etc/passwd"

    server = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}))
    assert server.check_syscall(_make_syscall(syscall="openat", path="/tmp/x")) == (True, None)
    assert server.check_syscall(_make_syscall(syscall="openat", path="/etc/passwd")) == (
        False,
        None,
    )


def test_check_syscall_uses_tuple_returning_hook():
    def hook(syscall):
        return (False, "sensitive path")

    server = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}))
    result = server.check_syscall(_make_syscall(syscall="openat", path="/etc/passwd"))
    assert result == (False, "sensitive path")


def test_check_syscall_denies_with_reason_when_hook_raises():
    def hook(syscall):
        raise ValueError("boom")

    server = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}))
    allowed, reason = server.check_syscall(_make_syscall(syscall="openat"))
    assert allowed is False
    assert "boom" in reason


def test_check_syscall_passes_the_full_event_object_to_the_hook():
    seen = {}

    def hook(syscall):
        seen["syscall"] = syscall
        return True

    server = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}))
    event = _make_syscall(syscall="openat", path="/tmp/x")
    server.check_syscall(event)
    assert seen["syscall"] is event


def test_check_syscall_falls_back_to_default_for_unregistered_syscall_name():
    def hook(syscall):
        return False

    server = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}, default_to_deny=False))
    assert server.check_syscall(_make_syscall(syscall="execve")) == (True, None)


# ---------------------------------------------------------------------------
# build_app / HTTP routes
# ---------------------------------------------------------------------------


def test_build_app_check_tool_route_allows():
    server = _make_server(policy=agpolicy())
    client = TestClient(server.build_app())
    response = client.post("/check_tool", json={"tool_name": "bash", "tool_input": {"cmd": "ls"}})
    assert response.status_code == 200
    assert response.json() == {"allowed": True, "reason": None}


def test_build_app_check_tool_route_denies_with_reason():
    def hook(tool_input):
        return (False, "nope")

    server = _make_server(policy=agpolicy(tool_hooks={"bash": hook}))
    client = TestClient(server.build_app())
    response = client.post("/check_tool", json={"tool_name": "bash", "tool_input": {"cmd": "ls"}})
    assert response.status_code == 200
    assert response.json() == {"allowed": False, "reason": "nope"}


def test_build_app_check_tool_route_denies_when_hook_raises():
    def hook(tool_input):
        raise ValueError("boom")

    server = _make_server(policy=agpolicy(tool_hooks={"bash": hook}))
    client = TestClient(server.build_app())
    response = client.post("/check_tool", json={"tool_name": "bash", "tool_input": {"cmd": "ls"}})
    assert response.status_code == 200
    body = response.json()
    assert body["allowed"] is False
    assert "boom" in body["reason"]


def test_build_app_has_no_daemon_command_polling_route():
    server = _make_server()
    client = TestClient(server.build_app())
    response = client.post("/check_inbox")
    assert response.status_code == 404


def test_build_app_check_syscall_route_allows():
    server = _make_server(policy=agpolicy())
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

    server = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}))
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

    server = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}))
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
# record_event / record_span
# ---------------------------------------------------------------------------


def test_record_event_delegates_to_data_collector():
    collector = _FakeDataCollector()
    server = _make_server(data_collector=collector)
    server.record_event("warning", {"message": "bad shape"}, call_label="dispatch")
    assert collector.events == [
        ("warning", {"message": "bad shape"}, "dispatch", False, None, False)
    ]


def test_record_event_forwards_term_message_and_flush():
    collector = _FakeDataCollector()
    server = _make_server(data_collector=collector)
    server.record_event(
        "agent_state",
        {"state": "agent_idle"},
        overwrite=True,
        term_message="[x] idle",
        flush=True,
    )
    assert collector.events == [
        ("agent_state", {"state": "agent_idle"}, None, True, "[x] idle", True)
    ]


def test_record_span_delegates_to_data_collector():
    collector = _FakeDataCollector()
    server = _make_server(data_collector=collector)
    server.record_span("llm:attempt", 0.0, 1.0, {"model": "x"}, cpu_ms=5.0)
    assert collector.spans == [
        ("llm:attempt", 0.0, 1.0, {"model": "x"}, 5.0, None, None, None, None)
    ]


def test_build_app_record_event_route_delegates_to_data_collector():
    collector = _FakeDataCollector()
    server = _make_server(data_collector=collector)
    client = TestClient(server.build_app())
    response = client.post(
        "/record_event", json={"type": "warning", "payload": {"message": "bad shape"}}
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert collector.events == [("warning", {"message": "bad shape"}, None, False, None, False)]


def test_build_app_record_event_route_forwards_term_message_and_flush():
    collector = _FakeDataCollector()
    server = _make_server(data_collector=collector)
    client = TestClient(server.build_app())
    response = client.post(
        "/record_event",
        json={
            "type": "agent_state",
            "payload": {"state": "agent_idle"},
            "overwrite": True,
            "term_message": "[x] idle",
            "flush": True,
        },
    )
    assert response.status_code == 200
    assert collector.events == [
        ("agent_state", {"state": "agent_idle"}, None, True, "[x] idle", True)
    ]


def test_build_app_has_no_final_attempt_result_callback_route():
    server = _make_server()
    client = TestClient(server.build_app())
    response = client.post("/report_attempt_result", json={"ok": True, "final_text": "hi"})
    assert response.status_code == 404


def test_build_app_record_span_route_delegates_to_data_collector():
    collector = _FakeDataCollector()
    server = _make_server(data_collector=collector)
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
