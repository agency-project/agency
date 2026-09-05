# Tests for host_interaction_server.py -- the tool/syscall mediation point.

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agency._agent_control import AgentControl
from agency.agpolicy import agpolicy
from agency.engine.host_servers.host_interaction_server import HostInteractionServer
from agency.harness._syscall_event import agsyscallevent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeDataLogger:
    def __init__(self):
        self.events = []
        self.spans = []

    def record_event(
        self,
        type,
        payload,
        call_label=None,
        update_latest_snapshot=False,
        term_message=None,
        flush=False,
    ):
        self.events.append((type, payload, call_label, update_latest_snapshot, term_message, flush))

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


def _make_server(policy=None, data_logger=None, invocation=None):
    skill = _make_skill(policy)
    data_logger = data_logger if data_logger is not None else _FakeDataLogger()
    return HostInteractionServer(skill, data_logger, invocation=invocation)


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


def test_checkpoint_is_bound_to_the_server_invocation_and_serializes_messages():
    class Invocation:
        def __init__(self):
            self.calls = []

        def _checkpoint(self, boundary_id, *, allow_messages, phase):
            self.calls.append((boundary_id, allow_messages, phase))
            return SimpleNamespace(
                cancelled=False,
                destroyed=False,
                invocation_messages=[SimpleNamespace(sequence=7, content="continue carefully")],
            )

    invocation = Invocation()
    server = _make_server(invocation=invocation)

    result = server.checkpoint("native:tool:0", allow_messages=True, phase="boundary")

    assert invocation.calls == [("native:tool:0", True, "boundary")]
    assert result == {
        "cancelled": False,
        "destroyed": False,
        "invocation_messages": [{"sequence": 7, "content": "continue carefully"}],
    }


def test_checkpoint_route_validates_wire_shape_and_uses_noop_without_invocation():
    client = TestClient(_make_server().build_app())

    invalid = client.post(
        "/checkpoint",
        json={"boundary_id": "", "allow_messages": "yes", "phase": ""},
    )
    valid = client.post(
        "/checkpoint",
        json={"boundary_id": "model:1", "allow_messages": False, "phase": "model"},
    )

    assert invalid.status_code == 400
    assert valid.status_code == 200
    assert valid.json() == {
        "cancelled": False,
        "destroyed": False,
        "invocation_messages": [],
    }


def test_paused_checkpoint_disconnect_wakes_and_joins_its_worker():
    control = AgentControl()
    invocation = control.begin_invocation("native")
    invocation.redirect("keep for reconnect")
    invocation.pause()
    worker_exited = threading.Event()
    original_checkpoint = invocation._checkpoint_interruptibly

    def checkpoint_with_exit_signal(*args, **kwargs):
        try:
            return original_checkpoint(*args, **kwargs)
        finally:
            worker_exited.set()

    invocation._checkpoint_interruptibly = checkpoint_with_exit_signal
    app = _make_server(invocation=invocation).build_app()

    async def scenario() -> list[dict]:
        receive_queue: "asyncio.Queue[dict]" = asyncio.Queue()
        body = json.dumps(
            {
                "boundary_id": "native:tool:1",
                "allow_messages": True,
                "phase": "boundary",
            }
        ).encode()
        await receive_queue.put({"type": "http.request", "body": body, "more_body": False})
        sent = []

        async def receive():
            return await receive_queue.get()

        async def send(message):
            sent.append(message)

        request_task = asyncio.create_task(
            app(
                {
                    "type": "http",
                    "asgi": {"version": "3.0", "spec_version": "2.3"},
                    "http_version": "1.1",
                    "method": "POST",
                    "scheme": "http",
                    "path": "/checkpoint",
                    "raw_path": b"/checkpoint",
                    "query_string": b"",
                    "root_path": "",
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                    "server": ("test", 80),
                    "client": ("test", 123),
                    "state": {},
                },
                receive,
                send,
            )
        )

        def wait_until_paused() -> bool:
            with control._condition:
                return control._condition.wait_for(
                    lambda: invocation._phase == "paused",
                    timeout=2.0,
                )

        assert await asyncio.to_thread(wait_until_paused)
        await receive_queue.put({"type": "http.disconnect"})
        await asyncio.wait_for(request_task, timeout=2.0)
        return sent

    sent = asyncio.run(scenario())

    assert worker_exited.is_set()
    assert any(
        message["type"] == "http.response.start" and message["status"] == 499 for message in sent
    )
    assert invocation.is_pause_requested() is True
    assert control.is_paused_actual() is False

    invocation.resume()
    retry = invocation._checkpoint(
        "native:tool:1",
        allow_messages=True,
        phase="boundary",
    )
    assert [entry.content for entry in retry.invocation_messages] == ["keep for reconnect"]


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


def test_record_event_delegates_to_data_logger():
    logger = _FakeDataLogger()
    server = _make_server(data_logger=logger)
    server.record_event("warning", {"message": "bad shape"}, call_label="dispatch")
    assert logger.events == [("warning", {"message": "bad shape"}, "dispatch", False, None, False)]


def test_record_event_forwards_term_message_and_flush():
    logger = _FakeDataLogger()
    server = _make_server(data_logger=logger)
    server.record_event(
        "agent_state",
        {"state": "agent_idle"},
        update_latest_snapshot=True,
        term_message="[x] idle",
        flush=True,
    )
    assert logger.events == [("agent_state", {"state": "agent_idle"}, None, True, "[x] idle", True)]


def test_record_span_delegates_to_data_logger():
    logger = _FakeDataLogger()
    server = _make_server(data_logger=logger)
    server.record_span("llm:attempt", 0.0, 1.0, {"model": "x"}, cpu_ms=5.0)
    assert logger.spans == [("llm:attempt", 0.0, 1.0, {"model": "x"}, 5.0, None, None, None, None)]


def test_build_app_record_event_route_delegates_to_data_logger():
    logger = _FakeDataLogger()
    server = _make_server(data_logger=logger)
    client = TestClient(server.build_app())
    response = client.post(
        "/record_event", json={"type": "warning", "payload": {"message": "bad shape"}}
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert logger.events == [("warning", {"message": "bad shape"}, None, False, None, False)]


def test_build_app_record_event_route_forwards_term_message_and_flush():
    logger = _FakeDataLogger()
    server = _make_server(data_logger=logger)
    client = TestClient(server.build_app())
    response = client.post(
        "/record_event",
        json={
            "type": "agent_state",
            "payload": {"state": "agent_idle"},
            "update_latest_snapshot": True,
            "term_message": "[x] idle",
            "flush": True,
        },
    )
    assert response.status_code == 200
    assert logger.events == [("agent_state", {"state": "agent_idle"}, None, True, "[x] idle", True)]


def test_build_app_has_no_final_attempt_result_callback_route():
    server = _make_server()
    client = TestClient(server.build_app())
    response = client.post("/report_attempt_result", json={"ok": True, "final_text": "hi"})
    assert response.status_code == 404


def test_build_app_record_span_route_delegates_to_data_logger():
    logger = _FakeDataLogger()
    server = _make_server(data_logger=logger)
    client = TestClient(server.build_app())
    response = client.post(
        "/record_span",
        json={"name": "llm:attempt", "start_ts": 0.0, "end_ts": 1.0, "attributes": {"model": "x"}},
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert logger.spans == [("llm:attempt", 0.0, 1.0, {"model": "x"}, None, None, None, None, None)]


def test_external_tool_admission_fences_redirects_before_policy_hook():
    handle = AgentControl().begin_invocation("external")
    called = []
    server = _make_server(
        policy=agpolicy(tool_hooks={"tool": lambda args: called.append(args) or True}),
        invocation=handle,
    )
    assert server.check_tool("tool", {"first": True})[0]
    handle.redirect("stop these actions")
    allowed, reason = server.check_tool("tool", {"stale": True})
    assert not allowed
    assert "Return to the model" in reason
    assert called == [{"first": True}]
    snapshot = handle._checkpoint("model", allow_messages=True, phase="model")
    assert not server.check_tool("tool", {})[0]
    handle._acknowledge_redirects(snapshot.invocation_messages)
    assert server.check_tool("tool", {"revised": True})[0]
