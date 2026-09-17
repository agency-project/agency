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
        print_to_terminal=True,
        flush=False,
    ):
        self.events.append(
            (
                type,
                payload,
                call_label,
                update_latest_snapshot,
                term_message,
                flush,
                print_to_terminal,
            )
        )

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


def _make_server(policy=None, data_logger=None, is_cancelled=None, agname="agent-1", sandbox=None):
    skill = _make_skill(policy)
    data_logger = data_logger if data_logger is not None else _FakeDataLogger()
    return HostInteractionServer(
        skill, data_logger, agname, sandbox=sandbox, is_cancelled=is_cancelled
    )


class _FakeSandbox:
    """Mimics agSandbox's GPU surface (no real daemon handle here, so
    ensure_gpu_acquired() is a direct passthrough, no pause/resume)."""

    def __init__(self, gpu_count_requested=0, gpu_ids=None):
        self._gpu_count_requested = gpu_count_requested
        self._gpu_ids = list(gpu_ids or [])
        self.acquire_calls = []

    def current_gpu_ids(self):
        if self._gpu_count_requested <= 0:
            return None
        return list(self._gpu_ids)

    def ensure_gpu_acquired(self, agname, *, is_cancelled=None):
        if self._gpu_count_requested <= 0 or self._gpu_ids:
            return
        self.acquire_calls.append(self._gpu_count_requested)
        self._gpu_ids = [0]


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
    assert server.check_syscall(_make_syscall()) == (True, None, None)


def test_check_syscall_denies_by_default_when_default_to_deny_set():
    server = _make_server(policy=agpolicy(default_to_deny=True))
    assert server.check_syscall(_make_syscall()) == (False, None, None)


def test_check_syscall_uses_bool_returning_hook():
    def hook(syscall):
        return syscall.path != "/etc/passwd"

    server = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}))
    assert server.check_syscall(_make_syscall(syscall="openat", path="/tmp/x")) == (
        True,
        None,
        None,
    )
    assert server.check_syscall(_make_syscall(syscall="openat", path="/etc/passwd")) == (
        False,
        None,
        None,
    )


def test_check_syscall_uses_tuple_returning_hook():
    def hook(syscall):
        return (False, "sensitive path")

    server = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}))
    result = server.check_syscall(_make_syscall(syscall="openat", path="/etc/passwd"))
    assert result == (False, "sensitive path", None)


def test_check_syscall_denies_with_reason_when_hook_raises():
    def hook(syscall):
        raise ValueError("boom")

    server = _make_server(policy=agpolicy(syscall_hooks={"openat": hook}))
    allowed, reason, _env_overrides = server.check_syscall(_make_syscall(syscall="openat"))
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
    assert server.check_syscall(_make_syscall(syscall="execve")) == (True, None, None)


# ---------------------------------------------------------------------------
# build_app / HTTP routes
# ---------------------------------------------------------------------------


def test_build_app_check_tool_route_allows():
    server = _make_server(policy=agpolicy())
    client = TestClient(server.build_app())
    response = client.post("/check_tool", json={"tool_name": "bash", "tool_input": {"cmd": "ls"}})
    assert response.status_code == 200
    body = response.json()
    assert body["allowed"] is True
    assert body["reason"] is None
    assert isinstance(body["call_id"], str) and body["call_id"]


def test_build_app_check_tool_route_denies_with_reason():
    def hook(tool_input):
        return (False, "nope")

    server = _make_server(policy=agpolicy(tool_hooks={"bash": hook}))
    client = TestClient(server.build_app())
    response = client.post("/check_tool", json={"tool_name": "bash", "tool_input": {"cmd": "ls"}})
    assert response.status_code == 200
    body = response.json()
    assert body["allowed"] is False
    assert body["reason"] == "nope"


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
    body = response.json()
    assert body["allowed"] is True
    assert body["reason"] is None
    assert isinstance(body["call_id"], str) and body["call_id"]


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
    body = response.json()
    assert body["allowed"] is False
    assert body["reason"] == "sensitive path"


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
    assert logger.events == [
        ("warning", {"message": "bad shape"}, "dispatch", False, None, False, True)
    ]


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
    assert logger.events == [
        ("agent_state", {"state": "agent_idle"}, None, True, "[x] idle", True, True)
    ]


def test_record_event_tracks_bootstrap_progress_pings():
    """ensure_harness_daemon()'s wait loop reads this directly (same host
    process, no network needed) to reset its deadline on each fresh ping
    from the sandbox-side bootstrap script."""
    server = _make_server(data_logger=_FakeDataLogger())
    assert server.last_bootstrap_ping_ts is None

    server.record_event("harness_bootstrap_progress", {"engine": "agent-1", "step": "x"})

    assert server.last_bootstrap_ping_ts is not None


def test_record_event_leaves_ping_timestamp_alone_for_other_event_types():
    server = _make_server(data_logger=_FakeDataLogger())
    server.record_event("agent_state", {"state": "agent_idle"})
    assert server.last_bootstrap_ping_ts is None


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
    assert logger.events == [("warning", {"message": "bad shape"}, None, False, None, False, True)]


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
    assert logger.events == [
        ("agent_state", {"state": "agent_idle"}, None, True, "[x] idle", True, True)
    ]


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


# ---------------------------------------------------------------------------
# admit_tool_call / complete_tool_call / admit_syscall / complete_syscall
# ---------------------------------------------------------------------------


def test_admit_tool_call_records_call_event_and_returns_call_id():
    logger = _FakeDataLogger()
    server = _make_server(policy=agpolicy(), data_logger=logger)
    result = server.admit_tool_call("bash", {"cmd": "ls"})
    assert result["allowed"] is True
    assert result["reason"] is None
    call_id = result["call_id"]
    assert isinstance(call_id, str) and call_id
    assert logger.events == [
        (
            "tool_call",
            {
                "tool": "bash",
                "arguments": {"cmd": "ls"},
                "gpu_ids": None,
                "call_id": call_id,
                "allowed": True,
            },
            None,
            False,
            None,
            False,
            True,
        ),
        (
            "agent_state",
            {"state": "running_tool", "tool": "bash"},
            None,
            True,
            "[agent-1] TOOL    ▶  bash  args={'cmd': 'ls'}",
            True,
            False,
        ),
    ]
    assert logger.spans == []


def test_admit_tool_call_with_large_arguments_keeps_full_term_message():
    # Truncation for display is a webui-only concern (agwebui/server.py's
    # _truncate_for_display) -- the term_message written here (terminal
    # print + the agent's own db) must carry the full text.
    logger = _FakeDataLogger()
    server = _make_server(policy=agpolicy(), data_logger=logger)
    big_content = "y" * 5000

    server.admit_tool_call("write_file", {"content": big_content})

    term_message = logger.events[1][4]
    assert term_message is not None
    assert big_content in term_message


def test_admit_tool_call_denied_term_message_still_shows_args():
    logger = _FakeDataLogger()
    server = _make_server(policy=agpolicy(default_to_deny=True), data_logger=logger)

    server.admit_tool_call("bash", {"cmd": "rm -rf /"})

    term_message = logger.events[1][4]
    assert "args={'cmd': 'rm -rf /'}" in term_message
    assert "DENIED" in term_message


def test_admit_tool_call_denied_records_call_event_but_no_pending_span():
    logger = _FakeDataLogger()
    server = _make_server(policy=agpolicy(default_to_deny=True), data_logger=logger)
    result = server.admit_tool_call("bash", {"cmd": "ls"})
    assert result["allowed"] is False
    server.complete_tool_call(result["call_id"], result={"ignored": True})
    # Denied call was never stashed as pending -- completion is a no-op, and
    # no tool_result/span is recorded for a call that never really ran.
    assert [e[0] for e in logger.events] == ["tool_call", "agent_state"]
    assert logger.events[1][1] == {"state": "tool_denied", "tool": "bash"}
    assert logger.spans == []


def test_complete_tool_call_records_result_event_and_span():
    logger = _FakeDataLogger()
    server = _make_server(policy=agpolicy(), data_logger=logger)
    call_id = server.admit_tool_call("bash", {"cmd": "ls"})["call_id"]
    server.complete_tool_call(call_id, result={"stdout": "ok"})
    assert [e[0] for e in logger.events] == ["tool_call", "agent_state", "tool_result"]
    result_payload = logger.events[2][1]
    assert result_payload["tool"] == "bash"
    assert result_payload["arguments"] == {"cmd": "ls"}
    assert result_payload["result"] == {"stdout": "ok"}
    assert result_payload["call_id"] == call_id
    assert len(logger.spans) == 1
    span_name, start_ts, end_ts, attributes = logger.spans[0][:4]
    assert span_name == "tool:bash"
    assert end_ts >= start_ts
    assert attributes["call_id"] == call_id


def test_complete_tool_call_with_unknown_call_id_is_a_noop():
    logger = _FakeDataLogger()
    server = _make_server(policy=agpolicy(), data_logger=logger)
    server.complete_tool_call("does-not-exist", result={"stdout": "ok"})
    assert logger.events == []
    assert logger.spans == []


def test_complete_tool_call_fires_once_even_if_called_twice():
    logger = _FakeDataLogger()
    server = _make_server(policy=agpolicy(), data_logger=logger)
    call_id = server.admit_tool_call("bash", {})["call_id"]
    server.complete_tool_call(call_id, result="first")
    server.complete_tool_call(call_id, result="second")
    assert [e[0] for e in logger.events] == ["tool_call", "agent_state", "tool_result"]
    assert len(logger.spans) == 1


def test_admit_syscall_and_complete_syscall_record_event_and_span():
    logger = _FakeDataLogger()
    server = _make_server(policy=agpolicy(), data_logger=logger)
    result = server.admit_syscall(_make_syscall(syscall="openat", path="/tmp/x"))
    assert result["allowed"] is True
    call_id = result["call_id"]
    server.complete_syscall(call_id, return_value=3)
    assert [e[0] for e in logger.events] == ["syscall_call", "agent_state", "syscall_result"]
    result_payload = logger.events[2][1]
    assert result_payload["syscall"] == "openat"
    assert result_payload["return_value"] == 3
    assert len(logger.spans) == 1
    assert logger.spans[0][0] == "syscall:openat"


def test_build_app_complete_tool_route_records_result_and_span():
    logger = _FakeDataLogger()
    server = _make_server(policy=agpolicy(), data_logger=logger)
    client = TestClient(server.build_app())
    admitted = client.post(
        "/check_tool", json={"tool_name": "bash", "tool_input": {"cmd": "ls"}}
    ).json()
    response = client.post(
        "/complete_tool", json={"call_id": admitted["call_id"], "result": {"stdout": "ok"}}
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert [e[0] for e in logger.events] == ["tool_call", "agent_state", "tool_result"]
    assert len(logger.spans) == 1


def test_build_app_complete_syscall_route_records_result_and_span():
    logger = _FakeDataLogger()
    server = _make_server(policy=agpolicy(), data_logger=logger)
    client = TestClient(server.build_app())
    admitted = client.post(
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
    ).json()
    response = client.post(
        "/complete_syscall", json={"call_id": admitted["call_id"], "return_value": 3}
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert [e[0] for e in logger.events] == ["syscall_call", "agent_state", "syscall_result"]
    assert len(logger.spans) == 1


def test_cached_routes_do_not_retain_server_after_app_is_released():
    import gc
    import weakref

    server = _make_server()
    reference = weakref.ref(server)
    app = server.build_app()
    # FastAPI caches endpoint classification independently of the app lifetime.
    from fastapi.routing import APIRoute

    cached_endpoints = [route.endpoint for route in app.routes if isinstance(route, APIRoute)]
    del server
    gc.collect()
    assert reference() is not None, "the live app must own its server"
    del app
    gc.collect()
    assert reference() is None, "cached endpoints must not retain completed requests"
    assert cached_endpoints


# ---------------------------------------------------------------------------
# GPU gating -- admit_tool_call() physically acquires, check_tool() and
# check_syscall() never do
# ---------------------------------------------------------------------------


def test_check_tool_never_physically_acquires_a_gpu():
    """check_tool() is the pure-policy check; only admit_tool_call() (its one
    real caller) performs the blocking acquire, so a policy-only check never
    blocks."""
    sandbox = _FakeSandbox(gpu_count_requested=1)
    server = _make_server(sandbox=sandbox)
    assert server.check_tool("bash", {"cmd": "ls"}) == (True, None)
    assert sandbox.acquire_calls == []
    assert sandbox._gpu_ids == []


def test_admit_tool_call_acquires_gpu_when_reserved_but_not_yet_held():
    sandbox = _FakeSandbox(gpu_count_requested=1)
    server = _make_server(sandbox=sandbox)
    result = server.admit_tool_call("bash", {"cmd": "ls"})
    assert (result["allowed"], result["reason"]) == (True, None)
    assert sandbox.acquire_calls == [1]
    assert sandbox._gpu_ids == [0]


def test_admit_tool_call_does_not_reacquire_once_gpu_already_held():
    sandbox = _FakeSandbox(gpu_count_requested=1, gpu_ids=[3])
    server = _make_server(sandbox=sandbox)
    result = server.admit_tool_call("bash", {"cmd": "ls"})
    assert (result["allowed"], result["reason"]) == (True, None)
    assert sandbox.acquire_calls == []
    assert sandbox._gpu_ids == [3]


def test_admit_tool_call_is_a_noop_when_sandbox_never_reserved_a_gpu():
    sandbox = _FakeSandbox(gpu_count_requested=0)
    server = _make_server(sandbox=sandbox)
    result = server.admit_tool_call("bash", {"cmd": "ls"})
    assert (result["allowed"], result["reason"]) == (True, None)
    assert sandbox.acquire_calls == []


def test_admit_tool_call_is_a_noop_when_no_sandbox_is_wired_up():
    server = _make_server(sandbox=None)
    result = server.admit_tool_call("bash", {"cmd": "ls"})
    assert (result["allowed"], result["reason"]) == (True, None)


def test_admit_tool_call_records_gpu_ids_in_its_term_message_and_event():
    sandbox = _FakeSandbox(gpu_count_requested=1, gpu_ids=[2])
    data_logger = _FakeDataLogger()
    server = _make_server(sandbox=sandbox, data_logger=data_logger)
    server.admit_tool_call("bash", {"cmd": "ls"})
    call_events = [e for e in data_logger.events if e[0] == "tool_call"]
    assert call_events[0][1]["gpu_ids"] == [2]
    term_messages = [e[4] for e in data_logger.events if e[4] and "TOOL" in e[4]]
    assert any("gpu=[2]" in m for m in term_messages)


def test_check_syscall_never_triggers_physical_gpu_acquisition():
    """Regression test: gating the blocking GPU acquire at the execve/execveat
    syscall layer wedged every agent forever, because a harness's own internal
    hook subprocesses and diagnostic probes (nvidia-smi, hostname, sed, ...)
    execve constantly and are indistinguishable from the agent's own workload
    at the syscall level -- so the very first such exec after reserve_resource
    blocked forever, long before the agent's real tool call ever ran. Physical
    acquisition must happen only in admit_tool_call(), never in
    check_syscall()."""
    sandbox = _FakeSandbox(gpu_count_requested=1)
    server = _make_server(sandbox=sandbox)
    for syscall_name in ("execve", "execveat"):
        allowed, reason, env_overrides = server.check_syscall(_make_syscall(syscall=syscall_name))
        assert (allowed, reason) == (True, None)
        assert sandbox.acquire_calls == []
        assert env_overrides == {
            "CUDA_VISIBLE_DEVICES": "NoDevFiles",
            "HIP_VISIBLE_DEVICES": "NoDevFiles",
        }


def test_admit_syscall_records_env_overrides_in_its_term_message_and_event():
    sandbox = _FakeSandbox(gpu_count_requested=1, gpu_ids=[2])
    data_logger = _FakeDataLogger()
    server = _make_server(sandbox=sandbox, data_logger=data_logger)
    server.admit_syscall(_make_syscall(syscall="execve"))
    call_events = [e for e in data_logger.events if e[0] == "syscall_call"]
    assert call_events[0][1]["env_overrides"] == {
        "CUDA_VISIBLE_DEVICES": "2",
        "HIP_VISIBLE_DEVICES": "2",
    }
    term_messages = [e[4] for e in data_logger.events if e[4] and "SYSCALL" in e[4]]
    assert any("env_overrides=" in m and "CUDA_VISIBLE_DEVICES" in m for m in term_messages)


def test_check_syscall_reflects_already_acquired_gpu_ids_as_env_overrides():
    sandbox = _FakeSandbox(gpu_count_requested=1, gpu_ids=[2])
    server = _make_server(sandbox=sandbox)
    allowed, reason, env_overrides = server.check_syscall(_make_syscall(syscall="execve"))
    assert (allowed, reason) == (True, None)
    assert env_overrides == {"CUDA_VISIBLE_DEVICES": "2", "HIP_VISIBLE_DEVICES": "2"}


def test_check_syscall_has_no_env_overrides_for_non_gated_syscalls():
    sandbox = _FakeSandbox(gpu_count_requested=1, gpu_ids=[2])
    server = _make_server(sandbox=sandbox)
    allowed, reason, env_overrides = server.check_syscall(_make_syscall(syscall="openat"))
    assert (allowed, reason) == (True, None)
    assert env_overrides is None


def test_check_syscall_has_no_env_overrides_when_sandbox_never_reserved_a_gpu():
    sandbox = _FakeSandbox(gpu_count_requested=0)
    server = _make_server(sandbox=sandbox)
    allowed, reason, env_overrides = server.check_syscall(_make_syscall(syscall="execve"))
    assert (allowed, reason) == (True, None)
    assert env_overrides is None
