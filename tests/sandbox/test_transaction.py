"""agSandbox transactions: AgentEngine.execute() holds the sandbox for a whole run, while
host tools and policy hooks called during that run execute on server worker threads and must
still reach it."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from unittest.mock import patch

import pytest

from agency.agdata import agdata
from agency.agskill import agskill
from agency.agtool import agtool
from agency.configs.agconfig import agconfig as agconfig_cls
from agency.agpolicy import agpolicy
from agency.engine.host_servers.host_interaction_server import HostInteractionServer
from agency.engine.host_servers.host_mcp_server import HostMcpServer
from agency.harness._syscall_event import agsyscallevent
from agency.sandbox.agsandbox import agSandbox

# Generous: a deadlock never finishes, so this only bounds how long a failure takes to report.
_WAIT_S = 5


class _FakeBackend:
    def __init__(self):
        self._gpu_count_requested = 0  # read by HostInteractionServer.check_syscall's GPU gate
        self.exec_threads: "list[str]" = []
        self.barrier: "threading.Barrier | None" = None

    def exec(self, command, **_kwargs):
        self.exec_threads.append(threading.current_thread().name)
        if self.barrier is not None:
            self.barrier.wait(timeout=_WAIT_S)
        return f"ran {command}", 0

    def destroy(self):
        pass


@pytest.fixture
def sandbox():
    cfg = agconfig_cls()
    cfg.sandbox.backend = "docker"
    with patch("agency.sandbox.base.shutil.which", return_value="/usr/bin/docker"):
        with patch("agency.sandbox.container._runtime_works", return_value=True):
            sb = agSandbox(str(uuid.uuid4()), agconfig=cfg)
    sb._backend = _FakeBackend()
    yield sb
    sb.destroy()


def _in_thread(fn, name):
    """Run fn on its own thread; return (thread, results) so a hang shows up as is_alive()."""
    results = []
    thread = threading.Thread(target=lambda: results.append(fn()), name=name, daemon=True)
    thread.start()
    thread.join(timeout=_WAIT_S)
    return thread, results


def test_host_tool_can_use_the_sandbox_while_the_engine_holds_it(sandbox):
    """The regression: a host tool calling sandbox.exec() from an MCP worker thread while the
    engine's thread holds the transaction used to wait forever on the engine's RLock."""
    tool = agtool(
        name="probe",
        description="runs one command in the sandbox",
        fn=lambda _arg: agdata(output=sandbox.exec("echo hi")[0]),
        params={"type": "object", "properties": {}},
    )
    skill = agskill(name="s", prompt="p", add_host_mcp_tools=[tool])
    server = HostMcpServer(sandbox, skill, resource_pool=None, data_logger=None)
    server.build_app()

    sandbox.begin_transaction()
    try:
        thread, results = _in_thread(
            lambda: asyncio.run(server._mcp_server.call_tool("probe", {})), "mcp-worker"
        )
        assert not thread.is_alive(), "host tool deadlocked on the engine's sandbox lock"
    finally:
        sandbox.end_transaction()
    assert json.loads(results[0].content[0].text) == {"output": "ran echo hi"}


def _interaction_server(sandbox, policy):
    skill = agskill(name="s", prompt="p", policy=policy)
    return HostInteractionServer(skill, data_logger=None, agname="a", sandbox=sandbox)


def test_a_tool_hook_can_use_the_sandbox_while_the_engine_holds_it(sandbox):
    """Policy hooks run on the interaction server's request thread, so they need the same
    hand-off as host tools: one that inspects the sandbox used to deadlock."""
    seen = []
    server = _interaction_server(
        sandbox,
        agpolicy(tool_hooks={"Bash": lambda _input: seen.append(sandbox.exec("pwd")) or True}),
    )
    sandbox.begin_transaction()
    try:
        thread, results = _in_thread(lambda: server.check_tool("Bash", {}), "policy-worker")
        assert not thread.is_alive(), "tool hook deadlocked on the engine's sandbox lock"
    finally:
        sandbox.end_transaction()
    assert results == [(True, None)]
    assert seen == [("ran pwd", 0)]


def test_a_syscall_hook_can_use_the_sandbox_while_the_engine_holds_it(sandbox):
    seen = []
    server = _interaction_server(
        sandbox,
        agpolicy(
            syscall_hooks={"execve": lambda _event: seen.append(sandbox.exec("git diff")) or True}
        ),
    )
    event = agsyscallevent(
        syscall="execve",
        pid=1,
        tid=1,
        argv=["python", "w.py"],
        envp=None,
        path="/usr/bin/python",
        timestamp=0.0,
        tool_name=None,
        tool_args=None,
    )
    sandbox.begin_transaction()
    try:
        thread, results = _in_thread(lambda: server.check_syscall(event), "policy-worker")
        assert not thread.is_alive(), "syscall hook deadlocked on the engine's sandbox lock"
    finally:
        sandbox.end_transaction()
    assert results[0][:2] == (True, None)
    assert seen == [("ran git diff", 0)]


def test_a_guest_does_not_wait_for_the_owner(sandbox):
    sandbox.begin_transaction()
    try:

        def guest():
            with sandbox.join_transaction():
                return sandbox.exec("true")

        thread, results = _in_thread(guest, "guest")
        assert not thread.is_alive()
        assert results == [("ran true", 0)]
    finally:
        sandbox.end_transaction()


def test_an_unrelated_thread_still_waits_for_the_transaction_to_end(sandbox):
    """Joining is opt-in: a thread that did not join is excluded exactly as before."""
    sandbox.begin_transaction()
    outsider = threading.Thread(target=lambda: sandbox.exec("true"), name="outsider", daemon=True)
    outsider.start()
    outsider.join(timeout=0.5)
    assert outsider.is_alive(), "a non-guest thread got in during another thread's transaction"
    sandbox.end_transaction()
    outsider.join(timeout=_WAIT_S)
    assert not outsider.is_alive()
    assert sandbox._backend.exec_threads == ["outsider"]


def test_guests_run_concurrently(sandbox):
    """Host tools run in parallel; admitting them must not serialize them behind each other."""
    sandbox._backend.barrier = threading.Barrier(2)
    sandbox.begin_transaction()
    try:

        def guest():
            with sandbox.join_transaction():
                return sandbox.exec("true")

        threads = [threading.Thread(target=guest, name=f"guest-{i}") for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=_WAIT_S)
        assert not any(t.is_alive() for t in threads), "guests were serialized"
    finally:
        sandbox._backend.barrier.abort()
        sandbox.end_transaction()


def test_a_guest_outliving_the_transaction_goes_back_to_taking_the_lock(sandbox):
    """end_transaction() does not wait for guests (one could block forever), so a guest still
    running afterwards must lose its pass rather than slip past the next transaction."""
    joined, second_holds, release_second = (threading.Event() for _ in range(3))

    def guest():
        with sandbox.join_transaction():
            joined.set()
            second_holds.wait(_WAIT_S)
            sandbox.exec("late")

    def second_owner():
        sandbox.begin_transaction()
        second_holds.set()
        release_second.wait(_WAIT_S)
        sandbox.end_transaction()

    sandbox.begin_transaction()
    late = threading.Thread(target=guest, name="late-guest", daemon=True)
    late.start()
    assert joined.wait(_WAIT_S)
    sandbox.end_transaction()

    second = threading.Thread(target=second_owner, daemon=True)
    second.start()
    assert second_holds.wait(_WAIT_S)
    late.join(timeout=0.5)
    assert late.is_alive(), "a stale guest ran inside a different thread's transaction"
    assert sandbox._backend.exec_threads == []

    release_second.set()
    late.join(timeout=_WAIT_S)
    second.join(timeout=_WAIT_S)
    assert sandbox._backend.exec_threads == ["late-guest"]


def test_transaction_state_survives_a_getstate_setstate_round_trip(sandbox):
    state = sandbox.__getstate__()
    assert not {"_lock", "_guest_lock", "_guest_threads", "_transaction_owner"} & state.keys()
    clone = agSandbox.__new__(agSandbox)
    clone.__setstate__(state)
    clone.begin_transaction()
    clone.end_transaction()
    clone._destroyed = True  # shares the original's backend; let the fixture tear it down
