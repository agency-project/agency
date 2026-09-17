"""Real Linux ptrace ownership transfer without requiring ZFS or a runtime."""

import os
import time
from types import SimpleNamespace

import pytest

from agency.configs.agconfig import agconfig
from agency.harness.ptrace.supervisor import agProxyPtrace, ptrace_available

pytestmark = pytest.mark.skipif(not ptrace_available(), reason="requires Linux x86_64 ptrace")


def wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("Timed out waiting for live PTY state")


@pytest.mark.timeout(90)
def test_detach_reattach_keeps_processes_stopped_until_policy_is_bound(tmp_path):
    from agency.harness.ptrace._checkpoint_handoff import task_status

    cfg = agconfig()
    cfg.sandbox.checkpoint_backend = "cow_zfs"
    cfg.sandbox.checkpoint_fast_resume = True
    decisions = []

    def check(_agent, event):
        decisions.append(event)
        return not (event.argv and event.argv[0] == "/usr/bin/touch")

    handle = agProxyPtrace(cfg, allow_initial_exec=True).launch(
        ["/bin/bash", "--noprofile", "--norc", "-i"],
        dict(os.environ, PS1="READY>"),
        cwd=str(tmp_path),
        pty_size=(120, 36),
        policy=SimpleNamespace(check=check),
        ag=None,
    )
    try:
        wait_for(lambda: "READY>" in handle.terminal_output())
        handle.write_terminal(b"stty -echo; TOKEN=$RANDOM; sleep 60 &\r")
        wait_for(lambda: len(handle.pids()) >= 2)
        # `sleep 60 &` forking is only proof `stty -echo` has *run* (it's
        # earlier in the same sequential command line) -- it's not proof the
        # tty driver has settled into non-echoing before the loop below
        # starts typing into it. A brief buffer here avoids the first
        # cycle's write occasionally getting echoed back and tripping the
        # "still stopped" assertion below.
        time.sleep(0.2)
        root = handle.root_pid
        for cycle in range(3):
            handle.checkpoint_detach()
            pids = handle.pids()
            assert root in pids
            for pid in pids:
                state = task_status(pid)
                assert int(state["TracerPid"]) == 0
                assert state["State"].strip().startswith("T")
            marker = f"ROUND_{cycle}"
            handle.write_terminal(
                f"/usr/bin/touch forbidden; printf '{marker}=%s\\n' $TOKEN\r".encode()
            )
            handle.checkpoint_reattach()
            assert all(int(task_status(pid)["TracerPid"]) != 0 for pid in pids)
            time.sleep(0.3)  # margin against a loaded CI runner, same reasoning as above
            assert marker + "=" not in handle.terminal_output()
            handle.resume()
            wait_for(lambda: marker + "=" in handle.terminal_output())
            assert handle.root_pid == root and handle.returncode is None
            assert not (tmp_path / "forbidden").exists()
        assert sum(bool(e.argv and e.argv[0] == "/usr/bin/touch") for e in decisions) == 3
    finally:
        handle.close()


@pytest.mark.timeout(45)
def test_handoff_seizes_every_thread_before_resuming(tmp_path):
    import sys
    from agency.harness.ptrace._checkpoint_handoff import task_status

    cfg = agconfig()
    cfg.sandbox.checkpoint_backend = "cow_zfs"
    cfg.sandbox.checkpoint_fast_resume = True
    program = """
import threading, time, sys
for _ in range(4):
    threading.Thread(target=lambda: time.sleep(60), daemon=True).start()
print('THREADS_READY', flush=True)
for line in sys.stdin:
    print('REPLIED_' + line.strip(), flush=True)
"""
    handle = agProxyPtrace(cfg, allow_initial_exec=True).launch(
        [sys.executable, "-c", program],
        dict(os.environ),
        cwd=str(tmp_path),
        pty_size=(120, 36),
        policy=SimpleNamespace(check=lambda *args: True),
        ag=None,
    )
    try:
        wait_for(lambda: "THREADS_READY" in handle.terminal_output())
        assert len(handle.pids()) >= 5
        for cycle in range(2):
            handle.checkpoint_detach()
            pids = handle.pids()
            assert all(int(task_status(pid)["TracerPid"]) == 0 for pid in pids)
            assert all(task_status(pid)["State"].strip().startswith("T") for pid in pids)
            handle.write_terminal(f"{cycle}\r".encode())
            handle.checkpoint_reattach()
            assert all(int(task_status(pid)["TracerPid"]) != 0 for pid in pids)
            time.sleep(0.1)
            assert f"REPLIED_{cycle}" not in handle.terminal_output()
            handle.resume()
            wait_for(lambda: f"REPLIED_{cycle}" in handle.terminal_output())
    finally:
        handle.close()
