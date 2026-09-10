"""Real Linux PTY/tracer mechanics, without a model or Claude installation."""

import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from agency.harness.ptrace.supervisor import agProxyPtrace, ptrace_available

pytestmark = pytest.mark.skipif(not ptrace_available(), reason="Linux ptrace required")


@pytest.mark.timeout(20)
def test_traced_pty_is_controlling_terminal_and_drains_bounded_output():
    class Policy:
        def check(self, *_):
            return True

    script = """import os,sys,struct,fcntl,termios,json
print(json.dumps({"tty": [os.isatty(i) for i in range(3)], "session": os.getsid(0)==os.getpid(), "size": list(struct.unpack("HHHH", fcntl.ioctl(0, termios.TIOCGWINSZ, b"\\0"*8))[:2])}),flush=True)
assert input()=="first"
sys.stdout.write("x" * (2*1024*1024)); print("DRAINED",flush=True)
assert input()=="second"
print("FINISHED",flush=True)
"""
    handle = agProxyPtrace(allow_initial_exec=True).launch(
        [sys.executable, "-c", script], {}, policy=Policy(), pty_size=(100, 30)
    )
    try:
        deadline = time.monotonic() + 5
        while '"tty"' not in handle.terminal_output() and time.monotonic() < deadline:
            time.sleep(0.01)
        initial = json.loads(handle.terminal_output().splitlines()[0])
        assert initial == {"tty": [True, True, True], "session": True, "size": [30, 100]}
        handle.write_terminal(b"first\n")
        deadline = time.monotonic() + 10
        while "DRAINED" not in handle.terminal_output() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "DRAINED" in handle.terminal_output()
        assert len(handle.terminal_output().encode()) <= 1024 * 1024
        handle.resize_terminal(120, 36)
        handle.write_terminal(b"second\n")
        stdout, stderr, rc = handle.wait(timeout=5)
        assert rc == 0 and not stderr and "FINISHED" in stdout
    finally:
        handle.close()


@pytest.mark.timeout(20)
def test_reap_finds_auto_attached_thread_when_clone_notification_is_lost(monkeypatch):
    from agency.harness.ptrace import _ctypes_defs as pt
    from agency.harness.ptrace._tracer_loop import TracerLoop

    lost_clone = threading.Event()
    original_dispatch = TracerLoop._dispatch

    def dispatch(loop, pid, status):
        # Reproduce SIGKILL overtaking the parent's clone notification: the
        # kernel has attached the child, but it never entered _known_pids.
        if status >> 16 == pt.PTRACE_EVENT_CLONE and not lost_clone.is_set():
            lost_clone.set()
            pt.ptrace(pt.PTRACE_CONT, pid, 0, 0)
            return
        return original_dispatch(loop, pid, status)

    monkeypatch.setattr(TracerLoop, "_dispatch", dispatch)
    handle = agProxyPtrace(allow_initial_exec=True).launch(
        [
            sys.executable,
            "-c",
            "import threading,time; threading.Thread(target=lambda:time.sleep(30)).start(); time.sleep(30)",
        ],
        {},
        pty_size=(80, 24),
        policy=SimpleNamespace(check=lambda *args: True),
    )
    try:
        assert lost_clone.wait(5)
        handle.close()
        assert handle.returncode is not None
        assert not handle.pids()
        assert not handle._loop._known_pids
    finally:
        handle.kill()


@pytest.mark.timeout(20)
def test_concurrent_tracers_do_not_consume_each_others_exit_events():
    handles = [
        agProxyPtrace(allow_initial_exec=True).launch(
            [sys.executable, "-c", f"import time; time.sleep(0.1); raise SystemExit({code})"],
            {},
            pty_size=(80, 24),
            policy=SimpleNamespace(check=lambda *args: True),
        )
        for code in [3, 7, 11]
    ]
    try:
        assert [handle.wait(timeout=5)[2] for handle in handles] == [3, 7, 11]
    finally:
        for handle in handles:
            handle.close()
