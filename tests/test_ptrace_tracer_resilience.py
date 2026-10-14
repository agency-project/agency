import errno
import os
import time

import pytest

from agency.harness.ptrace import _ctypes_defs as pt
from agency.harness.ptrace._tracer_loop import StopDecision, TracerLoop

pytestmark = pytest.mark.skipif(not os.path.exists("/proc/self/status"), reason="Linux ptrace only")


def _loop():
    return TracerLoop(["execve"], lambda stop: StopDecision(kind="allow"))


def test_vanished_tracee_during_dispatch_does_not_kill_the_tracer(monkeypatch):
    loop = _loop()
    real = loop._dispatch
    raised = []

    def flaky(pid, status):
        if not raised:
            raised.append(pid)
            real(pid, status)
            raise pt.PtraceError(errno.ESRCH, "ptrace(request=12, pid=%d) failed: No such process" % pid)
        return real(pid, status)

    monkeypatch.setattr(loop, "_dispatch", flaky)
    loop.start(["/bin/sh", "-c", "/bin/true; /bin/true; exit 3"], dict(os.environ), "/")
    assert loop.join(timeout=30) == 3
    assert raised and loop.tracer_error is None


def test_tracer_death_ends_the_run_instead_of_hanging(monkeypatch):
    loop = _loop()
    real = loop._dispatch
    calls = []

    def broken(pid, status):
        calls.append(pid)
        if len(calls) == 3:
            raise RuntimeError("tracer bug")
        return real(pid, status)

    monkeypatch.setattr(loop, "_dispatch", broken)
    started = time.time()
    loop.start(["/bin/sh", "-c", "/bin/true; /bin/true; sleep 30"], dict(os.environ), "/")
    assert loop.join(timeout=20) == -9
    assert time.time() - started < 20
    assert isinstance(loop.tracer_error, RuntimeError)
