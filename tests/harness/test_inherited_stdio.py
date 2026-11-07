"""The normal-terminal launch contract, without a second PTY or an I/O relay."""

from __future__ import annotations

import errno
import fcntl
import os
import select
import struct
import subprocess
import sys
import termios
import time

import pytest

from agency.harness.ptrace import supervisor
from agency.harness.terminal import InheritedTerminal


@pytest.mark.parametrize("other", [{"stdin_data": b"input"}, {"pty_size": (80, 24)}])
def test_inherited_stdio_rejects_conflicting_input_modes(other):
    with pytest.raises(ValueError, match="inherited stdio"):
        supervisor.agProxyPtrace().launch(
            [sys.executable], {}, inherit_stdio=True, policy=None, **other
        )


def test_inherited_launch_registers_scope_callback_before_start(monkeypatch):
    events = []

    class Loop:
        def __init__(self, **kwargs):
            self.callbacks = []

        def on_spawn(self, callback):
            self.callbacks.append(callback)

        def start(self, argv, env, cwd, **kwargs):
            assert kwargs == {"stdin_data": None, "inherit_stdio": True}
            for callback in self.callbacks:
                callback(123)
            events.append("exec")

    monkeypatch.setattr(supervisor, "TracerLoop", Loop)
    monkeypatch.setattr(
        supervisor._ProcessLifecycleProfiler, "for_active_session", lambda *a, **kw: None
    )
    supervisor.agProxyPtrace().launch(
        [sys.executable],
        {},
        inherit_stdio=True,
        on_spawn=lambda pid: events.append(("scope", pid)),
        policy=None,
    )
    assert events == [("scope", 123), "exec"]


def test_background_caller_does_not_take_terminal_ownership(monkeypatch):
    closed = []
    monkeypatch.setattr(os, "isatty", lambda fd: True)
    monkeypatch.setattr(os, "dup", lambda fd: 99)
    monkeypatch.setattr(os, "getpgrp", lambda: 11)
    monkeypatch.setattr(os, "tcgetpgrp", lambda fd: 22)
    monkeypatch.setattr(termios, "tcgetattr", lambda fd: [])
    monkeypatch.setattr(os, "close", closed.append)
    terminal = InheritedTerminal()
    assert not terminal.claim(33)
    terminal.restore()
    assert closed == [99]


def _read_until(fd, pending, marker, timeout=15):
    deadline = time.monotonic() + timeout
    while marker not in pending:
        assert time.monotonic() < deadline, pending.decode(errors="replace")
        if select.select([fd], [], [], 0.1)[0]:
            try:
                chunk = os.read(fd, 65536)
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise
                chunk = b""
            assert chunk, pending.decode(errors="replace")
            pending += chunk
    return pending


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux controlling-terminal test")
@pytest.mark.parametrize(
    "traced",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not supervisor.ptrace_available(), reason="ptrace not usable on this host"
            ),
        ),
    ],
)
def test_inherited_terminal_preserves_native_input_signals_resize_and_restores_terminal(traced):
    # This PTY is the test's user's terminal. The launch helper never creates
    # another one, and the target inherits this exact same terminal device.
    child = """
import json,os,signal,termios,tty
assert os.tcgetpgrp(0)==os.getpgrp()==os.getpid()
assert os.ttyname(0)==os.environ['EXPECTED_TTY']
tty.setcbreak(0)
signal.signal(signal.SIGINT,lambda *_: print('NATIVE_SIGINT',flush=True))
signal.signal(signal.SIGWINCH,lambda *_: print('NATIVE_SIZE',os.get_terminal_size(0).columns,flush=True))
print('CHILD_READY',flush=True)
assert os.read(0,1)==b'x'
print('NATIVE_INPUT',flush=True)
assert os.read(0,1)==b'q'
os._exit(7)  # leave cbreak mode behind: restore must repair forced exits too
"""
    launcher = """
import os,signal,subprocess,sys,termios
from agency.harness.terminal import InheritedTerminal
original=termios.tcgetattr(0)
env=dict(os.environ,EXPECTED_TTY=os.ttyname(0))
if sys.argv[2]=='True':
 from agency.harness.ptrace.supervisor import agProxyPtrace
 class Policy:
  def check(self,*_):return True
 handle=agProxyPtrace().launch([sys.executable,'-c',sys.argv[1]],env,inherit_stdio=True,policy=Policy())
 try: assert handle.wait(timeout=15)==('','',7)
 finally:handle.close()
else:
 terminal=InheritedTerminal()
 child=subprocess.Popen([sys.executable,'-c',sys.argv[1]],env=env,process_group=0)
 try:
  assert terminal.claim(child.pid)
  os.killpg(child.pid,signal.SIGCONT)
  assert child.wait(timeout=15)==7
 finally:
  if child.poll() is None: child.kill();child.wait()
  terminal.restore()
assert os.tcgetpgrp(0)==os.getpgrp()
assert termios.tcgetattr(0)==original
print('PARENT_RESTORED',flush=True)
"""
    master, slave = os.openpty()
    process = subprocess.Popen(
        ["setsid", "-c", sys.executable, "-c", launcher, child, str(traced)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
    )
    os.close(slave)
    try:
        pending = _read_until(master, b"", b"CHILD_READY")
        os.write(master, b"x")
        pending = _read_until(master, pending, b"NATIVE_INPUT")
        os.write(master, b"\x03")
        pending = _read_until(master, pending, b"NATIVE_SIGINT")
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 111, 0, 0))
        pending = _read_until(master, pending, b"NATIVE_SIZE 111")
        os.write(master, b"q")
        _read_until(master, pending, b"PARENT_RESTORED")
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)


@pytest.mark.skipif(not supervisor.ptrace_available(), reason="ptrace not usable on this host")
def test_traced_inherited_stdio_preserves_pipes_and_scope_before_exec(tmp_path):
    target = tmp_path / "executed"
    child = """
import os,pathlib,sys
pathlib.Path(sys.argv[1]).touch()
assert os.read(0,100)==b'native input\\n'
print('NATIVE_STDOUT',flush=True)
print('NATIVE_STDERR',file=sys.stderr,flush=True)
"""
    launcher = """
import os,sys
from agency.harness.ptrace.supervisor import agProxyPtrace
class Policy:
 def check(self,*_):return True
def spawned(pid):
 assert not os.path.exists(sys.argv[2])
 print('BEFORE_EXEC',flush=True)
h=agProxyPtrace().launch([sys.executable,'-c',sys.argv[1],sys.argv[2]],dict(os.environ),inherit_stdio=True,on_spawn=spawned,policy=Policy())
assert h._loop.pty_master is None
assert h._loop.stdout_r is None and h._loop.stderr_r is None
assert h.wait(timeout=15)==('','',0)
print('TRACE_FINISHED',flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", launcher, child, str(target)],
        input=b"native input\n",
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout.splitlines() == [b"BEFORE_EXEC", b"NATIVE_STDOUT", b"TRACE_FINISHED"]
    assert result.stderr.splitlines() == [b"NATIVE_STDERR"]


@pytest.mark.skipif(not supervisor.ptrace_available(), reason="ptrace not usable on this host")
@pytest.mark.parametrize("failure", ["scope", "foreground"])
def test_failed_inherited_launch_reaps_child_and_restores_foreground(failure):
    launcher = """
import os,sys,termios
from agency.harness.ptrace.supervisor import agProxyPtrace
from agency.harness.terminal import InheritedTerminal
original=termios.tcgetattr(0)
spawned=[]
def on_spawn(pid):
 spawned.append(pid)
 if sys.argv[1]=='scope':raise RuntimeError('scope setup failed')
original_claim=InheritedTerminal.claim
def claim_then_fail(self,pid):
 original_claim(self,pid)
 raise RuntimeError('foreground handoff failed')
if sys.argv[1]=='foreground':InheritedTerminal.claim=claim_then_fail
try:
 agProxyPtrace().launch([sys.executable,'-c','raise AssertionError("must not exec")'],dict(os.environ),inherit_stdio=True,on_spawn=on_spawn,policy=None)
except RuntimeError as exc:
 assert str(exc).endswith('failed')
else:raise AssertionError('failed launch unexpectedly succeeded')
assert len(spawned)==1
try:os.kill(spawned[0],0)
except ProcessLookupError:pass
else:raise AssertionError('failed launch left its child alive')
assert os.tcgetpgrp(0)==os.getpgrp()
assert termios.tcgetattr(0)==original
print('FAILED_LAUNCH_RESTORED',flush=True)
"""
    master, slave = os.openpty()
    process = subprocess.Popen(
        ["setsid", "-c", sys.executable, "-c", launcher, failure],
        stdin=slave,
        stdout=slave,
        stderr=slave,
    )
    os.close(slave)
    try:
        _read_until(master, b"", b"FAILED_LAUNCH_RESTORED")
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
