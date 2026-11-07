"""A native CLI on the caller's terminal, using Agency's optional supervisor."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..harness.terminal import InheritedTerminal

# Synchronize foreground/scope setup before the real CLI can read stdin or
# spawn children. Python's subprocess implementation forks safely without a
# Python preexec_fn; the tiny child then execs the CLI in the same PID.
_EXEC_CHILD = """import os, signal, sys
os.kill(os.getpid(), signal.SIGSTOP)
os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
"""


class NativeProcess:
    """Own the CLI lifetime while leaving its terminal bytes untouched.

    ``on_spawn`` runs once, for the stopped root before its first exec. This
    lets the session move the root into a cgroup before descendants exist.
    Ptrace owns traced descendants; otherwise process-group cleanup is
    supplemented by the session's cgroup controls.
    """

    def __init__(self, argv, env, cwd, *, config, policy, trace=True, on_spawn=None, agent=None):
        argv = list(argv)
        if not argv:
            raise ValueError("native CLI command is empty")
        if "/" not in argv[0]:
            executable = shutil.which(argv[0], path=env.get("PATH", os.defpath))
            if executable is None:
                raise FileNotFoundError(f"native CLI executable not found: {argv[0]}")
            argv[0] = executable
        elif not os.path.isabs(argv[0]):
            argv[0] = str(Path(cwd) / argv[0])
        self.exited = threading.Event()
        self.returncode = None
        self._error = None
        self._handle = self._process = None
        self._terminal = None
        self._root_exited = threading.Event()
        self._job_stopped = threading.Event()
        self._job_lock = threading.Lock()
        self._pause_requested = False
        self._pending_pause_stops = 0
        self._stopping = False
        self.pid = None

        def root_spawned(pid):
            if self.pid is None:
                self.pid = pid
                if on_spawn is not None:
                    on_spawn(pid)

        if trace:
            from ..harness.ptrace.supervisor import agProxyPtrace

            self._handle = agProxyPtrace(config, allow_initial_exec=True).launch(
                argv,
                env,
                cwd=str(cwd),
                policy=policy,
                ag=agent,
                inherit_stdio=True,
                on_spawn=root_spawned,
                on_job_stop=lambda _sig: self._job_stopped.set(),
            )
            self._handle.on_exit(self._root_exit, include_exit_code=True)
        else:
            self._terminal = InheritedTerminal()
            try:
                self._process = subprocess.Popen(
                    [sys.executable, "-I", "-S", "-c", _EXEC_CHILD, *argv],
                    cwd=cwd,
                    env=env,
                    process_group=0,
                )
                _, status = os.waitpid(self._process.pid, os.WUNTRACED)
                if not os.WIFSTOPPED(status):
                    self._process.returncode = os.waitstatus_to_exitcode(status)
                    raise RuntimeError("native CLI bootstrap exited before launch setup")
                root_spawned(self._process.pid)
                self._terminal.claim(self.pid)
                os.killpg(self.pid, signal.SIGCONT)
            except BaseException:
                if self._process is not None and self._process.poll() is None:
                    os.killpg(self._process.pid, signal.SIGKILL)
                    self._process.wait()
                self._terminal.restore()
                raise
        self._watcher = threading.Thread(target=self._watch, name="native-cli-exit", daemon=True)
        self._watcher.start()

    def _root_exit(self, pid, code):
        if pid == self.pid:
            self.returncode = code
            self._root_exited.set()

    def _watch(self):
        try:
            if self._handle is not None:
                self._root_exited.wait()
                # A CLI can leave background processes with inherited terminal
                # descriptors. Retire them, without replacing the root's code.
                self._handle.close()
            else:
                while True:
                    _, status = os.waitpid(self.pid, os.WUNTRACED | os.WCONTINUED)
                    if os.WIFSTOPPED(status):
                        with self._job_lock:
                            if os.WSTOPSIG(status) == signal.SIGSTOP and self._pending_pause_stops:
                                self._pending_pause_stops -= 1
                            elif not self._pause_requested and not self._stopping:
                                self._job_stopped.set()
                    elif os.WIFCONTINUED(status):
                        with self._job_lock:
                            self._pending_pause_stops = 0
                    else:
                        self.returncode = os.waitstatus_to_exitcode(status)
                        self._process.returncode = self.returncode
                        break
                self.signal_group(signal.SIGKILL)
        except BaseException as exc:
            self._error = exc
        finally:
            if self._terminal is not None:
                self._terminal.restore()
            self.exited.set()

    def signal_group(self, signum):
        if self.exited.is_set():
            return
        with self._job_lock:
            if signum in (signal.SIGTERM, signal.SIGKILL):
                self._stopping = True
                self._job_stopped.clear()
            elif self._handle is None and signum == signal.SIGSTOP:
                self._pause_requested = True
                self._pending_pause_stops += 1
            elif self._handle is None and signum == signal.SIGCONT:
                self._pause_requested = False
                self._job_stopped.clear()
        if self._handle is not None:
            control = {
                signal.SIGSTOP: self._handle.pause,
                signal.SIGCONT: self._handle.resume,
                signal.SIGTERM: self._handle.terminate,
                signal.SIGKILL: self._handle.kill,
            }.get(signum)
            if control is not None:
                control()
                return
            pids = self._handle.pids()
        else:
            pids = [-self.pid]
        for pid in pids:
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass

    def wait(self, timeout=None):
        if threading.current_thread() is threading.main_thread():
            deadline = time.monotonic() + timeout if timeout is not None else None
            while not self.exited.is_set():
                self.poll_job_control()
                remaining = deadline - time.monotonic() if deadline is not None else 0.05
                if remaining <= 0:
                    return None
                self.exited.wait(min(0.05, remaining))
        elif not self.exited.wait(timeout):
            return None
        if self._error is not None:
            raise self._error
        return self.returncode

    def poll_job_control(self):
        """Dispatch queued kernel job stops on the terminal-owning main thread."""
        with self._job_lock:
            if self._stopping or self.exited.is_set() or not self._job_stopped.is_set():
                return
            self._job_stopped.clear()
        if self._stopping or self.exited.is_set() or self._root_exited.is_set():
            return
        if self._handle is not None:
            self._handle.suspend_terminal()
        else:
            self._terminal.suspend(self.pid)

    def close(self):
        self.signal_group(signal.SIGKILL)
        if self.wait(10) is None:
            raise TimeoutError("native CLI process failed to exit")
