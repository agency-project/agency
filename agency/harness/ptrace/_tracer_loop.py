"""The waitpid()/ptrace-stop dispatch loop -- the core supervisor mechanism.

Runs entirely on ONE dedicated thread (the same thread that calls os.fork()):
ptrace's tracer identity is per-THREAD, not per-process -- only the thread
that attaches (via TRACEME/ATTACH/SEIZE) may subsequently ptrace()/waitpid()
that tracee, so the fork() and the whole dispatch loop must stay on one
thread for the lifetime of a launch.

Uses `waitpid(pid, WNOHANG)` polled per known pid, NOT `waitpid(-1, ...)`.
`waitpid(-1, ...)` reaps exit status for ANY child of the calling process,
not just ones this loop is tracing -- in a process that also spawns
subprocesses elsewhere (`docker`/`podman` via subprocess.run, harnesses, ...),
that would race with and could steal the exit status
those other call sites are waiting on. Polling WNOHANG per known pid avoids
this at the cost of a small, bounded poll latency -- verified during
development against a concurrent unrelated `subprocess.Popen` child (it gets
reaped correctly through subprocess's own machinery, untouched by this loop).

Decoupled from `agpolicy`/`agsyscallevent` on purpose: this
module takes a plain `syscall_hook` callback trading in the lightweight
`SeccompStop`/`StopDecision` shapes below, so `supervisor.py` is the only
place that adapts to the public `agpolicy` interface -- avoids a circular import and keeps this
package's only job "run the ptrace mechanics correctly."
"""

from __future__ import annotations

import ctypes
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Callable

from . import _ctypes_defs as pt
from . import _seccomp_filter


@dataclass
class SeccompStop:
    pid: int
    syscall: str
    syscall_nr: int
    argv: "list[str] | None"
    envp: "dict[str, str] | None"
    path: "str | None"
    timestamp: float


@dataclass
class StopDecision:
    kind: str  # "allow" | "deny" | "rewrite"
    new_args: "list[str] | None" = None
    # Correlates this admission with its later completion report (see
    # syscall_exit_hook below) -- opaque to this module, just threaded
    # through from whatever syscall_hook's own policy.check() returned.
    call_id: "str | None" = None


_EPERM = 1


def _is_thread_group_leader(pid: int) -> bool:
    """Return whether *pid* names a process rather than a non-leader thread.

    ``PTRACE_EVENT_CLONE`` covers both ``clone(CLONE_THREAD)`` and
    clone-created processes.  At the clone child's mandatory initial ptrace
    stop, ``/proc/<tid>/status`` is present and its ``Tgid`` distinguishes the
    two exactly.  If procfs is unexpectedly unavailable, retain the tracee as
    a process rather than silently losing a genuine subprocess.
    """
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("Tgid:"):
                    return int(line.split(":", 1)[1].strip()) == pid
    except (OSError, ValueError):
        return True
    return True


def _kernel_executable_path(pid: int) -> "str | None":
    """Read the executable image the kernel installed for stopped *pid*.

    This is called only at ``PTRACE_EVENT_EXEC``. Unlike ``argv[0]``, the
    procfs link identifies the actual image and cannot be changed by
    ``exec -a``.
    """
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return None


def _resolve_syscall_args(
    pid: int, regs: "pt.UserRegsStruct", syscall_nr: int
) -> "tuple[list[str] | None, dict[str, str] | None, str | None]":
    """Resolve the fields agsyscallevent cares about (argv/envp/path) for
    whichever syscall was intercepted. Returns all-None for any syscall
    number not in this table -- the event still gets delivered to the
    policy with just `syscall`/`pid`/`tid`/`timestamp` populated, it's just
    that this module doesn't yet know how to decode that syscall's specific
    argument registers."""
    if syscall_nr == pt.SYSCALL_NUMBERS["execve"]:
        # int execve(const char *pathname, char *const argv[], char *const envp[])
        path_ptr, argv_ptr, envp_ptr = regs.rdi, regs.rsi, regs.rdx
        path = pt.read_cstring(pid, path_ptr)
        return pt.resolve_argv(pid, argv_ptr), pt.resolve_envp(pid, envp_ptr), path
    if syscall_nr == pt.SYSCALL_NUMBERS["execveat"]:
        # int execveat(int dirfd, const char *pathname, char *const argv[],
        #              char *const envp[], int flags) -- args shift by one
        # register relative to execve() because of the leading dirfd.
        path_ptr, argv_ptr, envp_ptr = regs.rsi, regs.rdx, regs.r10
        path = pt.read_cstring(pid, path_ptr)
        return pt.resolve_argv(pid, argv_ptr), pt.resolve_envp(pid, envp_ptr), path
    if syscall_nr == pt.SYSCALL_NUMBERS["open"]:
        # int open(const char *pathname, int flags, mode_t mode)
        return None, None, pt.read_cstring(pid, regs.rdi)
    if syscall_nr == pt.SYSCALL_NUMBERS["openat"]:
        # int openat(int dirfd, const char *pathname, int flags, mode_t mode)
        # -- resolved as the raw pathname only; a relative path's real target
        # depends on dirfd, which this module does not resolve (that would
        # require reading the tracee's /proc/<pid>/fd/<dirfd> symlink) --
        # policies matching on relative paths should be aware of this.
        return None, None, pt.read_cstring(pid, regs.rsi)
    return None, None, None


class TracerLoop:
    """Owns one traced process tree, from fork() through exit. Construct a
    fresh instance per launch -- not reusable."""

    def __init__(
        self,
        syscalls: "list[str] | tuple[str, ...]",
        syscall_hook: "Callable[[SeccompStop], StopDecision]",
        poll_interval_s: float = 0.002,
        syscall_exit_hook: "Callable[[SeccompStop, str | None, int], None] | None" = None,
    ) -> None:
        self._syscalls = tuple(syscalls)
        self._syscall_hook = syscall_hook
        self._poll_interval_s = poll_interval_s
        # Called (stop, call_id, return_value) after an admitted syscall's
        # matching exit-stop -- None (the default) skips exit-tracing
        # entirely, resuming every admitted syscall with PTRACE_CONT exactly
        # as before this existed.
        self._syscall_exit_hook = syscall_exit_hook

        self.root_pid: "int | None" = None
        self.stdout_r: "int | None" = None
        self.stderr_r: "int | None" = None

        self._known_pids: "set[int]" = set()
        # pause()/resume(): _held_pids marks a pid that must not be
        # auto-continued the next time its SIGSTOP delivery-stop is
        # dispatched; _parked_pids marks one that has actually reached that
        # stop and is currently withheld there, awaiting resume()'s deferred
        # PTRACE_CONT. See _dispatch()'s generic-signal fallthrough.
        self._held_pids: "set[int]" = set()
        self._parked_pids: "set[int]" = set()
        # Pids resume() wants continued -- drained and actually PTRACE_CONT'd
        # by _run() on the dedicated tracer thread (see resume()'s docstring).
        self._resume_requests: "list[int]" = []
        self._process_pids: "set[int]" = set()
        self._pending_clone_pids: "set[int]" = set()
        self._pending_exec_paths: "dict[int, str | None]" = {}
        # pid -> (SeccompStop, call_id) for an admitted syscall resumed with
        # PTRACE_SYSCALL instead of PTRACE_CONT, awaiting its matching
        # syscall-exit-stop. Only populated when _syscall_exit_hook is set.
        self._pending_syscall_exit: "dict[int, tuple]" = {}
        self._options_applied: "set[int]" = set()
        self._returncode: "int | None" = None
        self._finished = threading.Event()
        # ``launch()`` must not return while the root process is still only
        # the forked Python image.  In particular, an immediately-ending
        # profiler session needs the kernel-confirmed executable name before
        # it interrupts the live lifecycle span.  This event is set after
        # the first root exec callback (or after the root exits without a
        # successful exec).
        self._root_image_ready = threading.Event()
        self._thread: "threading.Thread | None" = None
        self._lock = threading.Lock()

        self._stdout_buf = bytearray()
        self._stderr_buf = bytearray()
        self._stdout_reader: "threading.Thread | None" = None
        self._stderr_reader: "threading.Thread | None" = None
        self._stdin_writer: "threading.Thread | None" = None

        self._spawn_callbacks: "list[Callable[[int], None]]" = []
        self._exec_callbacks: "list[Callable[[int, str | None], None]]" = []
        self._exit_callbacks: "list[Callable[[int, int], None]]" = []
        self._spawn_log: "list[int]" = []
        self._exec_log: "list[tuple[int, str | None]]" = []
        self._exit_log: "list[tuple[int, int]]" = []

    # -- registration: replay-safe, a callback registered after some events
    #    have already happened still gets to see all of them -------------

    def on_spawn(self, callback: "Callable[[int], None]") -> None:
        with self._lock:
            backlog = list(self._spawn_log)
            self._spawn_callbacks.append(callback)
        for pid in backlog:
            callback(pid)

    def on_exec(self, callback: "Callable[[int, str | None], None]") -> None:
        with self._lock:
            backlog = list(self._exec_log)
            self._exec_callbacks.append(callback)
        for pid, executable_path in backlog:
            callback(pid, executable_path)

    def on_exit(self, callback: "Callable[[int, int], None]") -> None:
        with self._lock:
            backlog = list(self._exit_log)
            self._exit_callbacks.append(callback)
        for pid, code in backlog:
            callback(pid, code)

    def live_pids(self) -> "set[int]":
        with self._lock:
            return set(self._known_pids)

    # -- lifecycle ----------------------------------------------------------

    def start(
        self,
        argv: "list[str]",
        envp: "dict[str, str]",
        cwd: str,
        *,
        stdin_data: "bytes | None" = None,
    ) -> None:
        """Starts the fork + trace loop on ONE dedicated thread and blocks
        until the child exists and its first executable image is confirmed
        (or forking/the initial stop/exec failed). The fork itself MUST happen on the same
        thread that subsequently calls waitpid()/ptrace() on the child --
        ptrace's tracer identity is per-thread (only the thread that
        attaches, here via the child's PTRACE_TRACEME, may later
        ptrace()/waitpid() it) -- so os.fork() cannot happen on the
        caller's thread with the trace loop running on a different one:
        the resulting PTRACE_SETOPTIONS/waitpid calls would target a pid
        this thread was never the tracer of and fail with ESRCH."""
        started = threading.Event()
        start_error: "list[BaseException]" = []

        def run_with_fork() -> None:
            try:
                self._fork_and_exec(argv, envp, cwd, stdin_data)
            except BaseException as exc:  # noqa: BLE001 -- surfaced to start()'s caller below
                start_error.append(exc)
                started.set()
                return
            started.set()
            self._run()

        self._thread = threading.Thread(target=run_with_fork, name="agproxy_ptrace", daemon=True)
        self._thread.start()
        if not started.wait(timeout=30):
            raise RuntimeError("ptrace child did not reach its initial stop within 30s")
        if start_error:
            raise start_error[0]
        if not self._root_image_ready.wait(timeout=30):
            self.kill()
            raise RuntimeError("ptrace child did not exec or exit within 30s")

    def _fork_and_exec(
        self,
        argv: "list[str]",
        envp: "dict[str, str]",
        cwd: str,
        stdin_data: "bytes | None",
    ) -> None:
        """Runs on the dedicated tracer thread, before `_run()`. Forks,
        starts the output-reader threads, and blocks for the child's
        initial post-TRACEME stop + PTRACE_SETOPTIONS -- all on this
        thread, so `_run()`'s subsequent waitpid()/ptrace() calls are
        always issued by the same thread that attached."""
        stdout_r, stdout_w = os.pipe()
        stderr_r, stderr_w = os.pipe()
        stdin_r, stdin_w = os.pipe() if stdin_data is not None else (None, None)
        self.stdout_r, self.stderr_r = stdout_r, stderr_r

        try:
            pid = os.fork()
        except BaseException:
            for fd in (stdout_r, stdout_w, stderr_r, stderr_w, stdin_r, stdin_w):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            raise
        if pid == 0:
            self._child_exec(
                argv,
                envp,
                cwd,
                stdout_r,
                stdout_w,
                stderr_r,
                stderr_w,
                stdin_r,
                stdin_w,
            )
            os._exit(127)  # unreachable: _child_exec always execve()s or _exit()s

        os.close(stdout_w)
        os.close(stderr_w)
        if stdin_r is not None:
            os.close(stdin_r)
        self.root_pid = pid
        # Goes through _remember_spawn (not a bare _known_pids.add) so the
        # root pid reaches on_spawn callbacks too, not just its forked
        # descendants -- on_spawn's replay-log design means a caller that
        # registers a callback later (start()/launch() has already
        # returned by the time _run() and any real forking happens) still
        # sees this via the backlog, same as any other spawn event.
        self._remember_spawn(pid)

        self._stdout_reader = threading.Thread(
            target=self._drain_pipe,
            args=(stdout_r, self._stdout_buf),
            name=f"agproxy_ptrace-{pid}-stdout",
            daemon=True,
        )
        self._stderr_reader = threading.Thread(
            target=self._drain_pipe,
            args=(stderr_r, self._stderr_buf),
            name=f"agproxy_ptrace-{pid}-stderr",
            daemon=True,
        )
        self._stdout_reader.start()
        self._stderr_reader.start()

        try:
            _, status = os.waitpid(pid, 0)
            assert os.WIFSTOPPED(status), f"expected initial stop, got status={status:#x}"
            pt.ptrace(pt.PTRACE_SETOPTIONS, pid, 0, pt.ALL_TRACE_OPTIONS)
            with self._lock:
                self._options_applied.add(pid)
            pt.ptrace(pt.PTRACE_CONT, pid, 0, 0)
        except BaseException:
            if stdin_w is not None:
                os.close(stdin_w)
            raise

        if stdin_w is not None:
            writer = threading.Thread(
                target=self._write_stdin,
                args=(stdin_w, stdin_data),
                name=f"agproxy_ptrace-{pid}-stdin",
                daemon=True,
            )
            try:
                writer.start()
            except BaseException:
                os.close(stdin_w)
                raise
            self._stdin_writer = writer

    @staticmethod
    def _write_stdin(fd: int, data: bytes) -> None:
        """Write all of *data* off the tracer thread, then deliver EOF."""
        try:
            remaining = memoryview(data)
            while remaining:
                try:
                    written = os.write(fd, remaining)
                except InterruptedError:
                    continue
                except BrokenPipeError:
                    break
                if written <= 0:
                    break
                remaining = remaining[written:]
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def _drain_pipe(self, fd: int, buf: bytearray) -> None:
        """Runs on a dedicated reader thread for the lifetime of the launch
        -- reads until the write end closes (the traced process, and every
        process that inherited the fd, has exited), appending under
        `self._lock` so `read_output()` can snapshot safely from any
        thread. Continuously draining (rather than reading only in
        `wait()`) avoids the traced process blocking on a full pipe buffer
        if it writes more output than one read() call would drain."""
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            with self._lock:
                buf += chunk
        try:
            os.close(fd)
        except OSError:
            pass

    def read_output(self) -> "tuple[str, str]":
        with self._lock:
            stdout = bytes(self._stdout_buf)
            stderr = bytes(self._stderr_buf)
        return stdout.decode(errors="replace"), stderr.decode(errors="replace")

    def _child_exec(
        self,
        argv,
        envp,
        cwd,
        stdout_r,
        stdout_w,
        stderr_r,
        stderr_w,
        stdin_r,
        stdin_w,
    ) -> None:
        """Runs ONLY in the forked child, right up until execve replaces it
        (or it _exit()s on failure). No agency machinery is safe to touch
        here -- this is the traced target's process image until exec."""
        os.close(stdout_r)
        os.close(stderr_r)
        os.dup2(stdout_w, 1)
        os.dup2(stderr_w, 2)
        os.close(stdout_w)
        os.close(stderr_w)
        if stdin_r is None:
            # Never inherit the daemon's stdin. Without this, an open,
            # unfed non-tty pipe can make CLIs wait for input that will
            # never arrive.
            devnull_fd = os.open(os.devnull, os.O_RDONLY)
            os.dup2(devnull_fd, 0)
            if devnull_fd != 0:
                os.close(devnull_fd)
            else:
                os.set_inheritable(0, True)
        else:
            assert stdin_w is not None
            os.close(stdin_w)
            os.dup2(stdin_r, 0)
            if stdin_r != 0:
                os.close(stdin_r)
            else:
                os.set_inheritable(0, True)
        if cwd:
            os.chdir(cwd)
        pt.ptrace(pt.PTRACE_TRACEME, 0, 0, 0)
        # Synchronize with the parent: it must call PTRACE_SETOPTIONS(...,
        # PTRACE_O_TRACESECCOMP) before the filter below is installed and we
        # exec, or the filtered syscall fails with ENOSYS instead of
        # trapping (see _seccomp_filter.py's docstring).
        os.kill(os.getpid(), signal.SIGSTOP)
        _seccomp_filter.install_trace_filter(self._syscalls)
        try:
            os.execve(argv[0], argv, dict(envp))
        except BaseException as exc:
            # Anything that reaches here means the traced target never ran
            # at all -- write the reason directly to raw fd 2 (NOT via
            # sys.stderr / print(): a test runner like pytest that captures
            # output monkeypatches sys.stderr to a Python-level buffer
            # object *before* fork(), and the forked child inherits that
            # same monkeypatched object -- writing through it never reaches
            # the real fd 2 this process's stderr was dup2'd onto, so the
            # message would silently vanish under pytest's capture instead
            # of ending up in read_output() as intended).
            os.write(2, f"agproxy_ptrace: execve({argv[0]!r}) failed: {exc!r}\n".encode())
            os._exit(126)

    def _run(self) -> None:
        """Runs on the same dedicated tracer thread as `_fork_and_exec()`
        (which has already handled the root process's initial stop and
        PTRACE_SETOPTIONS by the time this is called -- see `start()`)."""
        assert self.root_pid is not None
        while True:
            with self._lock:
                pending = list(self._known_pids)
                to_resume = list(self._resume_requests)
                self._resume_requests.clear()
            # resume()'s PTRACE_CONT must be issued from this dedicated
            # tracer thread -- ptrace's tracer identity is per-thread, so a
            # call from resume()'s own (arbitrary) caller thread would fail.
            # resume() only queues the pids here; this loop is what actually
            # restarts them, on its very next iteration.
            for pid in to_resume:
                try:
                    pt.ptrace(pt.PTRACE_CONT, pid, 0, 0)
                except ProcessLookupError:
                    pass
            if not pending:
                break
            made_progress = False
            for wpid in pending:
                try:
                    got_pid, status = os.waitpid(wpid, os.WNOHANG)
                except ChildProcessError:
                    self._forget(wpid, -1)
                    continue
                if got_pid == 0:
                    continue
                made_progress = True
                self._dispatch(wpid, status)
            if not made_progress:
                time.sleep(self._poll_interval_s)

        self._finished.set()

    def _dispatch(self, pid: int, status: int) -> None:
        if os.WIFEXITED(status):
            self._forget(pid, os.WEXITSTATUS(status))
            return
        if os.WIFSIGNALED(status):
            self._forget(pid, -os.WTERMSIG(status))
            return
        assert os.WIFSTOPPED(status), (pid, status)
        sig = os.WSTOPSIG(status)
        event = (status >> 16) & 0xFF

        with self._lock:
            needs_options = pid not in self._options_applied
        if needs_options:
            # First-ever stop for this pid -- either the root process's
            # post-TRACEME/SIGSTOP stop (handled separately in _run() for
            # the root, so in practice this branch only fires for a newly
            # auto-attached fork/vfork/clone child) or, defensively, any
            # other pid we somehow see before applying options to it.
            self._classify_pending_clone(pid)
            pt.ptrace(pt.PTRACE_SETOPTIONS, pid, 0, pt.ALL_TRACE_OPTIONS)
            with self._lock:
                self._options_applied.add(pid)
            pt.ptrace(pt.PTRACE_CONT, pid, 0, 0)
            return

        if sig == signal.SIGTRAP and event == pt.PTRACE_EVENT_SECCOMP:
            self._handle_seccomp_stop(pid)
            return
        if sig == signal.SIGTRAP and event == 0:
            with self._lock:
                pending_exit = self._pending_syscall_exit.pop(pid, None)
            if pending_exit is not None:
                self._handle_syscall_exit_stop(pid, pending_exit)
                return
        if sig == signal.SIGTRAP and event in (
            pt.PTRACE_EVENT_FORK,
            pt.PTRACE_EVENT_VFORK,
            pt.PTRACE_EVENT_CLONE,
        ):
            new_pid = pt.get_eventmsg(pid)
            self._remember_spawn(
                new_pid,
                is_process=None if event == pt.PTRACE_EVENT_CLONE else True,
            )
            pt.ptrace(pt.PTRACE_CONT, pid, 0, 0)
            return
        if sig == signal.SIGTRAP and event == pt.PTRACE_EVENT_EXEC:
            self._commit_exec(pid)
            pt.ptrace(pt.PTRACE_CONT, pid, 0, 0)
            return
        if sig == signal.SIGTRAP and event == pt.PTRACE_EVENT_EXIT:
            # Process disappearance itself is handled via WIFEXITED or
            # WIFSIGNALED above; the pre-exit notification just resumes.
            pt.ptrace(pt.PTRACE_CONT, pid, 0, 0)
            return
        # A real signal being delivered to the tracee -- forward it
        # untouched, except don't forward a bare SIGTRAP (which shouldn't
        # occur here with the options above, but must never be forwarded
        # as a real signal if it somehow does).
        forward = 0 if sig == signal.SIGTRAP else sig
        if sig == signal.SIGSTOP:
            # Re-injecting SIGSTOP via PTRACE_CONT does not actually suspend
            # a tracee -- restarting it always resumes it regardless of
            # which signal is passed in the restart. The only way to truly
            # hold it stopped is to withhold the restart call entirely once
            # pause() has asked for this pid to stop -- see pause()/resume().
            with self._lock:
                if pid in self._held_pids:
                    self._parked_pids.add(pid)
                    return
        pt.ptrace(pt.PTRACE_CONT, pid, 0, forward)

    def pause(self) -> None:
        """Stop every currently-known pid in the traced tree. Unlike
        kill(), this must wait for each pid's own SIGSTOP delivery-stop to
        reach _dispatch() (on the dedicated tracer thread) before it is
        actually withheld -- see _dispatch()'s generic-signal fallthrough."""
        with self._lock:
            pids = list(self._known_pids)
            self._held_pids |= set(pids)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass

    def resume(self) -> None:
        """Queue the deferred PTRACE_CONT for every pid pause() withheld --
        actually issued by _run() on the dedicated tracer thread, since
        ptrace() calls are only valid from the thread that attached (this
        method itself may be called from any thread, e.g. a control-route
        handler)."""
        with self._lock:
            parked = list(self._parked_pids)
            self._parked_pids.clear()
            self._held_pids.clear()
            self._resume_requests.extend(parked)

    def _handle_seccomp_stop(self, pid: int) -> None:
        regs = pt.get_regs(pid)
        nr = regs.orig_rax
        name = pt.SYSCALL_NAMES_BY_NUMBER.get(nr, f"nr:{nr}")
        argv, envp, path = _resolve_syscall_args(pid, regs, nr)
        stop = SeccompStop(
            pid=pid,
            syscall=name,
            syscall_nr=nr,
            argv=argv,
            envp=envp,
            path=path,
            timestamp=time.time(),
        )
        decision = self._syscall_hook(stop)
        is_exec = nr in (pt.SYSCALL_NUMBERS["execve"], pt.SYSCALL_NUMBERS["execveat"])
        if decision.kind == "deny":
            # Skip the syscall (orig_rax=-1) and make it appear to have
            # returned -EPERM, in two separate GETREGS/SETREGS round-trips
            # -- validated this way during development; combining both
            # register writes into a single SETREGS call is unverified and
            # deliberately not attempted here.
            regs.orig_rax = ctypes.c_ulonglong(-1).value
            pt.set_regs(pid, regs)
            regs2 = pt.get_regs(pid)
            regs2.rax = ctypes.c_ulonglong((-_EPERM) & 0xFFFFFFFFFFFFFFFF).value
            pt.set_regs(pid, regs2)
        elif decision.kind == "rewrite" and decision.new_args:
            if nr not in (pt.SYSCALL_NUMBERS["execve"], pt.SYSCALL_NUMBERS["execveat"]):
                # `rewrite` only injects a new path+argv into the exec-family
                # argument registers. Path
                # redirection for openat/open is deliberately NOT done this
                # way (see the design doc: raw pointer rewriting for file
                # paths is fragile and agsandbox's mount mechanism already
                # solves "this path resolves somewhere else" properly) --
                # silently falls through to allow rather than corrupting
                # unrelated registers.
                pass
            else:
                path_addr, argv_addr = pt.inject_argv(
                    pid, regs.rsp, decision.new_args[0], decision.new_args
                )
                if nr == pt.SYSCALL_NUMBERS["execve"]:
                    regs.rdi = path_addr
                    regs.rsi = argv_addr
                else:
                    regs.rsi = path_addr
                    regs.rdx = argv_addr
                pt.set_regs(pid, regs)
        if is_exec:
            # The syscall entry is only a candidate: exec can still fail
            # (ENOENT, EACCES, malformed image, ...). Commit it only if the
            # kernel later reports PTRACE_EVENT_EXEC. A rewrite changes the
            # kernel pathname but never makes argv[0] authoritative.
            with self._lock:
                if decision.kind == "deny":
                    self._pending_exec_paths.pop(pid, None)
                else:
                    self._pending_exec_paths[pid] = (
                        decision.new_args[0]
                        if decision.kind == "rewrite" and decision.new_args
                        else path
                    )
        if decision.kind != "deny" and self._syscall_exit_hook is not None:
            # Request the matching syscall-exit-stop instead of resuming
            # freely, so the return value and duration can be reported. A
            # successful exec never reaches it (PTRACE_EVENT_EXEC fires
            # instead, see _commit_exec/_forget's own cleanup of this same
            # entry); a failed exec still returns normally and is handled
            # like any other syscall's exit.
            with self._lock:
                self._pending_syscall_exit[pid] = (stop, decision.call_id)
            pt.ptrace(pt.PTRACE_SYSCALL, pid, 0, 0)
        else:
            pt.ptrace(pt.PTRACE_CONT, pid, 0, 0)

    def _handle_syscall_exit_stop(self, pid: int, pending: tuple) -> None:
        stop, call_id = pending
        try:
            regs = pt.get_regs(pid)
            return_value = ctypes.c_longlong(regs.rax).value
        except OSError:
            # Tracee may have raced ahead to exit between the exit-stop
            # notification and this GETREGS -- best-effort, skip reporting
            # rather than crash the whole tracer loop over one lost value.
            return_value = None
        finally:
            pt.ptrace(pt.PTRACE_CONT, pid, 0, 0)
        if self._syscall_exit_hook is not None and return_value is not None:
            self._syscall_exit_hook(stop, call_id, return_value)

    def _remember_spawn(self, pid: int, *, is_process: "bool | None" = True) -> None:
        with self._lock:
            self._known_pids.add(pid)
            if is_process is None:
                self._pending_clone_pids.add(pid)
            elif is_process:
                self._process_pids.add(pid)
                self._spawn_log.append(pid)
            callbacks = list(self._spawn_callbacks) if is_process else []
        for cb in callbacks:
            cb(pid)

    def _classify_pending_clone(self, pid: int) -> None:
        with self._lock:
            if pid not in self._pending_clone_pids:
                return
            self._pending_clone_pids.discard(pid)
        if not _is_thread_group_leader(pid):
            return
        with self._lock:
            self._process_pids.add(pid)
            self._spawn_log.append(pid)
            callbacks = list(self._spawn_callbacks)
        for cb in callbacks:
            cb(pid)

    def _commit_exec(self, pid: int) -> None:
        # The staged syscall pathname is the user-facing command identity and
        # PTRACE_EVENT_EXEC confirms that exact attempt succeeded. This keeps
        # script launchers named for the requested script (for example
        # ``claude``) instead of procfs' shebang interpreter (``node``).
        # procfs remains the fallback when exec syscalls were not trapped or
        # execveat used an empty pathname.
        procfs_path = _kernel_executable_path(pid)
        with self._lock:
            staged_path = self._pending_exec_paths.pop(pid, None)
            # A successful exec never produces the syscall-exit-stop this
            # entry was waiting for -- PTRACE_EVENT_EXEC preempts it. Retain
            # the admission so the exec event can complete it below.
            pending_exit = self._pending_syscall_exit.pop(pid, None)
            executable_path = staged_path or procfs_path
            is_process = pid in self._process_pids
            if is_process:
                self._exec_log.append((pid, executable_path))
                callbacks = list(self._exec_callbacks)
            else:
                callbacks = []
        if pending_exit is not None and self._syscall_exit_hook is not None:
            stop, call_id = pending_exit
            # execve has no userspace return on success. The kernel exec event
            # is the authoritative successful completion boundary.
            self._syscall_exit_hook(stop, call_id, 0)
        for callback in callbacks:
            callback(pid, executable_path)
        if is_process and pid == self.root_pid:
            # Callbacks are synchronous while the tracee is stopped. Publish
            # readiness only after lifecycle consumers have applied the
            # kernel-confirmed executable name.
            self._root_image_ready.set()

    def _forget(self, pid: int, exit_code: int) -> None:
        with self._lock:
            self._known_pids.discard(pid)
            self._pending_clone_pids.discard(pid)
            self._pending_exec_paths.pop(pid, None)
            self._pending_syscall_exit.pop(pid, None)
            self._options_applied.discard(pid)
            if pid == self.root_pid:
                self._returncode = exit_code
            was_process = pid in self._process_pids
            self._process_pids.discard(pid)
            if was_process:
                self._exit_log.append((pid, exit_code))
            callbacks = list(self._exit_callbacks) if was_process else []
        for cb in callbacks:
            cb(pid, exit_code)
        if pid == self.root_pid:
            # Failed/denied execs never produce PTRACE_EVENT_EXEC. Unblock
            # launch after their exit callbacks preserve the failed
            # lifecycle under its honest ``<unknown>`` identity.
            self._root_image_ready.set()

    def join(self, timeout: "float | None" = None) -> "int | None":
        """Block until the root process (and everything it spawned) has
        exited, or *timeout* elapses. Returns the root process's exit code,
        or None if the timeout elapsed first. Also waits for the
        stdout/stderr reader threads to observe EOF, so `read_output()`
        is guaranteed complete once this returns non-None -- the pipe
        write ends only close once every process holding them (the root
        and everything it forked) has exited, which _finished already
        waits for, but the reader threads still need a moment to drain
        the last chunk and notice the resulting EOF themselves."""
        if not self._finished.wait(timeout):
            return None
        if self._thread is not None:
            self._thread.join()
        if self._stdout_reader is not None:
            self._stdout_reader.join()
        if self._stderr_reader is not None:
            self._stderr_reader.join()
        if self._stdin_writer is not None:
            self._stdin_writer.join()
        return self._returncode

    def kill(self) -> None:
        with self._lock:
            pids = list(self._known_pids)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
