"""Syscall-level supervisor for harness-driven agents.

`agProxyPtrace.launch()` starts a target process the same way a debugger
does -- fork(), the child calls `PTRACE_TRACEME`, installs a seccomp filter
that traps a small set of syscalls (`SECCOMP_RET_TRACE`, not a full deny-by-
default sandbox), then execve()s the real binary. The parent becomes that
process's tracer and, transitively, tracer of everything it forks/execs
(`PTRACE_O_TRACEFORK`/`_VFORK`/`_CLONE`), observing every trapped syscall
before it runs.

This requires no cooperation from the traced binary -- unlike each coding
harness's own hook system (Claude Code's `PreToolUse`, opencode's
`tool.execute.before`, ...), which only fires for calls the harness's own
tool-dispatch code chooses to report. See docs/Design_harness_integration.md
("Component 3") for the full design rationale.

x86_64 Linux only (see agproxy_ptrace_internal/_ctypes_defs.py's
`_arch_guard()`) -- `seccomp`+`PTRACE_EVENT_SECCOMP` has no macOS/BSD
equivalent.
"""

from __future__ import annotations

import os
import platform
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ..agconfig import GlobalConfigParam, DynamicConfigParam, _AgConfigViewBase
from ..agpolicy import agdecision
from .agproxy_ptrace_internal._tracer_loop import SeccompStop, StopDecision, TracerLoop

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agent import agent
    from ..agpolicy import agpolicy


_ptrace_available_cache: "bool | None" = None
_ptrace_available_lock = threading.Lock()


def ptrace_available() -> bool:
    """Return True if this host can actually fork a child, PTRACE_TRACEME
    it, and receive its stops -- cached for the process lifetime, mirroring
    agsandbox_backends/chroot.py's `chroot_available()` convention: a
    sysctl/arch check alone can false-positive (e.g. a stricter LSM policy
    than Yama's ptrace_scope denying it in practice), so the authoritative
    answer is a live smoke test, not just reading a config file."""
    global _ptrace_available_cache
    if _ptrace_available_cache is not None:
        return _ptrace_available_cache
    with _ptrace_available_lock:
        if _ptrace_available_cache is None:
            _ptrace_available_cache = _probe_ptrace_available()
        return _ptrace_available_cache


def _probe_ptrace_available() -> bool:
    if platform.machine() not in ("x86_64", "AMD64"):
        return False
    try:
        from .agproxy_ptrace_internal import _ctypes_defs as pt
    except Exception:
        return False
    try:
        pid = os.fork()
    except OSError:
        return False
    if pid == 0:
        try:
            import signal

            pt.ptrace(pt.PTRACE_TRACEME, 0, 0, 0)
            # PTRACE_TRACEME alone does not generate a stop the parent can
            # observe -- a stop only occurs on the next signal delivery or
            # exec (see TracerLoop._child_exec's identical use of this to
            # synchronize with the parent before installing a seccomp
            # filter). Raise SIGSTOP on ourselves to produce one.
            os.kill(os.getpid(), signal.SIGSTOP)
        except Exception:
            os._exit(1)
        os._exit(0)
    try:
        _, status = os.waitpid(pid, 0)
    except OSError:
        return False
    if not os.WIFSTOPPED(status):
        return False
    try:
        pt.ptrace(pt.PTRACE_CONT, pid, 0, 0)
        os.waitpid(pid, 0)
    except Exception:
        pass
    return True


@dataclass
class agsyscallevent:
    """One intercepted syscall, resolved into agpolicy-friendly shape.
    `argv`/`envp`/`path` are populated only for syscalls this module knows
    how to resolve arguments for (execve/execveat: all three; open/openat:
    `path` only) -- see agproxy_ptrace_internal/_tracer_loop.py's
    `_resolve_syscall_args`.

    `tool_name`/`tool_args` are unused here -- they exist only so
    `agtool.dispatch_tools()`'s native-tool-call retrofit (see
    docs/Design_harness_integration.md's later build phase) can hand
    `agpolicy.check()` the same event type a syscall-level mediation gets,
    with `syscall="tool_call"` and `argv`/`envp`/`path` left `None`, rather
    than inventing a second event type policies would need to special-case.
    """

    syscall: str
    pid: int
    tid: int
    argv: "list[str] | None"
    envp: "dict[str, str] | None"
    path: "str | None"
    timestamp: float
    tool_name: "str | None" = None
    tool_args: "dict | None" = None


class _AgPtraceFields:
    """Every agproxy_ptrace tunable, as config descriptors -- see
    docs/agconfig.md for the tier-1 (GlobalConfigParam) vs. tier-3
    (DynamicConfigParam) distinction."""

    syscalls = DynamicConfigParam(
        "agproxy_ptrace", default=("execve", "execveat")
    )  # which syscalls the seccomp filter traps; see _ctypes_defs.SYSCALL_NUMBERS
    # for the full set this module knows how to resolve arguments for.
    profiler = DynamicConfigParam(
        "agproxy_ptrace", default=None
    )  # reserved for a future profiler hook (perf/strace-equivalent); unused so far.
    disable_harness_native_sandbox = DynamicConfigParam(
        "agproxy_ptrace", default=True
    )  # advisory flag for agharness backends: prefer disabling a harness's own
    # OS-level sandboxing (bwrap/seatbelt/landlock) when running under
    # agproxy_ptrace, to avoid seccomp-filter-stacking surprises (see the
    # design doc's "Design Tensions" section). Not enforced by this module.
    attach_timeout_s = GlobalConfigParam(
        "agproxy_ptrace", default=30
    )  # ceiling on waiting for the traced process's initial post-TRACEME stop.

    def __init__(self, agconfig: "agConfig | None" = None) -> None:
        self._agconfig = agconfig


class agPtraceConfig(_AgConfigViewBase):
    _OWNER = "agproxy_ptrace"


class agProxyPtraceHandle:
    """A single launched, traced process tree. Returned by
    `agProxyPtrace.launch()`; not constructed directly."""

    def __init__(self, loop: TracerLoop) -> None:
        self._loop = loop

    def wait(self, timeout: "float | None" = None) -> "tuple[str, str, int]":
        """Block until the root process exits (or *timeout* elapses).
        Returns `(stdout, stderr, returncode)` -- deliberately the same
        shape family as `agSandbox.exec()`'s `(str, int)`, with stderr
        broken out separately since, unlike a shell wrapper script, there
        is no single combined-stream convention to lean on here. Output is
        continuously drained by dedicated reader threads for the lifetime
        of the launch (see `TracerLoop._drain_pipe`), so this only needs to
        join and read back whatever has accumulated."""
        returncode = self._loop.join(timeout=timeout)
        stdout, stderr = self._loop.read_output()
        return stdout, stderr, (returncode if returncode is not None else -1)

    def pids(self) -> "set[int]":
        return self._loop.live_pids()

    def on_spawn(self, callback: "Callable[[int], None]") -> None:
        self._loop.on_spawn(callback)

    def on_exit(self, callback: "Callable[[int], None]") -> None:
        """*callback* receives `(pid, exit_code)` -- exit_code follows
        `_tracer_loop._forget`'s convention: the process's real exit code if
        it exited normally, or the negative signal number if it was
        killed by a signal."""
        self._loop.on_exit(lambda pid, code: callback(pid))

    def kill(self) -> None:
        self._loop.kill()


class agProxyPtrace:
    """Entry point for launching a process under syscall-level supervision.
    One instance is reusable across multiple `launch()` calls -- it only
    holds config, not per-launch state (that lives on the returned
    `agProxyPtraceHandle`/`TracerLoop`)."""

    def __init__(self, agconfig: "agConfig | None" = None) -> None:
        self._agconfig = agconfig

    def launch(
        self,
        argv: "list[str]",
        envp: "dict[str, str]",
        *,
        cwd: str = "",
        policy: "agpolicy",
        ag: "agent | None" = None,
        sandbox=None,
    ) -> agProxyPtraceHandle:
        """*sandbox*, when given, selects the launch path: a docker/podman-
        backed sandbox (`IMAGE_KIND == "container"`) forks the traced child
        *inside the container* via a `docker/podman exec`-launched
        entrypoint (see `agproxy_ptrace_internal/_in_container_launcher.py`
        for why a host-side `fork()` cannot land a child in a different PID
        namespace). Any other sandbox (chroot, or none at all -- a bare
        host-level launch) uses the existing host-fork `TracerLoop` path,
        unchanged."""
        syscalls = _AgPtraceFields(self._agconfig).syscalls

        if sandbox is not None and getattr(sandbox._backend, "IMAGE_KIND", "") == "container":
            from .agproxy_ptrace_internal._in_container_launcher import InContainerRelay

            relay = InContainerRelay(sandbox=sandbox, policy=policy, ag=ag)
            relay.start(argv, envp, cwd, syscalls)
            return agProxyPtraceHandle(relay)

        def syscall_hook(stop: SeccompStop) -> StopDecision:
            event = agsyscallevent(
                syscall=stop.syscall,
                pid=stop.pid,
                tid=stop.pid,
                argv=stop.argv,
                envp=stop.envp,
                path=stop.path,
                timestamp=stop.timestamp,
            )
            decision = policy.check(ag, event)
            if decision.kind == "deny":
                return StopDecision(kind="deny")
            if decision.kind == "rewrite":
                return StopDecision(kind="rewrite", new_args=decision.new_args)
            return StopDecision(kind="allow")

        loop = TracerLoop(syscalls=syscalls, syscall_hook=syscall_hook)
        loop.start(argv, envp, cwd)
        return agProxyPtraceHandle(loop)


def wire_to_sandbox(handle: agProxyPtraceHandle, sandbox) -> None:
    """Feed *handle*'s fork/exit events into *sandbox*'s PID bookkeeping, so
    `sandbox.get_live_pids()`/`.wait_for_processes()`/`.pid_status_summary()`
    reflect a harness-driven agent's traced process tree exactly as they
    would a native agent's — see `agsandbox_backend.ingest_ptrace_pids()`
    (agsandbox_backends/base.py) for what "reflect" means precisely
    (ptrace-sourced pids are trusted independent of the sandbox's own
    `/proc` scan). Called by `agharness_backends` right after `launch()`;
    not required for launches that don't need sandbox-level PID tracking
    (e.g. a bare host-level launch with no agSandbox at all)."""
    handle.on_spawn(lambda pid: sandbox._backend.ingest_ptrace_pids(spawned={pid}))
    handle.on_exit(lambda pid: sandbox._backend.ingest_ptrace_pids(exited={pid}))


__all__ = [
    "agsyscallevent",
    "agdecision",
    "agPtraceConfig",
    "agProxyPtrace",
    "wire_to_sandbox",
    "agProxyPtraceHandle",
    "ptrace_available",
]
