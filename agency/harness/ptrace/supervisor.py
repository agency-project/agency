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
import re
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ...agconfig import GlobalConfigParam, DynamicConfigParam, _AgConfigViewBase
from ...agpolicy import agdecision
from .._syscall_event import agsyscallevent
from ._tracer_loop import SeccompStop, StopDecision, TracerLoop

if TYPE_CHECKING:
    from ...agconfig import agConfig
    from ...agent import agent
    from ...agpolicy import agpolicy


_ptrace_available_cache: "bool | None" = None
_ptrace_available_lock = threading.Lock()


def ptrace_available() -> bool:
    """Return True if this host can actually fork a child, PTRACE_TRACEME
    it, and receive its stops -- cached for the process lifetime, mirroring
    sandbox/chroot.py's `chroot_available()` convention: a
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
        from .ptrace import _ctypes_defs as pt
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
    except Exception:  # noqa: S110 - availability-probe cleanup is best-effort
        pass
    return True


@dataclass
class _ProcessSpanStart:
    """Host-clock timestamps retained until a traced process exits."""

    perf_ns: int
    wall_ns: int
    executable: str
    external_span: object | None


_SAFE_EXECUTABLE_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._+-"
)
_CREDENTIAL_SHAPED_EXECUTABLE = re.compile(
    r"(?:[0-9a-fA-F]{24,64}|"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}|"
    r"(?:sk|ghp|github_pat|xox[baprs])[-_][A-Za-z0-9._+-]{8,})"
)
_SENSITIVE_ENV_KEY_PARTS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "AUTH",
    "CREDENTIAL",
    "KEY",
)


def _executable_display_name(
    executable_path: "str | None", *, sensitive_values: "frozenset[str]" = frozenset()
) -> str:
    """Return a bounded, argument-free executable name for a trace span.

    Harness argv frequently contains both the full user prompt and bearer
    credentials (for example in an inline MCP configuration), and ``argv[0]``
    itself is attacker-controlled via facilities such as ``exec -a``. Process
    identity therefore comes only from the kernel-confirmed executable path
    delivered at ``PTRACE_EVENT_EXEC``. Only its basename is retained so a
    credential-bearing parent directory cannot reach a profiler artifact.
    """
    if not isinstance(executable_path, str):
        return "<unknown>"
    # procfs marks an unlinked image as ``/path/name (deleted)``. The suffix is
    # kernel metadata rather than part of the executable identity.
    if executable_path.endswith(" (deleted)"):
        executable_path = executable_path[: -len(" (deleted)")]
    basename = os.path.basename(executable_path.rstrip("/"))
    if not basename:
        return "<unknown>"
    # Span names are exported to JSON. Refuse rather than partially echo a
    # pathological filename: control/bidi characters, URI punctuation, query
    # strings, or overlong credential-shaped basenames must not be copied into
    # an artifact. Normal Unix command names fit this conservative alphabet.
    if len(basename) > 64 or any(
        character not in _SAFE_EXECUTABLE_CHARACTERS for character in basename
    ):
        return "<redacted>"
    if any(secret in basename for secret in sensitive_values) or (
        _CREDENTIAL_SHAPED_EXECUTABLE.fullmatch(basename)
    ):
        return "<redacted>"
    return basename


def _sensitive_environment_values(envp: "dict[str, str]") -> "frozenset[str]":
    """Extract exact launch credentials that must never become span names."""
    return frozenset(
        value
        for key, value in envp.items()
        if isinstance(value, str)
        and value
        and any(marker in key.upper() for marker in _SENSITIVE_ENV_KEY_PARTS)
    )


class _ProcessLifecycleProfiler:
    """Translate ptrace spawn/exec/exit observations into agprof records.

    A lifecycle is measured from the synchronous spawn callback through the
    synchronous exit callback. A successful PTRACE_EVENT_EXEC supplies a safe
    executable display name; raw argv/envp are intentionally never retained.
    """

    def __init__(
        self,
        *,
        parent_context,
        span_attributes: dict,
        sensitive_values: "frozenset[str]" = frozenset(),
        timing: str = "exact",
    ) -> None:
        self._parent_context = parent_context
        self._span_attributes = dict(span_attributes)
        self._sensitive_values = sensitive_values
        self._timing = timing
        self._processes: "dict[int, _ProcessSpanStart]" = {}
        self._finalized = False
        self._lock = threading.Lock()

    @classmethod
    def for_active_session(
        cls, ag: "agent | None", envp: "dict[str, str]", *, timing: str = "exact"
    ) -> "_ProcessLifecycleProfiler | None":
        from ...profiler import agprof

        if not agprof.enabled():
            return None
        attributes = agprof.current_span_attributes()
        if ag is not None:
            attributes.setdefault("agency.agent_id", str(ag.agname))
            parent_agent_id = getattr(ag, "_parent_agent_id", None)
            if parent_agent_id is not None:
                attributes.setdefault("agency.parent_agent_id", str(parent_agent_id))
        return cls(
            parent_context=agprof.current_span_context(),
            span_attributes=attributes,
            sensitive_values=_sensitive_environment_values(envp),
            timing=timing,
        )

    def on_spawn(self, pid: int) -> None:
        from ...profiler import agprof

        start_perf_ns = time.perf_counter_ns()
        start_wall_ns = time.time_ns()
        with self._lock:
            if self._finalized:
                return
            if pid in self._processes:
                return
            # Never derive process identity from launch argv: argv[0] is
            # attacker-controlled. PTRACE_EVENT_EXEC will update this only
            # after the kernel has successfully installed the image.
            executable = "<unknown>"
            metadata = {
                **self._span_attributes,
                "timing": self._timing,
                "provenance": "ptrace",
                "pid": pid,
                "executable": executable,
            }
            self._processes[pid] = _ProcessSpanStart(
                start_perf_ns,
                start_wall_ns,
                executable,
                agprof.start_external_span(
                    f"process:{executable}",
                    start_perf_ns=start_perf_ns,
                    start_wall_ns=start_wall_ns,
                    metadata=metadata,
                    parent_context=self._parent_context,
                ),
            )

    def on_exec(self, pid: int, executable_path: "str | None") -> None:
        """Commit a kernel-confirmed executable identity for a live process."""
        executable = _executable_display_name(
            executable_path, sensitive_values=self._sensitive_values
        )
        with self._lock:
            if self._finalized:
                return
            process = self._processes.get(pid)
            if process is None:
                return
            process.executable = executable
            if process.external_span is not None:
                process.external_span.update(f"process:{executable}", executable=executable)

    def on_exit(self, pid: int, exit_code: int) -> None:
        end_perf_ns = time.perf_counter_ns()
        end_wall_ns = time.time_ns()
        with self._lock:
            process = self._processes.pop(pid, None)
        if process is None:
            return

        metadata = {
            "executable": process.executable,
            "exit_code": exit_code,
            "outcome": "success" if exit_code == 0 else "failure",
        }
        if process.external_span is not None:
            process.external_span.end(
                end_perf_ns=max(process.perf_ns, end_perf_ns),
                end_wall_ns=max(process.wall_ns, end_wall_ns),
                metadata=metadata,
            )

    def finalize(self) -> None:
        """Preserve every still-live process as an interrupted agprof span."""
        from ...profiler import agprof

        with self._lock:
            if self._finalized:
                return
            self._finalized = True
            processes = list(self._processes.values())
            self._processes.clear()
        ended_perf_ns = time.perf_counter_ns()
        for process in processes:
            # Profiling is observational. A broken exporter must not mask the
            # launch failure whose cleanup called finalize().
            with suppress(Exception):
                agprof.interrupt_external_span(
                    process.external_span,
                    ended_perf_ns=max(process.perf_ns, ended_perf_ns),
                )


def _isolated_profiler_callback(callback: Callable) -> Callable:
    """Keep automatic telemetry failures off the ptrace supervision thread."""

    def invoke(*args) -> None:
        try:
            callback(*args)
        except Exception:
            # These callbacks run synchronously while a tracee is stopped. If
            # optional agprof/OTel code escapes, the tracer thread dies and the
            # child remains stopped forever. User callbacks are intentionally
            # not wrapped; this isolation is only for built-in telemetry.
            return

    return invoke


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
    )  # reserved for selecting a future heavyweight process profiler such as
    # perf. This is deliberately separate from the automatic, low-cost agprof
    # lifecycle records above: an active agprof session must see Tier-2
    # spawn/exit coverage without requiring an unrelated config selector.
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

    def __init__(
        self,
        loop: TracerLoop,
        process_profiler: "_ProcessLifecycleProfiler | None" = None,
    ) -> None:
        self._loop = loop
        self._process_profiler = process_profiler

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

    def on_exec(self, callback: "Callable[[int, str | None], None]") -> None:
        """Register a replay-safe successful-exec callback.

        The path is staged from the exec syscall's pathname and delivered only
        after ``PTRACE_EVENT_EXEC`` confirms success; ``/proc/<pid>/exe`` is a
        fallback when no pathname was trapped. It is never taken from argv[0].
        """
        self._loop.on_exec(callback)

    def on_exit(
        self,
        callback: "Callable[[int], None] | Callable[[int, int], None]",
        *,
        include_exit_code: bool = False,
    ) -> None:
        """Register a replay-safe process-exit callback.

        Existing one-argument callbacks continue to receive ``pid``.  Pass
        ``include_exit_code=True`` for ``callback(pid, exit_code)``; the code
        is the process's real exit status, or the negative signal number when
        it was killed.  Keeping this opt-in preserves the original public
        callback shape while making the ptrace-observed status available to
        lifecycle consumers such as agprof.
        """
        if include_exit_code:
            self._loop.on_exit(callback)
        else:
            self._loop.on_exit(lambda pid, _code: callback(pid))

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
        container_launch = (
            sandbox is not None and getattr(sandbox._backend, "IMAGE_KIND", "") == "container"
        )
        process_profiler = _ProcessLifecycleProfiler.for_active_session(
            ag,
            envp,
            # In-container spawn/exit notifications cross a UDS before their
            # host callbacks run, so their host-clock boundaries include
            # unmeasured relay jitter and must not claim exact timing.
            timing="host-observed" if container_launch else "exact",
        )

        if container_launch:
            from ..old_ptrace._in_container_launcher import InContainerRelay

            relay = InContainerRelay(
                sandbox=sandbox,
                policy=policy,
                ag=ag,
            )
            handle = agProxyPtraceHandle(relay, process_profiler)
            if process_profiler is not None:
                handle.on_spawn(_isolated_profiler_callback(process_profiler.on_spawn))
                handle.on_exec(_isolated_profiler_callback(process_profiler.on_exec))
                handle.on_exit(
                    _isolated_profiler_callback(process_profiler.on_exit),
                    include_exit_code=True,
                )
            try:
                relay.start(argv, envp, cwd, syscalls)
            except BaseException:
                if process_profiler is not None:
                    process_profiler.finalize()
                raise
            return handle

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
        handle = agProxyPtraceHandle(loop, process_profiler)
        if process_profiler is not None:
            handle.on_spawn(_isolated_profiler_callback(process_profiler.on_spawn))
            handle.on_exec(_isolated_profiler_callback(process_profiler.on_exec))
            handle.on_exit(
                _isolated_profiler_callback(process_profiler.on_exit),
                include_exit_code=True,
            )
        try:
            loop.start(argv, envp, cwd)
        except BaseException:
            if process_profiler is not None:
                process_profiler.finalize()
            raise
        return handle


def wire_to_sandbox(handle: agProxyPtraceHandle, sandbox) -> None:
    """Feed *handle*'s fork/exit events into *sandbox*'s PID bookkeeping, so
    `sandbox.get_live_pids()`/`.wait_for_processes()`/`.pid_status_summary()`
    reflect a harness-driven agent's traced process tree exactly as they
    would a native agent's — see `agsandbox_backend.ingest_ptrace_pids()`
    (sandbox/base.py) for what "reflect" means precisely
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
