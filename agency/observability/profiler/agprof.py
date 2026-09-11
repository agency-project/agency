"""agprof — profiling switch for the framework's built-in span annotations.

The framework's hot paths (skill runs, LLM calls, tool dispatch, sandbox ops,
agmap tasks) are permanently annotated with ``agprof.span(name)`` calls. With no
active session (the default) every one of them returns a shared no-op context
manager — one module-global load and an ``is None`` check. The OpenTelemetry
SDK remains an optional dependency and no spans are created until profiling
starts. Applications opt in with one line (or one env var):

    from agency import agprof

    with agprof.session(run_dir / "profile"):
        team.run()

    # or, zero application changes with process-lifetime scope:
    #   AGENCY_PROFILE=1 AGENCY_PROFILE_SCOPE=process python app.py

Backend: OpenTelemetry SDK with an always-on sampler. ``span()`` starts an OTel
span and records the same exact timestamps and attributes in the summary input
when the span ends. The profiler uses a private ``TracerProvider`` so it does
not replace or depend on an application's global provider.

CPU-vs-wait split: every span additionally records its thread's on-CPU time
(``time.thread_time_ns``) and run-queue wait (``/proc/self/task/<tid>/schedstat``)
across the span. The deltas are attached directly as OTel span attributes:

    cpu_ms       thread executed on a core        → compute
    runqueue_ms  runnable, waiting for a core     → scheduling/GIL contention
    blocked_ms   wall − cpu − runqueue            → waiting (model/IO/sync/GPU)

``summary_table()`` aggregates the same numbers per label after a session.
GPU activity is deliberately NOT part of this split — it happens outside the
calling thread and is attributed via lease intervals + device sampling, not thread clocks.
"""

from __future__ import annotations

import atexit
import copy
import importlib.util
import itertools
import json
import os
import re
import signal
import subprocess
import sysconfig
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager, nullcontext, suppress
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

_NULL = nullcontext()

_session = None  # None = profiling off (the fast path checks only this)
_profiler = None  # the live _OTelSession object, if any
_out_dir: "Path | None" = None
_state_lock = threading.Lock()
_process_shutdown_lock = threading.Lock()
_process_shutdown_started = False
_PROCESS_PROFILE_SIGNALS = (signal.SIGTERM, signal.SIGINT)

_counters: "dict[str, itertools.count]" = {}
_counters_lock = threading.Lock()

_profile_session_id: "str | None" = None
_profile_data_logger = None
_last_profile_records: "list[tuple]" = []
_open_spans: "dict[int, object]" = {}
_open_spans_lock = threading.Lock()
_interrupted_spans: "list[dict]" = []
_last_summary: "dict[str, dict] | None" = None
_last_run_summary: "dict | None" = None
_health = Counter()
_engine_coverage: dict[str, dict] = {}

_tls = threading.local()  # per-thread schedstat file cache only
_span_stack: "ContextVar[tuple[_TimedSpan, ...]]" = ContextVar("agprof_span_stack", default=())

# Sampler timeline + GPU lease intervals (see _Sampler / gpu_lease_*).
_samples: "list[tuple[int, str, float]]" = []  # (t_mono_ns, series, value)
_process_info: "dict[str, dict]" = {}  # pid-start_ticks identity -> trace/display metadata
_profile_root_pid: "int | None" = None
_thread_labels: "dict[tuple[int, int], tuple[int, str]]" = {}
_sampler: "_Sampler | None" = None
_leases_open: "dict[int, tuple[int, str]]" = {}  # gpu_id -> (t0_ns, label)
_leases: "list[tuple[int, int, int, str]]" = []  # (gpu_id, t0_ns, t1_ns, label)
_leases_lock = threading.Lock()
_session_started_ns: "int | None" = None
_session_sample_hz = 0.0
_session_sample_gpu = False

# Filtered automatic Python call intervals.
# These remain separate from semantic OTel spans so users can
# keep intentional agprof.span() overlays without losing unannotated work.
_auto_records: list[tuple] = []
_auto_stacks: "dict[tuple[int, int], list[tuple]]" = {}
_auto_code_labels: dict = {}
_auto_settings: "dict | None" = None
_auto_dropped = 0
_auto_filtered = Counter()
_auto_tool_in_use = False

_DEFAULT_AUTO_MIN_DURATION_MS = 1.0
_DEFAULT_AUTO_MAX_DEPTH = 32
_DEFAULT_AUTO_MAX_EVENTS = 250_000
_DYNAMIC_THREAD_MIN_DURATION_NS = 10_000_000

# Environment profiling is relaunched into a transient systemd cgroup before
# the workload starts. Prefer a system slice when passwordless sudo is
# available: its child scope contains the harness and Docker containers can be
# placed alongside that scope. Otherwise use an unprivileged user scope for the
# harness; Docker containers remain in their daemon-managed cgroups and are
# combined into the same trace through the container registry below.
_CGROUP_DIR_ENV = "AGENCY_PROFILE_CGROUP"
_CGROUP_PARENT_ENV = "AGENCY_PROFILE_CGROUP_PARENT"
_CGROUP_USER_SCOPE_ENV = "AGENCY_PROFILE_USER_SCOPE"
_CGROUP_SLICE_RE = re.compile(r"^agprof-[0-9a-f]+\.slice$")
_CGROUP_SCOPE_RE = re.compile(r"^agprof-[0-9a-f]+\.scope$")

# Marks a process already re-exec'd through the no-sudo systemd --user
# scope path (see _user_cgroup_reexec_command); its cgroup path isn't
# predictable in advance like the sudo path's, so it's discovered after
# landing instead.
_CGROUP_USER_REEXEC_ENV = "AGENCY_PROFILE_CGROUP_USER_REEXEC"

# Container cgroup registry — filled by the sandbox backends at container
# start (docker + podman)
_cg_registry: "dict[str, str]" = {}  # label (agname) -> cgroup dir
_daemon_cg: "dict[str, str]" = {}  # cgroup dir -> agg kind ("conmon"/"dockerd")
_cg_lock = threading.Lock()

# Pseudo thread-id lane for spans observed outside a local calling thread.
_DERIVED_TID = -1


def _agprof_print(message: str) -> None:
    """Emit shutdown diagnostics promptly even when stdout is redirected."""
    print(message, flush=True)


def _require_linux() -> None:
    """Reject profiling before any profiler output or workload is started."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "agprof: profiling is Linux-only; this operating system is unsupported "
            "because profiling requires cgroups v2 and Linux /proc kernel interfaces"
        )


def _current_cgroup_dir() -> Path:
    """Resolve this process's unified cgroup v2 directory."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
        relative = next(line.split("::", 1)[1] for line in lines if line.startswith("0::"))
    except (OSError, StopIteration, IndexError) as e:
        raise RuntimeError("agprof: profiling requires a readable Linux cgroup v2 hierarchy") from e
    return Path("/sys/fs/cgroup") / relative.lstrip("/")


def _process_cgroup_dir() -> Path:
    """Return the cgroup whose counters represent the profiled workload."""
    configured = os.environ.get(_CGROUP_DIR_ENV)
    cgroup_dir = Path(configured) if configured else _current_cgroup_dir()
    required = ("cpu.stat", "memory.current", "cgroup.procs")
    missing = [name for name in required if not (cgroup_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"agprof: workload cgroup {str(cgroup_dir)!r} is unusable "
            f"(missing {', '.join(missing)}); profiling requires Linux cgroups v2"
        )
    return cgroup_dir


def container_cgroup_parent() -> "str | None":
    """Docker cgroup parent for containers created by the profiled workload."""
    if os.environ.get(_CGROUP_USER_SCOPE_ENV):
        return None
    value = os.environ.get(_CGROUP_PARENT_ENV, "")
    return value if _CGROUP_SLICE_RE.fullmatch(value) else None


# wraps in cgroup
def _cgroup_reexec_command(slice_name: str, cgroup_dir: str) -> list[str]:
    """Build the privilege-separated systemd command used by env profiling."""
    uid = os.getuid()
    gid = os.getgid()
    user = os.environ.get("USER") or str(uid)
    home = os.environ.get("HOME") or str(Path.home())
    orig_argv = getattr(sys, "orig_argv", None)
    # argv[0] must be the resolved interpreter path: the reconstructed command
    # below runs through sudo/systemd-run/setpriv, whose secure_path can
    # override $PATH even with sudo -E, so a bare "python3" (whatever the
    # caller typed) can resolve to a different interpreter than the one
    # actually running -- silently losing this venv's installed packages.
    original_argv = [sys.executable, *orig_argv[1:]] if orig_argv else [sys.executable, *sys.argv]
    scope_name = slice_name.removesuffix(".slice") + ".scope"
    profiler_environment = [
        f"{key}={os.environ[key]}"
        for key in ("AGENCY_PROFILE", "AGENCY_PROFILE_DIR", "AGENCY_PROFILE_SCOPE")
        if key in os.environ
    ]
    return [
        "sudo",
        "-n",
        "-E",
        "systemd-run",
        "--scope",
        "--collect",
        "--quiet",
        "--same-dir",
        f"--slice={slice_name}",
        f"--unit={scope_name}",
        "setpriv",
        f"--reuid={uid}",
        f"--regid={gid}",
        "--init-groups",
        "env",
        f"HOME={home}",
        f"USER={user}",
        f"LOGNAME={user}",
        f"{_CGROUP_DIR_ENV}={cgroup_dir}",
        f"{_CGROUP_PARENT_ENV}={slice_name}",
        *profiler_environment,
        *original_argv,
    ]


def _user_scope_available() -> bool:
    """Probe whether systemd user-session cgroup delegation works here --
    a real subprocess, not a re-exec, so a failure can fall back cleanly."""

    try:
        result = subprocess.run(
            ["systemd-run", "--user", "--scope", "--collect", "--quiet", "true"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _user_cgroup_reexec_command() -> list[str]:
    """No-sudo systemd --user re-exec command -- no privilege separation
    needed since this never changes uid. Doesn't set _CGROUP_PARENT_ENV:
    a user-session scope isn't usable as a docker --cgroup-parent (dockerd
    runs as root, outside this delegated subtree)."""
    orig_argv = getattr(sys, "orig_argv", None)
    original_argv = [sys.executable, *orig_argv[1:]] if orig_argv else [sys.executable, *sys.argv]
    run_id = f"{os.getpid():x}{uuid.uuid4().hex[:8]}"
    scope_name = f"agprof-{run_id}.scope"
    profiler_environment = [
        f"{key}={os.environ[key]}"
        for key in ("AGENCY_PROFILE", "AGENCY_PROFILE_DIR", "AGENCY_PROFILE_SCOPE")
        if key in os.environ
    ]
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--collect",
        "--quiet",
        "--same-dir",
        f"--unit={scope_name}",
        "env",
        f"{_CGROUP_USER_REEXEC_ENV}=1",
        *profiler_environment,
        *original_argv,
    ]


def _ensure_environment_cgroup() -> None:
    """Re-exec environment-enabled profiling inside a dedicated cgroup.
    Tries the no-sudo systemd --user scope first, falling back to
    sudo/system-slice only if that's unavailable."""
    configured = os.environ.get(_CGROUP_DIR_ENV)
    if configured:
        cgroup_dir = _process_cgroup_dir()
        current = _current_cgroup_dir()
        try:
            current.relative_to(cgroup_dir)
        except ValueError as e:
            raise RuntimeError(
                f"agprof: process cgroup {current} is outside configured workload "
                f"cgroup {cgroup_dir}"
            ) from e
        return

    if os.environ.get(_CGROUP_USER_REEXEC_ENV):
        # Already landed inside the --user scope's own cgroup -- discover
        # it rather than re-exec'ing again.
        cgroup_dir = _current_cgroup_dir()
        required = ("cpu.stat", "memory.current", "cgroup.procs")
        missing = [name for name in required if not (cgroup_dir / name).is_file()]
        if missing:
            raise RuntimeError(
                f"agprof: user-scope workload cgroup {cgroup_dir} is unusable "
                f"(missing {', '.join(missing)}); profiling requires Linux cgroups v2"
            )
        os.environ[_CGROUP_DIR_ENV] = str(cgroup_dir)
        return

    if _user_scope_available():
        command = _user_cgroup_reexec_command()
        try:
            os.execvp(command[0], command)
        except OSError:
            pass  # fall through to the sudo/system-slice path below

    run_id = f"{os.getpid():x}{uuid.uuid4().hex[:8]}"
    slice_name = f"agprof-{run_id}.slice"
    cgroup_dir = f"/sys/fs/cgroup/agprof.slice/{slice_name}"
    command = _cgroup_reexec_command(slice_name, cgroup_dir)
    try:
        os.execvp(command[0], command)
    except OSError as e:
        raise RuntimeError(
            "agprof: unable to create the dedicated workload cgroup via either "
            "the systemd --user scope or sudo/systemd-run; profiling requires "
            "Linux cgroups v2 and either systemd user-session cgroup "
            "delegation or passwordless permission to create a transient "
            "systemd scope"
        ) from e


def container_started(
    label: str, cgroup_dir: str, daemon_cgroup_dir: "str | None" = None, daemon_kind: str = "conmon"
) -> None:
    """Register a container's cgroup for sampling (called by sandbox backends).

    *cgroup_dir* must be the kernel-reported cgroup v2 directory of the
    container (from /proc/<pid>/cgroup). Raises while a session is active if
    the directory has no cpu.stat — only docker/podman on cgroup v2 are
    supported, and a profiled run on anything else should fail loudly rather
    than silently produce no container metrics.
    """
    if _session is not None and not os.path.isfile(f"{cgroup_dir}/cpu.stat"):
        raise RuntimeError(
            f"agprof: cannot sample container cgroup {cgroup_dir!r} (no cpu.stat). "
            "Only docker/podman on cgroup v2 are supported."
        )
    with _cg_lock:
        _cg_registry[label] = cgroup_dir
        if daemon_cgroup_dir is not None:
            _daemon_cg[daemon_cgroup_dir] = daemon_kind


def container_stopped(label: str) -> None:
    """Drop a stopped or removed container from the sampling registry."""
    with _cg_lock:
        _cg_registry.pop(label, None)


def container_registered(label: str) -> bool:
    """True if *label* currently has a registered cgroup dir."""
    with _cg_lock:
        return label in _cg_registry


def _read_schedstat() -> "int | None":
    """This thread's cumulative run-queue wait (ns), or None if unavailable.

    Field 2 of /proc/self/task/<tid>/schedstat: time spent runnable but not
    executing. The fd is cached per thread (seek+read afterwards, ~µs).
    """
    try:
        f = getattr(_tls, "schedstat", None)
        if f is None:
            f = open(f"/proc/self/task/{threading.get_native_id()}/schedstat", "rb")
            _tls.schedstat = f
        f.seek(0)
        return int(f.read().split()[1])
    except Exception:
        return None


def detail_metadata(name: str, value, *, max_chars: int = 32_768) -> dict:
    """Snapshot readable span details with an explicit per-field size limit."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return {
        name: text[:max_chars],
        f"{name}_truncated": len(text) > max_chars,
        f"{name}_chars": len(text),
    }


def _otel_attribute(value):
    """Return an OTel-compatible scalar/sequence without losing metadata."""
    if isinstance(value, (bool, str, bytes, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, (bool, str, bytes, int, float)) for item in value
    ):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _record_data_logger_span(
    span_name: str,
    start_wall_ns: int,
    end_wall_ns: int,
    attributes: dict,
    *,
    cpu_ns: "int | None" = None,
    runqueue_ns: "int | None" = None,
    span_id: "int | None" = None,
    parent_span_id: "int | None" = None,
    profile_session_id: "str | None" = None,
    profile_span_name: "str | None" = None,
) -> bool:
    """Best-effort bridge from a completed profiler span to agprof's own agDataLogger."""
    data_logger = _profile_data_logger
    if data_logger is None:
        return False
    wall_ns = max(0, end_wall_ns - start_wall_ns)
    blocked_ns = (
        None
        if cpu_ns is None or runqueue_ns is None
        else max(0, attributes.get("agency.wall_ns", wall_ns) - cpu_ns - runqueue_ns)
    )
    logger_attributes = dict(attributes)
    if profile_session_id is not None:
        logger_attributes["agency.profile_session_id"] = profile_session_id
    if profile_span_name is not None and profile_span_name != span_name:
        logger_attributes["agency.profile_span_name"] = profile_span_name
    if span_id is not None:
        logger_attributes["agency.span_id"] = f"{span_id:016x}"
    if parent_span_id is not None:
        logger_attributes["agency.parent_span_id"] = f"{parent_span_id:016x}"
    try:
        data_logger.record_span(
            span_name,
            start_wall_ns / 1_000_000_000,
            end_wall_ns / 1_000_000_000,
            logger_attributes,
            cpu_ms=None if cpu_ns is None else cpu_ns / 1_000_000,
            runqueue_ms=None if runqueue_ns is None else runqueue_ns / 1_000_000,
            blocked_ms=None if blocked_ns is None else blocked_ns / 1_000_000,
            parent=None if parent_span_id is None else f"{parent_span_id:016x}",
            call_label=attributes.get("agency.run_id"),
        )
        return True
    except Exception as exc:
        # Profiling and persistence are observational. Neither may alter the
        # execution result whose span is being recorded.
        _health["span_export_failures"] += 1
        _agprof_print(f"[agprof] WARNING: datalogger span export failed: {exc}")
        return False


def _load_profile_records(profile_session_id: "str | None") -> list[tuple]:
    """Flush and read this session's spans from agprof's own data logger."""
    data_logger = _profile_data_logger
    if profile_session_id is None or data_logger is None:
        return []
    try:
        data_logger.flush()
        records = data_logger.read_profile_records(profile_session_id)
    except Exception as exc:
        _health["span_read_failures"] += 1
        _agprof_print(f"[agprof] WARNING: datalogger span read failed: {exc}")
        return []
    records.sort(key=lambda record: (record[2], record[0], record[1]))
    return records


class _TimedSpan:
    """OTel span plus exact wall/CPU/run-queue deltas from this thread."""

    __slots__ = (
        "_name",
        "_tracer",
        "_parent_context",
        "_span",
        "_scope",
        "_otel_t0",
        "_t0",
        "_cpu0",
        "_rq0",
        "_tid",
        "_metadata",
        "_interrupted",
        "_stack_token",
        "_profile_session_id",
    )

    def __init__(self, tracer, name: str, parent_context=None) -> None:
        self._name = name
        self._tracer = tracer
        self._parent_context = parent_context
        self._span = None
        self._scope = None
        self._metadata: dict = {}
        self._interrupted = False
        self._stack_token = None
        self._profile_session_id = _profile_session_id

    def __enter__(self) -> "_TimedSpan":
        self._t0 = time.perf_counter_ns()
        self._cpu0 = time.thread_time_ns()
        self._rq0 = _read_schedstat()
        self._tid = threading.get_native_id()
        semantic_thread = _span_thread_label(self._name)
        if semantic_thread is not None:
            _remember_thread_label(semantic_thread, priority=80, tid=self._tid)
        self._otel_t0 = time.time_ns()
        self._span = self._tracer.start_span(
            self._name,
            context=self._parent_context,
            start_time=self._otel_t0,
        )
        from opentelemetry.trace import use_span

        self._scope = use_span(self._span, end_on_exit=False)
        self._scope.__enter__()
        self._stack_token = _span_stack.set((*_span_stack.get(), self))
        with _open_spans_lock:
            _open_spans[id(self)] = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        stack_token = getattr(self, "_stack_token", None)
        if stack_token is not None:
            _span_stack.reset(stack_token)
        with _open_spans_lock:
            _open_spans.pop(id(self), None)
        if self._interrupted:
            if self._scope is not None:
                self._scope.__exit__(exc_type, exc_val, exc_tb)
            return
        t1 = time.perf_counter_ns()
        otel_t1 = time.time_ns()
        cpu1 = time.thread_time_ns()
        rq1 = _read_schedstat()
        runq = (rq1 - self._rq0) if (rq1 is not None and self._rq0 is not None) else None
        metadata = dict(self._metadata)
        metadata.setdefault("outcome", "failure" if exc_type is not None else "success")
        if exc_type is not None:
            metadata.setdefault("error_type", exc_type.__name__)
        measurements = {
            "agency.thread_id": self._tid,
            "agency.perf_start_ns": self._t0,
            "agency.wall_ns": t1 - self._t0,
            "agency.cpu_ns": cpu1 - self._cpu0,
        }
        if runq is not None:
            measurements["agency.runq_ns"] = runq
        for key, value in {**metadata, **measurements}.items():
            self._span.set_attribute(key, _otel_attribute(value))
        if exc_val is not None:
            self._span.record_exception(exc_val)
        context = self._span.get_span_context()
        parent = self._span.parent
        attributes = {**metadata, **measurements}
        span_id = context.span_id if context is not None else None
        parent_span_id = parent.span_id if parent is not None else None
        _record_data_logger_span(
            self._name,
            self._otel_t0,
            otel_t1,
            attributes,
            cpu_ns=cpu1 - self._cpu0,
            runqueue_ns=runq,
            span_id=span_id,
            parent_span_id=parent_span_id,
            profile_session_id=self._profile_session_id,
            profile_span_name=self._name,
        )
        self._span.end(end_time=otel_t1)
        if self._scope is not None:
            self._scope.__exit__(exc_type, exc_val, exc_tb)

    def interrupt(self, ended_perf_ns: int) -> None:
        """End this span in OTel while preserving incomplete-summary semantics."""
        self._interrupted = True
        attributes = {
            **self._metadata,
            "outcome": "interrupted",
            "agency.incomplete": True,
            "agency.thread_id": self._tid,
            "agency.perf_start_ns": self._t0,
            "agency.wall_ns": max(0, ended_perf_ns - self._t0),
        }
        for key, value in attributes.items():
            self._span.set_attribute(key, _otel_attribute(value))
        self._span.end(end_time=time.time_ns())

    def annotate(self, **metadata) -> None:
        """Attach JSON-safe outcome/metric fields to this span."""
        self._metadata.update(metadata)


class _ObservedSpan:
    """An exact host-observed interval with no executing host thread.

    Unlike :class:`_TimedSpan`, this handle can be opened by one callback and
    ended by a later callback without keeping an OTel context manager entered.
    Registering it in ``_open_spans`` makes the existing profiler shutdown
    path preserve a still-live external process as an interrupted span.
    """

    __slots__ = (
        "_name",
        "_tracer",
        "_parent_context",
        "_span",
        "_t0",
        "_wall0",
        "_tid",
        "_metadata",
        "_interrupted",
        "_ended",
        "_cancelled",
        "_lock",
        "_data_span_name",
        "_profile_session_id",
    )

    def __init__(
        self,
        tracer,
        name: str,
        *,
        start_perf_ns: int,
        start_wall_ns: int,
        metadata: "dict | None",
        parent_context,
        data_span_name: "str | None" = None,
    ) -> None:
        if parent_context is None:
            from opentelemetry.context import Context

            parent_context = Context()
        self._name = name
        self._tracer = tracer
        self._parent_context = parent_context
        self._t0 = start_perf_ns
        self._wall0 = start_wall_ns
        self._tid = _DERIVED_TID
        self._metadata = dict(metadata or {})
        self._interrupted = False
        self._ended = False
        self._cancelled = False
        self._lock = threading.Lock()
        self._data_span_name = data_span_name
        self._profile_session_id = _profile_session_id
        self._span = None
        with _open_spans_lock:
            _open_spans[id(self)] = self

    def context(self):
        """Durable parent identity, available before the interval completes."""
        from opentelemetry.trace import set_span_in_context

        with self._lock:
            if self._span is None:
                self._span = self._tracer.start_span(
                    self._name, context=self._parent_context, start_time=self._wall0
                )
            return set_span_in_context(self._span)

    def update(self, name: "str | None" = None, **metadata) -> None:
        with self._lock:
            if self._ended or self._interrupted or self._cancelled:
                return
            if name is not None and name != self._name:
                self._name = name
                if self._span is not None:
                    self._span.update_name(name)
            self._metadata.update(metadata)

    def end(
        self,
        *,
        end_perf_ns: int,
        end_wall_ns: int,
        metadata: "dict | None" = None,
        start_perf_ns: "int | None" = None,
        start_wall_ns: "int | None" = None,
    ) -> bool:
        if (start_perf_ns is None) != (start_wall_ns is None):
            raise ValueError("start_perf_ns and start_wall_ns must be overridden together")
        if self._span is not None and start_perf_ns is not None:
            raise ValueError("a span used as a parent cannot be retimed")
        with _open_spans_lock:
            if _open_spans.pop(id(self), None) is None:
                return False
        with self._lock:
            if self._ended or self._interrupted or self._cancelled:
                return False
            self._ended = True
            self._metadata.update(metadata or {})
            completed_metadata = dict(self._metadata)
            completed_metadata.setdefault("outcome", "success")
            effective_start_perf_ns = min(
                end_perf_ns,
                self._t0 if start_perf_ns is None else start_perf_ns,
            )
            effective_start_wall_ns = min(
                end_wall_ns,
                self._wall0 if start_wall_ns is None else start_wall_ns,
            )
            self._t0 = effective_start_perf_ns
            self._wall0 = effective_start_wall_ns
            wall_ns = max(0, end_perf_ns - effective_start_perf_ns)
            measurements = {
                "agency.thread_id": self._tid,
                "agency.perf_start_ns": effective_start_perf_ns,
                "agency.wall_ns": wall_ns,
            }
            span = self._span or self._tracer.start_span(
                self._name, context=self._parent_context, start_time=effective_start_wall_ns
            )
            self._span = span
            for key, value in {**completed_metadata, **measurements}.items():
                span.set_attribute(key, _otel_attribute(value))
            span_context = span.get_span_context()
            parent = span.parent
            attributes = {**completed_metadata, **measurements}
            span_id = span_context.span_id if span_context is not None else None
            parent_span_id = parent.span_id if parent is not None else None
            span.end(end_time=end_wall_ns)
            return _record_data_logger_span(
                self._data_span_name or self._name,
                effective_start_wall_ns,
                end_wall_ns,
                attributes,
                span_id=span_id,
                parent_span_id=parent_span_id,
                profile_session_id=self._profile_session_id,
                profile_span_name=self._name,
            )

    def interrupt(self, ended_perf_ns: int) -> None:
        with self._lock:
            if self._ended or self._interrupted or self._cancelled:
                return
            self._interrupted = True
            wall_ns = max(0, ended_perf_ns - self._t0)
            attributes = {
                **self._metadata,
                "outcome": "interrupted",
                "agency.incomplete": True,
                "agency.thread_id": self._tid,
                "agency.perf_start_ns": self._t0,
                "agency.wall_ns": wall_ns,
            }
            span = self._span or self._tracer.start_span(
                self._name, context=self._parent_context, start_time=self._wall0
            )
            self._span = span
            for key, value in attributes.items():
                span.set_attribute(key, _otel_attribute(value))
            span.end(end_time=self._wall0 + wall_ns)


class _OTelSession:
    """Own a private always-on OTel provider for one agprof session."""

    __slots__ = ("provider", "tracer")

    def __init__(self) -> None:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.sampling import ALWAYS_ON

        self.provider = TracerProvider(sampler=ALWAYS_ON)
        self.tracer = self.provider.get_tracer("agency.profiler")

    def span(self, name: str, parent_context=None) -> _TimedSpan:
        return _TimedSpan(self.tracer, name, parent_context=parent_context)

    def stop(self) -> None:
        self.provider.force_flush()
        self.provider.shutdown()


def enabled() -> bool:
    """True while a profiling session is active."""
    return _session is not None


def span(name: str, *, parent_context=None):
    """A timed, named interval in the current execution context.

    No-op (a shared ``nullcontext``) unless a session is active. Nesting in the
    same synchronous or async context produces parent/child spans in the trace.
    While active, each span also records its thread's CPU and run-queue time
    (see module doc).
    """
    s = _session
    if s is None:
        return _NULL
    return s.span(name, parent_context=parent_context)


def register_engine(engine: str) -> None:
    if not enabled():
        return
    _engine_coverage[str(engine)] = {
        "llm": "host_observed",
        "tools": "admission_completion_boundaries",
        "turns": "container_asserted" if engine == "native" else "unavailable",
        "retries": "unavailable",
        "automatic_functions": "host_only",
        "reason": "External harness internals require cooperative events; no transcript timing inference.",
    }


def telemetry_error(kind: str, count: int = 1) -> None:
    if enabled():
        _health[kind] += count


def ingest_auto_samples(
    pid: int, samples: "list[dict]", *, thread_label: str = "Remote thread"
) -> int:
    """Accept a batch of already-measured function-call samples from an
    external reporter (a harness profiling its own call stack, outside this
    process). Returns the number rejected. A no-op when profiling is off or
    automatic-function sampling was not requested for this session."""
    if not enabled() or _auto_settings is None:
        return 0
    now = time.perf_counter_ns()
    rejected = 0
    for sample in samples:
        try:
            name = sample["name"]
            if not isinstance(name, str) or not name or len(name) > 512:
                raise ValueError("invalid name")
            started = int(sample["perf_ns"])
            if started < _session_started_ns or started > now + 1_000_000_000:
                raise ValueError("timestamp outside profile session")
            duration = int(sample["duration_ns"])
            if duration < 0 or started + duration > now + 1_000_000_000:
                raise ValueError("invalid interval")
            if len(_auto_records) >= _auto_settings["max_events"]:
                telemetry_error("remote_auto_dropped")
                continue
            _auto_records.append(
                (
                    pid,
                    int(sample["tid"]),
                    name,
                    str(sample.get("filename", ""))[:1024],
                    int(sample.get("lineno", 0)),
                    started,
                    duration,
                    str(sample.get("outcome", "unknown"))[:32],
                    thread_label,
                )
            )
        except (KeyError, ValueError, TypeError, OverflowError):
            rejected += 1
            telemetry_error("remote_events_rejected")
    return rejected


def current_span_context():
    """Return a durable OTel context containing the active profiler span.

    The returned context can be stored by a host-side correlation registry
    and later supplied to :func:`span` from an unrelated request thread.  It
    is ``None`` when profiling is off or the current execution context has no
    active profiler span.
    """
    if _session is None or not _span_stack.get():
        return None
    from opentelemetry import trace

    active_span = _span_stack.get()[-1]._span
    return trace.set_span_in_context(active_span)


def current_span_attributes() -> dict:
    """Copy metadata attached to the active profiler span."""
    if _session is None or not _span_stack.get():
        return {}
    return dict(_span_stack.get()[-1]._metadata)


def spawn_traced(fn, *args, daemon: bool = True, **kwargs) -> threading.Thread:
    """Create a thread that inherits the caller's active OTel context.

    ``contextvars`` intentionally start empty in a new ``threading.Thread``.
    Capturing only OTel's context here preserves trace parentage without
    copying agprof's annotation stack into a thread that does not own those
    span handles.  The optional OTel dependency is never imported while
    profiling is disabled.
    """
    if _session is None:
        return threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=daemon)

    from opentelemetry import context as otel_context

    context = otel_context.get_current()

    def run() -> None:
        token = otel_context.attach(context)
        try:
            fn(*args, **kwargs)
        finally:
            otel_context.detach(token)

    return threading.Thread(target=run, daemon=daemon)


def annotate(**metadata) -> None:
    """Attach fields to the innermost active span in this execution context.

    This is a no-op when profiling is disabled or the current context has no
    open span. Context-local ownership prevents concurrent asyncio tasks on one
    thread from cross-annotating. Framework call sites use it for outcomes,
    token counts, TTFT, and other per-invocation metrics without adding work to
    the off path.
    """
    if _session is None:
        return
    stack = _span_stack.get()
    if stack:
        stack[-1].annotate(**metadata)


def start_external_span(
    name: str,
    *,
    start_perf_ns: int,
    start_wall_ns: int,
    metadata: "dict | None" = None,
    parent_context=None,
    data_span_name: "str | None" = None,
) -> "_ObservedSpan | None":
    """Open a host-owned span whose interval is ended by a later callback.

    It is used for ptrace process lifecycles and remotely reported starts where
    shutdown may happen before a matching end arrives. Open handles participate
    in agprof's normal interruption accounting. The caller may safely call
    ``handle.update(...)`` and ``handle.end(...)`` from unrelated threads.
    """
    # Registration and the session-state check are one transaction with
    # stop(): either this handle reaches _open_spans before stop owns the
    # state lock (and is drained as interrupted), or it observes the stopped
    # session and returns None. Without this lock, stop could drain the map
    # between the unsynchronised _session read and _ObservedSpan.__init__.
    # Lock order is always state -> open; no open-span path acquires state.
    with _state_lock:
        s = _session
        if s is None:
            return None
        return _ObservedSpan(
            s.tracer,
            name,
            start_perf_ns=start_perf_ns,
            start_wall_ns=start_wall_ns,
            metadata=metadata,
            parent_context=parent_context,
            data_span_name=data_span_name,
        )


def _append_interrupted_span(open_span, ended_perf_ns: int) -> None:
    if getattr(open_span, "_interrupted", False) or getattr(open_span, "_ended", False):
        return
    open_span.interrupt(ended_perf_ns)
    span = getattr(open_span, "_span", None)
    identity = {}
    if span is not None and hasattr(span, "get_span_context"):
        identity["span_id"] = f"{span.get_span_context().span_id:016x}"
        if span.parent is not None:
            identity["parent_span_id"] = f"{span.parent.span_id:016x}"
    _interrupted_spans.append(
        {
            "thread_id": open_span._tid,
            "label": open_span._name,
            "started_ns": open_span._t0,
            "duration_ms": round(max(0, ended_perf_ns - open_span._t0) / 1e6, 3),
            **copy.deepcopy(open_span._metadata),
            **identity,
            "outcome": "interrupted",
        }
    )


def interrupt_external_span(
    external_span: "_ObservedSpan | None", *, ended_perf_ns: "int | None" = None
) -> None:
    """Finalize an open external span as incomplete, once, if still live."""
    if external_span is None:
        return
    with _open_spans_lock:
        if _open_spans.pop(id(external_span), None) is None:
            return
    _append_interrupted_span(
        external_span,
        time.perf_counter_ns() if ended_perf_ns is None else ended_perf_ns,
    )


def cancel_external_span(external_span: "_ObservedSpan | None") -> None:
    """Silently discard a live external interval without emitting a span.

    This is for correlation reconcilers that learn the same interval will be
    represented by an authoritative fallback and must avoid double counting.
    Cancellation wins only while the handle is still registered; completion
    or profiler shutdown remains authoritative if it removed the handle first.
    """
    if external_span is None:
        return
    with _open_spans_lock:
        if _open_spans.pop(id(external_span), None) is None:
            return
        with external_span._lock:
            if external_span._ended or external_span._interrupted:
                return
            external_span._ended = True
            external_span._cancelled = True
            if external_span._span is not None:
                external_span._span.end(end_time=time.time_ns())


def _resolve_auto_roots(include) -> list[tuple[str, str]]:
    """Resolve include specs to (absolute directory, import-prefix) pairs."""
    if include is None:
        specs = [str(Path.cwd()), "agency"]
    elif isinstance(include, (str, os.PathLike)):
        specs = [str(include)]
    else:
        specs = [str(item) for item in include]
    roots: list[tuple[str, str]] = []
    for spec in specs:
        candidate = Path(spec).expanduser()
        if candidate.exists() or os.sep in spec:
            path = candidate.resolve()
            roots.append((str(path.parent if path.is_file() else path), ""))
            continue
        try:
            module_spec = importlib.util.find_spec(spec)
        except (ImportError, AttributeError, ValueError):
            module_spec = None
        if module_spec is None:
            continue
        locations = list(module_spec.submodule_search_locations or ())
        if locations:
            roots.extend((str(Path(location).resolve()), spec) for location in locations)
        elif module_spec.origin:
            roots.append((str(Path(module_spec.origin).resolve().parent), spec.rpartition(".")[0]))
    return sorted(set(roots), key=lambda item: len(item[0]), reverse=True)


def _make_auto_settings(
    *,
    include=None,
    exclude=None,
    min_duration_ms: float = _DEFAULT_AUTO_MIN_DURATION_MS,
    max_depth: int = _DEFAULT_AUTO_MAX_DEPTH,
    max_events: int = _DEFAULT_AUTO_MAX_EVENTS,
    include_dependencies: bool = False,
) -> dict:
    if min_duration_ms < 0:
        raise ValueError("agprof: auto_min_duration_ms must be >= 0")
    if max_depth < 1:
        raise ValueError("agprof: auto_max_depth must be >= 1")
    if max_events < 1:
        raise ValueError("agprof: auto_max_events must be >= 1")
    excluded = [] if exclude is None else ([exclude] if isinstance(exclude, str) else list(exclude))
    roots = _resolve_auto_roots(include)
    if include_dependencies:
        roots.extend(
            (path, "")
            for path in {sysconfig.get_path("purelib"), sysconfig.get_path("stdlib")}
            if path
        )
    if not roots:
        raise ValueError("agprof: auto_include did not resolve to any Python source roots")
    return {
        "roots": roots,
        "include_dependencies": include_dependencies,
        "exclude": tuple(str(item) for item in excluded),
        "min_duration_ns": int(min_duration_ms * 1e6),
        "min_duration_ms": float(min_duration_ms),
        "max_depth": int(max_depth),
        "max_events": int(max_events),
    }


def _auto_label_for_code(code) -> "tuple[str, str, int] | None":
    cached = _auto_code_labels.get(code, ...)
    if cached is not ...:
        return cached
    settings = _auto_settings
    if settings is None:
        return None
    filename = code.co_filename
    if not filename or filename.startswith("<"):
        _auto_code_labels[code] = None
        return None
    absolute = os.path.abspath(filename)
    normalized = absolute.replace(os.sep, "/")
    if (
        (
            not settings["include_dependencies"]
            and any(
                part in normalized
                for part in ("/site-packages/", "/dist-packages/", "/.venv/", "/venv/")
            )
        )
        or absolute.startswith(os.path.dirname(__file__) + os.sep)
        or any(
            pattern and (pattern in normalized or pattern.replace(".", "/") in normalized)
            for pattern in settings["exclude"]
        )
    ):
        _auto_code_labels[code] = None
        return None
    label = None
    for root, prefix in settings["roots"]:
        relative = os.path.relpath(absolute, root)
        if relative == os.pardir or relative.startswith(os.pardir + os.sep):
            continue
        module = os.path.splitext(relative)[0].replace(os.sep, ".")
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        if prefix and not (module == prefix or module.startswith(prefix + ".")):
            module = f"{prefix}.{module}" if module else prefix
        label = f"{module}.{code.co_qualname}".strip(".")
        break
    if label and any(
        label == pattern or label.startswith(pattern + ".")
        for pattern in settings["exclude"]
        if pattern
    ):
        label = None
    result = (label, absolute, code.co_firstlineno) if label else None
    if result is not None and _auto_tool_in_use:
        monitoring = sys.monitoring
        monitoring.set_local_events(
            monitoring.PROFILER_ID, code, monitoring.events.PY_RETURN | monitoring.events.PY_YIELD
        )
    _auto_code_labels[code] = result
    return result


def _remember_thread_label(
    name: str,
    *,
    priority: int = 20,
    pid: "int | None" = None,
    tid: "int | None" = None,
) -> None:
    label = " ".join(str(name).split()).strip()
    if not label:
        return
    key = (os.getpid() if pid is None else pid, threading.get_native_id() if tid is None else tid)
    previous = _thread_labels.get(key)
    if previous is None or priority >= previous[0]:
        _thread_labels[key] = (priority, label)


def _runtime_thread_label() -> str:
    current = threading.current_thread()
    name = current.name
    if current is threading.main_thread() or name == "MainThread":
        return "Agency main thread" if os.getpid() == _profile_root_pid else "Main thread"
    if name == "agprof-sampler":
        return "agprof sampler"
    match = re.fullmatch(r"ThreadPoolExecutor-(\d+)_(\d+)", name)
    if match:
        return f"Thread pool {match.group(1)} worker {match.group(2)}"
    if re.fullmatch(r"Thread-\d+(?: \(.+\))?", name):
        return "Python worker"
    return name or "Python thread"


def _span_thread_label(label: str) -> "str | None":
    match = re.fullmatch(r"run\d+:([^:]+):(.+)", label)
    if match:
        return f"{match.group(1)} — {match.group(2)}"
    return label if label.startswith("agmap:") else None


def _infer_auto_thread_label(records: list[tuple]) -> "str | None":
    if not records:
        return None
    labels = [record[2] for record in records]
    for label in labels:
        match = re.search(r"(?:^|\.)([A-Za-z_]\w*Team)\.(?:run|_run)(?:$|\.)", label)
        if match:
            return f"Agency team — {match.group(1)}"
    rules = (
        ("agsandbox_backends.base.run_with_unkillable_child_grace", "Sandbox subprocess waiter"),
        ("agutil._iter_batched.<locals>._drain", "LLM stream drainer"),
        ("agteam._wrap_run", "Agency team runner"),
        ("agmap._spawn", "Agency map worker"),
    )
    for needle, name in rules:
        if any(needle in label for label in labels):
            return name
    if any(label.startswith("agskill.") and "_traced_task" in label for label in labels):
        return "Agency skill runner"
    dominant = max(records, key=lambda record: record[6])
    if dominant[6] < _DYNAMIC_THREAD_MIN_DURATION_NS:
        return None
    label, filename = dominant[2], dominant[3]
    module = Path(filename).stem
    marker_index = label.find(f"{module}.")
    callable_name = label[marker_index:] if marker_index >= 0 else label
    callable_name = callable_name.replace(".<locals>.", ".")
    agency_dir = str(Path(__file__).resolve().parents[1]) + os.sep
    prefix = "Agency" if os.path.abspath(filename).startswith(agency_dir) else "Python"
    return f"{prefix} — {callable_name}"


def _thread_label(pid: int, tid: int) -> str:
    remembered = _thread_labels.get((pid, tid))
    return remembered[1] if remembered is not None else "Python thread"


def _record_auto_call(entry: tuple, ended_ns: int, outcome: str, *, identity=None) -> None:
    global _auto_dropped
    _code, started_ns, label, filename, lineno = entry
    if started_ns is None:
        _auto_filtered["depth"] += 1
        return
    duration_ns = max(0, ended_ns - started_ns)
    settings = _auto_settings
    if settings is None:
        return
    if duration_ns < settings["min_duration_ns"]:
        _auto_filtered["duration"] += 1
        return
    if len(_auto_records) >= settings["max_events"]:
        _auto_dropped += 1
        return
    pid, tid = identity or (os.getpid(), threading.get_native_id())
    _auto_records.append(
        (
            pid,
            tid,
            label,
            filename,
            lineno,
            started_ns,
            duration_ns,
            outcome,
            _thread_label(pid, tid),
        )
    )


def _auto_py_start(code, _instruction_offset) -> None:
    resolved = _auto_label_for_code(code)
    if resolved is None:
        return
    _remember_thread_label(_runtime_thread_label())
    key = (os.getpid(), threading.get_native_id())
    stack = _auto_stacks.setdefault(key, [])
    settings = _auto_settings
    if settings is None:
        return
    started_ns = time.perf_counter_ns() if len(stack) < settings["max_depth"] else None
    stack.append((code, started_ns, *resolved))


def _auto_py_end(code, _instruction_offset, _value, *, outcome: str) -> None:
    key = (os.getpid(), threading.get_native_id())
    stack = _auto_stacks.get(key)
    if not stack:
        return
    match = next((i for i in range(len(stack) - 1, -1, -1) if stack[i][0] is code), -1)
    if match < 0:
        return
    ended_ns = time.perf_counter_ns()
    entry, younger = stack[match], stack[match + 1 :]
    del stack[match:]
    for abandoned in younger:
        _record_auto_call(abandoned, ended_ns, "interrupted")
    _record_auto_call(entry, ended_ns, outcome)
    if not stack:
        _auto_stacks.pop(key, None)


def _auto_py_return(code, offset, value) -> None:
    _auto_py_end(code, offset, value, outcome="return")


def _auto_py_unwind(code, offset, exception) -> None:
    _auto_py_end(code, offset, exception, outcome="exception")


def _auto_py_resume(code, offset) -> None:
    _auto_py_start(code, offset)


def _auto_py_yield(code, offset, value) -> None:
    _auto_py_end(code, offset, value, outcome="yield")


def _enable_auto_functions(settings: dict) -> None:
    global _auto_settings, _auto_tool_in_use
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is None:
        raise RuntimeError("agprof: automatic function tracing requires Python 3.12 or newer")
    tool_id = monitoring.PROFILER_ID
    try:
        monitoring.use_tool_id(tool_id, "agency.agprof")
    except ValueError as e:
        raise RuntimeError(
            f"agprof: sys.monitoring profiler tool id is already in use by {monitoring.get_tool(tool_id)!r}"
        ) from e
    _auto_settings = copy.deepcopy(settings)
    _auto_code_labels.clear()
    _auto_stacks.clear()
    monitoring.register_callback(tool_id, monitoring.events.PY_START, _auto_py_start)
    monitoring.register_callback(tool_id, monitoring.events.PY_RESUME, _auto_py_resume)
    monitoring.register_callback(tool_id, monitoring.events.PY_RETURN, _auto_py_return)
    monitoring.register_callback(tool_id, monitoring.events.PY_YIELD, _auto_py_yield)
    monitoring.register_callback(tool_id, monitoring.events.PY_UNWIND, _auto_py_unwind)
    monitoring.set_events(
        tool_id,
        monitoring.events.PY_START | monitoring.events.PY_RESUME | monitoring.events.PY_UNWIND,
    )
    _auto_tool_in_use = True


def _disable_auto_functions() -> list[tuple]:
    global _auto_settings, _auto_tool_in_use
    monitoring = getattr(sys, "monitoring", None)
    if _auto_tool_in_use and monitoring is not None:
        tool_id = monitoring.PROFILER_ID
        monitoring.set_events(tool_id, 0)
        for code, label in _auto_code_labels.items():
            if label is not None:
                monitoring.set_local_events(tool_id, code, 0)
        for event in (
            monitoring.events.PY_START,
            monitoring.events.PY_RETURN,
            monitoring.events.PY_RESUME,
            monitoring.events.PY_UNWIND,
            monitoring.events.PY_YIELD,
        ):
            monitoring.register_callback(tool_id, event, None)
        monitoring.free_tool_id(tool_id)
        _auto_tool_in_use = False
    ended_ns = time.perf_counter_ns()
    for identity, stack in list(_auto_stacks.items()):
        for entry in stack:
            _record_auto_call(entry, ended_ns, "interrupted", identity=identity)
    _auto_stacks.clear()
    events = list(_auto_records)
    _auto_settings = None
    _auto_code_labels.clear()
    return events


def _unpack_record(record) -> tuple:
    """Return the six clock fields plus metadata from old/new record tuples."""
    metadata = record[6] if len(record) > 6 else {}
    return (*record[:6], metadata)


_PR_SET_NAME = 15  # linux prctl option
_libc = None


def thread_name(name: str) -> None:
    """Set the OS-level thread name for process/resource diagnostics.

    Linux truncates to 15 chars. No-op when profiling is off or prctl is
    unavailable; the full OTel span name is unaffected.
    """
    if _session is None:
        return
    semantic_name = _span_thread_label(name)
    _remember_thread_label(semantic_name or name, priority=100)
    global _libc
    with suppress(Exception):
        if _libc is None:
            import ctypes

            _libc = ctypes.CDLL(None, use_errno=True)
        _libc.prctl(_PR_SET_NAME, name[:15].encode(), 0, 0, 0)


def _os_thread_name() -> str:
    """This thread's OS-level name (what thread_name() set), or a tid tag."""
    remembered = _thread_labels.get((os.getpid(), threading.get_native_id()))
    if remembered is not None:
        return remembered[1]
    try:
        with open(f"/proc/self/task/{threading.get_native_id()}/comm", "rb") as f:
            return f.read().decode().strip()
    except Exception:
        return f"tid{threading.get_native_id()}"


def gpu_lease_begin(gpu_id: int) -> None:
    """Record the start of an exclusive GPU lease (called by agresources).

    Lease intervals are device-scope events joined to the trace at session
    stop as a synthetic per-device lane — they are NOT thread spans, because a
    lease can outlive the acquiring call (background processes hold the GPU
    across turns). No-op when profiling is off.
    """
    if _session is None:
        return
    with _leases_lock:
        _leases_open[gpu_id] = (time.perf_counter_ns(), _os_thread_name())


def gpu_lease_end(gpu_id: int) -> None:
    """Record the end of a GPU lease. Safe to call with no matching begin."""
    with _leases_lock:
        entry = _leases_open.pop(gpu_id, None)
        if entry is not None:
            _leases.append((gpu_id, entry[0], time.perf_counter_ns(), entry[1]))


def _is_python_executable(value: str) -> bool:
    return bool(re.fullmatch(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?", Path(value).name))


def _python_entrypoint(argv: list[str]) -> str:
    args = argv[1:]
    for index, arg in enumerate(args):
        if arg == "-m" and index + 1 < len(args):
            return f"python -m {args[index + 1]}"
        if arg == "-c":
            return "python -c"
        if arg == "-":
            return "python stdin"
        if not arg.startswith("-"):
            return Path(arg).name or "Python"
    return "Python"


def _command_role(comm: str, argv: list[str]) -> str:
    if not argv:
        return comm
    if _is_python_executable(argv[0]) or _is_python_executable(comm):
        return _python_entrypoint(argv)
    program = Path(argv[0]).name or comm
    if program in {"bash", "docker", "git", "podman", "sh", "sudo", "systemd-run"}:
        subcommand = next((arg for arg in argv[1:] if arg and not arg.startswith("-")), None)
        if subcommand is not None and len(subcommand) <= 40:
            return f"{program} {Path(subcommand).name}"
    return program


def _process_display_name(pid: int, comm: str, argv: list[str], sandbox: "str | None") -> str:
    """Semantic Perfetto process label with the PID retained for uniqueness."""
    command = _command_role(comm, argv)
    if sandbox is not None:
        role = f"sandbox:{sandbox} — {command}"
    elif pid == _profile_root_pid:
        role = f"Agency harness — {command}"
    elif argv and (_is_python_executable(argv[0]) or _is_python_executable(comm)):
        role = command if command == "Python" else f"Python — {command}"
    else:
        role = command
    return f"{role} (PID {pid})"


class _Sampler(threading.Thread):
    """Background poller for per-PID, workload-cgroup, and GPU metrics."""

    def __init__(self, hz: float, sample_gpu: bool) -> None:
        super().__init__(daemon=True, name="agprof-sampler")
        self._process_cgroup = _process_cgroup_dir()
        self._interval = 1.0 / hz
        self._stop_ev = threading.Event()
        self._nvml = None
        self._handles: list = []
        if sample_gpu:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = pynvml
                self._handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())
                ]
            except Exception:
                _health["gpu_initialization_failures"] += 1
                self._nvml = None
        self._proc_root = Path("/proc")
        self._clock_ticks = os.sysconf("SC_CLK_TCK")
        self._page_mb = os.sysconf("SC_PAGE_SIZE") / 2**20
        # Containers are sampled from the registry the sandbox backends fill
        # at container start (see container_started()) — no filesystem
        # discovery, no runtime naming assumptions.
        self._pid_labels: "dict[int, str]" = {}  # pid -> label (container or comm)
        self._proc_util_last: "dict[int, int]" = {}  # gpu idx -> last NVML sample ts

    def _workload_pids(self) -> "dict[int, str]":
        """Return every PID in the workload cgroup tree and its cgroup path.

        Reading every ``cgroup.procs`` file is the cgroup-v2 source of truth.
        A process may exit or move immediately afterward; callers therefore
        treat every subsequent /proc read as best-effort.
        """
        pids: "dict[int, str]" = {}
        try:
            for root, _dirs, files in os.walk(self._process_cgroup):
                if "cgroup.procs" not in files:
                    continue
                try:
                    with open(Path(root) / "cgroup.procs", "rb") as f:
                        for raw_pid in f.read().split():
                            pids[int(raw_pid)] = root
                except (OSError, ValueError):
                    continue
        except OSError:
            return {}
        return pids

    def _process_scope(self, cgroup_dir: str) -> "str | None":
        """Registered sandbox label owning *cgroup_dir*, if any."""
        with _cg_lock:
            entries = list(_cg_registry.items())
        for label, registered in entries:
            if cgroup_dir == registered or cgroup_dir.startswith(registered + "/"):
                return label
        return None

    def _sample_process(self, t: int, pid: int, cgroup_dir: str) -> None:
        """Sample one current cgroup member directly from /proc/<pid>."""
        try:
            process_dir = self._proc_root / str(pid)
            stat = (process_dir / "stat").read_bytes()
            close = stat.rfind(b") ")
            if close < 0:
                return
            comm = stat[stat.find(b"(") + 1 : close].decode(errors="replace")
            fields = stat[close + 2 :].split()
            cpu_s = (int(fields[11]) + int(fields[12])) / self._clock_ticks
            start_ticks = int(fields[19])
            vms_mb = int(fields[20]) / 2**20
            rss_mb = int(fields[21]) * self._page_mb
        except (OSError, ValueError, IndexError):
            return

        identity = f"{pid}-{start_ticks}"
        info = _process_info.get(identity)
        if info is None:
            sandbox = self._process_scope(cgroup_dir)
            try:
                raw_argv = (process_dir / "cmdline").read_bytes().split(b"\0")
                argv = [arg.decode(errors="replace") for arg in raw_argv if arg]
            except OSError:
                argv = []
            cmdline = " ".join(argv)
            display_name = _process_display_name(pid, comm, argv, sandbox)
            prior_pid_identity = next(
                (key for key, prior in _process_info.items() if prior["pid"] == pid),
                None,
            )
            trace_pid = pid if prior_pid_identity is None else 1_000_000_000 + len(_process_info)
            info = {
                "identity": identity,
                "pid": pid,
                "start_ticks": start_ticks,
                "comm": comm,
                "cmdline": cmdline,
                "argv": argv,
                "sandbox": sandbox,
                "cgroup": cgroup_dir,
                "display_name": display_name,
                "trace_pid": trace_pid,
                "first_seen_ns": t,
                "last_seen_ns": t,
            }
            _process_info[identity] = info
        else:
            info["last_seen_ns"] = t
            if info["sandbox"] is None:
                sandbox = self._process_scope(cgroup_dir)
                if sandbox is not None:
                    info["sandbox"] = sandbox
                    info["display_name"] = _process_display_name(
                        pid, comm, info.get("argv", []), sandbox
                    )

        prefix = f"proc:{identity}:"
        _samples.append((t, prefix + "cpu_s", cpu_s))
        _samples.append((t, prefix + "rss_mb", rss_mb))
        _samples.append((t, prefix + "vms_mb", vms_mb))
        try:
            io_fields = {}
            with (process_dir / "io").open("rb") as f:
                for line in f:
                    key, value = line.split(b":", 1)
                    io_fields[key] = int(value)
            _samples.append((t, prefix + "io_r", float(io_fields[b"read_bytes"])))
            _samples.append((t, prefix + "io_w", float(io_fields[b"write_bytes"])))
        except (OSError, ValueError, KeyError):
            # Linux may deny /proc/<pid>/io for a differently-owned container
            # process. CPU/RSS/VMS remain valid and the sandbox cgroup retains
            # its exact aggregate I/O counters.
            pass

    def _tick_processes(self, t: int) -> None:
        """Refresh workload membership and sample every PID still alive."""
        for pid, cgroup_dir in self._workload_pids().items():
            self._sample_process(t, pid, cgroup_dir)

    def _pid_label(self, pid: int) -> str:
        """Label for a GPU-using PID: the owning container's agname when the
        PID's cgroup falls under a registered container dir (same registry the
        CPU collector samples — runtime-agnostic), else the process comm name
        (e.g. vLLM's server process)."""
        label = self._pid_labels.get(pid)
        if label is not None:
            return label
        label = f"pid{pid}"
        cacheable = True
        try:
            cg = Path(f"/proc/{pid}/cgroup").read_text()
            path = next((l.split("::", 1)[1] for l in cg.splitlines() if l.startswith("0::")), "")
            full = "/sys/fs/cgroup" + path
            with _cg_lock:
                entries = list(_cg_registry.items())
            for reg_label, reg_dir in entries:
                if full == reg_dir or full.startswith(reg_dir + "/"):
                    label = reg_label
                    break
            else:
                label = Path(f"/proc/{pid}/comm").read_text().strip() or label
        except Exception:
            cacheable = False
        # comm names may contain ':' (e.g. "VLLM::EngineCor") — keep series
        # names parseable as gpu{i}:{label}:{metric}.
        label = label.replace(":", "_")
        if cacheable:
            self._pid_labels[pid] = label
        return label

    def _tick_gpu_procs(
        self, t: int, i: int, h, dev_util: float = 0.0, dev_power_w: float = 0.0
    ) -> None:
        """Per-PID GPU accounting for device *i*: VRAM per process (compute
        procs list) and SM utilization per process (NVML's sample buffer since
        the previous tick). This is what splits a device-wide curve into
        per-agent/per-process series under concurrency — cgroups can't meter
        GPUs, so this is the container-attribution path for the device.

        Power has NO per-process accounting anywhere (board sensors measure
        the whole card), so ``power_w_est`` is the device draw apportioned by
        utilization share — an APPORTIONED ESTIMATE, never a measurement,
        hence the ``_est`` suffix."""
        with suppress(Exception):
            for pr in self._nvml.nvmlDeviceGetComputeRunningProcesses(h):
                if pr.usedGpuMemory:
                    _samples.append(
                        (t, f"gpu{i}:{self._pid_label(pr.pid)}:mem_mb", pr.usedGpuMemory / 2**20)
                    )
        with suppress(Exception):
            last = self._proc_util_last.get(i, 0)
            util_samples = self._nvml.nvmlDeviceGetProcessUtilization(h, last)
            per_pid: "dict[int, list[int]]" = {}
            newest = last
            for s in util_samples:
                if s.timeStamp > last:  # NVML may replay old samples
                    per_pid.setdefault(s.pid, []).append(s.smUtil)
                    newest = max(newest, s.timeStamp)
            self._proc_util_last[i] = newest
            for pid, utils in per_pid.items():
                label = self._pid_label(pid)
                pid_util = sum(utils) / len(utils)
                _samples.append((t, f"gpu{i}:{label}:util_pct", pid_util))
                if dev_power_w > 0 and pid_util > 0:
                    share = min(1.0, pid_util / max(dev_util, 1.0))
                    _samples.append((t, f"gpu{i}:{label}:power_w_est", dev_power_w * share))

    def _tick_cgroups(self, t: int) -> None:
        """Sample every REGISTERED container (see container_started()) plus the
        registered runtime-daemon cgroups (conmon scopes / docker services),
        aggregated per kind. No discovery: the registry is the whole truth."""
        with _cg_lock:
            containers = list(_cg_registry.items())
            daemons = list(_daemon_cg.items())
        for label, cdir in containers:
            self._sample_container(t, label, cdir)
        agg: "dict[str, int]" = {}
        for ddir, kind in daemons:
            with suppress(Exception):  # daemon scope may disappear when its container stops
                with open(f"{ddir}/cpu.stat", "rb") as f:
                    agg[kind] = agg.get(kind, 0) + int(f.readline().split()[1])
        for kind, cpu_us in agg.items():
            _samples.append((t, f"cg:{kind}:cpu_us", float(cpu_us)))

    def _sample_container(self, t: int, label: str, cdir: str) -> None:
        try:
            with open(f"{cdir}/cpu.stat", "rb") as f:
                _samples.append((t, f"cg:{label}:cpu_us", float(int(f.readline().split()[1]))))
            with open(f"{cdir}/memory.current", "rb") as f:
                _samples.append((t, f"sandbox:{label}:mem_mb", int(f.read()) / 2**20))
        except Exception:
            return  # cgroup vanished (container stopped/hibernated) — skip all
        # Disk IO, tier 1: the cgroup's own io.stat (exact; present under
        # rootful docker and io-delegated rootless slices).
        io_done = False
        with suppress(Exception):
            rb = wb = 0
            with open(f"{cdir}/io.stat", "rb") as f:
                for line in f:
                    for tok in line.split():
                        if tok.startswith(b"rbytes="):
                            rb += int(tok[7:])
                        elif tok.startswith(b"wbytes="):
                            wb += int(tok[7:])
            _samples.append((t, f"cg:{label}:io_r", float(rb)))
            _samples.append((t, f"cg:{label}:io_w", float(wb)))
            io_done = True
        self._tick_io_net(t, label, cdir, io_from_pids=not io_done)

    def _tick_io_net(self, t: int, label: str, scope_path: str, io_from_pids: bool = True) -> None:
        """Network (always) and disk IO (tier-2 fallback only) for one
        container, via its PIDs.

        Network has no cgroup controller, so it must come from the container's
        netns: any one PID's /proc/<pid>/net/dev (per-container, stable across
        PIDs). Disk IO via summed /proc/<pid>/io is only used when the cgroup
        had no io.stat (io controller not delegated — typical rootless);
        caveats: subuid-owned PIDs unreadable, PIDs exiting between ticks lose
        their bytes, negative deltas dropped at injection.
        """
        # systemd cgroup driver parks the processes in a child cgroup of the
        # scope (scope/container/cgroup.procs); the scope's own procs file is
        # empty. Walk the scope so both layouts work.
        pids: "list[int]" = []
        try:
            for root, _dirs, files in os.walk(scope_path):
                if "cgroup.procs" in files:
                    with open(f"{root}/cgroup.procs", "rb") as f:
                        pids.extend(int(x) for x in f.read().split())
        except Exception:
            return
        if not pids:
            return
        if io_from_pids:
            io_r = io_w = 0
            io_seen = False
            for pid in pids:
                with suppress(Exception):  # subuid-owned or exited PIDs are skipped
                    with open(f"/proc/{pid}/io", "rb") as f:
                        for line in f:
                            if line.startswith(b"read_bytes:"):
                                io_r += int(line.split()[1])
                            elif line.startswith(b"write_bytes:"):
                                io_w += int(line.split()[1])
                    io_seen = True
            if io_seen:
                _samples.append((t, f"cg:{label}:io_r", float(io_r)))
                _samples.append((t, f"cg:{label}:io_w", float(io_w)))
        for pid in pids:
            sampled_network = False
            with suppress(Exception):
                with open(f"/proc/{pid}/net/dev", "rb") as f:
                    rx = tx = 0
                    for line in f.readlines()[2:]:
                        iface, _, rest = line.partition(b":")
                        if iface.strip() == b"lo":
                            continue
                        parts = rest.split()
                        rx += int(parts[0])
                        tx += int(parts[8])
                _samples.append((t, f"cg:{label}:net_rx", float(rx)))
                _samples.append((t, f"cg:{label}:net_tx", float(tx)))
                sampled_network = True
            if sampled_network:
                break  # one PID suffices — netns counters are container-wide

    def _tick_workload_cgroup(self, t: int) -> None:
        """Sample the clearly named aggregate workload cgroup."""
        try:
            cpu_fields = {}
            with (self._process_cgroup / "cpu.stat").open("rb") as f:
                for line in f:
                    key, value = line.split()[:2]
                    cpu_fields[key] = int(value)
            _samples.append((t, "workload:cpu_us", float(cpu_fields[b"usage_usec"])))

            with (self._process_cgroup / "memory.current").open("rb") as f:
                _samples.append((t, "workload:memory_mb", int(f.read()) / 2**20))

            io_r = io_w = 0
            with (self._process_cgroup / "io.stat").open("rb") as f:
                for line in f:
                    for token in line.split():
                        if token.startswith(b"rbytes="):
                            io_r += int(token[7:])
                        elif token.startswith(b"wbytes="):
                            io_w += int(token[7:])
            _samples.append((t, "workload:io_r", float(io_r)))
            _samples.append((t, "workload:io_w", float(io_w)))
        except (OSError, KeyError, ValueError):
            # A configured workload cgroup is validated before sampling. It
            # may disappear only during teardown, when dropping a final tick
            # is preferable to turning a completed benchmark into a failure.
            return

    def _tick(self) -> None:
        t = time.perf_counter_ns()
        self._tick_workload_cgroup(t)
        self._tick_processes(t)
        if self._nvml is not None:
            for i, h in enumerate(self._handles):
                with suppress(Exception):
                    u = self._nvml.nvmlDeviceGetUtilizationRates(h)
                    m = self._nvml.nvmlDeviceGetMemoryInfo(h)
                    p = self._nvml.nvmlDeviceGetPowerUsage(h)
                    _samples.append((t, f"gpu{i}:util_pct", float(u.gpu)))
                    _samples.append((t, f"gpu{i}:mem_mb", m.used / 2**20))
                    _samples.append((t, f"gpu{i}:power_w", p / 1000.0))
                    self._tick_gpu_procs(t, i, h, float(u.gpu), p / 1000.0)
        self._tick_cgroups(t)

    def run(self) -> None:
        _remember_thread_label("agprof sampler", priority=90)
        while not self._stop_ev.wait(self._interval):
            self._tick()
        if self._nvml is not None:
            with suppress(Exception):
                self._nvml.nvmlShutdown()

    def halt(self) -> None:
        self._stop_ev.set()
        if self.ident is not None:
            self.join(timeout=2)
        elif self._nvml is not None:
            # Thread.start() can fail before run() owns NVML shutdown.
            self._nvml.nvmlShutdown()


def next_index(key: str = "run") -> int:
    """Monotonic per-key counter for span labels (run0, run1, ...; agmap[0], ...)."""
    with _counters_lock:
        c = _counters.get(key)
        if c is None:
            c = _counters[key] = itertools.count()
    return next(c)


def start(
    out_dir=None,
    *,
    all_threads: bool = True,
    worker_name: "str | None" = None,
    sample_hz: float = 10.0,
    sample_gpu: bool = True,
    auto_functions: bool = True,
    auto_include_dependencies: bool = False,
    auto_include=None,
    auto_exclude=None,
    auto_min_duration_ms: float = _DEFAULT_AUTO_MIN_DURATION_MS,
    auto_max_depth: int = _DEFAULT_AUTO_MAX_DEPTH,
    auto_max_events: int = _DEFAULT_AUTO_MAX_EVENTS,
):
    """Start a profiling session. Prefer the ``session()`` context manager.

    *out_dir* receives ``agprof.trace.json``, ``summary.json``, and
    ``summary.md``. *sample_hz* and *sample_gpu* control the background gauge
    sampler (0 disables it). Automatic Python function intervals are captured
    with sys.monitoring and filtered before trace emission.
    ``all_threads`` and ``worker_name`` remain accepted for API compatibility.
    """
    _require_linux()
    global _session, _profiler, _out_dir, _last_summary, _last_run_summary
    global _sampler
    global _session_started_ns, _session_sample_hz, _session_sample_gpu
    global _profile_session_id, _profile_data_logger, _last_profile_records
    global _profile_root_pid, _auto_dropped
    auto_settings = (
        _make_auto_settings(
            include=auto_include,
            include_dependencies=auto_include_dependencies,
            exclude=auto_exclude,
            min_duration_ms=auto_min_duration_ms,
            max_depth=auto_max_depth,
            max_events=auto_max_events,
        )
        if auto_functions
        else None
    )
    with _state_lock:
        if _session is not None:
            raise RuntimeError("agprof: a profiling session is already active")
        try:
            otel_session = _OTelSession()
        except ImportError as e:
            # opentelemetry-sdk is a core dependency (see pyproject.toml) --
            # reaching this means the install is incomplete/stale, not that
            # an optional extra was skipped.
            raise RuntimeError(
                "agprof: profiling requires opentelemetry-sdk, a core agency "
                "dependency -- reinstall with `pip install -e .`"
            ) from e
        from ..agdatalogger import agDataLogger
        from ...configs.agconfig import agconfig as _agconfig_cls, dataloggerconfig

        profile_session_id = uuid.uuid4().hex
        profile_db_path = (
            str(Path(out_dir) / "profile_data.sqlite3") if out_dir is not None else ":memory:"
        )
        profile_data_logger = agDataLogger(
            _agconfig_cls(
                dataloggerconfig(
                    db_path=profile_db_path,
                    flush_batch_size=500,
                    flush_interval_s=1.0,
                )
            ),
            default_name="agprof",
            default_object="profiler",
        )
        try:
            profile_data_logger.start()
        except Exception as exc:
            # Profiling must not make the workload fail merely because its
            # optional on-disk span store is unavailable.
            _agprof_print(
                f"[agprof] WARNING: profiler datalogger start failed; using memory only: {exc}"
            )
            profile_data_logger = agDataLogger(
                _agconfig_cls(
                    dataloggerconfig(
                        db_path=":memory:",
                        flush_batch_size=500,
                        flush_interval_s=1.0,
                    )
                ),
                default_name="agprof",
                default_object="profiler",
            )
            profile_data_logger.start()
        _profile_session_id = profile_session_id
        _profile_data_logger = profile_data_logger
        _last_profile_records = []
        _health.clear()
        _engine_coverage.clear()
        _interrupted_spans.clear()
        with _open_spans_lock:
            _open_spans.clear()
        _samples.clear()
        _process_info.clear()
        _thread_labels.clear()
        _profile_root_pid = os.getpid()
        _remember_thread_label("Agency main thread", priority=90)
        _leases.clear()
        _leases_open.clear()
        _auto_records.clear()
        _auto_dropped = 0
        _auto_filtered.clear()
        _last_summary = None
        _last_run_summary = None
        started_ns = time.perf_counter_ns()
        try:
            if auto_settings is not None:
                _enable_auto_functions(auto_settings)
            _out_dir = Path(out_dir) if out_dir is not None else None
            _profiler = otel_session
            _session = otel_session
            _session_started_ns = started_ns
            _session_sample_hz = sample_hz
            _session_sample_gpu = sample_gpu
            if sample_hz > 0:
                _sampler = _Sampler(sample_hz, sample_gpu)
                _sampler.start()
        except BaseException:
            # A failed start must not reserve the process-wide session or
            # retain monitoring callbacks, a provider, or an open database.
            sampler = _sampler
            _session = _profiler = _sampler = None
            _out_dir = _session_started_ns = None
            _profile_session_id = _profile_data_logger = None
            cleanups = [_disable_auto_functions, otel_session.stop, profile_data_logger.stop]
            if sampler is not None:
                cleanups.insert(0, sampler.halt)
            for cleanup in cleanups:
                try:
                    cleanup()
                except Exception as exc:
                    _agprof_print(f"[agprof] WARNING: failed-start cleanup failed: {exc}")
            raise
    return otel_session


def stop():
    """Stop the active session and write its exact JSON/Markdown summaries."""
    global _session, _profiler, _out_dir, _last_summary, _last_run_summary, _sampler
    global _session_started_ns
    global _profile_session_id, _profile_data_logger, _last_profile_records
    with _state_lock:
        if _session is None:
            return None
        prof = _profiler
        out_dir = _out_dir
        sampler = _sampler
        started_ns = _session_started_ns
        profile_session_id = _profile_session_id
        profile_data_logger = _profile_data_logger
        _session = None
        _profiler = None
        _out_dir = None
        _sampler = None
        _session_started_ns = None
    if sampler is not None:
        sampler.halt()
    auto_settings = copy.deepcopy(_auto_settings)
    auto_events = _disable_auto_functions() if _auto_tool_in_use else list(_auto_records)
    t_end = time.perf_counter_ns()
    with _open_spans_lock:
        open_spans = list(_open_spans.values())
        _open_spans.clear()
    for open_span in open_spans:
        _append_interrupted_span(open_span, t_end)
    # Close any lease still open at session end so it renders to the stop edge.
    with _leases_lock:
        for gpu_id, (t0, label) in _leases_open.items():
            _leases.append((gpu_id, t0, t_end, label))
        _leases_open.clear()
    prof.stop()
    records = _load_profile_records(profile_session_id)
    _last_profile_records = list(records)
    if profile_data_logger is not None:
        try:
            profile_data_logger.stop()
        except Exception as exc:
            _agprof_print(f"[agprof] WARNING: profiler datalogger shutdown failed: {exc}")
    _profile_session_id = None
    _profile_data_logger = None
    samples = list(_samples)
    leases = list(_leases)
    interrupted_spans = list(_interrupted_spans)
    process_info = copy.deepcopy(_process_info)
    observations = []
    observations_error = None
    try:
        observations = _resource_observations(samples, process_info=process_info)
    except Exception as _e:
        observations_error = _e

    # The raw timestamps needed by the trace only live in this process. Write
    # that irreplaceable artifact before spending shutdown time on summaries.
    if out_dir is not None:
        _agprof_print(f"[agprof] writing trace: {out_dir / 'agprof.trace.json'}")
        try:
            from .agprof_trace import write_trace

            trace_path = write_trace(
                out_dir,
                records,
                samples,
                leases,
                process_info=process_info,
                interrupted_spans=interrupted_spans,
                started_ns=started_ns,
                observations=observations,
                automatic_records=auto_events,
                thread_labels=dict(_thread_labels),
                profile_root_pid=_profile_root_pid,
            )
            _agprof_print(f"[agprof] Perfetto trace: {trace_path}")
        except BaseException as _e:  # signals/SystemExit must be visible too
            _agprof_print(f"[agprof] WARNING: trace output failed: {_e}")
            if not isinstance(_e, Exception):
                raise

    _last_summary = _build_summary(records)
    try:
        if observations_error is not None:
            raise observations_error
        _last_run_summary = _build_run_summary(
            records,
            samples,
            leases,
            observations=observations,
            interrupted_spans=interrupted_spans,
            started_ns=started_ns,
            ended_ns=t_end,
            sample_hz=_session_sample_hz,
            sample_gpu=_session_sample_gpu,
            gpu_sampling_available=sampler is not None and sampler._nvml is not None,
            automatic_function_metrics={
                "enabled": auto_settings is not None,
                "captured": len(auto_events),
                "dropped": _auto_dropped,
                "filtered": dict(_auto_filtered),
                "include_dependencies": auto_settings["include_dependencies"]
                if auto_settings
                else False,
                "min_duration_ms": auto_settings["min_duration_ms"] if auto_settings else None,
                "max_depth": auto_settings["max_depth"] if auto_settings else None,
                "max_events": auto_settings["max_events"] if auto_settings else None,
            },
        )
    except Exception as _e:  # never let reporting kill the run
        _last_run_summary = None
        _agprof_print(f"[agprof] WARNING: summary generation failed: {_e}")
    if out_dir is not None and _last_run_summary is not None:
        try:
            _write_summary_files(out_dir, _last_run_summary)
        except Exception as _e:  # never let reporting kill the run
            _agprof_print(f"[agprof] WARNING: summary output failed: {_e}")
    return prof


_BYTE_KINDS = {
    "io_r": ("io_read_mb_s", "io read MB/s"),
    "io_w": ("io_write_mb_s", "io write MB/s"),
    "net_rx": ("net_receive_mb_s", "net rx MB/s"),
    "net_tx": ("net_transmit_mb_s", "net tx MB/s"),
}


def _resource_observations(samples, *, process_info=None) -> list[dict]:
    """Convert raw sampler series into gauges/rates with stable names and units."""
    process_info = _process_info if process_info is None else process_info
    observations: list[dict] = []
    prev: "dict[str, tuple[int, float]]" = {}
    for t, series, value in samples:
        parts = series.split(":")
        scope = parts[0]
        kind = parts[-1]
        identity = parts[1] if scope == "proc" and len(parts) == 3 else None
        process = process_info.get(identity) if identity is not None else None
        cumulative_scope = scope in ("cg", "workload", "proc")
        if kind in ("cpu_us", "cpu_s") or kind in _BYTE_KINDS:
            if not cumulative_scope:
                continue
            prior = prev.get(series)
            prev[series] = (t, value)
            if prior is None or t <= prior[0]:
                continue
            delta = value - prior[1]
            if delta < 0:
                continue
            dt_s = (t - prior[0]) / 1e9
            label = series.split(":")[1]
            if kind in ("cpu_us", "cpu_s"):
                cpu_s = delta * (1e-6 if kind == "cpu_us" else 1.0)
                measured = 100.0 * cpu_s / dt_s
                unit = "percent"
                total_value = cpu_s
                total_unit = "CPU seconds"
                if scope == "proc":
                    name = f"process:{identity}:cpu_pct"
                    display_name = f"{process['display_name'] if process else identity} CPU"
                    trace_name = "cpu_percent"
                elif scope == "workload":
                    name = "workload_total:cpu_pct"
                    display_name = "workload total CPU"
                    trace_name = "workload_total cpu %"
                elif label in ("conmon", "dockerd"):
                    name = f"{label}:cpu_pct"
                    display_name = f"{label} CPU"
                    trace_name = f"{label} cpu %"
                else:
                    name = f"sandbox:{label}:cpu_pct"
                    display_name = f"sandbox {label} CPU"
                    trace_name = name
            else:
                measured = delta / 2**20 / dt_s
                unit = "MB/s"
                total_value = delta / 2**20
                total_unit = "MB"
                canonical_kind, trace_kind = _BYTE_KINDS[kind]
                if scope == "proc":
                    name = f"process:{identity}:{canonical_kind}"
                    display_name = (
                        f"{process['display_name'] if process else identity} {trace_kind}"
                    )
                    trace_name = canonical_kind
                elif scope == "workload":
                    name = f"workload_total:{canonical_kind}"
                    display_name = f"workload total {trace_kind}"
                    trace_name = f"workload_total {trace_kind}"
                else:
                    name = f"sandbox:{label}:{canonical_kind}"
                    display_name = f"sandbox {label} {trace_kind}"
                    trace_name = display_name.replace(f"sandbox {label} ", f"sandbox:{label}:")
        else:
            measured = value
            if scope == "proc" and identity is not None:
                name = f"process:{identity}:{kind}"
                trace_name = kind
                display_name = f"{process['display_name'] if process else identity} {kind}"
                unit = "MB" if kind in ("rss_mb", "vms_mb") else "value"
            elif scope == "workload" and kind == "memory_mb":
                name = "workload_total:memory_mb"
                trace_name = "workload_total memory_mb"
                display_name = "workload total memory"
                unit = "MB"
            else:
                name = series
                trace_name = series
                display_name, unit = _gauge_description(series)
            total_value = None
            total_unit = None
        observation = {
            "timestamp_ns": t,
            "name": name,
            "display_name": display_name,
            "trace_name": trace_name,
            "unit": unit,
            "value": float(measured),
            "interval_total": total_value,
            "total_unit": total_unit,
        }
        if process is not None:
            observation["process_identity"] = identity
            observation["trace_pid"] = process["trace_pid"]
            observation["process_name"] = process["display_name"]
        observations.append(observation)
    return observations


def _gauge_description(series: str) -> "tuple[str, str]":
    """Human label and unit for a direct (non-cumulative) sampler series."""
    suffix = series.rsplit(":", 1)[-1]
    unit = {
        "util_pct": "percent",
        "mem_mb": "MB",
        "power_w": "W",
        "power_w_est": "W",
        "rss_mb": "MB",
    }.get(suffix, "value")
    parts = series.split(":")
    if parts[0] == "sandbox" and suffix == "mem_mb":
        return f"sandbox {parts[1]} memory", unit
    if parts[0].startswith("gpu"):
        gpu = parts[0][3:]
        owner = f" {parts[1]}" if len(parts) == 3 else ""
        metric = {
            "util_pct": "utilization",
            "mem_mb": "memory",
            "power_w": "power",
            "power_w_est": "estimated power",
        }.get(suffix, suffix)
        return f"GPU {gpu}{owner} {metric}", unit
    return series, unit


def _build_summary(records) -> "dict[str, dict]":
    """Aggregate records per label: calls, wall/cpu/runq/blocked totals (ms)."""
    out: "dict[str, dict]" = {}
    for record in records:
        _tid, name, _t0, wall, cpu, runq, _metadata = _unpack_record(record)
        # Collapse per-instance labels (run3:..., turn2, llm:attempt[1], agmap:f[0])
        # onto stable keys so the table stays readable.
        key = name.split("[")[0]
        if key.startswith("run") and ":" in key:
            key = "run:" + key.split(":", 2)[1]
        elif key.startswith("turn"):
            key = "turn"
        row = out.setdefault(
            key, {"calls": 0, "wall_ms": 0.0, "cpu_ms": 0.0, "runq_ms": 0.0, "blocked_ms": 0.0}
        )
        row["calls"] += 1
        row["wall_ms"] += wall / 1e6
        # A partial sum is not a total. Keep the entire aggregate unknown
        # if any constituent lacks the required counter.
        row["cpu_ms"] = (
            row["cpu_ms"] + cpu / 1e6 if row["cpu_ms"] is not None and cpu is not None else None
        )
        row["runq_ms"] = (
            row["runq_ms"] + runq / 1e6 if row["runq_ms"] is not None and runq is not None else None
        )
        row["blocked_ms"] = (
            row["blocked_ms"] + max(0, wall - cpu - runq) / 1e6
            if row["blocked_ms"] is not None and cpu is not None and runq is not None
            else None
        )
    return out


def _span_key(name: str) -> str:
    """Collapse per-instance suffixes onto the stable summary label."""
    key = name.split("[")[0]
    if key.startswith("run") and ":" in key:
        key = "run:" + key.split(":", 2)[1]
    elif key.startswith("turn"):
        key = "turn"
    return key


def _percentile(values: list[float], quantile: float) -> "float | None":
    """Linearly interpolated percentile, matching common dataframe defaults."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency_stats(values: list[float]) -> dict:
    if not values:
        return {
            "mean_ms": None,
            "min_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    return {
        "mean_ms": round(sum(values) / len(values), 3),
        "min_ms": round(min(values), 3),
        "p50_ms": round(_percentile(values, 0.50), 3),
        "p95_ms": round(_percentile(values, 0.95), 3),
        "p99_ms": round(_percentile(values, 0.99), 3),
        "max_ms": round(max(values), 3),
    }


def _build_run_summary(
    records,
    samples,
    leases,
    *,
    observations=None,
    interrupted_spans=(),
    started_ns: "int | None",
    ended_ns: int,
    sample_hz: float,
    sample_gpu: bool,
    gpu_sampling_available: bool,
    automatic_function_metrics: "dict | None" = None,
) -> dict:
    """Build the complete, JSON-safe summary for one profiler session."""
    completed_records = [_unpack_record(record) for record in records]
    interrupted = [copy.deepcopy(span) for span in interrupted_spans]
    for span in interrupted:
        started = span.pop("started_ns", None)
        if started is not None and started_ns is not None:
            span["started_offset_ms"] = round(max(0, started - started_ns) / 1e6, 3)
    interrupted.sort(key=lambda span: span["duration_ms"], reverse=True)

    interrupted_by_label: "dict[str, list[dict]]" = {}
    for span in interrupted:
        interrupted_by_label.setdefault(_span_key(span["label"]), []).append(span)

    completed_by_label: "dict[str, list[tuple]]" = {}
    for record in completed_records:
        completed_by_label.setdefault(_span_key(record[1]), []).append(record)

    span_rows = []
    base_summary = _build_summary(records)
    for label in sorted(
        set(completed_by_label) | set(interrupted_by_label),
        key=lambda key: -base_summary.get(key, {}).get("wall_ms", 0),
    ):
        row = base_summary.get(
            label,
            {
                "calls": 0,
                "wall_ms": 0.0,
                "cpu_ms": 0.0,
                "runq_ms": 0.0,
                "blocked_ms": 0.0,
            },
        )
        label_records = completed_by_label.get(label, [])
        outcomes = [record[6].get("outcome", "unknown") for record in label_records]
        wall_values = [record[3] / 1e6 for record in label_records]
        wall_ms = row["wall_ms"]
        span_rows.append(
            {
                "label": label,
                "calls": row["calls"],
                "started": row["calls"] + len(interrupted_by_label.get(label, [])),
                "succeeded": outcomes.count("success"),
                "unknown": outcomes.count("unknown"),
                "failed": sum(outcome in ("failure", "failed", "error") for outcome in outcomes),
                "interrupted": len(interrupted_by_label.get(label, [])),
                "wall_ms": round(wall_ms, 3),
                "cpu_ms": round(row["cpu_ms"], 3)
                if row["cpu_ms"] is not None and label_records
                else None,
                "runqueue_ms": round(row["runq_ms"], 3)
                if row["runq_ms"] is not None and label_records
                else None,
                "blocked_ms": round(row["blocked_ms"], 3)
                if label_records
                and all(r[4] is not None and r[5] is not None for r in label_records)
                else None,
                "cpu_percent": round(100 * row["cpu_ms"] / wall_ms, 1)
                if wall_ms and row["cpu_ms"] is not None and label_records
                else None,
                **_latency_stats(wall_values),
            }
        )

    grouped: "dict[str, list[dict]]" = {}
    if observations is None:
        observations = _resource_observations(samples)
    for observation in observations:
        grouped.setdefault(observation["name"], []).append(observation)
    resource_rows = []
    for name, grouped_observations in sorted(grouped.items()):
        values = [observation["value"] for observation in grouped_observations]
        resource_rows.append(
            {
                "name": name,
                "display_name": grouped_observations[0]["display_name"],
                "unit": grouped_observations[0]["unit"],
                "samples": len(values),
                "mean": round(sum(values) / len(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "last": round(values[-1], 3),
            }
        )
        totals = [
            observation["interval_total"]
            for observation in grouped_observations
            if observation["interval_total"] is not None
        ]
        if totals:
            resource_rows[-1]["total"] = round(sum(totals), 6)
            resource_rows[-1]["total_unit"] = grouped_observations[0]["total_unit"]

    raw_series: "dict[str, list[tuple[int, float]]]" = {}
    for timestamp, series, value in samples:
        raw_series.setdefault(series, []).append((timestamp, value))
    resource_by_name = {row["name"]: row for row in resource_rows}
    for series, points in raw_series.items():
        if not series.endswith(":power_w") or len(points) < 2:
            continue
        points.sort()
        energy_j = 0.0
        for (t0, v0), (t1, v1) in zip(points, points[1:]):
            if t1 > t0:
                energy_j += (t1 - t0) / 1e9 * (v0 + v1) / 2
        if series in resource_by_name:
            resource_by_name[series]["energy_j"] = round(energy_j, 3)

    lease_groups: "dict[tuple[int, str], list[float]]" = {}
    for gpu_id, t0, t1, label in leases:
        lease_groups.setdefault((gpu_id, label), []).append(max(0, t1 - t0) / 1e6)
    lease_rows = []
    for (gpu_id, label), durations in sorted(lease_groups.items()):
        lease_rows.append(
            {
                "gpu_id": gpu_id,
                "label": label,
                "leases": len(durations),
                "total_ms": round(sum(durations), 3),
                "mean_ms": round(sum(durations) / len(durations), 3),
                "max_ms": round(max(durations), 3),
            }
        )

    if started_ns is None:
        duration_ms = 0.0
    else:
        duration_ms = max(0, ended_ns - started_ns) / 1e6

    def is_run_label(name: str) -> bool:
        prefix, separator, _rest = name.partition(":")
        return separator == ":" and prefix.startswith("run") and prefix[3:].isdigit()

    completed_runs = [record for record in completed_records if is_run_label(record[1])]
    interrupted_runs = [span for span in interrupted if is_run_label(span["label"])]
    run_outcomes = [record[6].get("outcome", "unknown") for record in completed_runs]
    duration_s = duration_ms / 1e3
    run_metrics = {
        "started": len(completed_runs) + len(interrupted_runs),
        "completed": len(completed_runs),
        "succeeded": run_outcomes.count("success"),
        "failed": sum(outcome != "success" for outcome in run_outcomes),
        "interrupted": len(interrupted_runs),
        "completed_per_second": round(len(completed_runs) / duration_s, 6) if duration_s else 0.0,
        "successful_per_second": round(run_outcomes.count("success") / duration_s, 6)
        if duration_s
        else 0.0,
        **_latency_stats([record[3] / 1e6 for record in completed_runs]),
    }

    attempt_records = [
        record for record in completed_records if record[1].startswith("llm:attempt[")
    ]
    interrupted_attempts = [
        span for span in interrupted if span["label"].startswith("llm:attempt[")
    ]
    attempt_metadata = [record[6] for record in attempt_records]
    ttft_values = [
        float(metadata["ttft_ms"])
        for metadata in attempt_metadata
        if metadata.get("ttft_ms") is not None
    ]
    generation_pairs = [
        metadata
        for metadata in attempt_metadata
        if metadata.get("generation_ms") is not None and metadata.get("output_tokens") is not None
    ]
    generation_ms = sum(float(metadata["generation_ms"]) for metadata in generation_pairs)
    generated_tokens = sum(int(metadata["output_tokens"]) for metadata in generation_pairs)
    input_tokens = sum(int(metadata.get("input_tokens") or 0) for metadata in attempt_metadata)
    output_tokens = sum(int(metadata.get("output_tokens") or 0) for metadata in attempt_metadata)
    calls = sum(record[1].startswith("llm:attempt[0]") for record in attempt_records) + sum(
        span["label"].startswith("llm:attempt[0]") for span in interrupted_attempts
    )
    retry_indices = [metadata.get("retry_index") for metadata in attempt_metadata]
    retries = (
        sum(index > 0 for index in retry_indices)
        if retry_indices
        and all(isinstance(index, int) for index in retry_indices)
        and not interrupted_attempts
        else None
    )
    if retries is None and any(
        not record[1].startswith("llm:attempt[0]") for record in attempt_records
    ):
        retries = len(attempt_records) + len(interrupted_attempts) - calls
    llm_metrics = {
        "calls": calls,
        "successful_calls": sum(
            metadata.get("outcome", "unknown") == "success" for metadata in attempt_metadata
        ),
        "failed_calls": sum(
            metadata.get("outcome", "unknown") in ("failure", "failed", "error")
            and not metadata.get("retrying")
            for metadata in attempt_metadata
        ),
        "interrupted_calls": len(interrupted_attempts),
        "attempts": len(attempt_records) + len(interrupted_attempts),
        "retries": retries,
        "retry_coverage": "reported" if retries is not None else "unavailable",
        "retries_reported": sum(r[1].startswith("llm:retry_backoff") for r in completed_records)
        + sum(r["label"].startswith("llm:retry_backoff") for r in interrupted),
        "successful_attempts": sum(
            metadata.get("outcome", "unknown") == "success" for metadata in attempt_metadata
        ),
        "failed_attempts": sum(
            metadata.get("outcome", "unknown") in ("failure", "failed", "error")
            for metadata in attempt_metadata
        ),
        "interrupted_attempts": len(interrupted_attempts),
        "total_wait_ms": round(sum(record[3] for record in attempt_records) / 1e6, 3),
        "reported_input_tokens": input_tokens,
        "reported_output_tokens": output_tokens,
        "usage_missing_attempts": sum(
            m.get("input_tokens") is None or m.get("output_tokens") is None
            for m in attempt_metadata
        )
        + len(interrupted_attempts),
        "input_tokens": input_tokens
        if not interrupted_attempts
        and attempt_metadata
        and all(m.get("input_tokens") is not None for m in attempt_metadata)
        else None,
        "output_tokens": output_tokens
        if not interrupted_attempts
        and attempt_metadata
        and all(m.get("output_tokens") is not None for m in attempt_metadata)
        else None,
        "throughput_measured_attempts": len(generation_pairs),
        "output_tokens_per_second": round(generated_tokens / (generation_ms / 1e3), 3)
        if generation_ms
        else None,
        "latency": _latency_stats([record[3] / 1e6 for record in attempt_records]),
        "ttft": _latency_stats(ttft_values),
    }

    tool_records = [record for record in completed_records if record[1].startswith("tool:")]
    interrupted_tools = [span for span in interrupted if span["label"].startswith("tool:")]
    tool_outcomes = [record[6].get("outcome", "unknown") for record in tool_records]
    tool_metrics = {
        "started": len(tool_records) + len(interrupted_tools),
        "completed": len(tool_records),
        "succeeded": tool_outcomes.count("success"),
        "unknown": tool_outcomes.count("unknown"),
        "failed": sum(outcome in ("failure", "failed", "error") for outcome in tool_outcomes),
        "interrupted": len(interrupted_tools),
        "latency": _latency_stats(
            [
                record[3] / 1e6
                for record in tool_records
                if record[6].get("timing", "exact") == "exact"
            ]
        ),
        "latency_by_timing": {
            timing: _latency_stats(
                [
                    record[3] / 1e6
                    for record in tool_records
                    if record[6].get("timing", "exact") == timing
                ]
            )
            for timing in sorted({record[6].get("timing", "exact") for record in tool_records})
        },
        "by_tool": [],
    }
    tool_names = sorted(
        {record[1].removeprefix("tool:") for record in tool_records}
        | {span["label"].removeprefix("tool:") for span in interrupted_tools}
    )
    for tool_name in tool_names:
        matching = [record for record in tool_records if record[1] == f"tool:{tool_name}"]
        matching_interrupted = [
            span for span in interrupted_tools if span["label"] == f"tool:{tool_name}"
        ]
        matching_outcomes = [record[6].get("outcome", "unknown") for record in matching]
        tool_metrics["by_tool"].append(
            {
                "name": tool_name,
                "started": len(matching) + len(matching_interrupted),
                "completed": len(matching),
                "succeeded": matching_outcomes.count("success"),
                "unknown": matching_outcomes.count("unknown"),
                "failed": sum(
                    outcome in ("failure", "failed", "error") for outcome in matching_outcomes
                ),
                "interrupted": len(matching_interrupted),
                **_latency_stats(
                    [
                        record[3] / 1e6
                        for record in matching
                        if record[6].get("timing", "exact") == "exact"
                    ]
                ),
            }
        )

    workload_metrics = {}
    workload_cpu = resource_by_name.get("workload_total:cpu_pct")
    workload_memory = resource_by_name.get("workload_total:memory_mb")
    if workload_cpu:
        workload_metrics.update(
            cpu_average_percent=workload_cpu["mean"],
            cpu_peak_percent=workload_cpu["max"],
            cpu_time_seconds=workload_cpu.get("total"),
        )
    if workload_memory:
        workload_metrics.update(
            memory_average_mb=workload_memory["mean"],
            memory_peak_mb=workload_memory["max"],
        )
    for kind in ("io_read", "io_write"):
        row = resource_by_name.get(f"workload_total:{kind}_mb_s")
        if row:
            workload_metrics[f"{kind}_mb"] = row.get("total", 0.0)

    process_identities = sorted(
        {
            name.split(":", 2)[1]
            for name in resource_by_name
            if name.startswith("process:") and name.count(":") >= 2
        },
        key=lambda identity: _process_info.get(identity, {}).get("first_seen_ns", 0),
    )
    process_metrics = []
    for identity in process_identities:
        info = _process_info.get(identity, {})
        prefix = f"process:{identity}:"
        cpu = resource_by_name.get(prefix + "cpu_pct")
        rss = resource_by_name.get(prefix + "rss_mb")
        vms = resource_by_name.get(prefix + "vms_mb")
        read = resource_by_name.get(prefix + "io_read_mb_s")
        write = resource_by_name.get(prefix + "io_write_mb_s")
        process_metrics.append(
            {
                "identity": identity,
                "pid": info.get("pid"),
                "name": info.get("comm", identity),
                "display_name": info.get("display_name", identity),
                "cmdline": info.get("cmdline", ""),
                "sandbox": info.get("sandbox"),
                "cgroup": info.get("cgroup"),
                "cpu_average_percent": cpu["mean"] if cpu else None,
                "cpu_peak_percent": cpu["max"] if cpu else None,
                "cpu_time_seconds": cpu.get("total") if cpu else None,
                "rss_average_mb": rss["mean"] if rss else None,
                "rss_peak_mb": rss["max"] if rss else None,
                "vms_average_mb": vms["mean"] if vms else None,
                "vms_peak_mb": vms["max"] if vms else None,
                "io_read_mb": read.get("total") if read else None,
                "io_write_mb": write.get("total") if write else None,
                "samples": max(
                    (metric["samples"] for metric in (cpu, rss, vms, read, write) if metric),
                    default=0,
                ),
            }
        )

    gpu_ids = sorted(
        {
            int(name[3:].split(":", 1)[0])
            for name in resource_by_name
            if name.startswith("gpu")
            and name[3:].split(":", 1)[0].isdigit()
            and name.count(":") == 1
        }
    )
    gpu_metrics = []
    for gpu_id in gpu_ids:
        prefix = f"gpu{gpu_id}:"
        util = resource_by_name.get(prefix + "util_pct")
        memory = resource_by_name.get(prefix + "mem_mb")
        power = resource_by_name.get(prefix + "power_w")
        gpu_metrics.append(
            {
                "gpu_id": gpu_id,
                "utilization_average_percent": util["mean"] if util else None,
                "utilization_peak_percent": util["max"] if util else None,
                "memory_average_mb": memory["mean"] if memory else None,
                "memory_peak_mb": memory["max"] if memory else None,
                "power_average_w": power["mean"] if power else None,
                "power_peak_w": power["max"] if power else None,
                "energy_j": power.get("energy_j") if power else None,
            }
        )

    sandbox_labels = sorted(
        {
            name.split(":", 2)[1]
            for name in resource_by_name
            if name.startswith("sandbox:") and name.count(":") >= 2
        }
    )
    sandbox_metrics = []
    for label in sandbox_labels:
        prefix = f"sandbox:{label}:"
        cpu = resource_by_name.get(prefix + "cpu_pct")
        memory = resource_by_name.get(prefix + "mem_mb")
        row = {
            "label": label,
            "cpu_average_percent": cpu["mean"] if cpu else None,
            "cpu_peak_percent": cpu["max"] if cpu else None,
            "cpu_time_seconds": cpu.get("total") if cpu else None,
            "memory_average_mb": memory["mean"] if memory else None,
            "memory_peak_mb": memory["max"] if memory else None,
        }
        for kind in ("io_read", "io_write", "net_receive", "net_transmit"):
            metric = resource_by_name.get(prefix + f"{kind}_mb_s")
            row[f"{kind}_mb"] = metric.get("total") if metric else None
        sandbox_metrics.append(row)

    tick_times = sorted({timestamp for timestamp, _series, _value in samples})
    sampled_duration_s = (tick_times[-1] - tick_times[0]) / 1e9 if len(tick_times) >= 2 else 0.0
    return {
        "schema_version": 6,
        "data_source": "mixed"
        if any(
            record[6].get("provenance") == "container_asserted"
            or record[6].get("timing") in ("derived", "hook_boundary", "container_asserted")
            for record in completed_records
        )
        else "measured",
        "coverage": {
            "engines": copy.deepcopy(_engine_coverage),
            "tools": {
                "state": "partial" if tool_records or interrupted_tools else "unavailable",
                "reason": "Only instrumented admission/completion events are observable; absence is not zero.",
                "timing_counts": dict(
                    Counter(record[6].get("timing", "exact") for record in tool_records)
                ),
            },
            "resources": {
                "state": "sampled" if samples else "unavailable",
                "reason": "Gauges and PID membership are sampled; brief peaks and short-lived processes may be missed.",
                "gpu_process_power": "estimated",
                "cpu_attribution": "thread intervals, inclusive of concurrent work; remote CPU unavailable",
            },
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": round(duration_ms, 3),
        "run_metrics": run_metrics,
        "llm_metrics": llm_metrics,
        "tool_metrics": tool_metrics,
        "workload_metrics": workload_metrics,
        "process_metrics": process_metrics,
        "gpu_metrics": gpu_metrics,
        "sandbox_metrics": sandbox_metrics,
        "sampling": {
            "configured_hz": sample_hz,
            "effective_hz": round((len(tick_times) - 1) / sampled_duration_s, 3)
            if sampled_duration_s
            else 0.0,
            "sampled_duration_ms": round(sampled_duration_s * 1e3, 3),
            "gpu_requested": sample_gpu,
            "gpu_available": gpu_sampling_available,
            "raw_samples": len(samples),
            "telemetry_errors": dict(_health),
            "lossless": False,
        },
        "span_metrics": span_rows,
        "resource_metrics": resource_rows,
        "gpu_lease_metrics": lease_rows,
        "automatic_function_metrics": automatic_function_metrics or {"enabled": False},
        "incomplete_spans": interrupted,
    }


def summary_metrics() -> "dict | None":
    """A copy of the complete metrics document for the last completed session."""
    return copy.deepcopy(_last_run_summary)


def profile_records() -> list[tuple]:
    """A copy of completed span records loaded from the last session's dataloggers."""
    return copy.deepcopy(_last_profile_records)


def _markdown_escape(value) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _render_summary_markdown(summary: dict) -> str:
    """Render a complete profiler summary as a standalone Markdown report."""
    sampling = summary["sampling"]
    runs = summary["run_metrics"]
    llm = summary["llm_metrics"]
    tools = summary["tool_metrics"]
    automatic = summary.get("automatic_function_metrics", {"enabled": False})

    def number(value, digits=3):
        return "n/a" if value is None else f"{value:.{digits}f}"

    def milliseconds(value):
        return "n/a" if value is None else f"{value:.3f} ms"

    lines = [
        "# agprof summary",
        "",
    ]
    if summary.get("data_source") not in ("measured", "mixed"):
        lines.extend(
            [
                "> **MOCK DATA — illustrative only. These values were not measured.**",
                "",
            ]
        )
    lines.extend(
        [
            f"- Duration: **{summary['duration_ms'] / 1e3:.3f} s**",
            f"- Runs: **{runs['completed']}/{runs['started']} completed**, "
            f"{runs['succeeded']} succeeded, {runs['failed']} failed, "
            f"{runs['interrupted']} interrupted",
            f"- Completed throughput: **{runs['completed_per_second']:.3f} runs/s**",
            f"- LLM: **{llm['calls']} calls**, {llm['successful_calls']} succeeded, "
            f"{llm['failed_calls']} failed, {llm['interrupted_calls']} interrupted, "
            f"{number(llm['retries'], 0)} reported retries, "
            f"{llm['total_wait_ms'] / 1e3:.3f} s total wait",
            f"- Tools: **{tools['completed']}/{tools['started']} completed**, "
            f"{tools['failed']} failed, {tools['interrupted']} interrupted",
            (
                f"- Automatic Python calls: **{automatic.get('captured', 0)} captured**, "
                f"{automatic.get('dropped', 0)} dropped"
                if automatic.get("enabled")
                else "- Automatic Python calls: **disabled**"
            ),
            f"- Raw resource samples: **{sampling['raw_samples']}** "
            f"at {sampling['effective_hz']:g} Hz effective "
            f"({sampling['configured_hz']:g} Hz configured)",
            f"- GPU sampling: **{'available' if sampling['gpu_available'] else 'unavailable'}** "
            f"({'requested' if sampling['gpu_requested'] else 'not requested'})",
            "",
            "## Run, LLM, and tool metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Run latency p50 / p95 | {number(runs['p50_ms'])} / {number(runs['p95_ms'])} ms |",
            f"| LLM latency p50 / p95 | {number(llm['latency']['p50_ms'])} / "
            f"{number(llm['latency']['p95_ms'])} ms |",
            f"| LLM TTFT p50 / p95 | {number(llm['ttft']['p50_ms'])} / "
            f"{number(llm['ttft']['p95_ms'])} ms |",
            f"| LLM input / output tokens | {number(llm['input_tokens'], 0)} / {number(llm['output_tokens'], 0)} |",
            f"| LLM output throughput | {number(llm['output_tokens_per_second'])} tokens/s |",
            f"| LLM attempts | {llm['attempts']} total, {llm['successful_attempts']} succeeded, "
            f"{llm['failed_attempts']} failed, {llm['interrupted_attempts']} interrupted |",
            f"| Tool latency p50 / p95 | {number(tools['latency']['p50_ms'])} / "
            f"{number(tools['latency']['p95_ms'])} ms |",
            "",
            "### Tool outcomes",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "### Coverage",
            "",
            "Tool latency percentiles include exact intervals only; other timings are separated in JSON.",
            f"Unknown tool outcomes: {tools.get('unknown', 0)}. Missing measurements are not zero.",
            "Resources are sampled; per-process GPU power is estimated. Telemetry is not lossless.",
            f"Telemetry errors: {json.dumps(summary.get('sampling', {}).get('telemetry_errors', {}), sort_keys=True)}",
        ]
    )
    for engine, coverage in summary.get("coverage", {}).get("engines", {}).items():
        lines.append(
            f"- {_markdown_escape(engine)}: {_markdown_escape(json.dumps(coverage, sort_keys=True))}"
        )
    if tools["by_tool"]:
        lines.extend(
            [
                "| Tool | Completed/started | Succeeded | Failed | Interrupted | "
                "p50 latency | p95 latency |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for tool in tools["by_tool"]:
            lines.append(
                f"| {_markdown_escape(tool['name'])} | "
                f"{tool['completed']}/{tool['started']} | {tool['succeeded']} | "
                f"{tool['failed']} | {tool['interrupted']} | "
                f"{milliseconds(tool['p50_ms'])} | {milliseconds(tool['p95_ms'])} |"
            )
    else:
        lines.append("_No tool spans were recorded._")
    lines.extend(
        [
            "",
            "## Workload aggregate",
            "",
            "| CPU avg | CPU peak | CPU time | Memory avg | Memory peak | Disk read | Disk write |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {number(summary['workload_metrics'].get('cpu_average_percent'))}% | "
            f"{number(summary['workload_metrics'].get('cpu_peak_percent'))}% | "
            f"{number(summary['workload_metrics'].get('cpu_time_seconds'))} s | "
            f"{number(summary['workload_metrics'].get('memory_average_mb'))} MB | "
            f"{number(summary['workload_metrics'].get('memory_peak_mb'))} MB | "
            f"{number(summary['workload_metrics'].get('io_read_mb'), 6)} MB | "
            f"{number(summary['workload_metrics'].get('io_write_mb'), 6)} MB |",
            "",
            "## Per-process metrics",
            "",
        ]
    )
    if summary["process_metrics"]:
        lines.extend(
            [
                "| Process | PID | Sandbox | Samples | CPU avg | CPU peak | CPU time | "
                "RSS avg | RSS peak | VMS avg | VMS peak | Disk read | Disk write |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for process in summary["process_metrics"]:
            lines.append(
                f"| {_markdown_escape(process['name'])} | "
                f"{process['pid'] if process['pid'] is not None else 'n/a'} | "
                f"{_markdown_escape(process['sandbox'] or '')} | {process['samples']} | "
                f"{number(process['cpu_average_percent'])}% | "
                f"{number(process['cpu_peak_percent'])}% | "
                f"{number(process['cpu_time_seconds'])} s | "
                f"{number(process['rss_average_mb'])} MB | "
                f"{number(process['rss_peak_mb'])} MB | "
                f"{number(process['vms_average_mb'])} MB | "
                f"{number(process['vms_peak_mb'])} MB | "
                f"{number(process['io_read_mb'], 6)} MB | "
                f"{number(process['io_write_mb'], 6)} MB |"
            )
    else:
        lines.append("_No workload processes were sampled._")
    lines.extend(
        [
            "",
            "## GPU metrics",
            "",
        ]
    )
    if summary["gpu_metrics"]:
        lines.extend(
            [
                "| GPU | Util avg | Util peak | VRAM avg | VRAM peak | Power avg | "
                "Power peak | Energy |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for gpu in summary["gpu_metrics"]:
            lines.append(
                f"| {gpu['gpu_id']} | {number(gpu['utilization_average_percent'])}% | "
                f"{number(gpu['utilization_peak_percent'])}% | "
                f"{number(gpu['memory_average_mb'])} MB | {number(gpu['memory_peak_mb'])} MB | "
                f"{number(gpu['power_average_w'])} W | {number(gpu['power_peak_w'])} W | "
                f"{number(gpu['energy_j'])} J |"
            )
    else:
        lines.append("_No GPU samples were collected._")

    lines.extend(["", "## Sandbox metrics", ""])
    if summary["sandbox_metrics"]:
        lines.extend(
            [
                "| Sandbox | CPU avg | CPU peak | CPU time | Memory avg | Memory peak | "
                "Disk read | Disk write | Net receive | Net transmit |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for sandbox in summary["sandbox_metrics"]:
            lines.append(
                f"| {_markdown_escape(sandbox['label'])} | "
                f"{number(sandbox['cpu_average_percent'])}% | "
                f"{number(sandbox['cpu_peak_percent'])}% | "
                f"{number(sandbox['cpu_time_seconds'])} s | "
                f"{number(sandbox['memory_average_mb'])} MB | "
                f"{number(sandbox['memory_peak_mb'])} MB | "
                f"{number(sandbox['io_read_mb'], 6)} MB | "
                f"{number(sandbox['io_write_mb'], 6)} MB | "
                f"{number(sandbox['net_receive_mb'], 6)} MB | "
                f"{number(sandbox['net_transmit_mb'], 6)} MB |"
            )
    else:
        lines.append("_No sandbox resource samples were collected._")

    lines.extend(["", "## Incomplete spans", ""])
    if summary["incomplete_spans"]:
        lines.extend(
            [
                "| Label | Thread | Elapsed at stop (s) | Outcome |",
                "|---|---:|---:|---|",
            ]
        )
        for interrupted in summary["incomplete_spans"]:
            lines.append(
                f"| {_markdown_escape(interrupted['label'])} | "
                f"{interrupted['thread_id']} | {interrupted['duration_ms'] / 1e3:.3f} | "
                f"{interrupted['outcome']} |"
            )
    else:
        lines.append("_No spans were still open when profiling stopped._")

    lines.extend(
        [
            "",
            "## Span metrics",
            "",
        ]
    )
    if summary["span_metrics"]:
        lines.extend(
            [
                "| Label | Completed/started | Failed | Interrupted | Wall (s) | CPU (s) | "
                "Blocked (s) | Mean (ms) | p50 (ms) | p95 (ms) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["span_metrics"]:
            lines.append(
                f"| {_markdown_escape(row['label'])} | {row['calls']}/{row['started']} | "
                f"{row['failed']} | {row['interrupted']} | "
                f"{row['wall_ms'] / 1e3:.3f} | {number(None if row['cpu_ms'] is None else row['cpu_ms'] / 1e3)} | "
                f"{number(None if row['blocked_ms'] is None else row['blocked_ms'] / 1e3)} | {number(row['mean_ms'])} | "
                f"{number(row['p50_ms'])} | {number(row['p95_ms'])} |"
            )
    else:
        lines.append("_No completed spans._")

    lines.extend(["", "## Resource metrics", ""])
    if summary["resource_metrics"]:
        lines.extend(
            [
                "| Metric | Unit | Samples | Mean | Min | Max | Last | Total | Energy |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["resource_metrics"]:
            total_text = (
                f"{number(row['total'], 6)} {row['total_unit']}"
                if row.get("total") is not None
                else "n/a"
            )
            energy_text = (
                f"{number(row['energy_j'])} J" if row.get("energy_j") is not None else "n/a"
            )
            lines.append(
                f"| {_markdown_escape(row['display_name'])} | {row['unit']} | "
                f"{row['samples']} | {row['mean']:.3f} | {row['min']:.3f} | "
                f"{row['max']:.3f} | {row['last']:.3f} | {total_text} | {energy_text} |"
            )
    else:
        lines.append("_No resource samples were collected._")

    lines.extend(["", "## GPU lease metrics", ""])
    if summary["gpu_lease_metrics"]:
        lines.extend(
            [
                "| GPU | Holder | Leases | Total (s) | Mean (s) | Max (s) |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in summary["gpu_lease_metrics"]:
            lines.append(
                f"| {row['gpu_id']} | {_markdown_escape(row['label'])} | "
                f"{row['leases']} | {row['total_ms'] / 1e3:.3f} | "
                f"{row['mean_ms'] / 1e3:.3f} | {row['max_ms'] / 1e3:.3f} |"
            )
    else:
        lines.append("_No GPU leases were recorded._")
    return "\n".join(lines) + "\n"


def _write_summary_files(out_dir: Path, summary: dict) -> "tuple[Path, Path]":
    """Atomically write the machine- and human-readable per-run summaries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    markdown_path = out_dir / "summary.md"
    json_tmp = out_dir / f".summary.json.{os.getpid()}.tmp"
    markdown_tmp = out_dir / f".summary.md.{os.getpid()}.tmp"
    json_tmp.write_text(json.dumps(summary, indent=2) + "\n")
    markdown_tmp.write_text(_render_summary_markdown(summary))
    json_tmp.replace(json_path)
    markdown_tmp.replace(markdown_path)
    _agprof_print(f"[agprof] summaries: {json_path}, {markdown_path}")
    return json_path, markdown_path


def summary_table(sort_by: str = "wall_ms", row_limit: int = 30) -> str:
    """Formatted per-label CPU-vs-wait table for the last completed session."""
    if not _last_summary:
        return "(no completed agprof session)"
    rows = sorted(_last_summary.items(), key=lambda kv: kv[1][sort_by], reverse=True)
    lines = [
        f"{'label':<28} {'calls':>5} {'wall_s':>9} {'cpu_s':>8} {'runq_s':>8} {'blocked_s':>9} {'cpu%':>6}"
    ]
    lines.append("-" * len(lines[0]))

    def seconds(value):
        return "n/a" if value is None else f"{value / 1e3:.2f}"

    for name, r in rows[:row_limit]:
        cpu_pct = (
            f"{100 * r['cpu_ms'] / r['wall_ms']:.1f}%"
            if r["cpu_ms"] is not None and r["wall_ms"]
            else "n/a"
        )
        lines.append(
            f"{name:<28} {r['calls']:>5} {seconds(r['wall_ms']):>9} {seconds(r['cpu_ms']):>8} "
            f"{seconds(r['runq_ms']):>8} {seconds(r['blocked_ms']):>9} {cpu_pct:>6}"
        )
    return "\n".join(lines)


@contextmanager
def session(
    out_dir=None,
    *,
    all_threads: bool = True,
    worker_name: "str | None" = None,
    sample_hz: float = 10.0,
    sample_gpu: bool = True,
    auto_functions: bool = True,
    auto_include_dependencies: bool = False,
    auto_include=None,
    auto_exclude=None,
    auto_min_duration_ms: float = _DEFAULT_AUTO_MIN_DURATION_MS,
    auto_max_depth: int = _DEFAULT_AUTO_MAX_DEPTH,
    auto_max_events: int = _DEFAULT_AUTO_MAX_EVENTS,
):
    """Profile everything inside the block and write summaries on exit."""
    prof = start(
        out_dir,
        all_threads=all_threads,
        worker_name=worker_name,
        sample_hz=sample_hz,
        sample_gpu=sample_gpu,
        auto_functions=auto_functions,
        auto_include_dependencies=auto_include_dependencies,
        auto_include=auto_include,
        auto_exclude=auto_exclude,
        auto_min_duration_ms=auto_min_duration_ms,
        auto_max_depth=auto_max_depth,
        auto_max_events=auto_max_events,
    )
    try:
        yield prof
    finally:
        stop()


def profile_scope() -> str:
    """Configured environment profiling scope.

    Only ``process`` opts into process-lifetime profiling. Unset, empty, and
    invalid values all select the deterministic ``workload`` default.
    """
    value = os.environ.get("AGENCY_PROFILE_SCOPE", "").strip().lower()
    return "process" if value == "process" else "workload"


def _env_enabled() -> bool:
    """On by default -- unset means "profile". AGENCY_PROFILE=0 opts out."""
    if "AGENCY_PROFILE" not in os.environ:
        return True
    return os.environ["AGENCY_PROFILE"].strip().lower() in ("1", "true")


def _env_out_dir() -> str:
    # Lazy import: agutil is a low-level, widely-imported module and agprof
    # is imported from many places (container.py, agsandbox.py, ...);
    # importing at call time rather than module load time avoids adding a
    # module-level import-order constraint between the two.
    from ...utils.agutil import agency_run_dir_name, agency_runs_dir

    default = str(agency_runs_dir() / agency_run_dir_name() / "profiler")
    return os.environ.get("AGENCY_PROFILE_DIR", default)


@contextmanager
def workload():
    """Profile a workload boundary when environment profiling requests it.

    The context owns a session only for ``AGENCY_PROFILE_SCOPE=workload`` (the
    default) and only when no explicit session is already active. This keeps
    callers free of scope conditionals and prevents the context from stopping
    a session it did not start. A failed start() degrades to running
    unprofiled rather than blocking the actual work.
    """
    owns_session = False
    prof = _profiler
    if _env_enabled() and profile_scope() == "workload" and not enabled():
        try:
            prof = start(_env_out_dir())
            owns_session = True
        except RuntimeError as exc:
            _agprof_print(
                f"[agprof] WARNING: workload profiling failed to start, continuing unprofiled: {exc}"
            )
    try:
        yield prof
    finally:
        if owns_session:
            stop()


def _maybe_autostart() -> None:
    """Start process-lifetime profiling when explicitly requested."""
    if not _env_enabled() or profile_scope() != "process":
        return
    global _process_shutdown_started
    with _process_shutdown_lock:
        _process_shutdown_started = False
    start(_env_out_dir())
    _install_process_profile_signal_handlers()
    atexit.register(_shutdown_process_profile)


def _shutdown_process_profile() -> None:
    """Finalize process-scope traces exactly once."""
    global _process_shutdown_started
    with _process_shutdown_lock:
        if _process_shutdown_started:
            return
        _process_shutdown_started = True
    stop()


def _install_process_profile_signal_handlers() -> None:
    """Make catchable termination signals follow the process shutdown path."""
    for signum in _PROCESS_PROFILE_SIGNALS:
        signal.signal(signum, _process_profile_signal_handler)


def _process_profile_signal_handler(signum, _frame) -> None:
    """Finalize once, then preserve the signal's normal exit semantics."""
    # Prevent another SIGINT/SIGTERM from re-entering Python while the trace
    # is being written. The original signal is re-raised below after shutdown.
    for handled_signal in _PROCESS_PROFILE_SIGNALS:
        signal.signal(handled_signal, signal.SIG_IGN)
    try:
        _shutdown_process_profile()
    finally:
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)


def _initialize_environment_profiling() -> None:
    """Validate and isolate env-requested profiling before workload startup.
    Only AGENCY_PROFILE_SCOPE=process needs the dedicated cgroup this sets
    up; the default `workload` scope samples whatever cgroup this process
    is already in, no privilege needed. Degrades to unprofiled rather than
    blocking startup if the cgroup setup fails."""
    if not _env_enabled() or profile_scope() != "process":
        return
    try:
        _require_linux()
        _ensure_environment_cgroup()
    except RuntimeError as exc:
        _agprof_print(
            "[agprof] WARNING: environment profiling unavailable, continuing "
            f"unprofiled at the process level: {exc}"
        )
        return
    try:
        _maybe_autostart()
    except RuntimeError as exc:
        _agprof_print(f"[agprof] WARNING: process-scope profiling failed to start: {exc}")


# The native collector loads this stdlib-only module without starting a host session.
if __name__ != "_agency_container_profiler":
    _initialize_environment_profiling()
