"""agprof — profiling switch for the framework's built-in span annotations.

The framework's hot paths (skill runs, LLM calls, tool dispatch, sandbox ops,
agmap tasks) are permanently annotated with ``agprof.span(name)`` calls. With no
active session (the default) every one of them returns a shared no-op context
manager — one module-global load and an ``is None`` check; torch is never
imported. Applications opt in with one line (or one env var) and the framework
acts accordingly:

    from agency import agprof

    with agprof.session(run_dir / "tb_trace"):   # torch.profiler lifecycle
        team.run()

    # or, zero application changes with process-lifetime scope:
    #   AGENCY_PROFILE=1 AGENCY_PROFILE_SCOPE=process python app.py

Backend: torch.profiler (kineto). ``span()`` maps to
``torch.profiler.record_function`` and the session wraps ``profile(...)`` with
``profile_all_threads=True`` — REQUIRED, because the framework runs skills and
agmap tasks on plain ``threading.Thread`` daemons, and without a global observer
kineto silently drops every span opened on them. Traces land as
``*.pt.trace.json`` (TensorBoard torch-tb-profiler plugin / ui.perfetto.dev).

CPU-vs-wait split: every span additionally records its thread's on-CPU time
(``time.thread_time_ns``) and run-queue wait (``/proc/self/task/<tid>/schedstat``)
across the span. At session stop the deltas are injected into the trace file as
per-span ``args`` (click a span in the viewer):

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
import itertools
import json
import os
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext, suppress
from datetime import datetime, timezone
from pathlib import Path

_NULL = nullcontext()

_session = None  # None = profiling off (the fast path checks only this)
_profiler = None  # the live torch.profiler.profile object, if any
_out_dir: "Path | None" = None
_state_lock = threading.Lock()

_counters: "dict[str, itertools.count]" = {}
_counters_lock = threading.Lock()

# Completed-span records for the CPU-vs-wait split:
# (tid, name, t0_wall_ns, wall_ns, cpu_ns, runq_ns | None, metadata).
# list.append is atomic under the GIL, so no lock on the hot path.
_records: list[tuple] = []
_open_spans: "dict[int, _TimedSpan]" = {}
_open_spans_lock = threading.Lock()
_interrupted_spans: "list[dict]" = []
_last_summary: "dict[str, dict] | None" = None
_last_run_summary: "dict | None" = None

_tls = threading.local()

# Sampler timeline + GPU lease intervals (see _Sampler / gpu_lease_*).
_samples: "list[tuple[int, str, float]]" = []  # (t_mono_ns, series, value)
_process_info: "dict[str, dict]" = {}  # pid-start_ticks identity -> trace/display metadata
_sampler: "_Sampler | None" = None
_leases_open: "dict[int, tuple[int, str]]" = {}  # gpu_id -> (t0_ns, label)
_leases: "list[tuple[int, int, int, str]]" = []  # (gpu_id, t0_ns, t1_ns, label)
_leases_lock = threading.Lock()
_clock_mark_ns: "int | None" = None
_session_started_ns: "int | None" = None
_session_sample_hz = 0.0
_session_sample_gpu = False

# Environment profiling is relaunched in a transient systemd slice before the
# workload starts.  The slice is the aggregate accounting boundary; its child
# scope contains the harness and Docker containers are placed alongside that
# scope by the sandbox backend.
_CGROUP_DIR_ENV = "AGENCY_PROFILE_CGROUP"
_CGROUP_PARENT_ENV = "AGENCY_PROFILE_CGROUP_PARENT"
_CGROUP_SLICE_RE = re.compile(r"^agprof-[0-9a-f]+\.slice$")

# Container cgroup registry — filled by the sandbox backends at container
# start (docker + podman)
_cg_registry: "dict[str, str]" = {}  # label (agname) -> cgroup dir
_daemon_cg: "dict[str, str]" = {}  # cgroup dir -> agg kind ("conmon"/"dockerd")
_cg_lock = threading.Lock()


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
    value = os.environ.get(_CGROUP_PARENT_ENV, "")
    return value if _CGROUP_SLICE_RE.fullmatch(value) else None


# wraps in cgroup
def _cgroup_reexec_command(slice_name: str, cgroup_dir: str) -> list[str]:
    """Build the privilege-separated systemd command used by env profiling."""
    uid = os.getuid()
    gid = os.getgid()
    user = os.environ.get("USER") or str(uid)
    home = os.environ.get("HOME") or str(Path.home())
    original_argv = list(getattr(sys, "orig_argv", ())) or [sys.executable, *sys.argv]
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


def _ensure_environment_cgroup() -> None:
    """Re-exec environment-enabled profiling inside a dedicated cgroup."""
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

    run_id = f"{os.getpid():x}{uuid.uuid4().hex[:8]}"
    slice_name = f"agprof-{run_id}.slice"
    cgroup_dir = f"/sys/fs/cgroup/agprof.slice/{slice_name}"
    command = _cgroup_reexec_command(slice_name, cgroup_dir)
    try:
        os.execvp(command[0], command)
    except OSError as e:
        raise RuntimeError(
            "agprof: unable to create the dedicated workload cgroup with "
            "sudo/systemd-run; profiling requires Linux cgroups v2 and "
            "passwordless permission to create a transient systemd scope"
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
    """Drop a removed container from the sampling registry."""
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


class _TimedSpan:
    """Composite span: kineto record_function (the trace box) + wall/CPU/runq
    deltas captured on this thread (the numbers injected as the box's args)."""

    __slots__ = (
        "_name",
        "_rf",
        "_t0",
        "_cpu0",
        "_rq0",
        "_tid",
        "_metadata",
        "_interrupted",
    )

    def __init__(self, record_function_cls, name: str) -> None:
        self._name = name
        self._rf = record_function_cls(name)
        self._metadata: dict = {}
        self._interrupted = False

    def __enter__(self) -> "_TimedSpan":
        self._t0 = time.perf_counter_ns()
        self._cpu0 = time.thread_time_ns()
        self._rq0 = _read_schedstat()
        self._tid = threading.get_native_id()
        stack = getattr(_tls, "span_stack", None)
        if stack is None:
            stack = _tls.span_stack = []
        stack.append(self)
        with _open_spans_lock:
            _open_spans[id(self)] = self
        self._rf.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._rf.__exit__(exc_type, exc_val, exc_tb)
        stack = getattr(_tls, "span_stack", None)
        if stack:
            if stack[-1] is self:
                stack.pop()
            else:
                try:
                    stack.remove(self)
                except ValueError:
                    pass
        with _open_spans_lock:
            _open_spans.pop(id(self), None)
        if self._interrupted:
            return
        t1 = time.perf_counter_ns()
        cpu1 = time.thread_time_ns()
        rq1 = _read_schedstat()
        runq = (rq1 - self._rq0) if (rq1 is not None and self._rq0 is not None) else None
        metadata = dict(self._metadata)
        metadata.setdefault("outcome", "failure" if exc_type is not None else "success")
        if exc_type is not None:
            metadata.setdefault("error_type", exc_type.__name__)
        _records.append(
            (
                self._tid,
                self._name,
                self._t0,
                t1 - self._t0,
                cpu1 - self._cpu0,
                runq,
                metadata,
            )
        )

    def annotate(self, **metadata) -> None:
        """Attach JSON-safe outcome/metric fields to this span."""
        self._metadata.update(metadata)


class _TorchSession:
    """Maps span() to a kineto record_function + thread-clock capture."""

    __slots__ = ("_record_function",)

    def __init__(self, record_function) -> None:
        self._record_function = record_function

    def span(self, name: str) -> _TimedSpan:
        return _TimedSpan(self._record_function, name)


def enabled() -> bool:
    """True while a profiling session is active."""
    return _session is not None


def span(name: str):
    """A timed, named interval on the current thread.

    No-op (a shared ``nullcontext``) unless a session is active. Nesting on the
    same thread produces parent/child spans in the trace. While active, each
    span also records its thread's CPU and run-queue time (see module doc).
    """
    s = _session
    if s is None:
        return _NULL
    return s.span(name)


def annotate(**metadata) -> None:
    """Attach fields to the innermost active span on this thread.

    This is a no-op when profiling is disabled or the current thread has no
    open span. Framework call sites use it for outcomes, token counts, TTFT,
    and other per-invocation metrics without adding work to the off path.
    """
    if _session is None:
        return
    stack = getattr(_tls, "span_stack", None)
    if stack:
        stack[-1].annotate(**metadata)


def _unpack_record(record) -> tuple:
    """Return the six clock fields plus metadata from old/new record tuples."""
    metadata = record[6] if len(record) > 6 else {}
    return (*record[:6], metadata)


_PR_SET_NAME = 15  # linux prctl option
_libc = None


def thread_name(name: str) -> None:
    """Set the OS-level thread name — what kineto records as the trace lane
    label (Python's ``Thread.name`` never reaches the pthread name). Linux
    truncates to 15 chars. No-op when profiling is off or prctl is unavailable.
    """
    if _session is None:
        return
    global _libc
    with suppress(Exception):
        if _libc is None:
            import ctypes

            _libc = ctypes.CDLL(None, use_errno=True)
        _libc.prctl(_PR_SET_NAME, name[:15].encode(), 0, 0, 0)


def _os_thread_name() -> str:
    """This thread's OS-level name (what thread_name() set), or a tid tag."""
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


class _Sampler(threading.Thread):
    """Background poller for per-PID, workload-cgroup, and GPU metrics."""

    def __init__(self, hz: float, sample_gpu: bool) -> None:
        super().__init__(daemon=True, name="agprof-sampler")
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
                self._nvml = None
        self._process_cgroup = _process_cgroup_dir()
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
                cmdline = (
                    (process_dir / "cmdline")
                    .read_bytes()
                    .replace(b"\0", b" ")
                    .decode(errors="replace")
                    .strip()
                )
            except OSError:
                cmdline = ""
            display_name = f"{comm} (PID {pid})"
            if sandbox is not None:
                display_name = f"{comm} [{sandbox}] (PID {pid})"
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
                    info["display_name"] = f"{comm} [{sandbox}] (PID {pid})"

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
        while not self._stop_ev.wait(self._interval):
            self._tick()
        if self._nvml is not None:
            with suppress(Exception):
                self._nvml.nvmlShutdown()

    def halt(self) -> None:
        self._stop_ev.set()
        self.join(timeout=2)


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
):
    """Start a profiling session. Prefer the ``session()`` context manager.

    *out_dir*: directory for the TensorBoard trace (``None`` = no trace file;
    the returned profiler object still supports ``key_averages()`` and
    ``summary_table()`` still works). *sample_hz*/*sample_gpu* control the
    background gauge sampler (0 disables it entirely).
    """
    _require_linux()
    global _session, _profiler, _out_dir, _last_summary, _last_run_summary
    global _sampler, _clock_mark_ns
    global _session_started_ns, _session_sample_hz, _session_sample_gpu
    # Imported lazily: the framework must not require torch unless profiling.
    from torch.profiler import (
        ProfilerActivity,
        profile,
        record_function,
        tensorboard_trace_handler,
    )
    from torch._C._profiler import _ExperimentalConfig

    with _state_lock:
        if _session is not None:
            raise RuntimeError("agprof: a profiling session is already active")
        handler = (
            tensorboard_trace_handler(str(out_dir), worker_name=worker_name)
            if out_dir is not None
            else None
        )
        prof = profile(
            activities=[ProfilerActivity.CPU],
            experimental_config=_ExperimentalConfig(profile_all_threads=all_threads),
            on_trace_ready=handler,
        )
        prof.start()
        _records.clear()
        _interrupted_spans.clear()
        with _open_spans_lock:
            _open_spans.clear()
        _samples.clear()
        _process_info.clear()
        _leases.clear()
        _leases_open.clear()
        _last_summary = None
        _last_run_summary = None
        _out_dir = Path(out_dir) if out_dir is not None else None
        _profiler = prof
        _session = _TorchSession(record_function)
        _session_started_ns = time.perf_counter_ns()
        _session_sample_hz = sample_hz
        _session_sample_gpu = sample_gpu
        # Clock-sync marker: pairs a perf_counter_ns stamp with a kineto event
        # so sampler/lease timestamps can be mapped onto kineto's timebase at
        # injection time (spans don't need this — they match by order).
        _clock_mark_ns = time.perf_counter_ns()
        with _session.span("agprof:clock_sync"):
            pass
        if sample_hz > 0:
            _sampler = _Sampler(sample_hz, sample_gpu)
            _sampler.start()
    return prof


def stop():
    """Stop the active session: write the trace, inject per-span CPU/wait args
    into it, and build the label-level summary. Returns the profiler, or None
    if no session was active."""
    global _session, _profiler, _out_dir, _last_summary, _last_run_summary, _sampler
    global _session_started_ns
    with _state_lock:
        if _session is None:
            return None
        prof = _profiler
        out_dir = _out_dir
        sampler = _sampler
        started_ns = _session_started_ns
        _session = None
        _profiler = None
        _out_dir = None
        _sampler = None
        _session_started_ns = None
    if sampler is not None:
        sampler.halt()
    t_end = time.perf_counter_ns()
    with _open_spans_lock:
        open_spans = list(_open_spans.values())
        _open_spans.clear()
        for open_span in open_spans:
            open_span._interrupted = True
            _interrupted_spans.append(
                {
                    "thread_id": open_span._tid,
                    "label": open_span._name,
                    "started_ns": open_span._t0,
                    "duration_ms": round(max(0, t_end - open_span._t0) / 1e6, 3),
                    **copy.deepcopy(open_span._metadata),
                    "outcome": "interrupted",
                }
            )
    # Close any lease still open at session end so it renders to the stop edge.
    with _leases_lock:
        for gpu_id, (t0, label) in _leases_open.items():
            _leases.append((gpu_id, t0, t_end, label))
        _leases_open.clear()
    prof.stop()
    records = list(_records)
    _last_summary = _build_summary(records)
    try:
        _last_run_summary = _build_run_summary(
            records,
            list(_samples),
            list(_leases),
            interrupted_spans=list(_interrupted_spans),
            started_ns=started_ns,
            ended_ns=t_end,
            sample_hz=_session_sample_hz,
            sample_gpu=_session_sample_gpu,
            gpu_sampling_available=sampler is not None and sampler._nvml is not None,
        )
    except Exception as _e:  # never let reporting kill the run
        _last_run_summary = None
        print(f"[agprof] WARNING: summary generation failed: {_e}")
    if out_dir is not None:
        try:
            _inject_trace_args(out_dir, records, _interrupted_spans)
        except Exception as _e:  # never let post-processing kill the run
            print(f"[agprof] WARNING: trace arg injection failed: {_e}")
        if _last_run_summary is not None:
            try:
                _write_summary_files(out_dir, _last_run_summary)
            except Exception as _e:  # never let reporting kill the run
                print(f"[agprof] WARNING: summary output failed: {_e}")
    return prof


def _inject_trace_args(out_dir: Path, records, interrupted_spans=()) -> None:
    """Write cpu/runqueue/blocked args onto the matching spans in the newest
    trace file under *out_dir*.

    Matching is by (tid, name, order): same-thread same-name spans are strictly
    time-ordered (same-thread spans can only nest), so pairing the k-th kineto
    event with the k-th record needs no clock-base translation between kineto
    timestamps and perf_counter_ns.
    """
    traces = sorted(out_dir.glob("*.pt.trace.json"), key=lambda p: p.stat().st_mtime)
    if not traces:
        return
    path = traces[-1]
    data = json.loads(path.read_text())

    by_key_recs: "dict[tuple[int, str], list]" = {}
    for record in records:
        tid, name, t0, wall, cpu, runq, metadata = _unpack_record(record)
        by_key_recs.setdefault((tid, name), []).append((t0, wall, cpu, runq, metadata))
    for v in by_key_recs.values():
        v.sort()

    by_key_evs: "dict[tuple[int, str], list]" = {}
    for e in data.get("traceEvents", []):
        if e.get("ph") == "X":
            k = (e.get("tid"), e.get("name"))
            if k in by_key_recs:
                by_key_evs.setdefault(k, []).append(e)

    matched = 0
    for k, evs in by_key_evs.items():
        evs.sort(key=lambda e: e["ts"])
        for e, (t0, wall, cpu, runq, metadata) in zip(evs, by_key_recs[k]):
            blocked = max(0, wall - cpu - (runq or 0))
            args = dict(e.get("args") or {})
            args.update(
                cpu_ms=round(cpu / 1e6, 3),
                runqueue_ms=(round(runq / 1e6, 3) if runq is not None else "n/a"),
                blocked_ms=round(blocked / 1e6, 3),
                cpu_pct=(round(100 * cpu / wall, 1) if wall > 0 else 0.0),
            )
            args.update(metadata)
            e["args"] = args
            matched += 1

    for interrupted in interrupted_spans:
        candidates = [
            e
            for e in data.get("traceEvents", [])
            if e.get("ph") == "X"
            and e.get("tid") == interrupted["thread_id"]
            and e.get("name") == interrupted["label"]
            and "outcome" not in (e.get("args") or {})
        ]
        if candidates:
            event = max(candidates, key=lambda e: e.get("ts", 0))
            args = dict(event.get("args") or {})
            args.update(
                {
                    key: value
                    for key, value in interrupted.items()
                    if key not in ("thread_id", "label", "started_ns", "duration_ms")
                }
            )
            args["outcome"] = "interrupted"
            event["args"] = args

    n_counters, n_leases = _inject_timelines(data)
    path.write_text(json.dumps(data))
    print(
        f"[agprof] injected: cpu/wait args on {matched} spans, "
        f"{n_counters} counter samples, {n_leases} lease intervals ({path.name})"
    )


_BYTE_KINDS = {
    "io_r": ("io_read_mb_s", "io read MB/s"),
    "io_w": ("io_write_mb_s", "io write MB/s"),
    "net_rx": ("net_receive_mb_s", "net rx MB/s"),
    "net_tx": ("net_transmit_mb_s", "net tx MB/s"),
}


def _resource_observations(samples) -> list[dict]:
    """Convert raw sampler series into gauges/rates with stable names and units."""
    observations: list[dict] = []
    prev: "dict[str, tuple[int, float]]" = {}
    for t, series, value in samples:
        parts = series.split(":")
        scope = parts[0]
        kind = parts[-1]
        identity = parts[1] if scope == "proc" and len(parts) == 3 else None
        process = _process_info.get(identity) if identity is not None else None
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


def _inject_timelines(data) -> "tuple[int, int]":
    """Append sampler counter tracks (\"ph\":\"C\") and synthetic per-GPU lease
    lanes to the trace, mapping perf_counter_ns onto kineto's timebase via the
    clock-sync marker span emitted at session start."""
    evs = data.get("traceEvents", [])
    mark = next(
        (e for e in evs if e.get("ph") == "X" and e.get("name") == "agprof:clock_sync"), None
    )
    if mark is None or _clock_mark_ns is None:
        return 0, 0
    offset_us = mark["ts"] - _clock_mark_ns / 1e3
    pid = mark.get("pid", 0)

    def to_us(t_ns: int) -> float:
        return t_ns / 1e3 + offset_us

    new: list = []
    observations = _resource_observations(_samples)
    process_identities = {
        observation["process_identity"]
        for observation in observations
        if observation.get("process_identity") is not None
    }
    for sort_index, identity in enumerate(
        sorted(
            process_identities,
            key=lambda key: _process_info.get(key, {}).get("first_seen_ns", 0),
        ),
        start=1,
    ):
        info = _process_info.get(identity)
        if info is None:
            continue
        trace_pid = info["trace_pid"]
        new.extend(
            [
                {
                    "ph": "M",
                    "pid": trace_pid,
                    "tid": 0,
                    "name": "process_name",
                    "args": {"name": info["display_name"]},
                },
                {
                    "ph": "M",
                    "pid": trace_pid,
                    "tid": 0,
                    "name": "process_sort_index",
                    "args": {"sort_index": sort_index},
                },
                {
                    "ph": "M",
                    "pid": trace_pid,
                    "tid": 0,
                    "name": "process_labels",
                    "args": {
                        "labels": f"cgroup={info['cgroup']}; cmdline={info['cmdline'] or info['comm']}"
                    },
                },
            ]
        )
    # Cumulative series become rates; direct series stay gauges. A negative
    # delta means a counter reset and is dropped by _resource_observations.
    for observation in observations:
        new.append(
            {
                "ph": "C",
                "pid": observation.get("trace_pid", pid),
                "tid": 0,
                "ts": to_us(observation["timestamp_ns"]),
                "name": observation["trace_name"],
                "args": {"value": round(observation["value"], 2)},
            }
        )
    # Lease lanes: one synthetic "thread" per device, spans labeled by acquirer.
    lease_tids = set()
    for gpu_id, t0, t1, label in _leases:
        tid = f"gpu{gpu_id}-lease"
        lease_tids.add((gpu_id, tid))
        new.append(
            {
                "ph": "X",
                "pid": pid,
                "tid": tid,
                "ts": to_us(t0),
                "dur": max(1.0, (t1 - t0) / 1e3),
                "name": f"lease:{label}",
                "cat": "gpu_lease",
            }
        )
    for gpu_id, tid in lease_tids:
        new.append(
            {
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "name": "thread_name",
                "args": {"name": f"GPU {gpu_id} lease"},
            }
        )
    evs.extend(new)
    return sum(1 for e in new if e.get("ph") == "C"), len(_leases)


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
        row["cpu_ms"] += cpu / 1e6
        rq = runq or 0
        row["runq_ms"] += rq / 1e6
        row["blocked_ms"] += max(0, wall - cpu - rq) / 1e6
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
    interrupted_spans=(),
    started_ns: "int | None",
    ended_ns: int,
    sample_hz: float,
    sample_gpu: bool,
    gpu_sampling_available: bool,
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
        outcomes = [record[6].get("outcome", "success") for record in label_records]
        wall_values = [record[3] / 1e6 for record in label_records]
        wall_ms = row["wall_ms"]
        span_rows.append(
            {
                "label": label,
                "calls": row["calls"],
                "started": row["calls"] + len(interrupted_by_label.get(label, [])),
                "succeeded": outcomes.count("success"),
                "failed": sum(outcome not in ("success", "interrupted") for outcome in outcomes),
                "interrupted": len(interrupted_by_label.get(label, [])),
                "wall_ms": round(wall_ms, 3),
                "cpu_ms": round(row["cpu_ms"], 3),
                "runqueue_ms": round(row["runq_ms"], 3),
                "blocked_ms": round(row["blocked_ms"], 3),
                "cpu_percent": round(100 * row["cpu_ms"] / wall_ms, 1) if wall_ms else 0.0,
                **_latency_stats(wall_values),
            }
        )

    grouped: "dict[str, list[dict]]" = {}
    observations = _resource_observations(samples)
    for observation in observations:
        grouped.setdefault(observation["name"], []).append(observation)
    resource_rows = []
    for name, observations in sorted(grouped.items()):
        values = [observation["value"] for observation in observations]
        resource_rows.append(
            {
                "name": name,
                "display_name": observations[0]["display_name"],
                "unit": observations[0]["unit"],
                "samples": len(values),
                "mean": round(sum(values) / len(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "last": round(values[-1], 3),
            }
        )
        totals = [
            observation["interval_total"]
            for observation in observations
            if observation["interval_total"] is not None
        ]
        if totals:
            resource_rows[-1]["total"] = round(sum(totals), 6)
            resource_rows[-1]["total_unit"] = observations[0]["total_unit"]

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
    run_outcomes = [record[6].get("outcome", "success") for record in completed_runs]
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
    generation_ms = sum(float(metadata.get("generation_ms") or 0) for metadata in attempt_metadata)
    input_tokens = sum(int(metadata.get("input_tokens") or 0) for metadata in attempt_metadata)
    output_tokens = sum(int(metadata.get("output_tokens") or 0) for metadata in attempt_metadata)
    calls = sum(record[1].startswith("llm:attempt[0]") for record in attempt_records) + sum(
        span["label"].startswith("llm:attempt[0]") for span in interrupted_attempts
    )
    retries = len(attempt_records) + len(interrupted_attempts) - calls
    llm_metrics = {
        "calls": calls,
        "successful_calls": sum(
            metadata.get("outcome", "success") == "success" for metadata in attempt_metadata
        ),
        "failed_calls": sum(
            metadata.get("outcome", "success") != "success" and not metadata.get("retrying")
            for metadata in attempt_metadata
        ),
        "interrupted_calls": len(interrupted_attempts),
        "attempts": len(attempt_records) + len(interrupted_attempts),
        "retries": retries,
        "successful_attempts": sum(
            metadata.get("outcome", "success") == "success" for metadata in attempt_metadata
        ),
        "failed_attempts": sum(
            metadata.get("outcome", "success") != "success" for metadata in attempt_metadata
        ),
        "interrupted_attempts": len(interrupted_attempts),
        "total_wait_ms": round(sum(record[3] for record in attempt_records) / 1e6, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_tokens_per_second": round(output_tokens / (generation_ms / 1e3), 3)
        if generation_ms
        else None,
        "latency": _latency_stats([record[3] / 1e6 for record in attempt_records]),
        "ttft": _latency_stats(ttft_values),
    }

    tool_records = [record for record in completed_records if record[1].startswith("tool:")]
    interrupted_tools = [span for span in interrupted if span["label"].startswith("tool:")]
    tool_outcomes = [record[6].get("outcome", "success") for record in tool_records]
    tool_metrics = {
        "started": len(tool_records) + len(interrupted_tools),
        "completed": len(tool_records),
        "succeeded": tool_outcomes.count("success"),
        "failed": sum(outcome != "success" for outcome in tool_outcomes),
        "interrupted": len(interrupted_tools),
        "latency": _latency_stats([record[3] / 1e6 for record in tool_records]),
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
        matching_outcomes = [record[6].get("outcome", "success") for record in matching]
        tool_metrics["by_tool"].append(
            {
                "name": tool_name,
                "started": len(matching) + len(matching_interrupted),
                "completed": len(matching),
                "succeeded": matching_outcomes.count("success"),
                "failed": sum(outcome != "success" for outcome in matching_outcomes),
                "interrupted": len(matching_interrupted),
                **_latency_stats([record[3] / 1e6 for record in matching]),
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
            row[f"{kind}_mb"] = metric.get("total", 0.0) if metric else 0.0
        sandbox_metrics.append(row)

    tick_times = sorted({timestamp for timestamp, _series, _value in samples})
    sampled_duration_s = (tick_times[-1] - tick_times[0]) / 1e9 if len(tick_times) >= 2 else 0.0
    return {
        "schema_version": 4,
        "data_source": "measured",
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
        },
        "span_metrics": span_rows,
        "resource_metrics": resource_rows,
        "gpu_lease_metrics": lease_rows,
        "incomplete_spans": interrupted,
    }


def summary_metrics() -> "dict | None":
    """A copy of the complete metrics document for the last completed session."""
    return copy.deepcopy(_last_run_summary)


def _markdown_escape(value) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _render_summary_markdown(summary: dict) -> str:
    """Render a complete profiler summary as a standalone Markdown report."""
    sampling = summary["sampling"]
    runs = summary["run_metrics"]
    llm = summary["llm_metrics"]
    tools = summary["tool_metrics"]

    def number(value, digits=3):
        return "n/a" if value is None else f"{value:.{digits}f}"

    def milliseconds(value):
        return "n/a" if value is None else f"{value:.3f} ms"

    lines = [
        "# agprof summary",
        "",
    ]
    if summary.get("data_source") != "measured":
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
            f"{llm['retries']} {'retry' if llm['retries'] == 1 else 'retries'}, "
            f"{llm['total_wait_ms'] / 1e3:.3f} s total wait",
            f"- Tools: **{tools['completed']}/{tools['started']} completed**, "
            f"{tools['failed']} failed, {tools['interrupted']} interrupted",
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
            f"| LLM input / output tokens | {llm['input_tokens']} / {llm['output_tokens']} |",
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
                f"{row['wall_ms'] / 1e3:.3f} | {row['cpu_ms'] / 1e3:.3f} | "
                f"{row['blocked_ms'] / 1e3:.3f} | {number(row['mean_ms'])} | "
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
    print(f"[agprof] summaries: {json_path}, {markdown_path}")
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
    for name, r in rows[:row_limit]:
        cpu_pct = 100 * r["cpu_ms"] / r["wall_ms"] if r["wall_ms"] else 0.0
        lines.append(
            f"{name:<28} {r['calls']:>5} {r['wall_ms'] / 1e3:>9.2f} {r['cpu_ms'] / 1e3:>8.2f} "
            f"{r['runq_ms'] / 1e3:>8.2f} {r['blocked_ms'] / 1e3:>9.2f} {cpu_pct:>5.1f}%"
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
):
    """Profile everything inside the block; write + post-process the trace on
    exit. Yields the torch profiler object — after the block exits,
    ``prof.key_averages().table(...)`` and ``agprof.summary_table()`` give the
    per-span summaries.
    """
    prof = start(
        out_dir,
        all_threads=all_threads,
        worker_name=worker_name,
        sample_hz=sample_hz,
        sample_gpu=sample_gpu,
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
    return os.environ.get("AGENCY_PROFILE", "").strip().lower() in ("1", "true")


def _env_out_dir() -> str:
    return os.environ.get("AGENCY_PROFILE_DIR", "agprof_trace")


@contextmanager
def workload():
    """Profile a workload boundary when environment profiling requests it.

    The context owns a session only for ``AGENCY_PROFILE_SCOPE=workload`` (the
    default) and only when no explicit session is already active. This keeps
    callers free of scope conditionals and prevents the context from stopping
    a session it did not start.
    """
    owns_session = _env_enabled() and profile_scope() == "workload" and not enabled()
    prof = start(_env_out_dir()) if owns_session else _profiler
    try:
        yield prof
    finally:
        if owns_session:
            stop()


def _maybe_autostart() -> None:
    """Start process-lifetime profiling when explicitly requested."""
    if not _env_enabled() or profile_scope() != "process":
        return
    start(_env_out_dir())
    atexit.register(stop)


def _initialize_environment_profiling() -> None:
    """Validate and isolate env-requested profiling before workload startup."""
    if not _env_enabled():
        return
    _require_linux()
    _ensure_environment_cgroup()
    _maybe_autostart()


_initialize_environment_profiling()
