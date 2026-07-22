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

    # or, zero application changes:
    #   AGENCY_PROFILE=1 [AGENCY_PROFILE_DIR=path] python app.py

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
host thread and is attributed via lease intervals + device sampling, not thread clocks.
"""
from __future__ import annotations

import atexit
import itertools
import json
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

_NULL = nullcontext()

_session = None            # None = profiling off (the fast path checks only this)
_profiler = None           # the live torch.profiler.profile object, if any
_out_dir: "Path | None" = None
_state_lock = threading.Lock()

_counters: "dict[str, itertools.count]" = {}
_counters_lock = threading.Lock()

# Completed-span records for the CPU-vs-wait split:
# (tid, name, t0_wall_ns, wall_ns, cpu_ns, runq_ns | None).
# list.append is atomic under the GIL, so no lock on the hot path.
_records: "list[tuple[int, str, int, int, int, int | None]]" = []
_last_summary: "dict[str, dict] | None" = None

_tls = threading.local()

# Sampler timeline + GPU lease intervals (see _Sampler / gpu_lease_*).
_samples: "list[tuple[int, str, float]]" = []      # (t_mono_ns, series, value)
_sampler: "._Sampler | None" = None
_leases_open: "dict[int, tuple[int, str]]" = {}    # gpu_id -> (t0_ns, label)
_leases: "list[tuple[int, int, int, str]]" = []    # (gpu_id, t0_ns, t1_ns, label)
_leases_lock = threading.Lock()
_clock_mark_ns: "int | None" = None


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

    __slots__ = ("_name", "_rf", "_t0", "_cpu0", "_rq0")

    def __init__(self, record_function_cls, name: str) -> None:
        self._name = name
        self._rf = record_function_cls(name)

    def __enter__(self) -> "_TimedSpan":
        self._t0 = time.perf_counter_ns()
        self._cpu0 = time.thread_time_ns()
        self._rq0 = _read_schedstat()
        self._rf.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._rf.__exit__(exc_type, exc_val, exc_tb)
        t1 = time.perf_counter_ns()
        cpu1 = time.thread_time_ns()
        rq1 = _read_schedstat()
        runq = (rq1 - self._rq0) if (rq1 is not None and self._rq0 is not None) else None
        _records.append(
            (
                threading.get_native_id(),
                self._name,
                self._t0,
                t1 - self._t0,
                cpu1 - self._cpu0,
                runq,
            )
        )


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
    try:
        if _libc is None:
            import ctypes

            _libc = ctypes.CDLL(None, use_errno=True)
        _libc.prctl(_PR_SET_NAME, name[:15].encode(), 0, 0, 0)
    except Exception:
        pass


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
    """Background gauge poller: GPU util/VRAM/power per device (NVML) and host
    process CPU/RSS (/proc), appended to the _samples timeline. Never touches
    the span hot path; one batch of reads per tick (~1 ms with NVML)."""

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
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                    for i in range(pynvml.nvmlDeviceGetCount())
                ]
            except Exception:
                self._nvml = None
        self._page_mb = 4096 / 2**20
        try:
            self._page_mb = os.sysconf("SC_PAGE_SIZE") / 2**20
            self._clk_tck = os.sysconf("SC_CLK_TCK")
        except (ValueError, OSError):
            self._clk_tck = 100
        # Per-container cgroup accounting (rootless podman, cgroup v2 systemd
        # layout). Scopes appear/disappear with container incarnations; they
        # are re-discovered every tick by listing the user slice. Container
        # id -> name comes from podman's storage db (no CLI round-trips).
        uid = os.getuid()
        self._cg_base = Path(
            f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/user.slice"
        )
        self._cg_db = Path.home() / ".local/share/containers/storage/overlay-containers/containers.json"
        self._cg_labels: "dict[str, str]" = {}   # container full id -> label
        self._cg_db_mtime = -1.0
        self._pid_labels: "dict[int, str]" = {}  # pid -> label (container or comm)
        self._proc_util_last: "dict[int, int]" = {}  # gpu idx -> last NVML sample ts

    def _cg_label(self, cid: str) -> str:
        """Label for a container id: the agname part of 'sandbox-<run>-<agname>'."""
        label = self._cg_labels.get(cid)
        if label is not None:
            return label
        try:
            mtime = self._cg_db.stat().st_mtime
            if mtime != self._cg_db_mtime:
                self._cg_db_mtime = mtime
                db = json.loads(self._cg_db.read_text())
                for c in db:
                    names = c.get("names") or []
                    name = names[0] if names else c["id"][:12]
                    if name.startswith("sandbox-"):
                        # "sandbox-<runid>-sandbox_<agname>_<dedup>" -> "<agname>"
                        # (agSandbox allocates its own agname as sandbox_{agname}
                        # plus a 4-char dedup suffix — agsandbox.py __init__).
                        name = name.split("-", 2)[-1]
                        name = name.removeprefix("sandbox_")
                        base, _, suffix = name.rpartition("_")
                        if base and len(suffix) == 4:
                            name = base
                    self._cg_labels[c["id"]] = name
        except Exception:
            pass
        return self._cg_labels.get(cid, cid[:12])

    def _pid_label(self, pid: int) -> str:
        """Label for a GPU-using PID: its container's agname when the PID lives
        in a libpod cgroup (same id space as the CPU collector), else the
        process comm name (e.g. vLLM's server process)."""
        label = self._pid_labels.get(pid)
        if label is not None:
            return label
        label = f"pid{pid}"
        try:
            cg = Path(f"/proc/{pid}/cgroup").read_text()
            idx = cg.find("libpod-")
            if idx != -1 and not cg[idx:].startswith("libpod-conmon"):
                cid = cg[idx + 7 : idx + 7 + 64]
                label = self._cg_label(cid)
            else:
                label = Path(f"/proc/{pid}/comm").read_text().strip() or label
        except Exception:
            pass
        # comm names may contain ':' (e.g. "VLLM::EngineCor") — keep series
        # names parseable as gpu{i}:{label}:{metric}.
        label = label.replace(":", "_")
        self._pid_labels[pid] = label
        return label

    def _tick_gpu_procs(self, t: int, i: int, h) -> None:
        """Per-PID GPU accounting for device *i*: VRAM per process (compute
        procs list) and SM utilization per process (NVML's sample buffer since
        the previous tick). This is what splits a device-wide curve into
        per-agent/per-process series under concurrency — cgroups can't meter
        GPUs, so this is the container-attribution path for the device."""
        try:
            for pr in self._nvml.nvmlDeviceGetComputeRunningProcesses(h):
                if pr.usedGpuMemory:
                    _samples.append(
                        (t, f"gpu{i}:{self._pid_label(pr.pid)}:mem_mb",
                         pr.usedGpuMemory / 2**20)
                    )
        except Exception:
            pass
        try:
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
                _samples.append(
                    (t, f"gpu{i}:{self._pid_label(pid)}:util_pct",
                     sum(utils) / len(utils))
                )
        except Exception:
            pass  # NVMLError_NotFound when no samples since `last` — normal

    def _tick_cgroups(self, t: int) -> None:
        if not self._cg_base.is_dir():
            return
        conmon_cpu_us = 0
        saw_conmon = False
        try:
            entries = list(os.scandir(self._cg_base))
        except OSError:
            return
        for entry in entries:
            n = entry.name
            if not (n.startswith("libpod-") and n.endswith(".scope")):
                continue
            cid = n[len("libpod-"):-len(".scope")]
            if cid.startswith("conmon-"):
                # Container-runtime helper processes: aggregate as daemon cost.
                try:
                    with open(f"{entry.path}/cpu.stat", "rb") as f:
                        conmon_cpu_us += int(f.readline().split()[1])
                    saw_conmon = True
                except Exception:
                    pass
                continue
            label = self._cg_label(cid)
            try:
                with open(f"{entry.path}/cpu.stat", "rb") as f:
                    cpu_us = int(f.readline().split()[1])  # usage_usec
                _samples.append((t, f"cg:{label}:cpu_us", float(cpu_us)))
                with open(f"{entry.path}/memory.current", "rb") as f:
                    _samples.append((t, f"sandbox:{label}:mem_mb", int(f.read()) / 2**20))
            except Exception:
                pass  # scope vanished mid-read (container stopped) — skip
            self._tick_io_net(t, label, entry.path)
        if saw_conmon:
            _samples.append((t, "cg:conmon:cpu_us", float(conmon_cpu_us)))

    def _tick_io_net(self, t: int, label: str, scope_path: str) -> None:
        """Disk IO + network for one container, without the io controller.

        Rootless user slices typically don't get the `io` cgroup controller
        delegated, so instead: the scope's cgroup.procs lists the container's
        PIDs; /proc/<pid>/io (summed) gives cumulative disk bytes, and any one
        PID's /proc/<pid>/net/dev gives the container's network-namespace
        counters (the netns is per-container and stable across its PIDs).
        Caveats: per-PID io of processes under other subuids is unreadable and
        skipped; bytes of processes that exited between ticks are lost; the
        summed series can step down when a PID exits (negative deltas are
        dropped at injection).
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
        io_r = io_w = 0
        io_seen = False
        for pid in pids:
            try:
                with open(f"/proc/{pid}/io", "rb") as f:
                    for line in f:
                        if line.startswith(b"read_bytes:"):
                            io_r += int(line.split()[1])
                        elif line.startswith(b"write_bytes:"):
                            io_w += int(line.split()[1])
                io_seen = True
            except Exception:
                continue  # subuid-owned or exited — skip
        if io_seen:
            _samples.append((t, f"cg:{label}:io_r", float(io_r)))
            _samples.append((t, f"cg:{label}:io_w", float(io_w)))
        for pid in pids:
            try:
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
                break  # one PID suffices — netns counters are container-wide
            except Exception:
                continue

    def _tick(self) -> None:
        t = time.perf_counter_ns()
        try:
            with open("/proc/self/stat", "rb") as f:
                fields = f.read().rsplit(b") ", 1)[-1].split()
            cpu_s = (int(fields[11]) + int(fields[12])) / self._clk_tck
            _samples.append((t, "host:cpu_s", cpu_s))  # cumulative; %-ified at injection
            with open("/proc/self/statm", "rb") as f:
                rss_mb = int(f.read().split()[1]) * self._page_mb
            _samples.append((t, "host:rss_mb", rss_mb))
            # This process's own disk IO
            with open("/proc/self/io", "rb") as f:
                for line in f:
                    if line.startswith(b"read_bytes:"):
                        _samples.append((t, "host:io_r", float(line.split()[1])))
                    elif line.startswith(b"write_bytes:"):
                        _samples.append((t, "host:io_w", float(line.split()[1])))
            # Host-netns interface totals — SYSTEM scope: includes every
            # process on the box (vLLM traffic, ssh, ...), not just ours.
            with open("/proc/net/dev", "rb") as f:
                rx = tx = 0
                for line in f.readlines()[2:]:
                    iface, _, rest = line.partition(b":")
                    if iface.strip() == b"lo":
                        continue
                    parts = rest.split()
                    rx += int(parts[0])
                    tx += int(parts[8])
            _samples.append((t, "host:net_rx", float(rx)))
            _samples.append((t, "host:net_tx", float(tx)))
        except Exception:
            pass
        if self._nvml is not None:
            for i, h in enumerate(self._handles):
                try:
                    u = self._nvml.nvmlDeviceGetUtilizationRates(h)
                    m = self._nvml.nvmlDeviceGetMemoryInfo(h)
                    p = self._nvml.nvmlDeviceGetPowerUsage(h)
                    _samples.append((t, f"gpu{i}:util_pct", float(u.gpu)))
                    _samples.append((t, f"gpu{i}:mem_mb", m.used / 2**20))
                    _samples.append((t, f"gpu{i}:power_w", p / 1000.0))
                    self._tick_gpu_procs(t, i, h)
                except Exception:
                    pass
        self._tick_cgroups(t)

    def run(self) -> None:
        while not self._stop_ev.wait(self._interval):
            self._tick()
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

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
    global _session, _profiler, _out_dir, _last_summary, _sampler, _clock_mark_ns
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
        _samples.clear()
        _leases.clear()
        _leases_open.clear()
        _last_summary = None
        _out_dir = Path(out_dir) if out_dir is not None else None
        _profiler = prof
        _session = _TorchSession(record_function)
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
    global _session, _profiler, _out_dir, _last_summary, _sampler
    with _state_lock:
        if _session is None:
            return None
        prof = _profiler
        out_dir = _out_dir
        sampler = _sampler
        _session = None
        _profiler = None
        _out_dir = None
        _sampler = None
    if sampler is not None:
        sampler.halt()
    # Close any lease still open at session end so it renders to the stop edge.
    with _leases_lock:
        t_end = time.perf_counter_ns()
        for gpu_id, (t0, label) in _leases_open.items():
            _leases.append((gpu_id, t0, t_end, label))
        _leases_open.clear()
    prof.stop()
    records = list(_records)
    _last_summary = _build_summary(records)
    if out_dir is not None:
        try:
            _inject_trace_args(out_dir, records)
        except Exception as _e:  # never let post-processing kill the run
            print(f"[agprof] WARNING: trace arg injection failed: {_e}")
    return prof


def _inject_trace_args(out_dir: Path, records) -> None:
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
    for tid, name, t0, wall, cpu, runq in records:
        by_key_recs.setdefault((tid, name), []).append((t0, wall, cpu, runq))
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
        for e, (t0, wall, cpu, runq) in zip(evs, by_key_recs[k]):
            blocked = max(0, wall - cpu - (runq or 0))
            args = dict(e.get("args") or {})
            args.update(
                cpu_ms=round(cpu / 1e6, 3),
                runqueue_ms=(round(runq / 1e6, 3) if runq is not None else "n/a"),
                blocked_ms=round(blocked / 1e6, 3),
                cpu_pct=(round(100 * cpu / wall, 1) if wall > 0 else 0.0),
            )
            e["args"] = args
            matched += 1

    n_counters, n_leases = _inject_timelines(data)
    path.write_text(json.dumps(data))
    print(
        f"[agprof] injected: cpu/wait args on {matched} spans, "
        f"{n_counters} counter samples, {n_leases} lease intervals ({path.name})"
    )


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
    # Counters. Cumulative series are emitted as rates over each sample
    # interval ("cg:*" CPU -> %, byte counters -> MB/s); a negative delta
    # means the counter reset (new container incarnation under the same
    # label, or a PID exited from a summed series) and that interval is
    # skipped. Everything else is a direct gauge.
    _BYTE_KINDS = {"io_r": "io read MB/s", "io_w": "io write MB/s",
                   "net_rx": "net rx MB/s", "net_tx": "net tx MB/s"}
    prev: "dict[str, tuple[int, float]]" = {}
    for t, series, value in _samples:
        kind = series.rsplit(":", 1)[-1] if series.startswith(("cg:", "host:")) else None
        if kind in ("cpu_us", "cpu_s") or kind in _BYTE_KINDS:
            p = prev.get(series)
            prev[series] = (t, value)
            if p is None or t <= p[0]:
                continue
            delta = value - p[1]
            if delta < 0:
                continue  # incarnation reset / PID exit
            dt_s = (t - p[0]) / 1e9
            label = series.split(":")[1]
            if kind in ("cpu_us", "cpu_s"):
                d_cpu_s = delta * (1e-6 if kind == "cpu_us" else 1.0)
                rate = 100.0 * d_cpu_s / dt_s
                if series == "host:cpu_s":
                    cname = "host cpu %"
                elif label == "conmon":
                    cname = "conmon cpu %"
                else:
                    cname = f"sandbox:{label}:cpu_pct"
            elif series.startswith("host:"):
                rate = delta / 2**20 / dt_s
                cname = f"host {_BYTE_KINDS[kind]}"
            else:
                rate = delta / 2**20 / dt_s
                cname = f"sandbox:{label}:{_BYTE_KINDS[kind]}"
            new.append(
                {"ph": "C", "pid": pid, "tid": 0, "ts": to_us(t),
                 "name": cname, "args": {"value": round(rate, 2)}}
            )
        else:
            new.append(
                {"ph": "C", "pid": pid, "tid": 0, "ts": to_us(t),
                 "name": series, "args": {"value": round(value, 1)}}
            )
    # Lease lanes: one synthetic "thread" per device, spans labeled by acquirer.
    lease_tids = set()
    for gpu_id, t0, t1, label in _leases:
        tid = f"gpu{gpu_id}-lease"
        lease_tids.add((gpu_id, tid))
        new.append(
            {"ph": "X", "pid": pid, "tid": tid, "ts": to_us(t0),
             "dur": max(1.0, (t1 - t0) / 1e3), "name": f"lease:{label}",
             "cat": "gpu_lease"}
        )
    for gpu_id, tid in lease_tids:
        new.append(
            {"ph": "M", "pid": pid, "tid": tid, "name": "thread_name",
             "args": {"name": f"GPU {gpu_id} lease"}}
        )
    evs.extend(new)
    return sum(1 for e in new if e.get("ph") == "C"), len(_leases)


def _build_summary(records) -> "dict[str, dict]":
    """Aggregate records per label: calls, wall/cpu/runq/blocked totals (ms)."""
    out: "dict[str, dict]" = {}
    for _tid, name, _t0, wall, cpu, runq in records:
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
            f"{name:<28} {r['calls']:>5} {r['wall_ms']/1e3:>9.2f} {r['cpu_ms']/1e3:>8.2f} "
            f"{r['runq_ms']/1e3:>8.2f} {r['blocked_ms']/1e3:>9.2f} {cpu_pct:>5.1f}%"
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


def _maybe_autostart() -> None:
    """AGENCY_PROFILE=1 profiles an unmodified application for its whole life."""
    if os.environ.get("AGENCY_PROFILE", "").lower() not in ("1", "true"):
        return
    out_dir = os.environ.get("AGENCY_PROFILE_DIR", "agprof_trace")
    start(out_dir)
    atexit.register(stop)


_maybe_autostart()
