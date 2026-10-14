"""Time-series resource sampling for the host, the workload and every agent.

Everything recorded here is *measured* by the kernel (procfs, sysfs, cgroup v2
files). Nothing is inferred from application traces. Each metric source is
probed once; a source that is missing or unreadable on this host is listed in
``capabilities.json`` as unavailable and simply absent from the samples, so a
missing value is never mistaken for zero.

Units are fixed and recorded in ``capabilities.json``: CPU time and pressure
stall totals in microseconds, memory and I/O in bytes, scheduler run delay in
nanoseconds, timestamps in CLOCK_MONOTONIC nanoseconds.

Layout of one sample line (``samples.jsonl``):
    {"t": mono_ns, "active": n, "host": {...}, "workload": {...},
     "api": {...}, "agents": {"<slot>": {"host": {...}, "container": {...}}}}
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")

UNITS = {
    "t": "CLOCK_MONOTONIC nanoseconds",
    "cpu.usage_usec/user_usec/system_usec": "cumulative CPU microseconds for the cgroup subtree",
    "pressure.*.some_usec/full_usec": "cumulative stall microseconds (PSI total=)",
    "pressure.*.some_avg10/full_avg10": "percent of the last 10 s stalled",
    "memory.*": "bytes (events and vmstat-style counters are counts)",
    "io.rbytes/wbytes": "bytes",
    "threads.run_delay_ns": "cumulative ns runnable-but-not-running, summed over live threads",
    "threads.runnable": "threads in state R at the sample instant",
    "host.cpu.*": "jiffies (USER_HZ) summed over the named CPU set",
    "host.schedstat.run_delay_ns": "cumulative ns tasks waited on the runqueues of the pool CPUs",
}


def _read(path) -> "str | None":
    try:
        return Path(path).read_text()
    except OSError:
        return None


def parse_flat_keyed(text: str, keys=None) -> dict:
    """'key value' per line (cpu.stat, memory.stat, memory.events, vmstat)."""
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and (keys is None or parts[0] in keys):
            try:
                values[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return values


def parse_pressure(text: str) -> dict:
    """PSI file -> {some_usec, some_avg10, full_usec, full_avg10}."""
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] not in ("some", "full"):
            continue
        fields = dict(part.split("=", 1) for part in parts[1:] if "=" in part)
        try:
            values[f"{parts[0]}_usec"] = int(fields["total"])
            values[f"{parts[0]}_avg10"] = float(fields["avg10"])
        except (KeyError, ValueError):
            continue
    return values


def parse_io_stat(text: str) -> dict:
    """Sum cgroup io.stat over devices."""
    totals = dict(rbytes=0, wbytes=0, rios=0, wios=0)
    for line in text.splitlines():
        for part in line.split()[1:]:
            key, _, value = part.partition("=")
            if key in totals:
                try:
                    totals[key] += int(value)
                except ValueError:
                    continue
    return totals


def parse_meminfo(text: str, keys) -> dict:
    values = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name in keys:
            try:
                values[name] = int(rest.split()[0]) * 1024
            except (ValueError, IndexError):
                continue
    return values


def parse_proc_stat(text: str, cpu_sets: dict) -> dict:
    """Aggregate per-CPU jiffies over named CPU sets, plus global counters."""
    fields = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
    result = {name: dict.fromkeys(fields, 0) for name in cpu_sets}
    membership = {
        cpu: [n for n, cpus in cpu_sets.items() if cpu in cpus]
        for cpus in cpu_sets.values()
        for cpu in cpus
    }
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        head = parts[0]
        if head.startswith("cpu") and head[3:].isdigit():
            for name in membership.get(int(head[3:]), ()):
                for field, value in zip(fields, parts[1:9]):
                    result[name][field] += int(value)
        elif head in ("ctxt", "processes", "procs_running", "procs_blocked"):
            result[head] = int(parts[1])
    return result


def parse_schedstat(text: str, cpus) -> "dict | None":
    """/proc/schedstat per-CPU run delay (field 8) and timeslices (field 9)."""
    cpus = set(cpus)
    delay = slices = 0
    seen = False
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 10 and parts[0].startswith("cpu") and parts[0][3:].isdigit():
            if int(parts[0][3:]) in cpus:
                delay += int(parts[8])
                slices += int(parts[9])
                seen = True
    return dict(run_delay_ns=delay, timeslices=slices) if seen else None


def parse_diskstats(text: str, device: "str | None") -> "dict | None":
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 14 and parts[2] == device:
            return dict(
                reads=int(parts[3]),
                read_bytes=int(parts[5]) * 512,
                writes=int(parts[7]),
                write_bytes=int(parts[9]) * 512,
                in_flight=int(parts[11]),
                io_ms=int(parts[12]),
                weighted_io_ms=int(parts[13]),
            )
    return None


def device_for_path(path) -> "str | None":
    """Whole-disk device name backing ``path`` (for /proc/diskstats)."""
    try:
        device = os.stat(path).st_dev
        link = Path(f"/sys/dev/block/{os.major(device)}:{os.minor(device)}").resolve()
    except OSError:
        return None
    # A partition's sysfs directory sits inside its disk's directory.
    return link.parent.name if (link / "partition").exists() else link.name


MEMORY_STAT_KEYS = frozenset(
    "anon file kernel shmem file_mapped file_dirty file_writeback swapcached "
    "inactive_anon active_anon inactive_file active_file unevictable "
    "workingset_refault_anon workingset_refault_file workingset_activate_anon "
    "workingset_activate_file pgscan pgsteal pgscan_kswapd pgscan_direct pgsteal_kswapd "
    "pgsteal_direct pgfault pgmajfault pgactivate pgdeactivate thp_fault_alloc".split()
)
VMSTAT_KEYS = frozenset(
    "pgscan_kswapd pgscan_direct pgsteal_kswapd pgsteal_direct pswpin pswpout pgfault "
    "pgmajfault oom_kill allocstall_normal allocstall_movable workingset_refault_anon "
    "workingset_refault_file pgpgin pgpgout".split()
)
MEMINFO_KEYS = frozenset(
    "MemTotal MemFree MemAvailable Buffers Cached SwapCached Active Inactive Dirty "
    "Writeback AnonPages Mapped Shmem SwapTotal SwapFree Slab SReclaimable".split()
)


def read_cgroup(path: Path, *, detail: bool = True) -> "dict | None":
    """Counters for one cgroup directory; None when it no longer exists."""
    path = Path(path)
    cpu = _read(path / "cpu.stat")
    if cpu is None:
        return None
    record = dict(cpu=parse_flat_keyed(cpu))
    pressure = {}
    for resource in ("cpu", "memory", "io"):
        text = _read(path / f"{resource}.pressure")
        if text is not None:
            pressure[resource] = parse_pressure(text)
    record["pressure"] = pressure
    memory = {}
    for name in ("memory.current", "memory.peak", "memory.swap.current", "memory.swap.peak"):
        text = _read(path / name)
        if text is not None:
            try:
                memory[name.split(".", 1)[1].replace(".", "_")] = int(text.strip())
            except ValueError:
                pass
    events = _read(path / "memory.events")
    if events is not None:
        memory["events"] = parse_flat_keyed(events)
    if detail:
        stat = _read(path / "memory.stat")
        if stat is not None:
            memory["stat"] = parse_flat_keyed(stat, MEMORY_STAT_KEYS)
    record["memory"] = memory
    io = _read(path / "io.stat")
    if io is not None:
        record["io"] = dict(source="cgroup.io.stat", **parse_io_stat(io))
    pids = _read(path / "pids.current")
    if pids is not None:
        try:
            record["pids"] = int(pids.strip())
        except ValueError:
            pass
    return record


def cgroup_threads(path: Path) -> list:
    """All thread IDs in a cgroup subtree."""
    threads = []
    path = Path(path)
    try:
        directories = [path, *[p for p in path.rglob("*") if p.is_dir()]]
    except OSError:
        return threads
    for directory in directories:
        text = _read(directory / "cgroup.threads")
        if text:
            threads.extend(int(t) for t in text.split() if t.isdigit())
    return threads


def cgroup_processes(path: Path) -> list:
    processes = []
    path = Path(path)
    try:
        directories = [path, *[p for p in path.rglob("*") if p.is_dir()]]
    except OSError:
        return processes
    for directory in directories:
        text = _read(directory / "cgroup.procs")
        if text:
            processes.extend(int(p) for p in text.split() if p.isdigit())
    return processes


def scan_threads(path: Path) -> dict:
    """Scheduler state of the threads alive in a cgroup right now.

    Cumulative fields are summed over *live* threads only: a thread that exits
    takes its counters with it, so deltas between samples are lower bounds.
    """
    result = dict(
        threads=0,
        runnable=0,
        run_ns=0,
        run_delay_ns=0,
        timeslices=0,
        voluntary_switches=0,
        involuntary_switches=0,
        unreadable=0,
    )
    for tid in cgroup_threads(path):
        stat = _read(f"/proc/{tid}/stat")
        if stat is None:
            continue  # exited between listing and reading
        result["threads"] += 1
        # The state follows the parenthesised command, which may contain spaces.
        state = stat.rsplit(")", 1)[-1].split()[:1]
        if state == ["R"]:
            result["runnable"] += 1
        schedstat = _read(f"/proc/{tid}/schedstat")
        if schedstat:
            parts = schedstat.split()
            if len(parts) >= 3:
                result["run_ns"] += int(parts[0])
                result["run_delay_ns"] += int(parts[1])
                result["timeslices"] += int(parts[2])
        status = _read(f"/proc/{tid}/status")
        if status is None:
            result["unreadable"] += 1
            continue
        for line in status.splitlines()[-4:]:
            if line.startswith("voluntary_ctxt_switches:"):
                result["voluntary_switches"] += int(line.split()[1])
            elif line.startswith("nonvoluntary_ctxt_switches:"):
                result["involuntary_switches"] += int(line.split()[1])
    return result


def scan_process_io(path: Path) -> dict:
    """Per-process I/O accounting summed over a cgroup's live processes.

    Used where the cgroup io controller is not delegated (no io.stat). Subject
    to the same live-process caveat as ``scan_threads``; processes owned by a
    different mapped UID are counted in ``unreadable``.
    """
    totals = dict(
        source="proc.pid.io",
        processes=0,
        unreadable=0,
        rchar=0,
        wchar=0,
        read_bytes=0,
        write_bytes=0,
        syscr=0,
        syscw=0,
    )
    for pid in cgroup_processes(path):
        text = _read(f"/proc/{pid}/io")
        if text is None:
            totals["unreadable"] += 1
            continue
        totals["processes"] += 1
        for line in text.splitlines():
            key, _, value = line.partition(":")
            if key in totals:
                totals[key] += int(value)
    return totals


def probe_capabilities(workload_cgroup: "Path | None", pool_cpus, disk_device) -> dict:
    """Which measurements this host actually supports."""

    def available(path):
        return _read(path) is not None

    schedstat = _read("/proc/schedstat")
    cgroup = Path(workload_cgroup) if workload_cgroup else None
    return dict(
        host_psi={r: available(f"/proc/pressure/{r}") for r in ("cpu", "memory", "io")},
        host_schedstat=bool(schedstat and parse_schedstat(schedstat, pool_cpus)),
        host_diskstats=bool(disk_device),
        disk_device=disk_device,
        thread_schedstat=available(f"/proc/{os.getpid()}/schedstat"),
        cgroup=dict(
            path=str(cgroup) if cgroup else None,
            cpu_stat=bool(cgroup and available(cgroup / "cpu.stat")),
            cpu_pressure=bool(cgroup and available(cgroup / "cpu.pressure")),
            memory_pressure=bool(cgroup and available(cgroup / "memory.pressure")),
            io_pressure=bool(cgroup and available(cgroup / "io.pressure")),
            memory_stat=bool(cgroup and available(cgroup / "memory.stat")),
            memory_events=bool(cgroup and available(cgroup / "memory.events")),
            memory_peak=bool(cgroup and available(cgroup / "memory.peak")),
            io_stat=bool(cgroup and available(cgroup / "io.stat")),
            controllers=(_read(cgroup / "cgroup.controllers") or "").split() if cgroup else [],
        ),
        per_agent_io=(
            "cgroup io.stat"
            if cgroup and available(cgroup / "io.stat")
            else "per-process /proc/<pid>/io sums over live processes (io controller not delegated)"
        ),
        not_measured=[
            "per-thread scheduling latency of threads that exited between scans",
            "kernel scheduling events (requires perf/eBPF privileges); PSI and schedstat "
            "aggregates are recorded instead",
        ],
        units=UNITS,
    )


class ResourceSampler(threading.Thread):
    """Samples on a fixed cadence until stopped. Never raises into the runner."""

    def __init__(
        self,
        output,
        *,
        interval: float,
        pool_cpus,
        reserved_cpus,
        workload_cgroup=None,
        agents=None,
        api_snapshot=None,
        active=None,
        disk_path=".",
        agent_every: int = 2,
    ) -> None:
        super().__init__(name="tb-resource-sampler", daemon=True)
        self.output = Path(output)
        self.interval = interval
        self.cpu_sets = dict(pool=set(pool_cpus), reserved=set(reserved_cpus))
        self.workload_cgroup = Path(workload_cgroup) if workload_cgroup else None
        self._agents = agents or (lambda: {})
        self._api_snapshot = api_snapshot
        self._active = active
        self.disk_device = device_for_path(disk_path)
        self.agent_every = max(1, agent_every)
        self._stop_event = threading.Event()
        self.samples = 0
        self.errors = 0
        self.overruns = 0
        self.capabilities = probe_capabilities(self.workload_cgroup, pool_cpus, self.disk_device)

    def sample(self, *, agents: bool = True) -> dict:
        record = dict(t=time.monotonic_ns())
        if self._active is not None:
            record["active"] = self._active()
        host = {}
        text = _read("/proc/stat")
        if text is not None:
            host["cpu"] = parse_proc_stat(text, self.cpu_sets)
        pressure = {}
        for resource in ("cpu", "memory", "io"):
            text = _read(f"/proc/pressure/{resource}")
            if text is not None:
                pressure[resource] = parse_pressure(text)
        host["pressure"] = pressure
        text = _read("/proc/meminfo")
        if text is not None:
            host["memory"] = parse_meminfo(text, MEMINFO_KEYS)
        text = _read("/proc/vmstat")
        if text is not None:
            host["vmstat"] = parse_flat_keyed(text, VMSTAT_KEYS)
        text = _read("/proc/schedstat")
        if text is not None:
            schedstat = parse_schedstat(text, self.cpu_sets["pool"])
            if schedstat is not None:
                host["schedstat"] = schedstat
        text = _read("/proc/diskstats")
        if text is not None and self.disk_device:
            disk = parse_diskstats(text, self.disk_device)
            if disk is not None:
                host["disk"] = disk
        text = _read("/proc/loadavg")
        if text is not None:
            host["loadavg1"] = float(text.split()[0])
        record["host"] = host
        if self.workload_cgroup is not None:
            workload = read_cgroup(self.workload_cgroup)
            if workload is not None:
                record["workload"] = workload
        if self._api_snapshot is not None:
            try:
                snapshot = self._api_snapshot()
                snapshot.pop("totals", None)
                record["api"] = snapshot
            except Exception as exc:  # a broken ledger is reported by the runner
                record["api"] = dict(error=type(exc).__name__)
        if agents:
            per_agent = {}
            for slot, cgroups in self._agents().items():
                entry = {}
                for role, path in cgroups.items():
                    if not path:
                        continue
                    counters = read_cgroup(path)
                    if counters is None:
                        continue
                    counters["threads"] = scan_threads(path)
                    if "io" not in counters:
                        counters["io"] = scan_process_io(path)
                    entry[role] = counters
                if entry:
                    per_agent[str(slot)] = entry
            record["agents"] = per_agent
        return record

    def run(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        next_tick = time.monotonic()
        with self.output.open("a", buffering=1) as stream:
            while not self._stop_event.is_set():
                try:
                    record = self.sample(agents=self.samples % self.agent_every == 0)
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                    self.samples += 1
                except Exception:
                    self.errors += 1
                next_tick += self.interval
                delay = next_tick - time.monotonic()
                if delay <= 0:
                    self.overruns += 1
                    next_tick = time.monotonic()
                    delay = 0
                self._stop_event.wait(delay)

    def stop(self) -> dict:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=10)
        return dict(
            samples=self.samples,
            errors=self.errors,
            overruns=self.overruns,
            interval_seconds=self.interval,
            agent_every=self.agent_every,
        )
