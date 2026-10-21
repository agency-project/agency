"""System context for agents: what the machine was doing while a tool call ran.

No background sampler. Every source is a cumulative counter, so a window is the difference
between two snapshots, taken when a tool call is admitted and when it completes
(HostInteractionServer), or on demand by the ``get_system_state`` host tool:

    cgroup v2   <sandbox cgroup>/cpu.pressure  some/full ``total`` (us stalled)
                <sandbox cgroup>/cpu.stat      usage_usec, nr_throttled, throttled_usec
                <sandbox cgroup>/cgroup.procs  the sandbox's processes (host pids)
    host        /proc/stat cpuN lines for the sandbox's effective cpuset: busy / steal jiffies;
                busy minus the sandbox's own usage = load from other tenants
    ptrace      execve events Agency already traces (namespaced pid -> argv), so a process
                still running can be labelled as one the agent started

Facts only: rendered text reports numbers and never judges whether a measurement is reliable.

Config: ``agconfig.sysinfo`` (sysinfoconfig). ``record`` logs one ``sysinfo_window`` event per
tool call; ``expose`` decides what the agent sees ("none", "tool": the ``get_system_state``
host MCP tool, "context": the call's window is rendered and appended to that tool's result --
HostInteractionServer.complete_tool_call returns it, the native harness appends it, and the CLI
harnesses' PostToolUse hook hands it over as ``additionalContext`` -- "both").
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field

_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_RING = 512
_PROC = "/proc"  # tests point this at a fake procfs
# A process whose command line contains one of these is Agency's own machinery or the agent's
# harness, not something the agent ran; it is never listed as "your processes".
HARNESS_MARKERS = (
    "agency_harness",
    "agency.harness",
    "native_harness",
    "/opt/agency",
    "agency-daemon",
    "@anthropic-ai/claude-code",
    "/claude ",
    "codex",
    "sleep infinity",
    "tail -f /dev/null",
)


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcSample:
    host_pid: int
    ns_pid: int
    comm: str
    cmdline: str
    cpu_ticks: int


@dataclass(frozen=True)
class SysSnapshot:
    ts_ns: int
    psi_some_total_us: int
    psi_full_total_us: int
    cg_usage_usec: int
    nr_throttled: int
    throttled_usec: int
    cpu_jiffies: "dict[int, tuple[int, int, int]]"  # cpu -> (total, idle+iowait, steal)
    procs: "dict[int, ProcSample]"  # host pid -> sample


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _psi_totals(text: str) -> "tuple[int, int]":
    some = full = 0
    for line in text.splitlines():
        parts = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
        if line.startswith("some"):
            some = int(parts.get("total", 0))
        elif line.startswith("full"):
            full = int(parts.get("total", 0))
    return some, full


def _kv(text: str) -> "dict[str, int]":
    out = {}
    for line in text.splitlines():
        k, _, v = line.partition(" ")
        if v.strip().lstrip("-").isdigit():
            out[k] = int(v)
    return out


def parse_cpuset(text: str) -> "tuple[int, ...]":
    """'2-3,10-11' -> (2, 3, 10, 11)."""
    cpus: "list[int]" = []
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return tuple(cpus)


def _cpu_jiffies(proc_stat: str, cpus: "tuple[int, ...]") -> "dict[int, tuple[int, int, int]]":
    wanted = set(cpus)
    out = {}
    for line in proc_stat.splitlines():
        if not line.startswith("cpu") or line.startswith("cpu "):
            continue
        name, *vals = line.split()
        cpu = int(name[3:])
        if cpu not in wanted:
            continue
        v = [int(x) for x in vals] + [0] * 10
        user, nice, system, idle, iowait, irq, softirq, steal = v[:8]
        total = user + nice + system + idle + iowait + irq + softirq + steal
        out[cpu] = (total, idle + iowait, steal)
    return out


def _cgroup_pids(cgroup_dir: str) -> "list[int]":
    pids: "list[int]" = []
    for root, _dirs, files in os.walk(cgroup_dir):
        if "cgroup.procs" in files:
            pids.extend(int(x) for x in _read(os.path.join(root, "cgroup.procs")).split() if x)
    return pids


def _proc_sample(host_pid: int) -> "ProcSample | None":
    stat = _read(f"{_PROC}/{host_pid}/stat")
    if not stat:
        return None
    # comm may contain spaces; it is the text between the first '(' and the last ')'
    lpar, rpar = stat.find("("), stat.rfind(")")
    comm = stat[lpar + 1 : rpar]
    rest = stat[rpar + 2 :].split()
    ticks = int(rest[11]) + int(rest[12])  # utime + stime (fields 14, 15)
    ns_pid = host_pid
    for line in _read(f"{_PROC}/{host_pid}/status").splitlines():
        if line.startswith("NSpid:"):
            ns_pid = int(line.split()[-1])
            break
    cmdline = _read(f"{_PROC}/{host_pid}/cmdline").replace("\0", " ").strip()
    return ProcSample(host_pid, ns_pid, comm, cmdline[:200], ticks)


def snapshot(cgroup_dir: str, cpus: "tuple[int, ...]", with_procs: bool = True) -> SysSnapshot:
    """Read the counters listed in the module docstring: a few small file reads."""
    some, full = _psi_totals(_read(os.path.join(cgroup_dir, "cpu.pressure")))
    st = _kv(_read(os.path.join(cgroup_dir, "cpu.stat")))
    procs = {}
    if with_procs:
        for pid in _cgroup_pids(cgroup_dir):
            sample = _proc_sample(pid)
            if sample is not None:
                procs[pid] = sample
    return SysSnapshot(
        ts_ns=time.time_ns(),
        psi_some_total_us=some,
        psi_full_total_us=full,
        cg_usage_usec=st.get("usage_usec", 0),
        nr_throttled=st.get("nr_throttled", 0),
        throttled_usec=st.get("throttled_usec", 0),
        cpu_jiffies=_cpu_jiffies(_read(f"{_PROC}/stat"), cpus),
        procs=procs,
    )


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SysWindow:
    t0_ns: int
    t1_ns: int
    cpus: "tuple[int, ...]"
    busy_pct: float  # all users, over the cpuset
    own_pct: float  # this sandbox's share of the cpuset
    other_tenants_pct: float  # busy share not from this sandbox
    steal_pct: float
    psi_some_pct: float  # share of the window some sandbox task was runnable but waiting
    psi_full_pct: float
    throttled: int
    throttled_ms: float
    live_procs: "tuple[dict, ...]" = field(default=())  # still running at t1, not the harness

    @property
    def seconds(self) -> float:
        return (self.t1_ns - self.t0_ns) / 1e9

    def to_event(self) -> dict:
        d = asdict(self)
        d["cpus"] = list(self.cpus)
        d["live_procs"] = list(self.live_procs)
        d["seconds"] = round(self.seconds, 3)
        return d


def _is_harness(cmdline: str) -> bool:
    padded = cmdline + " "
    return any(m in padded for m in HARNESS_MARKERS)


def window(
    a: SysSnapshot,
    b: SysSnapshot,
    cpus: "tuple[int, ...]",
    execs: "dict[int, dict] | None" = None,
    max_procs: int = 5,
) -> SysWindow:
    """Difference two snapshots into percentages over [a.ts_ns, b.ts_ns]."""
    dt_s = max((b.ts_ns - a.ts_ns) / 1e9, 1e-6)
    tot = idle = steal = 0
    for cpu in cpus:
        if cpu in a.cpu_jiffies and cpu in b.cpu_jiffies:
            ta, ia, sa = a.cpu_jiffies[cpu]
            tb, ib, sb = b.cpu_jiffies[cpu]
            tot += tb - ta
            idle += ib - ia
            steal += sb - sa
    busy_pct = 100.0 * (tot - idle - steal) / tot if tot > 0 else 0.0
    steal_pct = 100.0 * steal / tot if tot > 0 else 0.0
    n = max(len(cpus), 1)
    own_pct = min(100.0, 100.0 * (b.cg_usage_usec - a.cg_usage_usec) / (dt_s * 1e6 * n))
    procs = []
    execs = execs or {}
    for pid, pb in b.procs.items():
        if _is_harness(pb.cmdline):
            continue
        pa = a.procs.get(pid)
        cpu_pct = 100.0 * (pb.cpu_ticks - pa.cpu_ticks) / (_CLK_TCK * dt_s) if pa else None
        started = execs.get(pb.ns_pid)
        procs.append(
            {
                "pid": pb.ns_pid,
                "command": (pb.cmdline or pb.comm)[:80],
                "cpu_pct": None if cpu_pct is None else round(cpu_pct, 1),
                "started_by_agent": started is not None,
                "started_ns": started["ts_ns"] if started else None,
            }
        )
    procs.sort(key=lambda p: -(p["cpu_pct"] or 0.0))
    return SysWindow(
        t0_ns=a.ts_ns,
        t1_ns=b.ts_ns,
        cpus=tuple(cpus),
        busy_pct=round(busy_pct, 1),
        own_pct=round(own_pct, 1),
        other_tenants_pct=round(max(0.0, busy_pct - own_pct), 1),
        steal_pct=round(steal_pct, 1),
        psi_some_pct=round(
            min(100.0, 100.0 * (b.psi_some_total_us - a.psi_some_total_us) / (dt_s * 1e6)), 1
        ),
        psi_full_pct=round(
            min(100.0, 100.0 * (b.psi_full_total_us - a.psi_full_total_us) / (dt_s * 1e6)), 1
        ),
        throttled=b.nr_throttled - a.nr_throttled,
        throttled_ms=round((b.throttled_usec - a.throttled_usec) / 1000.0, 1),
        live_procs=tuple(procs[:max_procs]),
    )


def _cpu_list(cpus: "tuple[int, ...]") -> str:
    return ",".join(str(c) for c in cpus)


def render(w: SysWindow, what: str = "this command") -> str:
    """Short fact block, no interpretation."""
    lines = [
        f"[system] during {what} ({w.seconds:.1f} s) on CPUs {_cpu_list(w.cpus)}: "
        f"busy {w.busy_pct:.0f}% (your sandbox {w.own_pct:.0f}%, other tenants "
        f"{w.other_tenants_pct:.0f}%), CPU pressure {w.psi_some_pct:.0f}% "
        f"(full {w.psi_full_pct:.0f}%), throttled {w.throttled} times, steal {w.steal_pct:.0f}%."
    ]
    if w.live_procs:
        shown = ", ".join(
            f"{p['command']} (pid {p['pid']}"
            + (f", {p['cpu_pct']:.0f}% CPU" if p["cpu_pct"] is not None else "")
            + (", started by you" if p["started_by_agent"] else "")
            + ")"
            for p in w.live_procs
        )
        lines.append(f"Processes still running in your sandbox: {shown}")
    else:
        lines.append("Processes still running in your sandbox: none besides the agent harness.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# per-sandbox tracker
# ---------------------------------------------------------------------------


class SysInfoTracker:
    """Per-sandbox host state: a snapshot when a tool call is admitted, a window when it
    completes, the execs the agent made, and a short ring of snapshots for
    ``get_system_state(seconds=...)``."""

    def __init__(
        self,
        cgroup_dir_fn,
        record: bool = True,
        expose: str = "none",
        max_procs: int = 5,
    ) -> None:
        self._cgroup_dir_fn = cgroup_dir_fn
        self.record = record
        self.expose = expose
        self.max_procs = max_procs
        self._lock = threading.Lock()
        self._open: "dict[str, tuple[SysSnapshot, str]]" = {}
        self._ring: "deque[SysSnapshot]" = deque(maxlen=_RING)
        self._execs: "dict[int, dict]" = {}

    def _dir_and_cpus(self) -> "tuple[str, tuple[int, ...]]":
        cg = self._cgroup_dir_fn()
        cpus = parse_cpuset(_read(os.path.join(cg, "cpuset.cpus.effective")))
        if not cpus:
            cpus = tuple(range(os.cpu_count() or 1))
        return cg, cpus

    def _snap(self) -> "tuple[SysSnapshot, tuple[int, ...]]":
        cg, cpus = self._dir_and_cpus()
        s = snapshot(cg, cpus)
        with self._lock:
            self._ring.append(s)
        return s, cpus

    def note_exec(self, ns_pid: int, argv: "list[str] | None") -> None:
        with self._lock:
            self._execs[ns_pid] = {"argv": list(argv or [])[:6], "ts_ns": time.time_ns()}

    def tool_started(self, call_id: str, tool_name: str = "") -> None:
        s, _ = self._snap()
        with self._lock:
            self._open[call_id] = (s, tool_name)

    def tool_completed(self, call_id: str) -> "tuple[SysWindow, str] | None":
        """(window, tool name) for an admitted call, or None if it was never started."""
        with self._lock:
            opened = self._open.pop(call_id, None)
        if opened is None:
            return None
        a, tool_name = opened
        b, cpus = self._snap()
        with self._lock:
            execs = dict(self._execs)
        return window(a, b, cpus, execs, self.max_procs), tool_name

    def recent(self, seconds: float) -> "SysWindow":
        """A window ending now, as close to ``seconds`` long as the snapshots allow."""
        b, cpus = self._snap()
        target = b.ts_ns - int(seconds * 1e9)
        with self._lock:
            older = [s for s in self._ring if s.ts_ns <= target]
            a = older[-1] if older else (self._ring[0] if len(self._ring) > 1 else None)
            execs = dict(self._execs)
        if a is None or a is b:
            # nothing to difference against yet: measure the next second
            time.sleep(1.0)
            a, b = b, self._snap()[0]
        return window(a, b, cpus, execs, self.max_procs)


# ---------------------------------------------------------------------------
# the pull tool
# ---------------------------------------------------------------------------

TOOL_NAME = "get_system_state"
DEFAULT_TOOL_WINDOW_S = 30.0
MAX_TOOL_WINDOW_S = 600.0


def system_state_tool():
    """The ``get_system_state`` host MCP tool (served when expose is "tool" or "both")."""
    from ..agdata import agdata, agerror
    from ..agtool import agtool

    def _get_system_state(arg, sandbox=None):
        tracker = sandbox.sysinfo() if sandbox is not None else None
        if tracker is None:
            return agerror("system state is not available for this sandbox")
        try:
            seconds = float(arg.get("seconds") or DEFAULT_TOOL_WINDOW_S)
        except (AttributeError, TypeError, ValueError):
            seconds = DEFAULT_TOOL_WINDOW_S
        seconds = min(max(seconds, 1.0), MAX_TOOL_WINDOW_S)
        w = tracker.recent(seconds)
        return agdata(state=render(w, what=f"the last {w.seconds:.0f} s"))

    return agtool(
        name=TOOL_NAME,
        description=(
            "Report what the machine running your sandbox was doing over a recent time window: "
            "how busy your CPUs were and how much of that came from other tenants, CPU "
            "pressure, throttling, steal, and processes still running in your sandbox."
        ),
        fn=_get_system_state,
        params={
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": (
                        f"window length ending now (default {DEFAULT_TOOL_WINDOW_S:g}, "
                        f"max {MAX_TOOL_WINDOW_S:g})"
                    ),
                }
            },
            "required": [],
        },
    )
