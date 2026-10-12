"""agency.observability.sysinfo: counter snapshots, windows, the tracker, and its wiring into
HostInteractionServer (sysinfo_window events) and HostMcpServer (get_system_state)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from agency.agskill import agskill
from agency.configs.agconfig import agconfig, sysinfoconfig
from agency.engine.host_servers.host_interaction_server import HostInteractionServer
from agency.engine.host_servers.host_mcp_server import HostMcpServer
from agency.harness._syscall_event import agsyscallevent
from agency.observability import sysinfo
from agency.observability.sysinfo import SysInfoTracker, parse_cpuset, render, snapshot, window

CPUS = (2, 3)


class FakeHost:
    """A cgroup dir and a procfs under tmp_path whose counters the test advances."""

    def __init__(self, tmp_path, monkeypatch):
        self.cg = tmp_path / "cg"
        self.proc = tmp_path / "proc"
        self.cg.mkdir()
        self.proc.mkdir()
        monkeypatch.setattr(sysinfo, "_PROC", str(self.proc))
        (self.cg / "cpuset.cpus.effective").write_text("2-3\n")
        self.jiffies = {2: [0] * 8, 3: [0] * 8}  # user nice system idle iowait irq softirq steal
        self.psi_some = self.psi_full = 0
        self.usage_usec = self.nr_throttled = self.throttled_usec = 0
        self.procs: "dict[int, dict]" = {}
        self.write()

    def add_proc(self, host_pid, ns_pid, cmdline, ticks=0):
        self.procs[host_pid] = {"ns": ns_pid, "cmd": cmdline, "ticks": ticks}

    def advance(self, seconds, own_cpu_s, other_cpu_s, steal_s=0.0, psi_some_s=0.0):
        """Spread own + other busy time evenly over the two CPUs (100 jiffies/s each)."""
        per_cpu = 100 * seconds
        for cpu in CPUS:
            busy = int(100 * (own_cpu_s + other_cpu_s) / len(CPUS))
            steal = int(100 * steal_s / len(CPUS))
            self.jiffies[cpu][0] += busy
            self.jiffies[cpu][7] += steal
            self.jiffies[cpu][3] += per_cpu - busy - steal
        self.usage_usec += int(own_cpu_s * 1e6)
        self.psi_some += int(psi_some_s * 1e6)
        self.write()

    def write(self):
        (self.cg / "cpu.pressure").write_text(
            f"some avg10=0.00 avg60=0.00 avg300=0.00 total={self.psi_some}\n"
            f"full avg10=0.00 avg60=0.00 avg300=0.00 total={self.psi_full}\n"
        )
        (self.cg / "cpu.stat").write_text(
            f"usage_usec {self.usage_usec}\nuser_usec 0\nsystem_usec 0\n"
            f"nr_periods 0\nnr_throttled {self.nr_throttled}\nthrottled_usec {self.throttled_usec}\n"
        )
        (self.cg / "cgroup.procs").write_text("".join(f"{p}\n" for p in self.procs))
        lines = ["cpu  0 0 0 0 0 0 0 0 0 0"]
        lines += [f"cpu{c} " + " ".join(map(str, v + [0, 0])) for c, v in self.jiffies.items()]
        (self.proc / "stat").write_text("\n".join(lines) + "\n")
        for pid, p in self.procs.items():
            d = self.proc / str(pid)
            d.mkdir(exist_ok=True)
            fields = ["S"] + ["0"] * 10 + [str(p["ticks"]), "0"] + ["0"] * 30
            comm = p["cmd"].split()[0].rsplit("/", 1)[-1]
            (d / "stat").write_text(f"{pid} ({comm}) " + " ".join(fields) + "\n")
            (d / "status").write_text(f"Name:\t{comm}\nNSpid:\t{pid}\t{p['ns']}\n")
            (d / "cmdline").write_text(p["cmd"].replace(" ", "\0") + "\0")


@pytest.fixture
def host(tmp_path, monkeypatch):
    return FakeHost(tmp_path, monkeypatch)


def test_parse_cpuset():
    assert parse_cpuset("2-3,10-11\n") == (2, 3, 10, 11)
    assert parse_cpuset("0") == (0,)
    assert parse_cpuset("") == ()


def test_window_splits_busy_into_own_and_other_tenants(host):
    a = snapshot(str(host.cg), CPUS)
    host.advance(seconds=10, own_cpu_s=4, other_cpu_s=6, steal_s=1, psi_some_s=3)
    b = snapshot(str(host.cg), CPUS)
    b = sysinfo.SysSnapshot(**{**b.__dict__, "ts_ns": a.ts_ns + 10 * 10**9})
    w = window(a, b, CPUS)
    assert w.busy_pct == pytest.approx(50.0)  # 10 cpu-s busy of 20 cpu-s
    assert w.own_pct == pytest.approx(20.0)  # 4 of 20
    assert w.other_tenants_pct == pytest.approx(30.0)
    assert w.steal_pct == pytest.approx(5.0)
    assert w.psi_some_pct == pytest.approx(30.0)


def test_processes_exclude_the_harness_and_mark_what_the_agent_started(host):
    host.add_proc(100, 1, "python3 -m agency.harness.daemon")
    host.add_proc(101, 7, "node /usr/lib/node_modules/@anthropic-ai/claude-code/cli.js")
    host.add_proc(102, 42, "python -m pytest tests/", ticks=0)
    host.write()
    a = snapshot(str(host.cg), CPUS)
    host.procs[102]["ticks"] = 150  # 1.5 cpu-s
    host.write()
    b = snapshot(str(host.cg), CPUS)
    b = sysinfo.SysSnapshot(**{**b.__dict__, "ts_ns": a.ts_ns + 10**9})
    w = window(a, b, CPUS, execs={42: {"argv": ["python"], "ts_ns": 1}})
    assert [p["pid"] for p in w.live_procs] == [42]
    assert w.live_procs[0]["started_by_agent"] is True
    assert w.live_procs[0]["cpu_pct"] == pytest.approx(150.0)
    text = render(w)
    assert "python -m pytest tests/ (pid 42, 150% CPU, started by you)" in text
    assert "unreliable" not in text.lower()  # facts only


def test_tracker_windows_one_tool_call(host):
    tracker = SysInfoTracker(lambda: str(host.cg))
    tracker.tool_started("c1", "Bash")
    host.advance(seconds=1, own_cpu_s=1, other_cpu_s=1)
    done = tracker.tool_completed("c1")
    assert done is not None
    w, tool_name = done
    assert tool_name == "Bash"
    assert w.cpus == CPUS
    assert w.busy_pct == pytest.approx(100.0)
    assert tracker.tool_completed("c1") is None  # once only


class _Logger:
    def __init__(self):
        self.events = []

    def record_event(self, type, payload, **_kw):
        self.events.append((type, payload))

    def record_span(self, *a, **kw):
        pass


def _sandbox_with(tracker, expose="none"):
    cfg = agconfig(sysinfoconfig(record=True, expose=expose))
    return SimpleNamespace(
        agconfig=cfg,
        sysinfo=lambda: tracker,
        ensure_gpu_acquired=lambda agname, *, is_cancelled=None: None,
        current_gpu_ids=lambda: None,
    )


def test_interaction_server_records_a_sysinfo_window_per_tool_call(host):
    tracker = SysInfoTracker(lambda: str(host.cg), record=True)
    logger = _Logger()
    server = HostInteractionServer(
        agskill(name="s", prompt="p"), logger, "a", sandbox=_sandbox_with(tracker)
    )
    call_id = server.admit_tool_call("Bash", {"command": "python w.py"})["call_id"]
    host.advance(seconds=1, own_cpu_s=0.5, other_cpu_s=1.0)
    server.complete_tool_call(call_id, result="ok")
    windows = [p for t, p in logger.events if t == "sysinfo_window"]
    assert len(windows) == 1
    assert windows[0]["tool"] == "Bash" and windows[0]["call_id"] == call_id
    assert windows[0]["busy_pct"] == pytest.approx(75.0)


def test_interaction_server_notes_agent_execs(host):
    tracker = SysInfoTracker(lambda: str(host.cg), record=True)
    server = HostInteractionServer(
        agskill(name="s", prompt="p"), _Logger(), "a", sandbox=_sandbox_with(tracker)
    )
    event = agsyscallevent(
        syscall="execve",
        pid=42,
        tid=42,
        argv=["python", "w.py"],
        envp=None,
        path="/usr/bin/python",
        timestamp=0.0,
        tool_name=None,
        tool_args=None,
    )
    server.admit_syscall(event)
    assert tracker._execs[42]["argv"] == ["python", "w.py"]


def test_a_broken_tracker_never_breaks_a_tool_call():
    def boom():
        raise RuntimeError("no cgroup")

    sandbox = _sandbox_with(None)
    sandbox.sysinfo = boom
    server = HostInteractionServer(agskill(name="s", prompt="p"), _Logger(), "a", sandbox=sandbox)
    admitted = server.admit_tool_call("Bash", {})
    assert admitted["allowed"] is True
    server.complete_tool_call(admitted["call_id"], result="ok")


@pytest.mark.parametrize("expose,served", [("none", False), ("tool", True), ("both", True)])
def test_host_mcp_server_serves_get_system_state_only_when_exposed(host, expose, served):
    tracker = SysInfoTracker(lambda: str(host.cg), expose=expose)
    sandbox = _sandbox_with(tracker, expose=expose)
    sandbox.join_transaction = lambda: __import__("contextlib").nullcontext()
    server = HostMcpServer(
        sandbox, agskill(name="s", prompt="p"), resource_pool=None, data_logger=None
    )
    server.build_app()
    names = {t.name for t in asyncio.run(server._mcp_server.list_tools())}
    assert ("get_system_state" in names) is served
    if served:
        tracker.tool_started("warm", "x")  # a snapshot to difference against
        host.advance(seconds=1, own_cpu_s=0.2, other_cpu_s=1.0)
        out = asyncio.run(server._mcp_server.call_tool("get_system_state", {"seconds": 5}))
        state = json.loads(out.content[0].text)["state"]
        assert state.startswith("[system] during the last")
        assert "other tenants" in state


def test_disabled_config_builds_no_tracker():
    from agency.sandbox.agsandbox import agSandbox

    stand_in = SimpleNamespace(agconfig=agconfig())  # record False, expose "none"
    assert agSandbox.sysinfo(stand_in) is None
