"""Where the workload runs: one shared systemd slice, and worker launchers.

All agents (their host-side Agency process and their container) live in one
transient user slice restricted to the experimental CPU pool. That slice is the
single place the shared CPU pool and the shared memory ceiling are enforced, and
the single cgroup whose counters describe the whole workload. No per-agent CPU
quota or CPU set exists anywhere.

Only runtime (non-persistent) systemd properties are set, and every property
changed on a pre-existing unit is restored from a lease file, even after a
crash (``restore_stale``).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .topology import format_cpu_list
from .util import atomic_json, capture, checked, read_json_or

SLICE = "agencytb.slice"
SESSION_UNITS = ("app.slice", "session.slice", "background.slice", "init.scope")
SESSION_PROPERTIES = ("AllowedCPUs",)


def systemd_available() -> bool:
    if not (shutil.which("systemd-run") and shutil.which("systemctl")):
        return False
    return (
        capture(["systemctl", "--user", "show", "--property=Version"], timeout=10)["returncode"]
        == 0
    )


class WorkloadSlice:
    """The shared cgroup for every agent of one run."""

    cgroup_root = Path("/sys/fs/cgroup")

    def __init__(self, layout, memory: dict, lease_path, *, confine_session: bool, name=SLICE):
        self.layout = layout
        self.memory = memory
        self.lease_path = Path(lease_path)
        self.confine_session = confine_session
        self.name = name
        self.cgroup: "Path | None" = None
        self.before: dict = {}
        self.applied: dict = {}

    def _show(self, unit, *properties) -> dict:
        result = capture(
            ["systemctl", "--user", "show", unit, *[f"--property={p}" for p in properties]]
        )
        if result["returncode"] != 0:
            return {}
        return dict(line.split("=", 1) for line in result["stdout"].splitlines() if "=" in line)

    def enter(self) -> dict:
        pool, reserved = self.layout.pool_arg, self.layout.reserved_arg
        checked(["systemctl", "--user", "start", self.name])

        def size(value):  # systemd spells "no limit" as infinity
            return "infinity" if str(value).lower() in ("max", "infinity") else value

        properties = [
            f"AllowedCPUs={pool}",
            f"MemoryMax={size(self.memory['workload_memory_max'])}",
            f"MemorySwapMax={size(self.memory['workload_swap_max'])}",
            f"TasksMax={self.memory['workload_tasks_max']}",
        ]
        checked(["systemctl", "--user", "set-property", "--runtime", self.name, *properties])
        if self.confine_session:
            for unit in SESSION_UNITS:
                current = self._show(unit, *SESSION_PROPERTIES)
                if current:
                    self.before[unit] = current
            # Persist the originals before changing anything.
            atomic_json(self.lease_path, dict(active=True, original=self.before, slice=self.name))
            for unit in self.before:
                checked(
                    [
                        "systemctl",
                        "--user",
                        "set-property",
                        "--runtime",
                        unit,
                        f"AllowedCPUs={reserved}",
                    ]
                )
        control_group = checked(
            ["systemctl", "--user", "show", self.name, "--property=ControlGroup", "--value"]
        )
        self.cgroup = self.cgroup_root / control_group.lstrip("/")
        effective = None
        try:
            effective = (self.cgroup / "cpuset.cpus.effective").read_text().strip()
        except OSError:
            pass
        if effective is None or effective != pool:
            raise RuntimeError(
                f"Workload slice cpuset is {effective!r}, expected {pool!r}; the cpuset "
                "controller must be delegated to the user manager"
            )
        self.applied = dict(
            slice=self.name,
            cgroup=str(self.cgroup),
            properties=properties,
            effective_cpus=effective,
            session_confined_to=reserved if self.confine_session else None,
            session_units=sorted(self.before),
            unmovable=(
                "Root-owned services, kernel threads and interrupts cannot be moved by an "
                "unprivileged user and may run on any CPU."
            ),
        )
        return self.applied

    def restore(self) -> dict:
        results = {}
        for unit, properties in self.before.items():
            assignments = [f"{key}={value}" for key, value in properties.items()]
            result = capture(
                ["systemctl", "--user", "set-property", "--runtime", unit, *assignments]
            )
            results[unit] = dict(returncode=result["returncode"], stderr=result["stderr"])
        if self.before and all(r["returncode"] == 0 for r in results.values()):
            atomic_json(self.lease_path, dict(active=False, original=self.before))
        stop = capture(["systemctl", "--user", "stop", self.name], timeout=120)
        results[self.name] = dict(returncode=stop["returncode"], stderr=stop["stderr"])
        return results


def restore_stale(lease_path) -> "dict | None":
    """Undo session confinement left behind by a run that did not exit cleanly."""
    lease = read_json_or(lease_path)
    if not lease or not lease.get("active"):
        return None
    results = {}
    for unit, properties in (lease.get("original") or {}).items():
        assignments = [f"{key}={value}" for key, value in properties.items()]
        result = capture(["systemctl", "--user", "set-property", "--runtime", unit, *assignments])
        results[unit] = result["returncode"]
    if all(code == 0 for code in results.values()):
        atomic_json(lease_path, dict(active=False, original=lease.get("original")))
    return results


# ------------------------------------------------------------------ launchers


class Handle:
    """A launched worker process the runner can poll, signal and reap."""

    name: str

    def poll(self) -> "int | None":
        raise NotImplementedError

    def terminate(self) -> None:
        raise NotImplementedError

    def kill(self) -> None:
        raise NotImplementedError


class _SystemdHandle(Handle):
    def __init__(self, unit, process):
        self.name, self._process = unit, process

    def poll(self):
        return self._process.poll()

    def terminate(self):
        capture(["systemctl", "--user", "kill", "--kill-whom=main", "--signal=SIGTERM", self.name])

    def kill(self):
        capture(["systemctl", "--user", "kill", "--signal=SIGKILL", self.name])
        capture(["systemctl", "--user", "stop", self.name], timeout=30)
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()


class SystemdLauncher:
    """Each worker is a transient user service inside the workload slice."""

    kind = "systemd"

    def __init__(
        self,
        *,
        slice_name,
        pool_cpus,
        python,
        working_directory,
        pythonpath,
        runtime_max_seconds,
        unit_prefix,
        environment=None,
    ):
        self.slice_name = slice_name
        self.pool = format_cpu_list(pool_cpus)
        self.python = str(python)
        self.working_directory = str(working_directory)
        self.pythonpath = pythonpath
        self.runtime_max = int(runtime_max_seconds)
        self.unit_prefix = unit_prefix
        self.environment = environment or {}

    def start(self, module: str, job_path, log_path, slot: int) -> Handle:
        unit = f"{self.unit_prefix}-t{slot:02d}.service"
        argv = [
            "systemd-run",
            "--user",
            "--wait",
            "--collect",
            "--quiet",
            f"--unit={unit}",
            f"--slice={self.slice_name}",
            f"--working-directory={self.working_directory}",
            f"--setenv=PYTHONPATH={self.pythonpath}",
            "--setenv=PYTHONUNBUFFERED=1",
            f"--property=AllowedCPUs={self.pool}",
            f"--property=CPUAffinity={self.pool}",
            f"--property=RuntimeMaxSec={self.runtime_max}",
            "--property=TimeoutStopSec=45s",
            "--property=KillMode=control-group",
            f"--property=StandardOutput=append:{log_path}",
            f"--property=StandardError=append:{log_path}",
        ]
        for key, value in self.environment.items():
            argv.append(f"--setenv={key}={value}")
        argv += [self.python, "-m", module, str(job_path)]
        process = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return _SystemdHandle(unit, process)

    def stale_units(self, prefix: str) -> list:
        result = capture(
            ["systemctl", "--user", "list-units", "--all", "--plain", "--no-legend", f"{prefix}*"]
        )
        return [line.split()[0] for line in result["stdout"].splitlines() if line.split()]


class _ProcessHandle(Handle):
    def __init__(self, name, process):
        self.name, self._process = name, process

    def poll(self):
        return self._process.poll()

    def _signal(self, signum):
        try:
            os.killpg(self._process.pid, signum)
        except (ProcessLookupError, PermissionError):
            pass

    def terminate(self):
        try:
            self._process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass

    def kill(self):
        self._signal(signal.SIGKILL)
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


class ProcessLauncher:
    """Plain child processes with CPU affinity. For hosts without a user
    systemd manager and for the mocked-worker validation suite. Affinity set
    this way is advisory (a process may widen it), so measured runs require
    ``SystemdLauncher``."""

    kind = "process"

    def __init__(
        self,
        *,
        pool_cpus=None,
        python=sys.executable,
        working_directory=".",
        pythonpath=None,
        environment=None,
        **_,
    ):
        self.pool = set(pool_cpus) if pool_cpus else None
        self.python = str(python)
        self.working_directory = str(working_directory)
        self.pythonpath = pythonpath
        self.environment = environment or {}

    def start(self, module: str, job_path, log_path, slot: int) -> Handle:
        env = {**os.environ, **self.environment, "PYTHONUNBUFFERED": "1"}
        if self.pythonpath:
            env["PYTHONPATH"] = self.pythonpath
        pool = self.pool

        def confine():
            # The runner itself sits on the reserved CPUs, so the pool is set
            # outright rather than intersected with the inherited mask.
            if pool:
                try:
                    os.sched_setaffinity(0, pool)
                except OSError:
                    pass  # pool not permitted by this cgroup; keep the inherited mask

        log = open(log_path, "ab")
        try:
            process = subprocess.Popen(
                [self.python, "-m", module, str(job_path)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=self.working_directory,
                env=env,
                start_new_session=True,
                preexec_fn=confine,
            )
        finally:
            log.close()
        return _ProcessHandle(f"process-{process.pid}", process)

    def stale_units(self, prefix: str) -> list:
        return []


def wait_for(predicate, timeout: float, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
