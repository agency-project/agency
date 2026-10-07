"""Read-only preflight and narrowly scoped cgroup lifecycle for Ubuntu."""

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from .config import ROOT, file_hash

CGROUP = Path("/sys/fs/cgroup/agencycpu.slice")
LOCK = Path("/run/lock/agency-cpu-experiment.lock")


def require_linux():
    if platform.system() != "Linux":
        raise RuntimeError("Execution/preflight requires Ubuntu Linux; use plan on macOS")


def command(args, *, timeout=30, **kwargs):
    require_linux()
    # Do not include command output or argv in errors: a tool may echo credentials.
    result = subprocess.run(args, capture_output=True, timeout=timeout, **kwargs)
    if result.returncode:
        raise RuntimeError(
            f"{Path(args[0]).name} failed (exit {result.returncode}); inspect prerequisites"
        )
    return result.stdout.decode() if isinstance(result.stdout, bytes) else result.stdout


def cpu_set(text):
    cpus = set()
    for item in text.strip().split(","):
        if not item:
            continue
        ends = item.split("-")
        cpus.update(range(int(ends[0]), int(ends[-1]) + 1))
    return cpus


def layout_from_rows(rows, condition):
    cores = {}
    for row in rows:
        cores.setdefault((row["socket"], row["core"]), []).append(row["cpu"])
    if len(cores) != 8 or any(len(cpus) != 1 for cpus in cores.values()):
        raise ValueError("Require exactly eight online physical cores with SMT disabled")
    if len({r["socket"] for r in rows}) != 1 or len({r["node"] for r in rows}) != 1:
        raise ValueError("Require one socket and one NUMA node for this baseline")
    ordered = [cores[key][0] for key in sorted(cores)]
    workers = ordered[2:]
    return {
        "topology": rows,
        "reserved": ordered[:2],
        "agent_cores": workers,
        "assignments": workers if condition == "DCB" else [workers[0]] * 6,
        "node": rows[0]["node"],
        "condition": condition,
    }


def topology(condition):
    from benchmarks.agent_stress.placement import parse_topology

    require_linux()
    rows = parse_topology(command(["lscpu", "-p=CPU,CORE,SOCKET,NODE,ONLINE"]))
    return layout_from_rows(rows, condition)


def processes(root=CGROUP):
    paths = [root / "cgroup.procs", *root.glob("**/cgroup.procs")]
    pids = set()
    for path in paths:
        try:
            pids.update(int(pid) for pid in path.read_text().split())
        except FileNotFoundError:
            continue
    return pids


def verify_cgroup(path, expected):
    if cpu_set((path / "cpuset.cpus.effective").read_text()) != set(expected):
        raise RuntimeError(f"Incorrect effective cpuset at {path}")
    for ancestor in (path, *path.parents):
        if ancestor == Path("/sys/fs"):
            break
        quota = ancestor / "cpu.max"
        if quota.exists() and quota.read_text().split()[0] != "max":
            raise RuntimeError(f"CPU quota would confound the experiment: {ancestor}")


def verify_background(layout):
    reserved = set(layout["reserved"])
    for name in ("system.slice", "user.slice", "init.scope"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            if not cpu_set((path / "cpuset.cpus.effective").read_text()) <= reserved:
                raise RuntimeError(f"Reserve the background cores for {name}; see isolation-script")
    # Custom top-level slices/scopes must not silently occupy dedicated cores.
    for path in Path("/sys/fs/cgroup").iterdir():
        if path.is_dir() and path != CGROUP and (path / "cpuset.cpus.effective").exists():
            if (
                processes(path)
                and not cpu_set((path / "cpuset.cpus.effective").read_text()) <= reserved
            ):
                raise RuntimeError(f"Background cgroup is not isolated: {path}")
    for pid in (Path("/sys/fs/cgroup") / "cgroup.procs").read_text().split():
        try:
            # Empty cmdline identifies kernel threads; IRQ/kernel isolation is best effort.
            if (Path("/proc") / pid / "cmdline").read_bytes() and not os.sched_getaffinity(
                int(pid)
            ) <= reserved:
                raise RuntimeError("A userspace process in the root cgroup can occupy agent cores")
        except (FileNotFoundError, ProcessLookupError):
            continue


def preflight(cfg, manifest, *, allow_active=False):
    require_linux()
    from .workload import verify_snapshot

    verify_snapshot(cfg, manifest)
    if os.geteuid() != 0:
        raise RuntimeError("Run on the dedicated Ubuntu host as root (cgroup management required)")
    layout = topology(cfg["condition"])
    if not Path("/sys/fs/cgroup/cgroup.controllers").exists():
        raise RuntimeError("Unified cgroup v2 is required")
    if not {"cpu", "cpuset"} <= set((CGROUP / "cgroup.subtree_control").read_text().split()):
        raise RuntimeError("Enable cpu and cpuset controllers in agencycpu.slice")
    verify_cgroup(CGROUP, layout["agent_cores"])
    if not os.access(CGROUP / "cgroup.kill", os.W_OK):
        raise RuntimeError("Kernel must provide writable cgroup.kill for deadline enforcement")
    if not allow_active and processes():
        raise RuntimeError("Experiment cgroup is occupied; recover/cleanup the previous run first")
    verify_background(layout)
    if not os.sched_getaffinity(0) <= set(layout["reserved"]):
        raise RuntimeError("Launch the orchestrator from a reserved-core background slice")
    if os.sched_getscheduler(0) != os.SCHED_OTHER or os.getpriority(os.PRIO_PROCESS, 0) != 0:
        raise RuntimeError("Use SCHED_OTHER with nice=0")
    info = json.loads(command(["docker", "info", "--format", "{{json .}}"]))
    if str(info["CgroupVersion"]) != "2" or info["CgroupDriver"] != "systemd":
        raise RuntimeError("Docker must use cgroup v2 with the systemd driver")
    if (
        not Path("/var/run/docker.sock").is_socket()
        or os.environ.get("DOCKER_HOST")
        or os.environ.get("DOCKER_CONTEXT")
    ):
        raise RuntimeError("Use the local rootful Docker socket and default context")
    endpoint = command(
        ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"]
    ).strip()
    if endpoint != "unix:///var/run/docker.sock":
        raise RuntimeError("Remote/rootless Docker is unsupported")
    url = os.environ.get(cfg["base_url_env"], "")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("Set a credential-free LiteLLM HTTP(S) base URL")
    if not os.environ.get(cfg["api_key_env"]):
        raise RuntimeError(f"Missing {cfg['api_key_env']} (value is never displayed)")
    if file_hash(cfg["codex_binary"]) != cfg["codex_sha256"]:
        raise RuntimeError("Linux Codex binary hash mismatch")
    binary = Path(cfg["codex_binary"])
    with binary.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise RuntimeError("Codex must be a Linux ELF binary")
    version = command([str(binary), "--version"]).strip()
    if version != "codex-cli " + cfg["codex_version"]:
        raise RuntimeError("Codex version differs from configuration")
    help_text = command([str(binary), "exec", "--help"])
    for option in (
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "--json",
        "--output-last-message",
    ):
        if option not in help_text:
            raise RuntimeError("Installed Codex exec interface is incompatible")
    images = sorted({task["image"] for group in manifest["agents"] for task in group})
    for image in images:
        found = command(["docker", "image", "inspect", "--format", "{{.Id}}", image]).strip()
        if found != image:
            raise RuntimeError("A prepared immutable task image is missing")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("agency", "openai", "httpx", "fastapi", "uvicorn", "swebench", "datasets")
    }
    if command(["git", "-C", str(ROOT), "status", "--porcelain"]).strip():
        raise RuntimeError("Commit/review implementation and manifest before freezing a trial")
    provenance = {
        "git_commit": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip(),
        "versions": versions,
        "python": platform.python_version(),
        "kernel": platform.release(),
        "cpu_model": next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "docker_version": info["ServerVersion"],
        "topology": layout["topology"],
        "codex_version": version,
        "endpoint_hash": hashlib.sha256(url.encode()).hexdigest(),
        "scheduler": "SCHED_OTHER/nice=0; cpuset only; no CPU quotas",
    }
    return layout, provenance


@contextmanager
def experiment_lock():
    require_linux()
    with LOCK.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another run, watchdog, evaluator, or cleanup owns this host"
            ) from None
        yield stream.fileno()


def worker_cgroup(name, cpu, node):
    path = CGROUP / name
    path.mkdir()
    (path / "cpuset.mems").write_text(str(node))
    (path / "cpuset.cpus").write_text(str(cpu))
    verify_cgroup(path, [cpu])
    return path


def join_cgroup(path, cpu):
    require_linux()
    (path / "cgroup.procs").write_text(str(os.getpid()))
    os.sched_setaffinity(0, {cpu})
    verify_cgroup(path, [cpu])
    if os.sched_getaffinity(0) != {cpu}:
        raise RuntimeError("Worker affinity was not enforced")


def terminate_tree(grace):
    """Signal *all* descendants, including detached tools; cgroup.kill is the fallback."""
    require_linux()
    began = time.monotonic()
    for pid in processes():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    while processes() and time.monotonic() - began < grace:
        time.sleep(0.02)
    if processes():
        (CGROUP / "cgroup.kill").write_text("1")
    limit = time.monotonic() + 5
    while processes() and time.monotonic() < limit:
        time.sleep(0.02)
    if processes():
        raise RuntimeError("cgroup remains populated after SIGKILL; host requires intervention")


def isolation_script(layout):
    reserved = ",".join(map(str, layout["reserved"]))
    agents = ",".join(map(str, layout["agent_cores"]))
    return f"""#!/bin/sh
set -eu
# REVIEW before running as root on the dedicated Ubuntu host.
# Save current AllowedCPUs properties before changing these runtime settings.
cat > /run/systemd/system/agencycpu.slice <<'UNIT'
[Unit]
Description=Agency CPU experiment owned slice
[Slice]
AllowedCPUs={agents}
CPUAccounting=yes
UNIT
systemctl daemon-reload
systemctl start agencycpu.slice
echo '+cpu +cpuset' > /sys/fs/cgroup/agencycpu.slice/cgroup.subtree_control
systemctl set-property --runtime system.slice AllowedCPUs={reserved}
systemctl set-property --runtime user.slice AllowedCPUs={reserved}
systemctl set-property --runtime init.scope AllowedCPUs={reserved}
# Reboot restores these runtime-only changes. See handoff for explicit restoration.
"""
