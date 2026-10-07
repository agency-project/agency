"""One independent condition per invocation, with a separate deadline watchdog."""

import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .backend import LABEL, container_cgroup, prepare_container, verify_running
from .config import (
    ROOT,
    assignments,
    atomic_json,
    comparison_identity,
    digest,
    private_directory,
    safe_json,
    validate_manifest,
)
from .linux import (
    CGROUP,
    command,
    experiment_lock,
    preflight,
    processes,
    require_linux,
    terminate_tree,
    worker_cgroup,
)
from .metrics import csv_text, reconcile, summarize


def read(path):
    return json.loads(Path(path).read_text())


def child_env():
    # Never inherit host provider credentials/config into Docker CLI helpers.
    return {
        "PATH": os.environ["PATH"],
        "HOME": "/root",
        "PYTHONPATH": str(ROOT),
        "PYTHONUNBUFFERED": "1",
        "AGENCY_PROFILE": "0",
    }


def journal_events(directory):
    events = []
    for path in Path(directory).glob("worker-*/journal.jsonl"):
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                # A killed writer may leave only its final record incomplete.
    return sorted(events, key=lambda event: event["time"])


def finalize(directory):
    plan = read(directory / "plan.json")
    timing = read(directory / "start.json")
    ending = read(directory / "end.json")
    cutoff = min(ending["cutoff"], timing["deadline"])
    rows = reconcile(
        assignments(plan["manifest"]),
        journal_events(directory),
        timing["start"],
        cutoff,
        ending["reason"],
    )
    # Offline grades survive repeated export/recovery.
    for row in rows:
        grade = directory / row["attempt_id"] / "grade.json"
        if row["status"] == "completed" and grade.exists():
            row["success"] = read(grade)["success"]
    metrics = summarize(rows, cutoff - timing["start"], plan["config"]["condition_timeout_seconds"])
    result = {
        "schema_version": 1,
        "run_id": plan["run_id"],
        "trial_id": plan["trial_id"],
        "condition": plan["config"]["condition"],
        "comparison_identity": plan["comparison_identity"],
        "deadline_reached": ending["reason"] == "deadline",
        "termination": ending,
        "metrics": metrics,
        "tasks": rows,
    }
    atomic_json(directory / "results.json", result)
    (directory / "tasks.csv").write_text(csv_text(rows))
    (directory / "metrics.csv").write_text(
        csv_text(
            [
                {
                    "condition": result["condition"],
                    "deadline_reached": result["deadline_reached"],
                    **metrics,
                }
            ]
        )
    )
    return result


def watchdog(directory, parent):
    """Independent reserved-core process retains the host lock if the coordinator dies."""
    require_linux()
    directory = Path(directory)
    cfg = read(directory / "plan.json")["config"]
    atomic_json(directory / "watchdog-ready.json", {"pid": os.getpid()})
    while not (directory / "start.json").exists():
        if os.getppid() != parent:
            terminate_tree(cfg["termination_grace_seconds"])
            return
        time.sleep(0.01)
    timing = read(directory / "start.json")
    while True:
        now = time.monotonic()
        if now >= timing["deadline"]:
            cutoff, reason = timing["deadline"], "deadline"
            break
        if os.getppid() != parent:
            cutoff, reason = now, "coordinator_lost"
            break
        request = directory / "stop.json"
        if request.exists():
            stop = read(request)
            cutoff, reason = min(stop["time"], timing["deadline"]), stop["reason"]
            break
        time.sleep(min(0.01, timing["deadline"] - now))
    # Persist the cutoff before any cleanup and never accept later completions.
    ending = {
        "cutoff": cutoff,
        "reason": reason,
        "signal_started": time.monotonic(),
        "cleanup_complete": False,
    }
    atomic_json(directory / "end.json", ending)
    try:
        terminate_tree(cfg["termination_grace_seconds"])
        ending["cleanup_complete"] = True
    except Exception as error:
        ending["cleanup_error"] = type(error).__name__
    ending["cleanup_finished"] = time.monotonic()
    atomic_json(directory / "end.json", ending)


def telemetry(directory):
    data = {
        "time": time.monotonic(),
        "cpu_stat": Path("/proc/stat").read_text(),
        "host_cpu_pressure": Path("/proc/pressure/cpu").read_text(),
        "cgroups": {},
    }
    for path in [CGROUP, *CGROUP.glob("**/cpu.stat")]:
        group = path if path == CGROUP else path.parent
        try:
            data["cgroups"][str(group)] = {
                name: (group / name).read_text()
                for name in ("cpu.stat", "cpu.pressure", "cpu.max", "cpuset.cpus.effective")
            }
        except FileNotFoundError:
            continue
    with (directory / "telemetry.jsonl").open("a") as stream:
        stream.write(json.dumps(data) + "\n")


def cleanup_owned(plan):
    errors = []
    for name in plan["containers"].values():
        try:
            # Missing containers are already clean; never delete by an unverified name.
            ids = command(
                [
                    "docker",
                    "ps",
                    "-aq",
                    "--filter",
                    f"name=^/{name}$",
                    "--filter",
                    f"label={LABEL}={plan['run_id']}",
                ]
            ).split()
            if ids:
                command(["docker", "rm", "-f", *ids])
        except Exception as error:
            errors.append({"container": name, "error": type(error).__name__})
    for raw in plan["worker_cgroups"]:
        path = Path(raw)
        if path.parent != CGROUP or not path.name.startswith(plan["run_id"]):
            raise ValueError("Refusing a cgroup outside the recorded experiment")
        if path.exists():
            try:
                path.rmdir()
            except OSError as error:
                errors.append({"cgroup": str(path), "error": type(error).__name__})
    return errors


def run(cfg, directory, trial_id):
    require_linux()
    manifest = validate_manifest(read(cfg["manifest"]))
    directory = Path(directory).resolve()
    with experiment_lock() as lock_fd:
        layout, provenance = preflight(cfg, manifest)
        private_directory(directory)
        run_id = "agencycpu-" + uuid.uuid4().hex[:16]
        plan = {
            "run_id": run_id,
            "trial_id": trial_id,
            "config": cfg,
            "manifest": manifest,
            "manifest_hash": digest(manifest),
            "layout": layout,
            "provenance": provenance,
            "comparison_identity": comparison_identity(cfg, manifest, provenance),
            "worker_cgroups": [],
            "containers": {},
            "container_cgroups": {},
        }
        atomic_json(directory / "plan.json", plan)
        atomic_json(directory / "assigned.json", assignments(manifest))
        workers = []
        guard = None
        interrupted = False

        def interrupt(signum, frame):
            nonlocal interrupted
            interrupted = True

        old_handlers = {
            sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            for slot, group in enumerate(manifest["agents"]):
                (directory / f"worker-{slot}").mkdir()
                cg = CGROUP / f"{run_id}-worker-{slot}"
                plan["worker_cgroups"].append(str(cg))
                atomic_json(directory / "plan.json", plan)
                worker_cgroup(cg.name, layout["assignments"][slot], layout["node"])
                for index, task in enumerate(group):
                    if interrupted:
                        raise InterruptedError("Interrupted during preparation")
                    attempt_id = f"a{slot}-t{index}"
                    (directory / attempt_id).mkdir()
                    name = f"{run_id}-{attempt_id}"
                    plan["containers"][attempt_id] = name
                    atomic_json(directory / "plan.json", plan)
                    plan["container_cgroups"][attempt_id] = prepare_container(
                        name, task, cfg, layout["assignments"][slot], run_id
                    )
            atomic_json(directory / "plan.json", plan)
            guard = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.cpu_oversubscription",
                    "_watchdog",
                    str(directory),
                    str(os.getpid()),
                ],
                cwd=ROOT,
                env=child_env(),
                pass_fds=(lock_fd,),
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for slot in range(6):
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.cpu_oversubscription",
                        "_worker",
                        str(directory),
                        str(slot),
                    ],
                    cwd=ROOT,
                    env=child_env(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                workers.append(process)
                process.stdin.write(
                    json.dumps(
                        {
                            "key": os.environ[cfg["api_key_env"]],
                            "url": os.environ[cfg["base_url_env"]],
                        }
                    ).encode()
                    + b"\n"
                )
                process.stdin.close()
            ready_limit = time.monotonic() + 120
            while not (
                (directory / "watchdog-ready.json").exists()
                and all((directory / f"ready-{s}.json").exists() for s in range(6))
            ):
                if (
                    interrupted
                    or guard.poll() is not None
                    or any(w.poll() is not None for w in workers)
                    or time.monotonic() >= ready_limit
                ):
                    raise RuntimeError(
                        "Workers/watchdog did not become ready; inspect worker errors"
                    )
                time.sleep(0.05)
            verify_running(plan)
            start = time.monotonic()
            atomic_json(
                directory / "start.json",
                {
                    "start": start,
                    "deadline": start + cfg["condition_timeout_seconds"],
                    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            next_sample = start
            while guard.poll() is None:
                now = time.monotonic()
                if interrupted:
                    atomic_json(
                        directory / "stop.json", {"time": now, "reason": "operator_interrupted"}
                    )
                    break
                if any(w.poll() not in (None, 0) for w in workers):
                    atomic_json(directory / "stop.json", {"time": now, "reason": "worker_failure"})
                    break
                if all((directory / f"done-{s}.json").exists() for s in range(6)):
                    end = max(read(directory / f"done-{s}.json")["time"] for s in range(6))
                    atomic_json(directory / "stop.json", {"time": end, "reason": "completed"})
                    break
                if now >= next_sample and now < start + cfg["condition_timeout_seconds"]:
                    telemetry(directory)
                    verify_running(plan)
                    next_sample = now + 1
                time.sleep(0.02)
            guard.wait(timeout=cfg["termination_grace_seconds"] + 15)
            if not (directory / "end.json").exists():
                raise RuntimeError("Watchdog failed without recording a cutoff")
        except BaseException as error:
            safe_json(
                directory / "controller-error.json",
                {"error": type(error).__name__, "message": str(error)},
            )
            if (directory / "start.json").exists():
                atomic_json(
                    directory / "stop.json",
                    {"time": time.monotonic(), "reason": "controller_failure"},
                )
            raise
        finally:
            # Pre-start failures and failed watchdogs also have a bounded shutdown.
            if guard is not None and guard.poll() is None:
                if not (directory / "start.json").exists():
                    guard.terminate()
                try:
                    guard.wait(timeout=cfg["termination_grace_seconds"] + 15)
                except subprocess.TimeoutExpired:
                    guard.kill()
                    guard.wait()
            if processes():
                try:
                    terminate_tree(cfg["termination_grace_seconds"])
                except Exception as error:
                    safe_json(directory / "termination-error.json", {"error": type(error).__name__})
                    if (directory / "end.json").exists():
                        ending = read(directory / "end.json")
                        atomic_json(directory / "end.json", {**ending, "cleanup_complete": False})
            for process in workers:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if (directory / "start.json").exists():
                if not (directory / "end.json").exists():
                    timing = read(directory / "start.json")
                    now = time.monotonic()
                    atomic_json(
                        directory / "end.json",
                        {
                            "cutoff": min(now, timing["deadline"]),
                            "reason": "controller_failure",
                            "cleanup_complete": not processes(),
                        },
                    )
                finalize(directory)
            safe_json(directory / "cleanup.json", {"errors": cleanup_owned(plan)})
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        return read(directory / "results.json")


def recover(directory):
    """Seal an interrupted run; never relaunch tasks or reset its budget."""
    require_linux()
    directory = Path(directory).resolve()
    with experiment_lock():
        plan = read(directory / "plan.json")
        # Creation intent is persisted before Docker starts a container. Recover
        # its cgroup by the exact recorded name AND label after a preparation crash.
        for attempt_id, name in plan["containers"].items():
            running = command(
                [
                    "docker",
                    "ps",
                    "-q",
                    "--filter",
                    f"name=^/{name}$",
                    "--filter",
                    f"label={LABEL}={plan['run_id']}",
                ]
            ).split()
            if running:
                plan["container_cgroups"][attempt_id] = str(container_cgroup(name))
        # Refuse to terminate unknown workloads in this exclusive slice.
        known = set(plan["worker_cgroups"]) | set(plan["container_cgroups"].values())
        for pid in processes():
            try:
                rel = (Path("/proc") / str(pid) / "cgroup").read_text().strip().split("0::", 1)[1]
                if str(Path("/sys/fs/cgroup") / rel.lstrip("/")) not in known:
                    raise RuntimeError(
                        "Unknown live process in experiment slice; manual review required"
                    )
            except FileNotFoundError:
                continue
        terminate_tree(plan["config"]["termination_grace_seconds"])
        if (directory / "end.json").exists():
            ending = read(directory / "end.json")
            atomic_json(
                directory / "end.json", {**ending, "cleanup_complete": True, "recovered": True}
            )
        timing_path = directory / "start.json"
        if timing_path.exists() and not (directory / "end.json").exists():
            timing = read(timing_path)
            same_boot = (
                timing["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            )
            # Without a durable cutoff, the last journal timestamp is all we can prove.
            last = max([timing["start"], *[e["time"] for e in journal_events(directory)]])
            atomic_json(
                directory / "end.json",
                {
                    "cutoff": min(math.nextafter(last, math.inf), timing["deadline"]),
                    "reason": "cutoff_unknown",
                    "cleanup_complete": True,
                    "same_boot": same_boot,
                    "invalid_for_comparison": True,
                },
            )
        safe_json(directory / "cleanup.json", {"errors": cleanup_owned(plan)})
        return finalize(directory) if timing_path.exists() else {"status": "preparation_failed"}
