"""Balanced asynchronous EC2 worker queues, durable attempt ledger, and safe resumption."""

from pathlib import Path
import asyncio
import fcntl
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from settings import validate_host

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def read(path):
    return json.loads(path.read_text())


def append(path, value):
    with path.open("a") as output:
        output.write(json.dumps(value, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def schedule():
    config = read(ROOT / "configuration.json")
    tasks = read(ROOT / "task_manifest.json")["tasks"]
    value = build_schedule(tasks, config)
    destination = ROOT / "run_schedule.json"
    if destination.exists():
        existing = read(destination)
        if existing != value:
            raise RuntimeError("Existing schedule differs from the frozen cohort configuration")
        return existing
    destination.write_text(json.dumps(value, indent=2) + "\n")
    return value


def build_schedule(tasks, config):
    seed = config["schedule_seed"]
    repetitions = config.get("repetitions", 10)
    first = config.get("repetition_start", 1)
    if (
        not isinstance(repetitions, int)
        or repetitions < 1
        or not isinstance(first, int)
        or first < 1
    ):
        raise ValueError("Repetition count and starting index must be positive integers")
    expected = len(tasks) * 2 * repetitions
    if config.get("executions", expected) != expected:
        raise ValueError("Configured execution count differs from task/arm/repetition dimensions")
    rng = random.Random(seed)
    lanes = {}
    for arm in ("standalone", "agency"):
        jobs = [
            {
                "run_id": f"t{t + 1}-r{r:02d}-{arm}",
                "instance_id": task["instance_id"],
                "repetition": r,
                "arm": arm,
                "diagnostic": False,
            }
            for t, task in enumerate(tasks)
            for r in range(first, first + repetitions)
        ]
        rng.shuffle(jobs)
        lanes[arm] = jobs
    value = {
        "seed": seed,
        "lanes": lanes,
        "policy": "Two continuously replenished workers per arm; independent seeded random order in each queue",
    }
    return value


async def execute(job):
    attempts = ROOT / ("diagnostics" if job.get("diagnostic") else "attempts") / job["run_id"]
    existing = sorted(attempts.glob("*/summary.json"))
    if existing:
        last = read(existing[-1])
        retryable = (
            last["status"] == "infrastructure_failed"
            and last.get("provider", {}).get("model_requests", 0) == 0
        )
        if not retryable or len(existing) >= 2:
            return last
    for attempt in range(len(existing) + 1, 3):
        spec = {**job, "attempt": attempt}
        directory = attempts / str(attempt)
        if directory.exists():
            raise RuntimeError("Unfinished attempt requires audit before resume: " + str(directory))
        active = execute.active
        active[job["run_id"]] = job["arm"]
        append(
            ROOT / "scheduling_events.jsonl",
            {
                "event": "start",
                "unix_s": time.time(),
                **spec,
                "active_total": len(active),
                "active_standalone": list(active.values()).count("standalone"),
                "active_agency": list(active.values()).count("agency"),
            },
        )
        logs = ROOT / "job_logs"
        logs.mkdir(exist_ok=True)
        with (logs / f"{job['run_id']}-{attempt}.log").open("ab") as log:
            proc = await asyncio.create_subprocess_exec(
                PYTHON,
                str(ROOT / "worker.py"),
                json.dumps(spec),
                cwd=ROOT,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            code = await proc.wait()
        active.pop(job["run_id"])
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            raise RuntimeError(
                f"Worker exited {code} without durable result: {job['run_id']}; audit required"
            )
        result = read(summary_path)
        append(ROOT / ("diagnostic_runs.jsonl" if job.get("diagnostic") else "runs.jsonl"), result)
        append(
            ROOT / "scheduling_events.jsonl",
            {
                "event": "finish",
                "unix_s": time.time(),
                **spec,
                "status": result["status"],
                "exit_code": code,
                "active_total": len(active),
            },
        )
        print(
            json.dumps({"run_id": job["run_id"], "attempt": attempt, "status": result["status"]}),
            flush=True,
        )
        if (
            result["status"] != "infrastructure_failed"
            or result["provider"]["model_requests"]
            or attempt == 2
        ):
            return result


execute.active = {}


async def resource_monitor(stop):
    while not stop.is_set():
        names = await asyncio.to_thread(
            subprocess.run,
            ["docker", "ps", "--filter", "name=ni-haiku-", "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
        )
        containers = []
        if names.stdout.split():
            inspection = await asyncio.to_thread(
                subprocess.run,
                ["docker", "inspect", *names.stdout.split()],
                capture_output=True,
                text=True,
            )
            if inspection.returncode == 0:
                containers = [
                    {
                        "name": x["Name"],
                        "cpus": x["HostConfig"]["NanoCpus"] / 1e9,
                        "memory_bytes": x["HostConfig"]["Memory"],
                    }
                    for x in json.loads(inspection.stdout)
                ]
        append(
            ROOT / "resource_samples.jsonl",
            {
                "unix_s": time.time(),
                "containers": containers,
                "cpu_quota_total": sum(c["cpus"] for c in containers),
                "host_load": os.getloadavg(),
                "mem_available_kb": int(
                    next(
                        line.split()[1]
                        for line in Path("/proc/meminfo").read_text().splitlines()
                        if line.startswith("MemAvailable:")
                    )
                ),
            },
        )
        if (
            len(containers) > 4
            or sum(c["cpus"] for c in containers) > 8
            or any(c["cpus"] > 2 for c in containers)
        ):
            # Stop admitting new work on any quota violation; running attempts remain auditable.
            raise RuntimeError("Study CPU/concurrency quota violation")
        try:
            await asyncio.wait_for(stop.wait(), timeout=10)
        except TimeoutError:
            continue


async def main(mode):
    actual_host, _ = validate_host(ROOT)
    configuration = read(ROOT / "configuration.json")
    assert actual_host == configuration["execution_host"], "Prepared host identity changed"
    assert (
        hashlib.sha256((ROOT / "task_manifest.json").read_bytes()).hexdigest()
        == configuration["manifest_sha256"]
    )
    if not mode.startswith("diagnostics"):
        preparation = read(ROOT / "preparation_complete.json")
        assert (
            hashlib.sha256((ROOT / "configuration.json").read_bytes()).hexdigest()
            == preparation["configuration_sha256"]
        )
    # A worker may have durably finished immediately before its parent was interrupted.
    for folder, journal in (("attempts", "runs.jsonl"), ("diagnostics", "diagnostic_runs.jsonl")):
        journal_path = ROOT / journal
        known = (
            {
                (r["run_id"], r["attempt"])
                for r in (json.loads(line) for line in journal_path.read_text().splitlines())
            }
            if journal_path.exists()
            else set()
        )
        for path in sorted((ROOT / folder).glob("*/*/summary.json")):
            result = read(path)
            if (result["run_id"], result["attempt"]) not in known:
                append(journal_path, result)
    plan = schedule()
    if mode.startswith("diagnostics"):
        tasks = read(ROOT / "task_manifest.json")["tasks"]
        jobs = [
            {
                "run_id": f"{mode}-t{i + 1}-{arm}",
                "instance_id": task["instance_id"],
                "arm": arm,
                "repetition": 0,
                "diagnostic": True,
            }
            for i, task in enumerate(tasks[:2])
            for arm in ("standalone", "agency")
        ]
        await asyncio.gather(*(execute(job) for job in jobs))
        return
    gate = read(ROOT / "diagnostic_gate.json")
    assert gate["passed"], "Diagnostic gate must pass before measured execution"
    available = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    workers = 2 if available >= configuration["memory_headroom_required_bytes"] else 1
    assert available >= 12 * 1024**3, "Insufficient memory for balanced two-worker minimum"
    append(
        ROOT / "scheduling_events.jsonl",
        {
            "event": "launch",
            "unix_s": time.time(),
            "workers_per_arm": workers,
            "available_memory_bytes": available,
            "reason": "Full four-worker headroom"
            if workers == 2
            else "Reduced concurrency to preserve memory headroom",
        },
    )
    queues = {}
    for arm, jobs in plan["lanes"].items():
        queue = asyncio.Queue()
        for job in jobs:
            queue.put_nowait(job)
        queues[arm] = queue

    async def worker(arm):
        while not queues[arm].empty():
            job = queues[arm].get_nowait()
            await execute(job)
            queues[arm].task_done()

    await asyncio.gather(*(worker(arm) for arm in queues for _ in range(workers)))
    (ROOT / "batch_complete.json").write_text(
        json.dumps({"unix_s": time.time(), "planned_executions": configuration["executions"]})
        + "\n"
    )


if __name__ == "__main__":
    with (ROOT / "experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        async def supervised():
            stop = asyncio.Event()
            monitor = asyncio.create_task(resource_monitor(stop))
            work = asyncio.create_task(main(sys.argv[1]))
            try:
                done, _ = await asyncio.wait((monitor, work), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                stop.set()
                if not work.done():
                    work.cancel()
                await monitor

        asyncio.run(supervised())
