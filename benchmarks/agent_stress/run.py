"""Real Agency sandbox experiment. Defaults intentionally stop at N=1,2.

Run from repository root: .venv/bin/python benchmarks/agent_stress/run.py --help
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
from pathlib import Path
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.agent_stress.placement import apply_driver_affinity


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"command": args, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": args, "returncode": -1, "error": str(exc)}


def peak(intervals):
    events = [
        (t, delta)
        for start, end in intervals
        if end > start
        for t, delta in ((start, 1), (end, -1))
    ]
    active = high = 0
    for _, delta in sorted(events):  # end before start: half-open intervals
        active += delta
        high = max(high, active)
    return high


def events(logger):
    logger.flush()
    with sqlite3.connect(logger.db_path) as db:
        return [
            {"id": i, "type": typ, "timestamp": ts, "name": name, "payload": json.loads(payload)}
            for i, typ, ts, name, payload in db.execute(
                "SELECT id,type,timestamp,name,payload FROM events ORDER BY id"
            )
        ]


def receipts(value):
    """Recover child evidence from Agency's existing tool_result event, even on failure."""
    if isinstance(value, dict):
        for child in value.values():
            yield from receipts(child)
    elif isinstance(value, list):
        for child in value:
            yield from receipts(child)
    elif isinstance(value, str):
        for line in value.splitlines():
            if line.startswith("STRESS_RECEIPT="):
                yield json.loads(line.removeprefix("STRESS_RECEIPT="))
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return
        if isinstance(parsed, (dict, list)):
            yield from receipts(parsed)


def replay(path, seconds, mode="normal"):
    # Synthetic source in the existing replay format; no patched backend or engine.
    code = f"""import json, os, time
from pathlib import Path
p = Path('/workspace/stress_state.json')
before = json.loads(p.read_text()) if p.exists() else {{'counter': 0, 'dirty': False}}
start = time.time()
mono = time.monotonic()
after = dict(before)
if {mode!r} != 'verify':
    after['counter'] += 1
    after['dirty'] = {mode == "dirty"!r}
    p.write_text(json.dumps(after))
time.sleep({seconds!r})
r = {{'start': start, 'end': time.time(), 'elapsed': time.monotonic()-mono,
     'pid': os.getpid(), 'before': before, 'after': after, 'mode': {mode!r},
     'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
     'cpuset_cpus_effective': Path('/sys/fs/cgroup/cpuset.cpus.effective').read_text().strip()
         if Path('/sys/fs/cgroup/cpuset.cpus.effective').exists() else None,
     'cpuset_mems_effective': Path('/sys/fs/cgroup/cpuset.mems.effective').read_text().strip()
         if Path('/sys/fs/cgroup/cpuset.mems.effective').exists() else None}}
Path('/workspace/stress_receipt.json').write_text(json.dumps(r))
print('STRESS_RECEIPT=' + json.dumps(r), flush=True)
"""
    bash = "python3 -c " + shlex.quote(code)
    blocks = [
        [
            {
                "type": "tool_use",
                "index": 0,
                "id": "stress_bash",
                "name": "bash",
                "arguments": json.dumps({"command": bash, "timeout": 120}),
            },
            {"type": "metadata", "stop_reason": "tool_use"},
        ],
        [
            {
                "type": "text",
                "index": 0,
                "text": json.dumps({"receipt": "/workspace/stress_receipt.json"}),
            },
            {"type": "metadata", "stop_reason": "end_turn"},
        ],
    ]
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE events (id INTEGER PRIMARY KEY, type TEXT, call_label TEXT, payload TEXT)"
        )
        for i, exchange in enumerate(blocks):
            db.executemany(
                "INSERT INTO events(type,call_label,payload) VALUES ('llm_block',?,?)",
                [(str(i), json.dumps(b)) for b in exchange],
            )
    return code


def save(out, report):
    (out / "results.json").write_text(json.dumps(report, indent=2))
    rows = report.get("phases", [])
    trace = []
    for row in rows:
        for inv in row["invocations"]:
            for lane, start_key, end_key in [
                ("engine", "start", "end"),
                ("child", "child_start", "child_end"),
            ]:
                if start_key in inv and end_key in inv:
                    trace.append(
                        {
                            "name": row["phase"],
                            "cat": lane,
                            "ph": "X",
                            "pid": lane,
                            "tid": inv["agent"],
                            "ts": inv[start_key] * 1e6,
                            "dur": (inv[end_key] - inv[start_key]) * 1e6,
                            "args": {"ordering_id": inv["ordering_id"], "status": inv["status"]},
                        }
                    )
    (out / "timeline.trace.json").write_text(json.dumps({"traceEvents": trace}, indent=2))
    metrics = [
        "phase",
        "total_agents",
        "total_invocations",
        "creation_wall_s",
        "submission_wall_s",
        "total_wall_s",
        "throughput_per_s",
        "successes",
        "failures",
        "peak_engine_backed_requests",
        "peak_children",
        "scheduler_running_count_peak",
        "passed",
    ]
    with (out / "metrics.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=metrics, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with (out / "invocations.csv").open("w") as f:
        fields = [
            "phase",
            "agent",
            "ordering_id",
            "request_id",
            "submitted_at",
            "start",
            "end",
            "child_start",
            "child_end",
            "status",
            "error",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            for inv in row["invocations"]:
                w.writerow({"phase": row["phase"], **inv})
    demonstrated = max(
        (
            r["total_agents"]
            for r in rows
            if r["phase"].startswith("sweep")
            and r["passed"]
            and r["peak_children"] == r["total_agents"]
        ),
        default=0,
    )
    report["largest_demonstrated_n"] = demonstrated
    (out / "results.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# Agency stress experiment",
        "",
        f"Status: {report['status']}",
        f"Largest N with all children overlapping and all requests successful: {demonstrated}.",
        "",
        "No unrun concurrency level is supported by this result.",
        "",
        "| Phase | N | Submit s | Total s | requests/s | Engine peak | Child peak | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['phase']} | {r['total_agents']} | {r['submission_wall_s']:.4f} | "
            f"{r['total_wall_s']:.3f} | {r['throughput_per_s']:.3f} | "
            f"{r['peak_engine_backed_requests']} | {r['peak_children']} | {r['passed']} |"
        )
    if report.get("error"):
        lines.extend(["", "Blocking error: " + report["error"]])
    lines.extend(
        [
            "",
            "Engine intervals are scheduler admission-to-terminal events, including startup/teardown.",
            "Child intervals come from real sandbox Python receipts in existing tool-result logs.",
            "Cold and subsequent invocations both include sandbox lifecycle costs; subsequent calls are not guaranteed hot.",
            "No LLM service is used. Without profiler resource data, poor scaling cannot be attributed to Agency.",
            "See README.md for the later sweep and bottleneck attribution protocol.",
        ]
    )
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def run(args, out, report):
    from agency import agent, agdata, agfile, agskill
    from agency.configs.agconfig import (
        agconfig,
        agentconfig,
        llmconfig,
        orchestratorconfig,
        sandboxconfig,
        resourcesconfig,
    )
    from agency.orchestrator import get_orchestrator

    skill = agskill(
        "stress",
        "Execute the prescribed deterministic workload.",
        output_schema=agdata(receipt=agfile),
        max_output_schema_retries=0,
    )
    owned = []
    configs = {}
    for name, duration, mode in [
        ("normal", args.seconds, "normal"),
        ("slow", args.slow_seconds, "normal"),
        ("dirty", args.seconds, "dirty"),
        ("verify", 0, "verify"),
    ]:
        source = out / f"replay_{name}.sqlite3"
        replay(source, duration, mode)
        configs[name] = agconfig(
            llmconfig(
                provider="mock",
                model="deterministic-replay",
                replay_db_path=str(source),
                timing_mode="instant",
                max_retries=0,
            ),
            agentconfig(log_dir=str(out / "logs"), output_dir=str(out / "outputs")),
            orchestratorconfig(db_path=str(out / "orchestrator.sqlite3")),
            sandboxconfig(backend=args.backend, base_image=args.image),
            resourcesconfig(idle_cpus=args.cpus, idle_memory=args.memory),
        )

    def make(name, config="normal", slot=0):
        cfg = configs[config].clone()
        cfg.sandbox.cpuset_cpus = str(report["cpu_layout"]["worker_cpus"][slot])
        cfg.sandbox.cpuset_mems = str(report["cpu_layout"]["numa_node"])
        a = agent(name, agconfig=cfg, harness="native")
        owned.append(a)
        return a

    def change_workload(a, name):
        cfg = a.get_config_copy()
        cfg.llm = configs[name].llm.clone()
        a.change_config(cfg)

    def phase(name, jobs, creation=0, kind="correctness"):
        phase_epoch = time.time()
        handles = []
        host_before = resource.getrusage(resource.RUSAGE_SELF)
        start_epoch = time.time()
        start = time.perf_counter()
        # CRITICAL: all run() calls finish before ANY handle is inspected/resolved.
        for a, steps in jobs:
            submitted = time.time()
            inv = a.run(skill, agdata(), max_steps=steps)
            handles.append((a, inv, submitted))
        submit_s = time.perf_counter() - start
        deadline = start + args.timeout
        invs = []
        for a, inv, submitted in handles:
            error = None
            try:
                inv.wait(timeout=max(0.001, deadline - time.perf_counter()))
                result = inv.to_dict()
                error = result.get("error")
            except Exception as exc:
                error = str(exc)
            invs.append(
                {
                    "agent": str(a.agname),
                    "ordering_id": inv.ordering_id,
                    "request_id": inv._request_id,  # telemetry correlation only
                    "submitted_at": submitted,
                    "status": inv.state,
                    "error": error,
                    "expected_cpuset_cpus": a.agconfig.sandbox.cpuset_cpus,
                    "expected_cpuset_mems": a.agconfig.sandbox.cpuset_mems,
                }
            )
        total = time.perf_counter() - start
        host_after = resource.getrusage(resource.RUSAGE_SELF)
        raw = events(get_orchestrator().data_logger)
        request_map = {i["request_id"]: i for i in invs}
        for event in raw:
            item = request_map.get(event["payload"].get("request_id"))
            if item is not None:
                if event["type"] == "request_started":
                    item["start"] = event["timestamp"]
                if event["type"] in {
                    "request_completed",
                    "request_failed",
                    "request_cancelled",
                    "request_destroyed",
                }:
                    item["end"] = event["timestamp"]
        for a in dict.fromkeys(a for a, _ in jobs):
            log = events(a.data_logger)
            (out / f"events_{a.agname}.json").write_text(json.dumps(log, indent=2))
            agent_invs = [i for i in invs if i["agent"] == str(a.agname)]
            if a.sandbox is not None:
                inspection = command(
                    [args.backend, "inspect", "--format", "{{json .HostConfig}}", a.sandbox._name]
                )
                # No container exec/start: inspect a hibernating container after completion.
                for item in agent_invs:
                    item["container_name"] = a.sandbox._name
                    item["container_host_config"] = inspection
            for i in agent_invs:
                found = [
                    r
                    for e in log
                    if e["type"] == "tool_result"
                    and i.get("start", float("inf")) <= e["timestamp"] <= i.get("end", 0)
                    for r in receipts(e["payload"])
                ]
                if len(found) == 1:
                    i["receipt"] = found[0]
                    i["child_start"], i["child_end"] = found[0]["start"], found[0]["end"]
                    i["affinity_verified"] = (
                        found[0].get("affinity") == [int(i["expected_cpuset_cpus"])]
                        and found[0].get("cpuset_cpus_effective") == i["expected_cpuset_cpus"]
                        and found[0].get("cpuset_mems_effective") == i["expected_cpuset_mems"]
                    )
        (out / "orchestrator_events.json").write_text(json.dumps(raw, indent=2))
        good = sum(i["status"] == "SUCCEEDED" and not i["error"] for i in invs)
        row = {
            "phase": name,
            "kind": kind,
            "started_at_epoch": phase_epoch,
            "driver_affinity": sorted(os.sched_getaffinity(0)),
            "total_agents": len(set(a for a, _ in jobs)),
            "total_invocations": len(jobs),
            "creation_wall_s": creation,
            "submission_wall_s": submit_s,
            "total_wall_s": total,
            "throughput_per_s": good / total,
            "successes": good,
            "failures": sum(i["status"] == "FAILED" for i in invs),
            "peak_engine_backed_requests": peak(
                [(i["start"], i["end"]) for i in invs if "start" in i and "end" in i]
            ),
            "peak_children": peak(
                [(i["child_start"], i["child_end"]) for i in invs if "receipt" in i]
            ),
            "host_cpu_s": (
                host_after.ru_utime
                + host_after.ru_stime
                - host_before.ru_utime
                - host_before.ru_stime
            ),
            "host_process_peak_rss_bytes": host_after.ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "scheduler_running_count_peak": max(
                (
                    e["payload"]["running_count"]
                    for e in raw
                    if e["type"] == "scheduler_state" and e["timestamp"] >= start_epoch
                ),
                default=0,
            ),
            "invocations": invs,
            "passed": good == len(jobs) and all(i.get("affinity_verified") for i in invs),
        }
        row["host_cpu_utilization_one_core_pct"] = 100 * row["host_cpu_s"] / total
        report["phases"].append(row)
        save(out, report)
        if any(i["status"] not in {"SUCCEEDED", "FAILED", "CANCELLED", "DESTROYED"} for i in invs):
            raise TimeoutError("Unfinished invocation; stop sweep and inspect saved events")
        return row

    try:
        for n in args.ns:
            t = time.perf_counter()
            agents = [make(f"sweep{n}_{j}", slot=j) for j in range(n)]
            creation = time.perf_counter() - t
            for warmup in range(args.warmups):
                row = phase(
                    f"warmup{n}_{warmup}",
                    [(a, 3) for a in agents],
                    creation if warmup == 0 else 0,
                    kind="warmup",
                )
                if not row["passed"]:
                    raise RuntimeError("Warmup failed; no measured sweep will run")
            for repeat in range(args.repeats):
                row = phase(
                    f"sweep{n}_{repeat}",
                    [(a, 3) for a in agents],
                    creation if repeat == 0 and args.warmups == 0 else 0,
                    kind="measured",
                )
                if not row["passed"]:
                    raise RuntimeError(
                        "Smallest failing case stops further stress; inspect evidence"
                    )
            for a in agents:
                a.destroy().wait()
        if args.scenarios:
            slow, short = make("blocking_slow", "slow"), make("blocking_short", slot=1)
            row = phase("cross_blocking", [(slow, 3), (short, 3)])
            s, f = row["invocations"]
            row["passed"] &= (
                "receipt" in s and "end" in f and s["child_start"] < f["end"] < s["child_end"]
            )
            ordered, other = make("ordered"), make("ordering_other", "slow", slot=1)
            row = phase("ordering", [(ordered, 3), (other, 3), (ordered, 3), (ordered, 3)])
            seq = [i for i in row["invocations"] if i["agent"] == str(ordered.agname)]
            row["passed"] &= (
                all(a.get("end", float("inf")) <= b.get("start", 0) for a, b in zip(seq, seq[1:]))
                and [i.get("receipt", {}).get("after", {}).get("counter") for i in seq] == [1, 2, 3]
                and row["peak_children"] >= 2
            )
        if args.rollback:
            target, other = make("rollback_target"), make("rollback_other", "slow", slot=1)
            baseline = phase("rollback_commit", [(target, 3), (other, 3)])
            if not baseline["passed"]:
                raise RuntimeError("Cannot test rollback without committed baseline")
            change_workload(target, "dirty")
            failed = phase("rollback_failure", [(target, 1), (other, 3)])
            a, b = failed["invocations"]
            failed["passed"] = (
                a["status"] == "FAILED"
                and "max_steps" in (a["error"] or "")
                and a.get("receipt", {}).get("after", {}).get("dirty") is True
                and b["status"] == "SUCCEEDED"
                and failed["peak_children"] == 2
                and all(i.get("affinity_verified") for i in failed["invocations"])
            )
            change_workload(target, "verify")
            change_workload(other, "verify")
            verified = phase("rollback_verify", [(target, 3), (other, 3)])
            verified["passed"] &= (
                verified["invocations"][0].get("receipt", {}).get("before")
                == {"counter": 1, "dirty": False}
                and verified["invocations"][1].get("receipt", {}).get("before")
                == {"counter": 2, "dirty": False}
                and failed["passed"]
            )
        report["status"] = "passed" if all(r["passed"] for r in report["phases"]) else "failed"
    finally:
        # Request all shutdowns before waiting; cleanup errors must not disappear.
        cleanup = []
        closings = []
        for a in owned:
            name = a.sandbox._name if a.sandbox is not None else None
            try:
                closings.append((str(a.agname), name, a.destroy()))
            except Exception as exc:
                cleanup.append({"agent": str(a.agname), "error": repr(exc)})
        for agent_name, container_name, closing in closings:
            try:
                closing.wait(timeout=args.timeout)
                remaining = command(
                    [
                        args.backend,
                        "ps",
                        "-a",
                        "--filter",
                        f"name=^{container_name}$",
                        "--format",
                        "{{.Names}}",
                    ]
                )
                if remaining["returncode"] != 0 or remaining["stdout"].strip():
                    raise RuntimeError(f"Container cleanup could not be verified: {remaining}")
                cleanup.append({"agent": agent_name, "container": container_name, "removed": True})
            except Exception as exc:
                cleanup.append({"agent": agent_name, "error": repr(exc)})
        report["cleanup"] = cleanup
        if any("error" in item for item in cleanup):
            report["status"] = "failed"
        save(out, report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ns", type=int, nargs="+", default=[1, 2])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--layout", type=Path, required=True)
    p.add_argument("--execute", action="store_true", help="Explicitly launch benchmark workloads")
    p.add_argument("--seconds", type=float, default=2)
    p.add_argument("--slow-seconds", type=float, default=15)
    p.add_argument("--scenarios", action="store_true")
    p.add_argument("--rollback", action="store_true")
    p.add_argument("--backend", choices=["docker", "podman"], default="docker")
    p.add_argument("--image", default="agency-sandbox:latest")
    p.add_argument("--cpus", type=float, default=1)
    p.add_argument("--memory", default="512m")
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if not args.execute:
        p.error(
            "No benchmark started. --execute is required; use placement.py for preparation only."
        )
    if (
        not args.ns
        or min(args.ns) < 1
        or len(set(args.ns)) != len(args.ns)
        or args.repeats < 1
        or args.warmups < 0
        or not 0 <= args.seconds <= 100
        or not args.seconds < args.slow_seconds <= 100
    ):
        p.error("Use distinct positive N, repeats >= 1, and 0 <= seconds < slow-seconds <= 100")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "preflight",
        "phases": [],
        "arguments": {**vars(args), "out": str(out), "layout": str(args.layout)},
        "host": {"platform": platform.platform(), "python": sys.version, "cpus": os.cpu_count()},
        "revision": command(["git", "rev-parse", "HEAD"]),
        "working_tree": command(["git", "status", "--short"]),
        "archive_source_revision": (ROOT / "SOURCE_REVISION").read_text().strip()
        if (ROOT / "SOURCE_REVISION").exists()
        else None,
    }
    try:
        layout = json.loads(args.layout.read_text())
        required_levels = list(args.ns)
        if args.scenarios or args.rollback:
            required_levels.append(2)
        apply_driver_affinity(layout, required_levels)
        report["cpu_layout"] = layout
        report["profile_environment"] = {
            name: os.environ.get(name)
            for name in ["AGENCY_PROFILE", "AGENCY_PROFILE_DIR", "AGENCY_PROFILE_SCOPE"]
        }
        if not shutil.which(args.backend):
            raise RuntimeError(f"{args.backend} executable unavailable")
        info = command([args.backend, "info", "--format", "json"])
        report["backend_info"] = info
        if info["returncode"]:
            raise RuntimeError(
                "Sandbox backend unavailable: " + info.get("stderr", info.get("error", ""))
            )
        image = command([args.backend, "image", "inspect", args.image])
        report["image_info"] = image
        if image["returncode"]:
            raise RuntimeError("Sandbox image missing; prepare the documented Agency image first")
        from agency.observability.profiler import agprof

        report["status"] = "running"
        with agprof.workload():
            run(args, out, report)
    except Exception as exc:
        report["status"] = "blocked" if not report["phases"] else "failed"
        report["error"] = repr(exc)
    finally:
        save(out, report)
    print(f"{report['status']}: {out / 'summary.md'}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
