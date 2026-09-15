"""Summarize COW-only cohorts and compare them with retained checkpoint results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def median(values):
    return statistics.median(values) if values else None


def trace_duration(path: Path, name: str):
    events = json.loads(path.read_text()).get("traceEvents", [])
    values = [
        event["dur"] / 1_000_000 for event in events if event.get("name") == name and "dur" in event
    ]
    return sum(values) if values else None


def cohort_rows(root: Path):
    rows = []
    for run_path in sorted((root / "runs").glob("*/run.json")):
        run = json.loads(run_path.read_text())
        if run.get("status") != "completed":
            continue
        report = json.loads((root / run["checkpoint_report_path"]).read_text())
        trace = root / run["profiler_path"] / "agprof.trace.json"
        # sandbox:ensure_daemon encloses harness:launch and
        # harness:startup_ready.  Use its wall duration rather than summing
        # nested spans and double-counting startup work.
        fresh_harness = trace_duration(trace, "sandbox:ensure_daemon")
        row = {
            "cohort": root.name,
            "runtime": run["sandbox_runtime"],
            "requested_bytes": run["requested_bytes"],
            "checkpoint_total_seconds": run["checkpoint_seconds"],
            "checkpoint_fs_snapshot_seconds": report["fs_snapshot_seconds"],
            "restore_fs_seconds": run["restore_seconds"],
            "container_start_seconds": run["container_start_seconds"],
            "fresh_harness_pty_seconds": fresh_harness,
            "payload_generation_seconds": run["payload"]["generation_seconds"],
        }
        row["hibernate_resume_seconds"] = (
            row["checkpoint_total_seconds"]
            + row["restore_fs_seconds"]
            + row["container_start_seconds"]
        )
        row["hibernate_resume_fresh_harness_seconds"] = (
            row["hibernate_resume_seconds"] + fresh_harness
        )
        rows.append(row)
    return rows


def summarize(rows):
    output = []
    keys = (
        "checkpoint_total_seconds",
        "checkpoint_fs_snapshot_seconds",
        "restore_fs_seconds",
        "container_start_seconds",
        "fresh_harness_pty_seconds",
        "payload_generation_seconds",
        "hibernate_resume_seconds",
        "hibernate_resume_fresh_harness_seconds",
    )
    groups = sorted({(row["cohort"], row["runtime"], row["requested_bytes"]) for row in rows})
    for cohort, runtime, size in groups:
        matches = [
            row
            for row in rows
            if (row["cohort"], row["runtime"], row["requested_bytes"]) == (cohort, runtime, size)
        ]
        item = {"cohort": cohort, "runtime": runtime, "requested_bytes": size, "n": len(matches)}
        for key in keys:
            values = [row[key] for row in matches if row[key] is not None]
            item["median_" + key] = median(values)
            item["min_" + key] = min(values) if values else None
            item["max_" + key] = max(values) if values else None
        output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--prior-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for root in args.roots for row in cohort_rows(root)]
    summary = summarize(rows)
    prior = json.loads(args.prior_summary.read_text())
    retained = [
        row
        for row in prior
        if row["root"].endswith(("bench-docker-image-commit", "bench-cow-criu"))
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "measurements.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "summary.json").write_text(
        json.dumps({"cow_only": summary, "retained_prior": retained}, indent=2) + "\n"
    )
    if rows:
        with (args.output / "measurements.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
