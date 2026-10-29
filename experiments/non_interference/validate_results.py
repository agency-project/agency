"""Validate the complete exported cohort and record the local analysis runtime."""

from collections import Counter
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root):
    read = lambda path: json.loads(path.read_text())
    frozen = read(root / "freeze_metadata.json")
    manifest = read(root / "task_manifest.json")
    config = read(root / "configuration.json")
    exported = read(root / "export_manifest.json")
    mismatches = [
        name
        for name, item in exported["files"].items()
        if not (root / name).is_file() or digest(root / name) != item["sha256"]
    ]
    rows = list(csv.DictReader((root / "metrics.csv").open()))
    raw = [json.loads(line) for line in (root / "runs.jsonl").read_text().splitlines()]
    latest = {}
    for result in raw:
        if (
            result["run_id"] not in latest
            or result["attempt"] > latest[result["run_id"]]["attempt"]
        ):
            latest[result["run_id"]] = result
    reports = [(path, read(path)) for path in (root / "attempts").glob("*/*/official-report.json")]
    starts = [
        json.loads(line)
        for line in (root / "scheduling_events.jsonl").read_text().splitlines()
        if json.loads(line).get("diagnostic") is False and json.loads(line).get("event") == "start"
    ]
    counts = Counter((r["instance_id"], r["arm"]) for r in rows)
    checks = {
        "export_file_hashes_match": not mismatches,
        "manifest_hash_matches_freeze": digest(root / "task_manifest.json")
        == frozen["manifest_sha256"]
        == config["manifest_sha256"],
        "expected_unique_raw_and_normalized_runs": len(rows)
        == len({r["run_id"] for r in raw})
        == len({r["run_id"] for r in rows})
        == config["executions"],
        "configured_runs_per_frozen_task_arm": counts
        == Counter(
            {
                (t["instance_id"], arm): config["repetitions"]
                for t in manifest["tasks"]
                for arm in ("standalone", "agency")
            }
        ),
        "all_attempt_records_preserved": len(raw)
        == len({(r["run_id"], r["attempt"]) for r in raw})
        == len(list(root.glob("attempts/*/*/summary.json"))),
        "every_run_officially_graded": len(latest) == config["executions"]
        and all(
            isinstance(read(root / r["directory"] / "official-report.json").get("resolved"), bool)
            and not read(root / r["directory"] / "official-report.json").get("grading_error")
            for r in latest.values()
        ),
        "grading_values_match_metrics": all(
            read(root / r["directory"] / "official-report.json").get("resolved")
            == (next(x for x in rows if x["run_id"] == r["run_id"])["resolved"] == "True")
            for r in latest.values()
        ),
        "execution_concurrency_within_budget": len(starts) == len(raw)
        and len({e["run_id"] for e in starts}) == config["executions"]
        and all(
            e["active_total"] <= 4 and e["active_agency"] <= 2 and e["active_standalone"] <= 2
            for e in starts
        ),
    }
    audit = read(root / "fidelity_audit.json")
    checks["full_trace_audit_passes"] = all(
        not audit[key]
        for key in (
            "request_model_mismatch_runs",
            "recording_coverage_mismatch_runs",
            "request_mismatch_runs",
            "response_mismatch_runs",
            "resource_or_starting_state_mismatch_runs",
        )
    ) and all(x["one_identical_initial_request"] for x in audit["initial_requests"].values())
    output = {
        "checks": checks,
        "passed": all(checks.values()),
        "export_hash_mismatches": mismatches,
        "raw_records": len(raw),
        "normalized_records": len(rows),
        "official_reports": len(reports),
        "cost_usage_complete_records": sum(r["usage_complete"] == "True" for r in rows),
        "unknown_cost_runs": [r["run_id"] for r in rows if r["usage_complete"] != "True"],
        "execution_status_counts": dict(Counter(r["status"] for r in rows)),
        "resolved_counts_by_arm": {
            arm: dict(Counter(r["resolved"] for r in rows if r["arm"] == arm))
            for arm in ("standalone", "agency")
        },
    }
    (root / "run_integrity.json").write_text(json.dumps(output, indent=2) + "\n")
    runtime = {
        "scope": "Offline statistical analysis only; all measured execution and official grading ran on the configured remote host",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "matplotlib", "pyarrow")
        },
        "bootstrap_seed": config["bootstrap_seed"],
        "bootstrap_resamples": config["bootstrap_resamples"],
        "source_and_input_sha256": {
            name: digest(root / name)
            for name in (
                "analysis.py",
                "analysis_secondary.py",
                "audit_results.py",
                "report.py",
                "validate_results.py",
                "task_manifest.json",
                "configuration.json",
                "pricing.json",
                "runs.jsonl",
                "metrics.csv",
                "task_summary.csv",
                "export_manifest.json",
            )
        },
        "usage_policy": "Unknown forwarded-request usage is not imputed; all cohort outcomes retained; exploratory cost analysis uses complete records only",
    }
    (root / "analysis_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if not output["passed"]:
        raise SystemExit("Final evidence validation failed; inspect run_integrity.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent)
