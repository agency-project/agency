"""Resume prepared cohorts without changing their pinned measurement runners.

Only an exact model refusal or a START_NS-only tool result may be archived and
retried. Checkpoint, integrity and cleanup failures still stop the run.
All original measurements, prompts, ordering and spread rules remain intact.
"""

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import subprocess
import time


def load_runner(root):
    spec = importlib.util.spec_from_file_location("pinned_checkpoint_runner", root / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke(runner, root, key_file, condition, validation=False):
    label = (
        "validation"
        if validation
        else (
            f"order-{condition['order']:02d}-bytes-{condition['requested_bytes']}-r{condition['replicate']}"
        )
    )
    directory = root / label if validation else root / "runs" / label
    for attempt in range(3):
        if not directory.exists():
            try:
                return runner.invoke_one(root, key_file, condition, validation=validation)
            except subprocess.CalledProcessError:
                if not (directory / "run.json").exists():
                    raise
        data = json.loads((directory / "run.json").read_text())
        result_path = directory / "call-result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        summary = result.get("summary", "").replace("’", "'").strip()
        if summary.lower() == "i'm sorry, but i can't comply with that request.":
            reason = "Model refused; required payload command did not produce a result"
        elif re.fullmatch(r"START_NS=\d+", summary):
            reason = "Native CLI tool returned only START_NS; payload completion record missing"
        else:
            reason = None
        if (
            data.get("status") != "failed"
            or data.get("cleanup_error")
            or data.get("exception_message")
            != "Could not recover the unique payload-generation result"
            or reason is None
        ):
            raise RuntimeError(f"Not an eligible invalid-workload retry: {directory}")
        manifest = json.loads((root / "manifest.json").read_text())
        archive = root / "invalid-workload-attempts"
        archive.mkdir(exist_ok=True)
        invalid = manifest.setdefault("invalid_workload_attempts", [])
        destination = archive / f"{label}-attempt-{1 + len(invalid)}"
        directory.rename(destination)
        invalid.append(
            {
                **condition,
                "path": str(destination.relative_to(root)),
                "reason": reason,
                "excluded_from_dataset": True,
                "original_status": data["status"],
            }
        )
        runner.save(root / "manifest.json", manifest)
        if attempt == 2:
            raise RuntimeError(
                "Three invalid workload attempts; stopping without changing the workload"
            )
    raise AssertionError("unreachable")


def resume(root, key_file):
    runner = load_runner(root)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("status") == "completed":
        return
    runner.validate_checkpoint_comparison(manifest["config"])
    manifest["resume_controller"] = {
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reason": "Resume unchanged measurements after a recorded invalid workload attempt",
    }
    runner.save(path, manifest)
    if manifest.get("validation", {}).get("status") != "completed":
        row = invoke(
            runner, root, key_file, {"requested_bytes": 4096, "replicate": 0, "order": 0}, True
        )
        if row["status"] != "completed":
            raise RuntimeError("Validation failed")
        manifest = json.loads(path.read_text())
        manifest["validation"] = {
            "status": "completed",
            "requested_bytes": 4096,
            "run_path": "validation",
            "excluded_from_dataset": True,
        }
        runner.save(path, manifest)

    def conditions(items):
        for condition in items:
            current = json.loads(path.read_text())
            existing = [
                r for r in current["actual_execution_order"] if r["order"] == condition["order"]
            ]
            if existing:
                if (
                    len(existing) != 1
                    or existing[0]["status"] != "completed"
                    or any(existing[0][key] != condition[key] for key in condition)
                ):
                    raise RuntimeError("Existing condition identity/status mismatch")
                continue
            row = invoke(runner, root, key_file, condition)
            if row["status"] != "completed" or row.get("cleanup_error"):
                raise RuntimeError(f"Condition failed: {row['run_id']}")
            current = json.loads(path.read_text())
            current["actual_execution_order"].append(
                {**condition, "run_id": row["run_id"], "status": row["status"]}
            )
            runner.save(path, current)

    conditions(manifest["initial_conditions"])
    by_size = {}
    for run in (root / "runs").glob("*/run.json"):
        row = json.loads(run.read_text())
        if row["replicate"] > 3:
            continue
        if row["status"] != "completed":
            raise RuntimeError(f"Incomplete initial condition: {run}")
        by_size.setdefault(row["requested_bytes"], []).append(
            row.get("checkpoint_seconds", row.get("report_commit_seconds"))
        )
    triggered = []
    for size, durations in sorted(by_size.items()):
        if len(durations) != 3:
            raise RuntimeError(f"Expected three initial measurements for {size}")
        median = sorted(durations)[1]
        if (max(durations) - min(durations)) / median > 0.10:
            triggered.append(size)
    extras = (
        runner.shuffled_conditions(triggered, (4, 5), runner.ORDER_SEED + 1) if triggered else []
    )
    for condition in extras:
        condition["order"] += len(manifest["initial_conditions"])
    manifest = json.loads(path.read_text())
    if manifest.get("extra_conditions") and manifest["extra_conditions"] != extras:
        raise RuntimeError("Extra-condition policy drift")
    manifest["extra_conditions"] = extras
    runner.save(path, manifest)
    conditions(extras)
    manifest = json.loads(path.read_text())
    manifest.update(
        status="completed",
        spread_triggered_extra_sizes_bytes=triggered,
        completed_wall_ns=time.time_ns(),
    )
    runner.save(path, manifest)
    (root / "REMOTE_COMPLETE").write_text(f"completed_wall_ns={time.time_ns()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    for name in ("bench-cow-criu", "bench-docker-image-commit", "bench-podman-image-commit"):
        resume(args.root / name, args.key_file)
    (args.root / "ALL_COHORTS_COMPLETE").write_text(str(time.time()) + "\n")
