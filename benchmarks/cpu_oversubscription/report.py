"""Explicit untimed official evaluation, resumable grades, and paired comparison."""

import importlib.metadata
import subprocess
from pathlib import Path

from .config import atomic_json, safe_json
from .linux import experiment_lock, processes, require_linux
from .runner import child_env, finalize, read


def evaluate(directory):
    require_linux()
    from benchmarks.tool_annotation_effect.adapters import SweBenchAdapter

    directory = Path(directory).resolve()
    with experiment_lock():
        if processes():
            raise RuntimeError("Stop/recover all experiment workloads before offline grading")
        result = finalize(directory)
        if not result["termination"].get("cleanup_complete"):
            raise RuntimeError("Cleanup was incomplete; recover before evaluation")
        plan = read(directory / "plan.json")
        if importlib.metadata.version("swebench") != plan["provenance"]["versions"]["swebench"]:
            raise RuntimeError("Restore the recorded SWE-bench version before grading")
        from .workload import verify_snapshot

        verify_snapshot(plan["config"], plan["manifest"])

        def official_command(args, **kwargs):
            # The official loader accepts local JSON datasets. Freeze gold/tests to
            # the same revision used for assignment instead of fetching moving HEAD.
            args = list(args)
            args[args.index("--dataset_name") + 1] = plan["config"]["dataset_snapshot"]
            return subprocess.run(args, env=child_env(), **kwargs)

        for row in result["tasks"]:
            target = directory / row["attempt_id"]
            if row["status"] != "completed":
                continue
            if (target / "grade.json").exists() and read(target / "grade.json")[
                "success"
            ] is not None:
                continue
            if not (target / "prediction.jsonl").exists():
                safe_json(
                    target / "grade.json",
                    {"success": None, "failure": "infrastructure", "error": "missing prediction"},
                )
                continue
            task = plan["manifest"]["agents"][row["agent"]][row["index"]]
            grade = SweBenchAdapter().evaluate(
                {"id": task["instance_id"], "dataset": plan["manifest"]["dataset"]},
                target / "prediction.jsonl",
                target,
                command_runner=official_command,
            )
            if grade["success"] is None:
                # New harness releases use logs/evaluation rather than run_evaluation.
                run_id = grade["evaluation_run_id"]
                report_root = target / "logs" / "evaluation" / run_id
                for path in sorted(report_root.rglob("*.json")):
                    imported = SweBenchAdapter().import_report(
                        {"id": task["instance_id"]}, read(path)
                    )
                    if imported["success"] is not None:
                        grade = {**imported, "evaluation_run_id": run_id}
                        break
            safe_json(target / "grade.json", grade)
        return finalize(directory)


def compare(directories, output):
    results = [read(Path(directory) / "results.json") for directory in directories]
    if not results or len({result["comparison_identity"] for result in results}) != 1:
        raise ValueError("Configurations, assignments, images, software, or hardware differ")
    pairs = {}
    for result in results:
        if result["metrics"]["grading_pending"]:
            raise ValueError("Complete official grading before comparison")
        if result["termination"]["reason"] not in ("completed", "deadline") or not result[
            "termination"
        ].get("cleanup_complete"):
            raise ValueError(
                "Infrastructure/operator interruption invalidates a baseline comparison"
            )
        pair = pairs.setdefault(result["trial_id"], {})
        if result["condition"] in pair:
            raise ValueError("Duplicate condition for a trial ID")
        pair[result["condition"]] = result
    if any(set(pair) != {"DCB", "LSB"} for pair in pairs.values()):
        raise ValueError("Each independent trial needs one DCB and one LSB result")
    lines = [
        "# CPU oversubscription comparison",
        "",
        "Pass rate uses all 30 assignments. Throughput counts completed attempts only.",
        "Mean duration includes observed censored durations; never-started tasks have no duration.",
        "",
        "| Trial | Condition | Deadline reached | Pass rate | Mean seconds | Completed/hour | Censored | Never started |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    paired = []
    for trial, pair in sorted(pairs.items()):
        for condition in ("DCB", "LSB"):
            result = pair[condition]
            m = result["metrics"]
            mean = (
                f"{m['average_task_wall_seconds']:.3f}"
                if m["average_task_wall_seconds"] is not None
                else "n/a"
            )
            lines.append(
                f"| {trial} | {condition} | {'**YES**' if result['deadline_reached'] else 'no'} | "
                f"{m['pass_rate']:.1%} | {mean} | {m['throughput_attempts_per_hour']:.3f} | "
                f"{m['censored_tasks']} | {m['unattempted_tasks']} |"
            )
        dcb, lsb = (pair[c]["metrics"] for c in ("DCB", "LSB"))
        paired.append(
            {
                "trial_id": trial,
                "pass_rate_difference_lsb_minus_dcb": lsb["pass_rate"] - dcb["pass_rate"],
                "throughput_ratio_lsb_over_dcb": lsb["throughput_attempts_per_hour"]
                / dcb["throughput_attempts_per_hour"]
                if dcb["throughput_attempts_per_hour"]
                else None,
                "DCB": dcb,
                "LSB": lsb,
            }
        )
    lines += [
        "",
        "Successful tasks/hour and paired throughput ratios are in comparison.json.",
        "Censored means are descriptive observations, not estimates of uncensored completion time.",
        "",
    ]
    aggregate = {}
    for condition in ("DCB", "LSB"):
        selected = [result for result in results if result["condition"] == condition]
        elapsed = sum(result["metrics"]["batch_elapsed_seconds"] for result in selected)
        completed = sum(result["metrics"]["completed_attempts"] for result in selected)
        successful = sum(result["metrics"]["successful_attempts"] for result in selected)
        durations = []
        for result in selected:
            durations.extend(
                row["duration_seconds"]
                for row in result["tasks"]
                if row["duration_seconds"] is not None
            )
        aggregate[condition] = {
            "trials": len(selected),
            "assigned": 30 * len(selected),
            "pass_rate": successful / (30 * len(selected)),
            "average_task_wall_seconds": sum(durations) / len(durations) if durations else None,
            "throughput_attempts_per_hour": completed * 3600 / elapsed if elapsed else 0,
            "successful_tasks_per_hour": successful * 3600 / elapsed if elapsed else 0,
            "deadline_reached_trials": sum(result["deadline_reached"] for result in selected),
        }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "comparison.json",
        {"trials": paired, "conditions": results, "aggregate": aggregate},
    )
    (output / "comparison.md").write_text("\n".join(lines))
    return {"report": str(output / "comparison.md"), "trials": len(paired)}
