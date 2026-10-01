"""Annotation-independent trajectories and task-clustered offline inference."""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .common import atomic_json

NORMALIZER_VERSION = "actions-v1"
CONTRASTS = [
    ("schema_only", "baseline"),
    ("purpose", "schema_only"),
    ("purpose_workstreams", "purpose"),
    ("purpose", "baseline"),
    ("purpose_workstreams", "baseline"),
]
METRICS = (
    "model_calls",
    "model_attempts",
    "tool_calls",
    "tool_errors",
    "input_tokens",
    "output_tokens",
    "cache_tokens",
    "annotation_characters",
    "annotation_token_estimate",
    "agent_seconds",
    "setup_seconds",
    "evaluation_seconds",
    "end_to_end_seconds",
    "compactions",
    "truncations",
    "validation_cycles",
    "repair_cycles",
    "agent_executions",
)


def normalize(events):
    admissions = {event["event_id"]: event for event in events if event["kind"] == "tool_admission"}
    results = {event["event_id"]: event for event in events if event["kind"] == "tool_result"}
    actions = []
    for event in events:
        if event["kind"] != "tool_annotation":
            continue
        arguments = dict(event.get("arguments") or {})
        arguments.pop("_agency", None)
        tool = event["tool_name"]
        targets = []
        for key in ("file_path", "path", "document_id", "url", "role"):
            if isinstance(arguments.get(key), str):
                targets.append(key + ":" + arguments[key])
        operation = {
            "read": "read",
            "write": "write",
            "edit": "edit",
            "glob": "search",
            "grep": "search",
            "bash": "shell",
            "webfetch": "fetch",
            "validate_ledger": "validation",
        }.get(tool, "unknown")
        command = arguments.get("command")
        validation = operation == "validation" or (
            tool == "bash"
            and isinstance(command, str)
            and bool(re.search(r"\b(pytest|rustc|gcc|clang|make|cargo\s+(test|check))\b", command))
        )
        result = results.get(event["event_id"], {})
        admission = admissions.get(event["event_id"], {})
        category = result.get("category", "interrupted")
        if admission.get("decision", {}).get("decision") == "deny":
            category = "denied"
        actions.append(
            {
                "version": NORMALIZER_VERSION,
                "tool": tool,
                "arguments": arguments,
                "arguments_json": json.dumps(arguments, sort_keys=True),
                "targets": sorted(targets),
                "operation": operation,
                "validation": validation,
                "result_category": category,
                "timestamp_ns": event.get("timestamp_ns"),
                "duration_ns": result.get("duration_ns"),
                "agent_id": event.get("agent_id"),
                "run_id": event.get("run_id"),
                "call_id": admission.get("call_id") or result.get("call_id"),
                "model_tool_call_id": event.get("model_tool_call_id"),
                "exchange": event.get("exchange"),
                "batch_size": event.get("batch_size"),
                "execution": event.get("execution"),
                "event_id": event["event_id"],
            }
        )
    return actions


def validation_cycles(actions):
    count = 0
    previous = False
    for action in actions:
        if action["validation"] and not previous:
            count += 1
        previous = action["validation"]
    return count


def repair_cycles(actions):
    state = {}
    count = 0
    for action in actions:
        role = action.get("agent_id")
        pending_error, repaired = state.get(role, (False, False))
        if action["result_category"] in ("error", "denied"):
            pending_error = True
        if pending_error and action["operation"] in ("edit", "write"):
            repaired = True
        if repaired and action["validation"]:
            count += 1
            pending_error, repaired = False, False
        state[role] = pending_error, repaired
    return count


def sequence_distance(left, right, *, lcs=False):
    row = [0] * (len(right) + 1) if lcs else list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        new = [0 if lcs else i]
        for j, b in enumerate(right, 1):
            if lcs:
                new.append(row[j - 1] + 1 if a == b else max(row[j], new[-1]))
            else:
                new.append(min(row[j] + 1, new[-1] + 1, row[j - 1] + (a != b)))
        row = new
    denominator = max(len(left), len(right), 1)
    return (1 - row[-1] / denominator) if lcs and (left or right) else row[-1] / denominator


def compare_actions(left, right):
    tools_left = [action["tool"] for action in left]
    tools_right = [action["tool"] for action in right]
    operations_left = [
        (action["tool"], tuple(action["targets"]), action["arguments_json"]) for action in left
    ]
    operations_right = [
        (action["tool"], tuple(action["targets"]), action["arguments_json"]) for action in right
    ]
    targets_left = set(itertools.chain.from_iterable(action["targets"] for action in left))
    targets_right = set(itertools.chain.from_iterable(action["targets"] for action in right))
    union = targets_left | targets_right
    return {
        "tool_lcs_distance": sequence_distance(tools_left, tools_right, lcs=True),
        "operation_edit_distance": sequence_distance(operations_left, operations_right),
        "target_overlap": len(targets_left & targets_right) / len(union) if union else None,
        "action_count_difference": len(left) - len(right),
        "validation_cycle_difference": validation_cycles(left) - validation_cycles(right),
    }


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summary(values):
    values = [value for value in values if value is not None]
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": percentile(values, 0.5),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
    }


def cluster_interval(task_means, *, seed=0, samples=2000):
    values = list(task_means.values())
    result = summary(values)
    result["task_clusters"] = len(values)
    result["interval"] = None
    if len(values) < 2:
        return result
    randomizer = random.Random(seed)
    boot = [statistics.mean(randomizer.choices(values, k=len(values))) for _ in range(samples)]
    result["interval"] = [percentile(boot, 0.025), percentile(boot, 0.975)]
    return result


def required_tasks(*, margin, task_difference_sd, power=0.8, alpha=0.05):
    """Normal approximation using paired TASK-average differences, not trial counts."""
    if margin <= 0 or task_difference_sd < 0 or not 0 < power < 1 or not 0 < alpha < 1:
        raise ValueError("Positive margin, nonnegative SD, and probabilities in (0,1) required")
    normal = statistics.NormalDist()
    z = normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)
    return max(2, math.ceil((z * task_difference_sd / margin) ** 2))


def estimate_cost(row, pricing):
    if not pricing.get("version") or not pricing.get("currency") == "USD":
        raise ValueError("Pricing requires an explicit version and USD currency")
    rates = pricing["models"].get(row["model_id"])
    if rates is None or row.get("input_tokens") is None or row.get("output_tokens") is None:
        return None
    if "cached_input_per_million" in rates and row.get("cache_tokens") is None:
        return None
    cached = row.get("cache_tokens") or 0
    return (
        (row["input_tokens"] - cached) * rates["input_per_million"]
        + cached * rates.get("cached_input_per_million", rates["input_per_million"])
        + row["output_tokens"] * rates["output_per_million"]
    ) / 1_000_000


def analyze(manifest, rows, *, bootstrap_samples=2000, pricing=None, tokenizer_file=None):
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    # A recovered interruption is an assigned attempt too. The original attempt
    # defines ITT; later recovery attempts are counted separately, never replacing it.
    first = {}
    for row in rows:
        first.setdefault(row["trial_id"], row)
    assigned = list(first.values())
    tokenizer_identity = None
    if tokenizer_file is not None:
        from .tokenization import annotation_tokens, local_tokenizer

        encode, tokenizer_identity = local_tokenizer(tokenizer_file)
        for row in assigned:
            row["annotation_token_estimate"] = annotation_tokens(row.get("events", []), encode)
    groups = defaultdict(list)
    for row in assigned:
        row["validation_cycles"] = validation_cycles(row.get("actions", []))
        row["repair_cycles"] = repair_cycles(row.get("actions", []))
        if pricing is not None:
            row["estimated_cost_usd"] = estimate_cost(row, pricing)
        groups[(row["suite"], row["model_id"])].append(row)
    report = {
        "planned_trials": len(manifest["schedule"]),
        "attempt_rows": len(rows),
        "recovery_attempts": len(rows) - len(assigned),
        "normalizer_version": NORMALIZER_VERSION,
        "failures": dict(Counter(row.get("failure") for row in rows)),
        "groups": [],
        "equivalence_claim": False,
        "inference": "exploratory task-clustered percentile bootstrap",
        "infrastructure_exclusion_rule": "complete-case sensitivity excludes infrastructure and missing verdicts only",
        "pricing_version": pricing.get("version") if pricing else None,
        "tokenizer": tokenizer_identity,
    }
    for (suite, model), group in groups.items():
        result = {
            "suite": suite,
            "model_id": model,
            "arms": {},
            "contrasts": [],
            "trajectory_pairs": [],
        }
        for arm in ("baseline", "schema_only", "purpose", "purpose_workstreams"):
            arm_rows = [row for row in group if row["arm"] == arm]
            observed = [row for row in arm_rows if isinstance(row.get("success"), bool)]
            successes = sum(row.get("success") is True for row in arm_rows)
            unknown = len(arm_rows) - len(observed)
            actions = list(
                itertools.chain.from_iterable(row.get("actions", []) for row in arm_rows)
            )
            observed_tasks = defaultdict(list)
            for row in observed:
                observed_tasks[row["task_id"]].append(int(row["success"]))
            success_interval = cluster_interval(
                {task: statistics.mean(values) for task, values in observed_tasks.items()},
                samples=bootstrap_samples,
            )
            result["arms"][arm] = {
                "assigned": len(arm_rows),
                "observed": len(observed),
                "unknown": unknown,
                "success_bounds": [successes / len(arm_rows), (successes + unknown) / len(arm_rows)]
                if arm_rows
                else None,
                "observed_success_rate": success_interval["mean"],
                "observed_success_task_clustered": success_interval,
                "metrics": {
                    metric: summary([row.get(metric) for row in arm_rows]) for metric in METRICS
                },
                "annotation_valid": sum(row.get("annotation_valid", 0) for row in arm_rows),
                "annotation_missing": sum(row.get("annotation_missing", 0) for row in arm_rows),
                "annotation_malformed": sum(row.get("annotation_malformed", 0) for row in arm_rows),
                "normalizer_coverage": sum(action["operation"] != "unknown" for action in actions)
                / len(actions)
                if actions
                else None,
            }
        for active, control in CONTRASTS:
            task_pairs = defaultdict(list)
            complete_pairs = defaultdict(list)
            metrics = {metric: defaultdict(list) for metric in METRICS}
            blocks = defaultdict(dict)
            for row in group:
                blocks[row["block_id"]][row["arm"]] = row
            missing_pairs = 0
            for block in blocks.values():
                a, b = block.get(active), block.get(control)
                if a is None or b is None:
                    missing_pairs += 1
                    continue
                task_id = a["task_id"]
                # Conservative assigned accounting, explicitly labeled. Unknown
                # verdicts remain unknown in exports and success bounds above.
                task_pairs[task_id].append(
                    int(a.get("success") is True) - int(b.get("success") is True)
                )
                if isinstance(a.get("success"), bool) and isinstance(b.get("success"), bool):
                    complete_pairs[task_id].append(int(a["success"]) - int(b["success"]))
                else:
                    missing_pairs += 1
                for metric in METRICS:
                    if a.get(metric) is not None and b.get(metric) is not None:
                        metrics[metric][task_id].append(a[metric] - b[metric])
            task_means = {task: statistics.mean(values) for task, values in task_pairs.items()}
            complete_means = {
                task: statistics.mean(values) for task, values in complete_pairs.items()
            }
            paired_metrics = {}
            for metric, task_values in metrics.items():
                paired_metrics[metric] = {
                    "paired_blocks": summary(
                        list(itertools.chain.from_iterable(task_values.values()))
                    ),
                    "task_clustered": cluster_interval(
                        {task: statistics.mean(values) for task, values in task_values.items()},
                        samples=bootstrap_samples,
                    ),
                }
            result["contrasts"].append(
                {
                    "active": active,
                    "control": control,
                    "assigned_confirmed_success_difference": cluster_interval(
                        task_means, samples=bootstrap_samples
                    ),
                    "complete_case_sensitivity": cluster_interval(
                        complete_means, samples=bootstrap_samples
                    ),
                    "missing_pairs": missing_pairs,
                    "metrics": paired_metrics,
                    "task_difference_sd": statistics.stdev(task_means.values())
                    if len(task_means) > 1
                    else None,
                }
            )
        task_runs = defaultdict(list)
        for row in group:
            # An empty completed trajectory is valid. A missing trace is not.
            if (
                row.get("attempt")
                and row.get("state") == "completed"
                and any(
                    event["kind"] in ("treatment", "model_exchange", "tool_annotation")
                    for event in row.get("events", [])
                )
            ):
                task_runs[row["task_id"]].append(row)
        pairs = defaultdict(lambda: defaultdict(list))
        per_agent = defaultdict(lambda: defaultdict(list))
        for task_id, runs in task_runs.items():
            for a, b in itertools.combinations(runs, 2):
                key = tuple(sorted((a["arm"], b["arm"])))
                comparison = compare_actions(a["actions"], b["actions"])
                comparison["artifact_different"] = (
                    None
                    if a.get("artifact_hash") is None or b.get("artifact_hash") is None
                    else a["artifact_hash"] != b["artifact_hash"]
                )
                comparison["final_output_different"] = a.get("final_output_hash") != b.get(
                    "final_output_hash"
                )
                pairs[key][task_id].append(comparison)
                roles = {action["agent_id"] for action in a["actions"]} | {
                    action["agent_id"] for action in b["actions"]
                }
                for role in roles:
                    per_agent[(key, role)][task_id].append(
                        compare_actions(
                            [action for action in a["actions"] if action["agent_id"] == role],
                            [action for action in b["actions"] if action["agent_id"] == role],
                        )
                    )
        baseline = pairs.get(("baseline", "baseline"), {})
        for pair, task_comparisons in pairs.items():
            record = {
                "arms": pair,
                "pairs": sum(map(len, task_comparisons.values())),
                "metrics": {},
                "excess_over_baseline": {},
            }
            metric_names = next(iter(task_comparisons.values()))[0].keys()
            for metric in metric_names:
                means = {}
                excess = {}
                for task_id, comparisons in task_comparisons.items():
                    values = [c[metric] for c in comparisons if c[metric] is not None]
                    if values:
                        means[task_id] = statistics.mean(values)
                        base_values = [
                            c[metric] for c in baseline.get(task_id, []) if c[metric] is not None
                        ]
                        if base_values:
                            excess[task_id] = means[task_id] - statistics.mean(base_values)
                record["metrics"][metric] = cluster_interval(means, samples=bootstrap_samples)
                record["excess_over_baseline"][metric] = cluster_interval(
                    excess, samples=bootstrap_samples
                )
            result["trajectory_pairs"].append(record)
        result["per_agent_trajectory"] = []
        for (pair, role), task_comparisons in per_agent.items():
            means = {
                task: statistics.mean(c["operation_edit_distance"] for c in comparisons)
                for task, comparisons in task_comparisons.items()
            }
            result["per_agent_trajectory"].append(
                {
                    "arms": pair,
                    "role": role,
                    "operation_edit_distance": cluster_interval(means, samples=bootstrap_samples),
                }
            )
        report["groups"].append(result)
    return report


def export_report(directory, manifest, rows, **kwargs):
    directory = Path(directory)
    report = analyze(manifest, rows, **kwargs)
    atomic_json(directory / "analysis.json", report)
    scalar_rows = [
        {key: value for key, value in row.items() if key not in ("events", "actions")}
        for row in rows
    ]
    atomic_json(directory / "results.json", scalar_rows)
    if scalar_rows:
        fields = sorted(set(itertools.chain.from_iterable(row.keys() for row in scalar_rows)))
        with (directory / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(scalar_rows)
    lines = [
        "# Tool annotation experiment",
        "",
        f"Planned trials: {report['planned_trials']}; attempt rows: {report['attempt_rows']}.",
        "Unknown verdicts and usage remain unknown. Repetitions are averaged within tasks.",
        "Intervals resample whole tasks. Exploratory comparisons; no equivalence claim.",
        "",
    ]
    for group in report["groups"]:
        lines += [
            f"## {group['suite']} / {group['model_id']}",
            "",
            "| Arm | Observed / assigned | Observed success | Unknown |",
            "|---|---:|---:|---:|",
        ]
        for arm, data in group["arms"].items():
            lines.append(
                f"| {arm} | {data['observed']} / {data['assigned']} | {data['observed_success_rate']} | {data['unknown']} |"
            )
        lines += ["", "Correctness differences and task-clustered intervals:", ""]
        for contrast in group["contrasts"]:
            effect = contrast["assigned_confirmed_success_difference"]
            lines.append(
                f"- {contrast['active']} − {contrast['control']}: confirmed-success difference {effect['mean']}; 95% interval {effect['interval']}; complete-case sensitivity {contrast['complete_case_sensitivity']['mean']}."
            )
        lines += ["", "Trajectory distance and excess over baseline variability:", ""]
        for comparison in group["trajectory_pairs"]:
            distance = comparison["metrics"]["operation_edit_distance"]
            excess = comparison["excess_over_baseline"]["operation_edit_distance"]
            lines.append(
                f"- {comparison['arms']}: distance {distance['mean']}, interval {distance['interval']}; excess {excess['mean']}, interval {excess['interval']}."
            )
        lines += [
            "",
            "Overhead (observed input/output tokens, execution seconds; unknowns excluded):",
            "",
        ]
        for arm, data in group["arms"].items():
            metrics = data["metrics"]
            lines.append(
                f"- {arm}: input {metrics['input_tokens']['mean']}; output {metrics['output_tokens']['mean']}; agent seconds {metrics['agent_seconds']['mean']}; valid/missing/malformed labels {data['annotation_valid']}/{data['annotation_missing']}/{data['annotation_malformed']}."
            )
        lines.append("")
    (directory / "report.md").write_text("\n".join(lines))
    return report


def review_export(directory, rows, *, seed=0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(seed)
    shuffled = list(rows)
    randomizer.shuffle(shuffled)
    raw, labels, key = [], [], []
    for index, row in enumerate(shuffled):
        packet_id = f"packet-{index:04d}"
        actions = [
            {
                key: action[key]
                for key in ("tool", "arguments", "result_category", "agent_id", "exchange")
            }
            for action in row.get("actions", [])
        ]
        raw.append(
            {
                "packet_id": packet_id,
                "actions": actions,
                "human_workstream_assignments": [],
                "review_notes": "",
            }
        )
        annotations = [
            event["annotation"]
            for event in row.get("events", [])
            if event["kind"] == "tool_annotation"
        ]
        labels.append(
            {
                "packet_id": packet_id,
                "agent_stated_labels": annotations,
                "utility_rating": None,
                "consistency_rating": None,
            }
        )
        key.append(
            {
                "packet_id": packet_id,
                "trial_id": row["trial_id"],
                "arm": row["arm"],
                "attempt": row.get("attempt"),
            }
        )
    atomic_json(directory / "raw-blinded.json", raw)
    atomic_json(directory / "labels-phase-two.json", labels)
    atomic_json(directory / "private-unblinding-key.json", key)
