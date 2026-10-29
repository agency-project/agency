"""Fixed-task stratified bootstrap, cost accounting, pilot figures and TOST planning.

Usage: python analysis.py [study_directory]. No run is fabricated or outcome-filtered.
"""

from pathlib import Path
import csv
import json
import math
import sys

import numpy as np


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def call_cost(call, pricing):
    usage = call["usage"]
    total_input = sum(
        usage.get(k, 0)
        for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    )
    suffix = "_above_100k_tokens" if total_input > 100000 else ""
    input_rate = pricing["input_cost_per_token" + suffix]
    output_rate = pricing["output_cost_per_token" + suffix]
    reference = total_input * input_rate + usage.get("output_tokens", 0) * output_rate
    creation = usage.get("cache_creation") or {}
    one_hour = creation.get("ephemeral_1h_input_tokens", 0)
    if one_hour > usage.get("cache_creation_input_tokens", 0):
        raise ValueError("Cache duration accounting inconsistent")
    five_min = usage.get("cache_creation_input_tokens", 0) - one_hour
    one_hour_key = "cache_creation_input_token_cost_above_1hr" + suffix
    estimated = (
        usage.get("input_tokens", 0) * input_rate
        + usage.get("output_tokens", 0) * output_rate
        + usage.get("cache_read_input_tokens", 0) * pricing["cache_read_input_token_cost" + suffix]
        + five_min * pricing["cache_creation_input_token_cost" + suffix]
        + one_hour * pricing[one_hour_key]
    )
    return reference, estimated


def metrics(result, pricing, report):
    provider = result.get("provider", {})
    calls = provider.get("calls", [])

    def not_forwarded(call):
        if call.get("forwarded_upstream") is False:
            return True
        # In the launch recorder, admission finishes before opening any HTTP client.
        # This specific error with no admission record therefore proves no inference occurred.
        return (
            call.get("status") is None
            and not call.get("admission")
            and "database is locked" in call.get("error", "").lower()
        )

    local_errors = [call for call in calls if not_forwarded(call)]
    upstream = [call for call in calls if not not_forwarded(call)]
    unknown_error_usage = sum(
        (c.get("status") is None or c.get("status", 0) >= 500) and not c.get("usage_valid")
        for c in upstream
    )
    valid = (
        any(c.get("usage_valid") for c in upstream)
        and not unknown_error_usage
        and all(c.get("usage_valid") for c in upstream if (c.get("status") or 0) < 400)
    )
    costs = [call_cost(c, pricing) for c in calls if c.get("usage_valid")]
    # Failed requests may be billed without usage; the totals remain observed-only.
    unknown = sum(c.get("status") == 200 and not c.get("usage_valid") for c in calls)
    usage = provider.get("usage", {})
    billed = [
        float(c["headers"]["x-litellm-response-cost"])
        for c in calls
        if c.get("headers", {}).get("x-litellm-response-cost") is not None
    ]
    return {
        "run_id": result["run_id"],
        "instance_id": result["instance_id"],
        "arm": result["arm"],
        "repetition": result["repetition"],
        "attempt": result["attempt"],
        "status": result["status"],
        "timeout": result.get("timeout", False),
        "exit_status": result.get("exit_status"),
        "resolved": report.get("resolved"),
        "grading_error": report.get("grading_error"),
        "usage_complete": valid,
        "unknown_successful_request_usage": unknown,
        "unknown_error_request_usage": unknown_error_usage,
        **{
            k: usage.get(k) if calls else None
            for k in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            )
        },
        "total_input_including_cache": sum(
            usage.get(k, 0)
            for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        )
        if calls
        else None,
        "llm_requests": len(upstream),
        "native_api_attempts": len(calls),
        "local_pre_inference_errors": len(local_errors),
        "local_pre_inference_error_wall_s": sum(
            c.get("end_unix_s", 0) - c.get("start_unix_s", 0) for c in local_errors
        ),
        "tool_calls": sum(c.get("tool_calls", 0) for c in calls),
        "reference_cost_usd": sum(c[0] for c in costs) if valid else None,
        "estimated_cached_cost_usd": sum(c[1] for c in costs) if valid else None,
        "router_reported_cost_usd": sum(billed)
        if len(billed) == len(upstream) and upstream
        else None,
        "router_cost_headers_present": len(billed),
        "http_429": sum(c.get("status") == 429 for c in calls),
        "provider_http_errors": sum(c.get("status", 0) >= 400 for c in calls),
        "provider_stream_errors": sum(bool(c.get("stream_errors")) for c in calls),
        "admission_wait_s": sum(c.get("admission", {}).get("wait_s", 0) for c in calls),
        "wall_s": result.get("wall_s"),
        "native_completion_wall_s": result.get("native_completion_wall_s"),
    }


def interval(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return [float(x) for x in np.quantile(finite, [0.025, 0.975])] if len(finite) else [None, None]


def bootstrap(groups, seed, count):
    rng = np.random.default_rng(seed)
    task_rows, differences, relatives, log_ratios = [], [], [], []
    for task_id, arms in groups.items():
        s, a = arms["standalone"], arms["agency"]
        if len(s) < 2 or len(a) < 2:
            raise ValueError("At least two complete observed runs per arm are required: " + task_id)
        bs = s[rng.integers(len(s), size=(count, len(s)))]
        ba = a[rng.integers(len(a), size=(count, len(a)))]
        ds = ba.mean(axis=1) - bs.mean(axis=1)
        differences.append(ds)
        relatives.append(ds / bs.mean(axis=1))
        vs, va = s.var(ddof=1), a.var(ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.log(ba.var(axis=1, ddof=1) / bs.var(axis=1, ddof=1))
        log_ratios.append(logs)
        row = {
            "instance_id": task_id,
            "n_standalone": len(s),
            "n_agency": len(a),
            "mean_standalone_usd": float(s.mean()),
            "mean_agency_usd": float(a.mean()),
            "difference_usd": float(a.mean() - s.mean()),
            "relative_difference_pct": float(100 * (a.mean() / s.mean() - 1)),
            "sd_standalone_usd": float(s.std(ddof=1)),
            "sd_agency_usd": float(a.std(ddof=1)),
            "cv_standalone": float(s.std(ddof=1) / s.mean()),
            "cv_agency": float(a.std(ddof=1) / a.mean()),
            "variance_ratio": float(va / vs) if vs else None,
            "difference_ci_low": interval(ds)[0],
            "difference_ci_high": interval(ds)[1],
            "variance_ratio_ci_low": interval(np.exp(logs))[0],
            "variance_ratio_ci_high": interval(np.exp(logs))[1],
            "degenerate_variance_bootstraps": int((~np.isfinite(logs)).sum()),
        }
        task_rows.append(row)
    difference_boot = np.mean(differences, axis=0)
    relative_boot = 100 * np.mean(relatives, axis=0)
    ratio_boot = np.exp(np.mean(log_ratios, axis=0))
    aggregate = {
        "difference_usd": float(np.mean([r["difference_usd"] for r in task_rows])),
        "difference_95ci": interval(difference_boot),
        "equal_task_relative_difference_pct": float(
            np.mean([r["relative_difference_pct"] for r in task_rows])
        ),
        "relative_difference_pct_95ci": interval(relative_boot),
        "geometric_variance_ratio": float(
            np.exp(np.mean([np.log(r["variance_ratio"]) for r in task_rows]))
        ),
        "variance_ratio_95ci": interval(ratio_boot),
        "scope": "Conditional on the five frozen tasks: independent resampling of runs within each task and arm; equal task weighting; no task resampling or population-level claim",
        "bootstrap_seed": seed,
        "bootstrap_resamples": count,
        "degenerate_aggregate_variance_bootstraps": int((~np.isfinite(ratio_boot)).sum()),
    }
    return task_rows, aggregate


def power_planning(task_rows):
    from scipy.stats import norm, t

    n = len(task_rows)
    within = np.mean(
        [
            (r["sd_agency_usd"] ** 2 + r["sd_standalone_usd"] ** 2) / r["mean_standalone_usd"] ** 2
            for r in task_rows
        ]
    )
    effects = np.array([r["relative_difference_pct"] / 100 for r in task_rows])
    noise = np.mean(
        [
            (
                r["sd_agency_usd"] ** 2 / r["n_agency"]
                + r["sd_standalone_usd"] ** 2 / r["n_standalone"]
            )
            / r["mean_standalone_usd"] ** 2
            for r in task_rows
        ]
    )
    between = max(0, effects.var(ddof=1) - noise)

    def power(k, reps, margin, theta=0):
        se = math.sqrt((between + within / reps) / k)
        if se == 0:
            return float(abs(theta) < margin)
        bound = margin - t.ppf(0.95, k - 1) * se
        if bound <= 0:
            return 0.0
        return float(norm.cdf((bound - theta) / se) - norm.cdf((-bound - theta) / se))

    rows = []
    for margin in (0.02, 0.05, 0.10):
        for reps in (10, 20, 50):
            lower, upper = 5, 100000
            if power(upper, reps, margin) < 0.8:
                required = None
            else:
                while lower < upper:
                    midpoint = (lower + upper) // 2
                    if power(midpoint, reps, margin) >= 0.8:
                        upper = midpoint
                    else:
                        lower = midpoint + 1
                required = lower
            rows.append(
                {
                    "margin_pct": margin * 100,
                    "repetitions_per_arm": reps,
                    "required_tasks": required,
                    "additional_tasks": max(0, required - n) if required else None,
                    "additional_repetitions_per_existing_task_arm": max(0, reps - 10),
                    "approx_power_zero_true_effect": power(required, reps, margin)
                    if required
                    else None,
                    "approx_power_observed_mean_effect": power(
                        required, reps, margin, effects.mean()
                    )
                    if required
                    else None,
                }
            )
    fixed_task = []
    for margin in (0.02, 0.05, 0.10):
        required_reps = max(
            10, math.ceil(within * (norm.ppf(0.95) + norm.ppf(0.90)) ** 2 / (n * margin**2))
        )
        fixed_task.append(
            {
                "margin_pct": margin * 100,
                "tasks": n,
                "repetitions_per_arm_task": required_reps,
                "additional_repetitions_per_arm_task": required_reps - 10,
            }
        )
    return {
        "method": "Approximate TOST at alpha=.05 (90% CI entirely within margin), target 80% power, normal sampling with task t critical value; planning assumes true mean effect zero",
        "within_run_relative_variance_sum": float(within),
        "between_task_relative_effect_variance": float(between),
        "assumptions": "Future tasks exchangeable with these five; fixed pilot mean denominators; independent runs; constant variance components. Only five task clusters: very uncertain and potentially optimistic. An observed effect at/outside a margin cannot be made equivalent by sample size alone.",
        "grid": rows,
        "fixed_task_only": fixed_task,
        "fixed_task_only_scope": "Normal approximation conditional on the same five tasks, zero true effect; excludes between-task variance because no population generalization is attempted",
        "tost_reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/",
    }


def figures(root, groups, tasks, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    labels = [t["instance_id"].replace("__", "/") for t in tasks]
    cost_labels = [
        f"{label} (n={t['n_standalone']}/{t['n_agency']} S/A)" for label, t in zip(labels, tasks)
    ]
    observed_n = sum(t["n_standalone"] + t["n_agency"] for t in tasks)
    cost_scope = f"complete usage, {observed_n}/100 runs"
    colors = {"standalone": "#31688e", "agency": "#b63679"}

    def save(fig, name):
        fig.tight_layout()
        for ext in ("pdf", "png", "svg"):
            fig.savefig(folder / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, len(tasks), figsize=(12, 3.4), sharey=False)
    for ax, task, label in zip(axes, tasks, labels):
        for x, arm in enumerate(("standalone", "agency")):
            data = groups[task["instance_id"]][arm]
            ax.boxplot(
                data, positions=[x], widths=0.5, showfliers=False, medianprops={"color": "black"}
            )
            ax.scatter(
                x + np.linspace(-0.12, 0.12, len(data)), data, s=17, color=colors[arm], alpha=0.75
            )
        ax.set_xticks(
            [0, 1],
            [f"Standalone\n(n={task['n_standalone']})", f"Agency\n(n={task['n_agency']})"],
            rotation=30,
        )
        ax.set_title(label, fontsize=8)
        ax.set_ylabel("Cache-neutral cost (USD)")
    fig.suptitle("PILOT · Repeated-run cost distributions · " + cost_scope, y=1.03)
    save(fig, "01_cost_distributions")

    for name, key, low, high, neutral, xlabel in (
        (
            "02_mean_differences",
            "difference_usd",
            "difference_ci_low",
            "difference_ci_high",
            0,
            "Agency − standalone cost (USD), 95% bootstrap CI",
        ),
        (
            "03_variance_ratios",
            "variance_ratio",
            "variance_ratio_ci_low",
            "variance_ratio_ci_high",
            1,
            "Variance ratio (Agency / standalone), 95% bootstrap CI",
        ),
    ):
        fig, ax = plt.subplots(figsize=(7.4, 3.4))
        means = np.array([t[key] for t in tasks])
        lower = np.array([t[low] for t in tasks])
        upper = np.array([t[high] for t in tasks])
        for y, mean, lo, hi in zip(range(len(tasks)), means, lower, upper):
            ax.plot([lo, hi], [y, y], color=colors["agency"])
            ax.scatter([mean], [y], color=colors["agency"], s=26)
        ax.axvline(neutral, ls="--", color="gray", lw=1)
        if neutral:
            ax.set_xscale("log")
        ax.set_yticks(range(len(tasks)), cost_labels)
        ax.set_xlabel(xlabel)
        ax.set_title(
            "PILOT · "
            + ("Mean cost differences" if neutral == 0 else "Run-to-run variance")
            + "\n"
            + cost_scope
        )
        save(fig, name)

    fig, ax = plt.subplots(figsize=(10, 4))
    fields = (
        "input_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "output_tokens",
    )
    palette = ("#31688e", "#35b779", "#e6b34b", "#b63679")
    for arm_i, arm in enumerate(("standalone", "agency")):
        base = np.zeros(len(tasks))
        for field, color in zip(fields, palette):
            values = [
                np.mean(
                    [
                        r[field]
                        for r in rows
                        if r["instance_id"] == t["instance_id"]
                        and r["arm"] == arm
                        and r["usage_complete"]
                    ]
                )
                / 1000
                for t in tasks
            ]
            ax.bar(
                np.arange(len(tasks)) + (arm_i - 0.5) * 0.36,
                values,
                0.34,
                bottom=base,
                color=color,
                label=field.replace("_", " ") if arm_i == 0 else None,
            )
            base += values
    ax.set_xticks(range(len(tasks)), labels, rotation=15, ha="right")
    ax.set_ylabel("Mean observed tokens (thousands)")
    ax.set_title("PILOT · Token breakdown (left: standalone; right: Agency)\n" + cost_scope)
    ax.legend(fontsize=8)
    save(fig, "04_token_breakdown")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, metric in zip(axes, ("tool_calls", "llm_requests")):
        for i, arm in enumerate(("standalone", "agency")):
            values = [
                np.mean(
                    [
                        r[metric]
                        for r in rows
                        if r["instance_id"] == t["instance_id"] and r["arm"] == arm
                    ]
                )
                for t in tasks
            ]
            ax.bar(
                np.arange(len(tasks)) + (i - 0.5) * 0.36, values, 0.34, label=arm, color=colors[arm]
            )
        ax.set_xticks(range(len(tasks)), labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean " + metric.replace("_", " "))
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
        ax.legend()
    fig.suptitle("PILOT · Supporting tool and LLM request counts · all 100 runs")
    save(fig, "05_call_counts")


def main(root):
    config = read(root / "configuration.json")
    pricing = read(root / "pricing.json")["rates"]
    results = [read(path) for path in sorted((root / "attempts").glob("*/*/summary.json"))]
    selected = {}
    for result in results:
        if (
            result["run_id"] not in selected
            or result["attempt"] > selected[result["run_id"]]["attempt"]
        ):
            selected[result["run_id"]] = result
    rows = []
    for result in selected.values():
        report_path = root / result["directory"] / "official-report.json"
        report = read(report_path) if report_path.exists() else {"grading_error": "Not graded"}
        rows.append(metrics(result, pricing, report))
    write_csv(root / "metrics.csv", rows)
    if len(rows) != 100:
        raise SystemExit(
            f"Recorded {len(rows)}/100 unique executions; refusing complete-study analysis"
        )
    tasks = read(root / "task_manifest.json")["tasks"]
    groups = {
        t["instance_id"]: {
            arm: np.array(
                [
                    r["reference_cost_usd"]
                    for r in rows
                    if r["instance_id"] == t["instance_id"]
                    and r["arm"] == arm
                    and r["reference_cost_usd"] is not None
                ]
            )
            for arm in ("standalone", "agency")
        }
        for t in tasks
    }
    summary, aggregate = bootstrap(groups, config["bootstrap_seed"], config["bootstrap_resamples"])
    write_csv(root / "task_summary.csv", summary)
    planning = power_planning(summary)
    (root / "sample_size_planning.json").write_text(json.dumps(planning, indent=2) + "\n")
    (root / "analysis_results.json").write_text(
        json.dumps({"aggregate": aggregate, "task_summary": summary}, indent=2) + "\n"
    )
    figures(root, groups, summary, rows)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent)
