"""Secondary estimated cache-aware cost analysis; the primary endpoint is unchanged."""

from pathlib import Path
import csv
import json
import sys

import numpy as np

from analysis import bootstrap, write_csv


def main(root):
    config = json.loads((root / "configuration.json").read_text())
    tasks = json.loads((root / "task_manifest.json").read_text())["tasks"]
    rows = list(csv.DictReader((root / "metrics.csv").open()))
    groups = {
        task["instance_id"]: {
            arm: np.array(
                [
                    float(r["estimated_cached_cost_usd"])
                    for r in rows
                    if r["instance_id"] == task["instance_id"]
                    and r["arm"] == arm
                    and r["estimated_cached_cost_usd"]
                ]
            )
            for arm in ("standalone", "agency")
        }
        for task in tasks
    }
    summary, aggregate = bootstrap(groups, config["bootstrap_seed"], config["bootstrap_resamples"])
    write_csv(root / "cache_aware_summary.csv", summary)
    output = {
        "primary_endpoint": False,
        "cost_type": "Estimate from observed cache-specific usage and frozen router rates; not an invoice",
        "aggregate": aggregate,
        "task_summary": summary,
    }
    (root / "cache_aware_results.json").write_text(json.dumps(output, indent=2) + "\n")
    ci = aggregate["difference_95ci"]
    ratio_ci = aggregate["variance_ratio_95ci"]
    section = [
        "## Secondary estimated cache-aware cost",
        "",
        "This estimate preserves observed cache-read and cache-write discounts/charges. It is a secondary endpoint; the primary cache-neutral reference-cost analysis above remains unchanged. Provider invoices are unavailable.",
        "",
        f"Equal-task Agency − standalone estimated cost: ${aggregate['difference_usd']:+.6f} (95% CI ${ci[0]:+.6f} to ${ci[1]:+.6f}); relative difference {aggregate['equal_task_relative_difference_pct']:+.2f}%.",
        f"Geometric variance ratio: {aggregate['geometric_variance_ratio']:.3f} (95% CI {ratio_ci[0]:.3f} to {ratio_ci[1]:.3f}).",
        "",
        "| Task | Mean standalone $ | Mean Agency $ | Difference % | Variance ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in summary:
        section.append(
            f"| {task['instance_id']} | {task['mean_standalone_usd']:.6f} | {task['mean_agency_usd']:.6f} | {task['relative_difference_pct']:+.2f} | {task['variance_ratio']:.3f} |"
        )
    section += [
        "",
        "The same fixed-task bootstrap and recorded seed are used. Unknown usage is not imputed. Differences between primary and secondary estimates can reflect cache-price effects; shared caches and different session paths remain possible billing/timing confounders. See `cache_aware_results.json` and `cache_aware_summary.csv`.",
        "",
        "To reproduce this appendix, run `python analysis_secondary.py` after `analysis.py` and `report.py`.",
        "",
    ]
    report = root / "results.md"
    original = report.read_text().split("\n## Secondary estimated cache-aware cost", 1)[0].rstrip()
    report.write_text(original + "\n\n" + "\n".join(section))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent)
