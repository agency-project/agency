#!/usr/bin/env python3
"""Plot saved E1A.1 trajectories and runtimes; this never launches agents.

Here a trajectory is the ordered sequence of recorded ``tool_call`` events in
each saved JSONL stream. Each shell call gets one broad action label, even if
its command contains several operations. Event order is retained; timestamps,
arguments, and tool results are excluded from the similarity score.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import statistics
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


TASKS = [
    ("psf__requests-1921", "Requests"),
    ("pydata__xarray-4094", "Xarray"),
    ("scikit-learn__scikit-learn-12585", "Scikit-learn"),
    ("sphinx-doc__sphinx-7889", "Sphinx"),
    ("sympy__sympy-22456", "SymPy"),
]
ARCHIVE_NAME = "campaign-all-epochs.tar.gz"
PRIMARY_EPOCH = "epoch-20260915T004153-87e87b9c"


def action_label(tool: str, arguments: dict) -> str:
    """Map one recorded tool call to one coarse action, in fixed priority order."""
    if tool == "apply_patch":
        return "edit"
    if tool != "Bash":
        return "other"
    command = str(arguments.get("command", ""))
    if re.search(r"\b(?:pytest|py\.test|tox)\b|\bbin/test\b", command):
        return "test"
    if re.search(r"\bgit\s+(?:diff|status|show|log)\b", command):
        return "review"
    if re.search(r"\b(?:rg|grep|find)\b", command):
        return "search"
    if re.search(r"\b(?:sed|cat|head|tail|ls)\b", command):
        return "read"
    if re.search(r"\bpython(?:[0-9.]*)\b", command):
        return "probe"
    return "other"


def read_actions(path: Path) -> list[str]:
    actions = []
    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("type") != "tool_call":
                continue
            payload = event["payload"]
            actions.append(action_label(payload["tool"], payload.get("arguments") or {}))
    return actions


def lcs_overlap(left: list[str], right: list[str]) -> float:
    """Order-preserving LCS length divided by the longer action sequence."""
    if not left and not right:
        return 1.0
    previous = [0] * (len(right) + 1)
    for item in left:
        current = [0] * (len(right) + 1)
        for column, other in enumerate(right, 1):
            if item == other:
                current[column] = previous[column - 1] + 1
            else:
                current[column] = max(previous[column], current[column - 1])
        previous = current
    return previous[-1] / max(len(left), len(right))


def load_patch_metadata(archive_path: Path) -> dict[str, dict]:
    """Read only saved final patches from the campaign archive."""
    patches = {}
    epoch_marker = f"/epochs/{PRIMARY_EPOCH}/attempts/"
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if epoch_marker not in member.name or not member.name.endswith("/worker/final.patch"):
                continue
            attempt_id = member.name.split("/attempts/", 1)[1].split("/", 1)[0]
            content = archive.extractfile(member).read()
            filenames = set(re.findall(rb"^diff --git a/(.*?) b/", content, flags=re.MULTILINE))
            patches[attempt_id] = {"content": content, "files": filenames}
    return patches


def load_pairs(root: Path) -> tuple[list[dict], dict]:
    catalog = root / "artifacts/e1a1/trajectory-catalog/trajectory-index.csv"
    campaign = root / "artifacts/e1a1/imported/campaign-e1bf2bbb354147149fb7015de83eb217"
    report = json.loads((campaign / "integrity-report.json").read_text())
    index = list(csv.DictReader(catalog.open(newline="")))
    if len(index) != 30 or len(report["rows"]) != 30:
        raise ValueError("Expected 30 saved measured runs")

    report_by_attempt = {row["attempt_id"]: row for row in report["rows"]}
    patch_by_attempt = load_patch_metadata(campaign / ARCHIVE_NAME)
    by_key = {}
    for row in index:
        attempt_id = row["attempt_id"]
        event_path = (
            root
            / "artifacts/e1a1/trajectory-catalog/trajectory-events"
            / Path(row["trajectory_events"]).name
        )
        actions = read_actions(event_path)
        if len(actions) != int(row["tool_call_count"]):
            raise ValueError(f"Tool call count mismatch: {attempt_id}")
        saved = report_by_attempt[attempt_id]
        if saved["resolved"] is not True or saved["worker_status"] != "completed":
            raise ValueError(f"Incomplete or unresolved saved run: {attempt_id}")
        key = (row["instance_id"], int(row["repetition"]), row["condition"])
        by_key[key] = {
            "attempt_id": attempt_id,
            "actions": actions,
            # The terminal integrity report is the source of the published
            # 1.747x ratio; catalog wall times are slightly shorter.
            "wall_seconds": float(saved["worker_wall_seconds"]),
            "patch": patch_by_attempt[attempt_id],
        }

    if len(by_key) != 30 or len(patch_by_attempt) != 30:
        raise ValueError("Missing saved run mapping or final patch")

    pairs = []
    for task, label in TASKS:
        baselines = [by_key[(task, repetition, "baseline")] for repetition in (1, 2, 3)]
        for repetition in (1, 2, 3):
            base = by_key[(task, repetition, "baseline")]
            contended = by_key[(task, repetition, "cpu_contended")]
            # Anchor both axes to this exact baseline run. With three baseline
            # runs, the median of its two peer similarities equals their mean.
            normal_similarity = statistics.median(
                lcs_overlap(base["actions"], other["actions"])
                for other in baselines
                if other["attempt_id"] != base["attempt_id"]
            )
            contended_similarity = lcs_overlap(base["actions"], contended["actions"])
            pairs.append(
                {
                    "task": task,
                    "task_label": label,
                    "repetition": repetition,
                    "baseline_attempt": base["attempt_id"],
                    "contended_attempt": contended["attempt_id"],
                    "baseline_actions": len(base["actions"]),
                    "contended_actions": len(contended["actions"]),
                    "baseline_baseline_similarity": normal_similarity,
                    "baseline_contended_similarity": contended_similarity,
                    "similarity_delta": contended_similarity - normal_similarity,
                    "baseline_wall_seconds": base["wall_seconds"],
                    "contended_wall_seconds": contended["wall_seconds"],
                    "wall_ratio": contended["wall_seconds"] / base["wall_seconds"],
                    "same_patch_files": base["patch"]["files"] == contended["patch"]["files"],
                    "same_patch_bytes": base["patch"]["content"] == contended["patch"]["content"],
                }
            )

    ratio = statistics.median(pair["wall_ratio"] for pair in pairs)
    if abs(ratio - float(report["median_paired_worker_wall_ratio"])) > 1e-9:
        raise ValueError("Paired wall ratio does not reproduce the terminal report")
    return pairs, report


def write_metrics(pairs: list[dict], output: Path) -> None:
    columns = list(pairs[0])
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(pairs)


def task_cluster_bootstrap_interval(pairs: list[dict]) -> tuple[float, float]:
    """Exploratory percentile interval for mean Δ, resampling five task means."""
    task_means = [
        statistics.mean(pair["similarity_delta"] for pair in pairs if pair["task"] == task)
        for task, _ in TASKS
    ]
    bootstrap = sorted(
        statistics.mean(sample) for sample in itertools.product(task_means, repeat=len(task_means))
    )
    last = len(bootstrap) - 1
    return bootstrap[int(0.025 * last)], bootstrap[int(0.975 * last)]


def make_figure(pairs: list[dict], output_dir: Path) -> None:
    dark = "#183044"
    blue = "#276F9A"
    orange = "#E46B49"
    muted = "#667786"
    task_colors = {
        "psf__requests-1921": "#286F96",
        "pydata__xarray-4094": "#238777",
        "scikit-learn__scikit-learn-12585": "#9365A0",
        "sphinx-doc__sphinx-7889": "#BE8140",
        "sympy__sympy-22456": "#D45D62",
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "axes.edgecolor": "#B9C4CC",
            "axes.labelcolor": dark,
            "xtick.color": muted,
            "ytick.color": dark,
        }
    )
    fig = plt.figure(figsize=(17.2, 9.7), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, left=0.07, right=0.97, bottom=0.28, top=0.75, width_ratios=[1.0, 1.25], wspace=0.34
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    y_positions = [14 - i for i in range(15)]

    limits = (0.25, 0.85)
    ax_a.plot(limits, limits, linestyle=(0, (5, 4)), color="#748592", linewidth=2.0, zorder=1)
    for pair in pairs:
        ax_a.scatter(
            pair["baseline_baseline_similarity"],
            pair["baseline_contended_similarity"],
            s=125,
            color=task_colors[pair["task"]],
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
    ax_a.set_xlim(limits)
    ax_a.set_ylim(limits)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax_a.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax_a.grid(color="#E7ECEF", linewidth=0.8, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.set_xlabel("Baseline ↔ baseline similarity", labelpad=12, fontsize=12.2)
    ax_a.set_ylabel("Baseline ↔ contended similarity", labelpad=13, fontsize=12.2)

    for i, (pair, y) in enumerate(zip(pairs, y_positions)):
        if i % 3 == 0:
            ax_b.axhspan(y - 2.48, y + 0.48, color="#F5F8FA", zorder=0)
        ax_b.plot(
            [pair["baseline_wall_seconds"], pair["contended_wall_seconds"]],
            [y, y],
            color="#AAB7C1",
            linewidth=2.7,
            zorder=1,
        )
        ax_b.scatter(
            pair["baseline_wall_seconds"],
            y,
            color=blue,
            s=70,
            zorder=3,
            edgecolor="white",
            linewidth=0.9,
        )
        ax_b.scatter(
            pair["contended_wall_seconds"],
            y,
            color=orange,
            s=70,
            zorder=3,
            edgecolor="white",
            linewidth=0.9,
        )

    labels = [f"{pair['task_label']}  ·  {pair['repetition']}" for pair in pairs]
    ax_b.set_yticks(y_positions, labels)
    ax_b.tick_params(axis="y", length=0, pad=11, labelsize=10.8)
    ax_b.set_ylim(-0.6, 14.7)
    ax_b.spines[["top", "right", "left"]].set_visible(False)
    ax_b.grid(axis="x", color="#E7ECEF", linewidth=0.8, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.set_xlim(0, 610)
    ax_b.set_xticks([0, 150, 300, 450, 600])
    ax_b.set_xlabel("Worker wall time  ·  seconds", labelpad=12, fontsize=12.2)

    fig.text(
        0.055,
        0.935,
        "CPU contention slows agents without increasing trajectory variation",
        color=dark,
        fontsize=22.5,
        weight="bold",
        ha="left",
    )
    fig.text(
        0.055,
        0.885,
        "E1A.1  ·  5 SWE-bench tasks  ·  15 matched baseline/contended pairs  ·  all 30 runs resolved",
        color=muted,
        fontsize=14,
        ha="left",
    )
    fig.text(
        0.095,
        0.800,
        "A   Compared with normal run variation",
        color=dark,
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.095, 0.772, "15 comparisons; diagonal means equal similarity", color=muted, fontsize=11.5
    )
    fig.text(
        0.555, 0.800, "B   Contention increases runtime", color=dark, fontsize=16, weight="bold"
    )
    fig.text(
        0.555,
        0.772,
        "Every contended run is slower than its matched baseline",
        color=muted,
        fontsize=11.5,
    )

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=task_colors[task],
                markeredgecolor="white",
                markersize=10,
                label=label,
            )
            for task, label in TASKS
        ],
        loc="lower left",
        bbox_to_anchor=(0.095, 0.175),
        frameon=False,
        ncol=5,
        fontsize=10.0,
        handlelength=0.8,
        columnspacing=0.8,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=blue,
                markeredgecolor="white",
                markersize=9,
                label="Baseline",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=orange,
                markeredgecolor="white",
                markersize=9,
                label="CPU contended",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.555, 0.175),
        frameon=False,
        ncol=2,
        fontsize=10.8,
        handlelength=1.0,
        columnspacing=1.3,
    )

    deltas = [pair["similarity_delta"] for pair in pairs]
    same_files = sum(pair["same_patch_files"] for pair in pairs)
    exact_patches = sum(pair["same_patch_bytes"] for pair in pairs)
    slower = sum(pair["wall_ratio"] > 1 for pair in pairs)
    median_ratio = statistics.median(pair["wall_ratio"] for pair in pairs)
    fig.text(
        0.095,
        0.145,
        f"Median similarity difference: {statistics.median(deltas):+.3f}",
        color=dark,
        fontsize=12.2,
        weight="bold",
    )
    fig.text(
        0.555,
        0.145,
        f"{slower}/15 pairs slower  ·  median paired slowdown {median_ratio:.3f}×",
        color=dark,
        fontsize=12.2,
        weight="bold",
    )
    fig.text(
        0.055,
        0.095,
        f"Trajectory variation under contention is comparable to normal run-to-run variation, "
        f"while median runtime increases {median_ratio:.3f}×.",
        color=dark,
        fontsize=13.0,
        weight="bold",
    )
    fig.text(
        0.055,
        0.061,
        f"Same files in final patch: {same_files}/15 pairs; byte-identical patches: {exact_patches}/15. "
        "Shared files and resolution do not imply identical patches or action paths.",
        color=muted,
        fontsize=10.6,
    )
    fig.text(
        0.055,
        0.030,
        "Trajectory similarity = longest common subsequence of ordered coarse tool-call categories ÷ longer sequence length. "
        "Each x is the median of its baseline run vs. the other two same-task baselines.",
        color=muted,
        fontsize=10.2,
    )

    for ext in ("png", "svg"):
        fig.savefig(
            output_dir / f"trajectory_invariance.{ext}",
            dpi=240,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.20,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    output_dir = Path(__file__).resolve().parent
    pairs, _report = load_pairs(args.root.resolve())
    write_metrics(pairs, output_dir / "pair_metrics.csv")
    make_figure(pairs, output_dir)
    print(f"Wrote {output_dir / 'trajectory_invariance.png'} and .svg")
    print(
        f"Median similarity difference: {statistics.median(p['similarity_delta'] for p in pairs):+.6f}"
    )
    interval = task_cluster_bootstrap_interval(pairs)
    print(
        f"Exploratory task-cluster bootstrap interval for mean difference: [{interval[0]:+.6f}, {interval[1]:+.6f}]"
    )
    print(f"Median paired wall ratio: {statistics.median(p['wall_ratio'] for p in pairs):.6f}")


if __name__ == "__main__":
    main()
