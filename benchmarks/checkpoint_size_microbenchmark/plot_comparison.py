"""Plot completed checkpoint cohorts (requires matplotlib)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "bench-cow-criu": ("COW + CRIU", "#087f6b"),
    "bench-docker-image-commit": ("Docker commit — fresh", "#b44b44"),
    "bench-podman-image-commit": ("Podman commit — same ZFS store", "#b07913"),
}
SIZES = {
    0: "0 B",
    65536: "64 KiB",
    1048576: "1 MiB",
    16777216: "16 MiB",
    134217728: "128 MiB",
    1073741824: "1 GiB",
}


def plot(summary, destination):
    rows = json.loads(summary.read_text())
    roots = list(dict.fromkeys(row["root"] for row in rows))
    sizes = sorted({row["requested_bytes"] for row in rows})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    for axis, selected, title in zip(
        axes,
        (sizes, [size for size in sizes if size <= 134217728]),
        ("Full payload range", "0–128 MiB view"),
    ):
        for index, root in enumerate(roots):
            series = {row["requested_bytes"]: row for row in rows if row["root"] == root}
            label, color = LABELS.get(Path(root).name, ("Docker commit — retained", "#727985"))
            points = [series[size] for size in selected]
            values = [point["median_seconds"] for point in points]
            low = [point["median_seconds"] - point["min_seconds"] for point in points]
            high = [point["max_seconds"] - point["median_seconds"] for point in points]
            positions = [
                position + (index - (len(roots) - 1) / 2) * 0.035
                for position in range(len(selected))
            ]
            axis.errorbar(
                positions,
                values,
                yerr=[low, high],
                color=color,
                marker="o",
                markersize=4,
                linewidth=1.5,
                capsize=3,
                label=label,
            )
        axis.set_xticks(range(len(selected)), [SIZES.get(size, str(size)) for size in selected])
        axis.set_ylim(bottom=0)
        axis.set_title(title, loc="left", fontsize=11)
        axis.set_xlabel("Controlled payload size (categories)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Checkpoint latency (seconds)")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Agency controlled checkpoint benchmark", x=0.06, ha="left", fontsize=15, fontweight="bold"
    )
    fig.text(
        0.06,
        0.015,
        "Medians and observed min–max. Legacy: native commit. COW: checkpoint API + ptrace detach. Payload generation excluded.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(destination.with_suffix("." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.summary, args.output)
