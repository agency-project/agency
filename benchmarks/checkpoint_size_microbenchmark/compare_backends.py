"""Compare completed cohorts without changing the retained Docker analysis."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def measurements(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError(f"Cohort is not complete: {root}")
    rows = []
    for path in sorted((root / "runs").glob("*/run.json")):
        data = json.loads(path.read_text())
        if data["status"] != "completed":
            raise ValueError(f"Incomplete benchmark condition: {path}")
        core = data.get("checkpoint_seconds", data.get("report_commit_seconds"))
        if core is None:
            raise ValueError(f"Missing checkpoint measurement: {path}")
        records = json.loads((path.parent / "records.json").read_text())

        def span(name):
            values = [row[3] / 1e9 for row in records if len(row) > 3 and row[1] == name]
            return sum(values) if values else None

        api_seconds = span("checkpoint.total")
        if api_seconds is None:
            api_seconds = span("sandbox:commit")
        report = {}
        if data.get("checkpoint_report_path"):
            report = json.loads((root / data["checkpoint_report_path"]).read_text())
        rows.append(
            {
                "root": str(root),
                "backend": manifest["config"].get("checkpoint_backend", "image_commit"),
                "runtime": manifest["config"]["backend"],
                "requested_bytes": data["requested_bytes"],
                "replicate": data["replicate"],
                "checkpoint_seconds": core,
                "checkpoint_including_handoff_seconds": data.get(
                    "checkpoint_including_handoff_seconds", core
                ),
                "boundary_ptrace_detach_seconds": data.get("boundary_ptrace_detach_seconds", 0),
                "restore_seconds": data.get("restore_seconds"),
                "checkpoint_api_seconds": api_seconds,
                "engine_teardown_seconds": span("teardown:commit"),
                "harness_retire_seconds": span("harness:retire"),
                "process_dump_seconds": report.get("process_dump_seconds"),
                "fs_snapshot_seconds": report.get("fs_snapshot_seconds"),
                "process_memory_rss_bytes": report.get("process_memory_rss_bytes"),
            }
        )
    if not rows:
        raise ValueError(f"No completed measurements in {root}")
    expected = manifest.get("actual_execution_order", [])
    if len(rows) != len(expected):
        raise ValueError(f"Missing or extra measurements in {root}")
    return manifest, rows


def compare(roots, output):
    cohorts = [measurements(root) for root in roots]
    fresh_sources = {
        (
            manifest["agency_source"]["executing_source_sha256"],
            manifest["agency_source"]["executing_runner_sha256"],
        )
        for manifest, _ in cohorts
        if manifest["agency_source"].get("executing_source_sha256")
    }
    if len(fresh_sources) > 1:
        raise ValueError("Fresh cohorts used different Agency source or measurement runners")
    cow_parents = {
        manifest["config"].get("checkpoint_zfs_parent")
        for manifest, _ in cohorts
        if manifest["config"].get("checkpoint_backend") == "cow_criu"
    }
    for manifest, _ in cohorts:
        config = manifest["config"]
        if (
            cow_parents
            and config["backend"] == "podman"
            and config.get("checkpoint_zfs_parent") not in cow_parents
        ):
            raise ValueError("Matched Podman cohorts used different ZFS parents")
    fixed = (
        "provider",
        "model",
        "temperature",
        "max_completion_tokens",
        "context_limit",
        "max_steps",
        "sample_hz",
        "sample_gpu",
        "sandbox_cpus",
        "sandbox_memory",
        "harness",
        "reasoning_effort",
        "file_access",
        "checkpoint_diagnostics",
        "checkpoint_diagnostics_extended",
        "max_concurrent_engines",
    )
    control = cohorts[0][0]
    for manifest, _ in cohorts[1:]:
        if manifest["payload"] != control["payload"]:
            raise ValueError("Controlled prompts or payload generator differ")
        if manifest["actual_sizes_bytes"] != control["actual_sizes_bytes"]:
            raise ValueError("Controlled payload sizes differ")
        for key in fixed:
            if manifest["config"].get(key) != control["config"].get(key):
                raise ValueError(f"Workload configuration differs: {key}")
        for key in ("instance_id", "repo_commit_in_creation_image", "creation_image_id"):
            if manifest["control"].get(key) != control["control"].get(key):
                raise ValueError(f"Pinned benchmark control differs: {key}")
    rows = [row for _, cohort in cohorts for row in cohort]
    summaries = []
    for root, (_, cohort) in zip(roots, cohorts):
        for size in sorted({row["requested_bytes"] for row in cohort}):
            group = [row for row in cohort if row["requested_bytes"] == size]
            values = [row["checkpoint_including_handoff_seconds"] for row in group]

            def optional_median(key):
                present = [row[key] for row in group if row[key] is not None]
                return statistics.median(present) if present else None

            summaries.append(
                {
                    "root": str(root),
                    "backend": group[0]["backend"],
                    "runtime": group[0]["runtime"],
                    "requested_bytes": size,
                    "n": len(group),
                    "median_seconds": statistics.median(values),
                    "min_seconds": min(values),
                    "max_seconds": max(values),
                    "median_checkpoint_api_seconds": statistics.median(
                        row["checkpoint_api_seconds"] for row in group
                    ),
                    "median_engine_teardown_seconds": statistics.median(
                        row["engine_teardown_seconds"] for row in group
                    ),
                    "median_core_seconds": statistics.median(
                        row["checkpoint_seconds"] for row in group
                    ),
                    "median_detach_seconds": statistics.median(
                        row["boundary_ptrace_detach_seconds"] for row in group
                    ),
                    "median_process_dump_seconds": optional_median("process_dump_seconds"),
                    "median_fs_snapshot_seconds": optional_median("fs_snapshot_seconds"),
                    "median_process_memory_rss_bytes": optional_median("process_memory_rss_bytes"),
                    "median_restore_seconds": statistics.median(
                        row["restore_seconds"]
                        for row in group
                        if row["restore_seconds"] is not None
                    )
                    if any(row["restore_seconds"] is not None for row in group)
                    else None,
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("measurements", rows), ("summary", summaries)):
        (output / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    labels = {
        "bench-cow-criu": "COW + CRIU",
        "bench-docker-image-commit": "Docker commit (fresh)",
        "bench-podman-image-commit": "Podman commit (same ZFS store)",
    }
    names = [labels.get(root.name, "Docker commit (retained)") for root in roots]
    sizes = sorted({row["requested_bytes"] for row in summaries})
    size_names = {
        0: "0 B",
        65536: "64 KiB",
        1048576: "1 MiB",
        16777216: "16 MiB",
        134217728: "128 MiB",
        1073741824: "1 GiB",
    }
    by_key = {(row["root"], row["requested_bytes"]): row for row in summaries}
    lines = [
        "# Controlled checkpoint comparison",
        "",
        "The legacy column uses the native commit duration from the original experiment. "
        "COW includes the complete checkpoint API plus ptrace detachment. Payload generation, "
        "initial provisioning, model calls and restore verification are excluded from that metric. "
        "The enclosing checkpoint API and engine teardown durations are reported separately below.",
        "",
        "Values are medians in seconds; brackets show observed min–max and sample count. "
        "These ranges are not confidence intervals. Each cohort has three initial repetitions per size "
        "and two more where the original 10% spread rule triggered.",
        "",
        "| Payload | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for size in sizes:
        values = []
        for root in roots:
            row = by_key[str(root), size]
            values.append(
                f"{row['median_seconds']:.3f} [{row['min_seconds']:.3f}–{row['max_seconds']:.3f}; n={row['n']}]"
            )
        lines.append("| " + size_names.get(size, str(size)) + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "![Checkpoint comparison](checkpoints.png)",
        "",
        "## Enclosing profiler measurements",
        "",
        "Native commit excludes legacy diagnostic collection/finalization. The following checkpoint "
        "API durations include that work. Engine teardown also includes its normal stop path. "
        "Neither is the entire Agent.run wall time.",
        "",
        "| Cohort | Payload | Checkpoint API median (s) | Engine teardown median (s) | Detach median (s) | COW restore median (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for root, name in zip(roots, names):
        for size in sizes:
            row = by_key[str(root), size]
            restore = row["median_restore_seconds"]
            lines.append(
                f"| {name} | {size_names.get(size, str(size))} | {row['median_checkpoint_api_seconds']:.3f} | {row['median_engine_teardown_seconds']:.3f} | {row['median_detach_seconds']:.3f} | "
                + (f"{restore:.3f}" if restore is not None else "—")
                + " |"
            )
    lines += [
        "",
        "## COW components",
        "",
        "Process dump includes the Podman/runc/CRIU operation. RSS is the sum of process resident "
        "sets, not a deduplicated physical-memory measurement. Native freeze/page statistics are "
        "retained in each cow-criu-checkpoint.json report.",
        "",
        "| Payload | Process dump median (s) | ZFS snapshot median (s) | Process RSS median (MiB) |",
        "|---|---:|---:|---:|",
    ]
    for row in summaries:
        if row["backend"] != "cow_criu":
            continue
        lines.append(
            f"| {size_names.get(row['requested_bytes'], str(row['requested_bytes']))} | "
            f"{row['median_process_dump_seconds']:.3f} | {row['median_fs_snapshot_seconds']:.3f} | "
            f"{row['median_process_memory_rss_bytes'] / 1024**2:.1f} |"
        )
    lines += [
        "",
        "## Controls and limits",
        "",
        "The comparison verifies matching prompts, deterministic payload generator, size matrix, "
        "pinned image identity, Requests repository commit, model, reasoning effort, token limits, "
        "CPU/memory settings, syscall/file-access profiling and repetition policy. The COW image's "
        "Podman configuration digest was verified against the original Docker OCI index's amd64 image.",
        "",
        "This is a system-level backend comparison. Docker uses its normal containerd storage; "
        "Podman commit and COW use the same per-sandbox ZFS clone layout; both Podman variants use native OverlayFS in those clones "
        "on one sparse file-backed pool. Runtime and storage differ between Docker and Podman; live-process preservation differs between commit and COW. "
        "The matched Podman commit baseline helps distinguish runtime/storage effects, but these results do not isolate "
        "ZFS as the sole cause of any speedup. Initial seed import and clone setup are outside checkpoint timing. "
        "Payload generation and its writes occur before the checkpoint boundary, so a cheap snapshot does not "
        "mean those writes are free. Model responses can vary despite fixed prompts and temperature; live process "
        "RSS is reported for each COW condition.",
        "",
        "See measurements.csv/json for every repetition, process dump, ZFS snapshot and RSS measurement; "
        "see each cohort's run.json, checkpoint report, records.json and Perfetto trace for raw evidence. "
        "Validation runs are excluded. CRIU's queued-inotify-event limitation and exact setup are documented "
        "in docs/fast-checkpoint.md.",
        "",
        "## Source identity",
        "",
        "| Cohort | Executed Agency digest / retained source manifest | Runner digest |",
        "|---|---|---|",
    ]
    for name, (manifest, _) in zip(names, cohorts):
        source = manifest["agency_source"]
        lines.append(
            f"| {name} | `{source.get('executing_source_sha256', source.get('manifest_sha256'))}` | `{source.get('executing_runner_sha256', 'retained original runner')}` |"
        )
    lines += ["", "## Invalid workload attempts", ""]
    for name, (manifest, _) in zip(names, cohorts):
        invalid = manifest.get("invalid_workload_attempts", [])
        reasons = "; ".join(item["reason"] for item in invalid)
        lines.append(
            f"- {name}: {len(invalid)} excluded invalid-workload attempt(s)."
            + (" " + reasons + "." if reasons else "")
        )
    lines += [
        "",
        "An exact model refusal or a START_NS-only CLI tool result without the required completion record is preserved "
        "under invalid-workload-attempts and retried with the same prompt/model/source/runner. "
        "Checkpoint, integrity and cleanup failures are not eligible for this retry. The separate "
        "resume controller retains the original measurement runner and shuffled repetition/spread policy; "
        "its hash is recorded in each resumed cohort manifest.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.roots, args.output), indent=2))
