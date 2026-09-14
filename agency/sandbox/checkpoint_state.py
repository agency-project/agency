"""Extended checkpoint observations, with explicit unknowns and bounded claims."""

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

FINGERPRINT = ("device", "inode", "size_bytes", "mtime_ns", "ctime_ns")


def category(path):
    parts = set(Path(path).parts)
    for label, names in [
        ("git", {".git"}),
        ("npm", {".npm"}),
        ("pip", {"pip"}),
        ("cargo", {".cargo", "cargo"}),
        ("cache", {".cache", "__pycache__", ".pytest_cache"}),
        ("build", {"build", "dist", "target", "node_modules"}),
    ]:
        if parts & names:
            return label
    return "repo" if path.startswith(("/testbed/", "/workspace/repo/")) else "other"


def totals(rows):
    result = {}
    for row in rows:
        group = result.setdefault(
            category(row["path"]), {"files": 0, "logical_bytes": 0, "unknown_size_files": 0}
        )
        group["files"] += 1
        if row.get("size_bytes") is None:
            group["unknown_size_files"] += 1
        else:
            group["logical_bytes"] += row["size_bytes"]
    return result


def fingerprint(row):
    values = tuple(row.get(k) for k in FINGERPRINT)
    return values if all(value is not None for value in values) else None


def incremental(previous, current):
    """Whole-file logical bytes, never a byte-level content diff."""
    rows = []
    for path in sorted(set(previous) | set(current)):
        before, after = previous.get(path), current.get(path)
        if after is None or after.get("kind") == "unavailable":
            status = (
                "deleted"
                if after and after.get("errno") == 2 and before and before.get("kind") == "file"
                else "unknown"
            )
        elif after.get("kind") != "file":
            status = "unknown"
        elif before is None or before.get("errno") == 2:
            status = "added" if after.get("change") == "A" else "newly_observed_modified"
        elif fingerprint(before) is None or fingerprint(after) is None:
            status = "unknown"
        elif fingerprint(before) != fingerprint(after):
            status = "modified"
        elif after.get("dirty"):
            status = "metadata_unchanged_dirty"
        else:
            status = "metadata_unchanged_clean"
        rows.append(
            {
                "path": path,
                "status": status,
                "before_bytes": before.get("size_bytes") if before else None,
                "after_bytes": after.get("size_bytes") if after else None,
            }
        )
    buckets = {}
    for row in rows:
        group = buckets.setdefault(row["status"], {"files": 0, "before_bytes": 0, "after_bytes": 0})
        group["files"] += 1
        group["before_bytes"] += row["before_bytes"] or 0
        group["after_bytes"] += row["after_bytes"] or 0
    return {
        "basis": "metadata comparison; equal metadata does not prove equal content",
        "files": rows,
        "totals": buckets,
    }


def directory(backend):
    cfg = backend._agconfig
    log_dir = cfg.agent.log_dir
    if not log_dir:
        log_dir = str(Path(cfg.data_logger.db_path).parent) if cfg.data_logger.db_path else "logs"
    return Path(log_dir).resolve() / "checkpoint-diagnostics"


def state_path(backend):
    name = hashlib.sha256(backend._container_name().encode()).hexdigest()
    return directory(backend) / f"{name}.state.json"


def load_previous(backend):
    path = state_path(backend)
    return json.loads(path.read_text()) if path.exists() else None


def save_previous(backend, report):
    path = state_path(backend)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(report, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def database_events(path):
    if not path:
        return [], "database_unavailable"
    try:
        connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT type,payload,id,timestamp FROM events WHERE type IN ('checkpoint_file_access','tool_call','tool_result') ORDER BY timestamp,id"
            ).fetchall()
        finally:
            connection.close()
        return [
            (
                kind,
                {
                    **json.loads(payload),
                    "_event_id": event_id,
                    "_recorded_wall_ns": round(timestamp * 1e9),
                },
            )
            for kind, payload, event_id, timestamp in rows
        ], "partial_observed_events"
    except (sqlite3.Error, OSError, ValueError):
        return [], "database_unavailable"


def reuse(snapshot, events, ended_wall_ns):
    start = snapshot["committed_wall_ns"]
    files = snapshot["snapshot"]
    evidence = {}
    for kind, event in events:
        if (
            kind != "checkpoint_file_access"
            or not start < event.get("started_wall_ns", 0) <= ended_wall_ns
        ):
            continue
        path = event.get("path")
        if path not in files or not files[path].get("dirty"):
            continue
        row = evidence.setdefault(
            path,
            {
                "opens": 0,
                "read_calls": 0,
                "returned_read_bytes": 0,
                "first_access_wall_ns": None,
                "version_mismatches": 0,
            },
        )
        if fingerprint(event) is None or fingerprint(event) != fingerprint(files[path]):
            row["version_mismatches"] += 1
            continue
        row["opens"] += int(event.get("successful_open", False))
        count = event.get("returned_bytes", 0)
        row["read_calls"] += int(count > 0)
        row["returned_read_bytes"] += max(0, count)
        if (
            row["first_access_wall_ns"] is None
            or event["started_wall_ns"] < row["first_access_wall_ns"]
        ):
            row["first_access_wall_ns"] = event["started_wall_ns"]
    groups = {}
    for path, value in files.items():
        if not value.get("dirty") or value.get("kind") != "file":
            continue
        row = evidence.setdefault(
            path,
            {
                "opens": 0,
                "read_calls": 0,
                "returned_read_bytes": 0,
                "first_access_wall_ns": None,
                "version_mismatches": 0,
            },
        )
        row["status"] = (
            "observed_read" if row["read_calls"] else "no_observed_read_partial_coverage"
        )
        row["time_to_first_access_s"] = (
            (row["first_access_wall_ns"] - start) / 1e9 if row["first_access_wall_ns"] else None
        )
        group = groups.setdefault(
            category(path),
            {
                "checkpointed_logical_bytes": 0,
                "logical_bytes_in_files_observed_read": 0,
                "returned_read_bytes": 0,
            },
        )
        size = value.get("size_bytes") or 0
        group["checkpointed_logical_bytes"] += size
        group["logical_bytes_in_files_observed_read"] += size if row["read_calls"] else 0
        group["returned_read_bytes"] += row["returned_read_bytes"]
    for group in groups.values():
        denominator = group["checkpointed_logical_bytes"]
        group["file_weighted_observed_reuse_fraction"] = (
            group["logical_bytes_in_files_observed_read"] / denominator if denominator else None
        )
    return {
        "checkpoint_id": snapshot["checkpoint_id"],
        "observed_until_wall_ns": ended_wall_ns,
        "coverage": "partial; no negative never-read inference; open is not read; mmap and untraced processes absent",
        "unique_bytes_read": None,
        "files": evidence,
        "categories": groups,
    }


def refresh_reuse(log_directory, db_path, *, ended_wall_ns=None):
    """Refresh all successful checkpoint windows, including reads several calls later.

    Call after the task's logger has flushed to include its final access events.
    Reports are immutable; refreshed observations live in private sidecar files.
    """
    ended_wall_ns = ended_wall_ns if ended_wall_ns is not None else time.time_ns()
    events, coverage = database_events(db_path)
    count = 0
    for path in Path(log_directory).glob("*.json"):
        if path.name.endswith((".state.json", ".reuse.json")):
            continue
        report = json.loads(path.read_text())
        if not report.get("commit_success") or not report.get("snapshot_complete"):
            continue
        value = reuse(report, events, ended_wall_ns)
        value["event_coverage"] = coverage
        fd, temporary = tempfile.mkstemp(prefix=".reuse-", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(value, stream, indent=2)
            os.replace(temporary, path.with_suffix(".reuse.json"))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        count += 1
    return count


def finalize(backend, report):
    """Read-only joins and runtime-reported sizes, not physical-write inference."""
    previous = report.pop("_previous", None)
    report["category_totals"] = totals(
        [r for r in report["changes"] or [] if r["change"] in ("A", "C")]
    )
    if (
        previous
        and report.get("snapshot_complete")
        and previous.get("container_id") == report.get("container_id")
    ):
        report["previous_checkpoint_id"] = previous["checkpoint_id"]
        report["incremental"] = incremental(previous["snapshot"], report["snapshot"])
    else:
        report["previous_checkpoint_id"] = None
        report["incremental"] = {
            "available": False,
            "reason": "no successful baseline for this container identity",
        }
    events, coverage = database_events(backend._agconfig.data_logger.db_path)
    report["event_coverage"] = coverage
    report["reuse_of_previous_checkpoint"] = (
        reuse(previous, events, time.time_ns())
        if previous and report["previous_checkpoint_id"]
        else None
    )
    report["action_window"] = {
        "start_wall_ns": previous.get("committed_wall_ns") if previous else None,
        "end_wall_ns": report["observed_wall_ns"],
        "attribution": "temporal association only; not causal per-tool byte deltas",
        "event_database": backend._agconfig.data_logger.db_path,
    }
    start = report["action_window"]["start_wall_ns"]
    report["action_window"]["events"] = [
        {
            "type": kind,
            "event_id": event["_event_id"],
            "recorded_wall_ns": event["_recorded_wall_ns"],
        }
        for kind, event in events
        if kind in ("tool_call", "tool_result")
        and start is not None
        and start <= event["_recorded_wall_ns"] <= report["observed_wall_ns"]
    ]
    refresh_reuse(directory(backend), backend._agconfig.data_logger.db_path)
    report["physical_bytes_written"] = None
    report["image_size_bytes"] = None
    report["image_size_delta_from_creation_image_bytes"] = None
    if report["commit_success"]:
        sizes = []
        for image in (report["image_id"], report.get("base_image_id")):
            if not image:
                sizes.append(None)
                continue
            result = backend._run(
                [backend._runtime, "image", "inspect", "--format={{.Size}}", image],
                check=True,
                timeout=backend._agconfig.sandbox.inspect_timeout_s,
            )
            sizes.append(int(result.stdout))
        report["image_size_bytes"] = sizes[0]
        if None not in sizes:
            report["image_size_delta_from_creation_image_bytes"] = sizes[0] - sizes[1]
    report["image_size_semantics"] = (
        "runtime-reported image size; not unique storage, compressed blob size, or physical writes"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Refresh observed checkpoint reuse after logger flush (no model calls)."
    )
    parser.add_argument("--diagnostics-dir", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--ended-wall-ns", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "reports_refreshed": refresh_reuse(
                    args.diagnostics_dir, args.db, ended_wall_ns=args.ended_wall_ns
                )
            }
        )
    )
