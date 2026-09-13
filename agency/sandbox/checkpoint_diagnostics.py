"""Best-effort checkpoint metadata; no file contents, no filesystem mutation."""

import json
import logging
import os
from pathlib import Path
import time
import uuid

from ..observability.profiler import agprof

_LOG = logging.getLogger(__name__)

# One read-only probe, with paths on stdin (not interpolated into shell code).
# lstat avoids following symlinks; directory sizes are not file sizes.
_STAT_SCRIPT = """
import json,os,re,stat,sys
rows=[]
# Runtime diff can list mount-point placeholders. lstat would otherwise count
# the mounted host payload (not committed) as checkpoint bytes, e.g. GPU libs.
with open('/proc/self/mountinfo') as stream:
    mounts=[re.sub(r'\\\\([0-7]{3})',lambda m:chr(int(m[1],8)),line.split()[4]) for line in stream]
mounts=[path for path in mounts if path!='/']
for path in json.load(sys.stdin):
    try:
        info=os.lstat(path)
        mounted=any(path==root or path.startswith(root.rstrip('/')+'/') for root in mounts)
        regular=stat.S_ISREG(info.st_mode) and not mounted
        rows.append({'path':path,'size_bytes':info.st_size if regular else None,
                     'kind':'mount_backed' if mounted else ('file' if regular else 'non_regular'),
                     'device':info.st_dev,'inode':info.st_ino,
                     'mtime_ns':info.st_mtime_ns,'ctime_ns':info.st_ctime_ns})
    except OSError as exc:
        rows.append({'path':path,'size_bytes':None,'kind':'unavailable',
                     'errno':exc.errno,'error':type(exc).__name__})
print(json.dumps(rows))
"""


def parse_diff(text):
    rows = []
    for line in text.splitlines():
        change, separator, path = line.partition(" ")
        if change not in {"A", "C", "D"} or not separator or not path.startswith("/"):
            raise ValueError("Unparseable runtime diff; newline-containing paths are unsupported")
        rows.append({"change": change, "path": path})
    return rows


def collect_before(backend):
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "checkpoint_id": uuid.uuid4().hex,
        "container": backend._container_name(),
        "observed_wall_ns": time.time_ns(),
        "errors": [],
        "diff_basis": "container creation image, not previous checkpoint",
        "snapshot_consistency": "best effort; running files may change during collection",
        "mount_size_semantics": "mounted payload excluded; runtime diff may include mount-point placeholders",
        "changes": None,
        "largest_files": [],
    }
    with agprof.span("checkpoint:diagnostics:collect"):
        try:
            result = backend._run(
                [backend._runtime, "diff", report["container"]],
                check=True,
                timeout=backend._agconfig.sandbox.inspect_timeout_s,
            )
            report["changes"] = parse_diff(result.stdout.decode("utf-8", errors="strict"))
            paths = [r["path"] for r in report["changes"] if r["change"] in {"A", "C"}]
            extended = backend._agconfig.sandbox.checkpoint_diagnostics_extended
            if extended:
                from .checkpoint_state import load_previous

                report["schema_version"] = 2
                identity = backend._run(
                    [backend._runtime, "inspect", "--format={{json .}}", report["container"]],
                    check=True,
                    timeout=backend._agconfig.sandbox.inspect_timeout_s,
                )
                identity = json.loads(identity.stdout)
                report["container_id"] = identity["Id"]
                report["base_image_id"] = identity["Image"]
                previous = load_previous(backend)
                if previous and previous.get("container_id") != report["container_id"]:
                    previous = None
                report["_previous"] = previous
                paths = sorted(
                    set(paths)
                    | {r["path"] for r in report["changes"]}
                    | set(previous["snapshot"] if previous else {})
                )
                report["snapshot"] = {}
            if paths:
                result = backend._run(
                    [
                        backend._runtime,
                        "exec",
                        "-i",
                        report["container"],
                        "python3",
                        "-B",
                        "-c",
                        _STAT_SCRIPT,
                    ],
                    input=json.dumps(paths).encode(),
                    check=True,
                    timeout=backend._agconfig.sandbox.exec_quick_timeout_s,
                )
                sizes = json.loads(result.stdout)
                by_path = {r["path"]: r for r in sizes}
                if set(by_path) != set(paths):
                    raise ValueError("Incomplete metadata probe")
                if extended:
                    dirty = {r["path"]: r["change"] for r in report["changes"]}
                    report["snapshot"] = {
                        path: {**row, "dirty": path in dirty, "change": dirty.get(path)}
                        for path, row in by_path.items()
                    }
                for row in report["changes"]:
                    if row["path"] in by_path:
                        row.update(by_path[row["path"]])
                files = [r for r in report["changes"] if r.get("size_bytes") is not None]
                report["largest_files"] = sorted(
                    files, key=lambda r: r["size_bytes"], reverse=True
                )[:20]
            if extended:
                report["snapshot_complete"] = all(
                    r.get("kind") != "unavailable" or r.get("errno") == 2
                    for r in report["snapshot"].values()
                )
        except Exception as exc:
            report["errors"].append({"phase": "before_commit", "type": type(exc).__name__})
    report["collection_seconds"] = time.perf_counter() - started
    return report


def write_after(backend, report, tag, commit_seconds, error):
    """Failure of diagnostics must never change commit success/failure semantics."""
    report.update(
        commit_seconds=commit_seconds,
        committed_wall_ns=time.time_ns(),
        commit_success=error is None,
        commit_error_type=type(error).__name__ if error else None,
        tag=tag,
        image_id=None,
        commit_timing_scope="plain commit including retries/backoff; excludes diagnostics and squash",
    )
    started = time.perf_counter()
    with agprof.span("checkpoint:diagnostics:finalize"):
        try:
            if error is None:
                result = backend._run(
                    [backend._runtime, "image", "inspect", "--format={{.Id}}", tag],
                    check=True,
                    timeout=backend._agconfig.sandbox.inspect_timeout_s,
                )
                report["image_id"] = result.stdout.decode().strip()
        except Exception as exc:
            report["errors"].append({"phase": "image_identity", "type": type(exc).__name__})
        if backend._agconfig.sandbox.checkpoint_diagnostics_extended:
            try:
                from .checkpoint_state import finalize

                finalize(backend, report)
            except Exception as exc:
                report["errors"].append({"phase": "extended", "type": type(exc).__name__})
        report.pop("_previous", None)
        report["collection_seconds"] += time.perf_counter() - started
    with agprof.span("checkpoint:diagnostics:write"):
        try:
            log_dir = backend._agconfig.agent.log_dir
            if not log_dir:
                db_path = backend._agconfig.data_logger.db_path
                log_dir = str(Path(db_path).parent) if db_path else "logs"
            directory = Path(log_dir).resolve() / "checkpoint-diagnostics"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{report['checkpoint_id']}.json"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(report, stream, indent=2)
            if (
                report.get("schema_version") == 2
                and report["commit_success"]
                and report.get("snapshot_complete")
            ):
                from .checkpoint_state import save_previous

                save_previous(backend, report)
        except Exception as exc:
            _LOG.warning("Checkpoint diagnostic output unavailable: %s", type(exc).__name__)
