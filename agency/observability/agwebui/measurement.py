"""User wall-clock measurement overrides, persisted beside the execution events.

Execution events stay immutable. A user's "end now", "keep running" and
"reset to automatic" choices are appended to ``run_dir/measurement.jsonl``;
the trajectory reducer overlays them on the automatic boundary it derives
from those events. No agency imports: the standalone server uses this too.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

LOG_NAME = "measurement.jsonl"
LEASE_NAME = "measurement.lease"
COMMANDS = ("end", "hold", "reset")
# A held-open measurement is renewed by a running server. After this long
# without renewal that server is gone and the measurement ended with it.
LEASE_INTERVAL = 5.0
LEASE_STALE = 30.0
MAX_RECORDS = 1000


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse(line) -> "dict | None":
    """Validate one stored record; a damaged or hand-edited line is rejected."""
    try:
        record = json.loads(line)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(record, dict) or record.get("type") not in COMMANDS:
        return None
    identifier = record.get("id")
    if not isinstance(identifier, str) or not identifier or len(identifier) > 128:
        return None
    if not finite(record.get("recorded_at")):
        return None
    if record["type"] == "end" and not finite(record.get("at")):
        return None
    return record


def read(path: Path, offset: int = 0) -> "tuple[list[dict], int, int]":
    """Return (records, next offset, rejected lines) from complete lines after *offset*."""
    try:
        with open(path, "rb") as stream:
            stream.seek(offset)
            data = stream.read()
    except OSError:
        return [], offset, 0
    # A line still being appended by another process is left for the next read.
    complete = data[: data.rfind(b"\n") + 1]
    records, rejected = [], 0
    for line in complete.splitlines():
        if not line.strip():
            continue
        record = parse(line)
        if record is None:
            rejected += 1
        else:
            records.append(record)
    return records, offset + len(complete), rejected


def append(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_lease(directory: Path, held_until: float) -> None:
    path = directory / LEASE_NAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"held_until": held_until}), encoding="utf-8")
    tmp.replace(path)


def read_lease(directory: Path) -> "float | None":
    try:
        value = json.loads((directory / LEASE_NAME).read_text(encoding="utf-8")).get("held_until")
    except (OSError, ValueError, AttributeError):
        return None
    return value if finite(value) else None
