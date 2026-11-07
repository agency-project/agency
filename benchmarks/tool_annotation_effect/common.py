"""Artifact primitives and lazy access to the standalone native package."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def native(module):
    # The standalone package intentionally avoids importing agency.__init__.
    package_root = str(ROOT / "agency")
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    return importlib.import_module("native_harness." + module)


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def source_hash():
    records = []
    for directory in (ROOT / "agency", ROOT / "benchmarks/tool_annotation_effect"):
        for path in sorted(directory.rglob("*.py")):
            records.append(
                (str(path.relative_to(ROOT)), hashlib.sha256(path.read_bytes()).hexdigest())
            )
    return digest(records)


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read_json(path) != value:
                raise ValueError(f"Immutable artifact differs: {path}")
    finally:
        os.unlink(temporary)


def read_events(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    events = []
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except ValueError:
            if index != len(lines) - 1:
                raise
            events.append({"kind": "truncated_trace", "line": index})
    return events


def reject_credentials(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {
                "api_key",
                "token",
                "password",
                "authorization",
                "secret",
                "aws_secret_key",
                "aws_access_key",
                "headers",
            }:
                raise ValueError(f"Credentials must use environment references, not {key}")
            reject_credentials(item)
    elif isinstance(value, list):
        for item in value:
            reject_credentials(item)
