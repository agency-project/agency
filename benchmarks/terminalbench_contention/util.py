"""Small shared helpers: durable JSON, hashing, redaction, subprocess capture."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent


def atomic_json(path, value, *, indent=2) -> None:
    """Write JSON so a reader never sees a partial file, and it survives a crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=indent, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_json_or(path, default=None):
    """Read JSON that may be absent or torn (a partially written artifact)."""
    try:
        return read_json(path)
    except (OSError, ValueError):
        return default


def read_jsonl(path, *, strict=False) -> list:
    """Read JSON lines. A torn trailing line is skipped unless ``strict``."""
    rows = []
    try:
        text = Path(path).read_text()
    except OSError:
        return rows
    for line in text.split("\n"):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            if strict:
                raise
    return rows


def append_jsonl(path, record) -> None:
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record, default=str) + "\n")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(directory, *, ignore=("__pycache__",)) -> "str | None":
    directory = Path(directory)
    if not directory.is_dir():
        return None
    outer = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        if any(part in ignore for part in path.parts) or path.suffix == ".pyc":
            continue
        outer.update(path.relative_to(directory).as_posix().encode() + b"\0")
        outer.update(bytes.fromhex(sha256_file(path)))
    return outer.hexdigest()


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def clock_anchor() -> dict:
    """One (wall, monotonic) pair. All traces use CLOCK_MONOTONIC ns, which is
    shared by every process on the host; this maps it to wall-clock time."""
    mono = time.monotonic_ns()
    wall = time.time_ns()
    return dict(
        monotonic_ns=mono,
        wall_ns=wall,
        epoch_offset_ns=wall - mono,
        clock="CLOCK_MONOTONIC",
        note="wall_ns = monotonic_ns + epoch_offset_ns",
    )


_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)(\"?\s*[:=]\s*\"?)[^\s\"',}]{6,}"),
)


def redact(text, extra_secrets=()) -> str:
    """Strip credentials from text bound for an artifact."""
    text = str(text)
    for secret in extra_secrets:
        if secret and len(secret) >= 6:
            text = text.replace(secret, "<redacted>")
    text = _SECRET_PATTERNS[0].sub("<redacted>", text)
    text = _SECRET_PATTERNS[1].sub("Bearer <redacted>", text)
    text = _SECRET_PATTERNS[2].sub(lambda m: f"{m.group(1)}{m.group(2)}<redacted>", text)
    return text


def capture(argv, timeout=60, **kwargs) -> dict:
    """Run a command and always return a record, never raise."""
    try:
        result = subprocess.run(
            [str(a) for a in argv], text=True, capture_output=True, timeout=timeout, **kwargs
        )
        return dict(
            argv=[str(a) for a in argv],
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(
            argv=[str(a) for a in argv], returncode=None, stdout="", stderr="", error=str(exc)
        )


def checked(argv, timeout=60, **kwargs) -> str:
    result = capture(argv, timeout=timeout, **kwargs)
    if result["returncode"] != 0:
        detail = result.get("error") or result["stderr"].strip() or result["stdout"].strip()
        raise RuntimeError(f"{' '.join(result['argv'][:4])} failed: {detail}")
    return result["stdout"].strip()


def load_env_file(path) -> dict:
    """Parse KEY=VALUE lines without exporting them or echoing values."""
    values = {}
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().removeprefix("export ").strip()] = value.strip().strip("'\"")
    return values


def git_state(repository=REPOSITORY) -> dict:
    head = capture(["git", "-C", repository, "rev-parse", "HEAD"])
    status = capture(["git", "-C", repository, "status", "--porcelain"])
    branch = capture(["git", "-C", repository, "rev-parse", "--abbrev-ref", "HEAD"])
    return dict(
        commit=head["stdout"].strip() or None,
        branch=branch["stdout"].strip() or None,
        dirty=bool(status["stdout"].strip()),
        dirty_paths=status["stdout"].splitlines()[:200],
    )
