"""On-disk session continuity for the standalone tandem harness.

A one-shot CLI in the Claude-Code mold has no in-memory carry-over between
invocations -- each `tandem_harness -p "..."` call is a fresh process, not
a long-lived one agency stays connected to -- so continuity has to live on
disk instead, the same way Claude Code's own `--resume <session_id>` works
against its own JSONL transcript file (see `claude_code.py`'s module
docstring for that design).

Deliberately a single JSON snapshot, overwritten on save, not an
append-only JSONL log the way Claude Code's real transcript is -- simpler,
and sufficient here since nothing needs crash-resilience mid-session the
way an interactive, long-running CLI session does; this harness only ever
runs one bounded `-p` invocation at a time."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


def new_session_id() -> str:
    return uuid.uuid4().hex


def session_path(session_dir: str, session_id: str) -> Path:
    return Path(session_dir) / f"{session_id}.json"


def load_session(session_dir: str, session_id: str) -> "list[dict] | None":
    """The saved message list for `session_id`, or None if no such session
    exists yet -- callers treat that as "start fresh," not an error."""
    path = session_path(session_dir, session_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    messages = data.get("messages")
    return messages if isinstance(messages, list) else None


def save_session(session_dir: str, session_id: str, model: str, messages: list) -> None:
    path = session_path(session_dir, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id,
        "model": model,
        "updated_at": time.time(),
        "messages": messages,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


__all__ = ["new_session_id", "session_path", "load_session", "save_session"]
