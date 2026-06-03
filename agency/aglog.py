from __future__ import annotations
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from .agdata import agdata


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class aglog:
    """Structured, thread-safe log of all agskill calls and lifecycle events on an agent.

    Automatically populated by agent — no manual calls required.

    Two views:
      entries   — skill calls only (list of dicts with type="skill")
      events    — full timeline: lifecycle events + skill calls in order

    Skill entry fields:
      type        : "skill"
      ts_start    : ISO-8601 when the skill was submitted
      ts_end      : ISO-8601 when the skill completed
      skill       : name of the agskill that ran
      input       : resolved input as a plain dict
      output      : result as a plain dict (or {"error": ...})
      history_len : number of messages in history after this call

    Lifecycle entry fields:
      type        : "lifecycle"
      event       : "created" | "forked" | "destroyed"
      ts          : ISO-8601 timestamp
      uuid        : agent UUID
      parent_uuid : (forked only) UUID of the source agent
    """

    def __init__(self, path: "Path | str | None" = None) -> None:
        self._entries: list[dict] = []   # skill calls only
        self._events:  list[dict] = []   # all events (lifecycle + skills)
        self._lock = threading.Lock()
        self._path = Path(path) if path is not None else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal — called by agent
    # ------------------------------------------------------------------

    def _record(
        self,
        skill: str,
        ts_start: str,
        ts_end: str,
        input_dict: dict,
        output_dict: dict,
        history_len: int,
        history_before: list[dict] | None = None,
        history_delta: list[dict] | None = None,
    ) -> None:
        entry = {
            "type":           "skill",
            "ts_start":       ts_start,
            "ts_end":         ts_end,
            "skill":          skill,
            "input":          input_dict,
            "output":         output_dict,
            "history_len":    history_len,
            "history_before": history_before if history_before is not None else [],
            "history_delta":  history_delta  if history_delta  is not None else [],
        }
        with self._lock:
            self._entries.append(entry)
            self._events.append(entry)
            self._write(entry)

    def _tool_call(
        self,
        tool: str,
        input_dict: dict,
        output_dict: dict,
        elapsed_ms: int,
    ) -> None:
        """Record a single tool invocation."""
        entry = {
            "type":       "tool",
            "ts":         _ts(),
            "tool":       tool,
            "input":      input_dict,
            "output":     output_dict,
            "elapsed_ms": elapsed_ms,
        }
        with self._lock:
            self._events.append(entry)
            self._write(entry)

    def _lifecycle(self, event: str, **kwargs) -> None:
        """Record a lifecycle event (created / forked / destroyed)."""
        entry = {"type": "lifecycle", "event": event, "ts": _ts(), **kwargs}
        with self._lock:
            self._events.append(entry)
            self._write(entry)

    def _write(self, entry: dict) -> None:
        """Append one JSON line to the log file (must be called under _lock)."""
        if self._path is not None:
            with self._path.open("a") as f:
                f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[dict]:
        """Skill-call entries only (backward-compatible view)."""
        with self._lock:
            return list(self._entries)

    @property
    def events(self) -> list[dict]:
        """Full event timeline: lifecycle events + skill calls in chronological order."""
        with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        """Number of completed skill calls (not counting lifecycle events)."""
        return len(self._entries)

    def __repr__(self) -> str:
        return f"aglog({len(self._entries)} skill calls, {len(self._events)} events total)"

    def dump(self) -> str:
        """Human-readable summary of the full event timeline."""
        lines = []
        for i, e in enumerate(self.events):
            if e["type"] == "lifecycle":
                extra = ""
                if "parent_uuid" in e:
                    extra = f"  ← forked from {e['parent_uuid'][:8]}"
                lines.append(
                    f"[{i}] {e['event'].upper()}  {e['ts']}  uuid={e['uuid'][:8]}{extra}"
                )
            else:
                delta_lines = []
                for m in e.get("history_delta", []):
                    role = m.get("role", "?")
                    if m.get("tool_calls"):
                        calls = ", ".join(
                            f"{tc['function']['name']}({tc['function']['arguments'][:60]})"
                            for tc in m["tool_calls"]
                        )
                        delta_lines.append(f"      [{role}] tool_calls: {calls}")
                    elif role == "tool":
                        delta_lines.append(
                            f"      [tool/{m.get('tool_call_id','')[:8]}] "
                            f"{str(m.get('content',''))[:120]}"
                        )
                    else:
                        delta_lines.append(
                            f"      [{role}] {str(m.get('content',''))[:120]}"
                        )
                delta_str = ("\n" + "\n".join(delta_lines)) if delta_lines else " (none)"
                lines.append(
                    f"[{i}] {e['skill']}  {e['ts_start']} → {e['ts_end']}\n"
                    f"    in      : {e['input']}\n"
                    f"    out     : {e['output']}\n"
                    f"    hist    : {e['history_len']} messages total\n"
                    f"    delta   :{delta_str}"
                )
        return "\n".join(lines) if lines else "(empty)"
