from __future__ import annotations
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from .agconfig import agConfig, DynamicConfigParam, _AgConfigViewBase


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# Exists to register aglog's config fields (via __set_name__ at import time)
# and hold their hardcoded defaults as plain class attributes -- aglog
# inherits from this below, so self.dump_content_truncate_len etc. work via
# the inherited ConfigParam descriptors exactly as if declared directly on aglog.
class _AgLogFields:
    dump_tool_args_truncate_len = DynamicConfigParam(
        "aglog", default=60
    )  # Max chars of tool call arguments shown in dump() human-readable summary
    dump_content_truncate_len = DynamicConfigParam(
        "aglog", default=120
    )  # Max chars of message content shown per history delta line in dump() output
    dump_tool_call_id_prefix_len = DynamicConfigParam(
        "aglog", default=8
    )  # Number of leading chars of tool_call_id shown in dump() output


class agLogConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting aglog tunables in one call::

        cfg = agConfig(agLogConfig(dump_content_truncate_len=200))

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics.
    """

    _OWNER = "aglog"


class aglog(_AgLogFields):
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
      parent_agname : (forked only) agname of the source agent
    """

    def __init__(
        self, path: "Path | str | None" = None, agconfig: "agConfig | None" = None
    ) -> None:
        self._entries: list[dict] = []  # skill calls only
        self._events: list[dict] = []  # all events (lifecycle + skills)
        self._lock = threading.Lock()
        self._path = Path(path) if path is not None else None
        self._agconfig = agconfig.clone() if agconfig is not None else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def change_config(self, agconfig: "agConfig | None") -> None:
        """Replace this log's agconfig with a clone of the given one."""
        self._agconfig = agconfig.clone() if agconfig is not None else None

    def get_config_copy(self) -> "agConfig | None":
        """Return a clone of this log's agconfig, or None if it has none."""
        return self._agconfig.clone() if self._agconfig is not None else None

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
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        entry = {
            "type": "skill",
            "ts_start": ts_start,
            "ts_end": ts_end,
            "skill": skill,
            "input": input_dict,
            "output": output_dict,
            "history_len": history_len,
            "history_before": history_before if history_before is not None else [],
            "history_delta": history_delta if history_delta is not None else [],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
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
            "type": "tool",
            "ts": _ts(),
            "tool": tool,
            "input": input_dict,
            "output": output_dict,
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
    def token_usage(self) -> dict:
        """Cumulative token usage across all completed skill calls.

        Returns {"input_tokens": int, "output_tokens": int, "total_tokens": int}.
        """
        with self._lock:
            inp = sum(e.get("input_tokens", 0) for e in self._entries)
            out = sum(e.get("output_tokens", 0) for e in self._entries)
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}

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
                if "parent_agname" in e:
                    extra = f"  ← forked from {e['parent_agname']}"
                lines.append(f"[{i}] {e['event'].upper()}  {e['ts']}  agname={e['agname']}{extra}")
            else:
                delta_lines = []
                for m in e.get("history_delta", []):
                    role = m.get("role", "?")
                    if m.get("tool_calls"):
                        calls = ", ".join(
                            f"{tc['function']['name']}({tc['function']['arguments'][: self.dump_tool_args_truncate_len]})"
                            for tc in m["tool_calls"]
                        )
                        delta_lines.append(f"      [{role}] tool_calls: {calls}")
                    elif role == "tool":
                        delta_lines.append(
                            f"      [tool/{m.get('tool_call_id', '')[: self.dump_tool_call_id_prefix_len]}] "
                            f"{str(m.get('content', ''))[: self.dump_content_truncate_len]}"
                        )
                    else:
                        delta_lines.append(
                            f"      [{role}] {str(m.get('content', ''))[: self.dump_content_truncate_len]}"
                        )
                delta_str = ("\n" + "\n".join(delta_lines)) if delta_lines else " (none)"
                inp = e.get("input_tokens", 0)
                out = e.get("output_tokens", 0)
                tok_str = f"in={inp}  out={out}  total={inp + out}" if (inp or out) else "n/a"
                lines.append(
                    f"[{i}] {e['skill']}  {e['ts_start']} → {e['ts_end']}\n"
                    f"    in      : {e['input']}\n"
                    f"    out     : {e['output']}\n"
                    f"    tokens  : {tok_str}\n"
                    f"    hist    : {e['history_len']} messages total\n"
                    f"    delta   :{delta_str}"
                )
        return "\n".join(lines) if lines else "(empty)"
