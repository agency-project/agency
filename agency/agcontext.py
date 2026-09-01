from __future__ import annotations
import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from concurrent.futures import Future


class agcontext:
    """Persistent conversation state owned by an agent and passed through each skill run.

    Fields
    ------
    recent_transcript   : reconstructed transcript of the most recent skill run
                           (overwritten each run, not accumulated)
    harness_sessions    : per-harness session continuity, e.g.
                           {"claude_code": {"session_id": ..., "blob_b64": ...}}
    retained_messages   : ordered user/system messages retained independently
                           of a harness transcript or restorable session
    harness_message_cursors
                        : last retained-message sequence incorporated into
                           each stateful harness session
    """

    def __init__(
        self,
        recent_transcript: "list[dict] | None" = None,
        harness_sessions: "dict[str, dict] | None" = None,
        retained_messages: "list[dict] | None" = None,
        harness_message_cursors: "dict[str, int] | None" = None,
        _future: "Future[agcontext] | None" = None,
    ) -> None:
        self.recent_transcript = recent_transcript if recent_transcript is not None else []
        self.harness_sessions = harness_sessions if harness_sessions is not None else {}
        self.retained_messages = retained_messages if retained_messages is not None else []
        self.harness_message_cursors = (
            harness_message_cursors if harness_message_cursors is not None else {}
        )
        self._future = _future

    # ------------------------------------------------------------------
    # Future / lazy-resolution support
    # ------------------------------------------------------------------

    def is_pending(self) -> bool:
        return self._future is not None

    def resolve_prev_dependencies(self) -> None:
        """Block until the pending future resolves and merge its state into self."""
        if self._future is None:
            return
        prev_ctx = self._future.result()
        self.recent_transcript = prev_ctx.recent_transcript
        self.harness_sessions = prev_ctx.harness_sessions
        self.retained_messages = prev_ctx.retained_messages
        self.harness_message_cursors = prev_ctx.harness_message_cursors
        self._future = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_resolved_transcript(self) -> "list[dict]":
        """Block until pending, then return a snapshot of the recent transcript."""
        self.resolve_prev_dependencies()
        return list(self.recent_transcript)

    def set_transcript(self, recent_transcript: "list[dict]") -> None:
        """Replace the recent transcript directly."""
        self.recent_transcript = list(recent_transcript)

    def append_retained_message(self, message: dict) -> int:
        """Append one validated, JSON-serializable retained message."""
        entry = copy.deepcopy(message)
        sequence = entry.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            raise ValueError("retained message sequence must be a positive integer")
        if entry.get("type") != "message":
            raise ValueError("retained message type must be 'message'")
        if entry.get("role") not in {"user", "system"}:
            raise ValueError("retained message role must be 'user' or 'system'")
        if not isinstance(entry.get("content"), str):
            raise ValueError("retained message content must be a string")
        if self.retained_messages and sequence <= int(self.retained_messages[-1]["sequence"]):
            raise ValueError("retained message sequences must be strictly increasing")
        self.retained_messages.append(entry)
        return sequence

    def pending_retained_messages(self, harness: str) -> "list[dict]":
        """Return retained entries not captured by *harness*'s session."""
        cursor = int(self.harness_message_cursors.get(str(harness), 0))
        return copy.deepcopy(
            [entry for entry in self.retained_messages if int(entry["sequence"]) > cursor]
        )

    def advance_retained_cursor(self, harness: str, sequence: int) -> None:
        """Monotonically mark retained messages through *sequence* as session-backed."""
        harness = str(harness)
        sequence = int(sequence)
        current = int(self.harness_message_cursors.get(harness, 0))
        if sequence < current:
            raise ValueError("retained message cursor cannot move backwards")
        self.harness_message_cursors[harness] = sequence

    def copy(self) -> "agcontext":
        """Return a deep copy of the resolved context (blocks if pending)."""
        self.resolve_prev_dependencies()
        return agcontext(
            recent_transcript=copy.deepcopy(self.recent_transcript),
            harness_sessions=copy.deepcopy(self.harness_sessions),
            retained_messages=copy.deepcopy(self.retained_messages),
            harness_message_cursors=copy.deepcopy(self.harness_message_cursors),
        )

    def __repr__(self) -> str:
        pending = " (pending)" if self._future is not None else ""
        return (
            f"agcontext(recent_transcript={len(self.recent_transcript)}"
            f"  harnesses={sorted(self.harness_sessions)}"
            f"  retained={len(self.retained_messages)}"
            f"){pending}"
        )
