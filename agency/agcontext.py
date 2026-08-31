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
    """

    def __init__(
        self,
        recent_transcript: "list[dict] | None" = None,
        harness_sessions: "dict[str, dict] | None" = None,
        _future: "Future[agcontext] | None" = None,
    ) -> None:
        self.recent_transcript = recent_transcript if recent_transcript is not None else []
        self.harness_sessions = harness_sessions if harness_sessions is not None else {}
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

    def copy(self) -> "agcontext":
        """Return a deep copy of the resolved context (blocks if pending)."""
        self.resolve_prev_dependencies()
        return agcontext(
            recent_transcript=copy.deepcopy(self.recent_transcript),
            harness_sessions=copy.deepcopy(self.harness_sessions),
        )

    def __repr__(self) -> str:
        pending = " (pending)" if self._future is not None else ""
        return (
            f"agcontext(recent_transcript={len(self.recent_transcript)}"
            f"  harnesses={sorted(self.harness_sessions)}"
            f"){pending}"
        )
