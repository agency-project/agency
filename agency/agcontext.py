from __future__ import annotations
import copy
from typing import TYPE_CHECKING

from . import agpause

if TYPE_CHECKING:
    from concurrent.futures import Future


class agcontext:
    """Persistent conversation state owned by an agent and passed through each skill run.

    Accumulates across all skill calls on the same agent so compaction state
    carries forward for the lifetime of the agent session.

    Fields
    ------
    recent_transcript   : reconstructed transcript of the most recent skill run
                           (overwritten each run, not accumulated)
    compaction_summary  : rolling summary produced by conversation compaction
    """

    def __init__(
        self,
        recent_transcript: "list[dict] | None" = None,
        compaction_summary: "str | None" = None,
        _future: "Future[agcontext] | None" = None,
    ) -> None:
        self.recent_transcript = recent_transcript if recent_transcript is not None else []
        self.compaction_summary = compaction_summary
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
        with agpause.note_blocked_on(agpause.producer_of(self._future)):
            prev_ctx = self._future.result()
        self.recent_transcript = prev_ctx.recent_transcript
        self.compaction_summary = prev_ctx.compaction_summary
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
            compaction_summary=self.compaction_summary,
        )

    def __repr__(self) -> str:
        pending = " (pending)" if self._future is not None else ""
        return (
            f"agcontext(recent_transcript={len(self.recent_transcript)}"
            f"  compact={'yes' if self.compaction_summary else 'no'}"
            f"){pending}"
        )
