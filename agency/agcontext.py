from __future__ import annotations
import copy
from typing import TYPE_CHECKING

from . import agpause

if TYPE_CHECKING:
    from concurrent.futures import Future


class agcontext:
    """Persistent conversation state owned by an agent and passed through each skill run.

    Accumulates across all skill calls on the same agent so token totals and
    compaction state carry forward for the lifetime of the agent session.

    Fields
    ------
    messages            : conversation history (excludes the per-skill system prompt)
    total_input_tokens  : cumulative input tokens across all LLM calls so far
    total_output_tokens : cumulative output tokens across all LLM calls so far
    compaction_summary  : rolling summary produced by conversation compaction
    """

    def __init__(
        self,
        messages: "list[dict] | None" = None,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
        compaction_summary: "str | None" = None,
        _future: "Future[agcontext] | None" = None,
    ) -> None:
        self.messages = messages if messages is not None else []
        self.total_input_tokens = total_input_tokens
        self.total_output_tokens = total_output_tokens
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
        self.messages = prev_ctx.messages
        self.total_input_tokens = prev_ctx.total_input_tokens
        self.total_output_tokens = prev_ctx.total_output_tokens
        self.compaction_summary = prev_ctx.compaction_summary
        self._future = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_resolved_messages(self) -> "list[dict]":
        """Block until pending, then return a snapshot of the message list."""
        self.resolve_prev_dependencies()
        return list(self.messages)

    def set_messages(self, messages: "list[dict]") -> None:
        """Replace the message list directly."""
        self.messages = list(messages)

    def copy(self) -> "agcontext":
        """Return a deep copy of the resolved context (blocks if pending)."""
        self.resolve_prev_dependencies()
        return agcontext(
            messages=copy.deepcopy(self.messages),
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            compaction_summary=self.compaction_summary,
        )

    def __repr__(self) -> str:
        pending = " (pending)" if self._future is not None else ""
        return (
            f"agcontext(msgs={len(self.messages)}"
            f"  in={self.total_input_tokens}"
            f"  out={self.total_output_tokens}"
            f"  compact={'yes' if self.compaction_summary else 'no'}"
            f"){pending}"
        )
