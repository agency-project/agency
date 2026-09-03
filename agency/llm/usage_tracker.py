"""Per-agent tracking of new (non-cumulative) prompt tokens across LLM exchanges.

A provider's reported `prompt_tokens` is the size of the ENTIRE conversation
sent for that one call, not just what's new since the previous exchange --
summing it raw across exchanges wildly overcounts. This isolates the delta
by matching each new exchange's messages against the tail state of whichever
branch it continues (if any), which stays correct even under harness-
internal compaction we have no visibility into: a shrink in prompt size just
floors the delta at zero instead of corrupting the running total, since
tokens already counted once stay counted.

Branch matching (not simple "compare to the previous exchange") is required
because the host server architecture admits multiple concurrent in-flight
LLM calls within one skill execution (see HostServerManager's attempt-lease
counter) -- two interleaved exchanges are not necessarily a continuation of
each other.
"""

from __future__ import annotations

import threading

_MAX_TRACKED_BRANCHES = 32


class LlmUsageTracker:
    """One instance per agent, living as long as the agent does -- so a
    delta can be computed against the previous exchange even across skill
    call boundaries, with no sqlite read required."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tips: "list[dict]" = []

    def resolve_new_prompt_tokens(
        self,
        messages: "list[dict]",
        response_message: dict,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> int:
        """Return this exchange's new prompt tokens and record its resulting
        state as the new tip of whichever branch it continues (or starts)."""
        with self._lock:
            match_index = None
            for index, tip in enumerate(self._tips):
                state_len = tip["state_len"]
                if state_len <= len(messages) and messages[:state_len] == tip["state"]:
                    match_index = index
                    break
            if match_index is not None:
                tip = self._tips.pop(match_index)
                new_prompt_tokens = max(
                    0, prompt_tokens - (tip["prompt_tokens"] + tip["completion_tokens"])
                )
            else:
                new_prompt_tokens = prompt_tokens
            new_state = messages + [response_message]
            self._tips.append(
                {
                    "state": new_state,
                    "state_len": len(new_state),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )
            if len(self._tips) > _MAX_TRACKED_BRANCHES:
                del self._tips[: len(self._tips) - _MAX_TRACKED_BRANCHES]
            return new_prompt_tokens
