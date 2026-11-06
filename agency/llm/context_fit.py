"""Fit a request into the model's context window by clearing old tool output.

Harnesses compact on their own (docs/Context_Window.md), but a step that adds
a lot at once, or a harness's own summary request, can still exceed the window.
The backend would reject that request, and most harnesses then end the turn
instead of recovering. `LlmHandlerServer` passes every request through
`fit_messages` first: below the limit it is untouched; above it the oldest tool
results lose their text, as Anthropic's `clear_tool_uses` context editing does,
until it fits. Tool calls and their results stay paired.

Sizes are estimated from the backend's own counts of earlier requests
(`SizeEstimator`). If the backend rejects a request for its size anyway, the
caller `shrink`s it and sends it again.
"""

from __future__ import annotations

import copy
import json
import re
from collections import deque

CLEARED = "[Cleared by Agency: the conversation outgrew the model's context window.]"
# Until a backend has counted anything: ~2.5 characters per token, on the dense
# side of prose (4) so that the first estimate errs toward fitting.
DEFAULT_RATE = 0.4
_MIN_RATE, _MAX_RATE = 0.2, 1.0
# Fit to this share of the limit: room for what the estimate gets wrong.
TARGET_SHARE = 0.95
# Steps whose rate the estimate remembers.
_STEPS_KEPT = 8


def message_chars(message: dict) -> int:
    chars = 0
    for block in message.get("blocks") or []:
        for key in ("text", "arguments", "thinking"):
            value = block.get(key)
            if isinstance(value, str):
                chars += len(value)
        if block.get("type") == "tool_definition":
            chars += len(json.dumps(block.get("tool") or {}))
        if "raw_content" in block:
            chars += len(json.dumps(block["raw_content"]))
        if block.get("type") == "openai_responses_item":
            chars += len(json.dumps(block.get("data") or {}))
    return chars


def _tool_result(block: dict) -> "tuple[dict, str] | None":
    """The output field in either normalized or native Responses history."""
    if block.get("type") == "tool_result":
        return block, "text"
    if block.get("type") == "openai_responses_item":
        item = block.get("data") or {}
        if item.get("type") in {"function_call_output", "custom_tool_call_output"}:
            return item, "output"
    return None


def measured_rate(prompt_tokens: int, chars: int) -> "float | None":
    """Tokens per character from one backend count, or None if it says nothing."""
    if prompt_tokens <= 0 or chars <= 0:
        return None
    return min(_MAX_RATE, max(_MIN_RATE, prompt_tokens / chars))


class SizeEstimator:
    """Estimates a request's tokens from the backend's counts of earlier ones.

    The last request's count is exact, and a conversation's next request is
    that one plus what was added. What was added is costed at the rate the
    last such step measured (tokens added / characters added), not at the
    whole conversation's: tool schemas and prose are cheap per character, a
    dense tool output is not."""

    def __init__(self):
        self.rate = DEFAULT_RATE
        self._steps: "deque[float]" = deque(maxlen=_STEPS_KEPT)
        self.last: "tuple[int, int] | None" = None  # (characters, tokens)
        self.last_fitted = False

    @property
    def step_rate(self) -> float:
        """The densest of the last few steps: what one adds varies (a short
        listing, then a file of dense text), and the next may be the dense
        kind."""
        return max(self._steps, default=DEFAULT_RATE)

    def observe(self, chars: int, tokens: int, fitted: bool = False) -> None:
        """Take the backend's count of a request of `chars` characters;
        `fitted` if this module cleared some of it, which makes it no
        measure of what one step adds."""
        rate = measured_rate(tokens, chars)
        if rate is None:
            return
        self.rate = rate
        clean = not fitted and not self.last_fitted
        if clean and self.last is not None and chars > self.last[0] and tokens > self.last[1]:
            step = measured_rate(tokens - self.last[1], chars - self.last[0])
            if step is not None:
                self._steps.append(step)
        self.last = (chars, tokens)
        self.last_fitted = fitted

    def budget(self, chars: int, limit: int) -> int:
        """How many characters a request may have to stay within `limit`."""
        target = int(limit * TARGET_SHARE)
        if self.last is not None and chars >= self.last[0] and target > self.last[1]:
            return self.last[0] + int((target - self.last[1]) / max(self.step_rate, self.rate))
        return int(target / max(self.step_rate, self.rate))


def fit_messages(messages: "list[dict]", budget: int) -> "tuple[list[dict], int]":
    """`messages` cut to `budget` characters (`SizeEstimator.budget`), and how
    many tool results were cleared. Unchanged (the same list) when they fit."""
    chars = [message_chars(m) for m in messages]
    total = sum(chars)
    if total <= budget:
        return messages, 0
    fitted = copy.deepcopy(messages)
    results = [
        i
        for i, m in enumerate(fitted)
        if any(_tool_result(b) is not None for b in m.get("blocks") or [])
    ]
    cleared = 0
    # Oldest first, keeping the newest result: it is what the model asked for.
    for i in results[:-1]:
        if total <= budget:
            break
        before = chars[i]
        for block in fitted[i]["blocks"]:
            result = _tool_result(block)
            if result is not None:
                output, field = result
                output[field] = CLEARED
                output.pop("raw_content", None)
        chars[i] = message_chars(fitted[i])
        total -= before - chars[i]
        cleared += before > chars[i]
    if total > budget and results:
        # The newest result alone is too big: keep its head and tail.
        last = fitted[results[-1]]
        for block in last["blocks"]:
            result = _tool_result(block)
            if result is not None and isinstance(result[0].get(result[1]), str):
                output, field = result
                text = output[field]
                before = message_chars(last)
                target = before - (total - budget)
                original = output.copy()
                output.pop("raw_content", None)

                def trimmed(half):
                    return text[:half] + "\n" + CLEARED + "\n" + text[len(text) - half :]

                # Native items are measured as JSON, so escaped newlines and
                # non-ASCII output need the same accounting while trimming.
                low, high = 0, len(text) // 2
                while low < high:
                    half = (low + high + 1) // 2
                    output[field] = trimmed(half)
                    if message_chars(last) <= target:
                        low = half
                    else:
                        high = half - 1
                output[field] = trimmed(low)
                after = message_chars(last)
                if after < before:
                    total -= before - after
                    cleared += 1
                else:
                    output.clear()
                    output.update(original)
                if total <= budget:
                    break
    return fitted, cleared


# How providers say a prompt exceeded the model's window (Anthropic, OpenAI,
# Bedrock, vLLM, LiteLLM's wrappers of them).
_CONTEXT_OVERFLOW = re.compile(
    r"prompt is too long|context[_ ]length[_ ]exceeded|maximum context length"
    r"|context length|context window|input is too long|too large for model"
    r"|exceeds? the maximum number of (?:input )?tokens",
    re.IGNORECASE,
)
_NOT_OVERFLOW = re.compile(r"rate limit|too many requests|per minute", re.IGNORECASE)


def is_context_overflow(message: str) -> bool:
    """Whether a backend error says the prompt exceeded the model's window."""
    return bool(_CONTEXT_OVERFLOW.search(message)) and not _NOT_OVERFLOW.search(message)


def shrink(messages: "list[dict]", share: float = 0.75) -> "tuple[list[dict], int]":
    """`messages` cut to `share` of their size, for a request the backend
    rejected although the estimate said it fit."""
    return fit_messages(messages, int(sum(message_chars(m) for m in messages) * share))


__all__ = [
    "CLEARED",
    "SizeEstimator",
    "fit_messages",
    "is_context_overflow",
    "measured_rate",
    "message_chars",
    "shrink",
]
