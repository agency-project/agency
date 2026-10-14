"""Is a provider response something an agent can act on?

A successful HTTP status says the provider answered, not that it produced a
turn. Three shapes come back as a "success" and must not be treated as one:

* **empty** -- no text and no tool call. The harness would take it as the
  model's final (empty) answer and end an unfinished task.
* **truncated** -- generation stopped at the output-token limit
  (stop_reason "length"). Whatever arrived, including a tool call whose
  arguments stop mid-string, is a fragment.
* **malformed tool call** -- arguments that are not a JSON object although the
  provider did not report truncation.

Everything here is provider-agnostic: it reads Agency's own message blocks and
the stop reason every backend already translates to the Chat Completions
vocabulary ("stop", "tool_calls", "length", "content_filter").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

EMPTY = "empty_response"
TRUNCATED = "truncated_output"
MALFORMED_TOOL_CALL = "malformed_tool_call"

_LENGTH_STOP_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})


def tool_arguments_complete(arguments) -> bool:
    """Whether a tool call's argument string is a whole JSON object.

    An empty string is the call "with no arguments" every backend already
    forwards as ``{}``. Anything else must parse, and to an object: a string
    cut off mid-value does not parse, which is what makes this the test for a
    tool call that was truncated in flight.
    """
    if arguments is None or arguments == "":
        return True
    if not isinstance(arguments, str):
        return False
    try:
        return isinstance(json.loads(arguments), dict)
    except ValueError:
        return False


@dataclass(frozen=True)
class ResponseVerdict:
    usable: bool
    reason: "str | None" = None
    detail: str = ""
    # ids of tool calls that must never be executed (arguments incomplete).
    incomplete_tool_calls: tuple = field(default_factory=tuple)


def inspect_response(message: "dict | None", stop_reason, *, empty_is_unusable: bool = True):
    """Classify one assembled assistant message."""
    blocks = [
        block for block in ((message or {}).get("blocks") or []) if block.get("type") != "metadata"
    ]
    tools = [block for block in blocks if block.get("type") == "tool_use"]
    incomplete = tuple(
        block.get("id") or ""
        for block in tools
        if not block.get("name") or not tool_arguments_complete(block.get("arguments"))
    )
    if stop_reason in _LENGTH_STOP_REASONS:
        return ResponseVerdict(
            False,
            TRUNCATED,
            f"generation stopped at the output-token limit (stop_reason={stop_reason!r}); "
            f"{len(incomplete)} of {len(tools)} tool call(s) incomplete",
            incomplete,
        )
    if incomplete:
        return ResponseVerdict(
            False,
            MALFORMED_TOOL_CALL,
            f"{len(incomplete)} tool call(s) have arguments that are not a JSON object "
            f"(stop_reason={stop_reason!r})",
            incomplete,
        )
    has_text = any(
        block.get("type") == "text" and (block.get("text") or "").strip() for block in blocks
    )
    # A reasoning block alone is not an answer; a provider-native block
    # (a refusal, a hosted-tool item) is, and passes through untouched.
    has_native = any(block.get("type") not in ("text", "thinking", "tool_use") for block in blocks)
    if not (tools or has_text or has_native) and empty_is_unusable:
        kinds = sorted({str(block.get("type")) for block in blocks}) or ["nothing"]
        return ResponseVerdict(
            False,
            EMPTY,
            f"response has no text and no tool call (stop_reason={stop_reason!r}, "
            f"blocks: {', '.join(kinds)})",
        )
    return ResponseVerdict(True)


def mark_incomplete_tool_calls(message: dict) -> list:
    """Flag every tool call whose arguments are incomplete so no harness
    adapter can mistake it for an executable call. Returns the flagged ids."""
    flagged = []
    for block in message.get("blocks") or []:
        if block.get("type") != "tool_use":
            continue
        if not block.get("name") or not tool_arguments_complete(block.get("arguments")):
            block["incomplete"] = True
            flagged.append(block.get("id") or "")
    return flagged
