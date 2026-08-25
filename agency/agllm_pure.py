"""Pure, dependency-free compaction algorithms shared by `agllm.py` (the
host-side, in-process LLM call/retry/compaction machinery used by
`execute_react()`) and the in-container native entrypoint
(`agharness_backends/_native_in_container_entrypoint.py`), which needs the
exact same token-estimation/tail-selection/pruning logic but has no
`agllm.py` to import (that module's own import chain pulls in `openai`/
`anthropic`/`boto3` at module level -- exactly what the entrypoint avoids
needing installed, same reasoning as `agtool_pure.py`, which this module
mirrors).

Deliberately zero imports beyond stdlib and zero relative imports -- loaded
by the entrypoint via `importlib.util.spec_from_file_location`, same
technique `agtool_pure.py` already uses.

What's NOT here, because it can't be made pure: actually making the
summarization LLM call. `agllm.py`'s own `compact()` uses `self.call()`
(its full retry/streaming machinery); the entrypoint uses its own
`_dispatch_via_terminus()` instead. Both sides use `build_summary_prompt_
messages()` below to build the same request shape, and `assemble_
compacted_messages()` to splice the result back in the same shape -- only
the actual network round-trip differs.
"""

from __future__ import annotations

CHARS_PER_TOKEN = 4
COMPACT_THRESHOLD = 0.9  # Fraction of context_limit that triggers compaction.
TAIL_FRACTION = 0.25
TAIL_MIN_TOKENS = 2_000
TAIL_MAX_TOKENS = 8_000
TOOL_OUTPUT_MAX_CHARS = 2_000
PRUNE_MIN_FREE_TOKENS = 20_000

DEFAULT_TAIL_TURNS = 3
DEFAULT_SUMMARY_TASK_INPUT_MAX_CHARS = 800
DEFAULT_SUMMARY_ASSISTANT_CONTENT_MAX_CHARS = 800
DEFAULT_SUMMARY_ROLE_CONTENT_MAX_CHARS = 1000

SUMMARY_SYSTEM = """\
You are a conversation summariser. Produce a concise structured summary of \
the conversation history provided. Preserve ALL critical details: decisions, \
file paths, error messages, constraints, user preferences, and tool outputs.

Format exactly (keep every heading, even if a section is empty):

## Goal
<one sentence describing the overall task>

## Constraints & Preferences
<bullet list — coding style, output format, naming conventions, user instructions \
that must be respected going forward>

## Progress
- Done: <completed subtasks>
- In progress: <current subtask>
- Blocked: <anything stuck and why>

## Key Decisions
<bullet list of decisions made and the reasons>

## Next Steps
<ordered bullet list of what remains to be done>

## Critical Context
<facts the agent must remember: variable values, flags, invariants, API responses>

## Relevant Files
<bullet list of every file path created, read, or modified>\
"""


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(msg: dict) -> int:
    chars = len(msg.get("content") or "")
    for tc in msg.get("tool_calls") or []:
        chars += len(tc.get("function", {}).get("arguments", ""))
    return max(1, chars // CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: "list[dict]") -> int:
    """Rough total token count for a list of messages (~4 chars per token)."""
    return sum(estimate_tokens(m) for m in messages)


def should_compact(prompt_tokens: int, context_limit: int) -> bool:
    return prompt_tokens >= int(context_limit * COMPACT_THRESHOLD)


# ---------------------------------------------------------------------------
# Tail-turn selection -- walk backward from the end of the conversation,
# keeping whole ReAct turns (an assistant message plus any tool-role
# messages immediately following it) until either `tail_turns` turns have
# been kept or the token budget for the tail is exhausted.
# ---------------------------------------------------------------------------


def tail_start(conv: "list[dict]", context_limit: int, tail_turns: int = DEFAULT_TAIL_TURNS) -> int:
    if not conv:
        return 0
    usable = int(context_limit * COMPACT_THRESHOLD)
    tail_budget = max(TAIL_MIN_TOKENS, min(TAIL_MAX_TOKENS, int(usable * TAIL_FRACTION)))
    turns_kept = 0
    tokens_kept = 0
    result = len(conv)
    i = len(conv) - 1
    while i >= 0 and turns_kept < tail_turns:
        if conv[i]["role"] != "assistant":
            i -= 1
            continue
        turn_end = i + 1
        while turn_end < len(conv) and conv[turn_end]["role"] == "tool":
            turn_end += 1
        turn_tokens = sum(estimate_tokens(conv[k]) for k in range(i, turn_end))
        if tokens_kept + turn_tokens > tail_budget and turns_kept > 0:
            break
        tokens_kept += turn_tokens
        turns_kept += 1
        result = i
        i -= 1
    return result


def prune_tool_outputs(messages: "list[dict]") -> "list[dict]":
    """Trim oversized tool results; only activates when savings reach
    PRUNE_MIN_FREE_TOKENS."""
    savings_chars = sum(
        len(m.get("content") or "") - TOOL_OUTPUT_MAX_CHARS
        for m in messages
        if m["role"] == "tool" and len(m.get("content") or "") > TOOL_OUTPUT_MAX_CHARS
    )
    if savings_chars // CHARS_PER_TOKEN < PRUNE_MIN_FREE_TOKENS:
        return messages
    result = []
    for m in messages:
        if m["role"] == "tool":
            content = m.get("content") or ""
            if len(content) > TOOL_OUTPUT_MAX_CHARS:
                m = {**m, "content": content[:TOOL_OUTPUT_MAX_CHARS] + "\n[truncated]"}
        result.append(m)
    return result


# ---------------------------------------------------------------------------
# Splitting + reassembly around a summarization call
# ---------------------------------------------------------------------------


def split_for_compaction(
    messages: "list[dict]", context_limit: int, tail_turns: int = DEFAULT_TAIL_TURNS
) -> "tuple[list[dict], list[dict], list[dict], list[dict]]":
    """Split into (sys_msg, task_input, head, tail). `head` is what would
    get summarized away -- empty means there's nothing worth compacting
    (the caller should return `messages` unchanged in that case)."""
    if messages and messages[0]["role"] == "system":
        sys_msg = [messages[0]]
        conv = messages[1:]
    else:
        sys_msg = []
        conv = list(messages)
    ts = tail_start(conv, context_limit, tail_turns)
    task_input = conv[:1]
    head = conv[1:ts]
    tail = conv[ts:]
    return sys_msg, task_input, head, tail


def build_summary_prompt_messages(
    task_input: "list[dict]",
    head: "list[dict]",
    previous_summary: "str | None" = None,
    *,
    task_input_max_chars: int = DEFAULT_SUMMARY_TASK_INPUT_MAX_CHARS,
    assistant_content_max_chars: int = DEFAULT_SUMMARY_ASSISTANT_CONTENT_MAX_CHARS,
    role_content_max_chars: int = DEFAULT_SUMMARY_ROLE_CONTENT_MAX_CHARS,
) -> "list[dict]":
    """The `{"role": "system"/"user", ...}` request messages to send for
    summarization -- send these as-is via whichever dispatch mechanism the
    caller has (agllm.py's `self.call()`, or the in-container entrypoint's
    `_dispatch_via_terminus()`)."""
    lines: "list[str]" = []
    if previous_summary:
        lines.append(
            f"Previous summary (update it — keep true facts, remove stale ones, "
            f"add new ones):\n{previous_summary}\n\nNew conversation to integrate:"
        )
    else:
        lines.append("Conversation to summarise:")
    if task_input:
        lines.append(f"[task input]: {(task_input[0].get('content') or '')[:task_input_max_chars]}")
    for m in head:
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        tool_calls = m.get("tool_calls")
        if role == "assistant" and tool_calls:
            names = ", ".join(tc["function"]["name"] for tc in tool_calls)
            lines.append(f"[assistant → tools: {names}]")
            if content:
                lines.append(f"  {content[:assistant_content_max_chars]}")
        elif role == "tool":
            lines.append(f"[tool result]: {content[:TOOL_OUTPUT_MAX_CHARS]}")
        elif content:
            lines.append(f"[{role}]: {content[:role_content_max_chars]}")
    return [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def assemble_compacted_messages(
    sys_msg: "list[dict]", task_input: "list[dict]", summary: str, tail: "list[dict]"
) -> "list[dict]":
    injection: "list[dict]" = [
        {
            "role": "user",
            "content": (
                "[HARNESS SYSTEM] [Conversation history summary — treat as established context, "
                "do not ask to re-confirm]\n" + summary
            ),
        },
        {
            "role": "assistant",
            "content": "[HARNESS SYSTEM] Understood. I'll continue from this context.",
        },
    ]
    return sys_msg + task_input + injection + tail
