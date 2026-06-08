from __future__ import annotations
import httpx
import openai

# --- Constants matching opencode's design -----------------------------------

# Fire when prompt is within this many tokens of the context limit.
# opencode default: 20 000.  Floor at 50% prevents nonsense on small models.
_RESERVED = 20_000

TAIL_TURNS = 2          # max recent assistant turns to keep verbatim
_TAIL_FRACTION = 0.25   # fraction of usable context budgeted for the tail
_TAIL_MIN_TOKENS = 2_000
_TAIL_MAX_TOKENS = 8_000

# Tool-output pruning: applied to the head before summarisation.
# Only activates when potential savings >= _PRUNE_MIN_FREE_TOKENS.
_TOOL_OUTPUT_MAX_CHARS = 2_000
_PRUNE_MIN_FREE_TOKENS = 20_000

# --- Summary prompt ----------------------------------------------------------
# Seven sections, matching opencode's structure exactly.

_SUMMARY_SYSTEM = """\
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


# --- Public helpers ----------------------------------------------------------

def fetch_context_limit(llm_config: dict) -> int | None:
    """Return the model's context window size, or None if unavailable.

    Priority:
    1. ``llm_config["context_limit"]`` — explicit user override
    2. vLLM ``max_model_len`` from ``GET /v1/models/{model}``
    """
    if "context_limit" in llm_config:
        return int(llm_config["context_limit"])
    try:
        client = openai.OpenAI(
            api_key=llm_config.get("api_key", ""),
            base_url=llm_config.get("base_url"),
        )
        info = client.models.retrieve(llm_config.get("model", ""))
        extra = getattr(info, "model_extra", None) or {}
        if "max_model_len" in extra:
            return int(extra["max_model_len"])
    except Exception:
        pass
    return None


def should_compact(prompt_tokens: int, context_limit: int) -> bool:
    """Return True when the prompt is within _RESERVED tokens of the context limit.

    A 50% floor prevents the threshold from going negative on small models.
    """
    threshold = max(context_limit - _RESERVED, context_limit // 2)
    return prompt_tokens >= threshold


# --- Internal helpers --------------------------------------------------------

def count_messages_tokens(messages: list[dict], llm_config: dict) -> int:
    """Return the token count for *messages*, using the vLLM /tokenize endpoint
    when available and falling back to the character-based estimate otherwise.

    The vLLM endpoint is at ``<base_url_without_v1>/tokenize`` and accepts::

        POST /tokenize
        {"model": "...", "messages": [...]}

    Response: ``{"tokens": [...], "count": N, "max_model_len": N}``
    """
    base_url: str = llm_config.get("base_url", "") or ""
    # Strip trailing /v1 (or /v1/) to reach the vLLM root
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    if root:
        try:
            resp = httpx.post(
                f"{root}/tokenize",
                json={
                    "model": llm_config.get("model", ""),
                    "messages": [{k: v for k, v in m.items() if not k.startswith("_")}
                                 for m in messages],
                },
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if "count" in data:
                return int(data["count"])
            if "tokens" in data:
                return len(data["tokens"])
        except Exception:
            pass
    return estimate_messages_tokens(messages)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Rough total token count for a list of messages (~4 chars per token)."""
    return sum(_estimate_tokens(m) for m in messages)


def _estimate_tokens(msg: dict) -> int:
    """Rough token count via the ~4 chars-per-token rule."""
    chars = len(msg.get("content") or "")
    for tc in (msg.get("tool_calls") or []):
        chars += len(tc.get("function", {}).get("arguments", ""))
    return max(1, chars // 4)


def _tail_start(conv: list[dict], context_limit: int,
                tail_turns: int = TAIL_TURNS) -> int:
    """Return the index of the first message in the verbatim tail.

    Uses a token budget (``_TAIL_FRACTION`` of usable context, bounded to
    ``_TAIL_MIN_TOKENS``–``_TAIL_MAX_TOKENS``) rather than a fixed turn count,
    matching opencode's approach.  The turn count is still an upper bound.

    A "turn" is one assistant message plus all immediately following
    tool-result messages.
    """
    if not conv:
        return 0

    usable = max(context_limit - _RESERVED, context_limit // 2)
    tail_budget = max(_TAIL_MIN_TOKENS, min(_TAIL_MAX_TOKENS,
                                            int(usable * _TAIL_FRACTION)))

    turns_kept = 0
    tokens_kept = 0
    result = len(conv)   # default: tail starts past the end (keep nothing)

    i = len(conv) - 1
    while i >= 0 and turns_kept < tail_turns:
        if conv[i]["role"] != "assistant":
            i -= 1
            continue

        # Measure this turn: assistant msg + following tool results
        turn_end = i + 1
        while turn_end < len(conv) and conv[turn_end]["role"] == "tool":
            turn_end += 1
        turn_tokens = sum(_estimate_tokens(conv[k]) for k in range(i, turn_end))

        # Accept the turn unless it blows the budget and we already have one
        if tokens_kept + turn_tokens > tail_budget and turns_kept > 0:
            break

        tokens_kept += turn_tokens
        turns_kept += 1
        result = i
        i -= 1

    return result


def _prune_tool_outputs(messages: list[dict]) -> list[dict]:
    """Trim oversized tool results in the messages list.

    Only activates when the potential token savings would reach
    ``_PRUNE_MIN_FREE_TOKENS``.  Each tool message is truncated to at most
    ``_TOOL_OUTPUT_MAX_CHARS`` characters.
    """
    savings_chars = sum(
        len(m.get("content") or "") - _TOOL_OUTPUT_MAX_CHARS
        for m in messages
        if m["role"] == "tool" and len(m.get("content") or "") > _TOOL_OUTPUT_MAX_CHARS
    )
    if savings_chars // 4 < _PRUNE_MIN_FREE_TOKENS:
        return messages

    result = []
    for m in messages:
        if m["role"] == "tool":
            content = m.get("content") or ""
            if len(content) > _TOOL_OUTPUT_MAX_CHARS:
                m = {**m, "content": content[:_TOOL_OUTPUT_MAX_CHARS] + "\n[truncated]"}
        result.append(m)
    return result


def compact(
    messages: list[dict],
    llm_config: dict,
    *,
    context_limit: int | None = None,
    tail_turns: int = TAIL_TURNS,
    previous_summary: str | None = None,
) -> tuple[list[dict], str]:
    """Summarise old messages and return a compacted list plus the new summary.

    Structure of the returned message list::

        [system]                         ← always preserved
        [user: task input]               ← conv[0], always preserved
        [user:  summary injection]       ← replaces the compacted head
        [assistant: "Understood…"]
        [tail turns verbatim]            ← last N turns kept as-is

    The task-input user message (the skill's entry point, conv[0]) is always
    kept outside the summarised region so the LLM retains the original goal.

    ``context_limit`` is used to size the tail token budget.  When omitted,
    a 128K default is used (tail budget capped at ``_TAIL_MAX_TOKENS``).

    Returns ``(new_messages, summary_text)``.
    """
    cl = context_limit if context_limit is not None else 128_000

    if messages and messages[0]["role"] == "system":
        sys_msg: list[dict] = [messages[0]]
        conv = messages[1:]
    else:
        sys_msg = []
        conv = list(messages)

    # conv[0] is the task-input user message — always kept verbatim.
    # The summarisable region is conv[1:ts] (the ReAct turns between the
    # task input and the retained tail).
    ts = _tail_start(conv, cl, tail_turns)
    task_input: list[dict] = conv[:1]    # always [conv[0]] or []
    head = conv[1:ts]                    # ReAct turns to summarise
    tail = conv[ts:]                     # recent turns to keep verbatim

    if not head:
        return messages, previous_summary or ""

    # Prune large tool outputs before summarising
    head = _prune_tool_outputs(head)

    # Build the summarisation prompt
    lines: list[str] = []
    if previous_summary:
        lines.append(
            f"Previous summary (update it — keep true facts, remove stale ones, "
            f"add new ones):\n{previous_summary}\n\nNew conversation to integrate:"
        )
    else:
        lines.append("Conversation to summarise:")

    # Include the task input so the summariser knows the original goal
    if task_input:
        lines.append(f"[task input]: {(task_input[0].get('content') or '')[:400]}")

    for m in head:
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        tool_calls = m.get("tool_calls")
        if role == "assistant" and tool_calls:
            names = ", ".join(tc["function"]["name"] for tc in tool_calls)
            lines.append(f"[assistant → tools: {names}]")
            if content:
                lines.append(f"  {content[:400]}")
        elif role == "tool":
            lines.append(f"[tool result]: {content[:_TOOL_OUTPUT_MAX_CHARS]}")
        elif content:
            lines.append(f"[{role}]: {content[:600]}")

    client = openai.OpenAI(
        api_key=llm_config.get("api_key", ""),
        base_url=llm_config.get("base_url"),
    )
    compact_kwargs: dict = dict(
        model=llm_config.get("model", "gpt-4o"),
        messages=[
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {"role": "user",   "content": "\n".join(lines)},
        ],
        max_tokens=1024,
    )
    if "extra_body" in llm_config:
        compact_kwargs["extra_body"] = llm_config["extra_body"]
    resp = client.chat.completions.create(**compact_kwargs)
    summary = (resp.choices[0].message.content or "").strip()

    injection: list[dict] = [
        {
            "role": "user",
            "content": (
                "[Conversation history summary — treat as established context, "
                "do not ask to re-confirm]\n" + summary
            ),
        },
        {
            "role": "assistant",
            "content": "Understood. I'll continue from this context.",
        },
    ]

    return sys_msg + task_input + injection + tail, summary
