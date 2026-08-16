"""Context compaction for the standalone native harness.

Same algorithm `agllm.py`'s own `maybe_compact()`/`compact()` use
(`agllm_pure.py`, loaded by path -- see `pure_loader.py`'s docstring),
dispatched through this harness's own `LLMClient` for the summarization
call, since there is no `agllm.py`/`agllm.call()` available here (`agllm.py`
imports `openai`/`anthropic`/`boto3` at module level and pulls in
`agent.py`/`agconfig.py` -- exactly the host-side weight this package
exists to avoid).

Context-limit lookup, not a hardcoded table: when bridged, asks
`agmanager_harness`'s `/internal/context_limit` (`bridge_client.py`), which
reuses `agllm.fetch_context_limit`'s real model-listing/known-limit lookup
host-side. When standalone (no bridge), there's no such lookup available
and no hardcoded guess is substituted for it either -- compaction simply
never triggers, exactly the same "graceful when unknown" behavior
`agllm.py`'s own `maybe_compact()` already has when `context_limit is
None`. A wrong guessed limit would be worse than no compaction at all: it
could compact away context that didn't need it, or (guessed too high)
never trigger before a real overflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .pure_loader import load_agllm_pure

if TYPE_CHECKING:
    from .llm_client import LLMClient

_agllm_pure = load_agllm_pure()


def maybe_compact(
    messages: list,
    context_limit: "int | None",
    llm: "LLMClient",
    model: str,
    previous_summary: "str | None",
) -> "tuple[list, str | None]":
    """Compact `messages` if they're near `context_limit`. Returns
    `(messages, previous_summary)` -- both unchanged if compaction doesn't
    trigger or the summarization call itself fails (best-effort: a failed
    summary attempt must not crash the whole run; the next turn's own
    dispatch will surface a real error if the context truly is too large,
    just one step later than agllm.py's own compact()-raises behavior)."""
    if context_limit is None:
        return messages, previous_summary
    token_count = _agllm_pure.estimate_messages_tokens(messages)
    if not _agllm_pure.should_compact(token_count, context_limit):
        return messages, previous_summary
    sys_msg, task_input, head, tail = _agllm_pure.split_for_compaction(messages, context_limit)
    if not head:
        return messages, previous_summary
    head = _agllm_pure.prune_tool_outputs(head)
    summary_messages = _agllm_pure.build_summary_prompt_messages(task_input, head, previous_summary)
    resp = llm.dispatch(model, summary_messages)
    if "error" in resp:
        return messages, previous_summary
    summary = (resp["message"].get("content") or "").strip()
    return _agllm_pure.assemble_compacted_messages(sys_msg, task_input, summary, tail), summary


__all__ = ["maybe_compact"]
