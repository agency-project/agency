"""Shared helper for tests that record an LLM exchange through the real
``agDataLogger.record_llm_exchange``."""

from __future__ import annotations

from agency.llm.message_store import MessageStore


def record_exchange(
    logger,
    call_label: str,
    *,
    exchange_type: str = "llm_block",
    prompt: "list[dict]" = (),
    response: "list[dict]" = (),
    **kwargs,
) -> "tuple[MessageStore, list[str], list[str]]":
    """Record one exchange whose prompt and response are the given agency message
    lists (``{"role": ..., "blocks": [...]}``), as an unpatched request would be.
    Returns the ``MessageStore`` that was passed to the logger and the message
    hashes of the prompt and of the response."""
    store = MessageStore()
    prompt_hashes = store.add_messages(list(prompt))
    response_hashes = store.add_messages(list(response))
    logger.record_llm_exchange(
        call_label,
        exchange_type=exchange_type,
        message_store=store,
        harness_message_hashes=prompt_hashes,
        patched_message_hashes=prompt_hashes,
        response_message_hashes=response_hashes,
        **kwargs,
    )
    return store, prompt_hashes, response_hashes
