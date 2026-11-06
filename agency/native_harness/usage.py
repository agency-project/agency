"""How big a prompt was, from the usage dict Agency's LLM backends report.

Stdlib only, so the native harness can use it as well as the host-side harness
adapters.

The backends fill `prompt_tokens` two ways:

* OpenAI and OpenAI Responses (agency/llm/openai.py, openai_responses.py) report
  the whole prompt; `cached_tokens` and `cache_write_tokens` are parts of it.
* Anthropic, and Claude or Converse on Bedrock (agency/llm/anthropic.py,
  bedrock.py), report Anthropic's `input_tokens`: only the part of the prompt
  that was neither read from nor written to the cache. Cache reads and writes
  are beside it, as `cache_read_tokens` and `cache_write_tokens`.

With prompt caching, most of a long Anthropic prompt is cache reads, so taking
`prompt_tokens` as the prompt's size can undercount it a hundredfold.
"""

from __future__ import annotations

from typing import NamedTuple


class PromptTokens(NamedTuple):
    total: int
    """The whole prompt, cached or not."""
    cache_read: int
    cache_write: int

    @property
    def uncached(self) -> int:
        return max(0, self.total - self.cache_read - self.cache_write)


def _int(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def prompt_tokens(usage: "dict | None") -> PromptTokens:
    """The whole prompt and its cached parts, whichever way the backend counted."""
    usage = usage or {}
    reported = _int(usage.get("prompt_tokens", usage.get("input_tokens")))
    if "cache_read_tokens" in usage or "cache_read_input_tokens" in usage:
        # Anthropic/Bedrock: the cache counts are beside prompt_tokens.
        read = _int(usage.get("cache_read_tokens", usage.get("cache_read_input_tokens")))
        write = _int(usage.get("cache_write_tokens", usage.get("cache_creation_input_tokens")))
        return PromptTokens(reported + read + write, read, write)
    # OpenAI: the cache counts are parts of prompt_tokens.
    read = _int(usage.get("cached_tokens"))
    write = _int(usage.get("cache_write_tokens"))
    return PromptTokens(max(reported, read + write), read, write)


__all__ = ["PromptTokens", "prompt_tokens"]
