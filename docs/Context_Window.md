# Context window

Every harness compacts its conversation itself. Agency tells each one the real
context limit of the model behind it (`agllm.fetch_context_limit`, or
`llmconfig(context_limit=...)`) and forwards token usage in the harness's own
format, so that it compacts at about 90% of that limit, as the native harness
does.

## What each harness is given

| Harness | Window | Compacts at |
|---|---|---|
| native | `context_limit` | 90%, on the backend's count of the last call plus an estimate of what came after |
| Claude Code | `--autocompact` (or `CLAUDE_CODE_AUTO_COMPACT_WINDOW` below 100K) and `CLAUDE_CODE_MAX_CONTEXT_TOKENS`; `[1m]` on Claude model names above 200K | 90%: the window passed is 90% plus the 33K Claude Code keeps free |
| Codex | `model_context_window`, and above 272K a one-model `model_catalog_json` | 90% |
| Grok | `context_window` | 85% (Grok's default) |
| OpenCode | `limit.context` and `limit.output` | 90%: `context - min(output, 32K)` |
| Kimi | `max_context_size`, `[loop_control] reserved_context_size` and `compaction_trigger_ratio` | 85%, which it reaches about a step late: ~90% |

The rules are those of the versions `tests/host/fixtures/Dockerfile.harnesses`
pins (Claude Code 2.1.296, Codex 0.147.0, Grok 1.0.0, OpenCode 1.18.15, Kimi
Code 0.42.0); `claude_code.py`, `codex.py`, `opencode.py` and `kimi.py` note
where each comes from.

## Token usage

Backends fill `prompt_tokens` two ways. OpenAI's is the whole prompt, with
`cached_tokens` a part of it; Anthropic's and Bedrock's leaves cache reads and
writes out (`cache_read_tokens`, `cache_write_tokens`). Adapters take the whole
prompt from `agency/native_harness/usage.py`'s `prompt_tokens()`, never from
`prompt_tokens` directly: with prompt caching, a long Claude conversation is
mostly cache reads, and a harness told only the uncached part never compacts.

A model listing (LiteLLM's, for instance) may give a model's whole window where
Agency knows its input cap; `fetch_context_limit` takes the smaller.

## A request that is too big anyway

A harness compacts before a call on the previous reply's count, so one large
tool output can carry the next request past the limit, and OpenCode's second
summary request re-sends the whole session. `LlmHandlerServer` passes every
request through `agency/llm/context_fit.py` first: over the limit, the oldest
tool results lose their text (`[Cleared by Agency: ...]`, the call and its
result stay paired), as Anthropic's `clear_tool_uses` does, until it fits; the
newest result, if too big alone, keeps its head and tail. Sizes come from the
backend's own count of the conversation's characters. A `context_fit` event
records each one. `llmconfig(fit_to_context_limit=False)` turns it off.

## Limits

* Codex caps a model it does not know at 272K. Above that the adapter writes
  `model_catalog.json`: Codex's own fallback entry for an unknown model, its
  base instructions (`codex_prompt.md`, Codex 0.147.0's `prompt.md`) and the
  real window. It replaces Codex's catalog, so `spawn_agent` no longer lists
  OpenAI models to switch to; Agency serves every request with its own model
  regardless. Moving the Codex pin means refreshing `codex_prompt.md`; the
  live test below compares Codex's requests with and without the catalog.
* Claude Code takes `--autocompact` up to 1M, so it compacts by 967K at the
  latest, and a Claude model name gets no window below 100K.
* If the backend rejects a request for its size anyway (the estimate missed,
  or the configured limit is above the model's real one), `LlmHandlerServer`
  cuts it to three quarters and sends it again, up to three times, so the
  harness gets an answer: Codex 0.147.0 and Grok 1.0.0 would end the turn on
  that error. Past that, Agency words the error so harnesses take it for an
  overflow (`streaming.context_overflow_message`: "prompt is too long" and
  code `context_length_exceeded`), which starts Claude Code's reactive
  compaction, and Kimi's and OpenCode's by their source.

## Tests

```sh
pytest tests/harness/test_context_window.py       # every backend, harness and known model
pytest tests/harness/test_context_window_live.py  # the installed claude, codex and grok binaries
```

The first fails when a backend, a harness or a model is added without saying
how it counts tokens or where it compacts. The second runs the installed
Claude Code, Codex and Grok against a local mock API and reads where they
compact; CI installs the latest Claude Code and Grok and the pinned Codex, so
a release that changes the rule fails it.
After moving a harness pin, re-check its `_threshold` function in the first
file against the new release, and rerun `benchmarks/context_window/run.py`
(real models, a Linux host) for each harness.
