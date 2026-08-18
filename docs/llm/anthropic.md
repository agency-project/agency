# Claude backend (`llm/anthropic.py`)

> `_AnthropicBackend` covers the first-party api.anthropic.com API. See [bedrock.md](bedrock.md) for the two other Anthropic-family backends (`_AnthropicBedrockBackend`, `_AnthropicAWSBackend`) that reuse this module's adapter machinery, and [base.md](base.md) for backend selection.

## Auth and client construction

```python
def make_client(self, timeout):
    anthropic_client = _anthropic_sdk.Anthropic(
        api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"),
        timeout=timeout,
        default_headers={"anthropic-workspace-id": workspace_id} if workspace_id else {},
    )
    return _AnthropicBedrockChatClient(anthropic_client)
```

`workspace_id` (config field, or `ANTHROPIC_WORKSPACE_ID` env var) is sent as the `anthropic-workspace-id` header only when set — omitted entirely otherwise, since plain api.anthropic.com doesn't use it (only Claude Platform on AWS requires it — see [bedrock.md](bedrock.md)'s `_AnthropicAWSBackend`).

Requires the `anthropic` package (`pip install anthropic`) — raises `RuntimeError` with that instruction if it's not installed. `_anthropic_sdk` is this module's own `try/except ImportError` binding, independent of the one in `.bedrock` or `.base` — each module that touches the SDK owns its own reference, so a test can patch (or a host can lack) the package for one backend family without affecting the others.

## The Messages-API adapter

The rest of `agllm.py` (streaming, tool-call parsing, retry, compaction) is built entirely around the OpenAI `chat.completions.create(**kwargs)` interface. Anthropic's Messages API has a different shape (`system` as a top-level param, content blocks instead of a flat string, `tool_use`/`tool_result` blocks instead of `tool_calls`, SSE event types instead of delta chunks), so this module translates between the two rather than teaching the rest of the codebase two request/response shapes.

- **`_openai_messages_to_anthropic(messages)`** → `(system_text, anthropic_messages)`. Extracts and joins consecutive `system` messages; converts assistant `tool_calls` into `tool_use` content blocks; merges consecutive `tool`-role results into one `tool_result`-block-bearing user message (Anthropic's API rejects a bare `tool` role).
- **`_openai_tools_to_anthropic(tools)`** — OpenAI's `{"type": "function", "function": {...}}` tool shape → Anthropic's flat `{"name", "description", "input_schema"}`.
- **`_anthropic_stream_to_openai_chunks(stream)`** — translates Anthropic's SSE event stream (`message_start`, `content_block_start/delta/stop`, `message_delta`) into the `_FakeChunk`/`_FakeChoice`/`_FakeDelta` shapes `agllm.call()`'s streaming loop expects. Tool-call JSON is buffered per content block and emitted as **one complete chunk** on `content_block_stop`, not streamed fragment-by-fragment — `agllm.call()` never renders partial tool-call arguments to the user, so nothing is lost, and one complete chunk avoids relying on every fragment individually surviving whatever consumes this generator. If the stream ends (e.g. `stop_reason="max_tokens"`) while a `tool_use` block is still open, `content_block_stop` never fires for it — previously this silently dropped the tool call entirely, producing a completely empty assistant turn that made callers reprompt forever; the partial JSON is now flushed instead, with a `WARNING` printed, so the caller sees a (possibly unparseable) attempt rather than silence.
- **`_AnthropicNonStreamResponse`** — mimics `openai.types.chat.ChatCompletion`'s `.choices[0].message.content` surface for `agllm.compact()`, which doesn't stream.
- **`_with_cache_control(content)`** / prompt caching — `_AnthropicBedrockCompletions.create()` attaches two `cache_control: {"type": "ephemeral"}` breakpoints per call: one on the system prompt (the largest, most static part of every request — tools render before system in Anthropic's prefix order, so this one breakpoint caches tools + system together), and one on the latest turn's last content block (so the *next* call in a growing agent-loop history reads everything up to, but not including, that turn from cache — the standard multi-turn caching pattern; earlier breakpoints don't need to be resent).

## `_AnthropicBedrockChatClient`

Drop-in replacement for the subset of `openai.OpenAI`'s interface `agllm.call()`/`.compact()` use (`.chat.completions.create()`, `.close()`), wrapping an already-constructed anthropic SDK client. Despite the name, it's not Bedrock-specific — it wraps whichever of `Anthropic`, `AnthropicBedrock`, or `AnthropicAWS` the caller constructed, since all three expose `.messages.create()` alike, which is all this wrapper needs. `_AnthropicBackend` here, and both Anthropic-family backends in [bedrock.md](bedrock.md), all construct one of these as their `make_client()` return value.

## Known context windows

`_known_anthropic_context_window(model)` is a static fallback lookup table (`_ANTHROPIC_CONTEXT_WINDOWS`), used when `list_models()`'s real lookup fails or a model isn't in the API's response yet. It strips the optional region + `anthropic.` prefix Bedrock model IDs carry (`_ANTHROPIC_BEDROCK_MODEL_RE`, a no-op substitution on plain api.anthropic.com IDs, which carry no such prefix) before matching — which is why this lookup, despite living in the "plain Anthropic" module, is shared as-is by [bedrock.md](bedrock.md)'s two AWS-hosted Anthropic-family backends rather than needing its own copy there.
