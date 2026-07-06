# agllm

`agllm` wraps an OpenAI-compatible LLM endpoint and provides streaming calls, message construction helpers, and conversation compaction. Use it directly when you need fine-grained control over an LLM call outside of the standard `agskill` ReAct loop; in normal skill execution, `agskill` creates and drives an `agllm` instance for you.

## Construction

```python
from agency.agllm import agllm

llm = agllm(
    config={
        "base_url":    "http://localhost:8000/v1",
        "api_key":     "EMPTY",
        "model":       "meta-llama/Llama-3.1-8B-Instruct",
        "temperature": 0.0,
        "max_completion_tokens": 4096,
    }
)
```

### Config dict keys

| Key | Type | Notes |
|---|---|---|
| `base_url` | str | OpenAI-compatible endpoint root. Omit to use the real OpenAI API. |
| `api_key` | str | Bearer token. Pass `"EMPTY"` for vLLM without auth. |
| `model` | str | Model identifier passed verbatim to the API. |
| `temperature` | float | Sampling temperature. |
| `max_completion_tokens` | int | Maximum tokens in the completion. Canonical framework key. |
| `max_output_tokens` | int | Provider-neutral alias for `max_completion_tokens`. |
| `max_tokens` | int | Deprecated alias for `max_completion_tokens`; still accepted for compatibility. |
| `top_p` | float | Nucleus sampling probability. |
| `top_k` | int | Top-k sampling (sent via `extra_body`). |
| `repetition_penalty` | float | Repetition penalty (sent via `extra_body`). |
| `extra_body` | dict | Arbitrary extra fields forwarded in the request body. Merged with per-param `extra_body` keys. |
| `context_limit` | int | Pin the model's context window size. Skips the auto-detect query at construction. |
| `provider` | str | Set to `"bedrock"` to enable Amazon Bedrock SigV4 authentication. |
| `region` | str | AWS region; used only when `provider == "bedrock"`. |

The second argument `context_limit` overrides `config["context_limit"]` and also skips the endpoint query. If neither is provided, `agllm` calls `fetch_context_limit()` once at construction time.

## Context limit detection

`agllm.fetch_context_limit(llm_config)` tries, in order:

1. `llm_config["context_limit"]` — explicit override, no network call made.
2. `GET /v1/models/{model}` — reads `max_model_len` from vLLM's model info response.
3. Falls back to `128 000` tokens and prints a warning.

## Main call

```python
result = llm.call(
    kwargs=llm.build_kwargs(messages, openai_tools),
    messages=messages,
    term=None,
    state_fn=None,
    live_messages_fn=None,
    update_ui_token_count_fn=None,
    total_input_tokens=0,
    total_output_tokens=0,
    skill_name="my_skill",
)
if not result.ok:
    # handle result.conn_error or result.context_exceeded
    ...
msg = agllm.build_assistant_msg(
    result.content_parts,
    result.reasoning_parts,
    result.tool_calls_raw,
)
messages.append(msg)
```

### `call` parameters

| Parameter | Type | Purpose |
|---|---|---|
| `kwargs` | dict | Pre-built request kwargs from `build_kwargs()`. Must not include `stream`; the method adds it. |
| `messages` | list[dict] | Full message history. A partial placeholder is appended during streaming and removed before returning. |
| `term` | agterm or None | Terminal logger for progress lines. |
| `state_fn` | callable or None | Called as `state_fn("llm", skill=skill_name)` at call start to update UI state. |
| `live_messages_fn` | callable or None | Called with `messages[1:]` during streaming for live UI updates. Fires at most once per `LIVE_REDRAW_CHAR_THRESHOLD` (100) new characters. |
| `update_ui_token_count_fn` | callable or None | Called as `fn(total_input, total_output)` after a successful call. |
| `total_input_tokens` | int | Running input token accumulator; incremented by this call's prompt tokens. |
| `total_output_tokens` | int | Running output token accumulator. |
| `skill_name` | str | Name shown in log lines and passed to `state_fn`. |
| `full_history_fn` | callable or None | Called with a dict on retry events for external logging. |

### Retry behavior

On `ssl.SSLError`, `OSError`, `httpx.TransportError`, `openai.APIConnectionError`, or the idle/stream timeout, the call sleeps `LLM_RETRY_SLEEP_S` (2 s) and retries up to `LLM_MAX_RETRIES` (10) times. A `BadRequestError` that matches context-limit error strings returns immediately with `context_exceeded=True`. Any other `BadRequestError` returns with `conn_error` set and does not retry.

### Timeouts

| Constant | Default | Meaning |
|---|---|---|
| `LLM_HTTP_CONNECT_TIMEOUT` | 10 s | TCP/TLS connection |
| `LLM_HTTP_WRITE_TIMEOUT` | 10 s | Sending the request body |
| `LLM_IDLE_TIMEOUT` | 300 s | Waiting for the first chunk |
| `LLM_STREAM_TIMEOUT` | 1800 s | Gap between consecutive chunks mid-stream |

## LLMCallResult

```python
@dataclass
class LLMCallResult:
    conn_error: Exception | None      # Set on network or API errors
    context_exceeded: bool            # True when the backend rejected the prompt as too long
    content_parts: list[str]          # Streamed text fragments; join to get full content
    reasoning_parts: list[str]        # Streamed reasoning/thinking fragments (if model emits them)
    tool_calls_raw: dict[int, dict]   # Keyed by tool-call index; each value is an OpenAI tool call dict
    prompt_tokens: int | None         # Tokens in the prompt as reported by the API
    total_input_tokens: int           # Accumulated input tokens including this call
    total_output_tokens: int          # Accumulated output tokens including this call
    elapsed_ms: int                   # Wall time for this call in milliseconds
```

`ok` is a computed property: `conn_error is None and not context_exceeded`.

## Building kwargs

```python
kwargs = agllm.build_llm_kwargs(llm_config, messages, openai_tools)
# or via instance:
kwargs = llm.build_kwargs(messages, openai_tools)
```

`build_llm_kwargs` strips private keys (those starting with `_`) from messages before sending. Generation params (`temperature`, `max_completion_tokens`, `top_p`, etc.) are copied from `llm_config` into the kwargs dict after `agconfig` normalizes aliases. Params that are not native OpenAI fields (`top_k`, `repetition_penalty`, `min_p`, `min_tokens`, `guided_json`, `guided_regex`) are placed under `extra_body` instead.

`agconfig` keeps the public dict API intact while centralizing config aliases and provider-specific wire names. OpenAI-compatible backends receive `max_completion_tokens`; Anthropic Messages API backends receive `max_tokens`.

## Building an assistant message

```python
msg = agllm.build_assistant_msg(
    result.content_parts,
    result.reasoning_parts,
    result.tool_calls_raw,
)
messages.append(msg)
```

This handles three sources of thinking/reasoning content in priority order:

1. `reasoning_parts` — explicit reasoning deltas emitted by the model in a dedicated field.
2. `<think>` / `<thinking>` tags embedded in `content_parts` — extracted and placed under `_thinking`.
3. Plain content with no thinking — stored directly as `content`.

Tool calls, if any, are sorted by index and stored under `tool_calls`.

## Round-robin config selection

When a list of configs is provided (for load balancing across multiple endpoints), use `pick_llm_config` to select one atomically:

```python
config = agllm.pick_llm_config(llm_config_list)
llm = agllm(config)
```

The counter is global and thread-safe.

## Compaction

Compaction summarises old conversation turns so the history fits within the model's context window.

### `maybe_compact`

The normal entry point. Checks whether compaction is needed and runs it if so.

```python
messages, token_count = llm.maybe_compact(
    ctx,
    messages,
    prompt_tokens=result.prompt_tokens,
    term=term,
    log=log,
    skill_name="my_skill",
    agname="my_agent",
)
```

Compaction fires when `prompt_tokens >= context_limit * 0.70`. When `prompt_tokens` is `None` a character-based estimate (`chars / 4`) is used instead. Pass `force=True` to compact unconditionally. The updated `ctx.compaction_summary` is written in place.

### `compact`

Force a single compaction pass directly.

```python
messages, summary = llm.compact(
    messages,
    context_limit=131072,
    tail_turns=2,
    previous_summary=ctx.compaction_summary,
)
```

The method keeps the system message (if present), the first user turn (task input), and the most recent `tail_turns` assistant turns. Everything in between is fed to the same LLM for summarisation. The resulting summary is injected as a synthetic user/assistant exchange. `previous_summary` is passed to the summariser so incremental summaries accumulate correctly rather than restarting from scratch.

### `_prune_tool_outputs`

Called internally during compaction. Truncates tool result messages longer than `_TOOL_OUTPUT_MAX_CHARS` (2 000 chars) to save tokens — but only activates when the total savings would exceed `_PRUNE_MIN_FREE_TOKENS` (20 000 tokens). Not normally called directly.

### Token counting

```python
# fast char-based estimate (~4 chars per token)
n = agllm.estimate_messages_tokens(messages)

# accurate count via vLLM /tokenize endpoint, falls back to estimate
n = agllm.count_messages_tokens(messages, llm_config)
```

## Amazon Bedrock

Set `provider` to `"bedrock"` and provide a `region`. Authentication is resolved in this order:

1. `api_key` starting with `"bedrock-api-key-"` — Bedrock Mantle API key.
2. No `api_key`, `aws_bedrock_token_generator` installed — automatic token generation.
3. `api_key` as `"ACCESS_KEY_ID:SECRET_ACCESS_KEY"` or `"ACCESS_KEY_ID:SECRET_ACCESS_KEY:SESSION_TOKEN"` — SigV4 with explicit credentials.
4. No `api_key` — SigV4 using boto3 ambient credentials (env vars, `~/.aws/credentials`, instance role).

```python
llm = agllm({
    "provider": "bedrock",
    "region":   "us-east-1",
    "model":    "anthropic.claude-3-5-sonnet-20241022-v2:0",
})
```

## Concurrency

A global semaphore (`LLM_CALL_MAX_CONCURRENCY = 128`) limits the number of simultaneously in-flight streaming calls. Each `call()` invocation acquires one slot for the duration of the stream and releases it on completion or error.

## Constraints and gotchas

- Messages containing keys that start with `_` (such as `_thinking`) are private and stripped before being sent to the API. They are preserved in the local history for UI rendering.
- The `messages` list passed to `call()` is mutated during streaming (a partial placeholder is appended) and then restored (placeholder removed) before the method returns. Do not read the list from another thread while a call is in progress.
- `compact()` does not retry the summarisation call. If the summarisation request fails, the exception propagates to the caller.
- `maybe_compact` returns `(messages, 0)` when `context_limit` is `None`. In that case the caller should not rely on the returned token count.
