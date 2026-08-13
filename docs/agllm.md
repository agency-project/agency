# agllm

`agllm` wraps an OpenAI-compatible LLM endpoint and provides streaming calls, message construction helpers, and conversation compaction. Use it directly when you need fine-grained control over an LLM call outside of the standard `agskill` ReAct loop; in normal skill execution, `agskill` creates and drives an `agllm` instance for you.

## Construction

`agent` requires an `agConfig` with LLM fields set under the `agllm_backend` owner — see the [README](../README.md#quick-start) for the `agent(agconfig=...)` pattern used day to day. `agllm` itself is lower-level and always takes an `agConfig` (no plain-dict shortcut), for fine-grained control outside the standard `agskill` ReAct loop:

```python
from agency.agllm import agllm
from agency.agllm_backends import agLLMBackendConfig
from agency.agconfig import agConfig

# 1. agConfig(agLLMBackendConfig(...)) -- set every field in one call, with
#    typo-checked field names. Recommended for most call sites.
cfg = agConfig(agLLMBackendConfig(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.0,
    max_tokens=4096,
))
llm = agllm(cfg)

# 2. Plain agConfig -- for setting/changing fields one at a time.
cfg = agConfig()
cfg.agllm_backend.base_url    = "http://localhost:8000/v1"
cfg.agllm_backend.api_key     = "EMPTY"
cfg.agllm_backend.model       = "meta-llama/Llama-3.1-8B-Instruct"
cfg.agllm_backend.temperature = 0.0
cfg.agllm_backend.max_tokens  = 4096
llm = agllm(cfg)
```

`agllm` clones whatever `agConfig` it's given (see "How config values are stored internally" below) — passing the *same* `cfg` to two different `agllm(cfg)` calls produces two independent LLM configs, not two views onto one shared config.

`agLLMBackendConfig(**fields)` is a small view over an `agConfig`, scoped to the `agllm_backend` owner — `agConfig(agLLMBackendConfig(**fields))` is equivalent to `agConfig({"agllm_backend": {**fields}})` plus a field-name check (an unknown keyword raises `TypeError` immediately instead of the field silently being ignored). Every other framework class with tunable fields has the same kind of view (`agAgentConfig`, `agSandboxConfig`, ...) — see `_AgConfigViewBase` in `agconfig.py`, and [`agconfig.md`](agconfig.md) for the full implementation. The canonical form is always `agConfig(agXXXConfig(...), ...)`, whether you're setting one owner's fields or composing several:

```python
from agency.agsandbox import agSandboxConfig

cfg = agConfig(
    agLLMBackendConfig(model="...", api_key="...", base_url="..."),
    agSandboxConfig().add_mount("out", path, "/agent_output"),
)
ag = agent(agconfig=cfg)
```

### Config fields

| Field | Type | Notes |
|---|---|---|
| `base_url` | str | OpenAI-compatible endpoint root. Omit to use the real OpenAI API. |
| `api_key` | str | Bearer token. Pass `"EMPTY"` for vLLM without auth. |
| `model` | str | Model identifier passed verbatim to the API. |
| `temperature` | float | Sampling temperature. |
| `reasoning_effort` | str | OpenAI reasoning effort (for example, `"none"` when Chat Completions tools require reasoning to be disabled). |
| `max_tokens` | int | Maximum tokens in the completion. |
| `top_p` | float | Nucleus sampling probability. |
| `top_k` | int | Top-k sampling (sent via `extra_body`). |
| `repetition_penalty` | float | Repetition penalty (sent via `extra_body`). |
| `extra_body` | dict | Arbitrary extra fields forwarded in the request body. Merged with per-param `extra_body` keys. |
| `context_limit` | int | Pin the model's context window size. Skips the auto-detect query at construction. |
| `provider` | str | Set to `"bedrock"` to enable Amazon Bedrock SigV4 authentication. |
| `region` | str | AWS region; used only when `provider == "bedrock"`. |

The second argument `context_limit` overrides the `context_limit` field and also skips the endpoint query. If neither is provided, `agllm` calls `fetch_context_limit()` once at construction time.

### How config values are stored internally

`agllm_backend.for_config()` builds one concrete backend — see **[agllm_backends/base.md](agllm_backends/base.md)** for the full backend-selection logic and config-field reference, and [openai.md](agllm_backends/openai.md)/[vllm.md](agllm_backends/vllm.md)/[anthropic.md](agllm_backends/anthropic.md)/[bedrock.md](agllm_backends/bedrock.md) for how each concrete backend actually works. Every `agllm_backend` inherits `AgLLMBackendFields`, which declares each LLM parameter (`model`, `api_key`, `base_url`, `temperature`, `top_k`, `workspace_id`, `aws_access_key`, ...) as a `DynamicConfigParam` — the same descriptor machinery every other framework class uses for its tunables (see `agllm.py`'s `_AgLLMFields`).

- The backend clones the `agConfig` it's given (`self._agconfig`) rather than storing it as-is — so its config is independent of the caller's, and `ag.llm._agconfig` is independent of both `ag.agconfig` and `ag.llm.backend._agconfig` too (three separate clones). Mutating the caller's original `agConfig`, or `ag.agconfig`, or even `ag.llm._agconfig` after construction has no effect on the backend — none of them are the object `build_llm_kwargs` actually reads from. To change the backend's config live, call `ag.llm.change_config(new_cfg)`: it clones `new_cfg` into `ag.llm._agconfig` and pushes that same clone into `ag.llm.backend._agconfig`, so the next call sees it. `ag.llm.get_config_copy()` returns a clone of `ag.llm`'s current agconfig (handy as a starting point for `new_cfg`). See [Design_configuration.md](Design_configuration.md#changing-a-dynamic-field-live) for the full example.

`model_listing_timeout_seconds` and `default_max_tokens` are different in kind: genuine process-wide tunables for the backend machinery itself (unrelated to any one call's parameters), so they stay tier-1 (`GlobalConfigParam`), overridable via `cfg.agllm_backend.default_max_tokens = ...` like any other global framework tunable.

## Context limit detection

`agllm.fetch_context_limit(llm_config)` (accepts an `agConfig` or an already-built backend) tries, in order:

1. `context_limit` — explicit override, no network call made.
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
| `state_fn` | callable or None | Called as `state_fn("llm", skill=skill_name)` at call start. In practice this is `agent._set_ui_state`, a thin wrapper around `agent_state.update_state(...)` — it drives the webui display *and* is part of the pause-synchronization state (see `agent.md`'s "Pause and resume"), not just a UI callback. |
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

`build_llm_kwargs` strips private keys (those starting with `_`) from messages before sending. Generation params (`temperature`, `max_tokens`, `top_p`, etc.) are copied from `llm_config` into the kwargs dict. Params that are not native OpenAI fields (`top_k`, `repetition_penalty`, `min_p`, `min_tokens`, `guided_json`, `guided_regex`) are placed under `extra_body` instead.

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

Set `provider` to `"bedrock"` and provide a `region`. See **[agllm_backends/bedrock.md](agllm_backends/bedrock.md)** for the full authentication-resolution order, the two different Bedrock backends `for_config()` picks between depending on the model ID, and the separate "Claude Platform on AWS" backend (`provider="anthropicAWS"`).

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
