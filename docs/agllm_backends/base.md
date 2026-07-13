# LLM backend selection (`agllm_backends/base.py`)

> This doc covers backend *selection*, the shared config machinery, and the abstract base class. For a specific backend's own mechanics see [openai.md](openai.md), [vllm.md](vllm.md), [anthropic.md](anthropic.md), or [bedrock.md](bedrock.md). For the `agllm` facade that sits in front of all of them, see [../agllm.md](../agllm.md).

An `agllm` instance builds exactly one `agllm_backend` from its config (via `agllm_backend.for_config()`) and reuses it for every client it needs — the streaming call in `agllm.call()`, the summarisation call in `agllm.compact()`, and the model-listing lookup in `agllm.fetch_context_limit()`.

Four concrete backends exist today, each a real subclass in its own module:

- **`_OpenAICompatibleBackend`** ([openai.md](openai.md)) — OpenAI, or any other OpenAI-compatible endpoint.
- vLLM ([vllm.md](vllm.md)) has no backend class of its own — it reuses `_OpenAICompatibleBackend` directly, since it speaks the same API.
- **`_AnthropicBackend`** ([anthropic.md](anthropic.md)) — Claude via the first-party api.anthropic.com API.
- **`_OpenAICompatibleBedrockBackend`**, **`_AnthropicBedrockBackend`**, **`_AnthropicAWSBackend`** ([bedrock.md](bedrock.md)) — the three AWS-hosted flavors: Bedrock's OpenAI-compatible Mantle gateway, Bedrock's native Anthropic Messages API, and Claude Platform on AWS.

## `for_config()` dispatch

```python
provider = agconfig.get("agllm_backend", "provider")
model = agconfig.get("agllm_backend", "model", "") or ""
if provider == "bedrock":
    ...  # _AnthropicBedrockBackend or _OpenAICompatibleBedrockBackend, by model ID
if provider in ("anthropicAWS", "anthropic_aws"):
    ...  # _AnthropicAWSBackend
if provider == "anthropic":
    ...  # _AnthropicBackend
if provider == "vllm" and not base_url:
    raise ValueError(...)  # vLLM has no well-known default endpoint
...  # falls through to _OpenAICompatibleBackend (the default: openai, vllm, or unset provider)
```

Every branch except the fallthrough imports its target class via a **lazy, function-local import** (`from .bedrock import ...`, `from .anthropic import _AnthropicBackend`, `from .openai import _OpenAICompatibleBackend`) rather than a module-level one — `.openai`/`.anthropic`/`.bedrock` all import `agllm_backend` *from* `base.py` to subclass it, so a module-level import the other way would be circular.

## Config field system

Every LLM parameter (`model`, `api_key`, `temperature`, `top_k`, `workspace_id`, `aws_access_key`, ...) is declared once on `AgLLMBackendFields` as a `DynamicConfigParam` — the same descriptor machinery every other framework class uses for its tunables (see `agllm.py`'s `_AgLLMFields`). `agllm_backend` inherits this class, so every concrete backend reads its parameters as plain attributes (`self.model`, `self.api_key`, ...) regardless of which one it is.

`model_listing_timeout_seconds` and `default_max_tokens` are different in kind — genuine process-wide tunables for the backend machinery itself, unrelated to any one call's parameters — so they stay tier-1 (`GlobalConfigParam`).

### Per-provider config views

`agLLMBackendConfig` is the generic view (any field, no restriction). Four provider-specific subclasses restrict `_ALLOWED_FIELDS` to what that backend actually reads, and fix `provider` so `for_config()` routes correctly without the caller setting it separately:

| Class | Module | `provider` | Notable restriction |
|---|---|---|---|
| `agVLLMBackendConfig` | `.vllm` | `"vllm"` | Full generation surface, including vLLM/sglang extensions (`top_k`, `repetition_penalty`, `min_p`, `min_tokens`, `guided_json`, `guided_regex`) |
| `agOpenAIBackendConfig` | `.openai` | `"openai"` | Excludes the vLLM/sglang-only extensions — real OpenAI's API rejects those in `extra_body` |
| `agAnthropicBackendConfig` | `.anthropic` | `"anthropic"` | Only `temperature`, `top_p`, `max_tokens`/`max_completion_tokens`, `extra_body["top_k"]` — everything else is silently dropped by the shared Anthropic adapter (see [anthropic.md](anthropic.md)), so it's excluded here rather than accepted and ignored |
| `agBedrockBackendConfig` | `.bedrock` | `"bedrock"` | Same generation surface as vLLM's for non-Anthropic Bedrock models; narrows to the Anthropic subset (like `agAnthropicBackendConfig`) when the model routes to `_AnthropicBedrockBackend` |

Passing a field a backend silently ignores raises `TypeError` immediately (`agConfigViewBase.update()`'s allowed-fields check) instead of the value quietly never reaching the API call.

## Exception-translation tuples

`BAD_REQUEST_EXCS`, `API_CONN_EXCS`, `RATE_LIMIT_EXCS`, `API_ERROR_EXCS` are tuples combining the `openai` package's exception classes with the `anthropic` package's equivalents (only if `anthropic` is installed — this module does its own `try/except ImportError` around `import anthropic as _anthropic_sdk`, matching the pattern `.anthropic`/`.bedrock` each use independently for their own SDK access). `agllm.call()` catches these without needing to import either SDK-specific exception type directly, regardless of which backend is in play.
