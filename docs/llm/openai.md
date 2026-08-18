# OpenAI backend (`llm/openai.py`)

> `_OpenAICompatibleBackend` is the default backend — see [base.md](base.md) for how `for_config()` routes here. Also reused as-is by [vllm.md](vllm.md), and subclassed by [bedrock.md](bedrock.md)'s `_OpenAICompatibleBedrockBackend`.

The simplest backend: it's a near-direct wrapper around `openai.OpenAI`.

```python
def make_client(self, timeout: httpx.Timeout) -> openai.OpenAI:
    return openai.OpenAI(
        api_key=self.api_key or "EMPTY",
        base_url=self.base_url,
        timeout=timeout,
    )
```

`api_key` defaults to `"EMPTY"` when unset — the conventional placeholder for vLLM/self-hosted endpoints with no auth configured; real OpenAI requires a real key regardless.

## `tokenize_url()`

Derives a vLLM-style `/tokenize` endpoint root from `base_url` by stripping a trailing `/v1`:

```python
def tokenize_url(self) -> "str | None":
    root = (self.base_url or "").rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root or None
```

Used by `agllm.count_messages_tokens()` for an accurate token count via the real tokenizer, falling back to a character-based estimate when unavailable (e.g. real OpenAI, which has no such endpoint) — see [../agllm.md](../agllm.md)'s "Token counting" section.

## `list_models()`

Not overridden — uses `agllm_backend`'s default implementation (`client.models.list()` via `make_client()`), since OpenAI's `/v1/models` is the canonical shape every other backend's own listing either matches directly or has to adapt to.
