# vLLM backend (`agllm_backends/vllm.py`)

> vLLM has no backend class of its own — see [openai.md](openai.md) for the implementation it reuses, and [base.md](base.md) for backend selection.

vLLM speaks the same OpenAI-compatible `chat.completions` API as real OpenAI, so `agllm_backend.for_config()` routes `provider="vllm"` straight to `._openai._OpenAICompatibleBackend` — with one extra check first:

```python
if provider == "vllm" and not agconfig.get("agllm_backend", "base_url"):
    raise ValueError(
        "agVLLMBackendConfig (provider='vllm') requires base_url "
        "-- point it at your vLLM/OpenAI-compatible endpoint (e.g. "
        "'http://localhost:8000/v1')."
    )
```

Unlike real OpenAI (which has a well-known default endpoint when `base_url` is omitted), a self-hosted vLLM server has no default to fall back to — omitting `base_url` is almost always a mistake, so it's caught immediately with an actionable error rather than surfacing later as an opaque connection failure.

## Config

`agVLLMBackendConfig` is the only thing this module actually defines. It allows the full generation surface `agOpenAIBackendConfig` does, *plus* the vLLM/sglang-specific sampling extensions (`top_k`, `repetition_penalty`, `min_p`, `min_tokens`, `guided_json`, `guided_regex`) — sent via `extra_body`, since they're not native OpenAI `chat.completions.create()` kwargs. Real OpenAI's API rejects unknown `extra_body` keys, which is why `agOpenAIBackendConfig` ([openai.md](openai.md)) deliberately excludes them rather than sharing one config class between the two providers.
