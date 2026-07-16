"""OpenAI (and, via `.vllm`, any other OpenAI-compatible endpoint) backend."""

from __future__ import annotations
import httpx
import openai

from .base import _AgProviderBackendConfig, _OPENAI_GEN_FIELDS, agllm_backend


class agOpenAIBackendConfig(_AgProviderBackendConfig):
    """agLLMBackendConfig restricted to the fields `_OpenAICompatibleBackend`
    reads for the real OpenAI API. Excludes the vLLM/sglang-only sampling
    extensions `agVLLMBackendConfig` allows (top_k, repetition_penalty,
    min_p, min_tokens, guided_json, guided_regex) -- OpenAI's API rejects
    those in extra_body. `provider` is fixed to "openai"."""

    _PROVIDER = "openai"
    _ALLOWED_FIELDS = (
        frozenset({"model", "api_key", "base_url", "context_limit"}) | _OPENAI_GEN_FIELDS
    )


class _OpenAICompatibleBackend(agllm_backend):
    """Default backend: OpenAI, vLLM, or any other OpenAI-compatible endpoint."""

    def make_client(self, timeout: httpx.Timeout, max_retries: "int | None" = None) -> openai.OpenAI:
        kwargs: dict = dict(
            api_key=self.api_key or "EMPTY",
            base_url=self.base_url,
            timeout=timeout,
        )
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        return openai.OpenAI(**kwargs)

    def tokenize_url(self) -> "str | None":
        base_url: str = self.base_url or ""
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        return root or None
