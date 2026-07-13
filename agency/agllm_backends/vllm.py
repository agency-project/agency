"""vLLM backend.

vLLM speaks the OpenAI-compatible chat.completions API, so it has no backend
class of its own -- `agllm_backend.for_config()` routes provider="vllm"
straight to `._openai._OpenAICompatibleBackend` (after checking base_url is
set, since unlike real OpenAI a vLLM endpoint has no well-known default
URL). This module only adds the vLLM-specific config surface: the extra
sampling knobs (top_k, repetition_penalty, min_p, min_tokens, guided_json,
guided_regex) vLLM/sglang accept via extra_body that OpenAI's own API
rejects.
"""

from __future__ import annotations

from .base import _AgProviderBackendConfig, _OPENAI_GEN_FIELDS, _VLLM_EXTRA_GEN_FIELDS
from .openai import _OpenAICompatibleBackend

# _OpenAICompatibleBackend isn't used below -- it's re-exported so callers
# that expect every agllm_backends.<provider> module to expose its own
# backend class (matching .openai/.anthropic/.bedrock) can still find one
# here, even though vLLM has no backend class of its own (see module
# docstring above).
__all__ = ["agVLLMBackendConfig", "_OpenAICompatibleBackend"]


class agVLLMBackendConfig(_AgProviderBackendConfig):
    """agLLMBackendConfig restricted to the fields `_OpenAICompatibleBackend`
    (.openai) reads for a vLLM (or other OpenAI-compatible) endpoint -- the
    full generation surface, including vLLM/sglang sampling extensions
    (top_k, repetition_penalty, min_p, min_tokens, guided_json, guided_regex)
    sent via extra_body. `provider` is fixed to "vllm"."""

    _PROVIDER = "vllm"
    _ALLOWED_FIELDS = (
        frozenset({"model", "api_key", "base_url", "context_limit"})
        | _OPENAI_GEN_FIELDS
        | _VLLM_EXTRA_GEN_FIELDS
    )
