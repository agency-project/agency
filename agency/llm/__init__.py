"""LLM backend abstraction for agllm.

Split by concrete backend: `.agllm` (the backend class, its config fields,
selection logic, and the cross-SDK exception tuples), `.openai`, `.vllm` (no
backend class of its own -- reuses `.openai`'s), `.anthropic`, `.bedrock`.
This package's own namespace re-exports the same public surface the
single-file `agllm_backend.py` module used to, so `from agency.llm import X`
(or `from agency import llm as m; m.X`) works exactly like the old `from
agency.agllm_backend import X` did.
"""

from .agllm import (
    agllm,
    AgLLMBackendFields,
    agLLMBackendConfig,
    BAD_REQUEST_EXCS,
    API_CONN_EXCS,
    RATE_LIMIT_EXCS,
    API_ERROR_EXCS,
)
from .openai import agOpenAIBackendConfig
from .vllm import agVLLMBackendConfig
from .anthropic import agAnthropicBackendConfig
from .bedrock import agBedrockBackendConfig
from .mock import agMockBackendConfig

__all__ = [
    "agllm",
    "AgLLMBackendFields",
    "agLLMBackendConfig",
    "agOpenAIBackendConfig",
    "agVLLMBackendConfig",
    "agAnthropicBackendConfig",
    "agBedrockBackendConfig",
    "agMockBackendConfig",
    "BAD_REQUEST_EXCS",
    "API_CONN_EXCS",
    "RATE_LIMIT_EXCS",
    "API_ERROR_EXCS",
]
