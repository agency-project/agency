"""LLM backend abstraction for agllm.

Split by concrete backend: `.base` (the abstract base class + config +
selection logic), `.openai`, `.vllm` (no backend class of its own -- reuses
`.openai`'s), `.anthropic`, `.bedrock`. This package's own namespace
re-exports the same public surface the single-file `agllm_backend.py` module
used to, so `from agency.agllm_backends import X` (or `from agency import
agllm_backends as m; m.X`) works exactly like the old `from
agency.agllm_backend import X` did.
"""

from .base import (
    AgLLMBackendFields,
    agLLMBackendConfig,
    agllm_backend,
    BAD_REQUEST_EXCS,
    API_CONN_EXCS,
    RATE_LIMIT_EXCS,
    API_ERROR_EXCS,
)
from .openai import agOpenAIBackendConfig
from .vllm import agVLLMBackendConfig
from .anthropic import agAnthropicBackendConfig
from .bedrock import agBedrockBackendConfig

__all__ = [
    "AgLLMBackendFields",
    "agLLMBackendConfig",
    "agllm_backend",
    "agOpenAIBackendConfig",
    "agVLLMBackendConfig",
    "agAnthropicBackendConfig",
    "agBedrockBackendConfig",
    "BAD_REQUEST_EXCS",
    "API_CONN_EXCS",
    "RATE_LIMIT_EXCS",
    "API_ERROR_EXCS",
]
