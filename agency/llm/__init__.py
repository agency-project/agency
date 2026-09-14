"""LLM backend abstraction for agllm.

Split by concrete backend: `.agllm` (the backend class, selection logic, and
the cross-SDK exception tuples), `.openai`, `.anthropic`, `.bedrock`,
`.mock`. Every field any backend reads lives on `agency.configs.agconfig.llmconfig`
-- there's no per-provider config class anymore; select a provider with
`agconfig(llmconfig(provider="..."))` and pass whatever fields that provider reads.
"""

from .agllm import (
    agllm,
    BAD_REQUEST_EXCS,
    API_CONN_EXCS,
    RATE_LIMIT_EXCS,
    API_ERROR_EXCS,
)

__all__ = [
    "agllm",
    "BAD_REQUEST_EXCS",
    "API_CONN_EXCS",
    "RATE_LIMIT_EXCS",
    "API_ERROR_EXCS",
]
