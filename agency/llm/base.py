"""Shared LLM backend config fields and cross-SDK exception tuples.

The actual backend class (config-holding, client-building, selection-
dispatching) is `agllm`, in the sibling `.agllm` module -- kept there rather
than here so `from agency.llm.agllm import agllm` keeps working as the one
stable import path regardless of internal reshuffling. This module holds
what's genuinely shared infrastructure instead: `AgLLMBackendFields` (every
backend's config fields, as descriptors), the per-provider `*BackendConfig`
view classes, and the cross-SDK exception-translation tuples.
"""

from __future__ import annotations
from typing import ClassVar
import openai

from ..agconfig import agConfig, GlobalConfigParam, DynamicConfigParam, _AgConfigViewBase

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None

# _OPENAI_GEN_FIELDS/_VLLM_EXTRA_GEN_FIELDS and the exception tuples below are
# never read within this module itself -- they exist for .openai/.vllm/
# .anthropic/.bedrock (the constants) and agllm.py (the exception tuples) to
# import. Declared here explicitly so static analysis recognizes them as
# intentional exports rather than dead globals.
__all__ = [
    "_OPENAI_GEN_FIELDS",
    "_VLLM_EXTRA_GEN_FIELDS",
    "BAD_REQUEST_EXCS",
    "API_CONN_EXCS",
    "RATE_LIMIT_EXCS",
    "API_ERROR_EXCS",
]


# ---------------------------------------------------------------------------
# Class-based LLM config -- every per-call LLM request parameter (model,
# api_key, temperature, ...) is a DynamicConfigParam, the same descriptor
# machinery every other framework class uses for its tunables (see agllm.py's
# _AgLLMFields). `agllm` (in the sibling `.agllm` module) inherits this
# class, so every concrete backend (._openai._OpenAICompatibleBackend,
# ._anthropic._AnthropicBackend, ...) reads its parameters as plain
# attributes (self.model, self.api_key, ...).
#
# Descriptors are registered once, at class-body-execution time, and shared
# via ordinary inheritance. Values are NOT stored privately per instance --
# `agllm.__init__` points self._agconfig at the exact agConfig the
# caller built its config on (typically via `cfg.agllm_backend.model = ...`
# before constructing anything), so a later `cfg.agllm_backend.temperature =
# 0.9` is visible on the next read, same as any other DynamicConfigParam.
# Two backends given two different (or cloned) agConfig instances still
# never see each other's values, for the usual agConfig reason: each one's
# data lives in its own `.data` dict.
#
# model_listing_timeout_seconds and default_max_tokens are different in
# kind: genuine process-wide tunables for the backend machinery itself
# (unrelated to any one call's parameters), so they stay tier-1
# (GlobalConfigParam), exactly as before.
# ---------------------------------------------------------------------------


class AgLLMBackendFields:
    """Every LLM config field used by any backend, as DynamicConfigParam
    descriptors, plus the tier-1 tunables for the backend machinery itself."""

    model_listing_timeout_seconds = GlobalConfigParam(
        "agllm_backend", default=10.0
    )  # httpx timeout for the best-effort /v1/models lookup.
    # Anthropic requires max_tokens; 4096 was too small — a call with no explicit
    # max_tokens (i.e. no "max_tokens" key in llm_config) could truncate mid-tool-call
    # on a large structured tool argument, silently dropping the call entirely (see
    # the flush-on-truncation handling in _anthropic_stream_to_openai_chunks) instead
    # of erroring. compact() always passes its own explicit max_tokens and never
    # hits this fallback.
    default_max_tokens = GlobalConfigParam("agllm_backend", default=128000)

    model = DynamicConfigParam("agllm_backend", default="")
    api_key = DynamicConfigParam("agllm_backend", default=None)
    base_url = DynamicConfigParam("agllm_backend", default=None)
    provider = DynamicConfigParam("agllm_backend", default=None)
    region = DynamicConfigParam("agllm_backend", default=None)
    context_limit = DynamicConfigParam("agllm_backend", default=None)
    temperature = DynamicConfigParam("agllm_backend", default=None)
    reasoning_effort = DynamicConfigParam("agllm_backend", default=None)
    max_completion_tokens = DynamicConfigParam("agllm_backend", default=None)
    max_tokens = DynamicConfigParam(
        "agllm_backend", default=None
    )  # deprecated alias for max_completion_tokens
    top_p = DynamicConfigParam("agllm_backend", default=None)
    frequency_penalty = DynamicConfigParam("agllm_backend", default=None)
    presence_penalty = DynamicConfigParam("agllm_backend", default=None)
    n = DynamicConfigParam("agllm_backend", default=None)
    stop = DynamicConfigParam("agllm_backend", default=None)
    logprobs = DynamicConfigParam("agllm_backend", default=None)
    seed = DynamicConfigParam("agllm_backend", default=None)
    extra_body = DynamicConfigParam("agllm_backend", default=None)
    top_k = DynamicConfigParam("agllm_backend", default=None)
    repetition_penalty = DynamicConfigParam("agllm_backend", default=None)
    min_p = DynamicConfigParam("agllm_backend", default=None)
    min_tokens = DynamicConfigParam("agllm_backend", default=None)
    guided_json = DynamicConfigParam("agllm_backend", default=None)
    guided_regex = DynamicConfigParam("agllm_backend", default=None)
    workspace_id = DynamicConfigParam("agllm_backend", default=None)
    aws_access_key = DynamicConfigParam("agllm_backend", default=None)
    aws_secret_key = DynamicConfigParam("agllm_backend", default=None)
    aws_session_token = DynamicConfigParam("agllm_backend", default=None)
    aws_profile = DynamicConfigParam("agllm_backend", default=None)
    aws_region = DynamicConfigParam("agllm_backend", default=None)

    _FIELD_NAMES: "tuple[str, ...]" = (
        "model",
        "api_key",
        "base_url",
        "provider",
        "region",
        "context_limit",
        "temperature",
        "reasoning_effort",
        "max_completion_tokens",
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "n",
        "stop",
        "logprobs",
        "seed",
        "extra_body",
        "top_k",
        "repetition_penalty",
        "min_p",
        "min_tokens",
        "guided_json",
        "guided_regex",
        "workspace_id",
        "aws_access_key",
        "aws_secret_key",
        "aws_session_token",
        "aws_profile",
        "aws_region",
    )

    def as_dict(self) -> dict:
        """Snapshot of every currently-set field -- used for logging and
        checkpoint serialization, where a dict is more convenient than
        reading each attribute individually."""
        return {name: v for name in self._FIELD_NAMES if (v := getattr(self, name)) is not None}


class agLLMBackendConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting agllm_backend fields in one call::

        cfg = agLLMBackendConfig(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            model="meta-llama/Llama-3.1-8B-Instruct",
        ).agconfig
        ag = agent(agconfig=cfg)

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics (targets
    an existing agConfig if given one, validates field names, composes with
    other owners' views via `agConfig(view_a, view_b, ...)`).
    """

    _OWNER = "agllm_backend"


# Common per-call generation params every OpenAI-compatible server accepts as
# top-level chat.completions.create() kwargs (see agllm.build_llm_kwargs's
# _OPENAI_GEN_PARAMS). vLLM (and other OpenAI-compatible servers with
# sampling extensions) additionally accept _VLLM_EXTRA_GEN_FIELDS via
# extra_body -- real OpenAI's API does not.
_OPENAI_GEN_FIELDS = frozenset(
    {
        "temperature",
        "max_completion_tokens",
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "n",
        "stop",
        "logprobs",
        "seed",
        "extra_body",
    }
)
_VLLM_EXTRA_GEN_FIELDS = frozenset(
    {
        "top_k",
        "repetition_penalty",
        "min_p",
        "min_tokens",
        "guided_json",
        "guided_regex",
    }
)


class _AgProviderBackendConfig(agLLMBackendConfig):
    """Shared base for the per-provider *BackendConfig classes below. Each
    subclass fixes `provider` to the value that routes `for_config()` to its
    corresponding backend class, and restricts `_ALLOWED_FIELDS` to what that
    backend actually reads -- passing a field it silently ignores (e.g.
    `frequency_penalty` on `agAnthropicBackendConfig`) raises immediately
    instead of the value quietly never reaching the API call.
    """

    _PROVIDER: "ClassVar[str]"

    def __init__(self, agconfig: "agConfig | None" = None, **fields) -> None:
        super().__init__(agconfig)
        self._agconfig.set(self._OWNER, "provider", self._PROVIDER)
        if fields:
            self.update(**fields)


# Exception-translation tuples so callers (e.g. LlmHandlerServer) can catch
# both backend families without importing the anthropic package directly.
BAD_REQUEST_EXCS: tuple = (openai.BadRequestError,) + (
    (_anthropic_sdk.BadRequestError,) if _anthropic_sdk else ()
)
API_CONN_EXCS: tuple = (openai.APIConnectionError,) + (
    (_anthropic_sdk.APIConnectionError,) if _anthropic_sdk else ()
)
RATE_LIMIT_EXCS: tuple = (openai.RateLimitError,) + (
    (_anthropic_sdk.RateLimitError,) if _anthropic_sdk else ()
)
# Base/catch-all API error classes — covers mid-stream server-side error frames
# (e.g. openai._streaming raises openai.APIError directly, with no HTTP status
# to build a more specific subclass from) and any other APIStatusError subclass
# not special-cased above (e.g. InternalServerError). Callers should check the
# more specific tuples above first — BadRequestError/RateLimitError/connection
# errors are all subclasses of these and get their own handling.
API_ERROR_EXCS: tuple = (openai.APIError,) + ((_anthropic_sdk.APIError,) if _anthropic_sdk else ())
