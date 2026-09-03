from __future__ import annotations
from typing import ClassVar, Generator
import httpx
import openai  # noqa: F401 — unused directly; tests patch agency.agllm.openai.OpenAI
from ..agconfig import agConfig, GlobalConfigParam, DynamicConfigParam, _AgConfigViewBase

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None


# Exists to register agllm's config fields (via __set_name__ at import time)
# and hold their hardcoded defaults as plain class attributes.
class _AgLLMFields:
    call_max_concurrency = GlobalConfigParam(
        "agllm", default=256
    )  # Max simultaneous in-flight LLM streaming calls across all skills.
    max_retries = DynamicConfigParam("agllm", default=12)
    idle_timeout = DynamicConfigParam(
        "agllm", default=900.0
    )  # seconds to wait for first chunk (High TTFT - server dead or overloaded?)
    stream_timeout = DynamicConfigParam(
        "agllm", default=1200.0
    )  # seconds to wait between chunks mid-stream
    retry_sleep_s = DynamicConfigParam(
        "agllm", default=2
    )  # base seconds for exponential backoff after a connection/timeout/API
    # error (see _retry_backoff_s) -- grows with attempt count and caps at
    # rate_limit_max_backoff_s, same as the 429 backoff path
    http_connect_timeout = DynamicConfigParam(
        "agllm", default=10.0
    )  # seconds for httpx to establish a TCP/TLS connection
    http_write_timeout = DynamicConfigParam(
        "agllm", default=10.0
    )  # seconds for httpx to finish writing the request body
    http_pool_timeout = DynamicConfigParam(
        "agllm", default=10.0
    )  # seconds httpx waits to acquire a connection from the pool
    live_redraw_char_threshold = DynamicConfigParam(
        "agllm", default=100
    )  # min new combined content+thinking chars before a UI redraw
    # 429 rate-limit backoff: prefer the server's Retry-After header (it knows exactly
    # when the org's per-minute window resets); exponential-with-jitter is only a
    # fallback for the rare case the header is missing. Uncapped exponential growth
    # isn't needed since 60s already covers a full per-minute rate-limit window.
    rate_limit_base_backoff_s = DynamicConfigParam("agllm", default=5.0)
    rate_limit_max_backoff_s = DynamicConfigParam("agllm", default=80.0)
    # Added on top of an honored Retry-After value, never subtracted from it — many
    # concurrently-throttled agents share the same org-wide window and so tend to
    # receive the same Retry-After, which would otherwise make them all wake up and
    # retry in the same instant.
    rate_limit_retry_after_jitter_s = DynamicConfigParam("agllm", default=5.0)
    default_context_limit = DynamicConfigParam(
        "agllm", default=200_000
    )  # Fallback context window size when model reports none.


class agLLMConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting agllm tunables in one call::

        cfg = agConfig(agLLMConfig(max_retries=5, idle_timeout=120))

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics.
    """

    _OWNER = "agllm"


# ---------------------------------------------------------------------------
# Class-based LLM config -- every per-call LLM request parameter (model,
# api_key, temperature, ...) is a DynamicConfigParam, the same descriptor
# machinery every other framework class uses for its tunables (see
# _AgLLMFields above). `agllm` inherits this class, so every concrete backend
# (._openai._OpenAICompatibleBackend, ._anthropic._AnthropicBackend, ...)
# reads its parameters as plain attributes (self.model, self.api_key, ...).
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
    # the flush-on-truncation handling in _AnthropicBackend._format_stream_to_agency)
    # instead of erroring. compact() always passes its own explicit max_tokens and
    # never hits this fallback.
    default_max_tokens = GlobalConfigParam("agllm_backend", default=128000)

    model = DynamicConfigParam("agllm_backend", default="")
    api_key = DynamicConfigParam("agllm_backend", default=None, sensitive=True)
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
    aws_access_key = DynamicConfigParam("agllm_backend", default=None, sensitive=True)
    aws_secret_key = DynamicConfigParam("agllm_backend", default=None, sensitive=True)
    aws_session_token = DynamicConfigParam("agllm_backend", default=None, sensitive=True)
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
# top-level chat.completions.create() kwargs (see agllm.build_kwargs's
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


# ---------------------------------------------------------------------------
# agllm class -- one instance per agconfig, built via agllm.for_config().
# Combines what used to be two classes: the per-provider backend (config
# holding, client building, capability hooks) and a thin outer wrapper
# around it. Inherits both AgLLMBackendFields (per-call backend params --
# model, api_key, temperature, ...) and _AgLLMFields (process-wide policy --
# retry/timeout/concurrency) so every concrete backend reads both as plain
# attributes.
# ---------------------------------------------------------------------------


class agllm(AgLLMBackendFields, _AgLLMFields):
    """One instance per agconfig — knows how to build a client, answer
    capability questions (model listing, tokenize endpoint, context limit),
    and build request kwargs. Use `agllm.for_config(agconfig)` to get the
    right concrete subclass; don't instantiate a subclass directly.

    The given agConfig is cloned (self._agconfig) -- so this instance's own
    config is independent of the caller's; mutating the caller's original
    agConfig afterward does not affect it. To change its live config, call
    ``change_config()`` (or, for one-off dynamic fields, mutate
    ``instance._agconfig`` directly since that object is used fresh on every
    call).
    """

    def __init__(self, agconfig: "agConfig") -> None:
        self._agconfig = agconfig.clone()

    def change_config(self, agconfig: "agConfig") -> None:
        """Replace this instance's agconfig with a clone of the given one."""
        self._agconfig = agconfig.clone()

    def get_config_copy(self) -> "agConfig":
        """Return a clone of this instance's agconfig."""
        return self._agconfig.clone()

    @staticmethod
    def for_config(agconfig: "agConfig") -> "agllm":
        from .bedrock import (
            _is_anthropic_bedrock_model,
            _AnthropicBedrockBackend,
            _OpenAICompatibleBedrockBackend,
            _AnthropicAWSBackend,
        )
        from .anthropic import _AnthropicBackend
        from .openai import _OpenAICompatibleBackend

        provider = agconfig.get("agllm_backend", "provider")
        model = agconfig.get("agllm_backend", "model", "") or ""
        if provider == "bedrock":
            if _is_anthropic_bedrock_model(model):
                return _AnthropicBedrockBackend(agconfig)
            return _OpenAICompatibleBedrockBackend(agconfig)
        if provider in ("anthropicAWS", "anthropic_aws"):
            return _AnthropicAWSBackend(agconfig)
        if provider == "anthropic":
            return _AnthropicBackend(agconfig)
        if provider == "vllm" and not agconfig.get("agllm_backend", "base_url"):
            raise ValueError(
                "agVLLMBackendConfig (provider='vllm') requires base_url "
                "-- point it at your vLLM/OpenAI-compatible endpoint (e.g. "
                "'http://localhost:8000/v1')."
            )
        return _OpenAICompatibleBackend(agconfig)

    def make_client(self, timeout: httpx.Timeout):
        """Build and return a client exposing `.chat.completions.create()` and `.close()`."""
        raise NotImplementedError

    def list_models(self) -> list:
        """Best-effort model listing, used for context-limit lookups. Exceptions
        propagate to the caller (fetch_context_limit already wraps this).
        Override to return [] for backends with no listing capability."""
        client = self.make_client(httpx.Timeout(self.model_listing_timeout_seconds))
        return list(client.models.list())

    def tokenize_url(self) -> "str | None":
        """Root URL for a vLLM-style /tokenize endpoint, or None if unsupported."""
        return None

    def known_context_limit(self, model: str) -> "int | None":
        """Static fallback context window for models with no listing API to
        query (e.g. Bedrock's native invoke_model). None if unknown — the
        caller (fetch_context_limit) falls back to _AgLLMFields.default_context_limit.default."""
        return None

    def fetch_context_limit(self) -> int:
        """Return this instance's model's context window size. Always live
        (never cached) -- call it fresh whenever the current value matters.

        Priority:
        1. ``self.context_limit`` — explicit user override
        2. Live API model listing — vLLM's ``max_model_len`` (a model_extra
           field) or the Anthropic API's ``max_input_tokens`` (a typed field)
        3. ``self.known_context_limit()`` — static fallback (e.g. Bedrock,
           which has no model-listing API at all)
        4. ``_AgLLMFields.default_context_limit.default`` — safe fallback so compaction always runs
        """
        if self.context_limit is not None:
            return int(self.context_limit)
        model_id = self.model or ""
        try:
            all_models = self.list_models()
            candidates = [m for m in all_models if m.id == model_id] or all_models
            for info in candidates:
                extra = getattr(info, "model_extra", None) or {}
                if "max_model_len" in extra:
                    return int(extra["max_model_len"])
                max_input_tokens = getattr(info, "max_input_tokens", None)
                if max_input_tokens is not None:
                    return int(max_input_tokens)
        except Exception as _e:
            print(f"[agllm] WARNING: failed to retrieve max_model_len from API: {_e}")
        known = self.known_context_limit(model_id)
        if known is not None:
            return known
        print(
            f"[agllm] WARNING: context limit unknown, falling back to {_AgLLMFields.default_context_limit.default}"
        )
        return _AgLLMFields.default_context_limit.default

    def _client_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.http_connect_timeout,
            read=self.stream_timeout,
            write=self.http_write_timeout,
            pool=self.http_pool_timeout,
        )

    def dispatch(self, request: dict) -> dict:
        backend_request = self._format_context_agency_to_backend(request)
        raw_result = self._call_backend(backend_request)
        return self._format_context_backend_to_agency(raw_result)

    def _format_context_agency_to_backend(self, request: dict) -> dict:
        raise NotImplementedError

    def _call_backend(self, backend_request: dict):
        raise NotImplementedError

    def _format_context_backend_to_agency(self, raw_result) -> dict:
        raise NotImplementedError

    def dispatch_stream(self, request: dict, on_client=None) -> "Generator[dict, None, None]":
        backend_request = self._format_context_agency_to_backend(request)
        raw_stream, client = self._call_backend_stream(backend_request, on_client)
        try:
            yield from self._format_stream_to_agency(raw_stream)
        finally:
            close = getattr(client, "close", None)
            if close:
                close()

    def _call_backend_stream(self, backend_request: dict, on_client=None):
        raise NotImplementedError

    def _format_stream_to_agency(self, raw_stream) -> "Generator[dict, None, None]":
        raise NotImplementedError

    def build_kwargs(self, messages: list[dict], openai_tools: "list | None" = None) -> dict:
        _OPENAI_GEN_PARAMS = {
            "temperature",
            "reasoning_effort",
            "max_completion_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "n",
            "stop",
            "logprobs",
            "seed",
        }
        _EXTRA_BODY_GEN_PARAMS = {
            "top_k",
            "repetition_penalty",
            "min_p",
            "min_tokens",
            "guided_json",
            "guided_regex",
        }
        wire_messages: list[dict] = []
        for m in messages:
            role = m["role"]
            blocks = m.get("blocks") or []
            if role == "assistant":
                text = "".join(b["text"] for b in blocks if b["type"] == "text")
                tool_calls = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {"name": b["name"], "arguments": b["arguments"]},
                    }
                    for b in blocks
                    if b["type"] == "tool_use"
                ]
                wire_msg = {"role": "assistant", "content": text or None}
                if tool_calls:
                    wire_msg["tool_calls"] = tool_calls
            elif role == "tool":
                result_block = next((b for b in blocks if b["type"] == "tool_result"), None)
                wire_msg = {
                    "role": "tool",
                    "tool_call_id": result_block.get("tool_call_id", "") if result_block else "",
                    "content": result_block.get("text", "") if result_block else "",
                }
            else:
                text = "".join(b["text"] for b in blocks if b["type"] == "text")
                wire_msg = {"role": role, "content": text}
            wire_messages.append(wire_msg)
        kwargs: dict = dict(
            model=self.model or "",
            messages=wire_messages,
        )
        for _p in _OPENAI_GEN_PARAMS:
            val = getattr(self, _p)
            if val is not None:
                kwargs[_p] = val
        if self.max_tokens is not None:
            print(
                "[agllm] WARNING: llm_config['max_tokens'] is deprecated; use 'max_completion_tokens' instead."
            )
            kwargs.setdefault("max_completion_tokens", self.max_tokens)
        _extra_body: dict = dict(self.extra_body or {})
        for _p in _EXTRA_BODY_GEN_PARAMS:
            val = getattr(self, _p)
            if val is not None:
                _extra_body[_p] = val
        if _extra_body:
            kwargs["extra_body"] = _extra_body
        if openai_tools:
            kwargs["tools"] = openai_tools
        return kwargs


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
