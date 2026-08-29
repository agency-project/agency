from __future__ import annotations
import httpx
import openai  # noqa: F401 — unused directly; tests patch agency.agllm.openai.OpenAI
from .base import AgLLMBackendFields
from ..agconfig import agConfig, GlobalConfigParam, DynamicConfigParam, _AgConfigViewBase


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
            wire_msg = {k: v for k, v in m.items() if not k.startswith("_")}
            if wire_msg.get("content") is None:
                wire_msg["content"] = ""
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
