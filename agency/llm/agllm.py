from __future__ import annotations
from typing import Generator
import httpx
import openai  # noqa: F401 — unused directly; tests patch agency.agllm.openai.OpenAI
from ..configs.agconfig import agconfig as agconfig_cls

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None


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


# ---------------------------------------------------------------------------
# agllm class -- one instance per agconfig, built via agllm.for_config().
# The per-provider backend (config holding, client building, capability
# hooks) and a thin outer wrapper around it, combined into one class. Reads
# every field (model, api_key, temperature, retry/timeout policy, ...)
# straight off self.agconfig -- see agency/configs/agconfig.py.
# ---------------------------------------------------------------------------


class agllm:
    """One instance per agconfig — knows how to build a client, answer
    capability questions (model listing, tokenize endpoint, context limit),
    and build request kwargs. Use `agllm.for_config(agconfig)` to get the
    right concrete subclass; don't instantiate a subclass directly.

    The given agconfig is cloned (self.agconfig) -- so this instance's own
    config is independent of the caller's; mutating the caller's original
    agconfig afterward does not affect it. To change its live config, call
    ``change_config()`` (or, for one-off dynamic fields, mutate
    ``instance.agconfig`` directly since that object is used fresh on every
    call).
    """

    def __init__(self, agconfig: "agconfig_cls") -> None:
        self.change_config(agconfig)

    @property
    def model(self) -> str:
        return self.agconfig.llm.model or ""

    def change_config(self, agconfig: "agconfig_cls") -> None:
        """Replace this instance's agconfig with a clone of the given one,
        then validate it's internally consistent for its own provider."""
        self.agconfig = agconfig.clone() if agconfig is not None else agconfig_cls()
        self._validate_config()

    def _validate_config(self) -> None:
        """No fields are unconditionally required at this base level: bedrock/
        anthropic credentials fall back to environment variables/IAM, and an
        empty model is a real (if useless) request rather than a malformed
        config. Concrete backends override this (calling super() first) to
        add their own checks -- e.g. _OpenAICompatibleBackend requires
        base_url, since that backend has no viable default endpoint for any
        provider name (openai/vllm/litellm/anything else)."""

    def get_config_copy(self) -> "agconfig_cls":
        """Return a clone of this instance's agconfig."""
        return self.agconfig.clone()

    @staticmethod
    def for_config(agconfig: "agconfig_cls") -> "agllm":
        from .bedrock import (
            _is_anthropic_bedrock_model,
            _needs_bedrock_converse,
            _AnthropicBedrockBackend,
            _BedrockConverseBackend,
            _OpenAICompatibleBedrockBackend,
            _AnthropicAWSBackend,
        )
        from .anthropic import _AnthropicBackend
        from .openai import _OpenAICompatibleBackend

        provider = agconfig.llm.provider
        model = agconfig.llm.model or ""
        if provider == "mock":
            from .mock import _MockBackend

            return _MockBackend(agconfig)
        if provider == "bedrock":
            if _is_anthropic_bedrock_model(model):
                return _AnthropicBedrockBackend(agconfig)
            if _needs_bedrock_converse(model):
                return _BedrockConverseBackend(agconfig)
            return _OpenAICompatibleBedrockBackend(agconfig)
        if provider in ("anthropicAWS", "anthropic_aws"):
            return _AnthropicAWSBackend(agconfig)
        if provider == "anthropic":
            return _AnthropicBackend(agconfig)
        # _OpenAICompatibleBackend.__init__ -> change_config() validates the
        # required base_url -- no need to duplicate that check here just to
        # fail one call frame earlier.
        return _OpenAICompatibleBackend(agconfig)

    def make_client(self, timeout: httpx.Timeout):
        """Build and return a client exposing `.chat.completions.create()` and `.close()`."""
        raise NotImplementedError

    def list_models(self) -> list:
        """Best-effort model listing, used for context-limit lookups. Exceptions
        propagate to the caller (fetch_context_limit already wraps this).
        Override to return [] for backends with no listing capability."""
        client = self.make_client(httpx.Timeout(self.agconfig.llm.model_listing_timeout_seconds))
        return list(client.models.list())

    def tokenize_url(self) -> "str | None":
        """Root URL for a vLLM-style /tokenize endpoint, or None if unsupported."""
        return None

    def known_context_limit(self, model: str) -> "int | None":
        """Static fallback context window with no listing API to query.
        None if unknown -- caller falls back to agconfig.default_context_limit."""
        return None

    def fetch_context_limit(self) -> int:
        """Return this instance's model's context window size. Always live
        (never cached) -- call it fresh whenever the current value matters.

        Priority:
        1. ``self.agconfig.llm.context_limit`` — explicit user override
        2. Live API model listing — vLLM's ``max_model_len`` (a model_extra
           field) or the Anthropic API's ``max_input_tokens`` (a typed field)
        3. ``self.known_context_limit()`` — static fallback (e.g. Bedrock,
           which has no model-listing API at all)
        4. ``self.agconfig.llm.default_context_limit`` — safe fallback so compaction always runs
        """
        if self.agconfig.llm.context_limit is not None:
            return int(self.agconfig.llm.context_limit)
        model_id = self.agconfig.llm.model or ""
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
            f"[agllm] WARNING: context limit unknown, falling back to {self.agconfig.llm.default_context_limit}"
        )
        return self.agconfig.llm.default_context_limit

    def _client_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.agconfig.llm.http_connect_timeout,
            read=self.agconfig.llm.stream_timeout,
            write=self.agconfig.llm.http_write_timeout,
            pool=self.agconfig.llm.http_pool_timeout,
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
                # Chat Completions permits null assistant content only when
                # the same message carries a tool call. Responses-style
                # clients can emit metadata-only assistant turns after their
                # hosted blocks are filtered, so preserve those as an empty
                # string instead of forwarding an invalid null-only message.
                wire_msg = {
                    "role": "assistant",
                    "content": text if text or not tool_calls else None,
                }
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
            model=self.agconfig.llm.model or "",
            messages=wire_messages,
        )
        for _p in _OPENAI_GEN_PARAMS:
            val = getattr(self.agconfig.llm, _p)
            if val is not None:
                kwargs[_p] = val
        if self.agconfig.llm.max_tokens is not None:
            print(
                "[agllm] WARNING: llm_config['max_tokens'] is deprecated; use 'max_completion_tokens' instead."
            )
            kwargs.setdefault("max_completion_tokens", self.agconfig.llm.max_tokens)
        _extra_body: dict = dict(self.agconfig.llm.extra_body or {})
        for _p in _EXTRA_BODY_GEN_PARAMS:
            val = getattr(self.agconfig.llm, _p)
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
