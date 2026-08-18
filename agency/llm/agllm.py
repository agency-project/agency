from __future__ import annotations
import httpx
import openai  # noqa: F401 — unused directly; tests patch agency.agllm.openai.OpenAI
from .. import agllm_pure
from ..agutil import _strip_thinking, _extract_thinking
from .base import agllm_backend, AgLLMBackendFields
from ..agconfig import agConfig, GlobalConfigParam, DynamicConfigParam, _AgConfigViewBase


# Exists to register agllm's config fields (via __set_name__ at import time)
# and hold their hardcoded defaults as plain class attributes -- agllm
# inherits from this below, so self.max_retries etc. work via the inherited
# ConfigParam descriptors exactly as if they were declared directly on agllm.
class _AgLLMFields:
    # Kept as plain (non-descriptor) class attributes because other code in
    # this file reads them directly in a @staticmethod, where there's no
    # instance/agconfig to read a ConfigParam descriptor through.
    CHARS_PER_TOKEN = 4  # Rough chars-per-token ratio for char-count token estimates.
    TOKENIZE_TIMEOUT_SECONDS = 5.0
    COMPACT_THRESHOLD = 0.9  # Fraction of context_limit that triggers compaction.
    TAIL_FRACTION = 0.25
    TAIL_MIN_TOKENS = 2_000
    TAIL_MAX_TOKENS = 8_000
    TOOL_OUTPUT_MAX_CHARS = 2_000
    PRUNE_MIN_FREE_TOKENS = 20_000

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
    summary_task_input_max_chars = DynamicConfigParam("agllm", default=800)
    summary_assistant_content_max_chars = DynamicConfigParam("agllm", default=800)
    summary_role_content_max_chars = DynamicConfigParam("agllm", default=1000)
    summary_max_tokens = DynamicConfigParam("agllm", default=20000)
    tail_turns = DynamicConfigParam("agllm", default=3)


class agLLMConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting agllm tunables in one call::

        cfg = agConfig(agLLMConfig(max_retries=5, idle_timeout=120))

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics.
    """

    _OWNER = "agllm"


# _llm_call_semaphore/_get_llm_call_semaphore/_llm_call_semaphore_slot were
# retired here along with agllm.call() itself, their only caller.


# _SUMMARY_SYSTEM lives in agllm_pure.py now (agllm_pure.SUMMARY_SYSTEM) --
# shared, unmodified, with the in-container native entrypoint's own
# compaction (see that module's docstring for why it's split out).


# LLMCallResult was retired here along with agllm.call() itself, its only
# production constructor.

# ---------------------------------------------------------------------------
# agllm class
# ---------------------------------------------------------------------------


class agllm(_AgLLMFields):
    """Encapsulates an LLM configuration and provides methods for building
    requests and executing streaming calls against that configuration."""

    def __init__(
        self,
        agconfig: "agConfig",
        context_limit: "int | None" = None,
    ) -> None:
        self._agconfig: agConfig = agconfig.clone()
        self.backend: agllm_backend = agllm_backend.for_config(self._agconfig)
        self.context_limit: int = (
            context_limit if context_limit is not None else agllm.fetch_context_limit(self.backend)
        )

    def change_config(self, agconfig: "agConfig") -> None:
        """Replace this llm's agconfig (and its backend's) with a clone of
        the given one. Mutating ``self._agconfig`` in place does not reach
        ``self.backend`` -- it holds its own independent clone -- so this is
        the supported way to push a live config change through to the next
        LLM call."""
        self._agconfig = agconfig.clone()
        self.backend.change_config(self._agconfig)

    def get_config_copy(self) -> "agConfig":
        """Return a clone of this llm's agconfig."""
        return self._agconfig.clone()

    # ------------------------------------------------------------------
    # Instance methods — delegate to static methods using self.backend
    # ------------------------------------------------------------------

    def build_kwargs(self, messages: list[dict], openai_tools: "list | None" = None) -> dict:
        return agllm.build_llm_kwargs(self.backend, messages, openai_tools)

    # _retry_backoff_s()/call() were retired here along with execute_react()
    # itself: call()'s only production caller. The terminus does its own
    # single-attempt streaming dispatch (never calls agllm.call()); native's
    # entrypoint dispatches via its own _dispatch_via_terminus with its own,
    # differently-scoped retry policy (see that function's docstring).

    # ------------------------------------------------------------------
    # Static methods — pure functions on config/data, no instance needed
    # ------------------------------------------------------------------

    @staticmethod
    def build_llm_kwargs(
        llm_config: "agConfig | AgLLMBackendFields",
        messages: list[dict],
        openai_tools: "list | None",
    ) -> dict:
        backend = (
            llm_config
            if isinstance(llm_config, AgLLMBackendFields)
            else agllm_backend.for_config(llm_config)
        )
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
            model=backend.model or "",
            messages=wire_messages,
        )
        for _p in _OPENAI_GEN_PARAMS:
            val = getattr(backend, _p)
            if val is not None:
                kwargs[_p] = val
        if backend.max_tokens is not None:
            print(
                "[agllm] WARNING: llm_config['max_tokens'] is deprecated; use 'max_completion_tokens' instead."
            )
            kwargs.setdefault("max_completion_tokens", backend.max_tokens)
        _extra_body: dict = dict(backend.extra_body or {})
        for _p in _EXTRA_BODY_GEN_PARAMS:
            val = getattr(backend, _p)
            if val is not None:
                _extra_body[_p] = val
        if _extra_body:
            kwargs["extra_body"] = _extra_body
        if openai_tools:
            kwargs["tools"] = openai_tools
        return kwargs

    @staticmethod
    def build_assistant_msg(
        content_parts: list[str],
        reasoning_parts: list[str],
        tool_calls_raw: dict[int, dict],
    ) -> dict:
        full_content = "".join(content_parts)
        full_reasoning = "".join(reasoning_parts)
        msg_dict: dict = {"role": "assistant"}
        if full_reasoning:
            msg_dict["_thinking"] = full_reasoning
            if full_content:
                msg_dict["content"] = full_content
        elif full_content:
            thinking = _extract_thinking(full_content)
            if thinking:
                msg_dict["_thinking"] = thinking
            msg_dict["content"] = _strip_thinking(full_content)
        if tool_calls_raw:
            msg_dict["tool_calls"] = [tool_calls_raw[i] for i in sorted(tool_calls_raw)]
        return msg_dict

    @staticmethod
    def fetch_context_limit(llm_config: "agConfig | agllm_backend") -> int:
        """Return the model's context window size.

        Priority:
        1. ``backend.context_limit`` — explicit user override
        2. Live API model listing — vLLM's ``max_model_len`` (a model_extra
           field) or the Anthropic API's ``max_input_tokens`` (a typed field)
        3. ``backend.known_context_limit()`` — static fallback (e.g. Bedrock,
           which has no model-listing API at all)
        4. ``_AgLLMFields.default_context_limit.default`` — safe fallback so compaction always runs
        """
        backend = (
            llm_config
            if isinstance(llm_config, AgLLMBackendFields)
            else agllm_backend.for_config(llm_config)
        )
        if backend.context_limit is not None:
            return int(backend.context_limit)
        model_id = backend.model or ""
        try:
            all_models = backend.list_models()
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
        known = backend.known_context_limit(model_id)
        if known is not None:
            return known
        print(
            f"[agllm] WARNING: context limit unknown, falling back to {_AgLLMFields.default_context_limit.default}"
        )
        return _AgLLMFields.default_context_limit.default

    # ------------------------------------------------------------------
    # Compaction — token estimation, pruning, summarisation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_messages_tokens(messages: list[dict]) -> int:
        """Rough total token count for a list of messages (~4 chars per
        token) -- delegates to agllm_pure, shared with the in-container
        native entrypoint's own compaction (see that module's docstring)."""
        return agllm_pure.estimate_messages_tokens(messages)

    @staticmethod
    def _estimate_tokens(msg: dict) -> int:
        return agllm_pure.estimate_tokens(msg)

    @staticmethod
    def count_messages_tokens(messages: list[dict], llm_config: "agConfig | agllm_backend") -> int:
        """Token count via the vLLM /tokenize endpoint, falling back to char estimate."""
        backend = (
            llm_config
            if isinstance(llm_config, AgLLMBackendFields)
            else agllm_backend.for_config(llm_config)
        )
        root = backend.tokenize_url()
        if root:
            try:
                resp = httpx.post(
                    f"{root}/tokenize",
                    json={
                        "model": backend.model or "",
                        "messages": [
                            {k: v for k, v in m.items() if not k.startswith("_")} for m in messages
                        ],
                    },
                    timeout=_AgLLMFields.TOKENIZE_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                data = resp.json()
                if "count" in data:
                    return int(data["count"])
                if "tokens" in data:
                    return len(data["tokens"])
            except Exception as _e:
                print(f"[agllm] remote tokenize endpoint failed, using local estimate: {_e}")
        return agllm.estimate_messages_tokens(messages)

    @staticmethod
    def should_compact(prompt_tokens: int, context_limit: int) -> bool:
        return agllm_pure.should_compact(prompt_tokens, context_limit)

    @staticmethod
    def _tail_start(
        conv: list[dict], context_limit: int, tail_turns: int = _AgLLMFields.tail_turns.default
    ) -> int:
        return agllm_pure.tail_start(conv, context_limit, tail_turns)

    @staticmethod
    def _prune_tool_outputs(messages: list[dict]) -> list[dict]:
        """Trim oversized tool results; only activates when savings reach
        agllm_pure.PRUNE_MIN_FREE_TOKENS."""
        return agllm_pure.prune_tool_outputs(messages)

    # compact()/maybe_compact() were retired here along with execute_react()
    # itself: their only production caller. Native's own compaction
    # (_native_in_container_entrypoint.py's _maybe_compact) uses the same
    # algorithm via the still-alive, still-tested agllm_pure.py (see
    # tests/test_agllm_pure.py and this file's own estimate_tokens/prune/
    # tail_start/should_compact static methods above, all thin delegates to
    # that module).
