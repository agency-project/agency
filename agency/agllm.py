from __future__ import annotations
import random
import re
import ssl
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable
import httpx
import openai  # noqa: F401 — unused directly; tests patch agency.agllm.openai.OpenAI
from .profiler import agprof
from .agutil import _iter_batched, _strip_thinking, _extract_thinking, _LLMIdleTimeout
from .agllm_backends import (
    agllm_backend,
    AgLLMBackendFields,
    BAD_REQUEST_EXCS,
    API_CONN_EXCS,
    RATE_LIMIT_EXCS,
    API_ERROR_EXCS,
)
from .agconfig import agConfig, GlobalConfigParam, DynamicConfigParam, _AgConfigViewBase

if TYPE_CHECKING:
    from .agterm import agterm
    from .aglog import aglog
    from .agcontext import agcontext


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


# Tier-1 (global class) config: lazily created on first use so a caller can
# override the limit via agllm.call_max_concurrency = N (or cfg.agllm.call_max_concurrency
# = N before any agllm exists) before the first LLM call in the process.
_llm_call_semaphore: threading.Semaphore | None = None
_llm_call_semaphore_init_lock = threading.Lock()


def _get_llm_call_semaphore() -> threading.Semaphore:
    global _llm_call_semaphore
    if _llm_call_semaphore is None:
        with _llm_call_semaphore_init_lock:
            if _llm_call_semaphore is None:
                limit = _AgLLMFields().call_max_concurrency
                _llm_call_semaphore = threading.Semaphore(limit)
    return _llm_call_semaphore


_SUMMARY_SYSTEM = """\
You are a conversation summariser. Produce a concise structured summary of \
the conversation history provided. Preserve ALL critical details: decisions, \
file paths, error messages, constraints, user preferences, and tool outputs.

Format exactly (keep every heading, even if a section is empty):

## Goal
<one sentence describing the overall task>

## Constraints & Preferences
<bullet list — coding style, output format, naming conventions, user instructions \
that must be respected going forward>

## Progress
- Done: <completed subtasks>
- In progress: <current subtask>
- Blocked: <anything stuck and why>

## Key Decisions
<bullet list of decisions made and the reasons>

## Next Steps
<ordered bullet list of what remains to be done>

## Critical Context
<facts the agent must remember: variable values, flags, invariants, API responses>

## Relevant Files
<bullet list of every file path created, read, or modified>\
"""


# ---------------------------------------------------------------------------
# Semaphore slot context manager
# ---------------------------------------------------------------------------


@contextmanager
def _llm_call_semaphore_slot():
    sem = _get_llm_call_semaphore()
    with agprof.span("llm:sync"):
        sem.acquire()
    try:
        yield
    finally:
        sem.release()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LLMCallResult:
    conn_error: "Exception | None" = None
    context_exceeded: bool = False
    content_parts: "list[str]" = field(default_factory=list)
    reasoning_parts: "list[str]" = field(default_factory=list)
    tool_calls_raw: "dict[int, dict]" = field(default_factory=dict)
    prompt_tokens: "int | None" = None
    completion_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    elapsed_ms: int = 0
    ttft_ms: "float | None" = None
    generation_ms: "float | None" = None
    output_tokens_per_second: "float | None" = None

    @property
    def ok(self) -> bool:
        return self.conn_error is None and not self.context_exceeded


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

    def _retry_backoff_s(self, exc: Exception, attempt: int) -> float:
        """Seconds to sleep before retrying `exc` at 0-indexed `attempt`.

        A rate limit honors the server's Retry-After header when present (it
        knows exactly when the org's per-minute window resets); otherwise --
        and for connection/timeout/API errors, which an overloaded backend
        raises just as often as a 429 -- this falls back to bounded
        exponential backoff with full jitter, so a sustained overload gets a
        growing wait instead of every retry hammering the backend at the same
        fixed interval.
        """
        if isinstance(exc, RATE_LIMIT_EXCS):
            _retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
            try:
                # Jitter is added on top, never subtracted -- the header is a
                # floor, not a target, so we never retry sooner than the
                # server said to.
                return float(_retry_after) + random.uniform(0, self.rate_limit_retry_after_jitter_s)
            except (TypeError, ValueError):
                pass
            _base = self.rate_limit_base_backoff_s
        else:
            _base = self.retry_sleep_s
        _backoff = min(self.rate_limit_max_backoff_s, _base * (2**attempt))
        return random.uniform(0, _backoff)

    def call(
        self,
        kwargs: dict,
        messages: list[dict],
        term: "agterm | None",
        state_fn: "Callable | None",
        live_messages_fn: "Callable | None",
        update_ui_token_count_fn: "Callable | None",
        total_input_tokens: int,
        total_output_tokens: int,
        skill_name: str,
        full_history_fn: "Callable | None" = None,
        call_tag: str = "",
    ) -> "LLMCallResult":
        """Execute a streaming LLM call, retrying on transient connection errors.

        `call_tag` is purely cosmetic -- it's stamped onto this call's log lines
        (e.g. "[compact]") so a caller that isn't the main ReAct loop (compact(),
        or any future one-off completion) is distinguishable in the log from a
        regular skill turn. Leave it blank for the default ReAct-loop call.

        Returns an LLMCallResult. Caller checks .ok and .conn_error.
        """
        backend = self.backend
        _tag = f"[{call_tag}] " if call_tag else ""

        kwargs = dict(kwargs)  # shallow copy so we don't mutate caller's dict
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        _initial_input_tokens = total_input_tokens
        _initial_output_tokens = total_output_tokens
        _llm_elapsed_ms = 0

        _PARTIAL_THINK_RE = re.compile(
            r"<think(?:ing)?>(.*?)(?:</think(?:ing)?>|$)", re.DOTALL | re.IGNORECASE
        )

        for attempt in range(self.max_retries):
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls_raw: dict[int, dict] = {}
            prompt_tokens: int | None = None
            completion_tokens = 0
            total_input_tokens = _initial_input_tokens
            total_output_tokens = _initial_output_tokens
            _retry_err: "Exception | None" = None
            _retry_sleep_s: float = self.retry_sleep_s
            _ttft_ms: "float | None" = None
            _generation_ms: "float | None" = None

            with _llm_call_semaphore_slot(), agprof.span(f"llm:attempt[{attempt}]"):
                client = self.backend.make_client(
                    httpx.Timeout(
                        connect=self.http_connect_timeout,
                        read=None,
                        write=self.http_write_timeout,
                        pool=self.http_pool_timeout,
                    ),
                )

                if term:
                    term.log(
                        "LLM ▶    ",
                        f"{_tag}model={(backend.model or '?')}  messages={len(messages)}  idle_timeout={self.idle_timeout:.0f}s  stream_timeout={self.stream_timeout:.0f}s",
                    )
                if state_fn:
                    state_fn("llm", skill=skill_name)

                _llm_t0 = time.monotonic()

                def _annotate_attempt(
                    outcome: str,
                    *,
                    error_type: "str | None" = None,
                    retrying: bool = False,
                ) -> None:
                    elapsed_ms = (time.monotonic() - _llm_t0) * 1000
                    generation_ms = (
                        max(0.0, elapsed_ms - _ttft_ms) if _ttft_ms is not None else None
                    )
                    output_tps = (
                        completion_tokens / (generation_ms / 1000)
                        if generation_ms and completion_tokens
                        else None
                    )
                    agprof.annotate(
                        outcome=outcome,
                        error_type=error_type,
                        retrying=retrying,
                        ttft_ms=round(_ttft_ms, 3) if _ttft_ms is not None else None,
                        generation_ms=round(generation_ms, 3)
                        if generation_ms is not None
                        else None,
                        input_tokens=prompt_tokens or 0,
                        output_tokens=completion_tokens,
                        output_tokens_per_second=round(output_tps, 3)
                        if output_tps is not None
                        else None,
                    )

                partial_msg: dict = {"role": "assistant", "content": ""}
                messages.append(partial_msg)
                if live_messages_fn:
                    live_messages_fn(messages[1:])
                _live_chars = 0

                try:
                    for batch in _iter_batched(
                        client.chat.completions.create(**kwargs),
                        idle_timeout=self.idle_timeout,
                        stream_timeout=self.stream_timeout,
                    ):
                        for chunk in batch:
                            if chunk.usage is not None:
                                prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                                completion_tokens = (
                                    getattr(chunk.usage, "completion_tokens", 0) or 0
                                )
                                total_input_tokens += prompt_tokens
                                total_output_tokens += completion_tokens
                                if term is not None:
                                    term._tokens = prompt_tokens
                            if not chunk.choices:
                                continue
                            delta = chunk.choices[0].delta

                            extra = getattr(delta, "model_extra", None) or {}
                            rc = getattr(delta, "reasoning_content", None)
                            if not isinstance(rc, str):
                                rc = extra.get("reasoning_content")
                            if not isinstance(rc, str):
                                rc = extra.get("reasoning")
                            if _ttft_ms is None and (
                                (isinstance(rc, str) and rc) or delta.content or delta.tool_calls
                            ):
                                _ttft_ms = (time.monotonic() - _llm_t0) * 1000
                            if isinstance(rc, str) and rc:
                                reasoning_parts.append(rc)
                                partial_msg["_thinking"] = "".join(reasoning_parts)

                            if delta.content:
                                content_parts.append(delta.content)
                                raw = "".join(content_parts)
                                m = _PARTIAL_THINK_RE.search(raw)
                                if m:
                                    partial_msg["_thinking"] = m.group(1).strip()
                                    partial_msg["content"] = _strip_thinking(raw)
                                else:
                                    partial_msg["content"] = raw

                            new_chars = len(partial_msg.get("content", "")) + len(
                                partial_msg.get("_thinking", "")
                            )
                            if (
                                live_messages_fn
                                and new_chars - _live_chars >= self.live_redraw_char_threshold
                            ):
                                live_messages_fn(messages[1:])
                                _live_chars = new_chars
                            if delta.tool_calls:
                                for tc_delta in delta.tool_calls:
                                    slot = tool_calls_raw.setdefault(
                                        tc_delta.index,
                                        {
                                            "id": "",
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""},
                                        },
                                    )
                                    if tc_delta.id:
                                        slot["id"] = tc_delta.id
                                    if tc_delta.function:
                                        if tc_delta.function.name:
                                            slot["function"]["name"] += tc_delta.function.name
                                        if tc_delta.function.arguments:
                                            slot["function"]["arguments"] += (
                                                tc_delta.function.arguments
                                            )

                except BAD_REQUEST_EXCS as _bad_req:
                    try:
                        client.close()
                    except Exception as _close_err:
                        print(
                            f"[agllm] WARNING: failed to close client during error handling: {_close_err}"
                        )
                    messages.pop()
                    _llm_elapsed_ms = int((time.monotonic() - _llm_t0) * 1000)
                    _err_str = str(_bad_req).lower()
                    if any(
                        kw in _err_str
                        for kw in (
                            "context_length_exceeded",
                            "maximum context length",
                            "context length",
                            "too long",
                            "reduce the length",
                        )
                    ):
                        if term:
                            term.log(
                                "LLM ✗    ",
                                f"{_tag}model={(backend.model or '?')}  context length exceeded — will compact and retry",
                            )
                        _annotate_attempt("failure", error_type="context_exceeded")
                        return LLMCallResult(context_exceeded=True, elapsed_ms=_llm_elapsed_ms)
                    if term:
                        term.log(
                            "LLM ✗    ",
                            f"{_tag}model={(backend.model or '?')}  bad request: {_bad_req}",
                        )
                    _annotate_attempt("failure", error_type=type(_bad_req).__name__)
                    return LLMCallResult(conn_error=_bad_req, elapsed_ms=_llm_elapsed_ms)

                except RATE_LIMIT_EXCS + (
                    (_LLMIdleTimeout, ssl.SSLError, OSError, httpx.TransportError)
                    + API_CONN_EXCS
                    + API_ERROR_EXCS
                ) as _transient_err:
                    try:
                        client.close()
                    except Exception as _close_err:
                        print(
                            f"[agllm] WARNING: failed to close client during error handling: {_close_err}"
                        )
                    messages.pop()
                    _llm_elapsed_ms = int((time.monotonic() - _llm_t0) * 1000)
                    if isinstance(_transient_err, RATE_LIMIT_EXCS):
                        _err_desc = f"rate limited: {_transient_err}"
                    elif isinstance(_transient_err, API_CONN_EXCS):
                        _err_desc = f"Connection error: LLM backend unreachable ({_transient_err.__cause__ or _transient_err})"
                    elif isinstance(_transient_err, API_ERROR_EXCS):
                        _err_desc = f"API error: {_transient_err}"
                    else:
                        _err_desc = str(_transient_err)
                    _retry_sleep_s = self._retry_backoff_s(_transient_err, attempt)
                    if attempt < self.max_retries - 1:
                        if term:
                            term.log(
                                "LLM ✗    ",
                                f"{_tag}model={(backend.model or '?')}  {_err_desc}  "
                                f"retry {attempt + 1}/{self.max_retries - 1} in {_retry_sleep_s:.1f}s",
                            )
                        _retry_err = _transient_err
                        _annotate_attempt(
                            "failure",
                            error_type=type(_transient_err).__name__,
                            retrying=True,
                        )
                    else:
                        if term:
                            term.log(
                                "LLM ✗    ",
                                f"{_tag}model={(backend.model or '?')}  {_err_desc}  all retries exhausted",
                            )
                        _annotate_attempt(
                            "failure",
                            error_type=type(_transient_err).__name__,
                        )
                        return LLMCallResult(conn_error=_transient_err, elapsed_ms=_llm_elapsed_ms)

                else:
                    messages.pop()  # remove partial placeholder
                    _llm_elapsed_ms = int((time.monotonic() - _llm_t0) * 1000)
                    _generation_ms = (
                        max(0.0, _llm_elapsed_ms - _ttft_ms) if _ttft_ms is not None else None
                    )
                    _annotate_attempt("success")

            if _retry_err is not None:
                if full_history_fn:
                    full_history_fn(
                        {"type": "llm_retry", "error": str(_retry_err), "attempt": attempt + 1}
                    )
                with agprof.span("llm:retry_backoff"):
                    time.sleep(_retry_sleep_s)
                continue
            break  # success

        if update_ui_token_count_fn is not None:
            try:
                update_ui_token_count_fn(total_input_tokens, total_output_tokens)
            except Exception as _e:
                print(f"[agllm] WARNING: update_ui_token_count_fn raised: {_e}")

        return LLMCallResult(
            content_parts=content_parts,
            reasoning_parts=reasoning_parts,
            tool_calls_raw=tool_calls_raw,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            elapsed_ms=_llm_elapsed_ms,
            ttft_ms=round(_ttft_ms, 3) if _ttft_ms is not None else None,
            generation_ms=round(_generation_ms, 3) if _generation_ms is not None else None,
            output_tokens_per_second=(
                round(completion_tokens / (_generation_ms / 1000), 3)
                if _generation_ms and completion_tokens
                else None
            ),
        )

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
        """Rough total token count for a list of messages (~4 chars per token)."""
        return sum(agllm._estimate_tokens(m) for m in messages)

    @staticmethod
    def _estimate_tokens(msg: dict) -> int:
        chars = len(msg.get("content") or "")
        for tc in msg.get("tool_calls") or []:
            chars += len(tc.get("function", {}).get("arguments", ""))
        return max(1, chars // _AgLLMFields.CHARS_PER_TOKEN)

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
        return prompt_tokens >= int(context_limit * _AgLLMFields.COMPACT_THRESHOLD)

    @staticmethod
    def _tail_start(
        conv: list[dict], context_limit: int, tail_turns: int = _AgLLMFields.tail_turns.default
    ) -> int:
        if not conv:
            return 0
        usable = int(context_limit * _AgLLMFields.COMPACT_THRESHOLD)
        tail_budget = max(
            _AgLLMFields.TAIL_MIN_TOKENS,
            min(_AgLLMFields.TAIL_MAX_TOKENS, int(usable * _AgLLMFields.TAIL_FRACTION)),
        )
        turns_kept = 0
        tokens_kept = 0
        result = len(conv)
        i = len(conv) - 1
        while i >= 0 and turns_kept < tail_turns:
            if conv[i]["role"] != "assistant":
                i -= 1
                continue
            turn_end = i + 1
            while turn_end < len(conv) and conv[turn_end]["role"] == "tool":
                turn_end += 1
            turn_tokens = sum(agllm._estimate_tokens(conv[k]) for k in range(i, turn_end))
            if tokens_kept + turn_tokens > tail_budget and turns_kept > 0:
                break
            tokens_kept += turn_tokens
            turns_kept += 1
            result = i
            i -= 1
        return result

    @staticmethod
    def _prune_tool_outputs(messages: list[dict]) -> list[dict]:
        """Trim oversized tool results; only activates when savings reach _AgLLMFields.PRUNE_MIN_FREE_TOKENS."""
        savings_chars = sum(
            len(m.get("content") or "") - _AgLLMFields.TOOL_OUTPUT_MAX_CHARS
            for m in messages
            if m["role"] == "tool"
            and len(m.get("content") or "") > _AgLLMFields.TOOL_OUTPUT_MAX_CHARS
        )
        if savings_chars // 4 < _AgLLMFields.PRUNE_MIN_FREE_TOKENS:
            return messages
        result = []
        for m in messages:
            if m["role"] == "tool":
                content = m.get("content") or ""
                if len(content) > _AgLLMFields.TOOL_OUTPUT_MAX_CHARS:
                    m = {
                        **m,
                        "content": content[: _AgLLMFields.TOOL_OUTPUT_MAX_CHARS] + "\n[truncated]",
                    }
            result.append(m)
        return result

    def compact(
        self,
        messages: list[dict],
        *,
        context_limit: "int | None" = None,
        tail_turns: "int | None" = None,
        previous_summary: "str | None" = None,
        term: "agterm | None" = None,
    ) -> "tuple[list[dict], str]":
        """Summarise old messages; return compacted list and new summary."""
        # tail_turns can't default to self.tail_turns in the signature — a default
        # expression binds once at function-definition time, so it would never
        # see a later agconfig override. Resolve it here instead.
        if tail_turns is None:
            tail_turns = self.tail_turns
        cl = (
            context_limit
            if context_limit is not None
            else (self.context_limit or self.default_context_limit)
        )
        if messages and messages[0]["role"] == "system":
            sys_msg: list[dict] = [messages[0]]
            conv = messages[1:]
        else:
            sys_msg = []
            conv = list(messages)
        ts = agllm._tail_start(conv, cl, tail_turns)
        task_input: list[dict] = conv[:1]
        head = conv[1:ts]
        tail = conv[ts:]
        if not head:
            return messages, previous_summary or ""
        head = agllm._prune_tool_outputs(head)
        lines: list[str] = []
        if previous_summary:
            lines.append(
                f"Previous summary (update it — keep true facts, remove stale ones, "
                f"add new ones):\n{previous_summary}\n\nNew conversation to integrate:"
            )
        else:
            lines.append("Conversation to summarise:")
        if task_input:
            lines.append(
                f"[task input]: {(task_input[0].get('content') or '')[: self.summary_task_input_max_chars]}"
            )
        for m in head:
            role = m.get("role", "?")
            content = (m.get("content") or "").strip()
            tool_calls = m.get("tool_calls")
            if role == "assistant" and tool_calls:
                names = ", ".join(tc["function"]["name"] for tc in tool_calls)
                lines.append(f"[assistant → tools: {names}]")
                if content:
                    lines.append(f"  {content[: self.summary_assistant_content_max_chars]}")
            elif role == "tool":
                lines.append(f"[tool result]: {content[: _AgLLMFields.TOOL_OUTPUT_MAX_CHARS]}")
            elif content:
                lines.append(f"[{role}]: {content[: self.summary_role_content_max_chars]}")
        compact_kwargs: dict = dict(
            model=self.backend.model or "",
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user", "content": "\n".join(lines)},
            ],
        )
        compact_kwargs["max_completion_tokens"] = self.summary_max_tokens
        if self.backend.extra_body:
            compact_kwargs["extra_body"] = self.backend.extra_body
        # `[]` below is scratch space call() uses to append/pop a live-streaming
        # placeholder -- it's not what's sent over the wire (that's
        # compact_kwargs["messages"] above), so an empty list is fine here.
        with agprof.span("llm:compact"):
            result = self.call(
                compact_kwargs,
                [],
                term,
                None,
                None,
                None,
                0,
                0,
                "compact",
                call_tag="compact",
            )
            agprof.annotate(
                outcome="success" if result.ok else "failure",
                error_type=(
                    type(result.conn_error).__name__
                    if result.conn_error is not None
                    else ("context_exceeded" if result.context_exceeded else None)
                ),
                ttft_ms=result.ttft_ms,
                generation_ms=result.generation_ms,
                input_tokens=result.prompt_tokens or 0,
                output_tokens=result.completion_tokens,
                output_tokens_per_second=result.output_tokens_per_second,
            )
        if not result.ok:
            raise result.conn_error or RuntimeError(
                "compact(): summarisation request itself exceeded the context limit"
            )
        summary = "".join(result.content_parts).strip()
        injection: list[dict] = [
            {
                "role": "user",
                "content": (
                    "[HARNESS SYSTEM] [Conversation history summary — treat as established context, "
                    "do not ask to re-confirm]\n" + summary
                ),
            },
            {
                "role": "assistant",
                "content": "[HARNESS SYSTEM] Understood. I'll continue from this context.",
            },
        ]
        return sys_msg + task_input + injection + tail, summary

    def maybe_compact(
        self,
        ctx: "agcontext",
        messages: list[dict],
        prompt_tokens: "int | None",
        *,
        term: "agterm | None" = None,
        log: "aglog | None" = None,
        _live_messages_fn: "Callable | None" = None,
        skill_name: str = "",
        agname: str = "",
        force: bool = False,
    ) -> "tuple[list[dict], int]":
        """Compact history if needed; mutates ctx.compaction_summary in place.

        Returns (messages, token_estimate) where token_estimate is:
          - chars/4 estimate of the messages when prompt_tokens is None
          - the API-reported count when prompt_tokens is provided
          - re-estimated on the compacted messages after compaction fires
          - 0 when context_limit is None (caller should not rely on the value)
        """
        if self.context_limit is None:
            return messages, 0
        if prompt_tokens is None:
            token_count = agllm.estimate_messages_tokens(messages)
            label = f"tokens~{token_count}/{self.context_limit}  msgs={len(messages)}  (pre-call estimate)"
        else:
            token_count = prompt_tokens
            label = f"tokens={token_count}/{self.context_limit}  msgs={len(messages)}"
        if not force and not agllm.should_compact(token_count, self.context_limit):
            return messages, token_count
        if term:
            term.log("COMPACT  ", f"skill={skill_name}  {label}")
        msgs_before = len(messages)
        messages, ctx.compaction_summary = self.compact(
            messages,
            context_limit=self.context_limit,
            previous_summary=ctx.compaction_summary,
            term=term,
        )
        if log:
            log._lifecycle(
                "compacted",
                agname=agname,
                skill=skill_name,
                prompt_tokens=token_count,
                context_limit=self.context_limit,
                msgs_before=msgs_before,
                msgs_after=len(messages),
            )
        if _live_messages_fn:
            _live_messages_fn(messages[1:])
        return messages, agllm.estimate_messages_tokens(messages)
