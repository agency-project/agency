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
from .agutil import _iter_batched, _strip_thinking, _extract_thinking, _LLMIdleTimeout
from .agllm_backend import agllm_backend, BAD_REQUEST_EXCS, API_CONN_EXCS, RATE_LIMIT_EXCS, API_ERROR_EXCS
from .agconfig import agConfig, GlobalConfigParam, DynamicConfigParam

if TYPE_CHECKING:
    from .agterm import agterm
    from .aglog import aglog
    from .agcontext import agcontext

# ---------------------------------------------------------------------------
# Constants -- not agconfig-backed, so kept as plain module constants.
# ---------------------------------------------------------------------------
_DEFAULT_CONTEXT_LIMIT = 128_000  # Fallback context window size when model reports none.

TOKENIZE_TIMEOUT_SECONDS = 5.0
CHARS_PER_TOKEN          = 4

_COMPACT_THRESHOLD      = 0.70
_TAIL_FRACTION          = 0.25
_TAIL_MIN_TOKENS        = 2_000
_TAIL_MAX_TOKENS        = 8_000
_TOOL_OUTPUT_MAX_CHARS  = 2_000
_PRUNE_MIN_FREE_TOKENS  = 20_000


# Exists to register agllm's config fields (via __set_name__ at import time)
# and hold their hardcoded defaults as plain class attributes -- agllm
# inherits from this below, so self.max_retries etc. work via the inherited
# ConfigParam descriptors exactly as if they were declared directly on agllm.
class _AgLLMFields:
    LLM_CALL_MAX_CONCURRENCY   = 256    # Maximum simultaneous in-flight LLM streaming calls across all skills.
    LLM_HTTP_CONNECT_TIMEOUT   = 10.0   # Seconds for httpx to establish a TCP/TLS connection.
    LLM_HTTP_WRITE_TIMEOUT     = 10.0   # Seconds for httpx to finish writing the request body.
    LLM_HTTP_POOL_TIMEOUT      = 10.0   # Seconds httpx waits to acquire a connection from the pool.
    LIVE_REDRAW_CHAR_THRESHOLD = 100    # Minimum new combined content+thinking chars before a UI redraw.
    LLM_RETRY_SLEEP_S          = 2      # Seconds to wait after a connection/SSL error before retrying.
    LLM_MAX_RETRIES            = 10
    LLM_IDLE_TIMEOUT           = 300.0  # seconds to wait for first chunk (server dead?)
    LLM_STREAM_TIMEOUT         = 1800.0 # seconds to wait between chunks mid-stream
    # 429 rate-limit backoff: prefer the server's Retry-After header (it knows exactly
    # when the org's per-minute window resets); exponential-with-jitter is only a
    # fallback for the rare case the header is missing. Uncapped exponential growth
    # isn't needed since 60s already covers a full per-minute rate-limit window.
    LLM_RATE_LIMIT_BASE_BACKOFF_S = 5.0
    LLM_RATE_LIMIT_MAX_BACKOFF_S  = 80.0
    # Added on top of an honored Retry-After value, never subtracted from it — many
    # concurrently-throttled agents share the same org-wide window and so tend to
    # receive the same Retry-After, which would otherwise make them all wake up and
    # retry in the same instant.
    LLM_RATE_LIMIT_RETRY_AFTER_JITTER_S = 5.0
    DEFAULT_CONTEXT_LIMIT               = 128_000
    SUMMARY_TASK_INPUT_MAX_CHARS        = 400
    SUMMARY_ASSISTANT_CONTENT_MAX_CHARS = 400
    SUMMARY_ROLE_CONTENT_MAX_CHARS      = 600
    SUMMARY_MAX_TOKENS                  = 4096
    TAIL_TURNS                          = 2

    call_max_concurrency = GlobalConfigParam("agllm", default=LLM_CALL_MAX_CONCURRENCY)
    # Not read via self.llm_config anywhere -- registered here purely so
    # agent.__init__ picks up cfg.agllm.llm_config = LLM_CONFIG when no
    # llm_config= is passed explicitly, without needing a raw cfg.set() call.
    llm_config: "dict | list[dict] | None" = DynamicConfigParam("agllm", default=None)
    max_retries = DynamicConfigParam("agllm", default=LLM_MAX_RETRIES)
    idle_timeout = DynamicConfigParam("agllm", default=LLM_IDLE_TIMEOUT)
    stream_timeout = DynamicConfigParam("agllm", default=LLM_STREAM_TIMEOUT)
    retry_sleep_s = DynamicConfigParam("agllm", default=LLM_RETRY_SLEEP_S)
    http_connect_timeout = DynamicConfigParam("agllm", default=LLM_HTTP_CONNECT_TIMEOUT)
    http_write_timeout = DynamicConfigParam("agllm", default=LLM_HTTP_WRITE_TIMEOUT)
    http_pool_timeout = DynamicConfigParam("agllm", default=LLM_HTTP_POOL_TIMEOUT)
    live_redraw_char_threshold = DynamicConfigParam("agllm", default=LIVE_REDRAW_CHAR_THRESHOLD)
    rate_limit_base_backoff_s = DynamicConfigParam("agllm", default=LLM_RATE_LIMIT_BASE_BACKOFF_S)
    rate_limit_max_backoff_s = DynamicConfigParam("agllm", default=LLM_RATE_LIMIT_MAX_BACKOFF_S)
    rate_limit_retry_after_jitter_s = DynamicConfigParam("agllm", default=LLM_RATE_LIMIT_RETRY_AFTER_JITTER_S)
    default_context_limit = DynamicConfigParam("agllm", default=DEFAULT_CONTEXT_LIMIT)
    summary_task_input_max_chars = DynamicConfigParam("agllm", default=SUMMARY_TASK_INPUT_MAX_CHARS)
    summary_assistant_content_max_chars = DynamicConfigParam("agllm", default=SUMMARY_ASSISTANT_CONTENT_MAX_CHARS)
    summary_role_content_max_chars = DynamicConfigParam("agllm", default=SUMMARY_ROLE_CONTENT_MAX_CHARS)
    summary_max_tokens = DynamicConfigParam("agllm", default=SUMMARY_MAX_TOKENS)
    tail_turns = DynamicConfigParam("agllm", default=TAIL_TURNS)


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
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    elapsed_ms: int = 0

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
        config: dict,
        context_limit: "int | None" = None,
        agconfig: "agConfig | None" = None,
    ) -> None:
        self.config: dict = config
        self.backend: agllm_backend = agllm_backend.for_config(config)
        self.context_limit: int = context_limit if context_limit is not None else agllm.fetch_context_limit(config)
        self._agconfig: "agConfig | None" = agconfig

    # ------------------------------------------------------------------
    # Round-robin config selector
    # ------------------------------------------------------------------

    _llm_config_counter: int = 0
    _llm_config_lock: threading.Lock = threading.Lock()

    @staticmethod
    def pick_llm_config(llm_config: "dict | list[dict]") -> dict:
        """Return a single config dict, round-robining across a list."""
        if not isinstance(llm_config, list):
            return llm_config
        with agllm._llm_config_lock:
            idx = agllm._llm_config_counter % len(llm_config)
            agllm._llm_config_counter += 1
        return llm_config[idx]

    # ------------------------------------------------------------------
    # Instance methods — delegate to static methods using self.config
    # ------------------------------------------------------------------

    def build_kwargs(self, messages: list[dict], openai_tools: "list | None" = None) -> dict:
        return agllm.build_llm_kwargs(self.config, messages, openai_tools)

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
    ) -> "LLMCallResult":
        """Execute a streaming LLM call, retrying on transient connection errors.

        Returns an LLMCallResult. Caller checks .ok and .conn_error.
        """
        llm_config = self.config

        kwargs = dict(kwargs)  # shallow copy so we don't mutate caller's dict
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        _initial_input_tokens  = total_input_tokens
        _initial_output_tokens = total_output_tokens
        _llm_elapsed_ms        = 0

        _PARTIAL_THINK_RE = re.compile(
            r"<think(?:ing)?>(.*?)(?:</think(?:ing)?>|$)", re.DOTALL | re.IGNORECASE
        )

        for attempt in range(self.max_retries):
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls_raw: dict[int, dict] = {}
            prompt_tokens: int | None = None
            total_input_tokens  = _initial_input_tokens
            total_output_tokens = _initial_output_tokens
            _retry_err: "Exception | None" = None
            _retry_sleep_s: float = self.retry_sleep_s

            with _llm_call_semaphore_slot():
                client = self.backend.make_client(
                    httpx.Timeout(connect=self.http_connect_timeout, read=None, write=self.http_write_timeout, pool=self.http_pool_timeout),
                )

                if term:
                    term.log("LLM ▶    ", f"model={llm_config.get('model','?')}  messages={len(messages)}  idle_timeout={self.idle_timeout:.0f}s  stream_timeout={self.stream_timeout:.0f}s")
                if state_fn:
                    state_fn("llm", skill=skill_name)

                _llm_t0 = time.monotonic()

                partial_msg: dict = {"role": "assistant", "content": ""}
                messages.append(partial_msg)
                if live_messages_fn:
                    live_messages_fn(messages[1:])
                _live_chars = 0

                try:
                    for batch in _iter_batched(client.chat.completions.create(**kwargs), idle_timeout=self.idle_timeout, stream_timeout=self.stream_timeout):
                        for chunk in batch:
                            if chunk.usage is not None:
                                prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                                total_input_tokens  += prompt_tokens
                                total_output_tokens += getattr(chunk.usage, "completion_tokens", 0) or 0
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

                            new_chars = len(partial_msg.get("content", "")) + len(partial_msg.get("_thinking", ""))
                            if live_messages_fn and new_chars - _live_chars >= self.live_redraw_char_threshold:
                                live_messages_fn(messages[1:])
                                _live_chars = new_chars
                            if delta.tool_calls:
                                for tc_delta in delta.tool_calls:
                                    slot = tool_calls_raw.setdefault(tc_delta.index, {
                                        "id": "", "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    })
                                    if tc_delta.id:
                                        slot["id"] = tc_delta.id
                                    if tc_delta.function:
                                        if tc_delta.function.name:
                                            slot["function"]["name"] += tc_delta.function.name
                                        if tc_delta.function.arguments:
                                            slot["function"]["arguments"] += tc_delta.function.arguments

                except BAD_REQUEST_EXCS as _bad_req:
                    try:
                        client.close()
                    except Exception:
                        pass
                    messages.pop()
                    _llm_elapsed_ms = int((time.monotonic() - _llm_t0) * 1000)
                    _err_str = str(_bad_req).lower()
                    if any(kw in _err_str for kw in ("context_length_exceeded", "maximum context length",
                                                      "context length", "too long", "reduce the length")):
                        if term:
                            term.log("LLM ✗    ", f"model={llm_config.get('model','?')}  context length exceeded — will compact and retry")
                        return LLMCallResult(context_exceeded=True, elapsed_ms=_llm_elapsed_ms)
                    if term:
                        term.log("LLM ✗    ", f"model={llm_config.get('model','?')}  bad request: {_bad_req}")
                    return LLMCallResult(conn_error=_bad_req, elapsed_ms=_llm_elapsed_ms)

                except RATE_LIMIT_EXCS as _rate_err:
                    try:
                        client.close()
                    except Exception:
                        pass
                    messages.pop()
                    _llm_elapsed_ms = int((time.monotonic() - _llm_t0) * 1000)
                    _retry_after = getattr(getattr(_rate_err, "response", None), "headers", {}).get("retry-after")
                    try:
                        # Jitter is added on top, never subtracted — the header is a floor,
                        # not a target, so we never retry sooner than the server said to.
                        _retry_sleep_s = float(_retry_after) + random.uniform(0, self.rate_limit_retry_after_jitter_s)
                    except (TypeError, ValueError):
                        # No (or unparseable) Retry-After header — exponential backoff with
                        # full jitter so many concurrently-throttled skills don't all wake
                        # up and retry in the same instant (thundering herd).
                        _backoff = min(self.rate_limit_max_backoff_s, self.rate_limit_base_backoff_s * (2 ** attempt))
                        _retry_sleep_s = random.uniform(0, _backoff)
                    if attempt < self.max_retries - 1:
                        if term:
                            term.log("LLM ✗    ", f"model={llm_config.get('model','?')}  rate limited: {_rate_err}  "
                                                    f"retry {attempt + 1}/{self.max_retries - 1} in {_retry_sleep_s:.1f}s")
                        _retry_err = _rate_err
                    else:
                        if term:
                            term.log("LLM ✗    ", f"model={llm_config.get('model','?')}  rate limited: {_rate_err}  all retries exhausted")
                        return LLMCallResult(conn_error=_rate_err, elapsed_ms=_llm_elapsed_ms)

                except (_LLMIdleTimeout, ssl.SSLError, OSError, httpx.TransportError) + API_CONN_EXCS + API_ERROR_EXCS as _conn_err:
                    try:
                        client.close()
                    except Exception:
                        pass
                    messages.pop()
                    _llm_elapsed_ms = int((time.monotonic() - _llm_t0) * 1000)
                    if isinstance(_conn_err, API_CONN_EXCS):
                        _err_desc = f"Connection error: LLM backend unreachable ({_conn_err.__cause__ or _conn_err})"
                    elif isinstance(_conn_err, API_ERROR_EXCS):
                        _err_desc = f"API error: {_conn_err}"
                    else:
                        _err_desc = str(_conn_err)
                    if attempt < self.max_retries - 1:
                        if term:
                            term.log("LLM ✗    ", f"model={llm_config.get('model','?')}  {_err_desc}  retry {attempt + 1}/{self.max_retries - 1}")
                        _retry_err = _conn_err
                    else:
                        if term:
                            term.log("LLM ✗    ", f"model={llm_config.get('model','?')}  {_err_desc}  all retries exhausted")
                        return LLMCallResult(conn_error=_conn_err, elapsed_ms=_llm_elapsed_ms)

                else:
                    messages.pop()  # remove partial placeholder
                    _llm_elapsed_ms = int((time.monotonic() - _llm_t0) * 1000)

            if _retry_err is not None:
                if full_history_fn:
                    full_history_fn({"type": "llm_retry", "error": str(_retry_err), "attempt": attempt + 1})
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
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            elapsed_ms=_llm_elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Static methods — pure functions on config/data, no instance needed
    # ------------------------------------------------------------------

    @staticmethod
    def build_llm_kwargs(llm_config: dict, messages: list[dict], openai_tools: "list | None") -> dict:
        _OPENAI_GEN_PARAMS = {"temperature", "max_completion_tokens", "top_p", "frequency_penalty", "presence_penalty", "n", "stop", "logprobs", "seed"}
        _EXTRA_BODY_GEN_PARAMS = {"top_k", "repetition_penalty", "min_p", "min_tokens", "guided_json", "guided_regex"}
        wire_messages: list[dict] = []
        for m in messages:
            wire_msg = {k: v for k, v in m.items() if not k.startswith("_")}
            if wire_msg.get("content") is None:
                wire_msg["content"] = ""
            wire_messages.append(wire_msg)
        kwargs: dict = dict(
            model=llm_config.get("model", ""),
            messages=wire_messages,
        )
        for _p in _OPENAI_GEN_PARAMS:
            if _p in llm_config:
                kwargs[_p] = llm_config[_p]
        if "max_tokens" in llm_config:
            print("[agllm] WARNING: llm_config['max_tokens'] is deprecated; use 'max_completion_tokens' instead.")
            kwargs.setdefault("max_completion_tokens", llm_config["max_tokens"])
        _extra_body: dict = dict(llm_config.get("extra_body") or {})
        for _p in _EXTRA_BODY_GEN_PARAMS:
            if _p in llm_config:
                _extra_body[_p] = llm_config[_p]
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
    def fetch_context_limit(llm_config: dict) -> int:
        """Return the model's context window size.

        Priority:
        1. ``llm_config["context_limit"]`` — explicit user override
        2. Live API model listing — vLLM's ``max_model_len`` (a model_extra
           field) or the Anthropic API's ``max_input_tokens`` (a typed field)
        3. ``backend.known_context_limit()`` — static fallback (e.g. Bedrock,
           which has no model-listing API at all)
        4. ``_DEFAULT_CONTEXT_LIMIT`` — safe fallback so compaction always runs
        """
        if "context_limit" in llm_config:
            return int(llm_config["context_limit"])
        model_id = llm_config.get("model", "")
        backend = agllm_backend.for_config(llm_config)
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
        print(f"[agllm] WARNING: context limit unknown, falling back to {_DEFAULT_CONTEXT_LIMIT}")
        return _DEFAULT_CONTEXT_LIMIT

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
        for tc in (msg.get("tool_calls") or []):
            chars += len(tc.get("function", {}).get("arguments", ""))
        return max(1, chars // CHARS_PER_TOKEN)

    @staticmethod
    def count_messages_tokens(messages: list[dict], llm_config: dict) -> int:
        """Token count via the vLLM /tokenize endpoint, falling back to char estimate."""
        root = agllm_backend.for_config(llm_config).tokenize_url()
        if root:
            try:
                resp = httpx.post(
                    f"{root}/tokenize",
                    json={
                        "model": llm_config.get("model", ""),
                        "messages": [{k: v for k, v in m.items() if not k.startswith("_")}
                                     for m in messages],
                    },
                    timeout=TOKENIZE_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                data = resp.json()
                if "count" in data:
                    return int(data["count"])
                if "tokens" in data:
                    return len(data["tokens"])
            except Exception:
                pass
        return agllm.estimate_messages_tokens(messages)

    @staticmethod
    def should_compact(prompt_tokens: int, context_limit: int) -> bool:
        return prompt_tokens >= int(context_limit * _COMPACT_THRESHOLD)

    @staticmethod
    def _tail_start(conv: list[dict], context_limit: int,
                    tail_turns: int = _AgLLMFields.TAIL_TURNS) -> int:
        if not conv:
            return 0
        usable = int(context_limit * _COMPACT_THRESHOLD)
        tail_budget = max(_TAIL_MIN_TOKENS, min(_TAIL_MAX_TOKENS,
                                                int(usable * _TAIL_FRACTION)))
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
        """Trim oversized tool results; only activates when savings reach _PRUNE_MIN_FREE_TOKENS."""
        savings_chars = sum(
            len(m.get("content") or "") - _TOOL_OUTPUT_MAX_CHARS
            for m in messages
            if m["role"] == "tool" and len(m.get("content") or "") > _TOOL_OUTPUT_MAX_CHARS
        )
        if savings_chars // 4 < _PRUNE_MIN_FREE_TOKENS:
            return messages
        result = []
        for m in messages:
            if m["role"] == "tool":
                content = m.get("content") or ""
                if len(content) > _TOOL_OUTPUT_MAX_CHARS:
                    m = {**m, "content": content[:_TOOL_OUTPUT_MAX_CHARS] + "\n[truncated]"}
            result.append(m)
        return result

    def compact(
        self,
        messages: list[dict],
        *,
        context_limit: "int | None" = None,
        tail_turns: "int | None" = None,
        previous_summary: "str | None" = None,
    ) -> "tuple[list[dict], str]":
        """Summarise old messages; return compacted list and new summary."""
        # tail_turns can't default to TAIL_TURNS in the signature — a default
        # expression binds once at function-definition time, so it would never
        # see a later agconfig override. Resolve it here instead.
        if tail_turns is None:
            tail_turns = self.tail_turns
        cl = context_limit if context_limit is not None else (self.context_limit or self.default_context_limit)
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
            lines.append(f"[task input]: {(task_input[0].get('content') or '')[:self.summary_task_input_max_chars]}")
        for m in head:
            role = m.get("role", "?")
            content = (m.get("content") or "").strip()
            tool_calls = m.get("tool_calls")
            if role == "assistant" and tool_calls:
                names = ", ".join(tc["function"]["name"] for tc in tool_calls)
                lines.append(f"[assistant → tools: {names}]")
                if content:
                    lines.append(f"  {content[:self.summary_assistant_content_max_chars]}")
            elif role == "tool":
                lines.append(f"[tool result]: {content[:_TOOL_OUTPUT_MAX_CHARS]}")
            elif content:
                lines.append(f"[{role}]: {content[:self.summary_role_content_max_chars]}")
        client = self.backend.make_client(httpx.Timeout(120.0))
        compact_kwargs: dict = dict(
            model=self.config.get("model", ""),
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user",   "content": "\n".join(lines)},
            ],
        )
        compact_kwargs["max_completion_tokens"] = SUMMARY_MAX_TOKENS
        if "extra_body" in self.config:
            compact_kwargs["extra_body"] = self.config["extra_body"]
        resp = client.chat.completions.create(**compact_kwargs)
        summary = (resp.choices[0].message.content or "").strip()
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
