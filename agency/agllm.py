from __future__ import annotations
import re
import ssl
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable
import httpx
import openai
from .agutil import _iter_batched, _strip_thinking, _extract_thinking, _LLMIdleTimeout

if TYPE_CHECKING:
    from .agterm import agterm
    from .aglog import aglog
    from .agcontext import agcontext

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_CONTEXT_LIMIT     = 128_000 # Fallback context window size when model reports none.
LLM_CALL_MAX_CONCURRENCY   = 128    # Maximum simultaneous in-flight LLM streaming calls across all skills.
LLM_HTTP_CONNECT_TIMEOUT   = 10.0   # Seconds for httpx to establish a TCP/TLS connection.
LLM_HTTP_WRITE_TIMEOUT     = 10.0   # Seconds for httpx to finish writing the request body.
LLM_HTTP_POOL_TIMEOUT      = 10.0   # Seconds httpx waits to acquire a connection from the pool.
LIVE_REDRAW_CHAR_THRESHOLD = 100    # Minimum new combined content+thinking chars before a UI redraw.
LLM_RETRY_SLEEP_S          = 2      # Seconds to wait after a connection/SSL error before retrying.
LLM_MAX_RETRIES            = 10
LLM_IDLE_TIMEOUT           = 300.0  # seconds to wait for first chunk (server dead?)
LLM_STREAM_TIMEOUT         = 1800.0 # seconds to wait between chunks mid-stream

_llm_call_semaphore = threading.Semaphore(LLM_CALL_MAX_CONCURRENCY)

# ---------------------------------------------------------------------------
# Compaction constants
# ---------------------------------------------------------------------------

TOKENIZE_TIMEOUT_SECONDS               = 5.0
CHARS_PER_TOKEN                        = 4
DEFAULT_CONTEXT_LIMIT                  = 128_000
SUMMARY_TASK_INPUT_MAX_CHARS           = 400
SUMMARY_ASSISTANT_CONTENT_MAX_CHARS    = 400
SUMMARY_ROLE_CONTENT_MAX_CHARS         = 600
SUMMARY_MAX_TOKENS                     = 1024

_COMPACT_THRESHOLD  = 0.70
TAIL_TURNS          = 2
_TAIL_FRACTION      = 0.25
_TAIL_MIN_TOKENS    = 2_000
_TAIL_MAX_TOKENS    = 8_000
_TOOL_OUTPUT_MAX_CHARS  = 2_000
_PRUNE_MIN_FREE_TOKENS  = 20_000

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
# AWS Bedrock SigV4 auth
# ---------------------------------------------------------------------------

class _BedrockSigV4Auth(httpx.Auth):
    """httpx auth handler that signs requests with AWS SigV4 for Amazon Bedrock."""

    def __init__(self, region: str, api_key: str | None = None) -> None:
        import boto3
        from botocore.credentials import Credentials
        self._region = region
        if api_key:
            parts = api_key.split(":", 2)
            if len(parts) < 2:
                raise ValueError(
                    "Bedrock api_key must be 'ACCESS_KEY_ID:SECRET_ACCESS_KEY' "
                    "or 'ACCESS_KEY_ID:SECRET_ACCESS_KEY:SESSION_TOKEN'."
                )
            self._creds = Credentials(
                access_key=parts[0],
                secret_key=parts[1],
                token=parts[2] if len(parts) == 3 else None,
            )
        else:
            creds = boto3.Session(region_name=region).get_credentials()
            if creds is None:
                raise RuntimeError(
                    "No AWS credentials found for Amazon Bedrock. "
                    "Set api_key='ACCESS_KEY_ID:SECRET_ACCESS_KEY' in the llm_config, "
                    "or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars, "
                    "or run: aws configure"
                )
            self._creds = creds

    def auth_flow(self, request: httpx.Request):
        import botocore.auth
        import botocore.awsrequest

        aws_req = botocore.awsrequest.AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content or b"",
            headers={k: v for k, v in request.headers.items()
                     if k.lower() not in ("host", "content-length")},
        )
        botocore.auth.SigV4Auth(self._creds.get_frozen_credentials(), "bedrock", self._region).add_auth(aws_req)
        for k, v in aws_req.headers.items():
            request.headers[k] = v
        yield request


# ---------------------------------------------------------------------------
# Semaphore slot context manager
# ---------------------------------------------------------------------------

@contextmanager
def _llm_call_semaphore_slot():
    _llm_call_semaphore.acquire()
    try:
        yield
    finally:
        _llm_call_semaphore.release()


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

class agllm:
    """Encapsulates an LLM configuration and provides methods for building
    requests and executing streaming calls against that configuration."""

    def __init__(self, config: dict, context_limit: "int | None" = None) -> None:
        self.config: dict = config
        self.context_limit: int = context_limit if context_limit is not None else agllm.fetch_context_limit(config)

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
        _LLM_MAX_RETRIES    = LLM_MAX_RETRIES
        _LLM_IDLE_TIMEOUT   = LLM_IDLE_TIMEOUT
        _LLM_STREAM_TIMEOUT = LLM_STREAM_TIMEOUT

        kwargs = dict(kwargs)  # shallow copy so we don't mutate caller's dict
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        _initial_input_tokens  = total_input_tokens
        _initial_output_tokens = total_output_tokens
        _llm_elapsed_ms        = 0

        _PARTIAL_THINK_RE = re.compile(
            r"<think(?:ing)?>(.*?)(?:</think(?:ing)?>|$)", re.DOTALL | re.IGNORECASE
        )

        for attempt in range(_LLM_MAX_RETRIES):
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls_raw: dict[int, dict] = {}
            prompt_tokens: int | None = None
            total_input_tokens  = _initial_input_tokens
            total_output_tokens = _initial_output_tokens
            _retry_err: "Exception | None" = None

            with _llm_call_semaphore_slot():
                client = agllm._make_client(
                    llm_config,
                    httpx.Timeout(connect=LLM_HTTP_CONNECT_TIMEOUT, read=None, write=LLM_HTTP_WRITE_TIMEOUT, pool=LLM_HTTP_POOL_TIMEOUT),
                )

                if term:
                    term.log("LLM ▶    ", f"model={llm_config.get('model','?')}  messages={len(messages)}  idle_timeout={_LLM_IDLE_TIMEOUT:.0f}s  stream_timeout={_LLM_STREAM_TIMEOUT:.0f}s")
                if state_fn:
                    state_fn("llm", skill=skill_name)

                _llm_t0 = time.monotonic()

                partial_msg: dict = {"role": "assistant", "content": ""}
                messages.append(partial_msg)
                if live_messages_fn:
                    live_messages_fn(messages[1:])
                _live_chars = 0

                try:
                    for batch in _iter_batched(client.chat.completions.create(**kwargs), idle_timeout=_LLM_IDLE_TIMEOUT, stream_timeout=_LLM_STREAM_TIMEOUT):
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
                            if live_messages_fn and new_chars - _live_chars >= LIVE_REDRAW_CHAR_THRESHOLD:
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

                except openai.BadRequestError as _bad_req:
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

                except (_LLMIdleTimeout, ssl.SSLError, OSError, httpx.TransportError,
                        openai.APIConnectionError) as _conn_err:
                    try:
                        client.close()
                    except Exception:
                        pass
                    messages.pop()
                    _llm_elapsed_ms = int((time.monotonic() - _llm_t0) * 1000)
                    if isinstance(_conn_err, openai.APIConnectionError):
                        _err_desc = f"Connection error: LLM backend unreachable ({_conn_err.__cause__ or _conn_err})"
                    else:
                        _err_desc = str(_conn_err)
                    if attempt < _LLM_MAX_RETRIES - 1:
                        if term:
                            term.log("LLM ✗    ", f"model={llm_config.get('model','?')}  {_err_desc}  retry {attempt + 1}/{_LLM_MAX_RETRIES - 1}")
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
                time.sleep(LLM_RETRY_SLEEP_S)
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
        _OPENAI_GEN_PARAMS = {"temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "n", "stop", "logprobs", "seed"}
        _EXTRA_BODY_GEN_PARAMS = {"top_k", "repetition_penalty", "min_p", "min_tokens", "guided_json", "guided_regex"}
        kwargs: dict = dict(
            model=llm_config.get("model", "gpt-4o"),
            messages=[{k: v for k, v in m.items() if not k.startswith("_")} for m in messages],
        )
        for _p in _OPENAI_GEN_PARAMS:
            if _p in llm_config:
                kwargs[_p] = llm_config[_p]
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
        2. vLLM ``max_model_len`` from ``GET /v1/models/{model}``
        3. ``_DEFAULT_CONTEXT_LIMIT`` — safe fallback so compaction always runs
        """
        if "context_limit" in llm_config:
            return int(llm_config["context_limit"])
        try:
            client = openai.OpenAI(
                api_key=llm_config.get("api_key", ""),
                base_url=llm_config.get("base_url"),
            )
            model_id = llm_config.get("model", "")
            all_models = list(client.models.list())
            candidates = [m for m in all_models if m.id == model_id] or all_models
            for info in candidates:
                extra = getattr(info, "model_extra", None) or {}
                if "max_model_len" in extra:
                    return int(extra["max_model_len"])
        except Exception as _e:
            print(f"[agllm] WARNING: failed to retrieve max_model_len from API: {_e}")
        print(f"[agllm] WARNING: context limit unknown, falling back to {_DEFAULT_CONTEXT_LIMIT}")
        return _DEFAULT_CONTEXT_LIMIT

    @staticmethod
    def _make_client(llm_config: dict, timeout: httpx.Timeout) -> openai.OpenAI:
        """Create an OpenAI-compatible client from llm_config.

        Supports two providers:
          - Default (OpenAI / vLLM / any OpenAI-compatible endpoint):
              {"base_url": "...", "api_key": "...", ...}
          - Amazon Bedrock (SigV4 auth, OpenAI-compatible endpoint):
              {"provider": "bedrock", "region": "us-east-2", "model": "<bedrock-model-id>", ...}
        """
        if llm_config.get("provider") == "bedrock":
            region  = llm_config.get("region", "us-east-1")
            api_key = llm_config.get("api_key") or None
            mantle_url  = f"https://bedrock-mantle.{region}.api.aws/v1"
            runtime_url = f"https://bedrock-runtime.{region}.amazonaws.com"
            if api_key and api_key.startswith("bedrock-api-key-"):
                return openai.OpenAI(api_key=api_key, base_url=mantle_url, timeout=timeout)
            if not api_key:
                try:
                    import os as _os
                    from aws_bedrock_token_generator import provide_token as _provide_token
                    _os.environ.setdefault("AWS_DEFAULT_REGION", region)
                    token = _provide_token(region=region)
                    return openai.OpenAI(api_key=token, base_url=mantle_url, timeout=timeout)
                except ImportError:
                    pass
            return openai.OpenAI(
                api_key="bedrock",
                base_url=runtime_url,
                http_client=httpx.Client(
                    auth=_BedrockSigV4Auth(region, api_key=api_key), timeout=timeout
                ),
            )
        return openai.OpenAI(
            api_key=llm_config.get("api_key", "") or "EMPTY",
            base_url=llm_config.get("base_url", None),
            timeout=timeout,
        )

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
        base_url: str = llm_config.get("base_url", "") or ""
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
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
                    tail_turns: int = TAIL_TURNS) -> int:
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
        tail_turns: int = TAIL_TURNS,
        previous_summary: "str | None" = None,
    ) -> "tuple[list[dict], str]":
        """Summarise old messages; return compacted list and new summary."""
        cl = context_limit if context_limit is not None else (self.context_limit or DEFAULT_CONTEXT_LIMIT)
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
            lines.append(f"[task input]: {(task_input[0].get('content') or '')[:SUMMARY_TASK_INPUT_MAX_CHARS]}")
        for m in head:
            role = m.get("role", "?")
            content = (m.get("content") or "").strip()
            tool_calls = m.get("tool_calls")
            if role == "assistant" and tool_calls:
                names = ", ".join(tc["function"]["name"] for tc in tool_calls)
                lines.append(f"[assistant → tools: {names}]")
                if content:
                    lines.append(f"  {content[:SUMMARY_ASSISTANT_CONTENT_MAX_CHARS]}")
            elif role == "tool":
                lines.append(f"[tool result]: {content[:_TOOL_OUTPUT_MAX_CHARS]}")
            elif content:
                lines.append(f"[{role}]: {content[:SUMMARY_ROLE_CONTENT_MAX_CHARS]}")
        client = openai.OpenAI(
            api_key=self.config.get("api_key", ""),
            base_url=self.config.get("base_url"),
        )
        compact_kwargs: dict = dict(
            model=self.config.get("model", "gpt-4o"),
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user",   "content": "\n".join(lines)},
            ],
            max_tokens=SUMMARY_MAX_TOKENS,
        )
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
