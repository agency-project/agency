from __future__ import annotations
import json
import queue
import re
import shlex
import ssl
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Generator, Iterable, TypeVar
import httpx2 as httpx
import openai

AGSKILL_REACT_MAX_STEPS = 4096

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LLM_CALL_MAX_CONCURRENCY     = 128   # Maximum number of simultaneous in-flight LLM streaming calls allowed across all skills.
LLM_HTTP_CONNECT_TIMEOUT     = 10.0  # Seconds for httpx to establish a TCP/TLS connection to the LLM endpoint before giving up.
LLM_HTTP_WRITE_TIMEOUT       = 10.0  # Seconds for httpx to finish writing the request body to the LLM endpoint.
LLM_HTTP_POOL_TIMEOUT        = 10.0  # Seconds httpx waits to acquire a connection from the pool before raising a timeout error.
LIVE_REDRAW_CHAR_THRESHOLD   = 100   # Minimum number of new combined content+thinking characters before pushing a UI redraw during streaming.
LLM_RETRY_SLEEP_S            = 2     # Seconds to wait after a connection/SSL error before retrying the LLM call, allowing drain thread exit and SSL teardown to complete.
AGBINARY_VALIDATE_EXEC_TIMEOUT = 5   # Seconds allowed for each container exec call when validating agbinary output file existence and size.


class _BedrockSigV4Auth(httpx.Auth):
    """httpx auth handler that signs requests with AWS SigV4 for Amazon Bedrock."""

    def __init__(self, region: str, api_key: str | None = None) -> None:
        import boto3
        from botocore.credentials import Credentials
        self._region = region
        if api_key:
            # Accept "ACCESS_KEY_ID:SECRET_ACCESS_KEY" or
            #        "ACCESS_KEY_ID:SECRET_ACCESS_KEY:SESSION_TOKEN"
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
            # Store the live credentials object (not a frozen snapshot) so that
            # boto3 can refresh SSO / assumed-role tokens automatically.
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
        # Resolve fresh credentials on every request — handles token refresh
        # for SSO, assumed-role, and other short-lived credential sources.
        botocore.auth.SigV4Auth(self._creds.get_frozen_credentials(), "bedrock", self._region).add_auth(aws_req)
        for k, v in aws_req.headers.items():
            request.headers[k] = v
        yield request


def _make_llm_client(llm_config: dict, timeout: httpx.Timeout) -> openai.OpenAI:
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
        # Bedrock's OpenAI-compatible endpoint (Mantle) uses a different base URL
        # than the raw bedrock-runtime endpoint.
        mantle_url  = f"https://bedrock-mantle.{region}.api.aws/v1"
        runtime_url = f"https://bedrock-runtime.{region}.amazonaws.com"
        if api_key and api_key.startswith("bedrock-api-key-"):
            # Bedrock API key (bearer token) — use the Mantle endpoint.
            return openai.OpenAI(api_key=api_key, base_url=mantle_url, timeout=timeout)
        if not api_key:
            # No pre-set key: generate a bearer token from the instance's IAM
            # role.  AWS_DEFAULT_REGION is set here (not by the caller) so that
            # botocore's internal credential-refresh client can resolve a region.
            try:
                import os as _os
                from aws_bedrock_token_generator import provide_token as _provide_token
                _os.environ.setdefault("AWS_DEFAULT_REGION", region)
                token = _provide_token(region=region)
                return openai.OpenAI(api_key=token, base_url=mantle_url, timeout=timeout)
            except ImportError:
                pass
        # SigV4 auth — use the raw bedrock-runtime endpoint.
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

_T = TypeVar("_T")
_BATCH_INTERVAL_S: float = 0.1   # main thread drains stream every 100 ms
_IDLE_CHECK_INTERVAL_S: float = 1.0  # how often to check idle timeout
_TOOL_OUTPUT_OFFLOAD_CHARS: int = 40_000  # minimum floor for tool-output offloading


class _LLMIdleTimeout(Exception):
    """Raised by _iter_batched when no chunk arrives within the applicable timeout."""


def _iter_batched(
    iterable: Iterable[_T],
    idle_timeout: float | None = None,
    stream_timeout: float | None = None,
) -> Generator[list[_T], None, None]:
    """Drain *iterable* in a background thread; yield batches to the caller.

    The background thread does minimal Python per item (one queue.put).
    The calling thread sleeps for _BATCH_INTERVAL_S between drains, releasing
    the GIL for that entire interval so other threads run unimpeded.
    GIL acquisitions drop from O(items) to O(items / avg_batch_size).

    *idle_timeout*   — seconds to wait for the **first** chunk before giving up
                       and treating the connection as dead (triggers a retry).
    *stream_timeout* — seconds to wait between chunks **after** streaming has
                       started.  A gap here means the model stalled mid-generation;
                       the partial response is discarded and the call retried.
                       Defaults to None (no mid-stream timeout — wait indefinitely
                       once tokens are flowing).
    """
    _SENTINEL = object()
    q: queue.SimpleQueue = queue.SimpleQueue()

    exc_box: list[BaseException] = []

    def _drain() -> None:
        try:
            for item in iterable:
                q.put(item)
        except BaseException as e:
            exc_box.append(e)
        finally:
            q.put(_SENTINEL)

    threading.Thread(target=_drain, daemon=True).start()

    _last_item = time.monotonic()
    _streaming = False  # True once the first chunk has been received

    while True:
        # Pick the applicable timeout: pre-first-chunk uses idle_timeout (tight,
        # detects dead servers); post-first-chunk uses stream_timeout (loose or
        # None, tolerates model thinking gaps without discarding partial output).
        _current_timeout = stream_timeout if _streaming else idle_timeout
        try:
            if _current_timeout is not None:
                item = q.get(timeout=_IDLE_CHECK_INTERVAL_S)
            else:
                item = q.get()
        except queue.Empty:
            if _current_timeout is not None and time.monotonic() - _last_item >= _current_timeout:
                label = "mid-stream" if _streaming else "pre-first-chunk"
                raise _LLMIdleTimeout(f"no chunk received for {_current_timeout:.0f}s ({label})")
            continue

        _last_item = time.monotonic()
        _streaming = True

        if item is _SENTINEL:
            if exc_box:
                raise exc_box[0]
            return

        # Sleep for one interval — background thread accumulates more items
        # while this thread holds no Python state (GIL fully released).
        time.sleep(_BATCH_INTERVAL_S)

        # Drain everything buffered during the sleep in one burst.
        batch: list[_T] = [item]
        while True:
            try:
                item = q.get_nowait()
                if item is _SENTINEL:
                    yield batch
                    return
                batch.append(item)
            except queue.Empty:
                break

        yield batch

_THINKING_RE = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(content: str) -> str:
    """Remove <think>…</think> / <thinking>…</thinking> blocks from model output."""
    return _THINKING_RE.sub("", content).strip()


def _extract_thinking(content: str) -> str:
    """Return the concatenated text of all thinking blocks, or empty string if none."""
    return "\n\n".join(m.group(1).strip() for m in _THINKING_RE.finditer(content))


def _build_llm_kwargs(llm_config: dict, messages: list[dict], openai_tools: list | None) -> dict:
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


def _build_assistant_msg(
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


def _drain_inbox(
    messages: list[dict],
    _inbox_fn: "Callable | None",
    _live_messages_fn: "Callable | None",
    _full_history_fn: "Callable | None",
) -> bool:
    """Drain pending inbox messages into the conversation. Returns True if any were appended."""
    had_inbox = False
    if _inbox_fn:
        while True:
            msg = _inbox_fn()
            if msg is None:
                break
            inbox_msg = {"role": "user", "content": msg}
            messages.append(inbox_msg)
            had_inbox = True
            if _live_messages_fn:
                _live_messages_fn(messages[1:])
            if _full_history_fn:
                _full_history_fn(inbox_msg)
    return had_inbox


from typing import get_args, get_origin
from .agdata import agdata, agerror, _fmt_exc
from .agtype import agtype, agimage, agrawstring
from .agtool import agtool
from .agcompaction import compact, should_compact, estimate_messages_tokens


_PATH_RE = re.compile(r"^(/[\w.\-]+)+$")


def _looks_like_path(s: str) -> bool:
    """Return True if s looks like a sandbox path (file or directory)."""
    return bool(_PATH_RE.match(s.strip())) if isinstance(s, str) else False


def _hint_to_json_type(hint) -> str:
    """Map a schema hint to a JSON Schema type string for tool parameter specs."""
    if isinstance(hint, type):
        if issubclass(hint, bool):   return "boolean"   # bool before int (bool is subclass of int)
        if issubclass(hint, int):    return "integer"
        if issubclass(hint, float):  return "number"
        if issubclass(hint, agtype): return "string"
        if issubclass(hint, (list, tuple)): return "array"
        if issubclass(hint, dict):   return "object"
        return "string"
    origin = get_origin(hint)
    if origin is list or origin is tuple: return "array"
    if origin is dict:                    return "object"
    if isinstance(hint, list):            return "array"   # [{"key": type, ...}] literal
    return "string"


def _example_for_hint(hint) -> str:
    """Return a short valid-JSON example for a type hint (used in tool descriptions/errors).

    Every returned string is parseable by json.loads.
    """
    if hint is bool:  return "true"
    if hint is int:   return "42"
    if hint is float: return "3.14"
    if hint is str:   return '"text"'
    origin = get_origin(hint)
    args   = get_args(hint)
    if hint is list or origin is list:
        return f"[{_example_for_hint(args[0])}]" if args else "[]"
    if hint is dict or origin is dict:
        if args and len(args) == 2:
            return "{" + f"{_example_for_hint(args[0])}: {_example_for_hint(args[1])}" + "}"
        return "{}"
    if hint is tuple or origin is tuple:
        return "[" + ", ".join(_example_for_hint(t) for t in args) + "]" if args else "[]"
    if isinstance(hint, list) and len(hint) == 1 and isinstance(hint[0], dict):
        obj = "{" + ", ".join(f'"{k}": {_example_for_hint(t)}' for k, t in hint[0].items()) + "}"
        return f"[{obj}]"
    if isinstance(hint, type) and issubclass(hint, agtype):
        return '"value"'
    return "null"


def _value_desc_for_hint(field: str, hint) -> str:
    """Return a value description with a concrete format example and a 'pass directly' note."""
    origin = get_origin(hint)
    args   = get_args(hint)
    ex     = _example_for_hint(hint)
    direct = "Pass directly — do not JSON-encode into a string."

    if hint is str:
        return (
            f"The complete string value for '{field}'. "
            "Pass the full content directly — not a file path."
        )
    if hint is bool:
        return f"Boolean for '{field}' (true or false)."
    if hint is int:
        return f"Integer for '{field}'. Example: {ex}."
    if hint is float:
        return f"Floating-point number for '{field}'. Example: {ex}."
    if hint is list or origin is list:
        if args:
            elem_type = args[0]
            elem_name = getattr(elem_type, "__name__", repr(elem_type))
            return (
                f"JSON array of {elem_name} values for '{field}'. "
                f"Example: {ex}. {direct}"
            )
        return f"JSON array for '{field}'. Example: {ex}. {direct}"
    if hint is dict or origin is dict:
        if args and len(args) == 2:
            kn = getattr(args[0], "__name__", repr(args[0]))
            vn = getattr(args[1], "__name__", repr(args[1]))
            return (
                f"JSON object with {kn} keys and {vn} values for '{field}'. "
                f"Example: {ex}. {direct}"
            )
        return f"JSON object for '{field}'. Example: {ex}. {direct}"
    if hint is tuple or origin is tuple:
        if args:
            types_str = ", ".join(getattr(t, "__name__", repr(t)) for t in args)
            return (
                f"JSON array of {len(args)} element(s) ({types_str}) for '{field}'. "
                f"Example: {ex}. {direct}"
            )
        return f"JSON array for '{field}'. Example: {ex}. {direct}"
    if isinstance(hint, list) and len(hint) == 1 and isinstance(hint[0], dict):
        keys = ", ".join(f'"{k}"' for k in hint[0])
        obj_ex = "{" + ", ".join(f'"{k}": {_example_for_hint(t)}' for k, t in hint[0].items()) + "}"
        return (
            f"JSON array of objects for '{field}'. Each object must have keys: {keys}. "
            f"Example: [{obj_ex}]. {direct}"
        )
    return f"Value for '{field}'."


def _return_tool_descriptions(field: str, hint) -> tuple[str, str]:
    """Return (tool_description, value_description) for a return_<field> tool."""
    tool_desc = f"Register the '{field}' output field."
    # agtype subclass — delegate to its classmethods
    if isinstance(hint, type) and issubclass(hint, agtype):
        return hint.return_tool_description(field), hint.return_value_description(field)
    # list[agtype] — delegate to the inner type, but include a JSON array example
    if get_origin(hint) is list:
        args = get_args(hint)
        if args and isinstance(args[0], type) and issubclass(args[0], agtype):
            inner = args[0]
            ex = _example_for_hint(hint)
            return (
                inner.return_tool_description(field) + " (as a JSON array)",
                inner.return_value_description(field) + f" Provide as a JSON array. Example: {ex}.",
            )
    return tool_desc, _value_desc_for_hint(field, hint)


def _make_return_output_tools(schema: "agdata") -> list[dict]:
    """Build one typed tool per output field from the schema.

    Each tool is named `return_<field>` and has a single `value` parameter
    with the correct JSON Schema type.  This avoids an untyped `value`
    parameter which confuses some model/parser combinations (e.g. qwen3_xml).
    """
    tools = []
    for field, hint in schema._data.items():
        json_type = _hint_to_json_type(hint)
        tool_desc, value_desc = _return_tool_descriptions(field, hint)
        value_schema: dict = {"type": json_type, "description": value_desc}
        tools.append({
            "type": "function",
            "function": {
                "name": f"return_{field}",
                "description": tool_desc,
                "parameters": {
                    "type": "object",
                    "properties": {"value": value_schema},
                    "required": ["value"],
                },
            },
        })
    return tools


def _make_return_output_tool(schema: "agdata") -> list[dict]:
    """Alias kept for test compatibility — returns the full per-field tool list."""
    return _make_return_output_tools(schema)


def _validate_value(hint, value) -> "str | None":
    """Recursively validate value against hint. Returns an error string or None."""
    origin = get_origin(hint)
    args   = get_args(hint)

    if isinstance(hint, type):
        if issubclass(hint, agtype):
            return None if isinstance(value, str) else f"expected str, got {type(value).__name__}"
        if issubclass(hint, bool):
            return None if isinstance(value, bool) else f"expected bool, got {type(value).__name__}"
        if issubclass(hint, (int, float, str)):
            return None if isinstance(value, hint) else f"expected {hint.__name__}, got {type(value).__name__}"
        # bare list/tuple/dict without type args
        if issubclass(hint, (list, tuple)):
            return None if isinstance(value, (list, tuple)) else f"expected array, got {type(value).__name__}"
        if issubclass(hint, dict):
            return None if isinstance(value, dict) else f"expected dict, got {type(value).__name__}"
        return None if isinstance(value, hint) else f"expected {hint.__name__}, got {type(value).__name__}"

    if origin is list:
        if not isinstance(value, list):
            return f"expected list, got {type(value).__name__}"
        if args:
            for i, item in enumerate(value):
                err = _validate_value(args[0], item)
                if err:
                    return f"item {i}: {err}"
        return None

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            return f"expected array, got {type(value).__name__}"
        if args:
            for i, (type_arg, item) in enumerate(zip(args, value)):
                err = _validate_value(type_arg, item)
                if err:
                    return f"item {i}: {err}"
        return None

    if origin is dict:
        if not isinstance(value, dict):
            return f"expected dict, got {type(value).__name__}"
        if len(args) == 2:
            for k, v in value.items():
                err = _validate_value(args[1], v)
                if err:
                    return f"key {k!r}: {err}"
        return None

    # [{"key": type, ...}] literal list-of-dicts schema
    if isinstance(hint, list) and len(hint) == 1 and isinstance(hint[0], dict):
        if not isinstance(value, list):
            return f"expected list, got {type(value).__name__}"
        template = hint[0]
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                return f"item {i}: expected dict, got {type(item).__name__}"
            for k, t in template.items():
                if k not in item:
                    return f"item {i}: missing key '{k}'"
                if isinstance(t, type) and not isinstance(item[k], t):
                    return f"item {i}.{k}: expected {t.__name__}, got {type(item[k]).__name__}"
        return None

    return None


def _validate_output_field(field: str, value, schema: "agdata") -> "str | None":
    """Validate a single (field, value) pair against the output schema hint.

    Returns an error string, or None if valid.
    """
    return _validate_value(schema._data[field], value)

if TYPE_CHECKING:
    from .agterm import agterm
    from .aglog import aglog
    from .agsandbox import agSandbox
    from .agresources import agResourcePool

_llm_call_semaphore = threading.Semaphore(LLM_CALL_MAX_CONCURRENCY)

LLM_MAX_RETRIES    = 10
LLM_IDLE_TIMEOUT   = 300.0   # seconds to wait for first chunk (server dead?)
LLM_STREAM_TIMEOUT = 1800.0  # seconds to wait between chunks mid-stream


@contextmanager
def _llm_call_semaphore_slot():
    _llm_call_semaphore.acquire()
    try:
        yield
    finally:
        _llm_call_semaphore.release()


def _maybe_compact(
    messages: list[dict],
    llm_config: dict,
    _context_limit: "int | None",
    prompt_tokens: "int | None",
    _compaction_summary: "str | None",
    term: "agterm | None",
    _compact_log_fn: "Callable | None",
    _live_messages_fn: "Callable | None",
    skill_name: str,
    force: bool = False,
) -> "tuple[list[dict], str | None]":
    """Run compaction if needed. Returns (messages, updated_compaction_summary).

    prompt_tokens=None → pre-call path (uses chars/4 estimate, log label includes tilde).
    prompt_tokens=int  → post-call path (uses actual API-reported token count).
    force=True         → skip the should_compact gate (used after a context-exceeded 400).
    """
    if _context_limit is None:
        return messages, _compaction_summary
    if prompt_tokens is None:
        token_count = estimate_messages_tokens(messages)
        label = f"tokens~{token_count}/{_context_limit}  msgs={len(messages)}  (pre-call estimate)"
    else:
        token_count = prompt_tokens
        label = f"tokens={token_count}/{_context_limit}  msgs={len(messages)}"
    if not force and not should_compact(token_count, _context_limit):
        return messages, _compaction_summary
    if term:
        term.log("COMPACT  ", f"skill={skill_name}  {label}")
    msgs_before = len(messages)
    messages, _compaction_summary = compact(
        messages, llm_config,
        context_limit=_context_limit,
        previous_summary=_compaction_summary,
    )
    if _compact_log_fn:
        _compact_log_fn(
            skill=skill_name,
            prompt_tokens=token_count,
            context_limit=_context_limit,
            msgs_before=msgs_before,
            msgs_after=len(messages),
        )
    if _live_messages_fn:
        _live_messages_fn(messages[1:])
    return messages, _compaction_summary


@dataclass
class _LLMCallResult:
    conn_error: "Exception | None" = None
    should_retry: bool = False
    context_exceeded: bool = False
    next_timeout_attempt: int = 0
    content_parts: "list[str]" = field(default_factory=list)
    reasoning_parts: "list[str]" = field(default_factory=list)
    tool_calls_raw: "dict[int, dict]" = field(default_factory=dict)
    prompt_tokens: "int | None" = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    elapsed_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.conn_error is None and not self.should_retry and not self.context_exceeded


@dataclass
class _FinalAnswerResult:
    kind: str  # "return" | "retry" | "error"
    return_tuple: "tuple | None" = None
    correction_msg: "dict | None" = None


def _llm_call(
    kwargs: dict,
    llm_config: dict,
    messages: list[dict],
    _timeout_attempt: int,
    term: "agterm | None",
    _state_fn: "Callable | None",
    _live_messages_fn: "Callable | None",
    _token_update_fn: "Callable | None",
    _total_input_tokens: int,
    _total_output_tokens: int,
    skill_name: str,
) -> "_LLMCallResult":
    """Execute one streaming LLM call inside the semaphore slot.

    Returns an _LLMCallResult. Caller checks .ok, .should_retry, .conn_error.
    """
    _LLM_MAX_RETRIES    = LLM_MAX_RETRIES
    _LLM_IDLE_TIMEOUT   = LLM_IDLE_TIMEOUT
    _LLM_STREAM_TIMEOUT = LLM_STREAM_TIMEOUT

    kwargs = dict(kwargs)  # shallow copy so we don't mutate caller's dict
    kwargs["stream"] = True
    kwargs["stream_options"] = {"include_usage": True}

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_raw: dict[int, dict] = {}
    prompt_tokens: int | None = None

    _PARTIAL_THINK_RE = re.compile(
        r"<think(?:ing)?>(.*?)(?:</think(?:ing)?>|$)", re.DOTALL | re.IGNORECASE
    )

    with _llm_call_semaphore_slot():
        client = _make_llm_client(
            llm_config,
            httpx.Timeout(connect=LLM_HTTP_CONNECT_TIMEOUT, read=None, write=LLM_HTTP_WRITE_TIMEOUT, pool=LLM_HTTP_POOL_TIMEOUT),
        )

        if term:
            term.log("LLM ▶    ", f"model={llm_config.get('model','?')}  messages={len(messages)}  idle_timeout={_LLM_IDLE_TIMEOUT:.0f}s  stream_timeout={_LLM_STREAM_TIMEOUT:.0f}s")
        if _state_fn:
            _state_fn("llm", skill=skill_name)

        _llm_t0 = time.monotonic()

        partial_msg: dict = {"role": "assistant", "content": ""}
        messages.append(partial_msg)
        if _live_messages_fn:
            _live_messages_fn(messages[1:])
        _live_chars = 0

        try:
            for batch in _iter_batched(client.chat.completions.create(**kwargs), idle_timeout=_LLM_IDLE_TIMEOUT, stream_timeout=_LLM_STREAM_TIMEOUT):
                for chunk in batch:
                    if chunk.usage is not None:
                        prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                        _total_input_tokens  += prompt_tokens
                        _total_output_tokens += getattr(chunk.usage, "completion_tokens", 0) or 0
                        if term is not None:
                            term._tokens = prompt_tokens
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    # Reasoning tokens — field name varies by model/backend:
                    # "reasoning_content" (DeepSeek-R1 / some vLLM builds)
                    # "reasoning"         (Kimi-K2 and others via model_extra)
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
                        # For <think>-tag models: expose thinking live even before closing tag
                        m = _PARTIAL_THINK_RE.search(raw)
                        if m:
                            partial_msg["_thinking"] = m.group(1).strip()
                            partial_msg["content"] = _THINKING_RE.sub("", raw).strip()
                        else:
                            partial_msg["content"] = raw

                    # Throttle UI redraws: push every ~100 new combined chars
                    new_chars = len(partial_msg.get("content", "")) + len(partial_msg.get("_thinking", ""))
                    if _live_messages_fn and new_chars - _live_chars >= LIVE_REDRAW_CHAR_THRESHOLD:
                        _live_messages_fn(messages[1:])
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
                return _LLMCallResult(context_exceeded=True, elapsed_ms=_llm_elapsed_ms)
            if term:
                term.log("LLM ✗    ", f"model={llm_config.get('model','?')}  bad request: {_bad_req}")
            return _LLMCallResult(conn_error=_bad_req, should_retry=False, elapsed_ms=_llm_elapsed_ms)

        except (_LLMIdleTimeout, ssl.SSLError, OSError, httpx.TransportError) as _conn_err:
            try:
                client.close()  # best-effort: unblock drain thread's ssl.read()
            except Exception:
                pass
            messages.pop()  # remove partial placeholder
            _llm_elapsed_ms = int((time.monotonic() - _llm_t0) * 1000)
            _err_desc = str(_conn_err) if not isinstance(_conn_err, _LLMIdleTimeout) else str(_conn_err)
            if _timeout_attempt < _LLM_MAX_RETRIES - 1:
                next_attempt = _timeout_attempt + 1
                if term:
                    term.log("LLM ✗    ", f"model={llm_config.get('model','?')}  {_err_desc}  retry {next_attempt}/{_LLM_MAX_RETRIES - 1}")
                return _LLMCallResult(
                    conn_error=_conn_err,
                    should_retry=True,
                    next_timeout_attempt=next_attempt,
                    elapsed_ms=_llm_elapsed_ms,
                )
            if term:
                term.log("LLM ✗    ", f"model={llm_config.get('model','?')}  {_err_desc}  all retries exhausted")
            return _LLMCallResult(
                conn_error=_conn_err,
                should_retry=False,
                next_timeout_attempt=_timeout_attempt,
                elapsed_ms=_llm_elapsed_ms,
            )

        messages.pop()  # remove partial placeholder
        _llm_elapsed_ms = int((time.monotonic() - _llm_t0) * 1000)

    if _token_update_fn is not None:
        try:
            _token_update_fn(_total_input_tokens, _total_output_tokens)
        except Exception as _e:
            print(f"[agskill] WARNING: token_update_fn raised: {_e}")

    return _LLMCallResult(
        content_parts=content_parts,
        reasoning_parts=reasoning_parts,
        tool_calls_raw=tool_calls_raw,
        prompt_tokens=prompt_tokens,
        total_input_tokens=_total_input_tokens,
        total_output_tokens=_total_output_tokens,
        elapsed_ms=_llm_elapsed_ms,
    )


def _dispatch_tools(
    tool_calls: list[dict],
    tool_map: dict,
    messages: list[dict],
    sandbox: "agSandbox",
    skill_name: str,
    _state_fn: "Callable | None",
    _live_messages_fn: "Callable | None",
    _full_history_fn: "Callable | None",
    term: "agterm | None",
    _intercept: "dict[str, Callable[[dict], str]] | None" = None,
    tool_offload_chars: int = _TOOL_OUTPUT_OFFLOAD_CHARS,
) -> bool:
    """Execute all tool calls from one LLM response, appending results to messages.

    Returns True if the read tool was lazily injected into tool_map during this
    dispatch (because a large output was offloaded and read was not already present).
    The caller should then add the read tool's schema to openai_tools so the LLM
    can use it on the next step.
    """
    _injected_read = False
    for tc in tool_calls:
        fn_name = tc["function"]["name"]
        fn_args = tc["function"]["arguments"]
        tc_id   = tc["id"]
        # Ensure arguments is valid JSON before it goes back into history.
        # A malformed string (truncated generation, Python repr, etc.) causes
        # vLLM to crash on the next request when it re-parses the history.
        try:
            json.loads(fn_args)
        except (json.JSONDecodeError, TypeError):
            fn_args = "{}"
            tc["function"]["arguments"] = fn_args

        # Framework-internal tools (e.g. return_output) are handled in the
        # calling thread before normal tool dispatch.
        if _intercept and fn_name in _intercept:
            try:
                args = json.loads(fn_args)
            except (json.JSONDecodeError, TypeError):
                args = {}
            result_content = _intercept[fn_name](args)
            tool_msg = {"role": "tool", "tool_call_id": tc_id, "content": result_content}
            messages.append(tool_msg)
            if _live_messages_fn:
                _live_messages_fn(messages[1:])
            if _full_history_fn:
                _full_history_fn(tool_msg)
            continue

        t = tool_map.get(fn_name)
        if t is None:
            if term:
                term.log("TOOL ✗   ", f"{fn_name}  → unknown tool")
            result_content = json.dumps({"error": f"unknown tool: {fn_name}"})
        else:
            try:
                if _state_fn:
                    _state_fn("tool", skill=skill_name, tool=fn_name)
                # Let the agent specify a custom timeout (seconds) via a
                # "timeout" key in the tool arguments.
                _tool_timeout: int | None = None
                try:
                    _parsed = json.loads(fn_args)
                    if isinstance(_parsed.get("timeout"), int):
                        _tool_timeout = _parsed["timeout"]
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
                result_content = t(agdata.from_json(fn_args), timeout=_tool_timeout).to_json()
                if _state_fn:
                    _state_fn("skill", skill=skill_name)
                # Offload large tool outputs regardless of whether the tool
                # itself uses the sandbox — fetch_paper and other add_tools have
                # need_sandbox=False but can still produce huge outputs that
                # bloat the context.
                if len(result_content) > tool_offload_chars:
                    safe_id = tc_id.replace("-", "")[:12]
                    offload_path = f"/workspace/long_tool_call_outputs/{fn_name}_{safe_id}.txt"
                    try:
                        try:
                            file_body = json.loads(result_content).get("content", result_content)
                        except (json.JSONDecodeError, AttributeError):
                            file_body = result_content
                        sandbox.write_file(offload_path, file_body)
                        result_content = json.dumps({
                            "note": f"Output was too large and has been saved to {offload_path}. Use the read tool to access it."
                        })
                        if "read" not in tool_map:
                            from .tools import make_read
                            _read_tool = make_read(sandbox)
                            tool_map["read"] = _read_tool
                            _injected_read = True
                    except Exception as _e:
                        print(f"[agskill] WARNING: failed to offload large tool output to {offload_path}: {_e}")
                if t.need_sandbox:
                    # A tool may signal failure via agdata(error=...) without raising —
                    # treat that the same as an exception: discard dirty state.
                    _result_errored = False
                    try:
                        if "error" in json.loads(result_content):
                            _result_errored = True
                    except (json.JSONDecodeError, TypeError):
                        pass
                    if _result_errored:
                        sandbox.stop(commit=False)
                        try:
                            _result_obj = json.loads(result_content)
                            _result_obj["workspace_reverted"] = (
                                "The workspace has been reverted to the state "
                                "before this tool call."
                            )
                            result_content = json.dumps(_result_obj)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    else:
                        sandbox.stop(commit=True)
            except Exception as e:
                if _state_fn:
                    _state_fn("skill", skill=skill_name)
                result_content = json.dumps({"error": _fmt_exc(e)})
                # On failure: remove without committing to discard dirty state.
                # The next tool call restores from the last successful checkpoint.
                if t.need_sandbox:
                    sandbox.stop(commit=False)
                    try:
                        _result_obj = json.loads(result_content)
                        _result_obj["workspace_reverted"] = (
                            "The workspace has been reverted to the state "
                            "before this tool call."
                        )
                        result_content = json.dumps(_result_obj)
                    except (json.JSONDecodeError, TypeError):
                        pass
        tool_msg = {"role": "tool", "tool_call_id": tc_id, "content": result_content}
        messages.append(tool_msg)
        if _live_messages_fn:
            _live_messages_fn(messages[1:])
        if _full_history_fn:
            _full_history_fn(tool_msg)
    return _injected_read


def _wait_for_processes(
    sandbox: "agSandbox",
    skill_name: str,
    term: "agterm | None",
    log: "aglog | None",
    agname: str,
    ping_interval_s: float,
    poll_interval_s: float,
    _state_fn: "Callable | None" = None,
) -> "str | None":
    """Wait for sandbox background processes after the LLM produces a final answer.

    Returns None if the sandbox is already clean (no action needed).
    Otherwise polls until all PIDs exit or ping_interval_s elapses, then
    returns a user-facing message to inject into the conversation so the
    LLM can act on the outcome.
    """
    # Quick pre-check: if no PIDs are being tracked, skip the expensive /proc scan.
    # Check isinstance(dict) to avoid triggering on MagicMock sandboxes in tests.
    watched = getattr(sandbox, "_watched_pids", None)
    if not isinstance(watched, dict) or not watched:
        return None
    get_live = sandbox.get_live_pids
    if not get_live():
        return None

    summary = sandbox.pid_status_summary()
    if _state_fn:
        _state_fn("proc_wait", skill=skill_name)
    if term:
        term.log("PROCS ▶  ", f"{skill_name}  monitoring: {summary}")
    if log:
        log._lifecycle("procs_started", agname=agname, skill=skill_name,
                       pids=list(get_live()), summary=summary)

    deadline = time.monotonic() + ping_interval_s
    while time.monotonic() < deadline:
        time.sleep(poll_interval_s)
        if not get_live():
            break

    live_now = get_live()

    if not live_now:
        if term:
            term.log("PROCS ✓  ", f"{skill_name}  all processes completed, re-entering agent")
        if log:
            log._lifecycle("procs_completed", agname=agname, skill=skill_name)
        return (
            "Background processes have completed. "
            "Read their output and act on the results."
        )

    summary = sandbox.pid_status_summary()
    if term:
        term.log("PROCS ⏳  ", f"{skill_name}  still running: {summary}")
    if log:
        log._lifecycle("procs_ping", agname=agname, skill=skill_name,
                       pids=list(live_now), summary=summary)
    return (
        f"Background processes are still running: {summary}. "
        f"You may check their output, wait, or proceed if appropriate. "
        f"If any of these processes are intentional long-running services "
        f"(daemons, servers, monitors) that should not block completion, "
        f"call daemon_release(pid) for each such PID to release it from monitoring."
    )


class agskill:
    """A named skill with its own system prompt and a self-contained ReAct loop.

    input_schema / output_schema are agdata objects whose keys define required
    fields and whose values are Python types (``str``, ``int``, ``float``,
    ``bool``, ``list``, ``dict``, or an ``agtype`` subclass such as ``agfile``).
    Both schemas are serialised and appended to the system prompt so the LLM
    knows the contract.

    Input is validated before the loop runs.  Output is validated after each
    final (non-tool-call) LLM response; on failure a correction message is
    injected and the loop retries up to max_output_schema_retries times.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        add_tools: list[agtool] | None = None,
        replace_tools: list[agtool] | None = None,
        input_schema: agdata | None = None,
        output_schema: agdata | None = None,
        max_output_schema_retries: int = 10,
        plan_mode: bool = False,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.add_tools = add_tools
        self.replace_tools = [] if plan_mode else replace_tools
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.max_output_schema_retries = max_output_schema_retries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _agtype_class(hint: object) -> type[agtype] | None:
        """Return the agtype subclass for a hint, handling both T and list[T]."""
        if isinstance(hint, type) and issubclass(hint, agtype):
            return hint
        if get_origin(hint) is list:
            args = get_args(hint)
            if args and isinstance(args[0], type) and issubclass(args[0], agtype):
                return args[0]
        return None

    @staticmethod
    def _output_field_desc(hint: object) -> str:
        """Return a human-readable type description with usage guidance for an output field."""
        if isinstance(hint, type) and issubclass(hint, agtype):
            return hint.schema_type()
        if hint is str:
            return "string — pass the complete text content as the value (not a file path)"
        if hint is int:
            return "integer — pass the numeric value directly"
        if hint is float:
            return "float — pass the numeric value directly"
        if hint is bool:
            return "boolean — pass true or false"
        if hint is list or hint is dict:
            return hint.__name__
        if get_origin(hint) is list:
            args = get_args(hint)
            inner = agskill._output_field_desc(args[0]) if args else "any"
            return f"array of {inner}"
        return str(hint)

    def _raw_input_key(self) -> "str | None":
        """Return the field key if input_schema is a single agrawstring field."""
        if self.input_schema is None:
            return None
        items = list(self.input_schema._data.items())
        if len(items) == 1:
            key, hint = items[0]
            if isinstance(hint, type) and issubclass(hint, agrawstring):
                return key
        return None

    def _raw_output_key(self) -> "str | None":
        """Return the field key if output_schema is a single agrawstring field."""
        if self.output_schema is None:
            return None
        items = list(self.output_schema._data.items())
        if len(items) == 1:
            key, hint = items[0]
            if isinstance(hint, type) and issubclass(hint, agrawstring):
                return key
        return None

    def _build_system_prompt(self, extra: str | None = None) -> str:
        parts = [self.system_prompt]

        # Collect agtype fields with extra prompt instructions and emit
        # them before the JSON format sections.
        extra_lines: list[str] = []
        for key, hint in (self.input_schema._data.items() if self.input_schema else []):
            cls = self._agtype_class(hint)
            if cls is not None:
                line = cls.extra_input_prompt(key)
                if line:
                    extra_lines.append(line)
        for key, hint in (self.output_schema._data.items() if self.output_schema else []):
            cls = self._agtype_class(hint)
            if cls is not None:
                line = cls.extra_output_prompt(key, self.name)
                if line:
                    extra_lines.append(line)

        if extra_lines:
            parts.append(
                "\nFile-backed fields — WARNING: these files are temporary and will "
                "be automatically deleted after this task ends:\n"
                + "\n".join(extra_lines)
            )

        if extra:
            parts.append(extra)

        if self.input_schema is not None and self._raw_input_key() is None:
            parts.append(f"\nInput JSON format:\n{self.input_schema.to_json()}")
        if self.output_schema is not None:
            if self._raw_output_key() is not None:
                parts.append("\nRespond with plain text only — no JSON wrapping, no markdown code fences.")
            else:
                field_tools = ", ".join(
                    f"return_{f}" for f in self.output_schema._data
                )
                field_lines = "\n".join(
                    f"  - {f}: {self._output_field_desc(h)}"
                    for f, h in self.output_schema._data.items()
                )
                parts.append(
                    f"\nTo return your results, call the appropriate return_<field> tool "
                    f"once for each required output field ({field_tools}). "
                    f"Required fields:\n"
                    f"{field_lines}\n\n"
                    "- Call each return_<field> tool separately — one field per call.\n"
                    "- Only call a return_<field> tool when you have the final value for that field.\n"
                    "- You may continue using other tools after registering outputs if needed."
                )
        return "\n".join(parts)

    def _build_user_content(self, input: agdata) -> "str | list":
        """Build the content value for the user message.

        Returns a plain string when there are no image fields, or a multimodal
        content array when agimage fields are present.  Image field values are
        replaced with a short placeholder in the text portion so the LLM doesn't
        see a raw base64 blob in the JSON.
        """
        # agrawstring input — send the value directly, no JSON wrapping.
        raw_key = self._raw_input_key()
        if raw_key is not None:
            val = input._data.get(raw_key, "")
            return val if isinstance(val, str) else str(val)

        schema = self.input_schema
        image_urls: list[str] = []
        image_keys: set[str] = set()

        if schema is not None:
            for key, hint in schema._data.items():
                # Single agimage
                if isinstance(hint, type) and issubclass(hint, agimage):
                    image_keys.add(key)
                    val = input._data.get(key)
                    if isinstance(val, str):
                        image_urls.append(val)
                # list[agimage]
                elif get_origin(hint) is list:
                    args = get_args(hint)
                    if args and isinstance(args[0], type) and issubclass(args[0], agimage):
                        image_keys.add(key)
                        vals = input._data.get(key, [])
                        if isinstance(vals, list):
                            image_urls.extend(v for v in vals if isinstance(v, str))

        if not image_urls:
            return f"[HARNESS SYSTEM] New Skill Input:\n{input.to_json()}"

        # Build a copy of the input dict with image fields replaced by placeholders
        # so the text portion stays compact.
        text_data = dict(input._data)
        for key in image_keys:  
            if key in text_data:
                hint = schema._data.get(key) if schema else None
                if get_origin(hint) is list:
                    count = len(text_data[key]) if isinstance(text_data[key], list) else 1
                    text_data[key] = f"[{count} image(s) attached]"
                else:
                    text_data[key] = "[image attached]"

        import json as _json
        text = _json.dumps(text_data)
        content: list = [{"type": "text", "text": f"New Skill Input:\n{text}"}]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        return content

    def _check_schema(self, data: agdata, schema: agdata) -> list[str]:
        """Return a list of error strings; empty list means the data is valid."""
        errors: list[str] = []
        for key, hint in schema._data.items():
            if key not in data._data:
                errors.append(f"missing required field '{key}'")
                continue
            actual = data._data[key]
            if isinstance(hint, type) and issubclass(hint, agtype):
                # agtype fields carry a string value after framework processing
                if not isinstance(actual, str):
                    errors.append(
                        f"field '{key}' ({hint.__name__}) must be a string"
                    )
                continue
            if isinstance(hint, list) and len(hint) == 1 and isinstance(hint[0], dict):
                # list[{key: type, ...}] — validate every item matches the template
                item_template = hint[0]
                if not isinstance(actual, list):
                    errors.append(f"field '{key}': expected list, got {type(actual).__name__}")
                    continue
                for i, item in enumerate(actual):
                    if not isinstance(item, dict):
                        errors.append(
                            f"field '{key}[{i}]': expected dict, got {type(item).__name__}"
                        )
                        continue
                    for item_key, item_type in item_template.items():
                        if item_key not in item:
                            errors.append(f"field '{key}[{i}]': missing key '{item_key}'")
                        elif isinstance(item_type, type) and not isinstance(item[item_key], item_type):
                            errors.append(
                                f"field '{key}[{i}].{item_key}': expected {item_type.__name__}, "
                                f"got {type(item[item_key]).__name__}"
                            )
                continue
            if isinstance(hint, type):
                if not isinstance(actual, hint):
                    errors.append(
                        f"field '{key}': expected {hint.__name__}, "
                        f"got {type(actual).__name__}"
                    )
        return errors

    def _parse_final_answer(
        self,
        msg_dict: dict,
        messages: list[dict],
        n_before: int,
        output_schema_retries_left: int,
        tokens: "tuple[int, int]",
    ) -> "_FinalAnswerResult":
        """Parse the LLM's final (non-tool-call) response.

        Returns a _FinalAnswerResult with kind:
          "return"  — caller should return return_tuple immediately
          "retry"   — caller should append correction_msg and continue the loop
          "error"   — caller should return an error return_tuple
        """
        # agrawstring output — capture the raw text and return immediately.
        raw_out_key = self._raw_output_key()
        if raw_out_key is not None:
            raw_content = msg_dict.get("content") or ""
            updated_history = agdata(messages=messages[1:])
            return _FinalAnswerResult(
                kind="return",
                return_tuple=(
                    agdata(**{raw_out_key: raw_content}),
                    updated_history,
                    [messages[0]] + messages[1:][n_before:],
                    tokens,
                ),
            )

        content = msg_dict.get("content") or "{}"
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped[stripped.find("\n") + 1:] if "\n" in stripped else stripped[3:]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
            content = stripped.strip()
        _parse_error: str = ""
        try:
            result = agdata.from_json(content)
        except json.JSONDecodeError as _e:
            if "Extra data" in str(_e):
                try:
                    _obj, _ = json.JSONDecoder().raw_decode(content.lstrip())
                    result = agdata(**_obj) if isinstance(_obj, dict) else agdata(result=content)
                    _parse_error = ""
                except (json.JSONDecodeError, TypeError):
                    _parse_error = str(_e)
                    result = agdata(result=content)
            else:
                _parse_error = str(_e)
                result = agdata(result=content)
        except TypeError as _e:
            _parse_error = str(_e)
            result = agdata(result=content)

        if self.output_schema is not None:
            errors = self._check_schema(result, self.output_schema)
            if _parse_error and "invalid json" not in " ".join(errors).lower():
                errors = [f"JSON parse error: {_parse_error}"] + errors
            if errors:
                if output_schema_retries_left > 0:
                    return _FinalAnswerResult(
                        kind="retry",
                        correction_msg={
                            "role": "user",
                            "content": (
                                f"[HARNESS SYSTEM] Your previous response could not be parsed. Errors: {errors}.\n"
                                "Common causes: unescaped quotes or backslashes inside a string value, "
                                "raw newlines inside a string (use \\n instead), trailing comma after "
                                "the last key, extra text or explanation outside the JSON object, "
                                "or markdown code fences around the JSON.\n"
                                "Respond ONLY with a single valid JSON object matching exactly: "
                                f"{self.output_schema.to_json()}"
                            ),
                        },
                    )
                _raw_out = (msg_dict.get("content") or "")[:2000]
                updated_history = agdata(messages=messages[1:])
                return _FinalAnswerResult(
                    kind="error",
                    return_tuple=(
                        agerror(
                            f"output schema error after retries: {errors}"
                            + (f"\nmodel output: {_raw_out!r}" if _raw_out else "")
                        ),
                        updated_history,
                        [messages[0]] + messages[1:][n_before:],
                        tokens,
                    ),
                )

        updated_history = agdata(messages=messages[1:])
        return _FinalAnswerResult(
            kind="return",
            return_tuple=(result, updated_history, [messages[0]] + messages[1:][n_before:], tokens),
        )

    def _validate_input(
        self,
        input: agdata,
        _is_continuation: bool,
        _extra_system: "str | None",
    ) -> "str | None":
        """Return an error string if input fails schema validation, else None."""
        if self.input_schema is None or _is_continuation:
            return None
        errors = self._check_schema(input, self.input_schema)
        if errors:
            return f"input schema error: {errors}"
        return None

    def _build_tools(
        self,
        sandbox: "agSandbox",
        pool: "agResourcePool | None",
        term: "agterm | None",
        log: "aglog | None",
        _ensure_read: bool = False,
    ) -> "tuple[list, dict, list | None]":
        """Build and return (active_tools, tool_map, openai_tools)."""
        from .tools import make_sandboxed_tools, make_read
        if self.replace_tools is not None:
            active_tools: list[agtool] = list(self.replace_tools)
        else:
            active_tools = make_sandboxed_tools(sandbox, pool)
            if self.add_tools:
                active_tools.extend(self.add_tools)
        # If input fields were offloaded to sandbox files, ensure the read tool
        # is available even for skills that use replace_tools without it.
        if _ensure_read:
            if not any(getattr(t, "name", None) == "read" for t in active_tools):
                active_tools.append(make_read(sandbox))
        for t in active_tools:
            t.attach_logger(term, log)
        tool_map = {t.name: t for t in active_tools}
        openai_tools = [t.to_openai_tool() for t in active_tools] or None
        return active_tools, tool_map, openai_tools

    def _build_initial_messages(
        self,
        input: agdata,
        history: agdata,
        _extra_system: "str | None",
        _live_messages_fn: "Callable | None",
        _full_history_fn: "Callable | None",
    ) -> "tuple[list[dict], int]":
        """Build initial messages list. Returns (messages, n_before)."""
        history_msgs: list[dict] = list(history._data.get("messages", []))
        n_before = len(history_msgs)
        messages: list[dict] = (
            [{"role": "system", "content": self._build_system_prompt(_extra_system)}]
            + history_msgs
            + [{"role": "user", "content": self._build_user_content(input)}]
        )
        if _live_messages_fn:
            _live_messages_fn(messages[1:])
        if _full_history_fn:
            _full_history_fn(messages[0])
            _full_history_fn(messages[-1])
        return messages, n_before

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------

    def run(
        self,
        llm_config: dict,
        input: agdata,
        history: agdata,
        sandbox: "agSandbox",
        pool: "agResourcePool | None" = None,
        max_steps: int = AGSKILL_REACT_MAX_STEPS,
        term: "agterm | None" = None,
        log: "aglog | None" = None,
        _is_continuation: bool = False,
        _state_fn: "Callable | None" = None,
        _live_messages_fn: "Callable | None" = None,
        _inbox_fn: "Callable | None" = None,
        _context_limit: "int | None" = None,
        _compact_log_fn: "Callable | None" = None,
        _full_history_fn: "Callable[[dict], None] | None" = None,
        _extra_system: "str | None" = None,
        _token_update_fn: "Callable[[int, int], None] | None" = None,
        _ping_interval_s: float = 300,
        _poll_interval_s: float = 5,
        _agname: str = "",
        _ensure_read: bool = False,
    ) -> tuple[agdata, agdata, list[dict], tuple[int, int]]:
        """Run the ReAct loop, including sandbox process monitoring.

        Returns (result_agdata, updated_history_agdata, history_delta).
        history_delta is the list of new messages added during this skill's
        execution (user input → tool calls / results → final answer).
        The system_prompt (+ schemas) is prepended to every call but is NOT
        persisted in history.

        When *_is_continuation* is True, input schema validation is skipped
        so continuation messages can flow through without matching the skill's
        declared input schema.

        After the LLM produces a final answer, if the sandbox has live
        background processes the loop does not exit — it waits up to
        *_ping_interval_s* seconds, then injects a status message and
        continues so the LLM can act on process completion or progress.
        """
        input_error = self._validate_input(input, _is_continuation, _extra_system)
        if input_error is not None:
            sys_msg = {"role": "system", "content": self._build_system_prompt(_extra_system)}
            return agerror(input_error), history, [sys_msg], (0, 0)

        _active_tools, tool_map, openai_tools = self._build_tools(sandbox, pool, term, log, _ensure_read=_ensure_read)

        # Tool-based output collection (return_output tool).
        # Used for all output schemas except agrawstring (raw text).
        _use_return_output = (
            self.output_schema is not None and self._raw_output_key() is None
        )
        _collected_outputs: dict = {}
        _required_fields: set[str] = set()
        _intercept: "dict[str, Callable[[dict], str]] | None" = None
        if _use_return_output:
            _required_fields = set(self.output_schema._data.keys())
            _return_tools = _make_return_output_tools(self.output_schema)
            openai_tools = _return_tools + (openai_tools or [])

            def _make_field_handler(field: str):
                from .agtype import agfile as _agfile, agbinary as _agbinary
                hint = self.output_schema._data[field]
                _is_agfile   = isinstance(hint, type) and issubclass(hint, _agfile)
                _is_agbinary = isinstance(hint, type) and issubclass(hint, _agbinary)
                _is_str      = hint is str

                def _handle(args: dict) -> str:
                    value = args.get("value")
                    err = _validate_output_field(field, value, self.output_schema)
                    if err is not None:
                        hint    = self.output_schema._data[field]
                        ex      = _example_for_hint(hint)
                        got_str = isinstance(value, str)
                        exp_arr = _hint_to_json_type(hint) == "array"
                        exp_obj = _hint_to_json_type(hint) == "object"
                        if got_str and exp_arr:
                            fix = (
                                f"You passed a JSON-encoded string; pass a JSON array directly. "
                                f"Example: {ex}"
                            )
                        elif got_str and exp_obj:
                            fix = (
                                f"You passed a JSON-encoded string; pass a JSON object directly. "
                                f"Example: {ex}"
                            )
                        else:
                            fix = f"Expected format: {ex}"
                        return json.dumps({"error": f"field '{field}': {err}. {fix}"})

                    # agfile: validate file exists, is a regular file, is non-empty,
                    # is UTF-8 text, and contains real content (not another path).
                    if _is_agfile and isinstance(value, str):
                        try:
                            content = sandbox.read_file(value)
                        except IsADirectoryError:
                            return json.dumps({"error": (
                                f"field '{field}': '{value}' is a directory, not a file. "
                                f"Pass the path to a specific output file "
                                f"(e.g. {value}/{field}.txt)."
                            )})
                        except UnicodeDecodeError:
                            return json.dumps({"error": (
                                f"field '{field}': file at '{value}' contains binary data "
                                f"and cannot be read as text. Write a UTF-8 text file instead."
                            )})
                        except Exception:
                            return json.dumps({"error": (
                                f"field '{field}': no file found at path '{value}'. "
                                f"Write your output to a file first, then call this "
                                f"tool with that file's path."
                            )})
                        if not content or not content.strip():
                            return json.dumps({"error": (
                                f"field '{field}': file at '{value}' is empty. "
                                f"Write the actual content to the file before "
                                f"registering the path."
                            )})
                        if _looks_like_path(content.strip()):
                            return json.dumps({"error": (
                                f"field '{field}': file at '{value}' contains only a "
                                f"path reference ('{content.strip()}'), not real content. "
                                f"Write the actual content to a file and return that "
                                f"file's path."
                            )})

                    # agbinary: check the file exists and is non-empty using a
                    # lightweight shell test — no content read needed.
                    if _is_agbinary and isinstance(value, str):
                        _, dir_rc = sandbox._container_exec(
                            f"test -d {shlex.quote(value)}", timeout=AGBINARY_VALIDATE_EXEC_TIMEOUT, shell="sh"
                        )
                        if dir_rc == 0:
                            return json.dumps({"error": (
                                f"field '{field}': '{value}' is a directory, not a file. "
                                f"Pass the path to a specific binary output file "
                                f"(e.g. {value}/{field}.bin)."
                            )})
                        _, exist_rc = sandbox._container_exec(
                            f"test -s {shlex.quote(value)}", timeout=AGBINARY_VALIDATE_EXEC_TIMEOUT, shell="sh"
                        )
                        if exist_rc != 0:
                            # -s fails for both missing and zero-byte files; distinguish them.
                            _, found_rc = sandbox._container_exec(
                                f"test -e {shlex.quote(value)}", timeout=AGBINARY_VALIDATE_EXEC_TIMEOUT, shell="sh"
                            )
                            if found_rc != 0:
                                return json.dumps({"error": (
                                    f"field '{field}': no file found at path '{value}'. "
                                    f"Write your binary output to a file first, then call "
                                    f"this tool with that file's path."
                                )})
                            return json.dumps({"error": (
                                f"field '{field}': file at '{value}' is empty. "
                                f"Write the actual binary content to the file before "
                                f"registering the path."
                            )})

                    # str: if the agent returned a file path instead of content,
                    # silently resolve it to the file's content.
                    if _is_str and isinstance(value, str) and _looks_like_path(value):
                        try:
                            resolved = sandbox.read_file(value)
                            if resolved and resolved.strip() \
                                    and not _looks_like_path(resolved.strip()):
                                value = resolved
                        except Exception:
                            pass  # leave value as-is; schema validation already passed

                    _collected_outputs[field] = value
                    remaining = _required_fields - set(_collected_outputs)
                    if remaining:
                        _remaining_tools = ", ".join(f"return_{f}" for f in sorted(remaining))
                        return json.dumps({"result": (
                            f"[HARNESS SYSTEM] ✓ '{field}' registered. "
                            f"Still needed: {sorted(remaining)}, call {_remaining_tools} tool(s)."
                        )})
                    return json.dumps({"result": (
                        f"[HARNESS SYSTEM] ✓ '{field}' registered. "
                        f"All required fields complete, please end your response now."
                    )})
                return _handle

            _intercept = {f"return_{f}": _make_field_handler(f) for f in _required_fields}

        _timeout_attempt = 0
        messages, n_before = self._build_initial_messages(
            input, history, _extra_system, _live_messages_fn, _full_history_fn,
        )
        output_schema_retries_left = self.max_output_schema_retries
        _compaction_summary: str | None = None
        _total_input_tokens:  int = 0
        _total_output_tokens: int = 0
        for _ in range(max_steps):
            kwargs: dict = _build_llm_kwargs(llm_config, messages, openai_tools)

            # Drain user inbox before firing — appended as user turns mid-conversation
            had_inbox = _drain_inbox(messages, _inbox_fn, _live_messages_fn, _full_history_fn)

            messages, _compaction_summary = _maybe_compact(
                messages, llm_config, _context_limit, None,
                _compaction_summary, term, _compact_log_fn, _live_messages_fn, self.name,
            )

            # Pre-call prompt size estimate (chars / 4).  Used for UI token display
            # and max_tokens clamping; precise accuracy is not required here.
            _pre_estimate = estimate_messages_tokens(messages)

            if _token_update_fn is not None:
                _token_update_fn(_total_input_tokens + _pre_estimate, _total_output_tokens)

            if _context_limit is not None:
                _headroom = max(1, _context_limit - _pre_estimate)
                if kwargs.get("max_tokens", _headroom) > _headroom:
                    kwargs = dict(kwargs)
                    kwargs["max_tokens"] = _headroom

            # --- Streaming call ----------------------------------------------
            llm_result = _llm_call(
                kwargs, llm_config, messages, _timeout_attempt,
                term, _state_fn, _live_messages_fn, _token_update_fn,
                _total_input_tokens, _total_output_tokens, self.name,
            )
            if llm_result.context_exceeded:
                if _full_history_fn:
                    _full_history_fn({"type": "llm_context_exceeded"})
                messages, _compaction_summary = _maybe_compact(
                    messages, llm_config, _context_limit, None,
                    _compaction_summary, term, _compact_log_fn, _live_messages_fn, self.name,
                    force=True,
                )
                continue
            if llm_result.should_retry:
                _timeout_attempt = llm_result.next_timeout_attempt
                if _full_history_fn:
                    _full_history_fn({"type": "llm_retry",
                                      "error": str(llm_result.conn_error),
                                      "attempt": _timeout_attempt})
                # Sleep before reconnecting: closing the client while the drain
                # thread is mid-ssl.read() corrupts process-wide OpenSSL state.
                # A 2 s gap lets the drain thread exit and the server's SSL
                # teardown complete before the next connection is attempted.
                # (Reproduced: 0 s → 3/5 SSL failures; 1 s+ → 5/5 clean.)
                time.sleep(LLM_RETRY_SLEEP_S)
                continue
            if not llm_result.ok:
                _err_msg = f"LLM connection error after 5 attempts: {llm_result.conn_error}"
                if _full_history_fn:
                    _full_history_fn({"type": "llm_error", "error": _err_msg})
                return agerror(_err_msg), history, [], (0, 0)
            _timeout_attempt = 0
            _total_input_tokens  = llm_result.total_input_tokens
            _total_output_tokens = llm_result.total_output_tokens
            if term:
                _ctx_str = f"/{_context_limit}" if _context_limit else ""
                _tok_str = f"  tokens={llm_result.prompt_tokens}{_ctx_str}" if llm_result.prompt_tokens else ""
                term.log("LLM ✓    ", f"model={llm_config.get('model','?')}  ({llm_result.elapsed_ms}ms){_tok_str}")
            if _state_fn:
                _state_fn("skill", skill=self.name)

            messages, _compaction_summary = _maybe_compact(
                messages, llm_config, _context_limit, llm_result.prompt_tokens,
                _compaction_summary, term, _compact_log_fn, _live_messages_fn, self.name,
            )

            # Build final assistant message dict from accumulated stream
            msg_dict: dict = _build_assistant_msg(llm_result.content_parts, llm_result.reasoning_parts, llm_result.tool_calls_raw)
            messages.append(msg_dict)
            if _live_messages_fn:
                _live_messages_fn(messages[1:])
            if _full_history_fn:
                _full_history_fn(msg_dict)

            if msg_dict.get("tool_calls"):
                _read_injected = _dispatch_tools(
                    msg_dict["tool_calls"], tool_map, messages, sandbox, self.name,
                    _state_fn, _live_messages_fn, _full_history_fn, term,
                    _intercept=_intercept,
                    tool_offload_chars=(
                        max(_TOOL_OUTPUT_OFFLOAD_CHARS, int(_context_limit * 0.1 * 4))
                        if _context_limit else _TOOL_OUTPUT_OFFLOAD_CHARS
                    ),
                )
                if _read_injected:
                    openai_tools = (openai_tools or []) + [tool_map["read"].to_openai_tool()]
                # If all required output fields are now registered, skip the next
                # LLM call and fall through directly to output validation and return.
                # Otherwise keep looping so the model can do more work.
                if not (_use_return_output and not (_required_fields - set(_collected_outputs))):
                    continue

            else:
                # Model stopped without tool calls.  If it was mid-inbox-exchange,
                # let the conversation complete before validating output.
                if had_inbox:
                    continue

            # ── Output-ready path ─────────────────────────────────────────────
            # Reached either when:
            #   (a) the model stopped naturally with no tool calls, or
            #   (b) all required return_<field> calls completed in the last batch.

            if _use_return_output:
                # Tool-based output path: check that all required fields were
                # registered via return_<field> before reaching here.
                missing = _required_fields - set(_collected_outputs)
                if missing:
                    if output_schema_retries_left > 0:
                        output_schema_retries_left -= 1
                        _missing_tools = " and ".join(
                            f"return_{f}" for f in sorted(missing)
                        )
                        reprompt = {
                            "role": "user",
                            "content": (
                                f"[HARNESS SYSTEM] You have not yet provided all required output fields. "
                                f"Still missing: {sorted(missing)}. "
                                f"Call {_missing_tools} tool(s) for each missing field."
                            ),
                        }
                        messages.append(reprompt)
                        if _live_messages_fn:
                            _live_messages_fn(messages[1:])
                        if _full_history_fn:
                            _full_history_fn(reprompt)
                        continue
                    _last_asst = next(
                        (m for m in reversed(messages) if m.get("role") == "assistant"), None
                    )
                    _last_out_str = ""
                    if _last_asst:
                        if _last_asst.get("content"):
                            _last_out_str = str(_last_asst["content"])[:2000]
                        elif _last_asst.get("tool_calls"):
                            _names = [
                                tc.get("function", {}).get("name", "?")
                                for tc in _last_asst["tool_calls"]
                            ]
                            _last_out_str = f"[tool calls: {_names}]"
                    updated_history = agdata(messages=messages[1:])
                    return (
                        agerror(
                            f"output schema error: missing fields after retries: {sorted(missing)}"
                            + f"\ncollected: {sorted(_collected_outputs.keys())}"
                            + (f"\nlast model output: {_last_out_str!r}" if _last_out_str else "")
                        ),
                        updated_history,
                        [messages[0]] + messages[1:][n_before:],
                        (_total_input_tokens, _total_output_tokens),
                    )
                # All fields collected — run optional validator.
                result = agdata(**_collected_outputs)
                proc_msg = _wait_for_processes(
                    sandbox, self.name, term, log, _agname,
                    _ping_interval_s, _poll_interval_s, _state_fn,
                )
                if proc_msg is not None:
                    messages.append({"role": "user", "content": proc_msg})
                    if _live_messages_fn:
                        _live_messages_fn(messages[1:])
                    if _full_history_fn:
                        _full_history_fn(messages[-1])
                    continue
                updated_history = agdata(messages=messages[1:])
                return (
                    result,
                    updated_history,
                    [messages[0]] + messages[1:][n_before:],
                    (_total_input_tokens, _total_output_tokens),
                )

            far = self._parse_final_answer(
                msg_dict, messages, n_before, output_schema_retries_left,
                (_total_input_tokens, _total_output_tokens),
            )
            if far.kind == "retry":
                output_schema_retries_left -= 1
                messages.append(far.correction_msg)
                continue
            if far.kind != "error":
                proc_msg = _wait_for_processes(
                    sandbox, self.name, term, log, _agname,
                    _ping_interval_s, _poll_interval_s, _state_fn,
                )
                if proc_msg is not None:
                    messages.append({"role": "user", "content": proc_msg})
                    if _live_messages_fn:
                        _live_messages_fn(messages[1:])
                    if _full_history_fn:
                        _full_history_fn(messages[-1])
                    continue
            return far.return_tuple

        updated_history = agdata(messages=messages[1:])
        return agerror("max_steps exceeded"), updated_history, [messages[0]] + messages[1:][n_before:], (_total_input_tokens, _total_output_tokens)

    def __repr__(self) -> str:
        return f"agskill(name={self.name!r})"
