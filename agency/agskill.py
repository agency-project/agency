from __future__ import annotations
import json
import queue
import re
import ssl
import threading
import time
from typing import TYPE_CHECKING, Callable, Generator, Iterable, TypeVar
import httpx
import openai


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
_TOOL_OUTPUT_OFFLOAD_CHARS: int = 80_000  # tool results longer than this are saved to a file


class _LLMIdleTimeout(Exception):
    """Raised by _iter_batched when no chunk arrives within idle_timeout seconds."""


def _iter_batched(
    iterable: Iterable[_T],
    idle_timeout: float | None = None,
) -> Generator[list[_T], None, None]:
    """Drain *iterable* in a background thread; yield batches to the caller.

    The background thread does minimal Python per item (one queue.put).
    The calling thread sleeps for _BATCH_INTERVAL_S between drains, releasing
    the GIL for that entire interval so other threads run unimpeded.
    GIL acquisitions drop from O(items) to O(items / avg_batch_size).

    If *idle_timeout* is given, raises _LLMIdleTimeout when no item arrives
    for that many seconds.  The drain thread is left as a daemon (it will die
    when the process exits or when the underlying socket is eventually closed).
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

    while True:
        # Block until the first item of the next batch arrives (GIL released).
        # When idle_timeout is set, poll every _IDLE_CHECK_INTERVAL_S so we
        # can detect dead connections (e.g. CLOSE-WAIT / stuck ssl.read()).
        try:
            if idle_timeout is not None:
                item = q.get(timeout=_IDLE_CHECK_INTERVAL_S)
            else:
                item = q.get()
        except queue.Empty:
            if time.monotonic() - _last_item >= idle_timeout:  # type: ignore[operator]
                raise _LLMIdleTimeout(f"no chunk received for {idle_timeout:.0f}s")
            continue

        _last_item = time.monotonic()

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
from typing import get_args, get_origin
from .agdata import agdata, _fmt_exc
from .agtype import agtype, agimage, agrawstring
from .agtool import agtool
from .agcompaction import compact, should_compact, count_messages_tokens

if TYPE_CHECKING:
    from .agterm import agterm
    from .aglog import aglog
    from .agsandbox import agSandbox
    from .agresources import agResourcePool

_skill_semaphore = threading.Semaphore(128)

class agskill:
    """A named skill with its own system prompt and a self-contained ReAct loop.

    input_schema / output_schema are agdata objects whose keys define required
    fields and whose values are Python types (``str``, ``int``, ``float``,
    ``bool``, ``list``, ``dict``, or an ``agtype`` subclass such as ``agfile``).
    Both schemas are serialised and appended to the system prompt so the LLM
    knows the contract.

    Input is validated before the loop runs.  Output is validated after each
    final (non-tool-call) LLM response; on failure a correction message is
    injected and the loop retries up to max_retries times.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        add_tools: list[agtool] | None = None,
        replace_tools: list[agtool] | None = None,
        input_schema: agdata | None = None,
        output_schema: agdata | None = None,
        output_validator: "Callable[[agdata], list[str]] | None" = None,
        max_retries: int = 10,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.add_tools = add_tools
        self.replace_tools = replace_tools
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.output_validator = output_validator   # extra check beyond type schema
        self.max_retries = max_retries

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
                parts.append(
                    f"\nOutput JSON format (respond ONLY with this JSON, no other text):\n"
                    f"{self.output_schema.to_json()}\n\n"
                    "JSON formatting rules — common failure modes to avoid:\n"
                    "- Do NOT wrap the JSON in markdown code fences (``` or ```json).\n"
                    "- Do NOT add any explanation, preamble, or trailing text outside the JSON object.\n"
                    "- All string values must use double quotes. Escape special characters inside strings:\n"
                    '  use \\" for a literal quote, \\\\ for a backslash, \\n for a newline.\n'
                    "  Never use raw newlines or unescaped quotes inside a string value.\n"
                    "- Every key must be a double-quoted string. Trailing commas are not allowed.\n"
                    "- The response must be a single JSON object {{ }} — not an array, not multiple objects.\n"
                    "- Include every required field exactly once. Do not nest the output inside an extra wrapper key."
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
            return input.to_json()

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
        content: list = [{"type": "text", "text": text}]
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
        max_steps: int = 100,
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
    ) -> tuple[agdata, agdata, list[dict], tuple[int, int]]:
        """Run the ReAct loop.

        Returns (result_agdata, updated_history_agdata, history_delta).
        history_delta is the list of new messages added during this skill's
        execution (user input → tool calls / results → final answer).
        The system_prompt (+ schemas) is prepended to every call but is NOT
        persisted in history.

        When *_is_continuation* is True (outer monitoring loop re-entry), input
        schema validation is skipped so process-status ping messages can flow
        through without matching the skill's declared input schema.
        """
        # --- Input validation ------------------------------------------------
        if self.input_schema is not None and not _is_continuation:
            errors = self._check_schema(input, self.input_schema)
            if errors:
                sys_msg = {"role": "system", "content": self._build_system_prompt(_extra_system)}
                return agdata(error=f"input schema error: {errors}"), history, [sys_msg], (0, 0)

        from .tools import make_sandboxed_tools
        if self.replace_tools is not None:
            active_tools: list[agtool] = list(self.replace_tools)
        elif sandbox is not None:
            active_tools = make_sandboxed_tools(sandbox, pool)
            if self.add_tools:
                active_tools.extend(self.add_tools)
        else:
            active_tools = list(self.add_tools or [])
        for t in active_tools:
            t.attach_logger(term, log)
        tool_map = {t.name: t for t in active_tools}
        openai_tools = [t.to_openai_tool() for t in active_tools] or None

        # Exponential backoff timeouts for LLM calls (seconds): 1,2,4,8,16 min
        _TIMEOUT_SEQUENCE = [60, 120, 240, 480, 960]
        _timeout_attempt = 0

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
            _full_history_fn(messages[0])          # system prompt
            _full_history_fn(messages[-1])         # user input

        retries_left = self.max_retries
        _compaction_summary: str | None = None
        _total_input_tokens:  int = 0
        _total_output_tokens: int = 0

        for _ in range(max_steps):
            had_inbox = False
            kwargs: dict = dict(
                model=llm_config.get("model", "gpt-4o"),
                # Strip private (_-prefixed) keys before sending to the API.
                # _thinking and similar fields are for internal/logging use only.
                messages=[{k: v for k, v in m.items() if not k.startswith("_")}
                          for m in messages],
            )
            # Forward standard OpenAI generation parameters from llm_config.
            _OPENAI_GEN_PARAMS = {"temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "n", "stop", "logprobs", "seed"}
            for _p in _OPENAI_GEN_PARAMS:
                if _p in llm_config:
                    kwargs[_p] = llm_config[_p]
            # Merge vLLM-specific parameters (not in the OpenAI spec) into extra_body.
            _EXTRA_BODY_GEN_PARAMS = {"top_k", "repetition_penalty", "min_p", "min_tokens", "guided_json", "guided_regex"}
            _extra_body: dict = dict(llm_config.get("extra_body") or {})
            for _p in _EXTRA_BODY_GEN_PARAMS:
                if _p in llm_config:
                    _extra_body[_p] = llm_config[_p]
            if _extra_body:
                kwargs["extra_body"] = _extra_body
            if openai_tools:
                kwargs["tools"] = openai_tools

            # Drain user inbox before firing — appended as user turns mid-conversation
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

            # Pre-call compaction: compact before the API call so an already-
            # oversized context doesn't cause the request to fail outright.
            # Uses a character-based token estimate because we haven't heard
            # back from the API yet.
            if _context_limit is not None:
                estimated = count_messages_tokens(messages, llm_config)
                if should_compact(estimated, _context_limit):
                    if term:
                        term.log(
                            "COMPACT  ",
                            f"skill={self.name}  "
                            f"tokens~{estimated}/{_context_limit}  "
                            f"msgs={len(messages)}  (pre-call estimate)",
                        )
                    msgs_before = len(messages)
                    messages, _compaction_summary = compact(
                        messages, llm_config,
                        context_limit=_context_limit,
                        previous_summary=_compaction_summary,
                    )
                    if _compact_log_fn:
                        _compact_log_fn(
                            skill=self.name,
                            prompt_tokens=estimated,
                            context_limit=_context_limit,
                            msgs_before=msgs_before,
                            msgs_after=len(messages),
                        )
                    if _live_messages_fn:
                        _live_messages_fn(messages[1:])

            read_timeout = _TIMEOUT_SEQUENCE[min(_timeout_attempt, len(_TIMEOUT_SEQUENCE) - 1)]
            _skill_semaphore.acquire()
            client = _make_llm_client(
                llm_config,
                httpx.Timeout(connect=30.0, read=None, write=180.0, pool=30.0),
            )

            if term:
                term.log("LLM ▶    ", f"model={llm_config.get('model','?')}  messages={len(messages)}  timeout={read_timeout}s")
            if _state_fn:
                _state_fn("llm", skill=self.name)

            # --- Streaming call ----------------------------------------------
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}

            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls_raw: dict[int, dict] = {}
            prompt_tokens: int | None = None
            _llm_t0 = time.monotonic()

            # Partial placeholder so live UI shows tokens as they arrive
            partial_msg: dict = {"role": "assistant", "content": ""}
            messages.append(partial_msg)
            # Push immediately so the placeholder appears before any tokens arrive
            if _live_messages_fn:
                _live_messages_fn(messages[1:])
            _live_chars = 0

            _PARTIAL_THINK_RE = re.compile(
                r"<think(?:ing)?>(.*?)(?:</think(?:ing)?>|$)", re.DOTALL | re.IGNORECASE
            )

            try:
                for batch in _iter_batched(client.chat.completions.create(**kwargs), idle_timeout=float(read_timeout)):
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
                        if _live_messages_fn and new_chars - _live_chars >= 100:
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

            except (_LLMIdleTimeout, ssl.SSLError, OSError, httpx.TransportError) as _conn_err:
                try:
                    client.close()  # best-effort: unblock drain thread's ssl.read()
                except Exception:
                    pass
                messages.pop()  # remove partial placeholder
                _llm_elapsed_ms = int((time.monotonic() - _llm_t0) * 1000)
                _skill_semaphore.release()
                _err_desc = (
                    f"timeout after {read_timeout}s"
                    if isinstance(_conn_err, _LLMIdleTimeout)
                    else f"connection error: {_conn_err}"
                )
                if _timeout_attempt < len(_TIMEOUT_SEQUENCE) - 1:
                    _timeout_attempt += 1
                    next_timeout = _TIMEOUT_SEQUENCE[_timeout_attempt]
                    if term:
                        term.log("LLM ✗    ", f"model={llm_config.get('model','?')}  {_err_desc}  retrying with {next_timeout}s")
                    continue
                if term:
                    term.log("LLM ✗    ", f"model={llm_config.get('model','?')}  {_err_desc}  all retries exhausted")
                return agdata(error=f"LLM connection error after {len(_TIMEOUT_SEQUENCE)} attempts: {_conn_err}"), history, [], (0, 0)

            messages.pop()  # remove partial placeholder
            _llm_elapsed_ms = int((time.monotonic() - _llm_t0) * 1000)
            _skill_semaphore.release()
            if term:
                term.log("LLM ✓    ", f"model={llm_config.get('model','?')}  ({_llm_elapsed_ms}ms)")
            if _token_update_fn is not None:
                try:
                    _token_update_fn(_total_input_tokens, _total_output_tokens)
                except Exception:
                    pass

            if _state_fn:
                _state_fn("skill", skill=self.name)

            # Auto-compaction: if we're burning through context, summarise old
            # messages now so the next iteration has headroom.
            if _context_limit is not None and prompt_tokens is not None:
                if should_compact(prompt_tokens, _context_limit):
                    if term:
                        term.log(
                            "COMPACT  ",
                            f"skill={self.name}  "
                            f"tokens={prompt_tokens}/{_context_limit}  "
                            f"msgs={len(messages)}",
                        )
                    msgs_before = len(messages)
                    messages, _compaction_summary = compact(
                        messages, llm_config,
                        context_limit=_context_limit,
                        previous_summary=_compaction_summary,
                    )
                    if _compact_log_fn:
                        _compact_log_fn(
                            skill=self.name,
                            prompt_tokens=prompt_tokens,
                            context_limit=_context_limit,
                            msgs_before=msgs_before,
                            msgs_after=len(messages),
                        )
                    if _live_messages_fn:
                        _live_messages_fn(messages[1:])

            # Build final assistant message dict from accumulated stream
            full_content = "".join(content_parts)
            full_reasoning = "".join(reasoning_parts)
            msg_dict: dict = {"role": "assistant"}
            if full_reasoning:
                # vLLM / DeepSeek-R1: thinking arrives in reasoning_content
                msg_dict["_thinking"] = full_reasoning
                if full_content:
                    msg_dict["content"] = full_content
            elif full_content:
                # <think>-tag models: thinking is embedded in content
                thinking = _extract_thinking(full_content)
                if thinking:
                    msg_dict["_thinking"] = thinking
                msg_dict["content"] = _strip_thinking(full_content)
            if tool_calls_raw:
                msg_dict["tool_calls"] = [tool_calls_raw[i] for i in sorted(tool_calls_raw)]
            messages.append(msg_dict)
            if _live_messages_fn:
                _live_messages_fn(messages[1:])
            if _full_history_fn:
                _full_history_fn(msg_dict)

            if msg_dict.get("tool_calls"):
                for tc in msg_dict["tool_calls"]:
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
                    t = tool_map.get(fn_name)
                    if t is None:
                        if term:
                            term.log("TOOL ✗   ", f"{fn_name}  → unknown tool")
                        result_content = json.dumps({"error": f"unknown tool: {fn_name}"})
                    else:
                        try:
                            if _state_fn:
                                _state_fn("tool", skill=self.name, tool=fn_name)
                            _ckpt_tag: str | None = None
                            if t.need_sandbox and sandbox is not None:
                                _ckpt_tag = f"agency/pretool-{sandbox._name}-{tc_id.replace('-','')[:8]}"
                                try:
                                    if not sandbox.commit(_ckpt_tag):
                                        _ckpt_tag = None  # nothing was committed, don't try to restore
                                except Exception:
                                    _ckpt_tag = None
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
                                _state_fn("skill", skill=self.name)
                        except Exception as e:
                            if _state_fn:
                                _state_fn("skill", skill=self.name)
                            result_content = json.dumps({"error": _fmt_exc(e)})
                        # On tool failure, revert the sandbox to the pre-call checkpoint
                        # and tell the agent the workspace was restored.
                        try:
                            _result_obj = json.loads(result_content)
                            if "error" in _result_obj and _ckpt_tag and sandbox is not None:
                                try:
                                    sandbox.restore(_ckpt_tag)
                                    _result_obj["workspace_reverted"] = (
                                        "The workspace has been reverted to the state "
                                        "before this tool call."
                                    )
                                    result_content = json.dumps(_result_obj)
                                except Exception:
                                    pass
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if sandbox is not None and len(result_content) > _TOOL_OUTPUT_OFFLOAD_CHARS:
                        safe_id = tc_id.replace("-", "")[:12]
                        offload_path = f"/workspace/long_tool_call_outputs/{fn_name}_{safe_id}.txt"
                        try:
                            # Write the plain `content` field so the file has natural
                            # line breaks and the read tool can paginate with offsets.
                            # Fall back to the raw JSON string if parsing fails.
                            try:
                                file_body = json.loads(result_content).get("content", result_content)
                            except (json.JSONDecodeError, AttributeError):
                                file_body = result_content
                            sandbox.write_file(offload_path, file_body)
                            result_content = json.dumps({
                                "note": f"Output was too large and has been saved to {offload_path}. Use the read tool to access it."
                            })
                        except Exception:
                            pass
                    tool_msg = {"role": "tool", "tool_call_id": tc_id, "content": result_content}
                    messages.append(tool_msg)
                    if _live_messages_fn:
                        _live_messages_fn(messages[1:])
                    if _full_history_fn:
                        _full_history_fn(tool_msg)

            else:
                # If this step consumed inbox messages, the LLM is mid-conversation
                # with the user — not producing a final answer yet. Continue the
                # loop so the exchange can complete before output validation runs.
                if had_inbox:
                    continue

                # --- Parse final answer --------------------------------------
                # full_content is already thinking-stripped via msg_dict["content"];
                # use msg_dict.get() so we don't re-process.

                # agrawstring output — capture the raw text and return immediately.
                raw_out_key = self._raw_output_key()
                if raw_out_key is not None:
                    raw_content = msg_dict.get("content") or ""
                    updated_history = agdata(messages=messages[1:])
                    return (
                        agdata(**{raw_out_key: raw_content}),
                        updated_history,
                        [messages[0]] + messages[1:][n_before:],
                    )

                content = msg_dict.get("content") or "{}"
                # Strip markdown code fences that some models add despite instructions
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
                    # If there is trailing garbage after a valid JSON object
                    # (e.g. a stray `"` the model appended), try extracting
                    # just the first complete JSON value via raw_decode.
                    if "Extra data" in str(_e):
                        try:
                            _obj, _ = json.JSONDecoder().raw_decode(content.lstrip())
                            result = agdata(**_obj) if isinstance(_obj, dict) else agdata(result=content)
                            _parse_error = ""  # recovered
                        except (json.JSONDecodeError, TypeError):
                            _parse_error = str(_e)
                            result = agdata(result=content)
                    else:
                        _parse_error = str(_e)
                        result = agdata(result=content)
                except TypeError as _e:
                    _parse_error = str(_e)
                    result = agdata(result=content)

                # --- Output validation + retry --------------------------------
                if self.output_schema is not None:
                    errors = self._check_schema(result, self.output_schema)
                    if not errors and self.output_validator is not None:
                        errors = self.output_validator(result)
                    if _parse_error and "invalid json" not in " ".join(errors).lower():
                        errors = [f"JSON parse error: {_parse_error}"] + errors
                    if errors:
                        if retries_left > 0:
                            retries_left -= 1
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"Your previous response could not be parsed. Errors: {errors}.\n"
                                    "Common causes: unescaped quotes or backslashes inside a string value, "
                                    "raw newlines inside a string (use \\n instead), trailing comma after "
                                    "the last key, extra text or explanation outside the JSON object, "
                                    "or markdown code fences around the JSON.\n"
                                    "Respond ONLY with a single valid JSON object matching exactly: "
                                    f"{self.output_schema.to_json()}"
                                ),
                            })
                            continue  # retry in the same loop
                        updated_history = agdata(messages=messages[1:])
                        return (
                            agdata(error=f"output schema error after retries: {errors}"),
                            updated_history,
                            [messages[0]] + messages[1:][n_before:],
                            (_total_input_tokens, _total_output_tokens),
                        )

                updated_history = agdata(messages=messages[1:])
                return result, updated_history, [messages[0]] + messages[1:][n_before:], (_total_input_tokens, _total_output_tokens)

        updated_history = agdata(messages=messages[1:])
        return agdata(error="max_steps exceeded"), updated_history, [messages[0]] + messages[1:][n_before:], (_total_input_tokens, _total_output_tokens)

    def __repr__(self) -> str:
        return f"agskill(name={self.name!r})"
