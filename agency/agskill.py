from __future__ import annotations
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, get_args, get_origin
from .agdata import agdata, agerror
from .agtype import agimage, agtype, output_field_desc, raw_schema_key, make_field_handler
from .agtool import agtool, make_return_output_tools, dispatch_tools, TOOL_OUTPUT_OFFLOAD_CHARS
from .agllm import LLM_RETRY_SLEEP_S, agllm
from .agcompaction import estimate_messages_tokens, maybe_compact
from .agsandbox import agSandbox

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGSKILL_REACT_MAX_STEPS        = 4096
AGBINARY_VALIDATE_EXEC_TIMEOUT = 5  # Seconds per container exec call when validating agbinary output.

@dataclass
class _FinalAnswerResult:
    kind: str  # "return" | "retry" | "error"
    return_tuple: "tuple | None" = None
    correction_msg: "dict | None" = None


def parse_final_answer(
    msg_dict: dict,
    messages: list[dict],
    n_before: int,
    output_schema_retries_left: int,
    tokens: "tuple[int, int]",
    output_schema: "agdata | None",
) -> _FinalAnswerResult:
    """Parse the LLM's final (non-tool-call) response.

    Returns a _FinalAnswerResult with kind:
      "return"  — caller should return return_tuple immediately
      "retry"   — caller should append correction_msg and continue the loop
      "error"   — caller should return an error return_tuple
    """
    raw_out_key = raw_schema_key(output_schema)
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

    if output_schema is not None:
        errors = result.check_schema(output_schema)
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
                            f"{output_schema.to_json()}"
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

if TYPE_CHECKING:
    from .agterm import agterm
    from .aglog import aglog
    from .agresources import agResourcePool


# ---------------------------------------------------------------------------
# ReAct loop helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Skill class
# ---------------------------------------------------------------------------

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

    def _build_system_prompt(self, extra: str | None = None) -> str:
        parts = [self.system_prompt]

        # Collect agtype fields with extra prompt instructions and emit
        # them before the JSON format sections.
        extra_lines: list[str] = []
        for key, hint in (self.input_schema._data.items() if self.input_schema else []):
            cls = agtype.from_hint(hint)
            if cls is not None:
                line = cls.extra_input_prompt(key)
                if line:
                    extra_lines.append(line)
        for key, hint in (self.output_schema._data.items() if self.output_schema else []):
            cls = agtype.from_hint(hint)
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

        if self.input_schema is not None and raw_schema_key(self.input_schema) is None:
            parts.append(f"\nInput JSON format:\n{self.input_schema.to_json()}")
        if self.output_schema is not None:
            if raw_schema_key(self.output_schema) is not None:
                parts.append("\nRespond with plain text only — no JSON wrapping, no markdown code fences.")
            else:
                field_tools = ", ".join(
                    f"return_{f}" for f in self.output_schema._data
                )
                field_lines = "\n".join(
                    f"  - {f}: {output_field_desc(h)}"
                    for f, h in self.output_schema._data.items()
                )
                parts.append(
                    f"\nTo return your results, call the appropriate return_<field> tool "
                    f"once for each required output field ({field_tools}). "
                    f"Required fields:\n"
                    f"{field_lines}\n\n"
                    "- Call each return_<field> tool separately — one field per call.\n"
                    "- Only call a return_<field> tool when you have the final value ready — "
                    "return the output itself as the tool argument. "
                    "Never call a return_<field> tool with empty or missing arguments.\n"
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
        raw_key = raw_schema_key(self.input_schema)
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

        text = json.dumps(text_data)
        content: list = [{"type": "text", "text": f"New Skill Input:\n{text}"}]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        return content

    def _build_tools(
        self,
        sandbox: "agSandbox",
        pool: "agResourcePool | None",
        term: "agterm | None",
        log: "aglog | None",
        _ensure_read: bool = False,
    ) -> "tuple[list, dict, list | None]":
        """Build and return (active_tools, tool_map, openai_tools)."""
        if self.replace_tools is not None:
            active_tools: list[agtool] = list(self.replace_tools)
        else:
            from .tools import make_sandboxed_tools, make_read
            active_tools = make_sandboxed_tools(sandbox, pool)
            if self.add_tools:
                active_tools.extend(self.add_tools)
        # If input fields were offloaded to sandbox files, ensure the read tool
        # is available even for skills that use replace_tools without it.
        if _ensure_read:
            if not any(getattr(t, "name", None) == "read" for t in active_tools):
                from .tools import make_read
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
        llm: "agllm",
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
        _full_history_fn: "Callable[[dict], None] | None" = None,
        _token_update_fn: "Callable[[int, int], None] | None" = None,
        _ping_interval_s: float = 300,
        _poll_interval_s: float = 5,
        _agname: str = "",
    ) -> tuple[agdata, agdata, list[dict], tuple[int, int]]:
        """Run the ReAct loop, including sandbox process monitoring."""
        # ── 1. Validate input against the skill's input schema.
        input_error = input.validate_input(self.input_schema, _is_continuation)
        if input_error is not None:
            sys_msg = {"role": "system", "content": self._build_system_prompt()}
            return agerror(input_error), history, [sys_msg], (0, 0)

        # ── 2. Prepare inputs — write agtype fields (e.g. agfile content → sandbox
        #    path) and offload oversized plain strings to temporary sandbox files.
        #    A timestamp suffix makes paths unique across back-to-back skill calls.
        _offloaded_paths: list[str] = []
        _input_suffix = f"_{int(time.time() * 1000)}"
        _offloaded_paths.extend(
            input.prepare_agtype_inputs(self.input_schema, sandbox, self.name, suffix=_input_suffix)
        )
        auto_paths, auto_fields = input.offload_large_fields(
            sandbox, self.name, schema=self.input_schema,
            suffix=_input_suffix, context_limit=llm.context_limit,
        )
        _offloaded_paths.extend(auto_paths)

        # Warn the model about auto-offloaded fields so it knows to read them.
        _extra_system: str | None = None
        if auto_fields:
            field_list = ", ".join(f"`{f}`" for f in auto_fields)
            _extra_system = (
                f"\nNote: The following input fields contain large content "
                f"that has been automatically saved to temporary files in "
                f"your sandbox: {field_list}. The file paths are shown in "
                f"the input JSON. Use the read tool to access the full "
                f"content. WARNING: these files are temporary and will be "
                f"automatically deleted after this task ends."
            )

        # ── 3. Build tool set — sandbox tools + read (if offloaded inputs exist).
        _active_tools, tool_map, openai_tools = self._build_tools(
            sandbox, pool, term, log, _ensure_read=bool(_offloaded_paths)
        )

        # ── 4. Set up structured output collection via return_<field> tools.
        #    Skipped for agrawstring schemas, which capture raw text directly.
        _use_return_output = (
            self.output_schema is not None and raw_schema_key(self.output_schema) is None
        )
        _collected_outputs: dict = {}
        _required_fields: set[str] = set()
        _intercept: "dict[str, Callable[[dict], str]] | None" = None
        if _use_return_output:
            _required_fields = set(self.output_schema._data.keys())
            _return_tools = make_return_output_tools(self.output_schema)
            openai_tools = _return_tools + (openai_tools or [])

            _intercept = {
                f"return_{f}": make_field_handler(
                    f, self.output_schema, sandbox,
                    _collected_outputs, _required_fields,
                    AGBINARY_VALIDATE_EXEC_TIMEOUT,
                )
                for f in _required_fields
            }

        # ── 5. Build the initial message list (system prompt + history + user turn).
        _timeout_attempt = 0
        messages, n_before = self._build_initial_messages(
            input, history, _extra_system, _live_messages_fn, _full_history_fn,
        )
        output_schema_retries_left = self.max_output_schema_retries
        _compaction_summary: str | None = None
        _total_input_tokens:  int = 0
        _total_output_tokens: int = 0

        # ── 6. ReAct loop — each iteration is one LLM call + tool dispatch cycle.
        for _ in range(max_steps):
            kwargs: dict = llm.build_kwargs(messages, openai_tools)

            # 6a. Drain any inbox messages injected by the orchestrator mid-loop.
            had_inbox = _drain_inbox(messages, _inbox_fn, _live_messages_fn, _full_history_fn)

            # 6b. Compact history if it is approaching the context limit.
            messages, _compaction_summary = maybe_compact(
                messages, llm.config, llm.context_limit, None,
                _compaction_summary, term, log, _live_messages_fn, self.name, agname=_agname,
            )

            # 6c. Cap max_tokens so the model's reply fits within what's left.
            _pre_estimate = estimate_messages_tokens(messages)

            if _token_update_fn is not None:
                _token_update_fn(_total_input_tokens + _pre_estimate, _total_output_tokens)

            if llm.context_limit is not None:
                _headroom = max(1, llm.context_limit - _pre_estimate)
                if kwargs.get("max_tokens", _headroom) > _headroom:
                    kwargs = dict(kwargs)
                    kwargs["max_tokens"] = _headroom

            # 6d. Call the LLM and handle transient errors (retry / context exceeded).
            llm_result = llm.call(
                kwargs, messages, _timeout_attempt,
                term, _state_fn, _live_messages_fn, _token_update_fn,
                _total_input_tokens, _total_output_tokens, self.name,
            )
            if llm_result.context_exceeded:
                if _full_history_fn:
                    _full_history_fn({"type": "llm_context_exceeded"})
                messages, _compaction_summary = maybe_compact(
                    messages, llm.config, llm.context_limit, None,
                    _compaction_summary, term, log, _live_messages_fn, self.name, agname=_agname,
                    force=True,
                )
                continue
            if llm_result.should_retry:
                _timeout_attempt = llm_result.next_timeout_attempt
                if _full_history_fn:
                    _full_history_fn({"type": "llm_retry",
                                      "error": str(llm_result.conn_error),
                                      "attempt": _timeout_attempt})
                time.sleep(LLM_RETRY_SLEEP_S)
                continue
            if not llm_result.ok:
                _err_msg = f"LLM connection error after 5 attempts: {llm_result.conn_error}"
                if _full_history_fn:
                    _full_history_fn({"type": "llm_error", "error": _err_msg})
                sandbox.remove_files(_offloaded_paths)
                return agerror(_err_msg), history, [], (0, 0)
            _timeout_attempt = 0
            _total_input_tokens  = llm_result.total_input_tokens
            _total_output_tokens = llm_result.total_output_tokens
            if term:
                _ctx_str = f"/{llm.context_limit}" if llm.context_limit else ""
                _tok_str = f"  tokens={llm_result.prompt_tokens}{_ctx_str}" if llm_result.prompt_tokens else ""
                term.log("LLM ✓    ", f"model={llm.config.get('model','?')}  ({llm_result.elapsed_ms}ms){_tok_str}")
            if _state_fn:
                _state_fn("skill", skill=self.name)

            # 6e. Post-response compaction: may compact again now that we know the
            #     actual prompt_tokens reported by the model.
            messages, _compaction_summary = maybe_compact(
                messages, llm.config, llm.context_limit, llm_result.prompt_tokens,
                _compaction_summary, term, log, _live_messages_fn, self.name, agname=_agname,
            )

            # 6f. Append the assistant turn to the message list.
            msg_dict: dict = agllm.build_assistant_msg(llm_result.content_parts, llm_result.reasoning_parts, llm_result.tool_calls_raw)
            messages.append(msg_dict)
            if _live_messages_fn:
                _live_messages_fn(messages[1:])
            if _full_history_fn:
                _full_history_fn(msg_dict)

            # 6g. Dispatch tool calls, or check if we can move to the output path.
            if msg_dict.get("tool_calls"):
                _read_injected = dispatch_tools(
                    msg_dict["tool_calls"], tool_map, messages, sandbox, self.name,
                    _state_fn, _live_messages_fn, _full_history_fn, term,
                    _intercept=_intercept,
                    tool_offload_chars=(
                        max(TOOL_OUTPUT_OFFLOAD_CHARS, int(llm.context_limit * 0.1 * 4))
                        if llm.context_limit else TOOL_OUTPUT_OFFLOAD_CHARS
                    ),
                )
                if _read_injected:
                    openai_tools = (openai_tools or []) + [tool_map["read"].to_openai_tool()]
                # Continue looping unless all required output fields are collected.
                if not (_use_return_output and not (_required_fields - set(_collected_outputs))):
                    continue

            else:
                # No tool calls — only continue if a mid-loop inbox message arrived,
                # which may need another LLM turn to process.
                if had_inbox:
                    continue

            # ── 7. Output-ready path — all required return_<field> calls received.
            if _use_return_output:
                missing = _required_fields - set(_collected_outputs)
                if missing:
                    # 7a. Missing fields — reprompt the model up to the retry limit.
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
                    sandbox.remove_files(_offloaded_paths)
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
                # 7b. All fields collected — wait for any background sandbox
                #     processes before returning (they may write output files).
                result = agdata(**_collected_outputs)
                proc_msg = agSandbox.wait_for_processes(
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
                # 7c. Recover agtype outputs (e.g. read file bytes back from sandbox)
                #     then clean up any auto-offloaded input files.
                updated_history = agdata(messages=messages[1:])
                result.recover_agtype_outputs(self.output_schema, sandbox)
                sandbox.remove_files(_offloaded_paths)
                return (
                    result,
                    updated_history,
                    [messages[0]] + messages[1:][n_before:],
                    (_total_input_tokens, _total_output_tokens),
                )

            # ── 8. Raw-text output path — model replied without tool calls.
            #    parse_final_answer validates the response against the output schema.
            far = parse_final_answer(
                msg_dict, messages, n_before, output_schema_retries_left,
                (_total_input_tokens, _total_output_tokens),
                self.output_schema,
            )
            if far.kind == "retry":
                # 8a. Schema mismatch — append a correction prompt and retry.
                output_schema_retries_left -= 1
                messages.append(far.correction_msg)
                continue
            if far.kind != "error":
                # 8b. Valid answer — wait for background processes before returning.
                proc_msg = agSandbox.wait_for_processes(
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
                far.return_tuple[0].recover_agtype_outputs(self.output_schema, sandbox)
            sandbox.remove_files(_offloaded_paths)
            return far.return_tuple

        # ── 9. Max steps exhausted — return an error with the accumulated history.
        updated_history = agdata(messages=messages[1:])
        sandbox.remove_files(_offloaded_paths)
        return agerror("max_steps exceeded"), updated_history, [messages[0]] + messages[1:][n_before:], (_total_input_tokens, _total_output_tokens)

    def __repr__(self) -> str:
        return f"agskill(name={self.name!r})"
