from __future__ import annotations
import json
import queue
import re
import threading
import time
from typing import TYPE_CHECKING, Callable, Generator, Iterable, TypeVar
import openai

_T = TypeVar("_T")
_BATCH_INTERVAL_S: float = 0.1   # main thread drains stream every 100 ms


def _iter_batched(iterable: Iterable[_T]) -> Generator[list[_T], None, None]:
    """Drain *iterable* in a background thread; yield batches to the caller.

    The background thread does minimal Python per item (one queue.put).
    The calling thread sleeps for _BATCH_INTERVAL_S between drains, releasing
    the GIL for that entire interval so other threads run unimpeded.
    GIL acquisitions drop from O(items) to O(items / avg_batch_size).
    """
    _SENTINEL = object()
    q: queue.SimpleQueue = queue.SimpleQueue()

    def _drain() -> None:
        try:
            for item in iterable:
                q.put(item)
        finally:
            q.put(_SENTINEL)

    threading.Thread(target=_drain, daemon=True).start()

    while True:
        # Block until the first item of the next batch arrives (GIL released).
        item = q.get()
        if item is _SENTINEL:
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
from .agdata import agdata
from .agtype import agtype
from .agtool import agtool
from .agcompaction import compact, should_compact

if TYPE_CHECKING:
    from .agterm import agterm
    from .aglog import aglog
    from .agsandbox import agSandbox
    from .agresources import agResourcePool

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
        max_retries: int = 3,
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

    def _build_system_prompt(self, extra: str | None = None) -> str:
        parts = [self.system_prompt]

        # Collect agtype fields with extra prompt instructions and emit
        # them before the JSON format sections.
        extra_lines: list[str] = []
        for key, hint in (self.input_schema._data.items() if self.input_schema else []):
            if isinstance(hint, type) and issubclass(hint, agtype):
                line = hint.extra_input_prompt(key)
                if line:
                    extra_lines.append(line)
        for key, hint in (self.output_schema._data.items() if self.output_schema else []):
            if isinstance(hint, type) and issubclass(hint, agtype):
                line = hint.extra_output_prompt(key, self.name)
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

        if self.input_schema is not None:
            parts.append(f"\nInput JSON format:\n{self.input_schema.to_json()}")
        if self.output_schema is not None:
            parts.append(
                f"\nOutput JSON format (respond ONLY with this JSON):\n"
                f"{self.output_schema.to_json()}"
            )
        return "\n".join(parts)

    def _check_schema(self, data: agdata, schema: agdata) -> list[str]:
        """Return a list of error strings; empty list means the data is valid."""
        errors: list[str] = []
        for key, hint in schema._data.items():
            if key not in data._data:
                errors.append(f"missing required field '{key}'")
                continue
            if isinstance(hint, type) and issubclass(hint, agtype):
                # agtype fields carry a string value after framework processing
                if not isinstance(data._data[key], str):
                    errors.append(
                        f"field '{key}' ({hint.__name__}) must be a string"
                    )
                continue
            if isinstance(hint, type):
                actual = data._data[key]
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
    ) -> tuple[agdata, agdata, list[dict]]:
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
                return agdata(error=f"input schema error: {errors}"), history, [sys_msg]

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

        client = openai.OpenAI(
            api_key=llm_config.get("api_key", ""),
            base_url=llm_config.get("base_url", None),
        )

        history_msgs: list[dict] = list(history._data.get("messages", []))
        n_before = len(history_msgs)
        messages: list[dict] = (
            [{"role": "system", "content": self._build_system_prompt(_extra_system)}]
            + history_msgs
            + [{"role": "user", "content": input.to_json()}]
        )
        if _live_messages_fn:
            _live_messages_fn(messages[1:])
        if _full_history_fn:
            _full_history_fn(messages[0])          # system prompt
            _full_history_fn(messages[-1])         # user input

        retries_left = self.max_retries
        _compaction_summary: str | None = None

        for _ in range(max_steps):
            had_inbox = False
            kwargs: dict = dict(
                model=llm_config.get("model", "gpt-4o"),
                # Strip private (_-prefixed) keys before sending to the API.
                # _thinking and similar fields are for internal/logging use only.
                messages=[{k: v for k, v in m.items() if not k.startswith("_")}
                          for m in messages],
            )
            if "extra_body" in llm_config:
                kwargs["extra_body"] = llm_config["extra_body"]
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

            if term:
                term.log("LLM      ", f"model={llm_config.get('model','?')}  messages={len(messages)}")
            if _state_fn:
                _state_fn("llm", skill=self.name)

            # --- Streaming call ----------------------------------------------
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}

            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls_raw: dict[int, dict] = {}
            prompt_tokens: int | None = None

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

            for batch in _iter_batched(client.chat.completions.create(**kwargs)):
                for chunk in batch:
                    if chunk.usage is not None:
                        prompt_tokens = chunk.usage.prompt_tokens
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

            messages.pop()  # remove partial placeholder

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
                    t = tool_map.get(fn_name)
                    if t is None:
                        if term:
                            term.log("TOOL ✗   ", f"{fn_name}  → unknown tool")
                        result_content = json.dumps({"error": f"unknown tool: {fn_name}"})
                    else:
                        try:
                            if _state_fn:
                                _state_fn("tool", skill=self.name, tool=fn_name)
                            result_content = t(agdata.from_json(fn_args)).to_json()
                            if _state_fn:
                                _state_fn("skill", skill=self.name)
                        except Exception as e:
                            if _state_fn:
                                _state_fn("skill", skill=self.name)
                            result_content = json.dumps({"error": str(e)})
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
                content = msg_dict.get("content") or "{}"
                # Strip markdown code fences that some models add despite instructions
                stripped = content.strip()
                if stripped.startswith("```"):
                    stripped = stripped[stripped.find("\n") + 1:] if "\n" in stripped else stripped[3:]
                    if stripped.endswith("```"):
                        stripped = stripped[:-3]
                    content = stripped.strip()
                try:
                    result = agdata.from_json(content)
                except (json.JSONDecodeError, TypeError):
                    result = agdata(result=content)

                # --- Output validation + retry --------------------------------
                if self.output_schema is not None:
                    errors = self._check_schema(result, self.output_schema)
                    if not errors and self.output_validator is not None:
                        errors = self.output_validator(result)
                    if errors:
                        if retries_left > 0:
                            retries_left -= 1
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"Output schema errors: {errors}. "
                                    f"Respond ONLY with valid JSON matching exactly: "
                                    f"{self.output_schema.to_json()}"
                                ),
                            })
                            continue  # retry in the same loop
                        updated_history = agdata(messages=messages[1:])
                        return (
                            agdata(error=f"output schema error after retries: {errors}"),
                            updated_history,
                            [messages[0]] + messages[1:][n_before:],
                        )

                updated_history = agdata(messages=messages[1:])
                return result, updated_history, [messages[0]] + messages[1:][n_before:]

        updated_history = agdata(messages=messages[1:])
        return agdata(error="max_steps exceeded"), updated_history, [messages[0]] + messages[1:][n_before:]

    def __repr__(self) -> str:
        return f"agskill(name={self.name!r})"
