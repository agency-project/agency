from __future__ import annotations
import json
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from .agdata import agdata, agerror
from .agtype import agtype
from .agschema import agschema
from .agcontext import agcontext
from .agtool import agtool, dispatch_tools, TOOL_OUTPUT_OFFLOAD_CHARS
from .agllm import agllm
from .agsandbox import agSandbox
from .agutil import format_exception
from .aglog import _ts

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGSKILL_REACT_MAX_STEPS        = 4096
AGBINARY_VALIDATE_EXEC_TIMEOUT = 5  # Seconds per container exec call when validating agbinary output.

if TYPE_CHECKING:
    from .agent import agent
    from .agterm import agterm
    from .aglog import aglog
    from .agresources import agResourcePool


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
        self.input_schema  = agschema(input_schema)  if input_schema  else None
        self.output_schema = agschema(output_schema) if output_schema else None
        self.max_output_schema_retries = max_output_schema_retries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, extra: str | None = None) -> str:
        parts = [self.system_prompt]

        # Each agtype subclass (agfile, agbinary, …) can inject extra prompt
        # lines describing how the LLM should handle that field (e.g. file paths,
        # binary encoding).  Collect these for both input and output schemas.
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

        # Emit the agtype instructions as a single block before the format sections.
        if extra_lines:
            parts.append(
                "\nFile-backed fields — WARNING: these files are temporary and will "
                "be automatically deleted after this task ends:\n"
                + "\n".join(extra_lines)
            )

        # Caller-supplied extra prompt (e.g. compaction summary injection).
        if extra:
            parts.append(extra)

        # Describe the input shape so the LLM knows what JSON keys to expect.
        # Skipped for agrawstring inputs (the value arrives as plain text, not JSON).
        if self.input_schema is not None and self.input_schema.raw_key() is None:
            parts.append(f"\nInput JSON format:\n{self.input_schema.to_json()}")

        if self.output_schema is not None:
            if self.output_schema.raw_key() is not None:
                # agrawstring output — model must reply with plain text, not a tool call.
                parts.append("\nRespond with plain text only — no JSON wrapping, no markdown code fences.")
            else:
                # Structured output — model must call one return_<field> tool per output field.
                field_tools = ", ".join(
                    f"return_{f}" for f in self.output_schema._data
                )
                field_lines = "\n".join(
                    f"  - {f}: {self.output_schema.field_desc(f)}"
                    for f in self.output_schema._data
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

    def _build_user_content(self, skill_input: agdata) -> "str | list":
        """Build the content value for the user message.

        Returns a plain string for simple inputs, or a multimodal content array
        when any agtype field contributes extra content blocks (e.g. agimage).
        Each agtype subclass declares its contribution via build_content_prompt.
        """
        # agrawstring input — send the value directly, no JSON wrapping.
        raw_key = self.input_schema.raw_key() if self.input_schema is not None else None
        if raw_key is not None:
            val = skill_input._data.get(raw_key, "")
            return val if isinstance(val, str) else str(val)

        schema = self.input_schema
        text_data = dict(skill_input._data)
        extra_blocks: list[dict] = []

        # Ask each agtype field for its contribution to the user message.
        # Fields with no agtype (e.g. plain str, int) are left as-is.
        if schema is not None:
            for key, hint in schema._data.items():
                cls = agtype.from_hint(hint)
                if cls is None:
                    continue
                placeholder, blocks = cls.build_content_prompt(key, skill_input._data.get(key))
                if placeholder is not None:
                    text_data[key] = placeholder
                extra_blocks.extend(blocks)

        # No extra blocks — return a plain JSON string (fast path).
        if not extra_blocks:
            return f"[HARNESS SYSTEM] New Skill Input:\n{skill_input.to_json()}"

        # Extra blocks present — build a multimodal content array: text first,
        # then the type-contributed blocks in schema field order.
        text = json.dumps(text_data)
        content: list = [{"type": "text", "text": f"New Skill Input:\n{text}"}]
        content.extend(extra_blocks)
        return content

    def _build_toolkit(
        self,
        agent_sandbox: "agSandbox",
        resource_pool: "agResourcePool | None",
        agent_terminal: "agterm | None",
        agent_log: "aglog | None",
        _ensure_read: bool = False,
    ) -> "tuple[dict[str, agtool], dict, set[str]]":
        """Build toolkit and structured-output collection state.

        Returns (toolkit, collected_outputs, required_fields).
        collected_outputs and required_fields are mutable containers that the
        return_<field> tools write into as they are called during the ReAct loop.
        required_fields is empty when the skill has no structured output schema.
        """
        if self.replace_tools is not None:
            active_tools: list[agtool] = list(self.replace_tools)
        else:
            from .tools import make_sandboxed_tools, make_read
            active_tools = make_sandboxed_tools(agent_sandbox, resource_pool)
            if self.add_tools:
                active_tools.extend(self.add_tools)

        if _ensure_read:
            if not any(getattr(t, "name", None) == "read" for t in active_tools):
                from .tools import make_read
                active_tools.append(make_read(agent_sandbox))

        collected_outputs: dict = {}
        required_fields: set[str] = set()
        if self.output_schema is not None and self.output_schema.raw_key() is None:
            required_fields = set(self.output_schema._data.keys())
            active_tools.extend(self.output_schema.make_return_output_agtool(
                agent_sandbox, collected_outputs, required_fields, AGBINARY_VALIDATE_EXEC_TIMEOUT
            ))

        for t in active_tools:
            t.attach_logger(agent_terminal, agent_log)

        return {t.name: t for t in active_tools}, collected_outputs, required_fields

    def _build_initial_messages(
        self,
        skill_input: agdata,
        agent_context: agcontext,
        _extra_system: "str | None",
        live_messages_fn: "Callable | None",
        full_history_fn: "Callable | None",
    ) -> "tuple[list[dict], int]":
        """Build initial messages list. Returns (messages, n_before)."""
        # n_before records how many messages were in agent_context before this skill run
        # started.  After the run, messages[n_before+1:] (skipping the leading
        # system prompt) is the "delta" — the new turns added by this call.
        n_before = len(agent_context.messages)

        # Three-part structure: [system] + persistent history from agent_context + [new user turn].
        messages: list[dict] = (
            [{"role": "system", "content": self._build_system_prompt(_extra_system)}]
            + list(agent_context.messages)
            + [{"role": "user", "content": self._build_user_content(skill_input)}]
        )

        # Push the conversation (minus system prompt) to the live UI view so the
        # user can see the running history before the first LLM response arrives.
        if live_messages_fn:
            live_messages_fn(messages[1:])

        # Log the system prompt and the new user message to the full-history sink
        # (e.g. aglog file writer) so they appear in debug transcripts.
        if full_history_fn:
            full_history_fn(messages[0])
            full_history_fn(messages[-1])
        return messages, n_before

    # ------------------------------------------------------------------
    # Scheduling wrapper — non-blocking, returns pending agdata
    # ------------------------------------------------------------------

    def run(
        self,
        ag: "agent",
        skill_input: agdata,
        max_steps: int = AGSKILL_REACT_MAX_STEPS,
    ) -> agdata:
        """Submit a skill run on *ag* and return a pending agdata immediately.

        Spawns a daemon thread that runs execute_react() and resolves futures
        when done.  Same-agent calls are serialized via the context future chain.
        """
        prev_ctx = ag.ctx
        result_future: Future[agdata] = Future()
        ctx_future: Future[agcontext] = Future()
        ts_start = _ts()
        resource_pool = type(ag).agresource_pool

        SKILL_ERROR_LOG_TRUNCATE = 300

        def _task() -> None:
            outer_result: agdata | None = None
            updated_ctx: agcontext = prev_ctx
            outer_delta: list[dict] = []
            history_before: list[dict] = []
            _prev_input_tokens: int = 0
            _prev_output_tokens: int = 0

            try:
                # ── 1. Unblock: wait for any in-flight predecessor to finish,
                #    then resolve any lazy input futures passed by the caller.
                prev_ctx.resolve_prev_dependencies()
                skill_input.resolve_input_dependencies()

                # ── 2. Provision sandbox — created once on first run and reused
                #    across subsequent runs via its internal checkpoint image.
                if not ag.is_external_sandbox and ag.sandbox is None:
                    _out = Path(type(ag).output_dir) / ag.agname if type(ag).output_dir else None
                    ag.sandbox = agSandbox(ag.agname, output_dir=_out)

                history_before = list(prev_ctx.messages)
                _prev_input_tokens = prev_ctx.total_input_tokens
                _prev_output_tokens = prev_ctx.total_output_tokens

                ag.terminal.log("SKILL ▶  ", f"{self.name}  input={list(skill_input._data.keys())}")
                ag._set_ui_state("skill", skill=self.name)
                ag._append_full_history({"type": "skill_start", "skill": self.name, "ts": ts_start})

                # ── 3. Run the ReAct loop.
                outer_result, updated_ctx, outer_delta = self.execute_react(
                    ag, prev_ctx, skill_input, max_steps,
                )

            except Exception as exc:
                outer_result = agerror(format_exception(exc))
                updated_ctx = prev_ctx
                outer_delta = []
                history_before = list(prev_ctx.messages)
                ag.terminal.log("SKILL ✗  ", f"{self.name}  exception={exc}")
            finally:
                # ── 4. Teardown — release GPU slot, commit container filesystem.
                _had_error = outer_result is not None and bool(outer_result._data.get("error"))
                ag._set_ui_state("error" if _had_error else "finished")
                if ag.sandbox is not None and ag.sandbox._gpu_id is not None:
                    resource_pool.release_gpu(ag.sandbox._gpu_id)
                if not ag.is_external_sandbox and ag.sandbox is not None:
                    ag.sandbox.stop(commit=True)

            # ── 5. Log result and commit token counts.
            ts_end = _ts()
            assert outer_result is not None
            input_dict  = skill_input.to_dict()
            result_dict = outer_result.to_dict()
            if result_dict.get("error"):
                ag.terminal.log("SKILL ✗  ", f"{self.name}  error={str(result_dict['error'])[:SKILL_ERROR_LOG_TRUNCATE]}")
                ag._append_full_history({"type": "skill_error", "skill": self.name,
                                         "error": str(result_dict["error"])})
            else:
                ag.terminal.log("SKILL ✓  ", f"{self.name}  output={list(result_dict.keys())}")
            outer_input_tokens  = updated_ctx.total_input_tokens  - _prev_input_tokens
            outer_output_tokens = updated_ctx.total_output_tokens - _prev_output_tokens
            try:
                ag.log._record(self.name, ts_start, ts_end,
                               input_dict, result_dict,
                               len(updated_ctx.messages),
                               history_before=history_before,
                               history_delta=outer_delta,
                               input_tokens=outer_input_tokens,
                               output_tokens=outer_output_tokens)
                type(ag)._add_global_tokens(outer_input_tokens, outer_output_tokens)
                _ag_usage = ag.log.token_usage
                _gl_usage = type(ag).global_token_usage()
                try:
                    from . import agwebui as _agwebui
                    if _agwebui._active is not None:
                        _agwebui._active.emitter.token_update(
                            ag.agname,
                            _ag_usage["input_tokens"],
                            _ag_usage["output_tokens"],
                            _gl_usage["input_tokens"],
                            _gl_usage["output_tokens"],
                        )
                except Exception as _e:
                    print(f"[agskill] WARNING: post-skill token_update push failed for {ag.agname}: {_e}")
            except Exception as log_exc:
                ag.terminal.log("SKILL ✗  ", f"[log error] {log_exc}")

            # ── 6. Resolve result future — unblocks the caller immediately.
            ag._snapshot_messages = list(updated_ctx.messages)
            result_future.set_result(outer_result)

            # ── 7. Prune history, then resolve ctx future for the next chained call.
            try:
                pruned_msgs = agllm._prune_tool_outputs(updated_ctx.messages)
                if pruned_msgs is not updated_ctx.messages:
                    updated_ctx.messages = pruned_msgs
                    ag.terminal.log("PRUNE    ", f"{self.name}  history pruned to {len(pruned_msgs)} msgs")
            except Exception as prune_exc:
                ag.terminal.log("PRUNE ✗  ", f"{self.name}  pruning failed: {prune_exc}")

            ctx_future.set_result(updated_ctx)

        threading.Thread(target=_task, daemon=True).start()
        ag.ctx = agcontext(_future=ctx_future)
        return agdata(_future=result_future)

    async def asyncio_run(
        self,
        ag: "agent",
        skill_input: agdata,
        max_steps: int = AGSKILL_REACT_MAX_STEPS,
    ) -> agdata:
        """Async wrapper around run() for use in asyncio event loops."""
        import asyncio
        loop = asyncio.get_event_loop()
        pending = self.run(ag, skill_input, max_steps)
        await loop.run_in_executor(None, pending._resolve)
        return pending

    # ------------------------------------------------------------------
    # ReAct loop — synchronous execution
    # ------------------------------------------------------------------

    def execute_react(
        self,
        ag: "agent",
        prev_ctx: agcontext,
        skill_input: agdata,
        max_steps: int = AGSKILL_REACT_MAX_STEPS,
    ) -> "tuple[agdata, agcontext, list[dict]]":
        """Run the ReAct loop synchronously against *ag*, return (result, ctx, delta)."""

        # ── 1. Validate input against the skill's input schema.
        input_error = self.input_schema.validate_input(skill_input) if self.input_schema is not None else None
        if input_error is not None:
            sys_msg = {"role": "system", "content": self._build_system_prompt()}
            return agerror(input_error), prev_ctx, [sys_msg]

        # ── 2. Prepare inputs — write agtype fields and offload oversized strings.
        _input_suffix = f"_{int(time.time() * 1000)}"
        _offloaded_paths, auto_fields = (
            self.input_schema.prepare_inputs_in_sandbox(
                skill_input, ag.sandbox, self.name,
                suffix=_input_suffix, context_limit=ag.llm.context_limit,
            )
            if self.input_schema is not None
            else ([], [])
        )

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

        # ── 3. Build toolkit with return_<field> tools for structured output.
        toolkit, _collected_outputs, _required_fields = self._build_toolkit(
            ag.sandbox, type(ag).agresource_pool, ag.terminal, ag.log,
            _ensure_read=bool(_offloaded_paths),
        )
        _use_return_output = bool(_required_fields)

        # ── 5. Build the initial message list (system prompt + history + user turn).
        messages, n_before = self._build_initial_messages(
            skill_input, prev_ctx, _extra_system,
            ag._push_live_messages, ag._append_full_history,
        )
        output_schema_retries_left = self.max_output_schema_retries
        _skill_tokens_in_start  = prev_ctx.total_input_tokens
        _skill_tokens_out_start = prev_ctx.total_output_tokens

        # ── 6. ReAct loop — each iteration is one LLM call + tool dispatch cycle.
        for _ in range(max_steps):
            # Derive wire-format tool schemas fresh each iteration — the toolkit dict
            # may grow mid-loop (e.g. read injected on large output offload).
            _tool_schemas = [t.to_openai_tool() for t in toolkit.values()] or None
            kwargs: dict = ag.llm.build_kwargs(messages, _tool_schemas)

            # 6a. Drain any inbox messages injected by the orchestrator mid-loop.
            had_inbox = ag._drain_inbox(messages)

            # 6b. Compact history if needed.
            messages, _pre_estimate = ag.llm.maybe_compact(
                prev_ctx, messages, None,
                term=ag.terminal, log=ag.log,
                _live_messages_fn=ag._push_live_messages,
                skill_name=self.name, agname=str(ag.agname),
            )

            ag.push_token_count_update_to_ui(
                prev_ctx.total_input_tokens - _skill_tokens_in_start + _pre_estimate,
                prev_ctx.total_output_tokens - _skill_tokens_out_start,
            )

            if ag.llm.context_limit is not None:
                _headroom = max(1, ag.llm.context_limit - _pre_estimate)
                if kwargs.get("max_tokens", _headroom) > _headroom:
                    kwargs = dict(kwargs)
                    kwargs["max_tokens"] = _headroom

            # 6c. Call the LLM (with internal retry on transient errors).
            llm_result = ag.llm.call(
                kwargs, messages,
                ag.terminal, ag._set_ui_state, ag._push_live_messages,
                ag.push_token_count_update_to_ui,
                prev_ctx.total_input_tokens, prev_ctx.total_output_tokens, self.name,
                full_history_fn=ag._append_full_history,
            )
            if llm_result.context_exceeded:
                if ag._append_full_history:
                    ag._append_full_history({"type": "llm_context_exceeded"})
                messages, _ = ag.llm.maybe_compact(
                    prev_ctx, messages, None,
                    term=ag.terminal, log=ag.log,
                    _live_messages_fn=ag._push_live_messages,
                    skill_name=self.name, agname=str(ag.agname), force=True,
                )
                continue
            if not llm_result.ok:
                _err_msg = f"LLM connection error after retries: {llm_result.conn_error}"
                if ag._append_full_history:
                    ag._append_full_history({"type": "llm_error", "error": _err_msg})
                ag.sandbox.remove_files(_offloaded_paths)
                return agerror(_err_msg), prev_ctx, []
            prev_ctx.total_input_tokens  = llm_result.total_input_tokens
            prev_ctx.total_output_tokens = llm_result.total_output_tokens
            if ag.terminal:
                _ctx_str = f"/{ag.llm.context_limit}" if ag.llm.context_limit else ""
                _tok_str = f"  tokens={llm_result.prompt_tokens}{_ctx_str}" if llm_result.prompt_tokens else ""
                ag.terminal.log("LLM ✓    ", f"model={ag.llm.config.get('model','?')}  ({llm_result.elapsed_ms}ms){_tok_str}")
            if ag._set_ui_state:
                ag._set_ui_state("skill", skill=self.name)

            # 6d. Post-response compaction.
            messages, _ = ag.llm.maybe_compact(
                prev_ctx, messages, llm_result.prompt_tokens,
                term=ag.terminal, log=ag.log,
                _live_messages_fn=ag._push_live_messages,
                skill_name=self.name, agname=str(ag.agname),
            )

            # 6e. Append the assistant turn to the message list.
            msg_dict: dict = agllm.build_assistant_msg(llm_result.content_parts, llm_result.reasoning_parts, llm_result.tool_calls_raw)
            messages.append(msg_dict)
            if ag._push_live_messages:
                ag._push_live_messages(messages[1:])
            if ag._append_full_history:
                ag._append_full_history(msg_dict)

            # 6f. Dispatch tool calls, or check if we can move to the output path.
            if msg_dict.get("tool_calls"):
                dispatch_tools(
                    msg_dict["tool_calls"], toolkit, messages, ag.sandbox, self.name,
                    ag._set_ui_state, ag._push_live_messages, ag._append_full_history, ag.terminal,
                    tool_offload_chars=(
                        max(TOOL_OUTPUT_OFFLOAD_CHARS, int(ag.llm.context_limit * 0.1 * 4))
                        if ag.llm.context_limit else TOOL_OUTPUT_OFFLOAD_CHARS
                    ),
                )
                # Continue looping unless all required output fields are collected.
                if not (_use_return_output and not (_required_fields - set(_collected_outputs))):
                    continue

            else:
                # No tool calls — only continue if a mid-loop inbox message arrived. If not, move onto the output path (step 7).
                if had_inbox:
                    continue

            # ── 7. Output-ready path.
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
                                f"Call {_missing_tools} tool(s) with your output as tool argument."
                            ),
                        }
                        messages.append(reprompt)
                        if ag._push_live_messages:
                            ag._push_live_messages(messages[1:])
                        if ag._append_full_history:
                            ag._append_full_history(reprompt)
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
                    prev_ctx.messages = messages[1:]
                    ag.sandbox.remove_files(_offloaded_paths)
                    return (
                        agerror(
                            f"output schema error: missing fields after retries: {sorted(missing)}"
                            + f"\ncollected: {sorted(_collected_outputs.keys())}"
                            + (f"\nlast model output: {_last_out_str!r}" if _last_out_str else "")
                        ),
                        prev_ctx,
                        [messages[0]] + messages[1:][n_before:],
                    )
                # 7b. All fields collected — wait for any background sandbox processes.
                result = agdata(**_collected_outputs)
                proc_msg = agSandbox.wait_for_processes(
                    ag.sandbox, self.name, ag.terminal, ag.log,
                    str(ag.agname), type(ag).ping_interval_s, type(ag).poll_interval_s, ag._set_ui_state,
                )
                if proc_msg is not None:
                    messages.append({"role": "user", "content": proc_msg})
                    if ag._push_live_messages:
                        ag._push_live_messages(messages[1:])
                    if ag._append_full_history:
                        ag._append_full_history(messages[-1])
                    continue
                # 7c. Recover agtype outputs then clean up auto-offloaded input files.
                prev_ctx.messages = messages[1:]
                self.output_schema.recover_outputs(result, ag.sandbox)
                ag.sandbox.remove_files(_offloaded_paths)
                return (
                    result,
                    prev_ctx,
                    [messages[0]] + messages[1:][n_before:],
                )

            # ── 8. Raw-text output path (agrawstring schema or no schema).
            assert self.output_schema is None or self.output_schema.raw_key() is not None, (
                f"BUG: reached raw-text path with structured output_schema on skill '{self.name}'. "
                "This should be unreachable — _use_return_output covers all schema cases."
            )
            out_key = self.output_schema.raw_key() if self.output_schema is not None else "result"
            result = agdata(**{out_key: msg_dict.get("content") or ""})
            proc_msg = agSandbox.wait_for_processes(
                ag.sandbox, self.name, ag.terminal, ag.log,
                str(ag.agname), type(ag).ping_interval_s, type(ag).poll_interval_s, ag._set_ui_state,
            )
            if proc_msg is not None:
                messages.append({"role": "user", "content": proc_msg})
                if ag._push_live_messages:
                    ag._push_live_messages(messages[1:])
                if ag._append_full_history:
                    ag._append_full_history(messages[-1])
                continue
            prev_ctx.messages = messages[1:]
            ag.sandbox.remove_files(_offloaded_paths)
            return (result, prev_ctx, [messages[0]] + messages[1:][n_before:])

        # ── 9. Max steps exhausted.
        prev_ctx.messages = messages[1:]
        ag.sandbox.remove_files(_offloaded_paths)
        return agerror("max_steps exceeded"), prev_ctx, [messages[0]] + messages[1:][n_before:]

    def __repr__(self) -> str:
        return f"agskill(name={self.name!r})"
