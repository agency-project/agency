from __future__ import annotations
import json
import time
from concurrent.futures import Future
from typing import TYPE_CHECKING, Callable
from .agdata import agdata, agerror
from .agpolicy import agpolicy
from .agtype import agtype
from . import agpause
from .profiler import agprof
from .agschema import agschema
from .agcontext import agcontext
from .agtool import agtool
from .llm.agllm import agllm
from .sandbox.agsandbox import agSandbox
from .agconfig import DynamicConfigParam, _AgConfigViewBase
from .agutil import format_exception
from .aglog import _ts


# Exists only to register agskill's config fields (via __set_name__ at import
# time). Constants are plain class attributes (not descriptors) so other code
# in this file needing the same hardcoded value can reference it directly.
# Reads use a throwaway instance -- _AgSkillFields(agconfig) -- since
# agskill instances don't hold their own agconfig (a skill runs on behalf of
# different agents with different agconfigs), so there's no self to hang a
# descriptor on.


# [REFACTOR] Maybe belongs in the harness code?
class _AgSkillFields:
    react_max_steps = DynamicConfigParam("agskill", default=4096)
    agbinary_validate_exec_timeout = DynamicConfigParam("agskill", default=5)
    error_log_truncate = DynamicConfigParam("agskill", default=300)
    last_output_log_truncate = DynamicConfigParam("agskill", default=2000)

    def __init__(self, agconfig=None) -> None:
        self._agconfig = agconfig


class agSkillConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting agskill tunables in one call::

        cfg = agConfig(agSkillConfig(react_max_steps=64))

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics.
    """

    _OWNER = "agskill"


if TYPE_CHECKING:
    from .agent import agent


# ---------------------------------------------------------------------------
# Default host-side MCP tools
# ---------------------------------------------------------------------------


def _memory_mb_to_docker_str(memory_mb: "float | None") -> "str | None":
    if memory_mb is None:
        return None
    return f"{int(memory_mb)}m"


def _reserve_resource(arg: agdata, sandbox, resource_pool) -> agdata:
    cpus = arg._data.get("cpus")
    memory_mb = arg._data.get("memory_mb")
    gpu = arg._data.get("gpu", 0)
    messages = []
    result: dict = {}

    if cpus is not None and cpus > resource_pool.total_cpus:
        return agerror(f"requested {cpus} cpus but pool only has {resource_pool.total_cpus}")
    if memory_mb is not None and memory_mb > resource_pool.total_memory_mb:
        return agerror(
            f"requested {memory_mb} MB but pool only has {resource_pool.total_memory_mb} MB"
        )
    if gpu and gpu > len(resource_pool.gpus):
        return agerror(f"requested {gpu} gpus but pool only has {len(resource_pool.gpus)}")

    if cpus is not None or memory_mb is not None:
        sandbox.update_limits(cpus=cpus, memory=_memory_mb_to_docker_str(memory_mb))
        if cpus is not None:
            sandbox._cpu_acquired += cpus
        if memory_mb is not None:
            sandbox._memory_acquired_mb += memory_mb
        resource_pool.notify_cpu_acquired(cpus or 0.0, memory_mb or 0)
        messages.append(f"cpus={cpus}, memory_mb={memory_mb}")

    if gpu:
        sandbox._gpu_count_requested = gpu
        sandbox._gpu_acquire_fn = resource_pool.acquire_gpus
        sandbox._gpu_release_fn = resource_pool.release_gpus
        result["gpu_count_requested"] = gpu
        messages.append(f"gpu reservation set to {gpu} (granted lazily on next exec)")

    if not messages:
        return agerror("reserve_resource called with nothing to reserve")
    result["message"] = "Reserved: " + "; ".join(messages)
    return agdata(**result)


def _release_resource(arg: agdata, sandbox, resource_pool) -> agdata:
    cpu = arg._data.get("cpu", False)
    memory = arg._data.get("memory", False)
    gpu = arg._data.get("gpu", False)
    messages = []

    if cpu or memory:
        held_cpus = sandbox._cpu_acquired if cpu else 0.0
        held_mb = sandbox._memory_acquired_mb if memory else 0
        sandbox.update_limits(
            cpus=resource_pool.idle_cpus if cpu else None,
            memory=resource_pool.idle_memory if memory else None,
        )
        if cpu:
            sandbox._cpu_acquired = 0.0
        if memory:
            sandbox._memory_acquired_mb = 0
        resource_pool.notify_cpu_released(held_cpus, held_mb)
        messages.append(f"cpu={cpu}, memory={memory} reset to idle")

    if gpu:
        if sandbox._gpu_count_requested > 0:
            if sandbox._gpu_ids:
                resource_pool.release_gpus(sandbox._gpu_ids)
            messages.append(f"gpu reservation ({sandbox._gpu_count_requested}) released")
            sandbox._gpu_ids = []
            sandbox._gpu_count_requested = 0
            sandbox._gpu_acquire_fn = None
            sandbox._gpu_release_fn = None
        else:
            messages.append("no gpu was reserved")

    if not messages:
        return agerror("release_resource called with nothing to release")
    return agdata(message="; ".join(messages))


def _get_current_resources(arg: agdata, sandbox, resource_pool) -> agdata:
    return agdata(
        cpus_acquired=sandbox._cpu_acquired,
        memory_mb_acquired=sandbox._memory_acquired_mb,
        gpu_count_requested=sandbox._gpu_count_requested,
        gpu_ids_held=list(sandbox._gpu_ids),
        total_cpus=resource_pool.total_cpus,
        total_memory_mb=resource_pool.total_memory_mb,
        total_gpus=len(resource_pool.gpus),
    )


def _daemon_release(arg: agdata, sandbox) -> agdata:
    pid = arg._data["pid"]
    sandbox.release_daemon(pid)
    return agdata(message=f"PID {pid} released as daemon -- will not block skill completion")


def _submit_output(arg: agdata, output_schema, submitted_output_store: dict) -> agdata:
    if output_schema is None:
        return agerror("this skill declares no output_schema -- nothing to submit")
    field = arg._data["field"]
    value = arg._data["value"]
    if field not in output_schema._data:
        return agerror(f"unknown output field {field!r}")
    err = output_schema.check_field(field, value)
    if err is not None:
        return agerror(err)
    submitted_output_store[field] = value
    required = set(output_schema._data.keys())
    still_missing = sorted(required - set(submitted_output_store.keys()))
    return agdata(result=f"field {field!r} recorded", still_missing=still_missing)


def _submitted_output(arg: agdata, submitted_output_store: dict) -> agdata:
    return agdata(**submitted_output_store)


_DEFAULT_HOST_MCP_TOOLS: "list[agtool]" = [
    agtool(
        name="reserve_resource",
        description=(
            "Reserve additional CPU/memory/GPU capacity for this sandbox. "
            "Any combination of cpus/memory_mb/gpu may be given in one call; "
            "omitted resources are left untouched."
        ),
        fn=_reserve_resource,
        params={
            "type": "object",
            "properties": {
                "cpus": {"type": "number", "description": "CPUs to reserve"},
                "memory_mb": {"type": "number", "description": "memory to reserve, in MB"},
                "gpu": {"type": "integer", "description": "number of GPUs to reserve"},
            },
            "required": [],
        },
    ),
    agtool(
        name="release_resource",
        description=(
            "Release previously reserved CPU/memory/GPU capacity for this "
            "sandbox. Any combination of cpu/memory/gpu may be given in one "
            "call; omitted resources are left untouched."
        ),
        fn=_release_resource,
        params={
            "type": "object",
            "properties": {
                "cpu": {"type": "boolean", "description": "release held CPU"},
                "memory": {"type": "boolean", "description": "release held memory"},
                "gpu": {"type": "boolean", "description": "release held GPU(s)"},
            },
            "required": [],
        },
    ),
    agtool(
        name="get_current_resources",
        description="Return this sandbox's current resource allocation and the pool's totals.",
        fn=_get_current_resources,
    ),
    agtool(
        name="daemon_release",
        description=(
            "Release a background process (by pid) from monitoring so the "
            "skill can finish without waiting for it."
        ),
        fn=_daemon_release,
        params={
            "type": "object",
            "properties": {"pid": {"type": "integer", "description": "pid to release"}},
            "required": ["pid"],
        },
    ),
    agtool(
        name="submit_output",
        description="Submit one required output field's value.",
        fn=_submit_output,
        params={
            "type": "object",
            "properties": {
                "field": {"type": "string", "description": "output field name"},
                "value": {"description": "the field's value"},
            },
            "required": ["field", "value"],
        },
        persistent_vars={"submitted_output_store": dict},
    ),
    agtool(
        name="submitted_output",
        description="Return the output fields submitted so far.",
        fn=_submitted_output,
        persistent_vars={"submitted_output_store": dict},
    ),
]


# ---------------------------------------------------------------------------
# Skill class
# ---------------------------------------------------------------------------


class agskill:
    """A named skill with its own system prompt and a self-contained ReAct loop. # [REFACTOR] Not anymore...

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
        add_host_mcp_tools: "list[agtool] | None" = None,
        add_sandbox_mcp_tools: "list[agtool] | None" = None,
        input_schema: agdata | None = None,
        output_schema: agdata | None = None,
        max_output_schema_retries: int = 10,  # [REFACTOR] Why here?
        policy: "agpolicy | None" = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.input_schema = agschema(input_schema) if input_schema else None
        self.output_schema = agschema(output_schema) if output_schema else None
        self.max_output_schema_retries = max_output_schema_retries
        self.policy = policy if policy is not None else agpolicy()
        self.host_mcp_tools = list(_DEFAULT_HOST_MCP_TOOLS) + (add_host_mcp_tools or [])
        self.sandbox_mcp_tools = list(add_sandbox_mcp_tools or [])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, extra: str | None = None) -> str:
        parts = [self.system_prompt]  # [REFACTOR] Maybe rename into skill_prompt?

        # Each agtype subclass (agfile, agbinary, …) can inject extra prompt
        # lines describing how the LLM should handle that field (e.g. file paths,
        # binary encoding).  Collect these for both input and output schemas.
        extra_lines: list[str] = []
        for key, hint in self.input_schema._data.items() if self.input_schema else []:
            cls = agtype.from_hint(hint)  # [REFACTOR] We need better variable names here.
            if cls is not None:
                line = cls.extra_input_prompt(key)
                if line:
                    extra_lines.append(line)
        for key, hint in self.output_schema._data.items() if self.output_schema else []:
            cls = agtype.from_hint(hint)  # [REFACTOR] Better names needed.
            if cls is not None:
                line = cls.extra_output_prompt(key, self.name)
                if line:
                    extra_lines.append(line)

        # Emit the agtype instructions as a single block before the format sections.
        if extra_lines:
            parts.append(
                "\nFile-backed fields — WARNING: these files are temporary and will "
                "be automatically deleted after this task ends:\n" + "\n".join(extra_lines)
            )

        # Caller-supplied extra prompt (e.g. compaction summary injection). # [REFACTOR] Check which method uses extras
        if extra:
            parts.append(extra)

        # Describe the input shape so the LLM knows what JSON keys to expect.
        # Skipped for agrawstring inputs (the value arrives as plain text, not JSON).
        if self.input_schema is not None and self.input_schema.raw_key() is None:
            parts.append(
                f"\nInput JSON format:\n{self.input_schema.to_json()}"
            )  # [REFACTOR] Do we have to explain the input format?

        if self.output_schema is not None:
            if self.output_schema.raw_key() is not None:
                # agrawstring output — model must reply with plain text, not a tool call.
                parts.append(
                    "\nRespond with plain text only — no JSON wrapping, no markdown code fences."
                )
            else:
                # Structured output — model must call one return_<field> tool per output field.
                field_tools = ", ".join(f"return_{f}" for f in self.output_schema._data)
                field_lines = "\n".join(
                    f"  - {f}: {self.output_schema.field_desc(f)}" for f in self.output_schema._data
                )
                parts.append(  # [REFACTOR] Better prompting
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

    def _build_user_content(
        self, skill_input: agdata
    ) -> "str | list":  # [REFACTOR] Are we only providing the per-turn inputs here?
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
        extra_blocks: list[dict] = []  # [REFACTOR] Better names - Why "extra"?

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

        # No extra blocks — return a plain JSON string (fast path). # [REFACTOR] what is skill input and what is extra_blocks - Maybe because of offloading? If so, need better names
        if not extra_blocks:
            return f"[HARNESS SYSTEM] New Skill Input:\n{skill_input.to_json()}"

        # Extra blocks present — build a multimodal content array: text first,
        # then the type-contributed blocks in schema field order.
        text = json.dumps(text_data)
        content: list = [{"type": "text", "text": f"New Skill Input:\n{text}"}]
        content.extend(extra_blocks)
        return content

    def _build_initial_messages(  # [REFACTOR] Unused?
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
            live_messages_fn(messages[1:])  # [REFACTOR] Why 1:?

        # Log the system prompt and the new user message to the full-history sink
        # (e.g. aglog file writer) so they appear in debug transcripts.
        if full_history_fn:
            full_history_fn(messages[0])
            full_history_fn(messages[-1])  # [REFACTOR] Why 0 and -1?
        return messages, n_before

    # ------------------------------------------------------------------
    # Scheduling wrapper — non-blocking, returns pending agdata
    # ------------------------------------------------------------------

    def run(
        self,
        ag: "agent",
        skill_input: agdata,
        max_steps: "int | None" = None,
    ) -> agdata:
        """Submit a skill run on *ag* and return a pending agdata immediately.

        Spawns a daemon thread that delegates execution to the agent's driver
        engine and resolves futures when done. Same-agent calls are serialized
        via the context future chain.
        """
        prev_ctx = ag.ctx
        result_future: Future[agdata] = Future()
        ctx_future: Future[agcontext] = Future()
        agpause.tag_producer(result_future, ag)
        agpause.tag_producer(ctx_future, ag)
        ts_start = _ts()

        def _task() -> None:  # [REFACTOR] Why wrap in task?
            outer_result: agdata | None = None
            updated_ctx: agcontext = prev_ctx
            outer_delta: list[dict] = []
            history_before: list[dict] = []
            _prev_input_tokens: int = 0
            _prev_output_tokens: int = 0
            # Fallback for the final logging step below if an exception hits
            # before the defensive copy further down is made.
            local_skill_input = skill_input
            agpause.set_current_worker_agent(ag)

            try:
                # ── 1. Unblock: wait for any in-flight predecessor to finish,
                #    then resolve any lazy input futures passed by the caller.
                with agprof.span("resolve"):
                    prev_ctx.resolve_prev_dependencies()
                    skill_input.resolve_input_dependencies()

                # Defensive shallow copy: prepare_inputs_in_sandbox() (called
                # below, via execute_harness) mutates its skill_input argument
                # in place (offloading oversized/agtype fields to sandbox
                # paths). If a caller hands the same agdata object to more
                # than one concurrent run() call (e.g. one shared input
                # fanned out to several agents), each run must mutate its
                # own private copy from here on rather than racing the
                # others on a shared one. A shallow copy is enough --
                # prepare_inputs_in_sandbox only ever reassigns top-level
                # keys on the object it's given, never mutates a nested
                # value's own contents in place.
                local_skill_input = agdata(**dict(skill_input._data))

                history_before = list(prev_ctx.messages)
                _prev_input_tokens = prev_ctx.total_input_tokens
                _prev_output_tokens = prev_ctx.total_output_tokens

                ag.terminal.log(
                    "SKILL ▶  ", f"{self.name}  input={list(local_skill_input._data.keys())}"
                )
                ag._set_ui_state("skill", skill=self.name)
                ag._append_full_history({"type": "skill_start", "skill": self.name, "ts": ts_start})

                # ── 2. Delegate actual execution to the Agent Engine.
                execution = ag.engine.execute(
                    context=prev_ctx,
                    skill=self,
                    skill_input=local_skill_input,
                    resource_pool=type(ag).agresource_pool,
                    max_steps=max_steps,
                )
                outer_result = execution.output
                updated_ctx = execution.context
                outer_delta = execution.delta

                outer_result = execution.output
                updated_ctx = execution.context
                outer_delta = execution.delta

            except Exception as exc:
                outer_result = agerror(format_exception(exc))
                updated_ctx = prev_ctx
                outer_delta = []
                history_before = list(prev_ctx.messages)
                ag.terminal.log("SKILL ✗  ", f"{self.name}  exception={exc}")
            finally:
                _had_error = outer_result is not None and bool(outer_result._data.get("error"))
                ag._set_ui_state("error" if _had_error else "finished")
                agpause.set_current_worker_agent(None)  # [REFACTOR] What does this do?

            # ── 3. Log result and commit token counts.
            ts_end = _ts()
            assert outer_result is not None
            input_dict = local_skill_input.to_dict()
            result_dict = outer_result.to_dict()
            if result_dict.get("error"):
                _error_log_truncate = (
                    _AgSkillFields(ag.agconfig).error_log_truncate
                )  # [REFACTOR] What is this? Why do we get it through ag.agconfig?
                ag.terminal.log(
                    "SKILL ✗  ",
                    f"{self.name}  error={str(result_dict['error'])[:_error_log_truncate]}",
                )
                ag._append_full_history(
                    {"type": "skill_error", "skill": self.name, "error": str(result_dict["error"])}
                )
            else:
                ag.terminal.log("SKILL ✓  ", f"{self.name}  output={list(result_dict.keys())}")
            outer_input_tokens = updated_ctx.total_input_tokens - _prev_input_tokens
            outer_output_tokens = updated_ctx.total_output_tokens - _prev_output_tokens
            try:
                ag.log._record(
                    self.name,
                    ts_start,
                    ts_end,
                    input_dict,
                    result_dict,
                    len(updated_ctx.messages),
                    history_before=history_before,
                    history_delta=outer_delta,
                    input_tokens=outer_input_tokens,
                    output_tokens=outer_output_tokens,
                )
                type(ag)._add_global_tokens(
                    outer_input_tokens, outer_output_tokens
                )  # [REFACTOR] Where is the add for local tokens??
                _ag_usage = (
                    ag.log.token_usage
                )  # [REFACTOR] Is ag.log the right place to get token usage?
                _gl_usage = type(ag).global_token_usage()
                try:
                    from . import agwebui as _agwebui  # [REFACTOR] Why lazy import?

                    if _agwebui._active is not None:
                        _agwebui._active.emitter.token_update(
                            ag.agname,
                            _ag_usage["input_tokens"],
                            _ag_usage["output_tokens"],
                            _gl_usage["input_tokens"],
                            _gl_usage["output_tokens"],
                        )
                except Exception as _e:
                    print(
                        f"[agskill] WARNING: post-skill token_update push failed for {ag.agname}: {_e}"
                    )
            except Exception as log_exc:
                ag.terminal.log("SKILL ✗  ", f"[log error] {log_exc}")

            # ── 6. Resolve result future — unblocks the caller immediately.
            ag._snapshot_messages = list(updated_ctx.messages)
            result_future.set_result(outer_result)

            # ── 7. Prune history, then resolve ctx future for the next chained call. # [REFACTOR] What kind of pruning and auto context management do we have?
            try:
                with agprof.span("prune"):
                    pruned_msgs = agllm._prune_tool_outputs(
                        updated_ctx.messages
                    )  # [REFACTOR] Why is this part of agllm?
                if pruned_msgs is not updated_ctx.messages:
                    updated_ctx.messages = pruned_msgs
                    ag.terminal.log(
                        "PRUNE    ", f"{self.name}  history pruned to {len(pruned_msgs)} msgs"
                    )
            except Exception as prune_exc:
                ag.terminal.log("PRUNE ✗  ", f"{self.name}  pruning failed: {prune_exc}")

            ctx_future.set_result(updated_ctx)

        # Set synchronously, before the thread even starts, so there is no
        # window where a run is genuinely in flight but ui_state still reads
        # "inactive".
        ag._set_ui_state(
            "skill", skill=self.name
        )  # [REFACTOR] Why not at the start of the run() function?

        def _traced_task() -> None:  # [REFACTOR] Maybe inline
            run_id = f"run{agprof.next_index()}"
            label = f"{run_id}:{self.name}:{ag.agname}"
            agprof.thread_name(label)
            with agprof.span(label):
                agprof.annotate(
                    **{
                        "agency.run_id": run_id,
                        "agency.agent_id": str(ag.agname),
                        "agency.parent_agent_id": getattr(
                            ag, "_parent_agent_id", None
                        ),  # [REFACTOR] Why do we need to track this?
                    }
                )
                _task()
                profile_result = result_future.result()
                profile_error = profile_result._data.get("error")
                agprof.annotate(
                    outcome="failure" if profile_error else "success",
                    error_type="skill_error" if profile_error else None,
                )

        agprof.spawn_traced(_traced_task).start()  # [REFACTOR] Why through "spawn_traced"?
        ag.ctx = agcontext(_future=ctx_future)
        return agdata(_future=result_future)

    async def asyncio_run(
        self,
        ag: "agent",
        skill_input: agdata,
        max_steps: "int | None" = None,
    ) -> agdata:
        """Async wrapper around run() for use in asyncio event loops."""
        import asyncio

        loop = asyncio.get_event_loop()
        pending = self.run(ag, skill_input, max_steps)
        await loop.run_in_executor(None, pending._resolve)
        return pending

    # ------------------------------------------------------------------
    # execute_react() (the old host-process ReAct loop -- LLM calls direct
    # from the host, tool dispatch via agtool.py's dispatch_tools() with a
    # per-tool-call sandbox hibernate) was retired here. Every engine,
    # native included, now runs through execute_harness() below -- native's
    # own loop lives in a persistent in-container process
    # (agharness_backends/native.py), not in this host process.
    # ------------------------------------------------------------------

    def execute_harness(  # [REFACTOR]  Can be inlined into run()?
        self,
        ag: "agent",
        prev_ctx: agcontext,
        skill_input: agdata,
        max_steps: "int | None" = None,
    ) -> "tuple[agdata, agcontext, list[dict]]":
        # [REFACTOR] Too much text
        """Run this skill against *ag* via its configured `agharness_backend`
        -- called unconditionally by `agskill.run()`'s `_task()` for every
        engine, native included (native is just another backend whose
        "binary" happens to be agency's own code). Same contract
        `execute_react()` used to promise on its own: `ctx` is the SAME
        `prev_ctx` object passed in, mutated in place (`.messages`/
        `.total_input_tokens`/`.total_output_tokens`); `delta` is
        `[system_prompt_message] + every message appended since this call
        started`. By the time `_task()` reaches this branch,
        `prev_ctx.resolve_prev_dependencies()` has already run (agskill.py's
        `_task()`), so `.messages` is already a concrete resolved list --
        this method does not need to resolve futures itself.

        See docs/Design_harness_integration.md for the design this
        implements: the skill's system prompt + input become a plain
        user-turn prompt (never injected as the harness's own system
        prompt or a tool), and the harness's own built-in tools/compaction
        run untouched -- mediation happens at the syscall level via
        agproxy_ptrace, not through this method.

        Also where every engine gets agtype/oversized-input offloading and
        agtype-output recovery -- the same `agschema.prepare_inputs_in_
        sandbox()`/`recover_outputs()` operations `execute_react()` used to
        call itself, hoisted up here so they're one shared, engine-agnostic
        implementation instead of five. A background-job wait
        (`agSandbox.wait_for_processes()`, `execute_react()`'s third such
        operation) is only called here for the `native` engine, NOT hoisted
        for all five -- see the call site's own comment for why the other
        four engines' ptrace-tracked child processes make that unsafe today.
        Both hoisted operations are
        host-side, sandbox-based operations with no dependency on which
        backend actually dispatched the call.
        """
        from .harness.adapters.base import agharness_backend

        input_error = (
            self.input_schema.validate_input(skill_input) if self.input_schema is not None else None
        )
        if input_error is not None:
            sys_msg = {"role": "system", "content": self._build_system_prompt()}
            return agerror(input_error), prev_ctx, [sys_msg]

        _input_suffix = f"_{int(time.time() * 1000)}"
        with agprof.span("input:prepare"):
            _offloaded_paths, auto_fields = (
                self.input_schema.prepare_inputs_in_sandbox(
                    skill_input,
                    ag.sandbox,
                    self.name,
                    suffix=_input_suffix,
                    context_limit=ag.llm.context_limit,
                    agconfig=ag.agconfig,
                )
                if self.input_schema is not None
                else ([], [])
            )
        extra_system: "str | None" = None
        if auto_fields:
            field_list = ", ".join(f"`{f}`" for f in auto_fields)
            extra_system = (
                f"\nNote: The following input fields contain large content "
                f"that has been automatically saved to temporary files in "
                f"your sandbox: {field_list}. The file paths are shown in "
                f"the input JSON. Use the read tool to access the full "
                f"content. WARNING: these files are temporary and will be "
                f"automatically deleted after this task ends."
            )

        backend = agharness_backend.for_config(
            ag.harness, ag.agconfig
        )  # [REFACTOR] ag.harness should be part of ag.config

        # Manager/bridge lifecycle lives HERE, at this one shared choke
        # point -- not duplicated per backend. See agharness_backends/
        # base.py's execute() docstring and agharness.py's own
        # get_or_create_host_manager()/ensure_harness_bridge() docstrings
        # for why this moved out of each backend's own execute().
        from .harness import agharness

        host_manager = agharness.get_or_create_host_manager(ag, ag.agconfig)
        # ensure_harness_bridge() itself picks container-backed vs
        # bare-host/chroot mode -- always returns a real base URL now,
        # never None; a backend that only supports one mode (native_harness
        # requires container-backed) checks agharness.is_container_backed()
        # itself, not harness_base_url's presence.
        harness_base_url = agharness.ensure_harness_bridge(ag.sandbox, host_manager)

        launch = host_manager.register_launch(
            skill=self,
            exact_tool_events=getattr(backend, "uses_exact_tool_events", False)
            and agprof.enabled(),
        )
        try:
            result, updated_ctx, delta = backend.execute(
                ag,
                prev_ctx,
                skill_input,
                max_steps,
                skill=self,
                extra_system=extra_system,
                host_manager=host_manager,
                harness_base_url=harness_base_url,
                launch=launch,
            )
        finally:
            launch.unregister()
            ag.sandbox.remove_files(_offloaded_paths)
        if not isinstance(result, agerror):
            # Give a background job the agent kicked off (e.g. `cmd &` via
            # a bash-style tool call) a chance to finish before this skill
            # call's container gets committed/stopped -- same protection
            # `execute_react()` gives itself.
            #
            # Scoped to `native` only, NOT hoisted for every engine as
            # originally planned: a real-Bedrock/real-`claude` regression
            # test run surfaced that the 4 external-harness engines leave
            # ptrace-tracked child PIDs in `agsandbox_backend._watched_pids`
            # that never receive an `ingest_ptrace_pids(exited=...)` call
            # even long after the harness CLI's own top-level process has
            # exited (confirmed: `wait_for_processes()` blocked for the
            # full 5-minute `ping_interval_s` on 7 real claude_code.py
            # end-to-end tests before this was narrowed to native-only).
            # That looks like a pre-existing gap in agproxy_ptrace's PID
            # exit-event delivery, never exercised before because nothing
            # called `wait_for_processes()` for a harness-driven engine
            # until this hoist -- a separate investigation, not something
            # to paper over here. Native's own persistent entrypoint
            # process is deliberately excluded from monitoring instead
            # (`agSandbox.release_daemon()`, see native.py's
            # `launch_in_container_entrypoint`), which is what makes this
            # safe for native specifically.
            # [REFACTOR] Why do we wait on the host side? Check process tracking implementation
            if ag.harness == "native":
                agSandbox.wait_for_processes(  # [REFACTOR] Returns a message, should be inside the container.
                    ag.sandbox,
                    self.name,
                    ag.terminal,
                    ag.log,
                    str(ag.agname),
                    type(ag).ping_interval_s,
                    type(ag).poll_interval_s,
                    ag._set_ui_state,
                )
            if self.output_schema is not None:
                self.output_schema.recover_outputs(result, ag.sandbox)
        return result, updated_ctx, delta

    def __repr__(self) -> str:
        return f"agskill(name={self.name!r})"
