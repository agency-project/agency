from __future__ import annotations
import functools
import json
from typing import TYPE_CHECKING
from .agdata import agdata, agerror
from .agpolicy import agpolicy
from .agtype import agtype
from .agschema import agschema
from .agtool import agtool
from .agconfig import DynamicConfigParam, _AgConfigViewBase


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
        resource_pool.acquire_cpu_mem(sandbox, cpus=cpus, memory_mb=memory_mb)
        messages.append(f"cpus={cpus}, memory_mb={memory_mb}")

    if gpu:
        sandbox._gpu_count_requested = gpu
        sandbox._gpu_acquire_fn = functools.partial(resource_pool.acquire_gpus, sandbox)
        sandbox._gpu_release_fn = functools.partial(resource_pool.release_gpus, sandbox)
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
        resource_pool.release_cpu_mem(sandbox, cpu=cpu, memory=memory)
        messages.append(f"cpu={cpu}, memory={memory} reset to idle")

    if gpu:
        if sandbox._gpu_count_requested > 0:
            if sandbox._gpu_ids:
                resource_pool.release_gpus(sandbox, sandbox._gpu_ids)
            messages.append(f"gpu reservation ({sandbox._gpu_count_requested}) released")
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

    # ------------------------------------------------------------------
    # Scheduling wrapper — non-blocking, returns pending agdata
    # ------------------------------------------------------------------

    def run(
        self,
        ag: "agent",
        skill_input: agdata,
        max_steps: "int | None" = None,
    ) -> agdata:
        """Submit through the process-wide event-driven orchestrator."""
        from .orchestrator import get_orchestrator

        orchestrator = get_orchestrator(ag.agconfig)
        return orchestrator.submit(ag, self, skill_input, max_steps=max_steps)

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

    def __repr__(self) -> str:
        return f"agskill(name={self.name!r})"
