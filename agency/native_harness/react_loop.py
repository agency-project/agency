"""The ReAct loop for the standalone native harness.

Ties together everything else in this package: dispatch (`llm_client.py`),
compaction (`compaction.py`), built-in tools (`tools.py`), MCP tools
(`mcp_client.py`), and the optional agency bridge (`bridge_client.py`) for
per-tool policy checks and lifecycle checkpoints. Same shape as the old
`_native_in_container_entrypoint.py`'s `_run_react_loop_inner`, adapted to
this package's own dependencies instead of a UDS connection to
`agllm_terminus`/`agmcp_server`/`agharness_messenger`.

Order per turn follows explicit safe boundaries: checkpoint and render
steering, compact, generate, checkpoint the model result, then checkpoint
after every published tool result."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import tools
from .bridge_client import (
    CONTROL_PHASE_BOUNDARY,
    CONTROL_PHASE_CLOSING,
    CONTROL_PHASE_MODEL,
)
from .compaction import maybe_compact

if TYPE_CHECKING:
    from .bridge_client import BridgeClient
    from .llm_client import LLMClient
    from .mcp_client import McpToolset

_DEFAULT_MAX_STEPS = 20
_AGENT_DESTROYED_MESSAGE = "agent destroyed"
_INVOCATION_CANCELLED_MESSAGE = "agent invocation cancelled"


@dataclass
class ReactLoopResult:
    status: str  # "done" | "error"
    messages: "list[dict] | None" = None
    final_text: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    message: str = ""
    turn_count: int = 0


def run_react_loop(
    messages: list,
    model: str,
    llm: "LLMClient",
    *,
    mcp: "McpToolset | None" = None,
    bridge: "BridgeClient | None" = None,
    context_limit: "int | None" = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
    offload_dir: str = "./long_tool_call_outputs",
) -> ReactLoopResult:
    messages = list(messages)
    total_input_tokens = 0
    total_output_tokens = 0
    previous_summary: "str | None" = None
    rendered_steering_sequences: "set[int]" = set()

    dispatch_table = dict(tools.TOOL_DISPATCH)
    tool_schemas = list(tools.BUILTIN_TOOL_SCHEMAS.values())
    have_tool = set(tools.BUILTIN_TOOL_SCHEMAS.keys())

    mcp_schemas = mcp.discover() if mcp is not None else []
    for schema in mcp_schemas:
        name = schema["function"]["name"]
        if name in have_tool:
            continue  # built-ins take precedence, same as tool-set-collision rules elsewhere
        tool_schemas.append(schema)
        have_tool.add(name)
        dispatch_table[name] = lambda args_json, _name=name: mcp.call(_name, args_json)

    for step in range(max_steps):
        if bridge is not None:
            decision = bridge.checkpoint(
                _native_boundary_id("generation", step, messages),
                allow_steering=True,
                phase=CONTROL_PHASE_MODEL,
            )
            stopped = _stopped_message(decision)
            if stopped is not None:
                return ReactLoopResult(status="error", message=stopped, turn_count=step)
            steering = []
            for entry in decision.get("steering") or []:
                try:
                    sequence = int(entry["sequence"])
                except (KeyError, TypeError, ValueError):
                    continue
                if sequence in rendered_steering_sequences:
                    continue
                rendered_steering_sequences.add(sequence)
                steering.append(str(entry.get("instructions", "")))
            if steering:
                messages.append(
                    {
                        "role": "user",
                        "content": "[AGENCY STEERING]\n" + "\n\n".join(steering),
                    }
                )

        messages, previous_summary = maybe_compact(
            messages, context_limit, llm, model, previous_summary
        )

        if bridge is not None:
            # Compaction bypasses ordinary user steering, but controls may
            # arrive while it is running. Observe them before task generation.
            decision = bridge.checkpoint(
                _native_boundary_id("post-compaction", step, messages),
                allow_steering=False,
                phase=CONTROL_PHASE_MODEL,
            )
            stopped = _stopped_message(decision)
            if stopped is not None:
                return ReactLoopResult(status="error", message=stopped, turn_count=step)

        resp = llm.dispatch(model, messages, tool_schemas or None)
        if "error" in resp:
            return ReactLoopResult(status="error", message=str(resp["error"]), turn_count=step + 1)

        if bridge is not None:
            response_has_tools = bool((resp.get("message") or {}).get("tool_calls"))
            decision = bridge.checkpoint(
                _native_boundary_id("model", step, messages),
                allow_steering=False,
                phase=(CONTROL_PHASE_BOUNDARY if response_has_tools else CONTROL_PHASE_CLOSING),
            )
            stopped = _stopped_message(decision)
            if stopped is not None:
                return ReactLoopResult(status="error", message=stopped, turn_count=step + 1)

        usage = resp.get("usage") or {}
        total_input_tokens += usage.get("prompt_tokens", 0) or 0
        total_output_tokens += usage.get("completion_tokens", 0) or 0

        message = resp["message"]
        messages.append(message)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return ReactLoopResult(
                status="done",
                messages=messages,
                final_text=message.get("content") or "",
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                turn_count=step + 1,
            )

        for tool_index, tc in enumerate(tool_calls):
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]
            handler = dispatch_table.get(fn_name)
            if handler is None:
                result_content = json.dumps({"error": f"unknown tool: {fn_name}"})
            elif bridge is not None:
                decision = bridge.check_tool_policy(fn_name, _parse_tool_input(fn_args))
                if decision.get("decision") == "deny":
                    result_content = json.dumps(
                        {"error": f"denied by policy: {decision.get('reason', 'no reason given')}"}
                    )
                else:
                    result_content = handler(fn_args)
            else:
                result_content = handler(fn_args)
            result_content = tools.offload_if_oversized(
                fn_name, tc["id"], result_content, offload_dir
            )
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result_content})
            if bridge is not None:
                decision = bridge.checkpoint(
                    f"native:tool:{step}:{tool_index}:{tc['id']}",
                    allow_steering=False,
                    phase=CONTROL_PHASE_BOUNDARY,
                )
                stopped = _stopped_message(decision)
                if stopped is not None:
                    return ReactLoopResult(
                        status="error",
                        message=stopped,
                        turn_count=step + 1,
                    )

    return ReactLoopResult(
        status="error",
        message=f"exceeded max_steps={max_steps} without a final answer",
        turn_count=max_steps,
    )


def _parse_tool_input(fn_args: str) -> dict:
    try:
        parsed = json.loads(fn_args) if fn_args else {}
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _native_boundary_id(kind: str, step: int, messages: list) -> str:
    payload = json.dumps(messages, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"native:{kind}:{step}:{digest}"


def _stopped_message(decision: dict) -> "str | None":
    if decision.get("destroyed"):
        return _AGENT_DESTROYED_MESSAGE
    if decision.get("cancelled"):
        return str(decision.get("error") or _INVOCATION_CANCELLED_MESSAGE)
    return None


__all__ = ["run_react_loop", "ReactLoopResult"]
