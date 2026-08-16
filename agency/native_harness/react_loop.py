"""The ReAct loop for the standalone native harness.

Ties together everything else in this package: dispatch (`llm_client.py`),
compaction (`compaction.py`), built-in tools (`tools.py`), MCP tools
(`mcp_client.py`), and the optional agency bridge (`bridge_client.py`) for
per-tool policy checks and pause/inbox check-in. Same shape as the old
`_native_in_container_entrypoint.py`'s `_run_react_loop_inner`, adapted to
this package's own dependencies instead of a UDS connection to
`agllm_terminus`/`agmcp_server`/`agharness_messenger`.

Order per turn matches `execute_react()`'s own (see the old entrypoint's
docstring): check pause/drain inbox, THEN compact, THEN dispatch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import tools
from .compaction import maybe_compact

if TYPE_CHECKING:
    from .bridge_client import BridgeClient
    from .llm_client import LLMClient
    from .mcp_client import McpToolset

_DEFAULT_MAX_STEPS = 20


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
    no_builtin_tools: bool = False,
) -> ReactLoopResult:
    messages = list(messages)
    total_input_tokens = 0
    total_output_tokens = 0
    previous_summary: "str | None" = None

    dispatch_table = {} if no_builtin_tools else dict(tools.TOOL_DISPATCH)
    tool_schemas = [] if no_builtin_tools else list(tools.BUILTIN_TOOL_SCHEMAS.values())
    have_tool = set() if no_builtin_tools else set(tools.BUILTIN_TOOL_SCHEMAS.keys())

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
            messages.extend(bridge.check_in())

        messages, previous_summary = maybe_compact(
            messages, context_limit, llm, model, previous_summary
        )

        resp = llm.dispatch(model, messages, tool_schemas or None)
        if "error" in resp:
            return ReactLoopResult(status="error", message=str(resp["error"]), turn_count=step + 1)

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

        for tc in tool_calls:
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


__all__ = ["run_react_loop", "ReactLoopResult"]
