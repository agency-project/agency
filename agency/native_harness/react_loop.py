"""The ReAct loop for the standalone native harness.

Ties together everything else in this package: dispatch (`llm_client.py`),
compaction (`compaction.py`), built-in tools (`tools.py`), MCP tools
(`mcp_client.py`), and the optional agency bridge (`bridge_client.py`) for
per-tool policy checks. Talks to this package's own in-process
dependencies directly, not over a UDS connection to
`agllm_terminus`/`agmcp_server`/`agharness_messenger`.

This loop carries no invocation-lifecycle state of its own: cancellation
propagates the same way it does for every other harness -- a denied tool
call (`bridge.check_tool_policy`) or a denied/errored model dispatch
(`llm.dispatch`), both mediated by the same host-side admission checks any
external harness goes through. There is no separate "checkpoint" concept."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import tools
from .profiling import profile_run, span as profile_span
from .compaction import maybe_compact
from .annotations import augment_schemas, extract, instruction, event_id, redact

if TYPE_CHECKING:
    from .bridge_client import BridgeClient
    from .llm_client import LLMClient
    from .mcp_client import McpToolset

_DEFAULT_MAX_STEPS = 4096


def _write_progress(
    progress_path: "str | None",
    messages: list,
    total_input_tokens: int,
    total_output_tokens: int,
    turn_count: int,
) -> None:
    """Best-effort checkpoint a host-side caller can poll for liveness and,
    if it gives up waiting, recover a partial answer from -- mtime is the
    activity signal, `final_text` is whatever the loop last actually said.
    Never lets a checkpoint failure interrupt the loop it's observing."""
    if progress_path is None:
        return
    final_text = ""
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            final_text = message["content"]
            break
    payload = {
        "turn_count": turn_count,
        "final_text": final_text,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }
    try:
        tmp_path = f"{progress_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, progress_path)
    except OSError:
        pass


@dataclass
class ReactLoopResult:
    status: str  # "done" | "error"
    messages: "list[dict] | None" = None
    final_text: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    message: str = ""
    turn_count: int = 0


@profile_run
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
    progress_path: "str | None" = None,
    annotation_arm: str = "baseline",
    observer=None,
    toolset: "tuple[list[dict], dict] | None" = None,
) -> ReactLoopResult:
    messages = list(messages)
    total_input_tokens = 0
    total_output_tokens = 0
    previous_summary: "str | None" = None

    dispatch_table = dict(tools.TOOL_DISPATCH) if toolset is None else dict(toolset[1])
    tool_schemas = (
        list(tools.BUILTIN_TOOL_SCHEMAS.values()) if toolset is None else list(toolset[0])
    )
    have_tool = {schema["function"]["name"] for schema in tool_schemas}

    mcp_schemas = mcp.discover() if mcp is not None else []
    for schema in mcp_schemas:
        name = schema["function"]["name"]
        if name in have_tool:
            continue  # built-ins take precedence, same as tool-set-collision rules elsewhere
        tool_schemas.append(schema)
        have_tool.add(name)
        dispatch_table[name] = lambda args_json, _name=name: mcp.call(_name, args_json)

    tool_schemas = augment_schemas(tool_schemas, annotation_arm)
    treatment_instruction = instruction(annotation_arm)
    if treatment_instruction:
        messages.insert(0, {"role": "system", "content": treatment_instruction})
    if observer is not None:
        observer(
            "treatment",
            {
                "arm": annotation_arm,
                "schemas": tool_schemas,
                "instruction": treatment_instruction,
                "initial_messages": messages,
            },
        )

    for step in range(max_steps):
        with profile_span(bridge, f"turn{step}"):
            before_compaction = messages
            messages, previous_summary = maybe_compact(
                messages, context_limit, llm, model, previous_summary
            )
            if observer is not None and messages != before_compaction:
                observer("compaction", {"exchange": step, "messages": messages})

            dispatch_started = time.perf_counter_ns()
            resp = llm.dispatch(model, messages, tool_schemas or None)
            if observer is not None:
                observer(
                    "model_exchange",
                    {
                        "exchange": step,
                        "response": resp,
                        "duration_ns": time.perf_counter_ns() - dispatch_started,
                    },
                )
            if "error" in resp:
                return ReactLoopResult(
                    status="error", message=str(resp["error"]), turn_count=step + 1
                )

            usage = resp.get("usage") or {}
            total_input_tokens += usage.get("prompt_tokens", 0) or 0
            total_output_tokens += usage.get("completion_tokens", 0) or 0

            message = resp["message"]
            tool_calls = message.get("tool_calls") or []
            messages.append(message)
            if not tool_calls:
                return ReactLoopResult(
                    status="done",
                    messages=messages,
                    final_text=message.get("content") or "",
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    turn_count=step + 1,
                )
            _write_progress(
                progress_path, messages, total_input_tokens, total_output_tokens, step + 1
            )

            prepared_calls = []
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"]["arguments"]
                fn_args, annotation = extract(fn_args, annotation_arm)
                trace_id = event_id() if observer is not None else None
                correlation = {
                    "event_id": trace_id,
                    "model_tool_call_id": tc["id"],
                    "exchange": step,
                    "batch_size": len(tool_calls),
                    "execution": "serial",
                    "tool_name": fn_name,
                }
                if observer is not None:
                    observer(
                        "tool_annotation",
                        {
                            **correlation,
                            "annotation": annotation,
                            "arguments": _parse_tool_input(fn_args),
                            "arguments_json": fn_args,
                            "call_id": None,
                        },
                    )
                prepared_calls.append((tc, fn_name, fn_args, correlation, annotation))

            for tc, fn_name, fn_args, correlation, annotation in prepared_calls:
                handler = dispatch_table.get(fn_name)
                call_id = None
                tool_duration_ns = None
                if handler is None:
                    result_content = json.dumps({"error": f"unknown tool: {fn_name}"})
                elif bridge is not None:
                    decision = bridge.check_tool_policy(
                        fn_name,
                        _parse_tool_input(fn_args),
                        annotation=redact(
                            {**annotation, "model_tool_call_id": tc["id"]},
                            (getattr(bridge, "token", None),),
                        ),
                    )
                    call_id = decision.get("call_id")
                    if observer is not None:
                        observer(
                            "tool_admission",
                            {**correlation, "call_id": call_id, "decision": decision},
                        )
                    if decision.get("decision") == "deny":
                        result_content = json.dumps(
                            {
                                "error": f"denied by policy: {decision.get('reason', 'no reason given')}"
                            }
                        )
                    else:
                        tool_started_ns = time.perf_counter_ns()
                        tool_started_wall_ns = time.time_ns()
                        try:
                            result_content = handler(fn_args)
                        except Exception as exc:
                            result_content = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
                        tool_duration_ns = time.perf_counter_ns() - tool_started_ns
                else:
                    observer_started_ns = time.perf_counter_ns() if observer is not None else None
                    try:
                        result_content = handler(fn_args)
                    except Exception as exc:
                        result_content = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
                    if observer_started_ns is not None:
                        tool_duration_ns = time.perf_counter_ns() - observer_started_ns
                result_error = _parse_tool_input(result_content).get("error")
                if observer is not None:
                    result_data = _parse_tool_input(result_content)
                    failed = bool(result_error) or result_data.get("isError") is True
                    failed = failed or any(
                        result_data.get(key) not in (None, 0) for key in ("returncode", "exit_code")
                    )
                    observer(
                        "tool_result",
                        {
                            **correlation,
                            "call_id": call_id,
                            "result": result_content,
                            "duration_ns": tool_duration_ns,
                            "category": "error" if failed else "ok",
                        },
                    )
                result_content = tools.offload_if_oversized(
                    fn_name, tc["id"], result_content, offload_dir
                )
                if bridge is not None:
                    if tool_duration_ns is not None:
                        bridge.complete_tool_policy(
                            call_id,
                            result_content,
                            duration_ns=tool_duration_ns,
                            started_wall_ns=tool_started_wall_ns,
                            error=str(result_error) if result_error is not None else None,
                        )
                    else:
                        bridge.complete_tool_policy(call_id, result_content)
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": result_content}
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


__all__ = ["run_react_loop", "ReactLoopResult"]
