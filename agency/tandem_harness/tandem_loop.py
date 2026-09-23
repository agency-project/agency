"""The two-model tandem loop: a supervisor model keeps the full running
task history and never sees the worker's tool schema; a worker model
carries out one supervisor order per fresh, stateless session (no state
carries from one order to the next -- whatever continuity is needed is the
supervisor's job to put in the next order's text), and reports back a
small structured result built directly from its own transcript, plus its
own natural-language summary of what it did.

Both levels are just `run_react_loop` (see that module's docstring): the
supervisor's turn-taking IS a `run_react_loop` call with its built-in tools
swapped out for three synthetic actions (`send_order`, `get_trace`,
`get_tool_call_detail`) plus one real one, `submit_output`, dispatched
straight through the shared `McpToolset` (the same connection the worker's
own MCP tools use) rather than through a synthetic handler; each worker
segment is a second, independent `run_react_loop` call (untouched
built-ins: Bash/Read/Write/Edit/Glob/Grep/WebFetch/TodoWrite/MCP) nested
inside `send_order`'s own handler. `submit_output` is a host-side
bookkeeping call (record one output field's value), not sandbox execution,
so it belongs to the supervisor -- the one with full task context -- not
the worker: routing task completion through send_order gave the supervisor
no clean way to recognize "done" and stop, so it would keep re-sending
"the task is complete" as if it were still an order for the worker to
carry out. No dedicated "finish" tool beyond that: the supervisor
completes the same way every ReAct loop in this package already does --
a turn with no tool call -- so a stray toolless turn and a deliberate
finish aren't different code paths, just like they aren't for
native/codex/claude_code today.

`segment_step_cap` (default 16) is a soft ceiling, not a tight per-order
budget: the worker may make several tool calls to satisfy one order if it
needs to, and `WORKER_SYSTEM` instructs it to finish with a concise
natural-language summary once it's done rather than trailing off. Each
`send_order` call returns exactly three fields -- `tool_calls` (mechanical,
harness-extracted, never model-authored: each entry has a `call_id` plus
`arguments`/`result` truncated to a fixed preview length as a cheap
cross-reference), `summary_text` (the worker's own natural-language account
of what happened), and `finish_reason` (`worker_finished` /
`step_cap_reached` / `worker_error` / `invalid_order`). `get_trace()` (no
arguments) returns the full, untruncated tool-call trace of the
immediately preceding `send_order` call; `get_tool_call_detail(call_id)`
returns just the one entry matching that `call_id`. Both are served from an
in-process cache of that same execution, so neither costs an extra worker
LLM call or re-execution."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

_DEBUG = bool(os.environ.get("TANDEM_DEBUG"))


def _debug(msg: str) -> None:
    if _DEBUG:
        print(f"[tandem_debug] {msg}", file=sys.stderr, flush=True)


from .react_loop import run_react_loop

if TYPE_CHECKING:
    from .bridge_client import BridgeClient
    from .llm_client import LLMClient
    from .mcp_client import McpToolset

_DEFAULT_MAX_SEGMENTS = 4096
# A soft ceiling, not a tight per-order budget -- see this module's docstring.
DEFAULT_SEGMENT_STEP_CAP = 16

# Preview lengths for the mechanical tool_calls cross-reference in a
# send_order report. Always applied (there is no "give me the full thing in
# this same call" flag) -- get_trace() is the escape hatch for full detail.
ARGUMENTS_TRUNCATE_CHARS = 50
RESULT_TRUNCATE_CHARS = 200

_SUPERVISOR_TOOL_NAMES = {"send_order", "get_trace", "get_tool_call_detail"}

SUPERVISOR_SYSTEM = """\
You are a helpful assistant that carries out a task given to you by a user.
You have access to a harness and execution environment to carry out the task through a natural language-based interface, accesible via the send_order tool.
You are ONLY given the following four tools:

- send_order(order): Give the harness an instruction to carry out. The instruction should be conscise and explicit and in natural language. You should provide specific instructions of the order. Returns a truncated tool_calls list, a summary_text, and a finish_reason.

- get_trace(): Get the full, untruncated trace of the most recent send_order execution, including the input and output of every tool call it made. Use this when send_order's own truncated tool_calls preview or summary_text doesn't have enough detail for your decision-making.

- get_tool_call_detail(call_id): Get the full, untruncated arguments and result of just ONE tool call from the most recent send_order execution, identified by the call_id in that send_order result's tool_calls entries. Prefer this over get_trace() when you only need one call's detail, not the whole trace.

- submit_output(field, value): Submit one required output field's value to the user. Use the exact field name and description given to you in this task's own instructions. Call it once per required field.

Example:
```
send_order("Find files that contain the string 'JSON' using glob.")
send_order("Read file ./file.json and return the keys using grep.")
send_order("Show me the function parse_json in the file ./file.json.")
send_order("Replace the function parse_json in the file ./file.json with ...")
send_order("Run JSONFieldTests.test_has_key_number via runtests.py")
get_tool_call_detail("call_1") -> returns the full, untruncated arguments/result of that one tool call.
get_trace() -> returns the full trace of the runtests.py execution, including the input and output of the tools.
... more orders ...
submit_output("field_name", "final value") -> once per required output field, when you have the confirmed final value to be reported to the user.
```
ALL other tools, such as "bash" or "read", are not available to you. You will have to use "send_order" to carry out the task. You may need to give explicit bash commands or code snippets to the worker to carry out the task.

For task execution, please keep going until the query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved.

If you are working on a codebase that has tests or the ability to build or run, consider using them to verify that your work is complete. Your philosophy should be to start as specific as possible to the code you changed so that you can catch issues efficiently, then make your way to broader tests as you build confidence. Fix the problem at the root cause rather than applying surface-level patches, when possible.

Once you have confirmed the final output values for every required output field, call submit_output for each field and provided a summary of the work you did (no further tool call) -- that ends the task.
"""

WORKER_SYSTEM = """
You are a tool-execution worker. You will be given a single instruction.
Carry out exactly what it asks using the tools available to you, then stop.
If the instruction is a question you can answer directly without a tool, just answer it in text. Return a consicise summary of the work you did, with the result of each tool call and the final result of the task achieved, within 1000 characters.
"""

_SEND_ORDER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "send_order",
        "description": (
            "Give the worker exactly one instruction to carry out. Returns "
            "{tool_calls, summary_text, finish_reason} -- tool_calls entries (each with a "
            "call_id) are truncated previews; call get_trace() or "
            "get_tool_call_detail(call_id) afterward for the full detail."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order": {
                    "type": "string",
                    "description": "The single instruction for the worker to carry out.",
                },
            },
            "required": ["order"],
        },
    },
}

_GET_TRACE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_trace",
        "description": (
            "Return the full, untruncated tool call trace (arguments and results) of the "
            "most recent send_order execution."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

_GET_TOOL_CALL_DETAIL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_tool_call_detail",
        "description": (
            "Return the full, untruncated arguments and result of one specific tool call "
            "(by call_id) from the most recent send_order execution's trace."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "call_id": {
                    "type": "string",
                    "description": "The call_id of the tool call, from a send_order tool_calls entry.",
                },
            },
            "required": ["call_id"],
        },
    },
}

# Mirrors the Agency MCP server's own submit_output tool schema
# (agskill.py's _DEFAULT_HOST_MCP_TOOLS) verbatim -- the supervisor is the
# one that calls this directly (see run_tandem_loop's docstring), never the
# worker, since it's a host-side bookkeeping call (record one output
# field's value), not sandbox execution.
_SUBMIT_OUTPUT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_output",
        "description": "Submit one required output field's value.",
        "parameters": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "description": "output field name"},
                "value": {"description": "the field's value"},
            },
            "required": ["field", "value"],
        },
    },
}


@dataclass
class TandemLoopResult:
    status: str  # "done" | "error"
    messages: "list[dict] | None" = None
    final_text: str = ""
    supervisor_input_tokens: int = 0
    supervisor_output_tokens: int = 0
    worker_input_tokens: int = 0
    worker_output_tokens: int = 0
    segment_count: int = 0
    message: str = ""


def _parse_args(fn_args: str) -> dict:
    try:
        parsed = json.loads(fn_args) if fn_args else {}
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "[TRUNCATED]"


def _extract_full_trace(result) -> "list[dict]":
    """The worker segment's raw tool-call trace, untruncated -- what
    get_trace() serves back verbatim, and what _build_report() truncates for
    send_order's own return value. `result` is the worker's own
    ReactLoopResult; `arguments`/`result` here are the raw strings the API
    actually carried, never re-parsed into a dict -- this is a mechanical
    transcript read, not model-authored content."""
    trace: "list[dict]" = []
    by_call_id: "dict[str, dict]" = {}
    for m in result.messages or []:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                entry = {
                    "call_id": tc["id"],
                    "tool": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"] or "",
                    "result": "",
                }
                trace.append(entry)
                by_call_id[tc["id"]] = entry
        elif m.get("role") == "tool":
            entry = by_call_id.get(m.get("tool_call_id"))
            if entry is not None:
                entry["result"] = m.get("content") or ""
    return trace


def _build_report(result) -> "tuple[dict, list[dict]]":
    """Returns (report, full_trace). `report` is exactly what send_order
    hands back to the supervisor -- tool_calls truncated to
    ARGUMENTS_TRUNCATE_CHARS/RESULT_TRUNCATE_CHARS as a cheap mechanical
    cross-reference, alongside the worker's own summary_text. `full_trace`
    is the untruncated version, cached by the caller for get_trace().
    `result` is the worker's own ReactLoopResult."""
    full_trace = _extract_full_trace(result)
    truncated_calls = [
        {
            "call_id": entry["call_id"],
            "tool": entry["tool"],
            "arguments": _truncate(entry["arguments"], ARGUMENTS_TRUNCATE_CHARS),
            "result": _truncate(entry["result"], RESULT_TRUNCATE_CHARS),
        }
        for entry in full_trace
    ]

    if result.status == "error" and (result.message or "").startswith("exceeded max_steps="):
        finish_reason = "step_cap_reached"
        summary_text = None
    elif result.status == "error":
        finish_reason = "worker_error"
        summary_text = result.message
    else:
        finish_reason = "worker_finished"
        summary_text = result.final_text or None

    report = {
        "tool_calls": truncated_calls,
        "summary_text": summary_text,
        "finish_reason": finish_reason,
    }
    return report, full_trace


def run_tandem_loop(
    messages: list,
    supervisor_model: str,
    worker_model: str,
    supervisor_llm: "LLMClient",
    worker_llm: "LLMClient",
    *,
    mcp: "McpToolset | None" = None,
    bridge: "BridgeClient | None" = None,
    worker_context_limit: "int | None" = None,
    segment_step_cap: int = DEFAULT_SEGMENT_STEP_CAP,
    max_segments: int = _DEFAULT_MAX_SEGMENTS,
    offload_dir: str = "./long_tool_call_outputs",
    progress_path: "str | None" = None,
) -> TandemLoopResult:
    worker_totals = {"input": 0, "output": 0}
    segment_counter = {"n": 0}
    # The full, untruncated trace of the most recent send_order execution --
    # get_trace() serves this back verbatim, no re-execution needed.
    last_trace: "dict[str, list[dict] | None]" = {"tool_calls": None}

    def send_order_handler(args_json: str) -> str:
        args = _parse_args(args_json)
        order = (args.get("order") or "").strip()
        if not order:
            return json.dumps(
                {
                    "tool_calls": [],
                    "summary_text": "send_order called with an empty order",
                    "finish_reason": "invalid_order",
                }
            )

        segment_index = segment_counter["n"]
        segment_counter["n"] += 1
        _debug(f"order[{segment_index}]: {order}")
        worker_messages = [
            # The bridged live transcript/webui has no structured way (yet)
            # to distinguish supervisor turns from worker-segment turns --
            # they share one agent identity end to end. The tag lives on the
            # user turn, not the system turn: the webui's live log doesn't
            # render system-role messages at all, so a tag placed there is
            # invisible -- confirmed by inspecting an actual run's log.
            # Each segment gets its own fresh dispatch, so this reappears
            # once per segment.
            {"role": "system", "content": WORKER_SYSTEM},
            {"role": "user", "content": f"[TANDEM WORKER segment {segment_index}] {order}"},
        ]
        result = run_react_loop(
            worker_messages,
            worker_model,
            worker_llm,
            mcp=mcp,
            bridge=bridge,
            context_limit=worker_context_limit,
            max_steps=segment_step_cap,
            offload_dir=offload_dir,
            span_prefix=f"worker_seg{segment_index}_turn",
            # Only meaningful when bridged: agmanager_harness interprets/strips
            # this field before forwarding to the real provider. Standalone
            # (bridge is None), there's nothing to strip it, and passing it
            # straight through to litellm/Bedrock is a bad request.
            internal_kind=("tandem_worker" if bridge is not None else None),
        )
        worker_totals["input"] += result.total_input_tokens
        worker_totals["output"] += result.total_output_tokens
        report, full_trace = _build_report(result)
        last_trace["tool_calls"] = full_trace
        _debug(f"report[{segment_index}]: {json.dumps(report)[:500]}")
        return json.dumps(report)

    def get_trace_handler(_args_json: str) -> str:
        if last_trace["tool_calls"] is None:
            return json.dumps({"error": "no previous send_order execution to trace"})
        return json.dumps({"tool_calls": last_trace["tool_calls"]})

    def get_tool_call_detail_handler(args_json: str) -> str:
        args = _parse_args(args_json)
        call_id = (args.get("call_id") or "").strip()
        if not call_id:
            return json.dumps({"error": "get_tool_call_detail called with an empty call_id"})
        trace = last_trace["tool_calls"]
        if trace is None:
            return json.dumps({"error": "no previous send_order execution to look up"})
        for entry in trace:
            if entry["call_id"] == call_id:
                return json.dumps(entry)
        return json.dumps(
            {"error": f"no tool call with call_id={call_id!r} in the most recent trace"}
        )

    supervisor_tool_schemas = [_SEND_ORDER_SCHEMA, _GET_TRACE_SCHEMA, _GET_TOOL_CALL_DETAIL_SCHEMA]
    supervisor_dispatch_table = {
        "send_order": send_order_handler,
        "get_trace": get_trace_handler,
        "get_tool_call_detail": get_tool_call_detail_handler,
    }
    if mcp is not None:
        # Discover once up front, not lazily inside the lambda below: the
        # supervisor may call submit_output as its very first action (there's
        # nothing stopping it from doing so before any send_order), so
        # McpToolset's tool_name -> server routing must already be populated
        # by the time that first call happens.
        mcp.discover()
        supervisor_tool_schemas.append(_SUBMIT_OUTPUT_SCHEMA)
        supervisor_dispatch_table["submit_output"] = lambda args_json: mcp.call(
            "submit_output", args_json
        )

    result = run_react_loop(
        messages,
        supervisor_model,
        supervisor_llm,
        bridge=bridge,
        max_steps=max_segments,
        offload_dir=offload_dir,
        progress_path=progress_path,
        tool_schemas=supervisor_tool_schemas,
        dispatch_table=supervisor_dispatch_table,
        # submit_output is a real host tool call, not a synthetic
        # control-flow action -- it goes through the same bridge policy
        # check any other real tool call does, so it's deliberately left
        # out of policy_exempt_tools.
        policy_exempt_tools=_SUPERVISOR_TOOL_NAMES,
        span_prefix="supervisor_turn",
        internal_kind=("tandem_supervisor" if bridge is not None else None),
    )

    return TandemLoopResult(
        status="done" if result.status == "done" else "error",
        messages=result.messages,
        final_text=result.final_text,
        supervisor_input_tokens=result.total_input_tokens,
        supervisor_output_tokens=result.total_output_tokens,
        worker_input_tokens=worker_totals["input"],
        worker_output_tokens=worker_totals["output"],
        segment_count=segment_counter["n"],
        message=result.message,
    )


__all__ = ["run_tandem_loop", "TandemLoopResult", "DEFAULT_SEGMENT_STEP_CAP"]
