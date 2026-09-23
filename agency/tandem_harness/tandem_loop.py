"""The two-model tandem loop: a supervisor model keeps the full running
task history and never sees the worker's tool schema; a worker model
carries out one smart_tool task per segment, and reports back a small
structured result built directly from its own transcript, plus its own
natural-language summary of what it did. The worker's own conversation
carries forward across segments by default -- up to `worker_history_turns`
(4096) past segments' worth of its own messages replayed as the start of
the next one's context (see `_trim_worker_history`) -- rather than each
segment being a wholly fresh session; a worker that already found where
the repo lives doesn't need the supervisor to keep re-stating it every
task, though the supervisor should still put whatever continuity actually
matters in the task text itself, both because `worker_history_turns` is a
bound, not a guarantee (old segments do eventually drop off), and because
that's a much shorter, more targeted signal than hoping the right detail
survives however many segments of replayed history.

Both levels are just `run_react_loop` (see that module's docstring): the
supervisor's turn-taking IS a `run_react_loop` call with its built-in tools
swapped out for three synthetic actions (`smart_tool`, `get_trace`,
`get_tool_call_detail`) plus one real one, `submit_output`, dispatched
straight through the shared `McpToolset` (the same connection the worker's
own MCP tools use) rather than through a synthetic handler; each worker
segment is a second, independent `run_react_loop` call (untouched
built-ins: Bash/Read/Write/Edit/Glob/Grep/WebFetch/TodoWrite/MCP) nested
inside `smart_tool`'s own handler. `smart_tool` is deliberately framed to
the supervisor as an ordinary tool call (natural-language task in, result
out) rather than as delegation to a subordinate -- a failed/incomplete
result should trigger the same "adjust the input and call again" reflex
any other tool failure would, not an executive decision to give up or
route around missing verification. `submit_output` is a host-side
bookkeeping call (record one output field's value), not sandbox execution,
so it belongs to the supervisor -- the one with full task context -- not
the worker: routing task completion through smart_tool gave the supervisor
no clean way to recognize "done" and stop, so it would keep re-sending
"the task is complete" as if it were still a task for the worker to
carry out. No dedicated "finish" tool beyond that: the supervisor
completes the same way every ReAct loop in this package already does --
a turn with no tool call -- so a stray toolless turn and a deliberate
finish aren't different code paths, just like they aren't for
native/codex/claude_code today.

`segment_step_cap` (default 16) is a soft ceiling, not a tight per-task
budget: the worker may make several tool calls to satisfy one task if it
needs to, and `WORKER_SYSTEM` instructs it to finish with a concise
natural-language summary once it's done rather than trailing off. Each
`smart_tool` call returns exactly three fields -- `tool_calls` (mechanical,
harness-extracted, never model-authored: each entry has a `call_id` plus
`arguments`/`result` truncated to a fixed preview length as a cheap
cross-reference), `tool_output` (the worker's own factual account of what
it ran and what happened -- `WORKER_SYSTEM` tells it to report like a tool
returning output, not offer its own diagnosis/beliefs), and `finish_reason`
(`worker_finished` /
`step_cap_reached` / `worker_error` / `invalid_task`). `get_trace()` (no
arguments) returns the full, untruncated tool-call trace of the
immediately preceding `smart_tool` call; `get_tool_call_detail(call_id)`
returns just the one entry matching that `call_id`. Both are served from an
in-process cache of that same execution, so neither costs an extra worker
LLM call or re-execution.

The worker also gets one extra tool of its own, `forward_tool_output`: for
a task a single tool call already answers in full (a file listing, a
grep/read result), retyping that content into a closing summary is pure
model-generated duplication of something already mechanically known --
slow, and pointless for a small worker model in particular. It takes no
arguments and always means "my most recently completed tool call" --
deliberately NOT a call_id the worker names, since a tool call's id is
OpenAI-protocol metadata (`tc["id"]`/`tool_call_id`), never text placed in
any message's own content, so there's no guarantee a given serving
template even renders it back into the worker's own prompt for the model
to read and reproduce. Position ("the last thing I just ran") needs no
such guarantee. `call_id_results`, populated by `react_loop.py` as it
dispatches each call (see that module's docstring), is an insertion-order
dict; forward_tool_output_handler just takes its last value and stages it
to be appended to `tool_output` verbatim once the segment ends -- no
extra worker LLM call, no regeneration."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

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

# How many previous smart_tool segments' own messages (task line through
# that segment's concluding turn) get replayed into a fresh segment's
# starting context, oldest dropped first once the cap is exceeded. Large by
# default -- for a real run this is well above the segment count most tasks
# ever reach, so in practice nothing gets dropped; it's a bound for
# pathological runs, not a tuned working set size yet.
DEFAULT_WORKER_HISTORY_TURNS = 4096

# Preview lengths for the mechanical tool_calls cross-reference in a
# smart_tool report. Always applied (there is no "give me the full thing in
# this same call" flag) -- get_trace() is the escape hatch for full detail.
ARGUMENTS_TRUNCATE_CHARS = 80
RESULT_TRUNCATE_CHARS = 160

_SUPERVISOR_TOOL_NAMES = {"smart_tool", "get_trace", "get_tool_call_detail"}

SUPERVISOR_SYSTEM = """\
You are a helpful assistant that carries out a task given to you by a user.
You are only given the following four tools:

- smart_tool(task): Execute a task expressed in natural language on the harness by decomposing it into a list of basic tool calls. Returns the tool_output of the task, the list of basic tool calls performed, and a finish_reason.

- get_trace(): Get the full, untruncated trace of the decomposed base tool calls from the most recent smart_tool call, including the input and output of every tool call. Use this when smart_tool's basic tool call list doesn't have enough detail for your decision-making.

- get_tool_call_detail(call_id): Get the full, untruncated arguments and result of just ONE tool call from the most recent smart_tool call, identified by the call_id in that smart_tool result's tool_calls entries. Prefer this over get_trace() when you only need one call's detail, not the whole trace.

- submit_output(field, value): Submit one required output field's value to the user. Use the exact field name and description given to you in this task's own instructions. Call it once per required field.

Example:
```
smart_tool("Find files that contain the string 'JSON' using glob.")
smart_tool("Read file ./file.json and return the keys using grep.")
smart_tool("Show me the function parse_json in the file ./file.json.")
smart_tool("Replace the function parse_json in the file ./file.json with ...")
smart_tool("Run JSONFieldTests.test_has_key_number via runtests.py")
get_tool_call_detail("call_1") -> returns the full, untruncated input and output  of that one tool call from the previous smart_tool call.
get_trace() -> returns the full tool call trace of the previous smart_tool call, including the input and output of the tools.
... more smart_tool calls ...
submit_output("field_name", "final value") -> once per required output field, when you have the confirmed final value to be reported to the user.
```

ALL basic tools, such as "bash", "grep", "read", etc., are NOT AVAILABLE TO YOU DIRECTLY. Use the smart_tool to run them. When given a high-level task, the smart tool may have trouble decomposing the task accurately. In this case, you may need to give explicit bash commands or code snippets in your task description. You should be able to learn what the smart tool is capable of as you go along. When needed, inspect what the smart tool has executed by calling get_trace() or get_tool_call_detail(call_id), to have a better understanding of the smart tool's capabilities.

If the smart tool is not returning the expected result, or only a summary of the desired result, inspect the basic tool call that contains the result yourself by calling get_tool_call_detail(call_id) to get the full, untruncated input and output of the basic tool call. Do not run the basic tool yourself, you do not have direct access to the basic tools.

For task execution, please keep going until the query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved.

If you are working on a codebase that has tests or the ability to build or run, consider using them to verify that your work is complete. Your philosophy should be to start as specific as possible to the code you changed so that you can catch issues efficiently, then make your way to broader tests as you build confidence. Fix the problem at the root cause rather than applying surface-level patches, when possible.

Once you have confirmed the final output values for every required output field, call submit_output for each field and provided a summary of the work you did (no further tool call) -- that ends the task.
"""

WORKER_SYSTEM = """
You are a tool-execution worker. You will be given a single task.
Do as exactly what the task asks to do using the tools available to you, then stop. Report back as conscise and objectively as possible on the result, within 500 characters. DO NOT add any diagnosis, conclusions, expectations, proposals, or guesses to the output and never address "the user" directly. If you could not complete the task, report exactly what was attempted and what happened. If the task is a question you can answer directly without a tool, answer with the face-value fact only.
If the task requires returning the full result of a tool call, such as returning the full list of files or the full grep orread output, call forward_tool_output() right after that tool call instead of retyping or summarizing its result. Never manually reproduce a tool result you can forward instead.
"""

_SMART_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "smart_tool",
        "description": (
            "Execute a task expressed in natural language on the harness by decomposing it into a list of basic tool calls. Returns the tool_output of the task, the list of basic tool calls performed, and a finish_reason."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to carry out.",
                },
            },
            "required": ["task"],
        },
    },
}

_GET_TRACE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_trace",
        "description": (
            "Get the full, untruncated trace of the decomposed base tool calls from the most recent smart_tool call, including the input and output of every tool call."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

_GET_TOOL_CALL_DETAIL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_tool_call_detail",
        "description": (
            "Get the full, untruncated arguments and result of just ONE tool call from the most recent smart_tool call, identified by the call_id in that smart_tool result's tool_calls entries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "call_id": {
                    "type": "string",
                    "description": "The call_id of the tool call, from a smart_tool tool_calls entry.",
                },
            },
            "required": ["call_id"],
        },
    },
}

_FORWARD_TOOL_OUTPUT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "forward_tool_output",
        "description": (
            "Stage your most recently completed tool call's full, untruncated result as "
            "your final tool_output, instead of retyping or summarizing it yourself. Takes "
            "no arguments -- call it right after the tool call whose result you want to "
            "forward, then finish with a short closing message (no further tool call)."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

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
    """Keeps the first and last half of `max_chars`, marker in between --
    not just the head. A command's arguments are usually most informative
    at the start, but a result's most informative part (a traceback's
    actual exception line, a command's final status) is often at the very
    end; a head-only preview would silently cut that off every time."""
    if len(text) <= max_chars:
        return text
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return text[:head_chars] + " ...[TRUNCATED]... " + text[len(text) - tail_chars :]


def _short_call_id(real_id: str, used_ids: "set[str]") -> str:
    """4-hex-char id derived from the API's own (long) tool-call id -- unique
    only within one segment's trace, which is all get_trace()/
    get_tool_call_detail() ever look it up against; the real id is single-use
    and discarded once the segment ends, so there's nothing for a short id to
    stay stable across. Falls back to a longer hex prefix on the rare
    within-segment collision so two different calls never end up sharing one
    id."""
    digest = hashlib.sha1(real_id.encode()).hexdigest()
    length = 4
    short = digest[:length]
    while short in used_ids and length < len(digest):
        length += 1
        short = digest[:length]
    used_ids.add(short)
    return short


def _extract_full_trace(result) -> "list[dict]":
    """The worker segment's raw tool-call trace, untruncated -- what
    get_trace() serves back verbatim, and what _build_report() truncates for
    smart_tool's own return value. `result` is the worker's own
    ReactLoopResult; `arguments`/`result` here are the raw strings the API
    actually carried, never re-parsed into a dict -- this is a mechanical
    transcript read, not model-authored content. `call_id` on each entry is
    a short hash (see `_short_call_id`), not the API's own long id -- results
    are still matched back to their call via that real id, just not exposed
    as such."""
    trace: "list[dict]" = []
    by_call_id: "dict[str, dict]" = {}
    used_short_ids: "set[str]" = set()
    for m in result.messages or []:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                entry = {
                    "call_id": _short_call_id(tc["id"], used_short_ids),
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


def _build_report(result, forwarded_tool_output: "str | None" = None) -> "tuple[dict, list[dict]]":
    """Returns (report, full_trace). `report` is exactly what smart_tool
    hands back to the supervisor -- tool_calls truncated to
    ARGUMENTS_TRUNCATE_CHARS/RESULT_TRUNCATE_CHARS as a cheap mechanical
    cross-reference, alongside the worker's own tool_output. `full_trace`
    is the untruncated version, cached by the caller for get_trace().
    `result` is the worker's own ReactLoopResult. `forwarded_tool_output`,
    if the worker called forward_tool_output, is that tool call's own raw
    result -- appended to tool_output verbatim (mechanical, never
    regenerated by the worker's own model) rather than replacing whatever
    closing text the worker did write."""
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
        tool_output = None
    elif result.status == "error":
        finish_reason = "worker_error"
        tool_output = result.message
    else:
        finish_reason = "worker_finished"
        tool_output = result.final_text or None

    if forwarded_tool_output is not None:
        tool_output = (
            f"{tool_output}\n\n{forwarded_tool_output}" if tool_output else forwarded_tool_output
        )

    report = {
        "tool_output": tool_output,
        "tool_calls": truncated_calls,
        "finish_reason": finish_reason,
    }
    return report, full_trace


def _trim_worker_history(conversation: "list[dict]", n: int) -> "list[dict]":
    """Keeps the leading system message (if any) plus at most the last *n*
    segments of the rest -- a segment is one smart_tool task's messages,
    starting at the "[TANDEM WORKER segment N] ..." user message that
    opens it. Whole segments are dropped from the front, oldest first;
    never split mid-segment, since a segment's own tool calls/results only
    make sense together. n <= 0 means no replayed history at all (the old
    fresh-worker-per-segment behavior)."""
    if not conversation:
        return conversation
    has_system = conversation[0]["role"] == "system"
    sys_prefix = conversation[:1] if has_system else []
    rest = conversation[1:] if has_system else conversation
    if n <= 0:
        return sys_prefix
    segment_starts = [i for i, m in enumerate(rest) if m["role"] == "user"]
    if len(segment_starts) <= n:
        return conversation
    return sys_prefix + rest[segment_starts[-n] :]


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
    worker_history_turns: int = DEFAULT_WORKER_HISTORY_TURNS,
    offload_dir: str = "./long_tool_call_outputs",
    progress_path: "str | None" = None,
    # Supervisor-only, deliberately not threaded into the worker's own
    # run_react_loop call below -- a worker segment is stateless/ephemeral
    # by design (see this module's docstring), so there is no session for
    # it to contribute to; only the supervisor's own conversation is ever
    # resumable.
    on_checkpoint: "Callable[[list], None] | None" = None,
) -> TandemLoopResult:
    worker_totals = {"input": 0, "output": 0}
    segment_counter = {"n": 0}
    # The full, untruncated trace of the most recent smart_tool call --
    # get_trace() serves this back verbatim, no re-execution needed.
    last_trace: "dict[str, list[dict] | None]" = {"tool_calls": None}
    # The worker's own running conversation, carried across segments so it
    # isn't rediscovering things (like where the repo actually lives) a
    # prior segment already found -- system prompt at index 0, then each
    # segment's "[TANDEM WORKER segment N] <task>" user message through
    # that segment's concluding turn, one after another. None until the
    # first smart_tool call. Trimmed to worker_history_turns segments (see
    # _trim_worker_history) before each new segment is appended, so it
    # never grows past that bound going into a dispatch.
    worker_conversation: "dict[str, list[dict] | None]" = {"messages": None}

    def smart_tool_handler(args_json: str) -> str:
        args = _parse_args(args_json)
        task = (args.get("task") or "").strip()
        if not task:
            return json.dumps(
                {
                    "tool_output": "smart_tool called with an empty task",
                    "tool_calls": [],
                    "finish_reason": "invalid_task",
                }
            )

        segment_index = segment_counter["n"]
        segment_counter["n"] += 1
        _debug(f"task[{segment_index}]: {task}")
        if worker_conversation["messages"] is None:
            # The bridged live transcript/webui has no structured way (yet)
            # to distinguish supervisor turns from worker-segment turns --
            # they share one agent identity end to end. The tag lives on
            # the user turn, not the system turn: the webui's live log
            # doesn't render system-role messages at all, so a tag placed
            # there is invisible -- confirmed by inspecting an actual run's
            # log. One system message for the whole carried-forward
            # conversation, not re-added per segment -- a system message
            # repeated mid-conversation is not normal chat-completions
            # shape and some providers/tokenizers handle it worse than a
            # single leading one.
            worker_conversation["messages"] = [{"role": "system", "content": WORKER_SYSTEM}]
        worker_conversation["messages"] = _trim_worker_history(
            worker_conversation["messages"], worker_history_turns
        )
        worker_conversation["messages"].append(
            {"role": "user", "content": f"[TANDEM WORKER segment {segment_index}] {task}"}
        )
        worker_messages = worker_conversation["messages"]
        # Populated by react_loop.py with every tool call this segment makes,
        # in call order ({tool_call_id: result_content}) -- so
        # forward_tool_output_handler can hand back "whatever I most
        # recently ran" (its last value) without the worker needing to name
        # a call_id it likely never actually saw as text (see this module's
        # docstring).
        call_id_results: "dict[str, str]" = {}
        forwarded = {"content": None}

        def forward_tool_output_handler(_args_json: str) -> str:
            if not call_id_results:
                return json.dumps({"error": "no previous tool call in this task to forward"})
            forwarded["content"] = next(reversed(call_id_results.values()))
            return json.dumps(
                {"result": "staged your most recent tool call's output as your final tool_output"}
            )

        result = run_react_loop(
            worker_messages,
            worker_model,
            worker_llm,
            mcp=mcp,
            bridge=bridge,
            context_limit=worker_context_limit,
            max_steps=segment_step_cap,
            offload_dir=offload_dir,
            # Without this, progress.json only refreshes on the
            # *supervisor's* own turns -- a long worker segment (many
            # chained tool calls with no supervisor turn in between) leaves
            # it stale for the segment's whole duration, however long that
            # actually is. The host-side adapter watches this file's mtime
            # as its sole liveness signal and treats a stale one as "idled
            # out" regardless of whether real work is still happening, so a
            # long-but-live segment could get mistaken for a hang.
            progress_path=progress_path,
            span_prefix=f"worker_seg{segment_index}_turn",
            # Only meaningful when bridged: agmanager_harness interprets/strips
            # this field before forwarding to the real provider. Standalone
            # (bridge is None), there's nothing to strip it, and passing it
            # straight through to litellm/Bedrock is a bad request.
            internal_kind=("tandem_worker" if bridge is not None else None),
            extra_tool_schemas=[_FORWARD_TOOL_OUTPUT_SCHEMA],
            extra_dispatch_table={"forward_tool_output": forward_tool_output_handler},
            call_id_results=call_id_results,
        )
        worker_totals["input"] += result.total_input_tokens
        worker_totals["output"] += result.total_output_tokens
        if result.messages is not None:
            # Becomes the base for the next segment's dispatch (trimmed to
            # worker_history_turns at that point, not here) -- whatever
            # run_react_loop actually ended up sending, including anything
            # maybe_compact() folded in along the way, not just our own
            # worker_messages input plus the new turns.
            worker_conversation["messages"] = result.messages
        report, full_trace = _build_report(result, forwarded_tool_output=forwarded["content"])
        last_trace["tool_calls"] = full_trace
        _debug(f"report[{segment_index}]: {json.dumps(report)[:500]}")
        return json.dumps(report)

    def get_trace_handler(_args_json: str) -> str:
        if last_trace["tool_calls"] is None:
            return json.dumps({"error": "no previous smart_tool call to trace"})
        return json.dumps({"tool_calls": last_trace["tool_calls"]})

    def get_tool_call_detail_handler(args_json: str) -> str:
        args = _parse_args(args_json)
        call_id = (args.get("call_id") or "").strip()
        if not call_id:
            return json.dumps({"error": "get_tool_call_detail called with an empty call_id"})
        trace = last_trace["tool_calls"]
        if trace is None:
            return json.dumps({"error": "no previous smart_tool call to look up"})
        for entry in trace:
            if entry["call_id"] == call_id:
                return json.dumps(entry)
        return json.dumps(
            {"error": f"no tool call with call_id={call_id!r} in the most recent trace"}
        )

    supervisor_tool_schemas = [_SMART_TOOL_SCHEMA, _GET_TRACE_SCHEMA, _GET_TOOL_CALL_DETAIL_SCHEMA]
    supervisor_dispatch_table = {
        "smart_tool": smart_tool_handler,
        "get_trace": get_trace_handler,
        "get_tool_call_detail": get_tool_call_detail_handler,
    }
    if mcp is not None:
        # Discover once up front, not lazily inside the lambda below: the
        # supervisor may call submit_output as its very first action (there's
        # nothing stopping it from doing so before any smart_tool call), so
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
        # The supervisor has no direct bash/read/edit/etc access -- it only
        # ever reaches for one of those by mistake (see SUPERVISOR_SYSTEM).
        # Point it back at smart_tool instead of leaving it to guess why an
        # otherwise-normal tool name came back "unknown".
        unknown_tool_hint="Use the smart_tool instead.",
        on_checkpoint=on_checkpoint,
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
