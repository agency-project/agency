"""The two-model tandem loop: a supervisor model keeps the full running
task history and never sees the worker's tool schema; a worker model
carries out one smart_tool task per segment, and reports back a small
structured result built directly from its own transcript, plus its own
natural-language summary of what it did. The worker's own conversation
carries forward across segments by default -- up to `worker_history_turns`
(3) past segments' worth of its own messages replayed as the start of
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
swapped out for two synthetic actions (`smart_tool`, `get_tool_call_detail`)
plus one real one, `submit_output`, dispatched straight through the shared
`McpToolset` (the same connection the worker's own MCP tools use, minus
`submit_output` itself -- see `exclude_mcp_tool_names` on the worker's own
call below) rather than through a synthetic handler; each worker
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

`segment_step_cap` (default 32) is a soft ceiling, not a tight per-task
budget: the worker may make several tool calls to satisfy one task if it
needs to, and `WORKER_SYSTEM` instructs it to finish with a concise
natural-language summary once it's done rather than trailing off. Each
`smart_tool` call returns `tool_calls` as a count only (mechanical,
harness-extracted, never model-authored; `get_tool_call_list` returns the
last call's entries: a `call_id`, its `arguments` truncated to a fixed
preview length, and its result's status and size), `tool_output` (the worker's own factual account of what
it ran and what happened -- `WORKER_SYSTEM` tells it to report like a tool
returning output, not offer its own diagnosis/beliefs), and `finish_reason`
(`tool_finished` / `empty_output` / `max_basic_tool_calls_reached` / `tool_error` /
`invalid_task`), rendered as plain text by `_render_report`.
`call_id`s are unique execution-wide (see `_short_call_id`) and kept in one
hash table (`all_calls`), so `get_tool_call_detail(call_id)` can look up any past call, not just the most recent segment's.

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
import re
import sys
from dataclasses import dataclass, replace
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
DEFAULT_SEGMENT_STEP_CAP = 32

# How many previous smart_tool segments get replayed into a new segment, oldest
# dropped first; replayed tool outputs otherwise grow every worker call's input.
DEFAULT_WORKER_HISTORY_TURNS = 3

_EMPTY_REPORT_MAX_REPROMPTS = 1
_EMPTY_REPORT_REPROMPT = (
    "Your last message was empty. Please reply with your report of what you ran and what happened."
)
_STEP_LIMIT_REPORT_PROMPT = (
    "You've used all the tool calls available for this task. Don't call any more tools. "
    "Reply now with your report: what the task asked for that you found or did, and what is left unfinished."
)

# Argument preview length for the tool_calls index in a smart_tool report; a
# result is listed only by status and size. get_tool_call_detail serves either in full.
ARGUMENTS_TRUNCATE_CHARS = 80

_SUPERVISOR_TOOL_NAMES = {"smart_tool", "get_tool_call_list", "get_tool_call_detail"}

SUPERVISOR_SYSTEM = """\
You complete the user's task using four tools:

- smart_tool(task, report): carries out `task`, a task written in natural language, using basic tools such as bash, read and edit, and returns what you asked for in `report`. Diagonose your task and decompose it into operations that can be carried out by the smart tool.
- get_tool_call_list(): the basic tool calls the last smart_tool call made, each with its call_id, arguments, status and output size. Use it to recover something the report is missing or contradicts, not to double-check a report that answers what you asked.
- get_tool_call_detail(call_id, lines, grep): the arguments and output of one basic tool call from an earlier smart_tool call. Use get_tool_call_list() first to get the call_id.
- submit_output(field, value): submit one required output field, using the exact field name from the task. Call it once per field, when its value is confirmed.

Writing a smart_tool call:
- `task`: what to do, in a sentence or two. smart_tool sees only what you write in `task`, not the user's message: include anything from the user's message that the step needs (a reproduction script, input data, expected output) verbatim, but don't paste content it can read from files. Group related steps into one call, such as reading several related places, or making an edit and rerunning the check that verifies it. Describe a code change by what should change and how to verify it; give exact code only if smart_tool got it wrong.
- `report`: Ask for information from the tool calls that you need to complete the task, e.g. "the column names of a table". Don't ask for what you already have, such as the query you just gave. Be precise and do not ask for more than you need. Think what you need to know to complete the task and ask for that.
- If you know something that saves trial and error (which command or library works, how to run the tests, where a file is), say it in task.
- If smart_tool gets a task wrong, say what was wrong and what you want instead in the next call.
- After you have the information you need, decide what to do next. Don't give the smart tool open-ended goals like "find the bug and fix it".

Examples:
smart_tool(task="Find where parse_json is defined and called.", report="File:line of the definition and of each call, one per line.")
smart_tool(task="Find how config.py picks the cache directory when HOME is unset.", report="The function and lines that do it, and the rule in one sentence.")
smart_tool(task="Run the tests in tests/test_api.py.", report="Pass/fail, and the name and error of each failing test.")
smart_tool(task="Make parse_json in util.py return None on empty input, then run tests/test_util.py.", report="The git diff and the test result.")
get_tool_call_list()
get_tool_call_detail("a1b2", grep="def parse_json")

Keep going until the task is fully resolved. When working on code, verify your change with the most specific test first, then broader ones, and fix the root cause rather than the symptom. Once every required output value is confirmed, call submit_output for each field.
"""

# Same prompt, with `report` guidance against whole-file and full-function reports.
_REPORT_BULLET = (
    '- `report`: Ask for information from the tool calls that you need to complete the task, e.g. "the column '
    'names of a table". Don\'t ask for what you already have, such as the query you just gave. Be precise and do '
    "not ask for more than you need. Think what you need to know to complete the task and ask for that.\n"
)
_BRIEF_REPORT_BULLET = (
    '- `report`: Ask for exactly the information you will use, e.g. "the column names of a table". Don\'t ask for '
    "what you already have, such as the query you just gave. Avoid asking for whole files or full function bodies: "
    "ask for the few lines that matter (file:line and those lines) or for the facts you need from them, and ask for "
    "more in a later call if you need it. Everything in the report takes up your context, so a short, precise "
    "report is better than a complete one.\n"
)
_BRIEF_EXAMPLE = (
    'smart_tool(task="Read how cache.py evicts entries when it is full.", report="File:line of each function '
    'involved, and only the lines that choose what to evict.")\n'
)
assert _REPORT_BULLET in SUPERVISOR_SYSTEM
SUPERVISOR_SYSTEM_BRIEF = SUPERVISOR_SYSTEM.replace(_REPORT_BULLET, _BRIEF_REPORT_BULLET).replace(
    "get_tool_call_list()\n", _BRIEF_EXAMPLE + "get_tool_call_list()\n", 1
)

# Qwen3.5 workers ended ~50% of turns inside <think> (no report) under the
# previous, rule-heavy prompt; short prompts measure ~4% (9B) / ~30% (4B) in replay.
WORKER_SYSTEM = (
    "You are a helpful assistant. Do the task with the available tools, then reply with a short report. "
    "Report must present the requested information in plain text, without headings, tables or bold. "
    "Note that only the final report reaches the user. If there is anything you changed from the user request, include it in the report. "
    "Do only what the task asks; if you think more is needed, say so in the report instead of doing it. "
    "If a step keeps failing after a few different attempts, stop and report what you tried and the exact error. "
    "To return a tool call's full output (a file, a listing, a grep result), call forward_tool_output() right after that call."
)

_SMART_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "smart_tool",
        "description": (
            "Execute a task expressed in natural language on the harness by decomposing it into a list of basic tool calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to carry out.",
                },
                "report": {
                    "type": "string",
                    "description": (
                        "What you want reported back from the tool: the specific information you are looking for. "
                    ),
                },
            },
            "required": ["task", "report"],
        },
    },
}

_GET_TOOL_CALL_DETAIL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_tool_call_detail",
        "description": (
            "Get the arguments and result of a tool call, identified by its call_id from get_tool_call_list. Pass lines or grep to get the information you need from the call, leave empty to get the full call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "call_id": {
                    "type": "string",
                    "description": "The call_id of the tool call, from get_tool_call_list.",
                },
                "lines": {
                    "type": "string",
                    "description": 'Optional 1-based inclusive line range of the result, e.g. "120-180" or "42".',
                },
                "grep": {
                    "type": "string",
                    "description": "Optional regular expression; returns only the result lines that match, with line numbers.",
                },
            },
            "required": ["call_id"],
        },
    },
}

_GET_TOOL_CALL_LIST_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_tool_call_list",
        "description": (
            "List the basic tool calls the most recent smart_tool call made: each one's call_id, "
            "arguments (shortened), status and output size. To see a call's output, pass its call_id to get_tool_call_detail."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

_FORWARD_TOOL_OUTPUT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "forward_tool_output",
        "description": (
            "Stage your most recently completed tool call's full, untruncated result "
            "into your report. Call it right after the tool call whose result you want to "
            "forward."
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


def _truncate(text: str, max_chars: int, marker: str = " … ") -> str:
    """Keeps the first and last half of `max_chars`, marker in between --
    not just the head. A command's arguments are usually most informative
    at the start, but a result's most informative part (a traceback's
    actual exception line, a command's final status) is often at the very
    end; a head-only preview would silently cut that off every time."""
    if len(text) <= max_chars:
        return text
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return text[:head_chars] + marker + text[len(text) - tail_chars :]


def _short_call_id(real_id: str, used_ids: "set[str]") -> str:
    """4-hex-char id from the real one; `used_ids` must persist across the
    whole execution (not per segment) for ids to stay unique."""
    digest = hashlib.sha1(real_id.encode()).hexdigest()
    length = 4
    short = digest[:length]
    while short in used_ids and length < len(digest):
        length += 1
        short = digest[:length]
    used_ids.add(short)
    return short


def _extract_full_trace(
    result, segment_marker: "str | None" = None, used_short_ids: "set[str] | None" = None
) -> "list[dict]":
    """This segment's tool-call trace, untruncated; `segment_marker` scopes
    it, `used_short_ids` must be the caller's persistent set."""
    trace: "list[dict]" = []
    by_call_id: "dict[str, dict]" = {}
    used_short_ids = used_short_ids if used_short_ids is not None else set()
    messages = result.messages or []
    start = 0
    if segment_marker is not None:
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if m.get("role") == "user" and m.get("content") == segment_marker:
                start = i + 1
                break
    for m in messages[start:]:
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


def _build_report(
    result,
    forwarded_tool_output: "str | None" = None,
    segment_marker: "str | None" = None,
    used_short_ids: "set[str] | None" = None,
) -> "tuple[dict, list[dict]]":
    """Returns (report, full_trace): report is smart_tool's truncated reply
    to the supervisor, full_trace the untruncated version fed into all_calls.
    `segment_marker`/`used_short_ids` pass straight through to _extract_full_trace."""
    full_trace = _extract_full_trace(result, segment_marker, used_short_ids)

    if result.status == "error" and (result.message or "").startswith("exceeded max_steps="):
        finish_reason = "max_basic_tool_calls_reached"
        tool_output = result.final_text or None
    elif result.status == "error":
        finish_reason = "tool_error"
        tool_output = result.message
    else:
        finish_reason = "tool_finished"
        tool_output = result.final_text or None

    if forwarded_tool_output is not None:
        status, text = _decode_result(forwarded_tool_output)
        forwarded = text or status
        tool_output = f"{tool_output}\n\n{forwarded}" if tool_output else forwarded
    if finish_reason == "tool_finished" and not (tool_output or "").strip():
        finish_reason = "empty_output"

    report = {
        "tool_output": tool_output,
        # Its content already lands in tool_output.
        "tool_calls": [e for e in full_trace if e["tool"] != "forward_tool_output"],
        "finish_reason": finish_reason,
    }
    return report, full_trace


_MAIN_RESULT_FIELDS = ("output", "content", "result", "error", "note")
_TOOL_CALLS_HEADER = "tool_calls ([id] tool: args / -> status · output size; get_tool_call_detail(call_id) for the output):"


def _field_value(value) -> str:
    if isinstance(value, str):
        return json.dumps(value) if (not value or any(c.isspace() for c in value)) else value
    return json.dumps(value)


def _decode_args(arguments: str) -> str:
    """A lone argument as its bare value, several as k=v pairs."""
    try:
        value = json.loads(arguments) if arguments else {}
    except (json.JSONDecodeError, TypeError):
        return arguments
    if not isinstance(value, dict):
        return arguments
    if len(value) == 1:
        only = next(iter(value.values()))
        return only if isinstance(only, str) else json.dumps(only)
    return " ".join(f"{k}={_field_value(v)}" for k, v in value.items())


def _decode_result(result: str, echoed: str = "") -> "tuple[str, str]":
    """(status, text): a tool's main text field unescaped, its other fields as k=v,
    minus fields that only echo a value already in the call's `echoed` arguments."""
    try:
        value = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return "", result
    if not isinstance(value, dict):
        return "", result
    main_key = next((k for k in _MAIN_RESULT_FIELDS if isinstance(value.get(k), str)), None)
    if main_key is None:
        main_key = next((k for k, v in value.items() if isinstance(v, list)), None)
    text = ""
    if main_key is not None:
        main = value[main_key]
        if isinstance(main, str):
            text = main
        else:
            text = "\n".join(
                ":".join(str(x).rstrip("\n") for x in item.values())
                if isinstance(item, dict)
                else str(item)
                for item in main
            )
    status = ["error"] if main_key == "error" else []
    status += [
        f"{k}=<{len(v)} items>" if isinstance(v, (list, dict)) else f"{k}={_field_value(v)}"
        for k, v in value.items()
        if k != main_key and not (isinstance(v, str) and v and json.dumps(v) in echoed)
    ]
    return " ".join(status), text.rstrip("\n")


def _render_call(entry: dict, *, full: bool, indent: str = "  ") -> "list[str]":
    args = _decode_args(entry["arguments"])
    status, text = _decode_result(entry["result"], echoed=entry["arguments"])
    if not full:
        args = _truncate(args.replace("\n", " ⏎ "), ARGUMENTS_TRUNCATE_CHARS)
    arg_lines = args.split("\n")
    lines = [f"{indent}[{entry['call_id']}] {entry['tool']}: {arg_lines[0]}"]
    lines += [f"{indent}    {line}" if line else "" for line in arg_lines[1:]]
    arrow = f"{indent}  -> "
    if not text:
        lines.append(arrow + (status or "no output"))
    elif not full:
        lines.append(arrow + " · ".join(p for p in (status, _size(text)) if p))
    elif "\n" not in text:
        lines.append(arrow + " · ".join(p for p in (status, text) if p))
    else:
        lines.append(arrow + " · ".join(p for p in (status, _size(text)) if p))
        lines += [f"{indent}     {line}" if line else "" for line in text.split("\n")]
    return lines


def _size(text: str) -> str:
    n = text.count("\n") + 1
    return f"{n} line" if n == 1 else f"{n} lines"


def _select_lines(text: str, lines_spec: str, pattern: str) -> "tuple[list[str], str]":
    """The result lines a get_tool_call_detail range/grep asks for, numbered, plus a note on what they are."""
    numbered = list(enumerate(text.split("\n"), 1))
    notes = []
    if lines_spec:
        m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", lines_spec)
        if m is None:
            return [], f'invalid lines {lines_spec!r}; use e.g. "120-180"'
        start, end = int(m.group(1)), int(m.group(2) or m.group(1))
        numbered = [(i, line) for i, line in numbered if start <= i <= end]
        notes.append(f"lines {start}-{end}")
    if pattern:
        try:
            regex = re.compile(pattern)
        except re.error:
            regex = re.compile(re.escape(pattern))
        numbered = [(i, line) for i, line in numbered if regex.search(line)]
        notes.append(f"{len(numbered)} lines matching {pattern!r}")
    note = f"{' within '.join(reversed(notes))} of {_size(text)}"
    return [f"{i}: {line}" for i, line in numbered], note


def _failed(entry: dict) -> bool:
    try:
        value = json.loads(entry["result"])
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(value, dict):
        return False
    return "error" in value or value.get("returncode") not in (None, 0)


def _render_report(report: dict) -> str:
    lines = [f"finish_reason: {report['finish_reason']}", "tool_output:"]
    lines += [
        f"  {line}" if line else "" for line in (report["tool_output"] or "(none)").split("\n")
    ]
    calls = report["tool_calls"]
    failed = sum(1 for entry in calls if _failed(entry))
    lines.append(f"tool_calls: {len(calls)}" + (f" ({failed} failed)" if failed else ""))
    return "\n".join(lines)


def _render_call_list(calls: "list[dict]") -> str:
    if not calls:
        return "The last smart_tool call made no basic tool calls."
    lines = [_TOOL_CALLS_HEADER]
    for entry in calls:
        lines += _render_call(entry, full=False)
    return "\n".join(lines)


def _trim_worker_history(
    conversation: "list[dict]", n: int, orders: "frozenset[str] | set[str]" = frozenset()
) -> "list[dict]":
    """Keeps the leading system message (if any) plus at most the last *n*
    segments of the rest -- a segment is one smart_tool task's messages,
    starting at the user message that opens it, one of the order texts in
    `orders`. Whole segments are dropped from the front, oldest first;
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
    segment_starts = [
        i for i, m in enumerate(rest) if m["role"] == "user" and m.get("content") in orders
    ]
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
    # Every tool-call entry from every segment so far, keyed by call_id.
    all_calls: "dict[str, dict]" = {}
    # The most recent smart_tool call's entries, for get_tool_call_list; None before the first call.
    last_calls: "dict[str, list[dict] | None]" = {"calls": None}
    # Persistent execution-wide set -- see _short_call_id.
    used_short_ids: "set[str]" = set()
    # The worker's own running conversation, carried across segments so it
    # isn't rediscovering things (like where the repo actually lives) a
    # prior segment already found -- system prompt at index 0, then each
    # segment's order (a plain user message, recorded in sent_orders) through
    # that segment's concluding turn, one after another. None until the
    # first smart_tool call. Trimmed to worker_history_turns segments (see
    # _trim_worker_history) before each new segment is appended, so it
    # never grows past that bound going into a dispatch.
    worker_conversation: "dict[str, list[dict] | None]" = {"messages": None}
    sent_orders: "set[str]" = set()

    def smart_tool_handler(args_json: str) -> str:
        args = _parse_args(args_json)
        task = (args.get("task") or "").strip()
        report_request = (args.get("report") or "").strip()
        last_calls["calls"] = []
        if not task:
            return _render_report(
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
            worker_conversation["messages"], worker_history_turns, sent_orders
        )
        # Plain user text: the worker sees an ordinary request, no marker; sent_orders finds it again.
        segment_marker = (
            f"{task}\n\nIn your reply, include: {report_request}" if report_request else task
        )
        sent_orders.add(segment_marker)
        worker_conversation["messages"].append({"role": "user", "content": segment_marker})
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

        def run_worker(segment_messages, max_steps=segment_step_cap):
            return run_react_loop(
                segment_messages,
                worker_model,
                worker_llm,
                mcp=mcp,
                bridge=bridge,
                context_limit=worker_context_limit,
                max_steps=max_steps,
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
                # Keep submit_output supervisor-only, not MCP-discoverable by the worker.
                exclude_mcp_tool_names=frozenset({"submit_output"}),
            )

        result = run_worker(worker_messages)
        worker_totals["input"] += result.total_input_tokens
        worker_totals["output"] += result.total_output_tokens
        reprompts = 0
        while (
            result.status == "done"
            and not (result.final_text or "").strip()
            and forwarded["content"] is None
            and reprompts < _EMPTY_REPORT_MAX_REPROMPTS
        ):
            reprompts += 1
            result = run_worker(
                result.messages + [{"role": "user", "content": _EMPTY_REPORT_REPROMPT}]
            )
            worker_totals["input"] += result.total_input_tokens
            worker_totals["output"] += result.total_output_tokens
        if (
            result.status == "error"
            and (result.message or "").startswith("exceeded max_steps=")
            and forwarded["content"] is None
        ):
            # One tool-less turn so the supervisor gets a partial report instead of nothing.
            wrap_up = run_worker(
                result.messages + [{"role": "user", "content": _STEP_LIMIT_REPORT_PROMPT}],
                max_steps=1,
            )
            worker_totals["input"] += wrap_up.total_input_tokens
            worker_totals["output"] += wrap_up.total_output_tokens
            if wrap_up.status == "done" and (wrap_up.final_text or "").strip():
                result = replace(wrap_up, status="error", message=result.message)
        if result.messages is not None:
            # Becomes the base for the next segment's dispatch (trimmed to
            # worker_history_turns at that point, not here) -- whatever
            # run_react_loop actually ended up sending, including anything
            # maybe_compact() folded in along the way, not just our own
            # worker_messages input plus the new turns.
            worker_conversation["messages"] = result.messages
        report, full_trace = _build_report(
            result,
            forwarded_tool_output=forwarded["content"],
            segment_marker=segment_marker,
            used_short_ids=used_short_ids,
        )
        for entry in full_trace:
            all_calls[entry["call_id"]] = entry
        last_calls["calls"] = report["tool_calls"]
        rendered = _render_report(report)
        _debug(f"report[{segment_index}]: {rendered[:500]}")
        return rendered

    def get_tool_call_detail_handler(args_json: str) -> str:
        args = _parse_args(args_json)
        call_id = (args.get("call_id") or "").strip()
        if not call_id:
            return "error: get_tool_call_detail called with an empty call_id"
        entry = all_calls.get(call_id)
        if entry is None:
            return f"error: no tool call with call_id={call_id!r} in this task's history"
        lines_spec = str(args.get("lines") or "").strip()
        pattern = str(args.get("grep") or "")
        if not lines_spec and not pattern:
            return "\n".join(_render_call(entry, full=True, indent=""))
        header = _render_call({**entry, "result": ""}, full=True, indent="")[:-1]
        status, text = _decode_result(entry["result"], echoed=entry["arguments"])
        selected, note = _select_lines(text, lines_spec, pattern)
        out = header + ["  -> " + " · ".join(p for p in (status, note) if p)]
        return "\n".join(out + [f"     {line}" for line in selected])

    def get_tool_call_list_handler(_args_json: str) -> str:
        if last_calls["calls"] is None:
            return "error: no smart_tool call has been made yet"
        return _render_call_list(last_calls["calls"])

    supervisor_tool_schemas = [
        _SMART_TOOL_SCHEMA,
        _GET_TOOL_CALL_LIST_SCHEMA,
        _GET_TOOL_CALL_DETAIL_SCHEMA,
    ]
    supervisor_dispatch_table = {
        "smart_tool": smart_tool_handler,
        "get_tool_call_list": get_tool_call_list_handler,
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
