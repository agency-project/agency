"""Tests for tandem_harness/tandem_loop.py -- the two-model orchestration:
a supervisor drives smart_tool/get_trace, completing the same way every
ReAct loop in this package does -- a turn with no tool call -- each
smart_tool call spins up a fresh, independent worker react-loop segment,
and the segment's own transcript is turned directly into a structured,
truncated report (mechanical, never model-authored), with get_trace()
serving the full untruncated version of the most recent one back on
demand."""

from __future__ import annotations

import copy
import json

from agency.tandem_harness import tools
from agency.tandem_harness.tandem_loop import (
    ARGUMENTS_TRUNCATE_CHARS,
    RESULT_TRUNCATE_CHARS,
    _build_report,
    _short_call_id,
    _trim_worker_history,
    _truncate,
    run_tandem_loop,
)
from agency.tandem_harness.react_loop import run_react_loop


class _Llm:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.requests = []

    def dispatch(self, model, messages, tools=None, **kwargs):
        self.requests.append((model, copy.deepcopy(messages), copy.deepcopy(tools), kwargs))
        return next(self._responses)


def _tool_call(name, arguments, call_id="call-0"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _smart_tool_response(task):
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("smart_tool", {"task": task})],
        },
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _get_trace_response():
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("get_trace", {})],
        },
        "usage": {"prompt_tokens": 6, "completion_tokens": 2},
    }


def _get_tool_call_detail_response(call_id):
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("get_tool_call_detail", {"call_id": call_id})],
        },
        "usage": {"prompt_tokens": 6, "completion_tokens": 2},
    }


def _supervisor_final_response(text):
    """A plain-text, no-tool-call turn -- the implicit completion signal
    every ReAct loop in this package uses, including the supervisor's."""
    return {
        "message": {"role": "assistant", "content": text},
        "usage": {"prompt_tokens": 8, "completion_tokens": 3},
    }


def _worker_tool_call_response(command="echo hi"):
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("bash", {"command": command})],
        },
        "usage": {"prompt_tokens": 20, "completion_tokens": 7},
    }


def _forward_tool_output_response(request_call_id="call-1"):
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("forward_tool_output", {}, call_id=request_call_id)],
        },
        "usage": {"prompt_tokens": 6, "completion_tokens": 2},
    }


def _submit_output_response(field, value):
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("submit_output", {"field": field, "value": value})],
        },
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }


class _FakeMcp:
    """Stands in for McpToolset -- discover()/call() are the only two
    methods run_tandem_loop's supervisor dispatch touches."""

    def __init__(self, schemas=None):
        self._schemas = list(schemas or [])
        self.discover_calls = 0
        self.calls = []

    def discover(self):
        self.discover_calls += 1
        return self._schemas

    def call(self, tool_name, arguments_json):
        self.calls.append((tool_name, arguments_json))
        return json.dumps({"result": f"field recorded via {tool_name}"})


def test_run_tandem_loop_one_order_then_implicit_completion(monkeypatch, tmp_path):
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: json.dumps({"ok": True}))
    supervisor_llm = _Llm(
        [_smart_tool_response("run echo"), _supervisor_final_response("all good")]
    )
    worker_llm = _Llm([_worker_tool_call_response()])

    result = run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        segment_step_cap=1,
        offload_dir=str(tmp_path),
    )

    assert result.status == "done"
    assert result.final_text == "all good"
    assert result.segment_count == 1
    # Token accounting is kept separate per model, not merged.
    assert result.supervisor_input_tokens == 10 + 8
    assert result.supervisor_output_tokens == 5 + 3
    assert result.worker_input_tokens == 20
    assert result.worker_output_tokens == 7


def test_worker_forward_tool_output_appends_the_last_calls_raw_result(monkeypatch, tmp_path):
    """The whole point: a worker that ran `ls` doesn't need to retype the
    listing into its own closing text -- forward_tool_output() (no call_id
    -- see tandem_loop.py's docstring on why) stages the most recently
    completed tool call's own raw result, and _build_report appends it to
    whatever (short) closing text the worker did write."""
    monkeypatch.setitem(
        tools.TOOL_DISPATCH, "bash", lambda args: json.dumps({"files": ["a.py", "b.py"]})
    )
    supervisor_llm = _Llm(
        [_smart_tool_response("list files"), _supervisor_final_response("all good")]
    )
    worker_llm = _Llm(
        [
            _worker_tool_call_response("ls"),
            _forward_tool_output_response(),
            _supervisor_final_response("done"),
        ]
    )

    result = run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        segment_step_cap=3,
        offload_dir=str(tmp_path),
    )

    assert result.status == "done"
    smart_tool_result = json.loads(supervisor_llm.requests[1][1][-1]["content"])
    assert smart_tool_result["tool_output"] == 'done\n\n{"files": ["a.py", "b.py"]}'


def test_worker_forward_tool_output_with_no_prior_tool_call_returns_an_error(monkeypatch, tmp_path):
    supervisor_llm = _Llm([_smart_tool_response("do it"), _supervisor_final_response("all good")])
    worker_llm = _Llm(
        [
            _forward_tool_output_response(request_call_id="call-0"),
            _supervisor_final_response("done"),
        ]
    )

    result = run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        segment_step_cap=2,
        offload_dir=str(tmp_path),
    )

    assert result.status == "done"
    # The worker sees the error as its own tool result and can still finish
    # normally -- nothing here is fatal to the segment.
    worker_tool_result = json.loads(worker_llm.requests[1][1][-1]["content"])
    assert "error" in worker_tool_result
    smart_tool_result = json.loads(supervisor_llm.requests[1][1][-1]["content"])
    assert smart_tool_result["tool_output"] == "done"


def test_worker_never_sees_supervisor_tool_schema_or_vice_versa(monkeypatch, tmp_path):
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: "{}")
    supervisor_llm = _Llm([_smart_tool_response("do it"), _supervisor_final_response("done")])
    worker_llm = _Llm([_worker_tool_call_response()])

    run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        segment_step_cap=1,
        offload_dir=str(tmp_path),
    )

    supervisor_tool_names = {s["function"]["name"] for s in supervisor_llm.requests[0][2]}
    assert supervisor_tool_names == {"smart_tool", "get_trace", "get_tool_call_detail"}

    worker_tool_names = {s["function"]["name"] for s in worker_llm.requests[0][2]}
    assert not worker_tool_names & supervisor_tool_names
    assert "bash" in worker_tool_names


def _supervisor_bash_call_response():
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("bash", {"command": "ls"})],
        },
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }


def test_supervisor_calling_a_real_tool_directly_is_pointed_back_at_smart_tool(tmp_path):
    """The supervisor has no direct bash/read/etc access -- it only ever
    reaches for one of those by mistake. The generic "unknown tool" error
    should tell it to use smart_tool instead of leaving it to guess."""
    supervisor_llm = _Llm([_supervisor_bash_call_response(), _supervisor_final_response("done")])
    worker_llm = _Llm([])  # would raise StopIteration if ever called

    run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        offload_dir=str(tmp_path),
    )

    tool_result = json.loads(supervisor_llm.requests[1][1][-1]["content"])
    assert tool_result == {"error": "unknown tool: bash -- use smart_tool to run this instead"}


def test_supervisor_gets_submit_output_tool_when_mcp_is_configured(tmp_path):
    """submit_output is a host-side bookkeeping call (record one output
    field's value), not sandbox execution, so the supervisor -- the one
    with full task context -- calls it directly instead of routing a
    completion message through smart_tool to the worker."""
    mcp = _FakeMcp()
    supervisor_llm = _Llm([_supervisor_final_response("done")])
    worker_llm = _Llm([])

    run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        mcp=mcp,
        offload_dir=str(tmp_path),
    )

    supervisor_tool_names = {s["function"]["name"] for s in supervisor_llm.requests[0][2]}
    assert "submit_output" in supervisor_tool_names
    # Discovered up front, not lazily on first use -- see run_tandem_loop.
    assert mcp.discover_calls >= 1


def test_supervisor_has_no_submit_output_tool_without_mcp(tmp_path):
    supervisor_llm = _Llm([_supervisor_final_response("done")])
    worker_llm = _Llm([])

    run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        offload_dir=str(tmp_path),
    )

    supervisor_tool_names = {s["function"]["name"] for s in supervisor_llm.requests[0][2]}
    assert "submit_output" not in supervisor_tool_names


def test_supervisor_submit_output_call_dispatches_through_mcp(tmp_path):
    mcp = _FakeMcp()
    supervisor_llm = _Llm(
        [
            _submit_output_response("path", "/workspace/note.txt"),
            _supervisor_final_response("done"),
        ]
    )
    worker_llm = _Llm([])

    run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        mcp=mcp,
        offload_dir=str(tmp_path),
    )

    assert mcp.calls == [
        ("submit_output", json.dumps({"field": "path", "value": "/workspace/note.txt"}))
    ]


def test_each_task_by_default_carries_the_prior_segments_own_messages(monkeypatch, tmp_path):
    """Per the current design: the worker's own conversation carries
    forward across segments by default (worker_history_turns), so a later
    segment doesn't have to rediscover what an earlier one already found."""
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: "{}")
    supervisor_llm = _Llm(
        [
            _smart_tool_response("first"),
            _smart_tool_response("second"),
            _supervisor_final_response("done"),
        ]
    )
    worker_llm = _Llm([_worker_tool_call_response("one"), _worker_tool_call_response("two")])

    run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        segment_step_cap=1,
        offload_dir=str(tmp_path),
    )

    first_segment_messages = worker_llm.requests[0][1]
    second_segment_messages = worker_llm.requests[1][1]
    assert len(first_segment_messages) == 2  # system + the task, nothing else
    assert second_segment_messages[-1]["content"].endswith("second")
    assert "first" in json.dumps(second_segment_messages)


def test_worker_history_turns_zero_gives_no_carryover(monkeypatch, tmp_path):
    """worker_history_turns=0 opts back into the old fresh-session-per-task
    behavior -- each worker segment starts from WORKER_SYSTEM + the task
    text alone, never a previous segment's own messages."""
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: "{}")
    supervisor_llm = _Llm(
        [
            _smart_tool_response("first"),
            _smart_tool_response("second"),
            _supervisor_final_response("done"),
        ]
    )
    worker_llm = _Llm([_worker_tool_call_response("one"), _worker_tool_call_response("two")])

    run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        segment_step_cap=1,
        worker_history_turns=0,
        offload_dir=str(tmp_path),
    )

    first_segment_messages = worker_llm.requests[0][1]
    second_segment_messages = worker_llm.requests[1][1]
    assert len(first_segment_messages) == 2  # system + the task, nothing else
    assert len(second_segment_messages) == 2
    assert second_segment_messages[-1]["content"].endswith("second")
    assert "first" not in json.dumps(second_segment_messages)


def test_worker_task_is_tagged_with_its_segment_index_on_the_user_turn(monkeypatch, tmp_path):
    """The webui's live log doesn't render system-role messages at all, so
    the segment-distinguishing tag has to live on the user turn -- verified
    against a real run's log, not just assumed."""
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: "{}")
    supervisor_llm = _Llm(
        [
            _smart_tool_response("first"),
            _smart_tool_response("second"),
            _supervisor_final_response("done"),
        ]
    )
    worker_llm = _Llm([_worker_tool_call_response("one"), _worker_tool_call_response("two")])

    run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        segment_step_cap=1,
        offload_dir=str(tmp_path),
    )

    first_task_message = worker_llm.requests[0][1][-1]
    second_task_message = worker_llm.requests[1][1][-1]
    assert first_task_message["role"] == "user"
    assert first_task_message["content"] == "[TANDEM WORKER segment 0] first"
    assert second_task_message["content"] == "[TANDEM WORKER segment 1] second"


def test_smart_tool_with_empty_task_short_circuits_without_invoking_worker(tmp_path):
    supervisor_llm = _Llm([_smart_tool_response(""), _supervisor_final_response("done")])
    worker_llm = _Llm([])  # would raise StopIteration if ever called

    result = run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        offload_dir=str(tmp_path),
    )

    assert result.status == "done"
    assert result.segment_count == 0


def test_get_trace_returns_the_full_untruncated_trace_of_the_previous_task(monkeypatch, tmp_path):
    long_command = "echo " + "x" * 100
    long_output = "y" * 500
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: long_output)
    supervisor_llm = _Llm(
        [
            _smart_tool_response("do a big thing"),
            _get_trace_response(),
            _supervisor_final_response("done"),
        ]
    )
    worker_llm = _Llm([_worker_tool_call_response(long_command)])

    run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        segment_step_cap=1,
        offload_dir=str(tmp_path),
    )

    # smart_tool's own tool result (fed back to the supervisor) is truncated.
    smart_tool_result = json.loads(supervisor_llm.requests[1][1][-1]["content"])
    assert len(smart_tool_result["tool_calls"][0]["arguments"]) == ARGUMENTS_TRUNCATE_CHARS + len(
        "[TRUNCATED]"
    )
    assert "[TRUNCATED]" in smart_tool_result["tool_calls"][0]["result"]

    # get_trace's result carries the same call, untruncated.
    trace_result = json.loads(supervisor_llm.requests[2][1][-1]["content"])
    assert trace_result["tool_calls"][0]["result"] == long_output
    assert long_command in trace_result["tool_calls"][0]["arguments"]


def test_get_trace_before_any_smart_tool_call_returns_an_error(tmp_path):
    supervisor_llm = _Llm([_get_trace_response(), _supervisor_final_response("done")])
    worker_llm = _Llm([])

    run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        offload_dir=str(tmp_path),
    )

    trace_result = json.loads(supervisor_llm.requests[1][1][-1]["content"])
    assert "error" in trace_result


def test_get_tool_call_detail_returns_the_full_untruncated_entry(monkeypatch, tmp_path):
    long_command = "echo " + "x" * 100
    long_output = "y" * 500
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: long_output)
    short_call_id = _short_call_id("call-0", set())
    supervisor_llm = _Llm(
        [
            _smart_tool_response("do a big thing"),
            _get_tool_call_detail_response(short_call_id),
            _supervisor_final_response("done"),
        ]
    )
    worker_llm = _Llm([_worker_tool_call_response(long_command)])

    run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        segment_step_cap=1,
        offload_dir=str(tmp_path),
    )

    detail = json.loads(supervisor_llm.requests[2][1][-1]["content"])
    assert detail["call_id"] == short_call_id
    assert detail["tool"] == "bash"
    assert long_command in detail["arguments"]
    assert detail["result"] == long_output  # untruncated, unlike smart_tool's own preview


def test_get_tool_call_detail_unknown_call_id_returns_an_error(monkeypatch, tmp_path):
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: "{}")
    supervisor_llm = _Llm(
        [
            _smart_tool_response("do it"),
            _get_tool_call_detail_response("does-not-exist"),
            _supervisor_final_response("done"),
        ]
    )
    worker_llm = _Llm([_worker_tool_call_response()])

    run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        segment_step_cap=1,
        offload_dir=str(tmp_path),
    )

    detail = json.loads(supervisor_llm.requests[2][1][-1]["content"])
    assert "error" in detail


def test_get_tool_call_detail_before_any_smart_tool_call_returns_an_error(tmp_path):
    supervisor_llm = _Llm(
        [_get_tool_call_detail_response("anything"), _supervisor_final_response("done")]
    )
    worker_llm = _Llm([])

    run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        offload_dir=str(tmp_path),
    )

    detail = json.loads(supervisor_llm.requests[1][1][-1]["content"])
    assert "error" in detail


def _seg(user_content, n_assistant_turns=1):
    """A minimal fake segment: one user task message plus n_assistant_turns
    bare assistant messages (no tool_calls -- content, not shape, is what
    these tests check)."""
    msgs = [{"role": "user", "content": user_content}]
    for i in range(n_assistant_turns):
        msgs.append({"role": "assistant", "content": f"{user_content}-reply-{i}"})
    return msgs


def test_trim_worker_history_keeps_system_and_last_n_whole_segments():
    conversation = (
        [{"role": "system", "content": "sys"}] + _seg("seg0") + _seg("seg1") + _seg("seg2")
    )
    trimmed = _trim_worker_history(conversation, 2)
    assert trimmed[0] == {"role": "system", "content": "sys"}
    assert "seg0" not in json.dumps(trimmed)
    assert "seg1" in json.dumps(trimmed)
    assert "seg2" in json.dumps(trimmed)


def test_trim_worker_history_n_zero_drops_all_segments():
    conversation = [{"role": "system", "content": "sys"}] + _seg("seg0") + _seg("seg1")
    trimmed = _trim_worker_history(conversation, 0)
    assert trimmed == [{"role": "system", "content": "sys"}]


def test_trim_worker_history_under_the_cap_is_unchanged():
    conversation = [{"role": "system", "content": "sys"}] + _seg("seg0")
    assert _trim_worker_history(conversation, 4096) == conversation


def test_trim_worker_history_empty_conversation():
    assert _trim_worker_history([], 4096) == []


def test_truncate_keeps_head_and_tail_with_marker_in_the_middle():
    short_text = "hello"
    assert _truncate(short_text, 50) == short_text

    # Digits so head/tail slices are unambiguous -- not just a length check.
    long_text = "".join(str(i % 10) for i in range(100))
    truncated = _truncate(long_text, 50)
    assert truncated == long_text[:25] + "[TRUNCATED]" + long_text[-25:]
    assert truncated.startswith(long_text[:25])
    assert truncated.endswith(long_text[-25:])
    assert len(truncated) == 50 + len("[TRUNCATED]")


def test_build_report_step_cap_reached_carries_the_executed_tool_call(monkeypatch, tmp_path):
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: json.dumps({"ok": True}))
    llm = _Llm([_worker_tool_call_response("echo hi")])
    result = run_react_loop(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "order"}],
        "worker-model",
        llm,
        max_steps=1,
        offload_dir=str(tmp_path),
    )

    report, full_trace = _build_report(result)
    short_call_id = _short_call_id("call-0", set())

    assert report["finish_reason"] == "step_cap_reached"
    assert report["tool_output"] is None
    assert full_trace == [
        {
            "call_id": short_call_id,
            "tool": "bash",
            "arguments": json.dumps({"command": "echo hi"}),
            "result": json.dumps({"ok": True}),
        }
    ]
    # report's own copy is the same call, just previewed to the (possibly
    # shorter) truncation lengths -- see test_build_report_truncates_* for
    # the truncation behavior itself.
    assert report["tool_calls"] == [
        {
            "call_id": short_call_id,
            "tool": "bash",
            "arguments": _truncate(json.dumps({"command": "echo hi"}), ARGUMENTS_TRUNCATE_CHARS),
            "result": _truncate(json.dumps({"ok": True}), RESULT_TRUNCATE_CHARS),
        }
    ]


def test_build_report_truncates_long_arguments_and_results():
    class _FakeResult:
        status = "done"
        final_text = ""
        message = ""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": "x" * 100}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "y" * 500},
        ]

    report, full_trace = _build_report(_FakeResult())

    truncated = report["tool_calls"][0]
    full_arguments = json.dumps({"command": "x" * 100})
    full_result = "y" * 500
    assert len(truncated["arguments"]) == ARGUMENTS_TRUNCATE_CHARS + len("[TRUNCATED]")
    assert "[TRUNCATED]" in truncated["arguments"]
    # The result's tail (e.g. a traceback's actual exception line, a
    # command's final status) is at least as informative as its head, so
    # the preview keeps both ends rather than only the start.
    assert truncated["arguments"].startswith(full_arguments[: ARGUMENTS_TRUNCATE_CHARS // 2])
    assert truncated["arguments"].endswith(
        full_arguments[-(ARGUMENTS_TRUNCATE_CHARS - ARGUMENTS_TRUNCATE_CHARS // 2) :]
    )
    assert len(truncated["result"]) == RESULT_TRUNCATE_CHARS + len("[TRUNCATED]")
    assert "[TRUNCATED]" in truncated["result"]
    assert truncated["result"].startswith(full_result[: RESULT_TRUNCATE_CHARS // 2])
    assert truncated["result"].endswith(
        full_result[-(RESULT_TRUNCATE_CHARS - RESULT_TRUNCATE_CHARS // 2) :]
    )

    full = full_trace[0]
    assert full["arguments"] == json.dumps({"command": "x" * 100})
    assert full["result"] == "y" * 500
    assert truncated["call_id"] == full["call_id"] == _short_call_id("c1", set())


def test_build_report_worker_finished_with_no_tool_call():
    class _FakeResult:
        status = "done"
        messages = [{"role": "assistant", "content": "no tool needed, answer is 42"}]
        final_text = "no tool needed, answer is 42"
        message = ""

    report, full_trace = _build_report(_FakeResult())

    assert report["finish_reason"] == "worker_finished"
    assert report["tool_calls"] == []
    assert report["tool_output"] == "no tool needed, answer is 42"
    assert full_trace == []


def test_build_report_worker_error_folds_the_dispatch_error_into_tool_output():
    class _FakeResult:
        status = "error"
        messages = None
        final_text = ""
        message = "llm endpoint unreachable: boom"

    report, full_trace = _build_report(_FakeResult())

    assert report["finish_reason"] == "worker_error"
    assert report["tool_output"] == "llm endpoint unreachable: boom"
    assert report["tool_calls"] == []
    assert full_trace == []


def test_build_report_forwarded_tool_output_stands_alone_with_no_closing_text():
    class _FakeResult:
        status = "done"
        messages = [{"role": "assistant", "content": ""}]
        final_text = ""
        message = ""

    report, _ = _build_report(_FakeResult(), forwarded_tool_output='{"files": ["a.py"]}')

    assert report["tool_output"] == '{"files": ["a.py"]}'


def test_build_report_forwarded_tool_output_is_appended_after_closing_text():
    class _FakeResult:
        status = "done"
        messages = [{"role": "assistant", "content": "done"}]
        final_text = "done"
        message = ""

    report, _ = _build_report(_FakeResult(), forwarded_tool_output='{"files": ["a.py"]}')

    assert report["tool_output"] == 'done\n\n{"files": ["a.py"]}'
