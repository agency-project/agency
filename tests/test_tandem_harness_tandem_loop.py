"""Tests for tandem_harness/tandem_loop.py -- the two-model orchestration:
a supervisor drives send_order/get_trace, completing the same way every
ReAct loop in this package does -- a turn with no tool call -- each
send_order spins up a fresh, independent worker react-loop segment, and the
segment's own transcript is turned directly into a structured, truncated
report (mechanical, never model-authored), with get_trace() serving the
full untruncated version of the most recent one back on demand."""

from __future__ import annotations

import copy
import json

from agency.tandem_harness import tools
from agency.tandem_harness.tandem_loop import (
    ARGUMENTS_TRUNCATE_CHARS,
    RESULT_TRUNCATE_CHARS,
    _build_report,
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


def _send_order_response(order):
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("send_order", {"order": order})],
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


def test_run_tandem_loop_one_order_then_implicit_completion(monkeypatch, tmp_path):
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: json.dumps({"ok": True}))
    supervisor_llm = _Llm(
        [_send_order_response("run echo"), _supervisor_final_response("all good")]
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


def test_worker_never_sees_supervisor_tool_schema_or_vice_versa(monkeypatch, tmp_path):
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: "{}")
    supervisor_llm = _Llm([_send_order_response("do it"), _supervisor_final_response("done")])
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
    assert supervisor_tool_names == {"send_order", "get_trace", "get_tool_call_detail"}

    worker_tool_names = {s["function"]["name"] for s in worker_llm.requests[0][2]}
    assert not worker_tool_names & supervisor_tool_names
    assert "bash" in worker_tool_names


def test_worker_output_instruction_is_appended_to_worker_system_only(monkeypatch, tmp_path):
    """The submit_output tool-usage instructions (exact required output
    field names) belong to the worker, since it's the worker that holds the
    submit_output tool -- see agskill.py's _build_output_instruction and
    daemon.py's _render_attempt_prompt. They must never reach the
    supervisor's own turns."""
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: "{}")
    supervisor_llm = _Llm([_send_order_response("do it"), _supervisor_final_response("done")])
    worker_llm = _Llm([_worker_tool_call_response()])

    run_tandem_loop(
        [{"role": "user", "content": "task"}],
        "supervisor-model",
        "worker-model",
        supervisor_llm,
        worker_llm,
        segment_step_cap=1,
        offload_dir=str(tmp_path),
        worker_output_instruction="Required fields:\n  - path: the file path",
    )

    worker_system_message = worker_llm.requests[0][1][0]
    assert worker_system_message["role"] == "system"
    assert "Required fields:\n  - path: the file path" in worker_system_message["content"]

    supervisor_messages_seen = json.dumps([r[1] for r in supervisor_llm.requests])
    assert "Required fields:\n  - path: the file path" not in supervisor_messages_seen


def test_each_order_gets_a_fresh_worker_session_with_no_carryover(monkeypatch, tmp_path):
    """Per the confirmed design: no state carries from one order to the
    next -- each worker segment starts from WORKER_SYSTEM + the order text
    alone, never the previous segment's own messages."""
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: "{}")
    supervisor_llm = _Llm(
        [
            _send_order_response("first"),
            _send_order_response("second"),
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
    assert len(first_segment_messages) == 2  # system + the order, nothing else
    assert len(second_segment_messages) == 2
    assert second_segment_messages[-1]["content"].endswith("second")
    assert "first" not in json.dumps(second_segment_messages)


def test_worker_order_is_tagged_with_its_segment_index_on_the_user_turn(monkeypatch, tmp_path):
    """The webui's live log doesn't render system-role messages at all, so
    the segment-distinguishing tag has to live on the user turn -- verified
    against a real run's log, not just assumed."""
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: "{}")
    supervisor_llm = _Llm(
        [
            _send_order_response("first"),
            _send_order_response("second"),
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

    first_order_message = worker_llm.requests[0][1][-1]
    second_order_message = worker_llm.requests[1][1][-1]
    assert first_order_message["role"] == "user"
    assert first_order_message["content"] == "[TANDEM WORKER segment 0] first"
    assert second_order_message["content"] == "[TANDEM WORKER segment 1] second"


def test_send_order_with_empty_order_short_circuits_without_invoking_worker(tmp_path):
    supervisor_llm = _Llm([_send_order_response(""), _supervisor_final_response("done")])
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


def test_get_trace_returns_the_full_untruncated_trace_of_the_previous_order(monkeypatch, tmp_path):
    long_command = "echo " + "x" * 100
    long_output = "y" * 500
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: long_output)
    supervisor_llm = _Llm(
        [
            _send_order_response("do a big thing"),
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

    # send_order's own tool result (fed back to the supervisor) is truncated.
    send_order_result = json.loads(supervisor_llm.requests[1][1][-1]["content"])
    assert len(send_order_result["tool_calls"][0]["arguments"]) == ARGUMENTS_TRUNCATE_CHARS + len(
        "[TRUNCATED]"
    )
    assert send_order_result["tool_calls"][0]["result"].endswith("[TRUNCATED]")

    # get_trace's result carries the same call, untruncated.
    trace_result = json.loads(supervisor_llm.requests[2][1][-1]["content"])
    assert trace_result["tool_calls"][0]["result"] == long_output
    assert long_command in trace_result["tool_calls"][0]["arguments"]


def test_get_trace_before_any_send_order_returns_an_error(tmp_path):
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
    supervisor_llm = _Llm(
        [
            _send_order_response("do a big thing"),
            _get_tool_call_detail_response("call-0"),
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
    assert detail["call_id"] == "call-0"
    assert detail["tool"] == "bash"
    assert long_command in detail["arguments"]
    assert detail["result"] == long_output  # untruncated, unlike send_order's own preview


def test_get_tool_call_detail_unknown_call_id_returns_an_error(monkeypatch, tmp_path):
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: "{}")
    supervisor_llm = _Llm(
        [
            _send_order_response("do it"),
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


def test_get_tool_call_detail_before_any_send_order_returns_an_error(tmp_path):
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


def test_truncate_appends_marker_only_when_over_the_limit():
    short_text = "hello"
    assert _truncate(short_text, 50) == short_text

    long_text = "x" * 100
    truncated = _truncate(long_text, 50)
    assert truncated == "x" * 50 + "[TRUNCATED]"


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

    assert report["finish_reason"] == "step_cap_reached"
    assert report["summary_text"] is None
    assert report["tool_calls"] == [
        {
            "call_id": "call-0",
            "tool": "bash",
            "arguments": json.dumps({"command": "echo hi"}),
            "result": json.dumps({"ok": True}),
        }
    ]
    assert full_trace == report["tool_calls"]  # both short enough to be untruncated


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
    assert len(truncated["arguments"]) == ARGUMENTS_TRUNCATE_CHARS + len("[TRUNCATED]")
    assert truncated["arguments"].endswith("[TRUNCATED]")
    assert len(truncated["result"]) == RESULT_TRUNCATE_CHARS + len("[TRUNCATED]")
    assert truncated["result"].endswith("[TRUNCATED]")

    full = full_trace[0]
    assert full["arguments"] == json.dumps({"command": "x" * 100})
    assert full["result"] == "y" * 500
    assert truncated["call_id"] == full["call_id"] == "c1"


def test_build_report_worker_finished_with_no_tool_call():
    class _FakeResult:
        status = "done"
        messages = [{"role": "assistant", "content": "no tool needed, answer is 42"}]
        final_text = "no tool needed, answer is 42"
        message = ""

    report, full_trace = _build_report(_FakeResult())

    assert report["finish_reason"] == "worker_finished"
    assert report["tool_calls"] == []
    assert report["summary_text"] == "no tool needed, answer is 42"
    assert full_trace == []


def test_build_report_worker_error_folds_the_dispatch_error_into_summary_text():
    class _FakeResult:
        status = "error"
        messages = None
        final_text = ""
        message = "llm endpoint unreachable: boom"

    report, full_trace = _build_report(_FakeResult())

    assert report["finish_reason"] == "worker_error"
    assert report["summary_text"] == "llm endpoint unreachable: boom"
    assert report["tool_calls"] == []
    assert full_trace == []
