"""Tests for tandem_harness/react_loop.py -- the generalization on top of
native_harness's own react loop that lets the tandem supervisor reuse the
same dispatch machinery with a caller-supplied tool surface
(tool_schemas/dispatch_table) and policy-check exemption for synthetic
control-flow actions (policy_exempt_tools). Completion is unchanged from
native's own convention: a turn with no tool call. See
tests/harness/test_native_lifecycle_boundary.py for the native-side
counterpart these are adapted from."""

from __future__ import annotations

import copy
import json

from agency.tandem_harness import tools
from agency.tandem_harness.react_loop import run_react_loop


class _Llm:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.requests = []

    def dispatch(self, model, messages, tools=None, **kwargs):
        self.requests.append((model, copy.deepcopy(messages), copy.deepcopy(tools), kwargs))
        return next(self._responses)


class _DenyingBridge:
    """A bridge whose check_tool_policy must never be called for an exempt
    tool -- raising makes an accidental call fail the test loudly."""

    def check_tool_policy(self, tool_name, tool_input):
        raise AssertionError(f"check_tool_policy should not be called for {tool_name!r}")

    def complete_tool_policy(self, *args, **kwargs):
        raise AssertionError("complete_tool_policy should not be called for an exempt tool")


def _tool_call_response(name="bash", arguments=None, text=None):
    return {
        "message": {
            "role": "assistant",
            "content": text,
            "tool_calls": [
                {
                    "id": "call-0",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments or {}),
                    },
                }
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _final_response(text="done"):
    return {
        "message": {"role": "assistant", "content": text},
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def test_max_steps_exhausted_preserves_transcript_of_the_executed_tool_call(tmp_path):
    """Regression test: hitting the step cap right after a tool call ran
    must not silently discard that tool call from the returned transcript --
    tandem_loop.py's report-building depends on result.messages surviving
    exactly this case (segment_step_cap=1 is the default)."""
    llm = _Llm([_tool_call_response(arguments={"command": "echo hi"})])

    result = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        max_steps=1,
        offload_dir=str(tmp_path),
    )

    assert result.status == "error"
    assert result.message == "exceeded max_steps=1 without a final answer"
    assert result.messages is not None
    tool_call_messages = [m for m in result.messages if m.get("role") == "assistant"]
    assert (
        tool_call_messages and tool_call_messages[0]["tool_calls"][0]["function"]["name"] == "bash"
    )
    tool_result_messages = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_result_messages) == 1
    assert result.total_input_tokens == 1
    assert result.total_output_tokens == 1


def test_custom_tool_schemas_bypass_builtins_and_mcp(tmp_path):
    llm = _Llm([_final_response()])
    custom_schema = [{"type": "function", "function": {"name": "smart_tool", "parameters": {}}}]

    result = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        tool_schemas=custom_schema,
        dispatch_table={"smart_tool": lambda args: "{}"},
        offload_dir=str(tmp_path),
    )

    assert result.status == "done"
    sent_tools = llm.requests[0][2]
    assert sent_tools == custom_schema
    names = {schema["function"]["name"] for schema in sent_tools}
    assert "bash" not in names and "Bash" not in names


def test_unknown_tool_error_carries_the_given_hint(tmp_path):
    llm = _Llm([_tool_call_response(name="bash", arguments={"command": "ls"}), _final_response()])

    result = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        tool_schemas=[{"type": "function", "function": {"name": "smart_tool", "parameters": {}}}],
        dispatch_table={"smart_tool": lambda args: "{}"},
        unknown_tool_hint="use smart_tool to run this instead",
        offload_dir=str(tmp_path),
    )

    assert result.status == "done"
    tool_result = json.loads(result.messages[2]["content"])
    assert tool_result == {"error": "unknown tool: bash -- use smart_tool to run this instead"}


def test_unknown_tool_error_has_no_hint_by_default(tmp_path):
    llm = _Llm([_tool_call_response(name="bash", arguments={"command": "ls"}), _final_response()])

    result = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        tool_schemas=[{"type": "function", "function": {"name": "smart_tool", "parameters": {}}}],
        dispatch_table={"smart_tool": lambda args: "{}"},
        offload_dir=str(tmp_path),
    )

    tool_result = json.loads(result.messages[2]["content"])
    assert tool_result == {"error": "unknown tool: bash"}


def test_policy_exempt_tools_skip_bridge_check(tmp_path):
    llm = _Llm(
        [_tool_call_response(name="smart_tool", arguments={"task": "do it"}), _final_response()]
    )
    handler_calls = []

    result = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        bridge=_DenyingBridge(),
        tool_schemas=[{"type": "function", "function": {"name": "smart_tool", "parameters": {}}}],
        dispatch_table={"smart_tool": lambda args: handler_calls.append(args) or "{}"},
        policy_exempt_tools={"smart_tool"},
        offload_dir=str(tmp_path),
    )

    assert result.status == "done"
    assert handler_calls == [json.dumps({"task": "do it"})]


def test_non_exempt_tool_still_goes_through_bridge_policy_check(tmp_path):
    class _DenyEverything:
        def check_tool_policy(self, tool_name, tool_input):
            return {"decision": "deny", "reason": "no", "call_id": "c1"}

        def complete_tool_policy(self, *args, **kwargs):
            pass

    llm = _Llm([_tool_call_response(arguments={"command": "echo hi"}), _final_response("finished")])

    result = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        bridge=_DenyEverything(),
        offload_dir=str(tmp_path),
    )

    assert result.status == "done"
    tool_result = json.loads([m for m in result.messages if m.get("role") == "tool"][0]["content"])
    assert "denied by policy" in tool_result["error"]


def test_span_prefix_disambiguates_supervisor_and_worker_spans(tmp_path, monkeypatch):
    recorded_spans = []

    class _RecordingCtx:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            recorded_spans.append(self.name)
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        "agency.tandem_harness.react_loop.profile_span", lambda bridge, name: _RecordingCtx(name)
    )
    llm = _Llm([_final_response()])

    run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        span_prefix="worker_seg3_turn",
        offload_dir=str(tmp_path),
    )

    assert recorded_spans == ["worker_seg3_turn0"]


def test_internal_kind_forwarded_to_dispatch(tmp_path):
    llm = _Llm([_final_response()])

    run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        internal_kind="tandem_supervisor",
        offload_dir=str(tmp_path),
    )

    assert llm.requests[0][3] == {"internal_kind": "tandem_supervisor"}


def test_builtin_path_is_unchanged_when_no_overrides_given(monkeypatch, tmp_path):
    """Sanity check that the generalization is additive: with no
    tool_schemas/dispatch_table override, behavior matches native's own
    react loop exactly (built-in tools, no stop-tool/policy-exempt logic)."""
    tool_events = []
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: tool_events.append(args) or "{}")
    llm = _Llm([_tool_call_response(arguments={"command": "echo hi"}), _final_response("done")])

    result = run_react_loop(
        [{"role": "user", "content": "start"}], "model", llm, offload_dir=str(tmp_path)
    )

    assert result.status == "done"
    assert result.final_text == "done"
    assert tool_events == [json.dumps({"command": "echo hi"})]


def test_extra_tool_schemas_are_layered_onto_the_builtin_path(monkeypatch, tmp_path):
    """extra_tool_schemas/extra_dispatch_table (the worker's own
    forward_tool_output) must not disturb the built-in tools/MCP path --
    both are available side by side."""
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: "{}")
    llm = _Llm(
        [_tool_call_response(name="extra_tool", arguments={"x": 1}), _final_response("done")]
    )
    handler_calls = []

    result = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        extra_tool_schemas=[
            {"type": "function", "function": {"name": "extra_tool", "parameters": {}}}
        ],
        extra_dispatch_table={"extra_tool": lambda args: handler_calls.append(args) or "{}"},
        offload_dir=str(tmp_path),
    )

    assert result.status == "done"
    assert handler_calls == [json.dumps({"x": 1})]
    sent_names = {schema["function"]["name"] for schema in llm.requests[0][2]}
    assert {"bash", "extra_tool"} <= sent_names


def test_call_id_results_is_populated_for_every_dispatched_tool_call(tmp_path):
    llm = _Llm([_tool_call_response(name="bash", arguments={"command": "ls"}), _final_response()])
    call_id_results: dict = {}

    result = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        tool_schemas=[{"type": "function", "function": {"name": "bash", "parameters": {}}}],
        dispatch_table={"bash": lambda args: json.dumps({"files": ["a.py"]})},
        call_id_results=call_id_results,
        offload_dir=str(tmp_path),
    )

    assert result.status == "done"
    assert call_id_results == {"call-0": json.dumps({"files": ["a.py"]})}


def test_on_checkpoint_fires_after_every_step_with_the_running_transcript(tmp_path):
    """A host that kills this process mid-run (an idle timeout, say) needs
    a way to persist the session as it goes, not just once at a clean
    finish -- on_checkpoint is that hook. Must fire on every non-final step
    (mirrors _write_progress, called from the same spot), and each call
    must see that step's own tool call already appended."""
    llm = _Llm(
        [
            _tool_call_response(name="bash", arguments={"command": "one"}),
            _tool_call_response(name="bash", arguments={"command": "two"}),
            _final_response(),
        ]
    )
    checkpoints: "list[list]" = []

    result = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        tool_schemas=[{"type": "function", "function": {"name": "bash", "parameters": {}}}],
        dispatch_table={"bash": lambda args: "{}"},
        on_checkpoint=lambda messages: checkpoints.append(copy.deepcopy(messages)),
        offload_dir=str(tmp_path),
    )

    assert result.status == "done"
    # Once per tool-calling step -- not on the final, toolless step, whose
    # completion is handled by the caller's own explicit save instead.
    assert len(checkpoints) == 2
    assert checkpoints[0][-1]["tool_calls"][0]["function"]["arguments"] == json.dumps(
        {"command": "one"}
    )
    assert checkpoints[1][-1]["tool_calls"][0]["function"]["arguments"] == json.dumps(
        {"command": "two"}
    )


def test_on_checkpoint_failure_does_not_interrupt_the_loop(tmp_path):
    """Best-effort, same as _write_progress -- a broken session save must
    never itself break the actual task."""
    llm = _Llm([_tool_call_response(arguments={"command": "one"}), _final_response()])

    def blows_up(_messages):
        raise RuntimeError("disk full")

    result = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        on_checkpoint=blows_up,
        offload_dir=str(tmp_path),
    )

    assert result.status == "done"
