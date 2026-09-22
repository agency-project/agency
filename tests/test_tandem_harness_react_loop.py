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
    custom_schema = [{"type": "function", "function": {"name": "send_order", "parameters": {}}}]

    result = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        tool_schemas=custom_schema,
        dispatch_table={"send_order": lambda args: "{}"},
        offload_dir=str(tmp_path),
    )

    assert result.status == "done"
    sent_tools = llm.requests[0][2]
    assert sent_tools == custom_schema
    names = {schema["function"]["name"] for schema in sent_tools}
    assert "bash" not in names and "Bash" not in names


def test_policy_exempt_tools_skip_bridge_check(tmp_path):
    llm = _Llm(
        [_tool_call_response(name="send_order", arguments={"order": "do it"}), _final_response()]
    )
    handler_calls = []

    result = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        bridge=_DenyingBridge(),
        tool_schemas=[{"type": "function", "function": {"name": "send_order", "parameters": {}}}],
        dispatch_table={"send_order": lambda args: handler_calls.append(args) or "{}"},
        policy_exempt_tools={"send_order"},
        offload_dir=str(tmp_path),
    )

    assert result.status == "done"
    assert handler_calls == [json.dumps({"order": "do it"})]


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
