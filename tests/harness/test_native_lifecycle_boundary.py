from __future__ import annotations

import copy
import json
import threading

import pytest

from agency._agent_control import AgentControl
from agency.native_harness import tools
from agency.native_harness.react_loop import run_react_loop


class _Llm:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.requests = []

    def dispatch(self, model, messages, tools=None, **_kwargs):
        self.requests.append((model, copy.deepcopy(messages), copy.deepcopy(tools)))
        return next(self._responses)


class _BlockingCompactionLlm:
    def __init__(self):
        self.compaction_entered = threading.Event()
        self.release_compaction = threading.Event()
        self.task_generation_started = threading.Event()
        self.requests = []

    def dispatch(self, model, messages, tools=None, **kwargs):
        self.requests.append((model, copy.deepcopy(messages), copy.deepcopy(tools), dict(kwargs)))
        if kwargs.get("internal_kind") == "compaction":
            self.compaction_entered.set()
            assert self.release_compaction.wait(timeout=2.0)
            return _final_response("compacted history")
        self.task_generation_started.set()
        return _final_response()


class _ControlBridge:
    def __init__(self, handle):
        self.handle = handle
        self.calls = []
        self.first_tool_checkpoint = threading.Event()

    def checkpoint(self, boundary_id, *, allow_messages, phase):
        self.calls.append((boundary_id, allow_messages, phase))
        if boundary_id.startswith("native:tool:0:0:"):
            self.first_tool_checkpoint.set()
        decision = self.handle._checkpoint(
            boundary_id,
            allow_messages=allow_messages,
            phase=phase,
        )
        return {
            "action_admitted": decision.action_admitted,
            "cancelled": decision.cancelled,
            "destroyed": decision.destroyed,
            "invocation_messages": [
                {"sequence": entry.sequence, "content": entry.content}
                for entry in decision.invocation_messages
            ],
        }

    @staticmethod
    def check_tool_policy(_tool_name, _tool_input):
        return {"decision": "allow", "reason": None}

    @staticmethod
    def complete_tool_policy(_call_id, _result):
        pass


def _tool_response(*labels):
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call-{index}",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"label": label}),
                    },
                }
                for index, label in enumerate(labels)
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _final_response(text="done"):
    return {
        "message": {"role": "assistant", "content": text},
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _wait_for_control(control, predicate, timeout=2.0):
    with control._condition:
        return control._condition.wait_for(predicate, timeout=timeout)


def _history_that_requires_compaction():
    messages = [{"role": "user", "content": "original task"}]
    for index in range(5):
        messages.extend(
            [
                {"role": "assistant", "content": f"old answer {index}"},
                {"role": "user", "content": f"follow-up {index}"},
            ]
        )
    return messages


@pytest.mark.parametrize("action", ["pause", "cancel"])
def test_native_controls_arriving_during_compaction_stop_before_task_generation(action, tmp_path):
    control = AgentControl()
    handle = control.begin_invocation("test")
    bridge = _ControlBridge(handle)
    llm = _BlockingCompactionLlm()
    result_holder = {}
    worker = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result",
            run_react_loop(
                _history_that_requires_compaction(),
                "model",
                llm,
                bridge=bridge,
                context_limit=1,
                max_steps=1,
                offload_dir=str(tmp_path),
            ),
        ),
        daemon=True,
    )
    worker.start()
    try:
        assert llm.compaction_entered.wait(timeout=2.0)
        getattr(handle, action)()
        llm.release_compaction.set()

        if action == "pause":
            assert _wait_for_control(control, control.is_paused_actual)
            assert worker.is_alive()
            assert not llm.task_generation_started.is_set()
            handle.resume()

        worker.join(timeout=2.0)
        assert not worker.is_alive()
    finally:
        if handle.is_pause_requested() and handle.phase != "closing":
            handle.resume()
        llm.release_compaction.set()
        worker.join(timeout=2.0)
        control.end_invocation(handle)

    result = result_holder["result"]
    if action == "pause":
        assert result.status == "done"
        assert llm.task_generation_started.is_set()
    else:
        assert result.status == "error"
        assert result.message == "agent invocation cancelled"
        assert not llm.task_generation_started.is_set()
    assert any(call[0].startswith("native:post-compaction:0:") for call in bridge.calls)


def test_native_pause_and_message_after_each_tool_preserve_multi_tool_protocol(
    monkeypatch, tmp_path
):
    control = AgentControl()
    handle = control.begin_invocation("test")
    bridge = _ControlBridge(handle)
    llm = _Llm([_tool_response("one", "two"), _final_response()])
    first_tool_entered = threading.Event()
    release_first_tool = threading.Event()
    tool_events = []

    def blocking_tool(arguments):
        label = json.loads(arguments)["label"]
        tool_events.append(label)
        if label == "one":
            first_tool_entered.set()
            assert release_first_tool.wait(timeout=2.0)
        return json.dumps({"ok": label})

    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", blocking_tool)
    result_holder = {}
    worker = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result",
            run_react_loop(
                [{"role": "user", "content": "start"}],
                "model",
                llm,
                bridge=bridge,
                context_limit=None,
                max_steps=3,
                offload_dir=str(tmp_path),
            ),
        )
    )
    worker.start()
    try:
        assert first_tool_entered.wait(timeout=2.0)
        handle.redirect("first instruction")
        handle.redirect("second instruction")
        handle.pause()
        release_first_tool.set()

        assert bridge.first_tool_checkpoint.wait(timeout=2.0)
        assert _wait_for_control(control, control.is_paused_actual)
        assert tool_events == ["one"]
        assert worker.is_alive()

        handle.resume()
        worker.join(timeout=2.0)
        assert not worker.is_alive()
    finally:
        if handle.is_pause_requested() and handle.phase != "closing":
            handle.resume()
        release_first_tool.set()
        worker.join(timeout=2.0)
        control.end_invocation(handle)

    result = result_holder["result"]
    assert result.status == "done"
    assert tool_events == ["one"]

    second_generation = llm.requests[1][1]
    roles = [message["role"] for message in second_generation]
    assert roles[-4:] == ["assistant", "tool", "tool", "user"]
    message_text = second_generation[-1]["content"]
    assert message_text.index("first instruction") < message_text.index("second instruction")
    assert sum("first instruction" in str(message) for message in second_generation) == 1
    assert sum("second instruction" in str(message) for message in second_generation) == 1


def test_native_cancel_during_tool_waits_for_result_then_skips_remaining_batch(
    monkeypatch, tmp_path
):
    control = AgentControl()
    handle = control.begin_invocation("test")
    bridge = _ControlBridge(handle)
    llm = _Llm([_tool_response("one", "two")])
    first_tool_entered = threading.Event()
    release_first_tool = threading.Event()
    tool_events = []

    def blocking_tool(arguments):
        label = json.loads(arguments)["label"]
        tool_events.append(label)
        if label == "one":
            first_tool_entered.set()
            assert release_first_tool.wait(timeout=2.0)
        return json.dumps({"ok": label})

    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", blocking_tool)
    result_holder = {}
    worker = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result",
            run_react_loop(
                [{"role": "user", "content": "start"}],
                "model",
                llm,
                bridge=bridge,
                context_limit=None,
                max_steps=3,
                offload_dir=str(tmp_path),
            ),
        )
    )
    worker.start()
    try:
        assert first_tool_entered.wait(timeout=2.0)
        handle.cancel()
        assert tool_events == ["one"]
        release_first_tool.set()
        worker.join(timeout=2.0)
        assert not worker.is_alive()
    finally:
        release_first_tool.set()
        worker.join(timeout=2.0)
        control.end_invocation(handle)

    assert result_holder["result"].status == "error"
    assert result_holder["result"].message == "agent invocation cancelled"
    assert tool_events == ["one"]
    assert len(llm.requests) == 1


@pytest.mark.parametrize("arrival", ["during-model", "before-action", "before-final"])
def test_native_redirect_discards_stale_action_or_final(monkeypatch, tmp_path, arrival):
    handle = AgentControl().begin_invocation("redirect")
    tool_events = []
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: tool_events.append(args))

    class Bridge(_ControlBridge):
        def checkpoint(self, boundary_id, *, allow_messages, phase):
            if (
                (arrival == "before-action" and phase == "action")
                or (arrival == "before-final" and phase == "closing")
            ) and not self.injected:
                self.injected = True
                handle.redirect("new instruction")
            return super().checkpoint(boundary_id, allow_messages=allow_messages, phase=phase)

        injected = False

    class Llm(_Llm):
        def dispatch(self, *args, **kwargs):
            result = super().dispatch(*args, **kwargs)
            if arrival == "during-model" and len(self.requests) == 1:
                handle.redirect("new instruction")
            return result

    first = _final_response("stale") if arrival == "before-final" else _tool_response("stale")
    llm = Llm([first, _final_response("revised")])
    result = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        bridge=Bridge(handle),
        max_steps=3,
        offload_dir=str(tmp_path),
    )
    assert result.final_text == "revised"
    assert tool_events == []
    assert "new instruction" in str(llm.requests[1][1])
    assert not handle._pending_messages
    assert handle.phase == "closing"


def test_native_failed_model_leaves_redirect_pending_for_new_attempt(tmp_path):
    handle = AgentControl().begin_invocation("retry")
    handle.redirect("retain me")
    failed = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        _Llm([{"error": "failed"}]),
        bridge=_ControlBridge(handle),
        offload_dir=str(tmp_path),
    )
    assert failed.status == "error"
    assert [entry.content for entry in handle._pending_messages] == ["retain me"]
    llm = _Llm([_final_response()])
    result = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        bridge=_ControlBridge(handle),
        offload_dir=str(tmp_path),
    )
    assert result.status == "done"
    assert "retain me" in str(llm.requests[0][1])
    assert not handle._pending_messages


def test_redirect_text_is_rendered_after_compaction(monkeypatch, tmp_path):
    handle = AgentControl().begin_invocation("compaction")
    handle.redirect("exact redirect text")
    compacted = [{"role": "user", "content": "summary"}]
    monkeypatch.setattr(
        "agency.native_harness.react_loop.maybe_compact",
        lambda *args: (copy.deepcopy(compacted), "summary"),
    )
    llm = _Llm([_final_response()])
    result = run_react_loop(
        [{"role": "user", "content": "old"}],
        "model",
        llm,
        bridge=_ControlBridge(handle),
        offload_dir=str(tmp_path),
    )
    assert result.status == "done"
    assert "exact redirect text" in llm.requests[0][1][-1]["content"]
    assert not handle._pending_messages
