"""native_harness checkpoints only complete turns, so a resumed session
never starts from an assistant tool_use without its tool result."""

from __future__ import annotations

import copy
import json

from agency.native_harness import tools
from agency.native_harness.react_loop import run_react_loop


class _Llm:
    def __init__(self, responses):
        self._responses = iter(responses)

    def dispatch(self, model, messages, tools=None, **kwargs):
        return next(self._responses)


def _bash_call(command, call_id):
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": command})},
                }
            ],
        },
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def test_checkpoint_is_taken_only_after_tool_results(monkeypatch, tmp_path):
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: json.dumps({"ok": True}))
    llm = _Llm(
        [
            _bash_call("one", "c1"),
            _bash_call("two", "c2"),
            {"message": {"role": "assistant", "content": "done"}, "usage": {}},
        ]
    )
    checkpoints = []
    result = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        on_checkpoint=lambda messages: checkpoints.append(copy.deepcopy(messages)),
        offload_dir=str(tmp_path),
    )
    assert result.status == "done"
    assert [c[-1]["tool_call_id"] for c in checkpoints] == ["c1", "c2"]
    assert all(c[-1]["role"] == "tool" for c in checkpoints)


class _Bridge:
    """Admits every call; completion hands back system context (sysinfo expose "context")."""

    def __init__(self, context):
        self._context = context
        self.completed = []

    def check_tool_policy(self, name, tool_input):
        return {"decision": "allow", "call_id": f"call-{len(self.completed)}"}

    def complete_tool_policy(self, call_id, result, **kwargs):
        self.completed.append(call_id)
        return self._context


def test_system_context_from_completion_is_appended_to_the_tool_result(monkeypatch, tmp_path):
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: json.dumps({"ok": True}))
    llm = _Llm(
        [
            _bash_call("one", "c1"),
            {"message": {"role": "assistant", "content": "done"}, "usage": {}},
        ]
    )
    checkpoints = []
    run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        bridge=_Bridge("[system] during this command (1.0 s) ..."),
        on_checkpoint=lambda messages: checkpoints.append(copy.deepcopy(messages)),
        offload_dir=str(tmp_path),
    )
    tool_message = checkpoints[0][-1]
    assert tool_message["content"] == '{"ok": true}\n\n[system] during this command (1.0 s) ...'


def test_no_system_context_leaves_the_tool_result_unchanged(monkeypatch, tmp_path):
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: json.dumps({"ok": True}))
    llm = _Llm(
        [
            _bash_call("one", "c1"),
            {"message": {"role": "assistant", "content": "done"}, "usage": {}},
        ]
    )
    checkpoints = []
    run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        bridge=_Bridge(None),
        on_checkpoint=lambda messages: checkpoints.append(copy.deepcopy(messages)),
        offload_dir=str(tmp_path),
    )
    assert checkpoints[0][-1]["content"] == '{"ok": true}'
