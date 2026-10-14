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
