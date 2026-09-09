from __future__ import annotations

import copy
import json

from agency.native_harness import tools
from agency.native_harness.react_loop import run_react_loop


class _Llm:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.requests = []

    def dispatch(self, model, messages, tools=None, **_kwargs):
        self.requests.append((model, copy.deepcopy(messages), copy.deepcopy(tools)))
        return next(self._responses)


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


def test_native_max_steps_counts_model_turns_not_tool_calls(monkeypatch, tmp_path):
    tool_events = []
    monkeypatch.setitem(tools.TOOL_DISPATCH, "bash", lambda args: tool_events.append(args) or "{}")
    llm = _Llm([_tool_response("one", "two"), _tool_response("three"), _final_response()])

    result = run_react_loop(
        [{"role": "user", "content": "start"}],
        "model",
        llm,
        max_steps=2,
        offload_dir=str(tmp_path),
    )

    assert result.status == "error"
    assert result.message == "exceeded max_steps=2 without a final answer"
    assert result.turn_count == 2
    assert len(llm.requests) == 2
    assert len(tool_events) == 3
