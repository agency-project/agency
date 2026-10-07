"""T39 (end the run once every output field is submitted) and T41 (dropped tools) in both harness loops."""

from __future__ import annotations

import copy
import json

import pytest

from agency.native_harness import react_loop as native_loop
from agency.tandem_harness import react_loop as tandem_loop


class _Llm:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.requests = []

    def dispatch(self, model, messages, tools=None, **kwargs):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        return next(self._responses)


class _Mcp:
    def __init__(self, missing_after):
        self.missing_after = list(missing_after)

    def discover(self):
        return [{"type": "function", "function": {"name": n, "parameters": {"type": "object", "properties": {}}}}
                for n in ("submit_output", "todowrite")]

    def call(self, name, args_json):
        missing = self.missing_after.pop(0)
        return json.dumps({"result": "field recorded", "missing_output_fields": missing})


def _turn(*calls, text=None):
    tcs = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
    return {"message": {"role": "assistant", "content": text, "tool_calls": tcs}, "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def _closing():
    return {"message": {"role": "assistant", "content": "Done."}, "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


@pytest.mark.parametrize("loop", [native_loop, tandem_loop])
@pytest.mark.parametrize("end_on_submit", [False, True])
def test_end_on_submit_skips_the_closing_turn(monkeypatch, tmp_path, loop, end_on_submit):
    monkeypatch.setattr(loop, "END_ON_SUBMIT", end_on_submit)
    llm = _Llm([_turn(("submit_output", {"field": "a", "value": 1})),
                _turn(("submit_output", {"field": "b", "value": 2})),
                _closing()])
    result = loop.run_react_loop([{"role": "user", "content": "go"}], "m", llm, mcp=_Mcp([["b"], None]),
                                 offload_dir=str(tmp_path))
    assert result.status == "done"
    assert len(llm.requests) == (2 if end_on_submit else 3)


@pytest.mark.parametrize("loop", [native_loop, tandem_loop])
def test_end_on_submit_waits_for_every_field(monkeypatch, tmp_path, loop):
    monkeypatch.setattr(loop, "END_ON_SUBMIT", True)
    llm = _Llm([_turn(("submit_output", {"field": "a", "value": 1})), _closing()])
    loop.run_react_loop([{"role": "user", "content": "go"}], "m", llm, mcp=_Mcp([["b"]]), offload_dir=str(tmp_path))
    assert len(llm.requests) == 2


@pytest.mark.parametrize("loop", [native_loop, tandem_loop])
def test_dropped_tools_are_not_offered(monkeypatch, tmp_path, loop):
    monkeypatch.setattr(loop, "DROP_TOOLS", frozenset({"todowrite", "webfetch"}))
    llm = _Llm([_closing()])
    loop.run_react_loop([{"role": "user", "content": "go"}], "m", llm, mcp=_Mcp([]), offload_dir=str(tmp_path))
    names = {t["function"]["name"] for t in llm.requests[0][1]}
    assert "submit_output" in names and "bash" in names
    assert not names & {"todowrite", "webfetch"}
