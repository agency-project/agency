"""agency/llm/context_fit.py: a request over the context limit loses its
oldest tool output, not its structure."""

import copy

import pytest

from agency.llm.context_fit import (
    CLEARED,
    SizeEstimator,
    fit_messages,
    measured_rate,
    message_chars,
)


def _conversation(results: int, size: int) -> "list[dict]":
    messages = [{"role": "user", "blocks": [{"type": "text", "index": 0, "text": "task"}]}]
    for i in range(results):
        messages.append(
            {
                "role": "assistant",
                "blocks": [
                    {
                        "type": "tool_use",
                        "index": 0,
                        "id": f"c{i}",
                        "name": "bash",
                        "arguments": "{}",
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "blocks": [
                    {"type": "tool_result", "index": 0, "tool_call_id": f"c{i}", "text": "x" * size}
                ],
            }
        )
    return messages


def _texts(messages):
    return [m["blocks"][0].get("text") for m in messages if m["role"] == "tool"]


def test_a_request_that_fits_is_untouched():
    messages = _conversation(3, 1_000)
    fitted, cleared = fit_messages(messages, budget=10_000)
    assert fitted is messages and cleared == 0


def test_the_oldest_tool_results_are_cleared_first_and_pairs_stay():
    messages = _conversation(5, 1_000)
    # ~5,000 characters, budget 3,500: two results must go.
    fitted, cleared = fit_messages(messages, budget=3_500)
    assert cleared == 2
    assert _texts(fitted) == [CLEARED, CLEARED, "x" * 1_000, "x" * 1_000, "x" * 1_000]
    assert [b["type"] for m in fitted for b in m["blocks"]] == [
        b["type"] for m in messages for b in m["blocks"]
    ]
    assert [b.get("tool_call_id") for m in fitted for b in m["blocks"]] == [
        b.get("tool_call_id") for m in messages for b in m["blocks"]
    ]
    assert sum(message_chars(m) for m in fitted) <= 3_500
    assert _texts(messages)[0] == "x" * 1_000  # the input is not modified


def test_a_newest_result_too_big_on_its_own_keeps_its_head_and_tail():
    messages = _conversation(2, 1_000)
    messages[-1]["blocks"][0]["text"] = "a" * 5_000 + "b" * 5_000
    fitted, _ = fit_messages(messages, budget=4_000)
    newest = _texts(fitted)[-1]
    assert newest.startswith("a") and newest.endswith("b") and CLEARED in newest
    assert sum(message_chars(m) for m in fitted) <= 4_000


@pytest.mark.parametrize("kind", ["function", "custom_tool"])
@pytest.mark.parametrize("newest_size", [1_000, 10_000])
@pytest.mark.parametrize("character", ["y", "\u754c\n"])
def test_native_responses_history_is_fitted_and_serialized_without_losing_call_ids(
    kind, newest_size, character
):
    from agency.configs.agconfig import agconfig
    from agency.llm.message_store import split_tool_schema
    from agency.llm.responses_native import (
        NATIVE_REQUEST,
        context_from_request,
        request_from_agency,
    )

    items = [
        {"type": "message", "role": "user", "content": "task"},
        {"type": f"{kind}_call", "call_id": "c0", "name": "bash", "input": "pwd"}
        if kind == "custom_tool"
        else {"type": "function_call", "call_id": "c0", "name": "bash", "arguments": "{}"},
        {"type": f"{kind}_call_output", "call_id": "c0", "output": "x" * 5_000},
        {"type": f"{kind}_call", "call_id": "c1", "name": "bash", "input": "pwd"}
        if kind == "custom_tool"
        else {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": "{}"},
        {"type": f"{kind}_call_output", "call_id": "c1", "output": character * newest_size},
    ]
    original = copy.deepcopy(items)
    context = context_from_request({"model": "test", "input": items})
    assert sum(message_chars(m) for m in context["messages"]) > 4_000
    fitted, cleared = fit_messages(context["messages"], budget=4_000)
    assert cleared > 0
    assert sum(message_chars(m) for m in fitted) <= 4_000
    messages, tools, choice, extra = split_tool_schema(fitted)
    rendered = request_from_agency(messages, tools, choice, extra[NATIVE_REQUEST], agconfig().llm)
    outputs = rendered["extra_body"]["input"]
    assert outputs[2] == {"type": f"{kind}_call_output", "call_id": "c0", "output": CLEARED}
    assert outputs[0:2] == original[0:2]
    assert outputs[3] == original[3]
    assert outputs[4]["call_id"] == "c1"
    if newest_size == 1_000 and character == "y":
        assert outputs[4] == original[4]
    else:
        assert outputs[4]["output"].startswith(character)
        assert original[4]["output"].endswith(outputs[4]["output"].split(CLEARED)[-1][1:])
        assert CLEARED in outputs[4]["output"]
    assert items == original


def test_measured_rate_is_bounded():
    assert measured_rate(1_000, 4_000) == 0.25
    assert measured_rate(10_000, 1_000) == 1.0
    assert measured_rate(10, 1_000) == 0.2
    assert measured_rate(0, 1_000) is None


def test_the_estimate_costs_what_was_added_at_the_rate_the_last_step_measured():
    size = SizeEstimator()
    # 40K characters of schemas and prose counted 10K tokens (0.25/char) ...
    size.observe(40_000, 10_000)
    # ... then 10K characters of dense tool output added 7K tokens (0.7/char).
    size.observe(50_000, 17_000)
    assert size.step_rate == 0.7
    # Limit 60K, target 57K: 40K tokens to spare from 17K, at 0.7/char.
    assert size.budget(55_000, 60_000) == 50_000 + int(40_000 / 0.7)
    # A request smaller than the last (after a compaction) falls back to the
    # whole conversation's rate, never below the step's.
    assert size.budget(30_000, 60_000) == int(57_000 / 0.7)


def test_a_fitted_request_is_no_measure_of_a_step():
    """It cleared as much as it added: its tiny net growth says nothing about
    what the next step's content costs."""
    size = SizeEstimator()
    size.observe(40_000, 10_000)
    size.observe(70_000, 31_000)  # 30K characters added 21K tokens: 0.7
    size.observe(71_000, 31_100, fitted=True)
    size.observe(73_000, 32_000)  # after a fitted one: not a step either
    assert size.step_rate == 0.7


def test_a_light_step_does_not_hide_a_dense_one():
    """A short listing between two dense files must not lower the rate the
    next file is costed at (Kimi sent 63K against a 60K limit that way)."""
    size = SizeEstimator()
    size.observe(40_000, 10_000)
    size.observe(70_000, 31_000)  # dense: 0.7
    size.observe(80_000, 34_000)  # a listing: 0.3
    assert size.step_rate == 0.7
