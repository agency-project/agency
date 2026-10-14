"""Tests for agency.llm.response_check: which provider responses are a turn an
agent can act on, and which are an empty, cut-off or malformed answer that
must not be mistaken for one."""

from __future__ import annotations

import pytest

from agency.llm.response_check import (
    EMPTY,
    MALFORMED_TOOL_CALL,
    TRUNCATED,
    inspect_response,
    mark_incomplete_tool_calls,
    tool_arguments_complete,
)

# Arguments of a real tool call that stopped at the output-token limit.
CUT = '{"cmd":"cat <<\'EOF\' > notes.txt\\n1. Linux scheduling determines which'


def _message(*blocks):
    return {"role": "assistant", "blocks": [dict(b, index=i) for i, b in enumerate(blocks)]}


def _text(text):
    return {"type": "text", "text": text}


def _tool(arguments, name="exec_command", call_id="call_1"):
    return {"type": "tool_use", "id": call_id, "name": name, "arguments": arguments}


METADATA = {"type": "metadata", "usage": None, "stop_reason": "stop"}
REASONING = {"type": "thinking", "text": "", "signature": "openai_responses:rsn"}


@pytest.mark.parametrize(
    "arguments,complete",
    [
        ("", True),
        (None, True),
        ("{}", True),
        ('{"cmd":"ls -la"}', True),
        (CUT, False),
        ('{"cmd":"ls"', False),
        ("not json", False),
        ("[1, 2]", False),
        ('"a string"', False),
        ({"cmd": "ls"}, False),  # arguments are always a JSON string on the wire
    ],
)
def test_tool_arguments_complete(arguments, complete):
    assert tool_arguments_complete(arguments) is complete


@pytest.mark.parametrize(
    "message",
    [
        None,
        _message(),
        _message(METADATA),
        _message(_text("")),
        _message(_text("  \n")),
        _message(REASONING),
        _message(REASONING, METADATA),
    ],
    ids=["none", "no-blocks", "metadata-only", "empty-text", "blank-text", "reasoning", "both"],
)
def test_response_without_text_or_tool_call_is_empty(message):
    verdict = inspect_response(message, "stop")
    assert not verdict.usable and verdict.reason == EMPTY
    assert "stop_reason='stop'" in verdict.detail


def test_empty_response_is_usable_only_when_explicitly_allowed():
    assert inspect_response(_message(), "stop", empty_is_unusable=False).usable


@pytest.mark.parametrize(
    "message,stop_reason",
    [
        (_message(_text("done")), "stop"),
        (_message(_tool('{"cmd":"ls"}')), "tool_calls"),
        (_message(_text(""), _tool('{"cmd":"ls"}')), "stop"),
        (_message(REASONING, _tool("")), "tool_calls"),
        (_message({"type": "openai_responses_refusal", "data": {"refusal": "no"}}), "stop"),
        (_message(_text("done")), None),
    ],
    ids=["text", "tool", "empty-text-plus-tool", "reasoning-plus-tool", "refusal", "no-reason"],
)
def test_usable_responses(message, stop_reason):
    verdict = inspect_response(message, stop_reason)
    assert verdict.usable and verdict.reason is None and verdict.incomplete_tool_calls == ()


@pytest.mark.parametrize("stop_reason", ["length", "max_tokens", "max_output_tokens"])
def test_stop_at_the_output_limit_is_truncated_whatever_arrived(stop_reason):
    for message in (_message(_text("The answer is")), _message(_tool('{"cmd":"ls"}')), _message()):
        verdict = inspect_response(message, stop_reason)
        assert not verdict.usable and verdict.reason == TRUNCATED


def test_truncation_separates_whole_tool_calls_from_cut_off_ones():
    message = _message(_tool('{"cmd":"ls"}', call_id="whole"), _tool(CUT, call_id="cut"))
    verdict = inspect_response(message, "length")
    assert verdict.reason == TRUNCATED
    assert verdict.incomplete_tool_calls == ("cut",)
    assert "1 of 2 tool call(s) incomplete" in verdict.detail


def test_unparseable_arguments_without_truncation_are_a_malformed_tool_call():
    verdict = inspect_response(_message(_tool(CUT)), "tool_calls")
    assert not verdict.usable and verdict.reason == MALFORMED_TOOL_CALL
    assert verdict.incomplete_tool_calls == ("call_1",)


def test_tool_call_without_a_name_is_malformed():
    assert inspect_response(_message(_tool("{}", name="")), "tool_calls").reason == (
        MALFORMED_TOOL_CALL
    )


def test_mark_incomplete_flags_only_the_calls_that_cannot_run():
    message = _message(
        _text("x"), _tool('{"cmd":"ls"}', call_id="whole"), _tool(CUT, call_id="cut")
    )
    assert mark_incomplete_tool_calls(message) == ["cut"]
    whole, cut = message["blocks"][1], message["blocks"][2]
    assert "incomplete" not in whole and cut["incomplete"] is True
    assert cut["id"] == "cut" and cut["arguments"] == CUT  # evidence is kept, not rewritten
