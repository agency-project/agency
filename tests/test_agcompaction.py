"""Tests for agcompaction — context-limit fetch, tail selection, pruning, and compact()."""
from __future__ import annotations
import json
from unittest.mock import MagicMock, patch

import pytest

from agency.agdata import agdata as _agdata

from agency.agcompaction import (
    TAIL_TURNS,
    _RESERVED,
    _TAIL_MAX_TOKENS,
    _TAIL_MIN_TOKENS,
    _TAIL_FRACTION,
    _TOOL_OUTPUT_MAX_CHARS,
    _PRUNE_MIN_FREE_TOKENS,
    _estimate_tokens,
    _prune_tool_outputs,
    _tail_start,
    compact,
    fetch_context_limit,
    should_compact,
)

LLM_CONFIG = {"api_key": "test", "model": "gpt-4o", "base_url": "http://localhost/v1"}

# Module-level so ProcessPoolExecutor can pickle it.
# Returns content large enough to push estimated tokens well past BIG_CTX threshold.
_LARGE_CONTENT = "x" * (400_000)  # ~100k tokens estimated (chars // 4)

def _large_file_tool_fn(arg: _agdata) -> _agdata:
    return _agdata(output=_LARGE_CONTENT)
# Use a large context so the threshold is clearly context_limit - _RESERVED.
BIG_CTX = 100_000


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------

def test_estimate_tokens_content():
    msg = {"role": "assistant", "content": "a" * 400}  # 400 chars → 100 tokens
    assert _estimate_tokens(msg) == 100


def test_estimate_tokens_tool_args():
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"arguments": "x" * 800}}],
    }
    assert _estimate_tokens(msg) == 200


def test_estimate_tokens_minimum_one():
    assert _estimate_tokens({"role": "user", "content": ""}) == 1


# ---------------------------------------------------------------------------
# _prune_tool_outputs
# ---------------------------------------------------------------------------

def _big_tool_msg(chars: int) -> dict:
    return {"role": "tool", "content": "x" * chars, "tool_call_id": "t1"}


def test_prune_does_nothing_when_savings_below_threshold():
    # One tool result slightly over limit but savings < 20K tokens
    msg = _big_tool_msg(_TOOL_OUTPUT_MAX_CHARS + 100)
    msgs = [msg]
    result = _prune_tool_outputs(msgs)
    assert result[0]["content"] == msg["content"]  # unchanged


def test_prune_trims_when_savings_above_threshold():
    # Many large tool results — total savings > 20K tokens
    big = _TOOL_OUTPUT_MAX_CHARS + _PRUNE_MIN_FREE_TOKENS * 4 + 1000
    msgs = [_big_tool_msg(big)]
    result = _prune_tool_outputs(msgs)
    assert len(result[0]["content"]) == _TOOL_OUTPUT_MAX_CHARS + len("\n[truncated]")


def test_prune_leaves_small_tool_results_intact():
    small = {"role": "tool", "content": "small result", "tool_call_id": "t2"}
    # Add enough large results to cross the threshold, but keep one small
    big = _TOOL_OUTPUT_MAX_CHARS + _PRUNE_MIN_FREE_TOKENS * 4 + 1000
    msgs = [_big_tool_msg(big), small]
    result = _prune_tool_outputs(msgs)
    assert result[1]["content"] == "small result"


def test_prune_leaves_non_tool_messages_intact():
    big = _TOOL_OUTPUT_MAX_CHARS + _PRUNE_MIN_FREE_TOKENS * 4 + 1000
    asst = {"role": "assistant", "content": "x" * big}
    msgs = [asst, _big_tool_msg(big)]
    result = _prune_tool_outputs(msgs)
    assert result[0]["content"] == asst["content"]   # assistant untouched


# ---------------------------------------------------------------------------
# _tail_start
# ---------------------------------------------------------------------------

def _conv(*roles: str) -> list[dict]:
    return [{"role": r, "content": f"msg-{i}"} for i, r in enumerate(roles)]


def test_tail_start_empty():
    assert _tail_start([], BIG_CTX) == 0


def test_tail_start_fewer_turns_than_requested():
    # Two assistant turns exist but tail_turns=3 requested — keep all ReAct turns.
    # In compact(), this means head = conv[1:1] = [] → no compaction.
    conv = _conv("user", "assistant", "user", "assistant")
    ts = _tail_start(conv, BIG_CTX, tail_turns=3)
    # tail starts at or before index 1 (the first assistant), meaning nothing left to summarise
    assert ts <= 1


def test_tail_start_exact_one_turn():
    conv = _conv("user", "assistant", "user", "assistant")
    ts = _tail_start(conv, BIG_CTX, tail_turns=1)
    # Keep only the last assistant turn (index 3)
    assert ts == 3
    assert conv[ts]["role"] == "assistant"


def test_tail_start_two_turns():
    conv = _conv("user", "assistant", "user", "assistant", "user", "assistant")
    ts = _tail_start(conv, BIG_CTX, tail_turns=2)
    # Keep last two assistant turns; tail starts at second-to-last assistant (index 3)
    assert ts == 3
    assert conv[ts]["role"] == "assistant"


def test_tail_start_with_tool_messages():
    conv = [
        {"role": "user",      "content": "start"},
        {"role": "assistant", "content": None, "tool_calls": [{}]},
        {"role": "tool",      "content": "r1"},
        {"role": "assistant", "content": None, "tool_calls": [{}]},
        {"role": "tool",      "content": "r2"},
        {"role": "user",      "content": "next"},
        {"role": "assistant", "content": "done"},
    ]
    ts = _tail_start(conv, BIG_CTX, tail_turns=2)
    # Second-to-last assistant is at index 3
    assert ts == 3
    assert conv[ts]["role"] == "assistant"


def test_tail_start_respects_token_budget():
    # Create a turn whose token estimate exceeds _TAIL_MAX_TOKENS on its own.
    # It should still be kept as the first (and only) turn — the budget cap
    # only applies when a second turn would be added.
    huge_content = "x" * (_TAIL_MAX_TOKENS * 4 * 2)   # >> _TAIL_MAX_TOKENS tokens
    conv = [
        {"role": "user",      "content": "task"},
        {"role": "assistant", "content": "small first turn"},
        {"role": "assistant", "content": huge_content},
    ]
    ts = _tail_start(conv, BIG_CTX, tail_turns=2)
    # The huge turn is always kept (single turn always accepted).
    # Whether the small first turn is also kept depends on budget.
    # With budget=8000 and huge turn >> 8000, only the huge turn fits → ts=2.
    assert ts == 2


# ---------------------------------------------------------------------------
# should_compact
# ---------------------------------------------------------------------------

def test_should_compact_below_threshold():
    # threshold = max(BIG_CTX - _RESERVED, BIG_CTX // 2) = max(80000, 50000) = 80000
    assert not should_compact(79_999, BIG_CTX)


def test_should_compact_above_threshold():
    assert should_compact(80_001, BIG_CTX)


def test_should_compact_at_exact_threshold():
    assert should_compact(80_000, BIG_CTX)


def test_should_compact_floor_for_small_models():
    # Model smaller than _RESERVED: threshold should be 50% floor, not negative
    small_ctx = 10_000
    threshold = max(small_ctx - _RESERVED, small_ctx // 2)
    assert threshold == 5_000
    assert not should_compact(4_999, small_ctx)
    assert should_compact(5_000, small_ctx)


# ---------------------------------------------------------------------------
# fetch_context_limit
# ---------------------------------------------------------------------------

def test_fetch_context_limit_from_config():
    cfg = {**LLM_CONFIG, "context_limit": "32768"}
    assert fetch_context_limit(cfg) == 32768


def test_fetch_context_limit_from_vllm():
    mock_info = MagicMock()
    mock_info.model_extra = {"max_model_len": 131072}
    mock_client = MagicMock()
    mock_client.models.retrieve.return_value = mock_info

    with patch("agency.agcompaction.openai.OpenAI", return_value=mock_client):
        result = fetch_context_limit(LLM_CONFIG)
    assert result == 131072


def test_fetch_context_limit_config_wins_over_vllm():
    cfg = {**LLM_CONFIG, "context_limit": 8192}
    mock_info = MagicMock()
    mock_info.model_extra = {"max_model_len": 131072}
    mock_client = MagicMock()
    mock_client.models.retrieve.return_value = mock_info

    with patch("agency.agcompaction.openai.OpenAI", return_value=mock_client):
        result = fetch_context_limit(cfg)
    assert result == 8192


def test_fetch_context_limit_none_when_unavailable():
    mock_client = MagicMock()
    mock_client.models.retrieve.side_effect = Exception("not found")

    with patch("agency.agcompaction.openai.OpenAI", return_value=mock_client):
        result = fetch_context_limit(LLM_CONFIG)
    assert result is None


# ---------------------------------------------------------------------------
# compact()
# ---------------------------------------------------------------------------

def _mock_compact_response(summary_text: str):
    msg = MagicMock()
    msg.content = summary_text
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def _make_messages(n_turns: int, *, with_tools: bool = False) -> list[dict]:
    """Build: system + user(task) + N assistant turns (optionally with tool calls)."""
    msgs: list[dict] = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user",   "content": '{"task": "do the thing"}'},
    ]
    for i in range(n_turns):
        if with_tools:
            msgs.append({"role": "assistant", "content": None,
                          "tool_calls": [{"id": f"c{i}", "type": "function",
                                          "function": {"name": "bash",
                                                       "arguments": f'{{"command":"cmd{i}"}}'}}]})
            msgs.append({"role": "tool", "content": f"result {i}", "tool_call_id": f"c{i}"})
        else:
            msgs.append({"role": "assistant", "content": f"assistant reply {i}"})
    return msgs


def test_compact_returns_shorter_list():
    messages = _make_messages(6)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_compact_response("## Goal\nTest.")

    with patch("agency.agcompaction.openai.OpenAI", return_value=mock_client):
        new_msgs, summary = compact(messages, LLM_CONFIG, context_limit=BIG_CTX)

    assert summary == "## Goal\nTest."
    assert len(new_msgs) < len(messages)
    assert new_msgs[0]["role"] == "system"
    assert any("summary" in m.get("content", "").lower() for m in new_msgs)


def test_compact_preserves_task_input():
    """conv[0] (the skill task input) must always appear verbatim after the system msg."""
    messages = _make_messages(6)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_compact_response("summary")

    with patch("agency.agcompaction.openai.OpenAI", return_value=mock_client):
        new_msgs, _ = compact(messages, LLM_CONFIG, context_limit=BIG_CTX)

    # system at [0], task user input at [1]
    assert new_msgs[0]["role"] == "system"
    assert new_msgs[1]["role"] == "user"
    assert new_msgs[1]["content"] == '{"task": "do the thing"}'


def test_compact_preserves_tail_turns():
    messages = _make_messages(6)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_compact_response("summary")

    with patch("agency.agcompaction.openai.OpenAI", return_value=mock_client):
        new_msgs, _ = compact(messages, LLM_CONFIG, context_limit=BIG_CTX, tail_turns=2)

    assistant_bodies = [
        m["content"] for m in new_msgs
        if m["role"] == "assistant" and "Understood" not in (m.get("content") or "")
    ]
    assert "assistant reply 4" in assistant_bodies
    assert "assistant reply 5" in assistant_bodies


def test_compact_nothing_to_summarise():
    # Only 1 assistant turn → tail_turns=2 means nothing in head → no-op
    messages = _make_messages(1)
    mock_client = MagicMock()

    with patch("agency.agcompaction.openai.OpenAI", return_value=mock_client):
        new_msgs, summary = compact(messages, LLM_CONFIG, context_limit=BIG_CTX, tail_turns=2)

    assert new_msgs == messages
    assert summary == ""
    mock_client.chat.completions.create.assert_not_called()


def test_compact_incremental_with_previous_summary():
    messages = _make_messages(4)
    prev = "## Goal\nPrevious task."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_compact_response("updated summary")

    with patch("agency.agcompaction.openai.OpenAI", return_value=mock_client):
        _, summary = compact(messages, LLM_CONFIG, context_limit=BIG_CTX,
                             previous_summary=prev)

    assert summary == "updated summary"
    user_content = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert prev in user_content


def test_compact_task_input_included_in_summarisation_prompt():
    """The task input message should appear in the summarisation prompt so the
    LLM knows what goal the compacted turns were working toward."""
    messages = _make_messages(4)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_compact_response("s")

    with patch("agency.agcompaction.openai.OpenAI", return_value=mock_client):
        compact(messages, LLM_CONFIG, context_limit=BIG_CTX)

    user_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "do the thing" in user_prompt   # task content appears in prompt


def test_compact_prunes_large_tool_outputs():
    """When head contains large tool results that hit the prune threshold,
    they should be trimmed before being sent to the summariser."""
    # Build messages with a giant tool result in the head
    huge = _TOOL_OUTPUT_MAX_CHARS + _PRUNE_MIN_FREE_TOKENS * 4 + 100
    messages = [
        {"role": "system",    "content": "sys"},
        {"role": "user",      "content": '{"task": "x"}'},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c0", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        ]},
        {"role": "tool",      "content": "y" * huge, "tool_call_id": "c0"},
        # Two more turns kept as tail
        {"role": "assistant", "content": "turn 1"},
        {"role": "assistant", "content": "turn 2"},
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_compact_response("s")

    with patch("agency.agcompaction.openai.OpenAI", return_value=mock_client):
        compact(messages, LLM_CONFIG, context_limit=BIG_CTX, tail_turns=2)

    user_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    # Pruned content appears truncated in the prompt
    assert "[truncated]" in user_prompt or len(user_prompt) < huge // 2


# ---------------------------------------------------------------------------
# Integration: agskill triggers compaction when threshold exceeded
# ---------------------------------------------------------------------------

def _make_stream(content: str, prompt_tokens: int) -> list:
    """Build a minimal list of streaming chunks that agskill.run() can iterate."""
    # Chunk 1: content delta
    delta1 = MagicMock()
    delta1.content = content
    delta1.tool_calls = None
    choice1 = MagicMock()
    choice1.delta = delta1
    choice1.finish_reason = None
    chunk1 = MagicMock()
    chunk1.choices = [choice1]
    chunk1.usage = None

    # Chunk 2: final chunk carrying usage, no content
    delta2 = MagicMock()
    delta2.content = None
    delta2.tool_calls = None
    choice2 = MagicMock()
    choice2.delta = delta2
    choice2.finish_reason = "stop"
    chunk2 = MagicMock()
    chunk2.choices = [choice2]
    chunk2.usage = MagicMock(prompt_tokens=prompt_tokens)

    return [chunk1, chunk2]


def test_agskill_triggers_compaction_when_over_threshold():
    from agency.agskill import agskill
    from agency.agdata import agdata

    skill = agskill(name="test", system_prompt="You are helpful.", replace_tools=[])

    limit = BIG_CTX
    over = max(limit - _RESERVED, limit // 2) + 1

    compact_calls = []

    def fake_compact(messages, llm_config, **kw):
        compact_calls.append(len(messages))
        return messages, "summary"

    with patch("agency.agskill.openai.OpenAI") as MockClient, \
         patch("agency.agskill.compact", side_effect=fake_compact):
        MockClient.return_value = MagicMock()
        MockClient.return_value.chat.completions.create.return_value = \
            _make_stream('{"result": "done"}', over)
        skill.run(LLM_CONFIG, agdata(task="x"), agdata(messages=[]),
                  sandbox=None, _context_limit=limit)

    assert len(compact_calls) == 1


def test_agskill_skips_compaction_when_under_threshold():
    from agency.agskill import agskill
    from agency.agdata import agdata

    skill = agskill(name="test", system_prompt="You are helpful.", replace_tools=[])

    limit = BIG_CTX
    under = max(limit - _RESERVED, limit // 2) - 1

    compact_calls = []

    def fake_compact(messages, llm_config, **kw):
        compact_calls.append(True)
        return messages, ""

    with patch("agency.agskill.openai.OpenAI") as MockClient, \
         patch("agency.agskill.compact", side_effect=fake_compact):
        MockClient.return_value = MagicMock()
        MockClient.return_value.chat.completions.create.return_value = \
            _make_stream('{"result": "ok"}', under)
        skill.run(LLM_CONFIG, agdata(task="x"), agdata(messages=[]),
                  sandbox=None, _context_limit=limit)

    assert len(compact_calls) == 0


def test_agskill_passes_context_limit_to_compact():
    """compact() must receive the context_limit kwarg so tail sizing is correct."""
    from agency.agskill import agskill
    from agency.agdata import agdata

    skill = agskill(name="test", system_prompt="You are helpful.", replace_tools=[])
    limit = BIG_CTX
    over = max(limit - _RESERVED, limit // 2) + 1

    received_kwargs = {}

    def fake_compact(messages, llm_config, **kw):
        received_kwargs.update(kw)
        return messages, "summary"

    with patch("agency.agskill.openai.OpenAI") as MockClient, \
         patch("agency.agskill.compact", side_effect=fake_compact):
        MockClient.return_value = MagicMock()
        MockClient.return_value.chat.completions.create.return_value = \
            _make_stream('{"result": "done"}', over)
        skill.run(LLM_CONFIG, agdata(task="x"), agdata(messages=[]),
                  sandbox=None, _context_limit=limit)

    assert received_kwargs.get("context_limit") == limit


# ---------------------------------------------------------------------------
# Helpers for tool-call streaming mocks
# ---------------------------------------------------------------------------

class _TCFn:
    def __init__(self, name: str, args: str) -> None:
        self.name = name
        self.arguments = args


class _TC:
    def __init__(self, name: str, args_json: str, call_id: str = "c1") -> None:
        self.id = call_id
        self.index = 0
        self.function = _TCFn(name, args_json)


def _make_tool_call_stream(name: str, args: dict, call_id: str = "c1") -> list:
    """Minimal streaming chunks representing a tool-call LLM response."""
    tc = _TC(name, json.dumps(args), call_id)

    delta1 = MagicMock()
    delta1.content = None
    delta1.tool_calls = [tc]
    delta1.model_extra = {}
    delta1.reasoning_content = None
    choice1 = MagicMock()
    choice1.delta = delta1
    chunk1 = MagicMock()
    chunk1.choices = [choice1]
    chunk1.usage = None

    delta2 = MagicMock()
    delta2.content = None
    delta2.tool_calls = None
    delta2.model_extra = {}
    delta2.reasoning_content = None
    choice2 = MagicMock()
    choice2.delta = delta2
    chunk2 = MagicMock()
    chunk2.choices = [choice2]
    chunk2.usage = MagicMock(prompt_tokens=500)

    return [chunk1, chunk2]


# ---------------------------------------------------------------------------
# Pre-call compaction: tool result overloads context
# ---------------------------------------------------------------------------

def test_agskill_compacts_before_llm_call_when_tool_result_overloads_context():
    """A single tool call that returns a large blob should trigger pre-call
    compaction before the next LLM request, even if the context was under the
    threshold before the tool call ran."""
    from agency.agskill import agskill
    from agency.agdata import agdata
    from agency.agtool import agtool

    limit = BIG_CTX  # 100_000
    # threshold = max(100_000 - 20_000, 50_000) = 80_000
    # _LARGE_CONTENT is 400_000 chars → ~100_000 estimated tokens → over threshold

    read_tool = agtool(
        name="read_file",
        description="Read a file",
        fn=_large_file_tool_fn,
        params={"type": "object", "properties": {"path": {"type": "string"}},
                "required": ["path"]},
    )
    skill = agskill(name="test", system_prompt="You are helpful.",
                    replace_tools=[read_tool])

    compact_calls: list[int] = []

    def fake_compact(messages, llm_config, **kw):
        compact_calls.append(len(messages))
        # Return a drastically shorter list so the second LLM call succeeds
        return messages[:3], "summary"

    responses = [
        _make_tool_call_stream("read_file", {"path": "/workspace/big.txt"}),
        _make_stream('{"result": "done"}', 100),
    ]
    response_iter = iter(responses)

    with patch("agency.agskill.openai.OpenAI") as MockClient, \
         patch("agency.agskill.compact", side_effect=fake_compact), \
         patch("agency.agcompaction.httpx.post", side_effect=ConnectionError("no vllm")):
        MockClient.return_value = MagicMock()
        MockClient.return_value.chat.completions.create.side_effect = \
            lambda **kw: next(response_iter)
        skill.run(LLM_CONFIG, agdata(task="read the file"), agdata(messages=[]),
                  sandbox=None, _context_limit=limit)

    assert len(compact_calls) >= 1, \
        "compact() should have been triggered before the second LLM call " \
        "because the tool result pushed estimated tokens past the threshold"
