"""Tests for agcompaction — context-limit fetch, tail selection, pruning, and compact()."""
from __future__ import annotations
import json
from unittest.mock import MagicMock, patch

import pytest

from agency.agdata import agdata as _agdata

from agency.agcompaction import (
    TAIL_TURNS,
    _COMPACT_THRESHOLD,
    _TAIL_MAX_TOKENS,
    _TAIL_MIN_TOKENS,
    _TAIL_FRACTION,
    _TOOL_OUTPUT_MAX_CHARS,
    _PRUNE_MIN_FREE_TOKENS,
    DEFAULT_CONTEXT_LIMIT,
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
# Use a large context so the 70% threshold is well-defined.
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
    threshold = int(BIG_CTX * _COMPACT_THRESHOLD)
    assert not should_compact(threshold - 1, BIG_CTX)


def test_should_compact_above_threshold():
    threshold = int(BIG_CTX * _COMPACT_THRESHOLD)
    assert should_compact(threshold + 1, BIG_CTX)


def test_should_compact_at_exact_threshold():
    threshold = int(BIG_CTX * _COMPACT_THRESHOLD)
    assert should_compact(threshold, BIG_CTX)


def test_should_compact_small_model():
    small_ctx = 10_000
    threshold = int(small_ctx * _COMPACT_THRESHOLD)
    assert not should_compact(threshold - 1, small_ctx)
    assert should_compact(threshold, small_ctx)


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
    mock_client.models.list.return_value = [mock_info]

    with patch("agency.agcompaction.openai.OpenAI", return_value=mock_client):
        result = fetch_context_limit(LLM_CONFIG)
    assert result == 131072


def test_fetch_context_limit_model_with_slash_in_name():
    """Model names like 'nvidia/foo' must not trigger a 404 via retrieve()."""
    cfg = {**LLM_CONFIG, "model": "nvidia/MiniMax-M2.7-NVFP4"}
    mock_info = MagicMock()
    mock_info.id = "nvidia/MiniMax-M2.7-NVFP4"
    mock_info.model_extra = {"max_model_len": 196000}
    mock_client = MagicMock()
    mock_client.models.list.return_value = [mock_info]

    with patch("agency.agcompaction.openai.OpenAI", return_value=mock_client):
        result = fetch_context_limit(cfg)
    assert result == 196000
    mock_client.models.retrieve.assert_not_called()


def test_fetch_context_limit_config_wins_over_vllm():
    cfg = {**LLM_CONFIG, "context_limit": 8192}
    mock_info = MagicMock()
    mock_info.model_extra = {"max_model_len": 131072}
    mock_client = MagicMock()
    mock_client.models.list.return_value = [mock_info]

    with patch("agency.agcompaction.openai.OpenAI", return_value=mock_client):
        result = fetch_context_limit(cfg)
    assert result == 8192


def test_fetch_context_limit_fallback_when_unavailable():
    """When the API is unreachable, fall back to DEFAULT_CONTEXT_LIMIT (not None)."""
    mock_client = MagicMock()
    mock_client.models.list.side_effect = Exception("connection refused")

    with patch("agency.agcompaction.openai.OpenAI", return_value=mock_client):
        result = fetch_context_limit(LLM_CONFIG)
    assert result == DEFAULT_CONTEXT_LIMIT


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
    over = int(limit * _COMPACT_THRESHOLD) + 1

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
                  sandbox=MagicMock(), _context_limit=limit)

    assert len(compact_calls) == 1


def test_agskill_skips_compaction_when_under_threshold():
    from agency.agskill import agskill
    from agency.agdata import agdata

    skill = agskill(name="test", system_prompt="You are helpful.", replace_tools=[])

    limit = BIG_CTX
    under = int(limit * _COMPACT_THRESHOLD) - 1

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
                  sandbox=MagicMock(), _context_limit=limit)

    assert len(compact_calls) == 0


def test_agskill_passes_context_limit_to_compact():
    """compact() must receive the context_limit kwarg so tail sizing is correct."""
    from agency.agskill import agskill
    from agency.agdata import agdata

    skill = agskill(name="test", system_prompt="You are helpful.", replace_tools=[])
    limit = BIG_CTX
    over = int(limit * _COMPACT_THRESHOLD) + 1

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
                  sandbox=MagicMock(), _context_limit=limit)

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
# _maybe_compact force parameter
# ---------------------------------------------------------------------------

def test_maybe_compact_force_bypasses_threshold():
    """force=True must trigger compaction even when token estimate is below threshold."""
    from agency.agskill import _maybe_compact

    # Tiny messages — well below threshold
    messages = [
        {"role": "system",    "content": "sys"},
        {"role": "user",      "content": "task"},
        {"role": "assistant", "content": "reply"},
    ]
    compact_calls = []

    def fake_compact(msgs, cfg, **kw):
        compact_calls.append(True)
        return msgs, "summary"

    with patch("agency.agskill.compact", side_effect=fake_compact):
        _maybe_compact(messages, LLM_CONFIG, BIG_CTX, None, None, None, None, None, "test", force=True)

    assert len(compact_calls) == 1


def test_maybe_compact_no_force_below_threshold_does_nothing():
    """Without force, _maybe_compact must not fire when estimate is below threshold."""
    from agency.agskill import _maybe_compact

    messages = [
        {"role": "system",    "content": "sys"},
        {"role": "user",      "content": "task"},
        {"role": "assistant", "content": "reply"},
    ]
    compact_calls = []

    def fake_compact(msgs, cfg, **kw):
        compact_calls.append(True)
        return msgs, "summary"

    with patch("agency.agskill.compact", side_effect=fake_compact):
        _maybe_compact(messages, LLM_CONFIG, BIG_CTX, None, None, None, None, None, "test", force=False)

    assert len(compact_calls) == 0


# ---------------------------------------------------------------------------
# Reactive compaction: context-exceeded 400 error
# ---------------------------------------------------------------------------

def test_agskill_context_exceeded_triggers_forced_compaction():
    """A context_exceeded result must call compact() (bypassing the threshold) and retry the LLM.

    force=True is an _maybe_compact parameter, not forwarded to compact() itself.
    We verify forced compaction fired by checking that compact() was called even
    though the message estimate is far below the threshold (proactive path would
    not fire on tiny messages).
    """
    from agency.agskill import agskill, _LLMCallResult
    from agency.agdata import agdata

    skill = agskill(name="test", system_prompt="You are helpful.", replace_tools=[])

    compact_calls: list[int] = []

    def fake_compact(messages, llm_config, **kw):
        compact_calls.append(len(messages))
        return messages, "summary"

    call_count = 0

    def fake_llm_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _LLMCallResult(context_exceeded=True)
        result = _LLMCallResult()
        result.content_parts = ['{"result": "done"}']
        result.prompt_tokens = 100
        result.total_input_tokens = 100
        result.total_output_tokens = 10
        result.elapsed_ms = 50
        return result

    with patch("agency.agskill.compact", side_effect=fake_compact), \
         patch("agency.agskill._llm_call", side_effect=fake_llm_call):
        skill.run(LLM_CONFIG, agdata(task="x"), agdata(messages=[]),
                  sandbox=MagicMock(), _context_limit=BIG_CTX)

    # compact() must have been called — proactive path won't fire on tiny messages,
    # so any call means the reactive (force=True) path triggered it.
    assert len(compact_calls) >= 1, \
        "compact() must be called after context_exceeded (forced, bypassing threshold)"
    assert call_count == 2, "LLM must be retried after forced compaction"


def test_agskill_context_exceeded_does_not_count_as_retry():
    """context_exceeded compaction must not consume a connection-retry slot."""
    from agency.agskill import agskill, _LLMCallResult
    from agency.agdata import agdata

    skill = agskill(name="test", system_prompt="You are helpful.", replace_tools=[])

    call_count = 0

    def fake_llm_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _LLMCallResult(context_exceeded=True)
        result = _LLMCallResult()
        result.content_parts = ['{"result": "done"}']
        result.prompt_tokens = 100
        result.total_input_tokens = 100
        result.total_output_tokens = 10
        result.elapsed_ms = 50
        return result

    def fake_compact(messages, llm_config, **kw):
        return messages, "summary"

    with patch("agency.agskill.compact", side_effect=fake_compact), \
         patch("agency.agskill._llm_call", side_effect=fake_llm_call):
        result, *_ = skill.run(LLM_CONFIG, agdata(task="x"), agdata(messages=[]),
                               sandbox=MagicMock(), _context_limit=BIG_CTX)

    # Must succeed — context_exceeded is not a connection error
    assert result.error is None
