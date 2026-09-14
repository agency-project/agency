"""Tests for the pure token-estimation/tail-selection/pruning/prompt-
building algorithms in native_harness/compaction.py."""

from __future__ import annotations

from agency.native_harness import compaction


def test_estimate_tokens_content():
    msg = {"role": "assistant", "content": "a" * 400}
    assert compaction.estimate_tokens(msg) == 100


def test_estimate_tokens_minimum_one():
    assert compaction.estimate_tokens({"role": "user", "content": ""}) == 1


def test_should_compact_threshold():
    assert compaction.should_compact(900, 1000) is True
    assert compaction.should_compact(100, 1000) is False


def test_tail_start_empty():
    assert compaction.tail_start([], 100_000) == 0


def test_prune_tool_outputs_leaves_small_results_intact():
    msgs = [{"role": "tool", "content": "small", "tool_call_id": "t1"}]
    assert compaction.prune_tool_outputs(msgs) == msgs


def test_split_for_compaction_separates_system_task_input_head_tail():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old turn 1"},
        {"role": "user", "content": "old turn 1 followup"},
        {"role": "assistant", "content": "recent turn"},
    ]
    sys_msg, task_input, head, tail = compaction.split_for_compaction(
        messages, context_limit=100_000, tail_turns=1
    )
    assert sys_msg == [messages[0]]
    assert task_input == [messages[1]]
    assert tail and tail[0]["content"] == "recent turn"


def test_split_before_first_assistant_turn_keeps_fresh_invocation_message_in_tail():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "user", "content": "[AGENCY INVOCATION MESSAGE]\nMUST_KEEP_EXACT"},
        {"role": "user", "content": "additional current-turn context"},
    ]

    sys_msg, task_input, head, tail = compaction.split_for_compaction(messages, context_limit=1)

    assert sys_msg == [messages[0]]
    assert task_input == [messages[1]]
    assert head == []
    assert tail == messages[2:]


def test_build_summary_prompt_messages_shape():
    task_input = [{"role": "user", "content": "the task"}]
    head = [{"role": "assistant", "content": "did something"}]
    messages = compaction.build_summary_prompt_messages(task_input, head)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == compaction.SUMMARY_SYSTEM
    assert "the task" in messages[1]["content"]
    assert "did something" in messages[1]["content"]


def test_assemble_compacted_messages_injects_summary_and_keeps_tail():
    sys_msg = [{"role": "system", "content": "sys"}]
    task_input = [{"role": "user", "content": "task"}]
    tail = [{"role": "assistant", "content": "recent"}]
    result = compaction.assemble_compacted_messages(sys_msg, task_input, "the summary", tail)
    assert result[0] == sys_msg[0]
    assert result[1] == task_input[0]
    assert "the summary" in result[2]["content"]
    assert result[-1] == tail[0]
