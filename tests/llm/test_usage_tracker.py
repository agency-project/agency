"""Tests for LlmUsageTracker -- the branch-matching delta calculation that
isolates each exchange's own new prompt tokens from a provider's cumulative
prompt_tokens figure."""

from __future__ import annotations

from agency.llm.usage_tracker import LlmUsageTracker


def _msg(role: str, text: str) -> dict:
    return {"role": role, "blocks": [{"type": "text", "index": 0, "text": text}]}


def test_first_exchange_has_no_baseline_to_subtract():
    tracker = LlmUsageTracker()
    messages = [_msg("user", "hello")]
    response = _msg("assistant", "hi there")

    new_prompt_tokens = tracker.resolve_new_prompt_tokens(messages, response, 100, 20)

    assert new_prompt_tokens == 100


def test_second_exchange_subtracts_previous_prompt_plus_completion():
    tracker = LlmUsageTracker()
    first_messages = [_msg("user", "hello")]
    first_response = _msg("assistant", "hi there")
    tracker.resolve_new_prompt_tokens(first_messages, first_response, 100, 20)

    # A continuation: the previous exchange's messages + its own response,
    # plus one new user turn -- exactly what a real follow-up call sends.
    second_messages = first_messages + [first_response, _msg("user", "and then?")]
    second_response = _msg("assistant", "then this")

    new_prompt_tokens = tracker.resolve_new_prompt_tokens(second_messages, second_response, 135, 15)

    # 135 - (100 + 20) == 15 tokens of genuinely new prompt content.
    assert new_prompt_tokens == 15


def test_shrunk_prompt_after_compaction_floors_at_zero_not_negative():
    tracker = LlmUsageTracker()
    first_messages = [_msg("user", "a very long conversation") for _ in range(50)]
    first_response = _msg("assistant", "ok")
    tracker.resolve_new_prompt_tokens(first_messages, first_response, 5000, 20)

    # Something outside our visibility (harness-internal compaction)
    # replaced the conversation with a much shorter summary before the next
    # call -- the new messages array does not extend the previous state at
    # all, so no branch matches and this is treated as a fresh baseline.
    compacted_messages = [_msg("user", "[summary] ..."), _msg("user", "continue")]
    compacted_response = _msg("assistant", "continuing")

    new_prompt_tokens = tracker.resolve_new_prompt_tokens(
        compacted_messages, compacted_response, 300, 10
    )

    assert new_prompt_tokens == 300  # no match found -> full value, never negative


def test_unrelated_branch_does_not_match_and_is_tracked_separately():
    tracker = LlmUsageTracker()
    branch_a_messages = [_msg("user", "branch A task")]
    branch_a_response = _msg("assistant", "working on A")
    tracker.resolve_new_prompt_tokens(branch_a_messages, branch_a_response, 200, 30)

    # A completely different, concurrent conversation -- shares no prefix
    # with branch A, so it must not be diffed against it.
    branch_b_messages = [_msg("user", "branch B task")]
    branch_b_response = _msg("assistant", "working on B")
    new_prompt_tokens_b = tracker.resolve_new_prompt_tokens(
        branch_b_messages, branch_b_response, 150, 25
    )
    assert new_prompt_tokens_b == 150

    # Branch A can still be continued correctly afterward -- its tip wasn't
    # clobbered by branch B's unrelated exchange.
    branch_a_continued = branch_a_messages + [branch_a_response, _msg("user", "more A")]
    new_prompt_tokens_a = tracker.resolve_new_prompt_tokens(
        branch_a_continued, _msg("assistant", "still A"), 260, 10
    )
    assert new_prompt_tokens_a == 260 - (200 + 30)


def test_registry_stays_bounded_under_many_unrelated_branches():
    tracker = LlmUsageTracker()
    for i in range(200):
        tracker.resolve_new_prompt_tokens(
            [_msg("user", f"task {i}")], _msg("assistant", "ok"), 10, 1
        )
    assert len(tracker._tips) <= 32
