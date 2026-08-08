"""Tests for agllm_pure.py -- the dependency-free compaction algorithms
shared by agllm.py (host-side, execute_react()'s compaction) and the
in-container native entrypoint's own compaction.

agllm.py's own compaction tests (tests/test_agllm.py) already exercise
this logic thoroughly through agllm's delegating static methods -- these
tests exist to (a) prove the module works when imported directly (not just
through agllm.py's thin wrappers), and (b) prove it's loadable by raw file
path, the same way the in-container entrypoint actually loads it (mirrors
test_agtool_pure.py's identical check for the same reason).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from agency import agllm_pure


def test_estimate_tokens_content():
    msg = {"role": "assistant", "content": "a" * 400}
    assert agllm_pure.estimate_tokens(msg) == 100


def test_estimate_tokens_minimum_one():
    assert agllm_pure.estimate_tokens({"role": "user", "content": ""}) == 1


def test_should_compact_threshold():
    assert agllm_pure.should_compact(900, 1000) is True
    assert agllm_pure.should_compact(100, 1000) is False


def test_tail_start_empty():
    assert agllm_pure.tail_start([], 100_000) == 0


def test_prune_tool_outputs_leaves_small_results_intact():
    msgs = [{"role": "tool", "content": "small", "tool_call_id": "t1"}]
    assert agllm_pure.prune_tool_outputs(msgs) == msgs


def test_split_for_compaction_separates_system_task_input_head_tail():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old turn 1"},
        {"role": "user", "content": "old turn 1 followup"},
        {"role": "assistant", "content": "recent turn"},
    ]
    sys_msg, task_input, head, tail = agllm_pure.split_for_compaction(
        messages, context_limit=100_000, tail_turns=1
    )
    assert sys_msg == [messages[0]]
    assert task_input == [messages[1]]
    assert tail and tail[0]["content"] == "recent turn"


def test_build_summary_prompt_messages_shape():
    task_input = [{"role": "user", "content": "the task"}]
    head = [{"role": "assistant", "content": "did something"}]
    messages = agllm_pure.build_summary_prompt_messages(task_input, head)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == agllm_pure.SUMMARY_SYSTEM
    assert "the task" in messages[1]["content"]
    assert "did something" in messages[1]["content"]


def test_assemble_compacted_messages_injects_summary_and_keeps_tail():
    sys_msg = [{"role": "system", "content": "sys"}]
    task_input = [{"role": "user", "content": "task"}]
    tail = [{"role": "assistant", "content": "recent"}]
    result = agllm_pure.assemble_compacted_messages(sys_msg, task_input, "the summary", tail)
    assert result[0] == sys_msg[0]
    assert result[1] == task_input[0]
    assert "the summary" in result[2]["content"]
    assert result[-1] == tail[0]


def test_loadable_by_raw_file_path_like_the_entrypoint_does():
    """The in-container native entrypoint can't `from agency.agllm_pure
    import ...` (that would execute agency/__init__.py first, pulling in
    the same heavy host-venv-only dependency chain this module exists to
    avoid) -- it loads this exact file by path instead, same as
    agtool_pure.py. Prove that path actually works."""
    module_path = Path(agllm_pure.__file__)
    spec = importlib.util.spec_from_file_location("agllm_pure_standalone", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.estimate_tokens({"role": "user", "content": "abcd"}) == 1
    assert module.should_compact(900, 1000) is True

    sys.modules.pop("agllm_pure_standalone", None)
