"""Regression coverage for Claude's pasted-tab normalization and turn fence."""

from types import SimpleNamespace
import json
import shlex
import sys

import pytest

from agency.configs.agconfig import agconfig
from agency.harness.adapters.base import AdapterRuntime
from agency.harness.adapters.claude_code import ClaudeCodeAdapter, ClaudeDriver
from agency.harness.adapters.pty.execution import PtyExecution


@pytest.mark.parametrize(
    "prompt",
    [
        "[Agency run]\nclass Example:\n\tdef method(self):\n\t\treturn 1",
        "[Agency run]\ninline\ttab and café\ttext",
        "[Agency redirect]\n\tfirst\n\t\tsecond\t",
    ],
)
def test_claude_acknowledges_pasted_tabs_after_composer_expands_them(prompt):
    driver = object.__new__(ClaudeDriver)
    assert driver.prompt_matches(prompt.replace("\t", "    "), prompt)
    assert driver.prompt_matches(prompt, prompt)


@pytest.mark.parametrize(
    "received",
    [
        None,
        "[Agency redirect]\n    current",
        "[Agency run]\n  current",
        "[Agency run]\n    different",
    ],
)
def test_claude_tab_normalization_does_not_accept_missing_or_different_prompt(received):
    driver = object.__new__(ClaudeDriver)
    assert not driver.prompt_matches(received, "[Agency run]\n\tcurrent")


def test_claude_acknowledgment_requires_current_turn_even_when_text_matches():
    driver = object.__new__(ClaudeDriver)
    driver._expected_turn_id = "current-turn"
    driver._expected_prompt = "[Agency run]\n\tcurrent"
    driver.root = None
    execution = PtyExecution(driver, SimpleNamespace())
    execution._turn_id = "current-turn"
    execution._expected_prompt = driver._expected_prompt

    def submit(turn, prompt):
        event = driver._event_from_payload(
            {"turn_id": turn, "payload": {"hook_event_name": "UserPromptSubmit", "prompt": prompt}}
        )
        driver.events = lambda: [event] if event is not None else []
        execution._poll()

    submit("old-turn", "[Agency run]\n    current")
    assert not execution._acknowledged
    assert execution._turn_id == "current-turn"
    submit("current-turn", "[Agency run]\n    wrong")
    assert not execution._acknowledged
    submit("current-turn", "[Agency run]\n    current")
    assert execution._acknowledged
    assert execution._turn_id == "current-turn"


def test_claude_hooks_use_the_harness_interpreter_without_replacing_task_python(
    tmp_path, monkeypatch
):
    config = agconfig()
    runtime = AdapterRuntime(config, "model", "agent", "http://daemon", "key", SimpleNamespace())
    interpreter = "/opt/harness runtime/bin/python"
    with monkeypatch.context() as patch:
        patch.setattr(sys, "executable", interpreter)
        argv, _ = ClaudeCodeAdapter(config).prepare_pty(runtime, tmp_path)
    hooks = json.loads(argv[argv.index("--settings") + 1])["hooks"]
    for entries in hooks.values():
        for entry in entries:
            assert shlex.split(entry["hooks"][0]["command"])[0] == interpreter


def test_claude_completed_turn_survives_damaged_usage_rows_without_changing_native_snapshot():
    driver = object.__new__(ClaudeDriver)
    driver._attempt_offset = 0
    native = (
        b'{"message":{"usage":{"input_tokens":2,"output_tokens":3}}}\n'
        b'{"message":{"usage": partially written record}\n'
        b"\xff\n"
        b'{"message":null}\n'
        b'{"message":{"usage":[]}}\n'
        b'{"message":{"usage":{"input_tokens":5,"output_tokens":7}}}\n'
        b'{"unfinished":'
    )
    event = {"text": "Task finished."}
    driver._transcript = lambda: native
    assert driver.completed(event)
    assert event["input_tokens"] == 7
    assert event["output_tokens"] == 10
    assert driver.snapshot() == native
