"""Tests for the Codex CLI adapter's availability and output parser."""

from __future__ import annotations

import json

from agency.harness.adapters.codex import _CodexBackend, codex_available


def test_codex_available_reflects_real_which():
    import shutil

    assert codex_available() == (shutil.which("codex") is not None)


def test_parse_output_events_ignores_non_agent_message_items():
    stdout = "\n".join(
        [
            json.dumps(
                {"type": "item.completed", "item": {"type": "reasoning", "text": "thinking"}}
            ),
            json.dumps(
                {"type": "item.completed", "item": {"type": "agent_message", "text": "final"}}
            ),
        ]
    )
    assert _CodexBackend._parse_output_events(stdout) == "final"


def test_parse_output_events_falls_back_to_raw_text():
    assert _CodexBackend._parse_output_events("plain text") == "plain text"
