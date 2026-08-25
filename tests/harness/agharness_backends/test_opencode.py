"""Tests for the OpenCode CLI adapter's availability and output parser."""

from __future__ import annotations

import json

from agency.harness.adapters.opencode import _OpencodeBackend, opencode_available


def test_opencode_available_reflects_real_which():
    import shutil

    assert opencode_available() == (shutil.which("opencode") is not None)


def test_parse_output_events_extracts_last_text_from_ndjson():
    stdout = "\n".join(
        [
            json.dumps({"type": "turn.started"}),
            json.dumps({"type": "item.completed", "text": "partial"}),
            json.dumps({"type": "item.completed", "text": "final answer"}),
        ]
    )
    assert _OpencodeBackend._parse_output_events(stdout) == "final answer"


def test_parse_output_events_falls_back_to_raw_text():
    assert _OpencodeBackend._parse_output_events("just plain text, no JSON") == (
        "just plain text, no JSON"
    )
