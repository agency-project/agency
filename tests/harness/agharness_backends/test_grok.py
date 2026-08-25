"""Tests for the Grok CLI adapter's availability and result parser."""

from __future__ import annotations

import json

from agency.harness.adapters.grok import _GrokBackend, grok_available


def test_grok_available_reflects_real_which():
    import shutil

    assert grok_available() == (shutil.which("grok") is not None)


def test_parse_result_json_extracts_text_usage_and_session():
    payload = json.dumps(
        {"text": "abc", "usage": {"input_tokens": 1, "output_tokens": 2}, "sessionId": "s1"}
    )
    text, usage, session_id = _GrokBackend._parse_result_json(payload)
    assert text == "abc"
    assert usage == {"input_tokens": 1, "output_tokens": 2}
    assert session_id == "s1"


def test_parse_result_json_falls_back_on_malformed_json():
    text, usage, session_id = _GrokBackend._parse_result_json("not json")
    assert text == "not json"
    assert usage == {}
    assert session_id is None
