"""Tests for agrawstring — raw string bypass mode."""

import json

from agency.agdata import agdata
from agency.agtype import agtype, agrawstring
from agency.agskill import agskill


# ---------------------------------------------------------------------------
# agrawstring class
# ---------------------------------------------------------------------------


def test_agrawstring_is_agtype_subclass():
    assert issubclass(agrawstring, agtype)


def test_agrawstring_schema_type():
    assert agrawstring.schema_type() == "str"


def test_agrawstring_needs_no_sandbox():
    assert agrawstring.needs_sandbox() is False


def test_agrawstring_prepare_passthrough():
    val, paths = agrawstring.prepare("hello world", None, "sk", "content")
    assert val == "hello world"
    assert paths == []


def test_agrawstring_recover_passthrough():
    val, paths = agrawstring.recover("some output", None)
    assert val == "some output"
    assert paths == []


# ---------------------------------------------------------------------------
# _build_system_prompt — JSON blocks omitted for agrawstring
# ---------------------------------------------------------------------------


def test_system_prompt_no_input_json_when_raw_input():
    sk = agskill("t", "Write a story.", input_schema=agdata(prompt=agrawstring))
    prompt = sk._build_system_prompt()
    assert "Input JSON format" not in prompt


def test_system_prompt_no_output_json_when_raw_output():
    sk = agskill("t", "Write a story.", output_schema=agdata(story=agrawstring))
    prompt = sk._build_system_prompt()
    assert "Output JSON format" not in prompt
    assert "plain text" in prompt.lower()


def test_system_prompt_keeps_json_for_normal_output():
    sk = agskill("t", "Summarise.", output_schema=agdata(summary=str))
    prompt = sk._build_system_prompt()
    assert "return_summary" in prompt


def test_system_prompt_keeps_input_json_for_normal_input():
    sk = agskill("t", "Summarise.", input_schema=agdata(text=str))
    prompt = sk._build_system_prompt()
    assert "Input JSON format" in prompt


# ---------------------------------------------------------------------------
# build_prompt_payload — raw passthrough for agrawstring input
# ---------------------------------------------------------------------------


def test_build_prompt_payload_raw_input_returns_plain_string():
    sk = agskill("t", "", input_schema=agdata(prompt=agrawstring))
    inp = agdata(prompt="Tell me a story about a robot.")
    content = sk.build_prompt_payload(inp)
    assert content == "Tell me a story about a robot."


def test_build_prompt_payload_raw_input_no_json_wrapping():
    sk = agskill("t", "", input_schema=agdata(prompt=agrawstring))
    inp = agdata(prompt="Hello!")
    content = sk.build_prompt_payload(inp)
    assert not content.startswith("{")


def test_build_prompt_payload_normal_input_still_json():
    sk = agskill("t", "", input_schema=agdata(text=str))
    inp = agdata(text="hello")
    content = sk.build_prompt_payload(inp)
    assert isinstance(content, str)
    parsed = json.loads(content.split("\n", 1)[1])
    assert parsed["text"] == "hello"


# ---------------------------------------------------------------------------
# agskill.run — raw output captures full response, no JSON parsing
# ---------------------------------------------------------------------------


# test_raw_output_captures_full_response / test_raw_output_does_not_parse_as_json /
# test_raw_output_with_raw_input / test_raw_output_preserves_newlines_and_quotes
# (and their _make_chunk()/_make_mock_agent()/_run_skill_with_mock_response()
# helpers) were retired here: they exercised execute_react()'s raw-text
# output path via the full loop. Native's `_NativeBackend.execute()` shares
# the identical raw_key()-passthrough logic (`agdata(**{out_key: final_text})`)
# one level above the entrypoint's own react loop -- fast, no-Docker
# coverage of the entrypoint's own final_text plumbing (which this logic
# wraps) lives in tests/harness/agharness_backends/
# test_native_loop_fast.py; the agdata-wrapping step itself is Docker-only
# coverage today (test_native.py's TestNativeBackendRealEndToEnd), same
# tier gap noted for the return_output-family tests elsewhere in this
# retirement pass.
