"""Tests for agpath — path-only string agskill schema field.

Unlike agfile/agbinary, agpath never reads or writes the sandbox filesystem;
it only checks that a value looks like a path, on both the input and output
side. It exists because a plain `str` output field silently auto-resolves a
path-looking value to that file's contents (see agschema.make_field_handler),
which is wrong for a field whose value is meant to *stay* a path.
"""

import json
from agency.agdata import agdata
from agency.agschema import agschema
from agency.agtype import agtype, agpath
from agency.agskill import agskill


# ---------------------------------------------------------------------------
# agpath class — basic classmethods
# ---------------------------------------------------------------------------


def test_agpath_is_agtype_subclass():
    assert issubclass(agpath, agtype)


def test_agpath_schema_type():
    assert agpath.schema_type() == "path"


def test_agpath_needs_sandbox():
    assert agpath.needs_sandbox() is False


def test_agpath_prepare_passthrough():
    val, paths = agpath.prepare("/workspace/out.txt", None, "skill", "field")
    assert val == "/workspace/out.txt"
    assert paths == []


def test_agpath_recover_passthrough():
    val, paths = agpath.recover("/workspace/out.txt", None)
    assert val == "/workspace/out.txt"
    assert paths == []


def test_agpath_extra_input_prompt_mentions_path():
    assert "path" in agpath.extra_input_prompt("dest").lower()


def test_agpath_extra_output_prompt_warns_against_content():
    prompt = agpath.extra_output_prompt("dest", "skill")
    assert "path" in prompt.lower()
    assert "content" in prompt.lower()


def test_agpath_get_return_tool_value_description_warns_against_content():
    desc = agpath.get_return_tool_value_description("dest")
    assert "content" in desc.lower()


def test_agpath_validate_input_value_accepts_path():
    assert agpath.validate_input_value("/workspace/out.txt") is None


def test_agpath_validate_input_value_rejects_non_path_string():
    err = agpath.validate_input_value("this is not a path")
    assert err is not None
    assert "does not look like a path" in err


def test_agpath_validate_input_value_rejects_non_string():
    err = agpath.validate_input_value(123)
    assert err is not None
    assert "must be a string" in err


def test_agpath_validate_output_accepts_path():
    assert agpath.validate_output("dest", "/data/note.txt", None, 5) is None


def test_agpath_validate_output_rejects_non_path_value():
    err = agpath.validate_output("dest", "The quick brown fox jumps.", None, 5)
    assert err is not None
    assert "does not look like a path" in err


def test_agpath_validate_output_rejects_non_string_value():
    assert agpath.validate_output("dest", 123, None, 5) is not None


# ---------------------------------------------------------------------------
# agdata serialization
# ---------------------------------------------------------------------------


def test_agdata_serializes_agpath_as_path():
    d = agdata(dest=agpath)
    assert json.loads(d.to_json()) == {"dest": "path"}


# ---------------------------------------------------------------------------
# agschema.check — schema validation
# ---------------------------------------------------------------------------


def test_check_schema_agpath_hint_accepts_path_string():
    s = agschema(agdata(dest=agpath))
    assert s.check(agdata(dest="/workspace/out.txt")) == []


def test_check_schema_agpath_hint_rejects_non_path_string():
    s = agschema(agdata(dest=agpath))
    errors = s.check(agdata(dest="not a path"))
    assert errors
    assert "does not look like a path" in errors[0]


def test_check_schema_agpath_hint_rejects_non_string():
    s = agschema(agdata(dest=agpath))
    errors = s.check(agdata(dest=123))
    assert errors


# ---------------------------------------------------------------------------
# agskill._build_system_prompt — agpath prompts injected
# ---------------------------------------------------------------------------


def test_system_prompt_includes_agpath_input_instructions():
    sk = agskill(
        "move",
        "Do stuff.",
        input_schema=agdata(dest=agpath),
        output_schema=agdata(result=str),
    )
    prompt = sk._build_system_prompt()
    assert "dest" in prompt
    assert "path" in prompt.lower()


def test_system_prompt_includes_agpath_output_instructions():
    sk = agskill(
        "move",
        "Do stuff.",
        input_schema=agdata(theme=str),
        output_schema=agdata(moved_to=agpath),
    )
    prompt = sk._build_system_prompt()
    assert "moved_to" in prompt
    assert "content" in prompt.lower()


def test_system_prompt_agpath_type_shown_as_path():
    sk = agskill("t", "", input_schema=agdata(dest=agpath))
    prompt = sk._build_system_prompt()
    assert '"dest": "path"' in prompt


# test_return_agpath_valid_path_is_accepted /
# test_return_agpath_non_path_value_returns_error_and_reprompts /
# test_return_agpath_non_path_value_exhausts_retries /
# test_return_agpath_does_not_auto_resolve_to_file_contents /
# test_return_str_path_field_still_auto_resolves_and_warns (and their
# shared _run_skill_with_sandbox() helper) were retired here: they tested
# agschema.py's make_field_handler()'s agtype-specific validation chain (as
# invoked by the retired per-field `return_<field>` tool handler), only
# ever reachable via execute_react(). Native's `submit_output` MCP tool
# does NOT run this same validation chain today -- a real, documented gap
# for agtype OUTPUT fields specifically, noted in agmcp_server.py's own
# module docstring (found while retiring test_agfile.py's equivalent tests)
# rather than silently dropped.


def test_input_agpath_rejects_non_path_value_before_llm_call():
    # input_schema validation is shared, engine-agnostic code -- testing it
    # directly here needs no LLM/loop at all (see test_agskill.py's
    # equivalent conversion for input_schema tests).
    sk = agskill("move", "", input_schema=agdata(dest=agpath), output_schema=agdata(result=str))
    error = sk.input_schema.validate_input(agdata(dest="not a path"))
    assert error is not None
    assert "does not look like a path" in error
