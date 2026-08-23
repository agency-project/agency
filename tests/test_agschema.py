"""Tests for agschema — schema wrapper for agskill input/output schemas."""

import json
import pytest
from unittest.mock import MagicMock

from agency.agdata import agdata, agerror
from agency.agschema import agschema
from agency.agtype import agfile, agbinary, agrawstring, agpath


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_from_agdata():
    s = agschema(agdata(task=str, count=int))
    assert s._data == {"task": str, "count": int}


def test_construction_from_agschema_copies():
    s1 = agschema(agdata(x=str))
    s2 = agschema(s1)
    assert s2._data == s1._data


def test_construction_rejects_non_agdata():
    with pytest.raises(TypeError):
        agschema({"task": str})


def test_construction_rejects_plain_string():
    with pytest.raises(TypeError):
        agschema("task=str")


def test_repr():
    s = agschema(agdata(x=str))
    assert "agschema" in repr(s)


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------


def test_to_json_round_trips():
    s = agschema(agdata(question=str, count=int))
    parsed = json.loads(s.to_json())
    assert "question" in parsed
    assert "count" in parsed


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def test_check_valid_data_returns_empty():
    s = agschema(agdata(x=int, name=str))
    assert s.check(agdata(x=5, name="hi")) == []


def test_check_missing_field():
    s = agschema(agdata(x=int, name=str))
    errors = s.check(agdata(x=5))
    assert any("name" in e for e in errors)


def test_check_wrong_type():
    s = agschema(agdata(x=int))
    errors = s.check(agdata(x="not_an_int"))
    assert errors


def test_check_agtype_field_accepts_string():
    s = agschema(agdata(doc=agfile))
    assert s.check(agdata(doc="/workspace/out.txt")) == []


def test_check_agtype_field_rejects_non_string():
    s = agschema(agdata(doc=agfile))
    errors = s.check(agdata(doc=123))
    assert errors


def test_check_agpath_field_accepts_path():
    s = agschema(agdata(dest=agpath))
    assert s.check(agdata(dest="/workspace/out.txt")) == []


def test_check_agpath_field_rejects_non_path_string():
    s = agschema(agdata(dest=agpath))
    errors = s.check(agdata(dest="not a path at all"))
    assert errors
    assert "does not look like a path" in errors[0]


def test_check_agpath_field_rejects_non_string():
    s = agschema(agdata(dest=agpath))
    errors = s.check(agdata(dest=123))
    assert errors


def test_check_list_of_dicts_schema_valid():
    s = agschema(agdata(items=[{"name": str, "count": int}]))
    assert s.check(agdata(items=[{"name": "a", "count": 1}])) == []


def test_check_list_of_dicts_schema_missing_key():
    s = agschema(agdata(items=[{"name": str, "count": int}]))
    errors = s.check(agdata(items=[{"name": "a"}]))
    assert any("count" in e for e in errors)


def test_check_list_of_dicts_schema_wrong_item_type():
    s = agschema(agdata(items=[{"name": str}]))
    errors = s.check(agdata(items=[{"name": 123}]))
    assert errors


# ---------------------------------------------------------------------------
# check_field
# ---------------------------------------------------------------------------


def test_check_field_valid_returns_none():
    s = agschema(agdata(x=int))
    assert s.check_field("x", 5) is None


def test_check_field_invalid_returns_string():
    s = agschema(agdata(x=int))
    assert s.check_field("x", "oops") is not None


def test_check_field_agtype_accepts_str():
    s = agschema(agdata(doc=agfile))
    assert s.check_field("doc", "/path/to/file") is None


def test_check_field_agpath_accepts_path():
    s = agschema(agdata(dest=agpath))
    assert s.check_field("dest", "/path/to/file") is None


def test_check_field_agpath_rejects_non_path():
    s = agschema(agdata(dest=agpath))
    assert s.check_field("dest", "not a path") is not None


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------


def test_validate_input_valid_returns_none():
    s = agschema(agdata(question=str))
    assert s.validate_input(agdata(question="hi")) is None


def test_validate_input_missing_field_returns_error():
    s = agschema(agdata(question=str))
    err = s.validate_input(agdata())
    assert err is not None
    assert "question" in err


def test_validate_input_agpath_valid_returns_none():
    s = agschema(agdata(dest=agpath))
    assert s.validate_input(agdata(dest="/data/out.txt")) is None


def test_validate_input_agpath_invalid_returns_error():
    s = agschema(agdata(dest=agpath))
    err = s.validate_input(agdata(dest="not a path"))
    assert err is not None
    assert "does not look like a path" in err


# ---------------------------------------------------------------------------
# raw_key
# ---------------------------------------------------------------------------


def test_raw_key_single_agrawstring_returns_key():
    s = agschema(agdata(content=agrawstring))
    assert s.raw_key() == "content"


def test_raw_key_no_agrawstring_returns_none():
    s = agschema(agdata(content=str))
    assert s.raw_key() is None


def test_raw_key_multiple_fields_returns_none():
    s = agschema(agdata(a=agrawstring, b=str))
    assert s.raw_key() is None


def test_raw_key_none_on_empty():
    s = agschema(agdata())
    assert s.raw_key() is None


# ---------------------------------------------------------------------------
# field_desc
# ---------------------------------------------------------------------------


def test_field_desc_str():
    s = agschema(agdata(answer=str))
    desc = s.field_desc("answer")
    assert "string" in desc.lower()


def test_field_desc_agfile():
    s = agschema(agdata(doc=agfile))
    desc = s.field_desc("doc")
    assert "file" in desc.lower()


def test_field_desc_agpath():
    s = agschema(agdata(dest=agpath))
    desc = s.field_desc("dest")
    assert "path" in desc.lower()


def test_field_desc_int():
    s = agschema(agdata(count=int))
    desc = s.field_desc("count")
    assert "int" in desc.lower()


# ---------------------------------------------------------------------------
# get_return_tool_descriptions
# ---------------------------------------------------------------------------


def test_get_return_tool_description_prompt_returns_two_strings():
    s = agschema(agdata(answer=str))
    tool_desc, val_desc = s.get_return_tool_descriptions("answer")
    assert isinstance(tool_desc, str) and len(tool_desc) > 0
    assert isinstance(val_desc, str) and len(val_desc) > 0


def test_get_return_tool_description_prompt_agfile_mentions_path():
    s = agschema(agdata(doc=agfile))
    tool_desc, val_desc = s.get_return_tool_descriptions("doc")
    assert "path" in val_desc.lower()


def test_get_return_tool_description_prompt_agbinary_mentions_path():
    s = agschema(agdata(audio=agbinary))
    tool_desc, val_desc = s.get_return_tool_descriptions("audio")
    assert "path" in val_desc.lower()


def test_get_return_tool_description_prompt_agpath_warns_against_content():
    s = agschema(agdata(dest=agpath))
    tool_desc, val_desc = s.get_return_tool_descriptions("dest")
    assert "path" in val_desc.lower()
    assert "content" in val_desc.lower()


# make_return_output_tools/make_field_handler tests were retired here along
# with execute_react() itself: both only ever supported the per-field
# `return_<field>` tool mechanism (agschema.make_return_output_agtool),
# whose only caller was agskill.py's (now-deleted) _build_toolkit(). Native's
# structured output uses a different mechanism (`submit_output`, see
# agmcp_server.py) -- fast coverage for that lives in
# tests/harness/agharness_backends/test_native_loop_fast.py.


# ---------------------------------------------------------------------------
# prepare_inputs_in_sandbox — size offload
# ---------------------------------------------------------------------------


def test_prepare_inputs_in_sandbox_replaces_long_string():
    from agency.agschema import _AgSchemaFields

    s = agschema(agdata(text=str))
    sb = MagicMock()
    data = agdata(text="x" * (_AgSchemaFields.input_offload_chars.default + 1))
    paths, fields = s.prepare_inputs_in_sandbox(data, sb, "skill")
    assert fields == ["text"]
    assert len(paths) == 1


def test_prepare_inputs_in_sandbox_skips_short_strings():
    s = agschema(agdata(text=str))
    sb = MagicMock()
    data = agdata(text="short")
    paths, fields = s.prepare_inputs_in_sandbox(data, sb, "skill")
    assert paths == [] and fields == []


# ---------------------------------------------------------------------------
# prepare_inputs_in_sandbox — agtype fields
# ---------------------------------------------------------------------------


def test_prepare_inputs_in_sandbox_calls_prepare_on_agfile_field():
    s = agschema(agdata(doc=agfile))
    sb = MagicMock()
    sb.write_file = MagicMock()
    data = agdata(doc="the file contents")
    paths, _ = s.prepare_inputs_in_sandbox(data, sb, "skill")
    sb.write_file.assert_called_once()
    assert paths


def test_prepare_inputs_in_sandbox_no_agtype_fields_returns_empty():
    s = agschema(agdata(text=str))
    sb = MagicMock()
    data = agdata(text="hello")
    paths, _ = s.prepare_inputs_in_sandbox(data, sb, "skill")
    assert paths == []


def test_recover_outputs_reads_agfile_path():
    s = agschema(agdata(doc=agfile))
    sb = MagicMock()
    sb.read_file.return_value = "the recovered content"
    data = agdata(doc="/workspace/out.txt")
    paths = s.recover_outputs(data, sb)
    sb.read_file.assert_called_once_with("/workspace/out.txt")
    assert data.doc == "the recovered content"
    assert paths == ["/workspace/out.txt"]


def test_recover_outputs_skips_agerror():
    s = agschema(agdata(doc=agfile))
    sb = MagicMock()
    result = agerror("something failed")
    paths = s.recover_outputs(result, sb)
    sb.read_file.assert_not_called()
    assert paths == []


def test_recover_outputs_no_agtype_fields_returns_empty():
    s = agschema(agdata(answer=str))
    sb = MagicMock()
    data = agdata(answer="done")
    paths = s.recover_outputs(data, sb)
    assert paths == []


# ---------------------------------------------------------------------------
# validate_and_recover() -- whole-schema validation + recovery from a
# harness's single raw final-answer text (agskill.execute_harness()'s path)
# ---------------------------------------------------------------------------


def test_validate_and_recover_valid_json_round_trips():
    s = agschema(agdata(answer=str))
    sb = MagicMock()
    data, paths = s.validate_and_recover('{"answer": "done"}', sb)
    assert not isinstance(data, agerror)
    assert data.answer == "done"
    assert paths == []


def test_validate_and_recover_malformed_json_returns_agerror():
    s = agschema(agdata(answer=str))
    sb = MagicMock()
    data, paths = s.validate_and_recover("not json at all {", sb)
    assert isinstance(data, agerror)
    assert "could not parse harness output as JSON" in data.error
    assert paths == []


def test_validate_and_recover_non_object_json_returns_agerror():
    s = agschema(agdata(answer=str))
    sb = MagicMock()
    data, paths = s.validate_and_recover("[1, 2, 3]", sb)
    assert isinstance(data, agerror)
    assert "must be a JSON object" in data.error
    assert paths == []


def test_validate_and_recover_missing_field_returns_agerror():
    s = agschema(agdata(answer=str, confidence=float))
    sb = MagicMock()
    data, paths = s.validate_and_recover('{"answer": "done"}', sb)
    assert isinstance(data, agerror)
    assert "output schema error" in data.error
    assert "confidence" in data.error
    assert paths == []


def test_validate_and_recover_wrong_type_returns_agerror():
    s = agschema(agdata(answer=str))
    sb = MagicMock()
    data, paths = s.validate_and_recover('{"answer": 42}', sb)
    assert isinstance(data, agerror)
    assert "output schema error" in data.error


def test_validate_and_recover_recovers_agtype_field():
    s = agschema(agdata(doc=agfile))
    sb = MagicMock()
    sb.read_file.return_value = "the recovered content"
    data, paths = s.validate_and_recover('{"doc": "/workspace/out.txt"}', sb)
    assert not isinstance(data, agerror)
    sb.read_file.assert_called_once_with("/workspace/out.txt")
    assert data.doc == "the recovered content"
    assert paths == ["/workspace/out.txt"]
