"""Tests for the agtype base class, agdata serialization, and input offloading."""
import json
import pytest
from unittest.mock import MagicMock
from agency.agdata import agdata
from agency.agtype import agtype, agfile, agimage
from agency.agskill import agskill


# ---------------------------------------------------------------------------
# agtype base class
# ---------------------------------------------------------------------------

def test_agtype_schema_type_default():
    assert agtype.schema_type() == "str"

def test_agtype_needs_sandbox_default():
    assert agtype.needs_sandbox() is False

def test_agtype_prepare_passthrough():
    val, paths = agtype.prepare("hello", None, "skill", "field")
    assert val == "hello"
    assert paths == []

def test_agtype_recover_passthrough():
    val, paths = agtype.recover("hello", None)
    assert val == "hello"
    assert paths == []

def test_agtype_extra_input_prompt_empty():
    assert agtype.extra_input_prompt("x") == ""

def test_agtype_extra_output_prompt_empty():
    assert agtype.extra_output_prompt("x", "skill") == ""


# ---------------------------------------------------------------------------
# agdata serialization with type objects
# ---------------------------------------------------------------------------

def test_agdata_serializes_python_types():
    d = agdata(x=str, n=int, f=float, b=bool, items=list, mapping=dict)
    parsed = json.loads(d.to_json())
    assert parsed == {
        "x": "str", "n": "int", "f": "float",
        "b": "bool", "items": "list", "mapping": "dict",
    }

def test_agdata_serializes_custom_agtype():
    class agblob(agtype):
        @classmethod
        def schema_type(cls): return "blob"
    d = agdata(data=agblob)
    assert json.loads(d.to_json()) == {"data": "blob"}


# ---------------------------------------------------------------------------
# agskill._check_schema — Python type object hints
# ---------------------------------------------------------------------------

def test_check_schema_accepts_python_type_objects():
    sk = agskill("t", "", input_schema=agdata(x=int, name=str))
    assert sk._check_schema(agdata(x=5, name="hi"), sk.input_schema) == []

def test_check_schema_type_mismatch_with_type_object():
    sk = agskill("t", "", input_schema=agdata(x=int))
    errors = sk._check_schema(agdata(x="bad"), sk.input_schema)
    assert len(errors) == 1
    assert "x" in errors[0]
    assert "int" in errors[0]

def test_system_prompt_type_names_shown_correctly():
    sk = agskill(
        "t", "",
        input_schema=agdata(n=int, s=str, doc=agfile),
        output_schema=agdata(result=float),
    )
    prompt = sk._build_system_prompt()
    assert '"n": "int"' in prompt       # input schema still uses to_json()
    assert '"s": "str"' in prompt
    assert '"doc": "file"' in prompt
    assert "result" in prompt            # output field listed by name
    assert "float" in prompt             # output field type shown as "float"


# ---------------------------------------------------------------------------
# Input offloading (_offload_large_fields)
# ---------------------------------------------------------------------------

def test_offload_large_fields_replaces_long_string():
    from agency.agent import _offload_large_fields, INPUT_OFFLOAD_CHARS
    sandbox = MagicMock()
    long_val = "x" * (INPUT_OFFLOAD_CHARS + 1)
    inp = agdata(text=long_val, small="hi")
    paths, fields = _offload_large_fields(inp, sandbox, "mskill")
    sandbox.write_file.assert_called_once()
    assert "text" in fields
    assert "small" not in fields
    assert len(paths) == 1
    assert "mskill_text" in paths[0]
    assert "content saved to" in inp._data["text"]
    assert inp._data["small"] == "hi"

def test_offload_large_fields_skips_short_strings():
    from agency.agent import _offload_large_fields
    sandbox = MagicMock()
    inp = agdata(text="short")
    paths, fields = _offload_large_fields(inp, sandbox, "skill")
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert fields == []

def test_offload_large_fields_skips_non_string_scalars():
    from agency.agent import _offload_large_fields
    sandbox = MagicMock()
    inp = agdata(n=42)
    paths, fields = _offload_large_fields(inp, sandbox, "skill")
    sandbox.write_file.assert_not_called()
    assert paths == []

def test_offload_large_fields_sandbox_failure_leaves_field_unchanged():
    from agency.agent import _offload_large_fields, INPUT_OFFLOAD_CHARS
    sandbox = MagicMock()
    sandbox.write_file.side_effect = OSError("no space")
    long_val = "y" * (INPUT_OFFLOAD_CHARS + 1)
    inp = agdata(text=long_val)
    paths, fields = _offload_large_fields(inp, sandbox, "skill")
    assert paths == []
    assert fields == []
    assert inp._data["text"] == long_val

def test_offload_large_fields_list_large_strings_replaced_with_paths():
    from agency.agent import _offload_large_fields, INPUT_OFFLOAD_CHARS
    sandbox = MagicMock()
    long_a = "a" * (INPUT_OFFLOAD_CHARS + 1)
    long_b = "b" * (INPUT_OFFLOAD_CHARS + 1)
    inp = agdata(items=[long_a, long_b])
    paths, fields = _offload_large_fields(inp, sandbox, "sk")
    assert sandbox.write_file.call_count == 2
    assert "items" in fields
    assert len(paths) == 2
    assert "sk_items_0" in paths[0]
    assert "sk_items_1" in paths[1]
    result = inp._data["items"]
    assert result[0] == paths[0]
    assert result[1] == paths[1]

def test_offload_large_fields_list_short_strings_unchanged():
    from agency.agent import _offload_large_fields
    sandbox = MagicMock()
    inp = agdata(items=["short", "also short"])
    paths, fields = _offload_large_fields(inp, sandbox, "sk")
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert fields == []
    assert inp._data["items"] == ["short", "also short"]

def test_offload_large_fields_list_mixed_only_large_replaced():
    from agency.agent import _offload_large_fields, INPUT_OFFLOAD_CHARS
    sandbox = MagicMock()
    long_val = "x" * (INPUT_OFFLOAD_CHARS + 1)
    inp = agdata(items=["short", long_val])
    paths, fields = _offload_large_fields(inp, sandbox, "sk")
    sandbox.write_file.assert_called_once()
    result = inp._data["items"]
    assert result[0] == "short"
    assert result[1] == paths[0]

def test_offload_large_fields_list_non_string_elements_skipped():
    from agency.agent import _offload_large_fields
    sandbox = MagicMock()
    inp = agdata(items=[42, None, {"key": "val"}])
    paths, fields = _offload_large_fields(inp, sandbox, "sk")
    sandbox.write_file.assert_not_called()
    assert paths == []

def test_offload_large_fields_list_sandbox_failure_leaves_element_unchanged():
    from agency.agent import _offload_large_fields, INPUT_OFFLOAD_CHARS
    sandbox = MagicMock()
    sandbox.write_file.side_effect = OSError("no space")
    long_val = "x" * (INPUT_OFFLOAD_CHARS + 1)
    inp = agdata(items=[long_val])
    paths, fields = _offload_large_fields(inp, sandbox, "sk")
    assert paths == []
    assert fields == []
    assert inp._data["items"] == [long_val]

def test_offload_large_fields_skips_agtype_list_fields():
    from agency.agent import _offload_large_fields, INPUT_OFFLOAD_CHARS
    sandbox = MagicMock()
    data_url = "data:image/jpeg;base64," + "A" * (INPUT_OFFLOAD_CHARS + 1)
    inp = agdata(frames=[data_url, data_url])
    schema = agdata(frames=list[agimage])
    paths, fields = _offload_large_fields(inp, sandbox, "sk", schema=schema)
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert inp._data["frames"] == [data_url, data_url]

def test_offload_large_fields_skips_single_agtype_field():
    from agency.agent import _offload_large_fields, INPUT_OFFLOAD_CHARS
    sandbox = MagicMock()
    data_url = "data:image/jpeg;base64," + "A" * (INPUT_OFFLOAD_CHARS + 1)
    inp = agdata(photo=data_url)
    schema = agdata(photo=agimage)
    paths, fields = _offload_large_fields(inp, sandbox, "sk", schema=schema)
    sandbox.write_file.assert_not_called()
    assert inp._data["photo"] == data_url
