"""Tests for the agtype base class, agdata serialization, and input offloading."""
import json
import pytest
from typing import get_origin, get_args
from unittest.mock import MagicMock
from agency.agdata import agdata
from agency.agtype import agtype, agfile, agimage, agbinary, agrawstring


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
# Deep nesting — _validate_value
# ---------------------------------------------------------------------------

def test_validate_value_nested_list_str():
    from agency.agtype import _validate_value
    assert _validate_value(list[list[str]], [["a", "b"], ["c"]]) is None
    assert _validate_value(list[list[str]], [["a", 1]]) is not None

def test_validate_value_nested_list_agfile():
    from agency.agtype import _validate_value
    # agfile serialises as str; nested list of paths should validate
    assert _validate_value(list[list[agfile]], [["/a.txt", "/b.txt"]]) is None
    assert _validate_value(list[list[agfile]], [[123]]) is not None

def test_validate_value_dict_of_list_str():
    from agency.agtype import _validate_value
    assert _validate_value(dict[str, list[int]], {"k": [1, 2, 3]}) is None
    assert _validate_value(dict[str, list[int]], {"k": ["not_int"]}) is not None

def test_validate_value_tuple_with_nested_list():
    from agency.agtype import _validate_value
    assert _validate_value(tuple[list[str], int], [["a", "b"], 42]) is None
    assert _validate_value(tuple[list[str], int], [["a", "b"], "not_int"]) is not None


# ---------------------------------------------------------------------------
# _hint_to_json_type — moved from test_agskill.py
# ---------------------------------------------------------------------------

def test_hint_to_json_type_coverage():
    """_hint_to_json_type maps all common Python types to correct JSON Schema types."""
    from agency.agtype import _hint_to_json_type
    assert _hint_to_json_type(str)         == "string"
    assert _hint_to_json_type(int)         == "integer"
    assert _hint_to_json_type(float)       == "number"
    assert _hint_to_json_type(bool)        == "boolean"
    assert _hint_to_json_type(list)        == "array"
    assert _hint_to_json_type(list[str])   == "array"
    assert _hint_to_json_type(tuple)       == "array"
    assert _hint_to_json_type(tuple[str, int]) == "array"
    assert _hint_to_json_type(dict)        == "object"
    assert _hint_to_json_type(dict[str, int]) == "object"
    assert _hint_to_json_type([{"k": str}]) == "array"  # literal list-of-dicts


def test_return_output_list_str_type_error():
    """list[str] with a non-str element returns a validation error."""
    from agency.agtype import _validate_output_field
    schema = agdata(tags=list[str])
    err = _validate_output_field("tags", ["good", 42], schema)
    assert err is not None
    assert "int" in err
