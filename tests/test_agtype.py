"""Tests for the agtype base class, agdata serialization, and input offloading."""

import json
from agency.agdata import agdata
from agency.agtype import agtype, agfile, agpath


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


def test_agtype_validate_input_value_accepts_str():
    assert agtype.validate_input_value("hello") is None


def test_agtype_validate_input_value_rejects_non_str():
    assert agtype.validate_input_value(123) is not None


def test_agtype_validate_output_default_none():
    assert agtype.validate_output("x", "anything", None, 5) is None


# ---------------------------------------------------------------------------
# agdata serialization with type objects
# ---------------------------------------------------------------------------


def test_agdata_serializes_python_types():
    d = agdata(x=str, n=int, f=float, b=bool, items=list, mapping=dict)
    parsed = json.loads(d.to_json())
    assert parsed == {
        "x": "str",
        "n": "int",
        "f": "float",
        "b": "bool",
        "items": "list",
        "mapping": "dict",
    }


def test_agdata_serializes_custom_agtype():
    class agblob(agtype):
        @classmethod
        def schema_type(cls):
            return "blob"

    d = agdata(data=agblob)
    assert json.loads(d.to_json()) == {"data": "blob"}


# ---------------------------------------------------------------------------
# Deep nesting — validate_value_against_type_hint
# ---------------------------------------------------------------------------


def test_validate_value_nested_list_str():
    from agency.agtype import validate_value_against_type_hint

    assert validate_value_against_type_hint(list[list[str]], [["a", "b"], ["c"]]) is None
    assert validate_value_against_type_hint(list[list[str]], [["a", 1]]) is not None


def test_validate_value_nested_list_agfile():
    from agency.agtype import validate_value_against_type_hint

    # agfile serialises as str; nested list of paths should validate
    assert validate_value_against_type_hint(list[list[agfile]], [["/a.txt", "/b.txt"]]) is None
    assert validate_value_against_type_hint(list[list[agfile]], [[123]]) is not None


def test_validate_value_agpath_accepts_path():
    from agency.agtype import validate_value_against_type_hint

    assert validate_value_against_type_hint(agpath, "/workspace/out.txt") is None


def test_validate_value_agpath_rejects_non_path():
    from agency.agtype import validate_value_against_type_hint

    assert validate_value_against_type_hint(agpath, "not a path") is not None


def test_validate_value_nested_list_agpath():
    from agency.agtype import validate_value_against_type_hint

    assert validate_value_against_type_hint(list[agpath], ["/a.txt", "/b.txt"]) is None
    assert validate_value_against_type_hint(list[agpath], ["/a.txt", "not a path"]) is not None


def test_validate_value_dict_of_list_str():
    from agency.agtype import validate_value_against_type_hint

    assert validate_value_against_type_hint(dict[str, list[int]], {"k": [1, 2, 3]}) is None
    assert validate_value_against_type_hint(dict[str, list[int]], {"k": ["not_int"]}) is not None


def test_validate_value_tuple_with_nested_list():
    from agency.agtype import validate_value_against_type_hint

    assert validate_value_against_type_hint(tuple[list[str], int], [["a", "b"], 42]) is None
    assert (
        validate_value_against_type_hint(tuple[list[str], int], [["a", "b"], "not_int"]) is not None
    )


# ---------------------------------------------------------------------------
# type_hint_to_string_type — moved from test_agskill.py
# ---------------------------------------------------------------------------


def testtype_hint_to_string_type_coverage():
    """type_hint_to_string_type maps all common Python types to correct JSON Schema types."""
    from agency.agtype import type_hint_to_string_type

    assert type_hint_to_string_type(str) == "string"
    assert type_hint_to_string_type(int) == "integer"
    assert type_hint_to_string_type(float) == "number"
    assert type_hint_to_string_type(bool) == "boolean"
    assert type_hint_to_string_type(list) == "array"
    assert type_hint_to_string_type(list[str]) == "array"
    assert type_hint_to_string_type(tuple) == "array"
    assert type_hint_to_string_type(tuple[str, int]) == "array"
    assert type_hint_to_string_type(dict) == "object"
    assert type_hint_to_string_type(dict[str, int]) == "object"
    assert type_hint_to_string_type([{"k": str}]) == "array"  # literal list-of-dicts


def test_return_output_list_str_type_error():
    """list[str] with a non-str element returns a validation error."""
    from agency.agtype import validate_output_field_against_schema

    schema = agdata(tags=list[str])
    err = validate_output_field_against_schema("tags", ["good", 42], schema)
    assert err is not None
    assert "int" in err
