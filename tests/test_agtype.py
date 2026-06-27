"""Tests for the agtype base class, agdata serialization, and input offloading."""
import json
import pytest
from typing import get_origin, get_args
from unittest.mock import MagicMock
from agency.agdata import agdata
from agency.agtype import agtype, agfile, agimage, agbinary, agrawstring
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

def test_offload_large_fields_skips_single_agbinary_field():
    from agency.agent import _offload_large_fields, INPUT_OFFLOAD_CHARS
    sandbox = MagicMock()
    # After agbinary.prepare() the value is a short sandbox path, but the skip
    # should fire on the schema hint alone — verify with a long string too.
    long_path = "/workspace/inputs/" + "a" * (INPUT_OFFLOAD_CHARS + 1)
    inp = agdata(audio=long_path)
    schema = agdata(audio=agbinary)
    paths, fields = _offload_large_fields(inp, sandbox, "sk", schema=schema)
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert inp._data["audio"] == long_path

def test_offload_large_fields_skips_agbinary_list_fields():
    from agency.agent import _offload_large_fields, INPUT_OFFLOAD_CHARS
    sandbox = MagicMock()
    long_path = "/workspace/inputs/" + "b" * (INPUT_OFFLOAD_CHARS + 1)
    inp = agdata(clips=[long_path, long_path])
    schema = agdata(clips=list[agbinary])
    paths, fields = _offload_large_fields(inp, sandbox, "sk", schema=schema)
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert inp._data["clips"] == [long_path, long_path]

def test_offload_large_fields_skips_agfile_field():
    from agency.agent import _offload_large_fields, INPUT_OFFLOAD_CHARS
    sandbox = MagicMock()
    long_path = "/workspace/inputs/" + "c" * (INPUT_OFFLOAD_CHARS + 1)
    inp = agdata(doc=long_path)
    schema = agdata(doc=agfile)
    paths, fields = _offload_large_fields(inp, sandbox, "sk", schema=schema)
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert inp._data["doc"] == long_path

def test_offload_large_fields_offloads_agrawstring_when_long():
    from agency.agent import _offload_large_fields, INPUT_OFFLOAD_CHARS
    sandbox = MagicMock()
    long_text = "x" * (INPUT_OFFLOAD_CHARS + 1)
    inp = agdata(prompt=long_text)
    schema = agdata(prompt=agrawstring)
    paths, fields = _offload_large_fields(inp, sandbox, "sk", schema=schema)
    sandbox.write_file.assert_called_once()
    assert len(paths) == 1
    assert "prompt" in fields
    assert "sk_prompt" in inp._data["prompt"]

def test_offload_large_fields_preserves_short_agrawstring():
    from agency.agent import _offload_large_fields
    sandbox = MagicMock()
    inp = agdata(prompt="short")
    schema = agdata(prompt=agrawstring)
    paths, fields = _offload_large_fields(inp, sandbox, "sk", schema=schema)
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert inp._data["prompt"] == "short"

def test_offload_large_fields_skips_dict_agtype_field():
    from agency.agent import _offload_large_fields, INPUT_OFFLOAD_CHARS
    sandbox = MagicMock()
    long_val = "data:image/jpeg;base64," + "A" * (INPUT_OFFLOAD_CHARS + 1)
    inp = agdata(images={"a": long_val})
    schema = agdata(images=dict[str, agimage])
    paths, fields = _offload_large_fields(inp, sandbox, "sk", schema=schema)
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert inp._data["images"] == {"a": long_val}

def test_offload_large_fields_skips_tuple_agtype_field():
    from agency.agent import _offload_large_fields, INPUT_OFFLOAD_CHARS
    sandbox = MagicMock()
    long_val = "data:image/jpeg;base64," + "A" * (INPUT_OFFLOAD_CHARS + 1)
    inp = agdata(pair=(long_val, "label"))
    schema = agdata(pair=tuple[agimage, str])
    paths, fields = _offload_large_fields(inp, sandbox, "sk", schema=schema)
    sandbox.write_file.assert_not_called()
    assert paths == []
    assert inp._data["pair"] == (long_val, "label")


# ---------------------------------------------------------------------------
# _prepare_agtype_inputs — dict and tuple
# ---------------------------------------------------------------------------

def test_prepare_agtype_inputs_dict_agimage_encodes_values(tmp_path):
    from agency.agent import _prepare_agtype_inputs
    import base64
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    img_a.write_bytes(b"bytes_a")
    img_b.write_bytes(b"bytes_b")
    inp = agdata(images={"x": str(img_a), "y": str(img_b)})
    schema = agdata(images=dict[str, agimage])
    paths = _prepare_agtype_inputs(inp, schema, MagicMock(), "sk")
    assert paths == []
    assert inp._data["images"]["x"].startswith("data:image/jpeg;base64,")
    assert inp._data["images"]["y"].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(inp._data["images"]["x"].split(",", 1)[1]) == b"bytes_a"

def test_prepare_agtype_inputs_tuple_encodes_agtype_positions(tmp_path):
    from agency.agent import _prepare_agtype_inputs
    import base64
    img = tmp_path / "img.png"
    img.write_bytes(b"png_bytes")
    inp = agdata(pair=(str(img), "label"))
    schema = agdata(pair=tuple[agimage, str])
    _prepare_agtype_inputs(inp, schema, MagicMock(), "sk")
    result = inp._data["pair"]
    assert result[0].startswith("data:image/png;base64,")
    assert base64.b64decode(result[0].split(",", 1)[1]) == b"png_bytes"
    assert result[1] == "label"


# ---------------------------------------------------------------------------
# _recover_agtype_outputs — list, dict, tuple
# ---------------------------------------------------------------------------

def test_recover_agtype_outputs_list_agfile_reads_each():
    from agency.agent import _recover_agtype_outputs
    from agency.agdata import agdata as _agdata
    sandbox = MagicMock()
    sandbox.read_file.side_effect = ["content_a", "content_b"]
    result = _agdata(docs=["/workspace/a.txt", "/workspace/b.txt"])
    schema = _agdata(docs=list[agfile])
    _recover_agtype_outputs(result, schema, sandbox)
    assert result._data["docs"] == ["content_a", "content_b"]
    assert sandbox.read_file.call_count == 2

def test_recover_agtype_outputs_dict_agfile_reads_values():
    from agency.agent import _recover_agtype_outputs
    from agency.agdata import agdata as _agdata
    sandbox = MagicMock()
    sandbox.read_file.side_effect = ["content_x", "content_y"]
    result = _agdata(docs={"x": "/workspace/x.txt", "y": "/workspace/y.txt"})
    schema = _agdata(docs=dict[str, agfile])
    _recover_agtype_outputs(result, schema, sandbox)
    assert result._data["docs"] == {"x": "content_x", "y": "content_y"}

def test_recover_agtype_outputs_tuple_recovers_agtype_positions():
    from agency.agent import _recover_agtype_outputs
    from agency.agdata import agdata as _agdata
    sandbox = MagicMock()
    sandbox.read_file.return_value = "file_content"
    result = _agdata(pair=["/workspace/out.txt", 42])
    schema = _agdata(pair=tuple[agfile, int])
    _recover_agtype_outputs(result, schema, sandbox)
    assert result._data["pair"][0] == "file_content"
    assert result._data["pair"][1] == 42
    sandbox.read_file.assert_called_once()


# ---------------------------------------------------------------------------
# Deep nesting — _prepare_agtype_inputs and _recover_agtype_outputs
# ---------------------------------------------------------------------------

def test_prepare_agtype_inputs_nested_list_agimage(tmp_path):
    from agency.agent import _prepare_agtype_inputs
    import base64
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    img_a.write_bytes(b"img_a")
    img_b.write_bytes(b"img_b")
    inp = agdata(batches=[[str(img_a)], [str(img_b)]])
    schema = agdata(batches=list[list[agimage]])
    _prepare_agtype_inputs(inp, schema, MagicMock(), "sk")
    assert inp._data["batches"][0][0].startswith("data:image/jpeg;base64,")
    assert inp._data["batches"][1][0].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(inp._data["batches"][0][0].split(",", 1)[1]) == b"img_a"

def test_prepare_agtype_inputs_dict_of_list_agimage(tmp_path):
    from agency.agent import _prepare_agtype_inputs
    import base64
    img = tmp_path / "x.jpg"
    img.write_bytes(b"img_x")
    inp = agdata(groups={"g": [str(img)]})
    schema = agdata(groups=dict[str, list[agimage]])
    _prepare_agtype_inputs(inp, schema, MagicMock(), "sk")
    assert inp._data["groups"]["g"][0].startswith("data:image/jpeg;base64,")

def test_recover_agtype_outputs_nested_list_agfile():
    from agency.agent import _recover_agtype_outputs
    from agency.agdata import agdata as _agdata
    sandbox = MagicMock()
    sandbox.read_file.side_effect = ["content_a", "content_b"]
    result = _agdata(batches=[["/workspace/a.txt"], ["/workspace/b.txt"]])
    schema = _agdata(batches=list[list[agfile]])
    _recover_agtype_outputs(result, schema, sandbox)
    assert result._data["batches"] == [["content_a"], ["content_b"]]

def test_recover_agtype_outputs_dict_of_list_agfile():
    from agency.agent import _recover_agtype_outputs
    from agency.agdata import agdata as _agdata
    sandbox = MagicMock()
    sandbox.read_file.side_effect = ["content_x", "content_y"]
    result = _agdata(groups={"g": ["/workspace/x.txt", "/workspace/y.txt"]})
    schema = _agdata(groups=dict[str, list[agfile]])
    _recover_agtype_outputs(result, schema, sandbox)
    assert result._data["groups"] == {"g": ["content_x", "content_y"]}


# ---------------------------------------------------------------------------
# Deep nesting — _offload_large_fields skip
# ---------------------------------------------------------------------------

def test_offload_skips_nested_list_agimage():
    from agency.agent import _offload_large_fields, INPUT_OFFLOAD_CHARS
    sandbox = MagicMock()
    long_url = "data:image/jpeg;base64," + "A" * (INPUT_OFFLOAD_CHARS + 1)
    inp = agdata(batches=[[long_url], [long_url]])
    schema = agdata(batches=list[list[agimage]])
    paths, fields = _offload_large_fields(inp, sandbox, "sk", schema=schema)
    sandbox.write_file.assert_not_called()
    assert paths == []

def test_offload_skips_dict_of_list_agimage():
    from agency.agent import _offload_large_fields, INPUT_OFFLOAD_CHARS
    sandbox = MagicMock()
    long_url = "data:image/jpeg;base64," + "A" * (INPUT_OFFLOAD_CHARS + 1)
    inp = agdata(groups={"g": [long_url]})
    schema = agdata(groups=dict[str, list[agimage]])
    paths, fields = _offload_large_fields(inp, sandbox, "sk", schema=schema)
    sandbox.write_file.assert_not_called()
    assert paths == []


# ---------------------------------------------------------------------------
# Deep nesting — _validate_value
# ---------------------------------------------------------------------------

def test_validate_value_nested_list_str():
    from agency.agskill import _validate_value
    assert _validate_value(list[list[str]], [["a", "b"], ["c"]]) is None
    assert _validate_value(list[list[str]], [["a", 1]]) is not None

def test_validate_value_nested_list_agfile():
    from agency.agskill import _validate_value
    # agfile serialises as str; nested list of paths should validate
    assert _validate_value(list[list[agfile]], [["/a.txt", "/b.txt"]]) is None
    assert _validate_value(list[list[agfile]], [[123]]) is not None

def test_validate_value_dict_of_list_str():
    from agency.agskill import _validate_value
    assert _validate_value(dict[str, list[int]], {"k": [1, 2, 3]}) is None
    assert _validate_value(dict[str, list[int]], {"k": ["not_int"]}) is not None

def test_validate_value_tuple_with_nested_list():
    from agency.agskill import _validate_value
    assert _validate_value(tuple[list[str], int], [["a", "b"], 42]) is None
    assert _validate_value(tuple[list[str], int], [["a", "b"], "not_int"]) is not None


# ---------------------------------------------------------------------------
# Fuzz: _prepare_agtype_inputs — random nested schemas
# ---------------------------------------------------------------------------

def test_random_prepare_agtype_inputs_fuzz():
    """100 randomly generated nested schemas: _prepare_agtype_inputs must not
    crash, must preserve plain-Python values unchanged, and must call the
    correct sandbox method for each agfile/agbinary leaf it encounters.
    """
    import random
    from agency.agent import _prepare_agtype_inputs

    rng = random.Random(20240628)

    # agimage uses URL passthrough (no sandbox); agfile/agbinary use sandbox writes;
    # agrawstring is a no-op prepare.
    AGTYPE_LEAVES = [agimage, agfile, agbinary, agrawstring]
    PLAIN_LEAVES  = [str, int, float, bool]
    ALL_LEAVES    = AGTYPE_LEAVES + PLAIN_LEAVES

    def rand_hint(depth):
        if depth >= 3 or (depth > 0 and rng.random() < 0.4 * depth):
            return rng.choice(ALL_LEAVES)
        kind = rng.choice(("list", "dict", "tuple"))
        if kind == "list":
            return list[rand_hint(depth + 1)]
        if kind == "dict":
            return dict[str, rand_hint(depth + 1)]
        n = rng.randint(1, 3)
        inners = tuple(rand_hint(depth + 1) for _ in range(n))
        return tuple[inners] if len(inners) > 1 else tuple[inners[0]]

    def rand_value(hint):
        if hint is bool:        return rng.choice([True, False])
        if hint is int:         return rng.randint(-9, 9)
        if hint is float:       return round(rng.uniform(-9.0, 9.0), 1)
        if hint is str:         return rng.choice(["hello", "world"])
        if hint is agimage:     return f"https://example.com/img{rng.randint(0,9)}.jpg"
        if hint is agfile:      return rng.choice(["some text", "more text"])
        if hint is agbinary:    return b"raw bytes"
        if hint is agrawstring: return rng.choice(["raw", "text"])
        origin, args = get_origin(hint), get_args(hint)
        if origin is list:
            return [rand_value(args[0]) for _ in range(rng.randint(1, 3))]
        if origin is dict:
            return {f"k{i}": rand_value(args[1]) for i in range(rng.randint(1, 3))}
        if origin is tuple:
            return [rand_value(t) for t in args]
        return None

    def count_leaves(hint, value, *leaf_types):
        """Count how many values at agtype leaf positions match leaf_types."""
        if isinstance(hint, type) and issubclass(hint, tuple(leaf_types)):
            return 1
        origin, args = get_origin(hint), get_args(hint)
        if origin is list and args and isinstance(value, list):
            return sum(count_leaves(args[0], v, *leaf_types) for v in value)
        if origin is dict and len(args) == 2 and isinstance(value, dict):
            return sum(count_leaves(args[1], v, *leaf_types) for v in value.values())
        if origin is tuple and args and isinstance(value, (list, tuple)):
            return sum(count_leaves(ta, v, *leaf_types) for ta, v in zip(args, value))
        return 0

    failures = []
    for trial in range(100):
        hint  = rand_hint(0)
        value = rand_value(hint)
        sandbox = MagicMock()

        inp    = agdata(f=value)
        schema = agdata(f=hint)

        try:
            paths = _prepare_agtype_inputs(inp, schema, sandbox, "sk")
        except Exception as exc:
            failures.append(f"[{trial}] raised {exc!r} for hint={hint!r}")
            continue

        # agfile leaves each produce one sandbox.write_file call and one path
        n_agfile = count_leaves(hint, value, agfile)
        if sandbox.write_file.call_count != n_agfile:
            failures.append(
                f"[{trial}] write_file called {sandbox.write_file.call_count}x, "
                f"expected {n_agfile} for hint={hint!r}"
            )

        # agbinary leaves each produce one sandbox.write_file_bytes call and one path
        n_agbinary = count_leaves(hint, value, agbinary)
        if sandbox.write_file_bytes.call_count != n_agbinary:
            failures.append(
                f"[{trial}] write_file_bytes called {sandbox.write_file_bytes.call_count}x, "
                f"expected {n_agbinary} for hint={hint!r}"
            )

        # agimage (URL) and agrawstring and plain types produce no sandbox calls
        if len(paths) != n_agfile + n_agbinary:
            failures.append(
                f"[{trial}] paths={paths!r}, expected {n_agfile + n_agbinary} entries "
                f"for hint={hint!r}"
            )

    assert not failures, f"{len(failures)}/100 trials failed:\n" + "\n".join(failures[:20])


# ---------------------------------------------------------------------------
# Fuzz: _offload_large_fields — agtype fields skipped, plain/agrawstring offloaded
# ---------------------------------------------------------------------------

def test_random_offload_agtype_skip_fuzz():
    """100 randomly generated single-field schemas: _offload_large_fields must
    skip fields whose hint contains any non-agrawstring agtype at any nesting
    depth, and must offload plain str and agrawstring fields when the value
    exceeds INPUT_OFFLOAD_CHARS.
    """
    import random
    from agency.agent import _offload_large_fields, INPUT_OFFLOAD_CHARS

    rng = random.Random(20240629)

    NON_RAW_AGTYPES = [agimage, agfile, agbinary]
    ALL_LEAVES      = NON_RAW_AGTYPES + [agrawstring, str, int, float, bool]

    def rand_hint(depth):
        if depth >= 3 or (depth > 0 and rng.random() < 0.4 * depth):
            return rng.choice(ALL_LEAVES)
        kind = rng.choice(("list", "dict", "tuple"))
        if kind == "list":
            return list[rand_hint(depth + 1)]
        if kind == "dict":
            return dict[str, rand_hint(depth + 1)]
        n = rng.randint(1, 3)
        inners = tuple(rand_hint(depth + 1) for _ in range(n))
        return tuple[inners] if len(inners) > 1 else tuple[inners[0]]

    def hint_has_non_raw_agtype(hint) -> bool:
        if isinstance(hint, type) and issubclass(hint, agtype):
            return not issubclass(hint, agrawstring)
        origin, args = get_origin(hint), get_args(hint)
        if origin is list and args:
            return hint_has_non_raw_agtype(args[0])
        if origin is dict and len(args) == 2:
            return hint_has_non_raw_agtype(args[1])
        if origin is tuple and args:
            return any(hint_has_non_raw_agtype(a) for a in args)
        return False

    long_str = "x" * (INPUT_OFFLOAD_CHARS + 1)

    failures = []
    for trial in range(100):
        hint    = rand_hint(0)
        sandbox = MagicMock()
        inp     = agdata(f=long_str)
        schema  = agdata(f=hint)

        try:
            paths, fields = _offload_large_fields(inp, sandbox, "sk", schema=schema)
        except Exception as exc:
            failures.append(f"[{trial}] raised {exc!r} for hint={hint!r}")
            continue

        has_non_raw = hint_has_non_raw_agtype(hint)

        if has_non_raw:
            # Field must be skipped entirely — no sandbox write, value untouched
            if sandbox.write_file.called:
                failures.append(
                    f"[{trial}] write_file called for non-raw agtype hint={hint!r}"
                )
            if paths or fields:
                failures.append(
                    f"[{trial}] unexpected paths/fields for non-raw agtype hint={hint!r}"
                )
        else:
            # The offloader checks the actual value type, not the hint — our value
            # IS a long string, so write_file must always be called here.
            if not sandbox.write_file.called:
                failures.append(
                    f"[{trial}] write_file NOT called for hint={hint!r}"
                )

    assert not failures, f"{len(failures)}/100 trials failed:\n" + "\n".join(failures[:20])
