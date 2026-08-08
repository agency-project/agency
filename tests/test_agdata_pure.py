"""Tests for agdata_pure.py -- the minimal, standalone-loadable stand-in for
`agdata`/`agerror` used to call a shipped `add_tools`/`replace_tools`
closure inside the native in-container entrypoint (see that module's
`_make_custom_tool_handler` and agdata_pure.py's own docstring for why a
full drop-in isn't needed and can't be loaded there the way this module is).

One thing every test here indirectly protects: this module must stay
loadable via `importlib.util.spec_from_file_location` with NO relative
imports and NO imports beyond stdlib -- same reasoning as
`test_agtool_pure.py`'s own `test_loadable_by_raw_file_path_like_the_entrypoint_does`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from agency import agdata_pure


class TestAgData:
    def test_construction_and_attribute_access(self):
        a = agdata_pure.agdata(x=1, y="hi")
        assert a.x == 1
        assert a.y == "hi"

    def test_missing_field_raises_attribute_error(self):
        a = agdata_pure.agdata(x=1)
        with pytest.raises(AttributeError, match="no field 'missing'"):
            a.missing

    def test_setattr(self):
        a = agdata_pure.agdata(x=1)
        a.y = 2
        assert a.y == 2

    def test_to_dict(self):
        a = agdata_pure.agdata(x=1, y="hi")
        assert a.to_dict() == {"x": 1, "y": "hi"}

    def test_to_json_round_trip(self):
        a = agdata_pure.agdata(x=1, y="hi")
        restored = agdata_pure.agdata.from_json(a.to_json())
        assert restored.to_dict() == {"x": 1, "y": "hi"}

    def test_from_dict(self):
        a = agdata_pure.agdata.from_dict({"x": 1})
        assert a.x == 1

    def test_equality(self):
        assert agdata_pure.agdata(x=1) == agdata_pure.agdata(x=1)
        assert agdata_pure.agdata(x=1) != agdata_pure.agdata(x=2)

    def test_repr(self):
        assert "x" in repr(agdata_pure.agdata(x=1))


class TestAgError:
    def test_error_field_accessible(self):
        e = agdata_pure.agerror("bad thing")
        assert e.error == "bad thing"

    def test_other_field_raises_agerror(self):
        e = agdata_pure.agerror("bad thing")
        with pytest.raises(agdata_pure.AgError, match="bad thing"):
            e.other_field

    def test_non_str_message_raises_type_error(self):
        with pytest.raises(TypeError):
            agdata_pure.agerror(123)

    def test_to_dict(self):
        e = agdata_pure.agerror("bad thing")
        assert e.to_dict() == {"error": "bad thing"}

    def test_repr(self):
        assert "bad thing" in repr(agdata_pure.agerror("bad thing"))


def test_loadable_by_raw_file_path_like_the_entrypoint_does():
    """The in-container native entrypoint can't `from agency.agdata_pure
    import ...` (that would execute agency/__init__.py first, pulling in
    the same heavy host-venv-only dependency chain this module exists to
    avoid) -- it loads this exact file by path instead. Prove that path
    actually works, against the real file on disk, not just that a normal
    package import succeeds."""
    module_path = Path(agdata_pure.__file__)
    spec = importlib.util.spec_from_file_location("agdata_pure_standalone", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    a = module.agdata(n=21)
    assert a.n == 21
    e = module.agerror("boom")
    assert e.error == "boom"

    sys.modules.pop("agdata_pure_standalone", None)
