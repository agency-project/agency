"""Tests for agtool_pure.py -- the dependency-free replace/paginate
algorithms shared by the host-side edit/read tools and the in-container
native entrypoint.

One thing every test here indirectly protects: this module must stay
loadable via `importlib.util.spec_from_file_location` with NO relative
imports and NO imports beyond stdlib -- that's the entire reason it's split
out from tools/edit.py and tools/read.py (see its own module docstring).
`test_loadable_by_raw_file_path_like_the_entrypoint_does` checks this
directly, the same way the in-container entrypoint actually loads it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from agency import agtool_pure


class TestReplace:
    def test_simple_replace(self):
        result = agtool_pure.replace("def foo():\n    return 1\n", "return 1", "return 2")
        assert "return 2" in result

    def test_not_found_raises(self):
        with pytest.raises(ValueError, match="Could not find"):
            agtool_pure.replace("def foo(): pass\n", "MISSING", "x")

    def test_identical_strings_raises(self):
        with pytest.raises(ValueError):
            agtool_pure.replace("hello\n", "hello", "hello")

    def test_replace_all(self):
        result = agtool_pure.replace("a a a\n", "a", "b", replace_all=True)
        assert result == "b b b\n"

    def test_ambiguous_raises(self):
        with pytest.raises(ValueError, match="multiple matches"):
            agtool_pure.replace("x\nx\n", "x", "y")

    def test_line_trimmed_fallback(self):
        content = "    def foo():\n        pass\n"
        result = agtool_pure.replace(content, "def foo():\n    pass", "def bar():\n    pass")
        assert "bar" in result


class TestPaginateText:
    def test_returns_plain_dict_not_agdata(self):
        result = agtool_pure.paginate_text("a\nb\nc\n", offset=1, limit=10)
        assert type(result) is dict

    def test_pagination_fields(self):
        result = agtool_pure.paginate_text("a\nb\nc\n", offset=1, limit=2)
        assert result["lines_shown"] == 2
        assert result["total_lines"] == 3
        assert result["truncated"] is True
        assert result["content"] == "1: a\n2: b"

    def test_offset_into_middle(self):
        result = agtool_pure.paginate_text("a\nb\nc\n", offset=2, limit=10)
        assert result["content"] == "2: b\n3: c"
        assert result["truncated"] is False


def test_loadable_by_raw_file_path_like_the_entrypoint_does():
    """The in-container native entrypoint can't `from agency.agtool_pure
    import ...` (that would execute agency/__init__.py first, pulling in
    the same heavy host-venv-only dependency chain this module exists to
    avoid) -- it loads this exact file by path instead. Prove that path
    actually works, against the real file on disk, not just that a normal
    package import succeeds."""
    module_path = Path(agtool_pure.__file__)
    spec = importlib.util.spec_from_file_location("agtool_pure_standalone", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.replace("x = 1\n", "x = 1", "x = 2") == "x = 2\n"
    assert module.paginate_text("only line\n", 1, 10)["total_lines"] == 1

    sys.modules.pop("agtool_pure_standalone", None)
