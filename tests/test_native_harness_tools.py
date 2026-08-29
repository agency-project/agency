"""Tests for the pure replace/paginate algorithms in
native_harness/tools.py, shared by that harness's edit/read tool
dispatch."""

from __future__ import annotations

import pytest

from agency.native_harness import tools


class TestReplace:
    def test_simple_replace(self):
        result = tools.replace("def foo():\n    return 1\n", "return 1", "return 2")
        assert "return 2" in result

    def test_not_found_raises(self):
        with pytest.raises(ValueError, match="Could not find"):
            tools.replace("def foo(): pass\n", "MISSING", "x")

    def test_identical_strings_raises(self):
        with pytest.raises(ValueError):
            tools.replace("hello\n", "hello", "hello")

    def test_replace_all(self):
        result = tools.replace("a a a\n", "a", "b", replace_all=True)
        assert result == "b b b\n"

    def test_ambiguous_raises(self):
        with pytest.raises(ValueError, match="multiple matches"):
            tools.replace("x\nx\n", "x", "y")

    def test_line_trimmed_fallback(self):
        content = "    def foo():\n        pass\n"
        result = tools.replace(content, "def foo():\n    pass", "def bar():\n    pass")
        assert "bar" in result


class TestPaginateText:
    def test_returns_plain_dict(self):
        result = tools.paginate_text("a\nb\nc\n", offset=1, limit=10)
        assert type(result) is dict

    def test_pagination_fields(self):
        result = tools.paginate_text("a\nb\nc\n", offset=1, limit=2)
        assert result["lines_shown"] == 2
        assert result["total_lines"] == 3
        assert result["truncated"] is True
        assert result["content"] == "1: a\n2: b"

    def test_offset_into_middle(self):
        result = tools.paginate_text("a\nb\nc\n", offset=2, limit=10)
        assert result["content"] == "2: b\n3: c"
        assert result["truncated"] is False
