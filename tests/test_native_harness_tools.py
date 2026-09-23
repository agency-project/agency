"""Tests for the pure replace/paginate algorithms in
native_harness/tools.py, shared by that harness's edit/read tool
dispatch; also the bash built-in's workdir/timeout handling (see
TestBash -- both were silently ignored before)."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from mcp.types import ImageContent, TextContent

from agency.native_harness import tools
from agency.native_harness.mcp_client import _decode_tool_result


def _run_bash(**kwargs):
    return json.loads(tools.TOOL_DISPATCH["bash"](json.dumps(kwargs)))


class TestBash:
    def test_honors_workdir(self, tmp_path):
        result = _run_bash(command="pwd", workdir=str(tmp_path))
        assert result["output"].strip() == str(tmp_path)
        assert result["returncode"] == 0

    def test_defaults_workdir_to_process_cwd_when_omitted(self):
        result = _run_bash(command="pwd")
        assert result["output"].strip() == os.getcwd()

    def test_reports_a_clear_error_for_a_missing_workdir(self, tmp_path):
        missing = str(tmp_path / "does-not-exist")
        result = _run_bash(command="pwd", workdir=missing)
        assert "error" in result
        assert missing in result["error"]

    def test_honors_a_custom_timeout(self):
        result = _run_bash(command="sleep 5", timeout=1)
        assert result == {"error": "command timed out after 1s"}

    def test_without_a_timeout_uses_the_120s_default(self):
        result = _run_bash(command="true")
        assert result["returncode"] == 0


class TestMcpToolResults:
    def test_uses_structured_content_when_present(self):
        result = SimpleNamespace(structured_content={"r": 1}, content=[])
        assert _decode_tool_result(result) == {"r": 1}

    def test_decodes_json_object_from_text_content(self):
        result = SimpleNamespace(
            structured_content=None,
            content=[SimpleNamespace(text='{\n  "r": 1\n}')],
        )
        assert _decode_tool_result(result) == {"r": 1}

    def test_wraps_plain_text_for_compatibility(self):
        result = SimpleNamespace(
            structured_content=None,
            content=[SimpleNamespace(text="plain text")],
        )
        assert _decode_tool_result(result) == {"result": "plain text"}

    def test_preserves_multiple_and_non_text_content_blocks(self):
        result = SimpleNamespace(
            structured_content=None,
            content=[
                TextContent(type="text", text="caption"),
                ImageContent(type="image", data="aGVsbG8=", mimeType="image/png"),
            ],
        )

        assert _decode_tool_result(result) == {
            "content": [
                {"type": "text", "text": "caption"},
                {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
            ]
        }


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
