"""Unit tests for tool logic that runs on the host (no container needed).

Sandboxed tool integration tests (bash, read, write, edit, glob, grep)
live in test_agsandbox.py::TestSandboxedTools which runs against a real
container.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from agency.agdata import agdata


# ---------------------------------------------------------------------------
# edit — _replace fuzzy logic (pure Python, no container)
# ---------------------------------------------------------------------------

class TestEditLogic:
    def setup_method(self):
        from agency.tools.edit import _replace
        self._replace = _replace

    def test_simple_replace(self):
        result = self._replace("def foo():\n    return 1\n", "return 1", "return 2")
        assert "return 2" in result

    def test_not_found_raises(self):
        with pytest.raises(ValueError, match="Could not find"):
            self._replace("def foo(): pass\n", "MISSING", "x")

    def test_identical_strings_raises(self):
        with pytest.raises(ValueError):
            self._replace("hello\n", "hello", "hello")

    def test_replace_all(self):
        result = self._replace("a a a\n", "a", "b", replace_all=True)
        assert result == "b b b\n"

    def test_ambiguous_raises(self):
        with pytest.raises(ValueError, match="multiple matches"):
            self._replace("x\nx\n", "x", "y")

    def test_line_trimmed_fallback(self):
        content = "    def foo():\n        pass\n"
        result = self._replace(content, "def foo():\n    pass", "def bar():\n    pass")
        assert "bar" in result


# ---------------------------------------------------------------------------
# webfetch
# ---------------------------------------------------------------------------

class TestWebfetch:
    def setup_method(self):
        from agency.tools.webfetch import webfetch
        # Call .fn() directly: tests fetch/convert logic in-process with mocked
        # httpx. The process-pool mechanism is covered by test_agtool.py.
        self.fn = webfetch.fn

    def _mock_response(self, text: str, content_type: str = "text/html"):
        mock_resp = MagicMock()
        mock_resp.text = text
        mock_resp.content = text.encode()
        mock_resp.headers = {"content-type": content_type}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_html_to_markdown(self):
        html = "<html><body><h1>Hello</h1><p>World</p></body></html>"
        with patch("httpx.get", return_value=self._mock_response(html)):
            result = self.fn(agdata(url="https://example.com"))
        assert "Hello" in result.output
        assert getattr(result, "error", None) is None

    def test_plain_text(self):
        with patch("httpx.get", return_value=self._mock_response("plain text", "text/plain")):
            result = self.fn(agdata(url="https://example.com", format="text"))
        assert "plain text" in result.output

    def test_invalid_url(self):
        result = self.fn(agdata(url="ftp://bad"))
        assert result.error is not None

    def test_http_error(self):
        import httpx as _httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        exc = _httpx.HTTPStatusError("404", request=MagicMock(), response=mock_resp)
        with patch("httpx.get", side_effect=exc):
            result = self.fn(agdata(url="https://example.com/missing"))
        assert result.error is not None


# ---------------------------------------------------------------------------
# todowrite
# ---------------------------------------------------------------------------

class TestTodowrite:
    def setup_method(self):
        import agency.tools.todowrite as m
        m._store = []
        from agency.tools.todowrite import todowrite
        self.tool = todowrite

    def test_set_todos(self):
        todos = [
            {"content": "task 1", "status": "pending", "priority": "high"},
            {"content": "task 2", "status": "completed", "priority": "low"},
        ]
        result = self.tool(agdata(todos=todos))
        assert result.count == 2
        assert result.pending == 1

    def test_overwrite(self):
        self.tool(agdata(todos=[{"content": "old", "status": "pending", "priority": "low"}]))
        result = self.tool(agdata(todos=[{"content": "new", "status": "in_progress", "priority": "high"}]))
        assert result.count == 1
        assert result.todos[0]["content"] == "new"

    def test_missing_todos_field(self):
        result = self.tool(agdata(wrong="field"))
        assert result.error is not None

    def test_output_is_valid_json(self):
        todos = [{"content": "t", "status": "pending", "priority": "medium"}]
        result = self.tool(agdata(todos=todos))
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
