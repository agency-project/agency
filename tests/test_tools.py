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


# ---------------------------------------------------------------------------
# _log functions must not raise on error results
#
# getattr(agdata_with_error, "field", default) raises AgError instead of
# returning the default, because agdata.__getattr__ raises AgError whenever
# the data dict contains an "error" key.  All tool _log functions must use
# result._data.get() instead so they survive error results gracefully.
# ---------------------------------------------------------------------------

class TestToolLogOnErrorResult:
    """Verify that _log functions in sandboxed tools never raise when the
    tool returns an error result (regression: AgError propagation in log)."""

    def _make_term(self):
        from unittest.mock import MagicMock
        term = MagicMock()
        term.log = MagicMock()
        return term

    def _make_tool_with_term(self, tool):
        term = self._make_term()
        tool._term = term
        return tool, term

    def test_read_log_does_not_raise_on_error(self):
        from agency.tools.read import make_read
        from unittest.mock import MagicMock
        sb = MagicMock()
        tool = make_read(sb)
        tool, term = self._make_tool_with_term(tool)
        error_result = agdata(error="Not found: /workspace/missing.txt")
        # Must not raise AgError
        tool._log_fn(tool, agdata(filePath="/workspace/missing.txt"), error_result, 42)
        # Log was called with the error path, not the success path
        assert term.log.called
        logged = term.log.call_args[0]
        assert "✗" in logged[0] or "error" in str(logged).lower()

    def test_write_log_does_not_raise_on_error(self):
        from agency.tools.write import make_write
        from unittest.mock import MagicMock
        sb = MagicMock()
        tool = make_write(sb)
        tool, term = self._make_tool_with_term(tool)
        error_result = agdata(error="Permission denied")
        tool._log_fn(tool, agdata(filePath="/workspace/out.txt"), error_result, 10)
        assert term.log.called
        logged = term.log.call_args[0]
        assert "✗" in logged[0] or "error" in str(logged).lower()

    def test_bash_log_does_not_raise_on_error(self):
        from agency.tools.bash import make_bash
        from unittest.mock import MagicMock
        sb = MagicMock()
        tool = make_bash(sb)
        tool, term = self._make_tool_with_term(tool)
        error_result = agdata(error="timed out")
        tool._log_fn(tool, agdata(command="sleep 999"), error_result, 30000)
        assert term.log.called

    def test_glob_log_does_not_raise_on_error(self):
        from agency.tools.glob import make_glob
        from unittest.mock import MagicMock
        sb = MagicMock()
        tool = make_glob(sb)
        tool, term = self._make_tool_with_term(tool)
        error_result = agdata(error="pattern error")
        tool._log_fn(tool, agdata(pattern="**/*.py"), error_result, 5)
        assert term.log.called

    def test_grep_log_does_not_raise_on_error(self):
        from agency.tools.grep import make_grep
        from unittest.mock import MagicMock
        sb = MagicMock()
        tool = make_grep(sb)
        tool, term = self._make_tool_with_term(tool)
        error_result = agdata(error="search failed")
        tool._log_fn(tool, agdata(pattern="TODO"), error_result, 5)
        assert term.log.called
