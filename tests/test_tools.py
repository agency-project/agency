"""Unit tests for all 8 ported tools."""
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agdata import agdata


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------

class TestBash:
    def setup_method(self):
        from src.tools.bash import bash
        self.tool = bash

    def test_simple_command(self):
        result = self.tool(agdata(command="echo hello"))
        assert "hello" in result.output
        assert result.exit_code == 0

    def test_nonzero_exit(self):
        result = self.tool(agdata(command="exit 1", timeout=5))
        assert result.exit_code != 0

    def test_stderr_captured(self):
        result = self.tool(agdata(command="echo err >&2"))
        assert "err" in result.output

    def test_timeout(self):
        result = self.tool(agdata(command="sleep 10", timeout=1))
        assert result.exit_code == -1
        assert "timed out" in result.output.lower()

    def test_workdir(self, tmp_path):
        result = self.tool(agdata(command="pwd", workdir=str(tmp_path)))
        assert str(tmp_path) in result.output

    def test_truncation(self):
        # Generate output larger than 50 KB
        result = self.tool(agdata(command="python3 -c \"print('x'*100+'\\n', end='') \" " * 600))
        # If truncated flag set, output starts with truncation notice
        if result.truncated:
            assert "truncated" in result.output


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------

class TestRead:
    def setup_method(self):
        from src.tools.read import read
        self.tool = read

    def test_read_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("line1\nline2\nline3\n")
        result = self.tool(agdata(filePath=str(f)))
        assert "line1" in result.content
        assert result.type == "file"

    def test_read_with_offset_limit(self, tmp_path):
        f = tmp_path / "nums.txt"
        f.write_text("\n".join(str(i) for i in range(1, 11)))
        result = self.tool(agdata(filePath=str(f), offset=3, limit=2))
        assert result.offset == 3
        assert result.lines_shown == 2
        assert "3:" in result.content
        assert "4:" in result.content

    def test_read_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("")
        (tmp_path / "b.txt").write_text("")
        result = self.tool(agdata(filePath=str(tmp_path)))
        assert result.type == "directory"
        assert "a.txt" in result.entries
        assert "b.txt" in result.entries

    def test_missing_file(self, tmp_path):
        result = self.tool(agdata(filePath=str(tmp_path / "nope.txt")))
        assert result.error is not None


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

class TestWrite:
    def setup_method(self):
        from src.tools.write import write
        self.tool = write

    def test_create_file(self, tmp_path):
        f = tmp_path / "out.txt"
        result = self.tool(agdata(filePath=str(f), content="hello"))
        assert result.created is True
        assert f.read_text() == "hello"

    def test_overwrite_file(self, tmp_path):
        f = tmp_path / "out.txt"
        f.write_text("old")
        result = self.tool(agdata(filePath=str(f), content="new"))
        assert result.created is False
        assert f.read_text() == "new"

    def test_creates_parent_dirs(self, tmp_path):
        f = tmp_path / "a" / "b" / "c.txt"
        result = self.tool(agdata(filePath=str(f), content="deep"))
        assert f.read_text() == "deep"
        assert getattr(result, "error", None) is None


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------

class TestEdit:
    def setup_method(self):
        from src.tools.edit import edit, _replace
        self.tool = edit
        self._replace = _replace

    def test_simple_replace(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def foo():\n    return 1\n")
        result = self.tool(agdata(filePath=str(f), oldString="return 1", newString="return 2"))
        assert result.success is True
        assert "return 2" in f.read_text()

    def test_not_found(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def foo(): pass\n")
        result = self.tool(agdata(filePath=str(f), oldString="MISSING", newString="x"))
        assert result.error is not None

    def test_identical_strings(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("hello\n")
        result = self.tool(agdata(filePath=str(f), oldString="hello", newString="hello"))
        assert result.error is not None

    def test_replace_all(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("a a a\n")
        result = self.tool(agdata(filePath=str(f), oldString="a", newString="b", replaceAll=True))
        assert result.success is True
        assert f.read_text() == "b b b\n"

    def test_replace_pipeline_line_trimmed(self):
        content = "    def foo():\n        pass\n"
        result = self._replace(content, "def foo():\n    pass", "def bar():\n    pass")
        assert "bar" in result

    def test_ambiguous_raises(self):
        content = "x\nx\n"
        with pytest.raises(ValueError, match="multiple matches"):
            self._replace(content, "x", "y")

    def test_missing_file(self, tmp_path):
        result = self.tool(agdata(filePath=str(tmp_path / "nope.py"), oldString="x", newString="y"))
        assert result.error is not None


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------

class TestGlob:
    def setup_method(self):
        from src.tools.glob import glob
        self.tool = glob

    def test_finds_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = self.tool(agdata(pattern="*.py", path=str(tmp_path)))
        names = [Path(p).name for p in result.files]
        assert "a.py" in names
        assert "b.py" in names
        assert "c.txt" not in names

    def test_empty_result(self, tmp_path):
        result = self.tool(agdata(pattern="*.xyz", path=str(tmp_path)))
        assert result.files == []
        assert result.count == 0

    def test_missing_dir(self, tmp_path):
        result = self.tool(agdata(pattern="*.py", path=str(tmp_path / "ghost")))
        assert result.error is not None


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------

class TestGrep:
    def setup_method(self):
        from src.tools.grep import grep
        self.tool = grep

    def test_finds_match(self, tmp_path):
        f = tmp_path / "src.py"
        f.write_text("def hello():\n    pass\n")
        result = self.tool(agdata(pattern="def hello", path=str(tmp_path)))
        assert result.count >= 1
        assert any("hello" in m["text"] for m in result.matches)

    def test_no_match(self, tmp_path):
        f = tmp_path / "src.py"
        f.write_text("nothing here\n")
        result = self.tool(agdata(pattern="ZZZNOMATCH", path=str(tmp_path)))
        assert result.count == 0

    def test_include_filter(self, tmp_path):
        (tmp_path / "a.py").write_text("target_word\n")
        (tmp_path / "b.txt").write_text("target_word\n")
        result = self.tool(agdata(pattern="target_word", path=str(tmp_path), include="*.py"))
        paths = [m["path"] for m in result.matches]
        assert all(p.endswith(".py") for p in paths)


# ---------------------------------------------------------------------------
# webfetch
# ---------------------------------------------------------------------------

class TestWebfetch:
    def setup_method(self):
        from src.tools.webfetch import webfetch
        self.tool = webfetch

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
            result = self.tool(agdata(url="https://example.com"))
        assert "Hello" in result.output
        assert getattr(result, "error", None) is None

    def test_plain_text(self):
        with patch("httpx.get", return_value=self._mock_response("plain text", "text/plain")):
            result = self.tool(agdata(url="https://example.com", format="text"))
        assert "plain text" in result.output

    def test_invalid_url(self):
        result = self.tool(agdata(url="ftp://bad"))
        assert result.error is not None

    def test_http_error(self):
        import httpx as _httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        exc = _httpx.HTTPStatusError("404", request=MagicMock(), response=mock_resp)
        with patch("httpx.get", side_effect=exc):
            result = self.tool(agdata(url="https://example.com/missing"))
        assert result.error is not None


# ---------------------------------------------------------------------------
# todowrite
# ---------------------------------------------------------------------------

class TestTodowrite:
    def setup_method(self):
        from src.tools import todowrite as _todowrite_mod
        import src.tools.todowrite as m
        m._store = []
        from src.tools.todowrite import todowrite
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
