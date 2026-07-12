"""Unit tests for tool logic that runs on the host (no container needed).

Sandboxed tool integration tests (bash, read, write, edit, glob, grep)
live in test_agsandbox.py::TestSandboxedTools which runs against a real
container.
"""

import json
import threading
import pytest
from unittest.mock import patch, MagicMock

from agency.agdata import agdata, agerror


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
        assert not isinstance(result, agerror)

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
        result = self.tool(
            agdata(todos=[{"content": "new", "status": "in_progress", "priority": "high"}])
        )
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
# _log functions must not raise on agerror results
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
        error_result = agerror("Not found: /workspace/missing.txt")
        # Must not raise AgError
        tool._log_fn(tool, agdata(file_path="/workspace/missing.txt"), error_result, 42)
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
        error_result = agerror("Permission denied")
        tool._log_fn(tool, agdata(file_path="/workspace/out.txt"), error_result, 10)
        assert term.log.called
        logged = term.log.call_args[0]
        assert "✗" in logged[0] or "error" in str(logged).lower()

    def test_bash_log_does_not_raise_on_error(self):
        from agency.tools.bash import make_bash
        from unittest.mock import MagicMock

        sb = MagicMock()
        tool = make_bash(sb)
        tool, term = self._make_tool_with_term(tool)
        error_result = agerror("timed out")
        tool._log_fn(tool, agdata(command="sleep 999"), error_result, 30000)
        assert term.log.called

    def test_glob_log_does_not_raise_on_error(self):
        from agency.tools.glob import make_glob
        from unittest.mock import MagicMock

        sb = MagicMock()
        tool = make_glob(sb)
        tool, term = self._make_tool_with_term(tool)
        error_result = agerror("pattern error")
        tool._log_fn(tool, agdata(pattern="**/*.py"), error_result, 5)
        assert term.log.called

    def test_grep_log_does_not_raise_on_error(self):
        from agency.tools.grep import make_grep
        from unittest.mock import MagicMock

        sb = MagicMock()
        tool = make_grep(sb)
        tool, term = self._make_tool_with_term(tool)
        error_result = agerror("search failed")
        tool._log_fn(tool, agdata(pattern="TODO"), error_result, 5)
        assert term.log.called


# ---------------------------------------------------------------------------
# Sandbox tools must have run_in_subprocess=False so they run in the calling thread.
#
# Background: sandbox tools (bash, read, write, edit, glob, grep) close over
# an agSandbox instance. After reserve_gpu is called, the sandbox holds
# references to pool.acquire_gpu / pool.release_gpu — bound methods on an
# agResourcePool which contains threading.Semaphore objects. threading.Semaphore
# wraps _thread.lock, which cloudpickle cannot serialise. If any sandbox tool
# had run_in_subprocess=True, cloudpickle.dumps(t.fn) would raise
# "TypeError: cannot pickle '_thread.lock' object" the moment the LLM tried
# to call bash after calling reserve_gpu.
#
# The fix: all sandbox tools use run_in_subprocess=False, running in the calling
# thread (subprocess calls inside them already release the GIL, so no process
# pool is needed for GIL relief). GPU acquisition inside exec() also runs on
# the real agResourcePool in the calling thread, not a deserialized copy in a
# worker process.
# ---------------------------------------------------------------------------


class TestSandboxToolsRunInSubprocessFalse:
    """Regression tests for the run_in_subprocess=False requirement on all sandbox tools."""

    SANDBOX_TOOL_FACTORIES = [
        ("bash", "make_bash", ("bash.py", "make_bash")),
        ("read", "make_read", ("read.py", "make_read")),
        ("write", "make_write", ("write.py", "make_write")),
        ("edit", "make_edit", ("edit.py", "make_edit")),
        ("glob", "make_glob", ("glob.py", "make_glob")),
        ("grep", "make_grep", ("grep.py", "make_grep")),
    ]

    def _make_sandbox_mock(self):
        return MagicMock()

    def _make_pool_with_semaphore(self):
        """Return a mock pool whose acquire_gpu attribute holds a real threading.Semaphore,
        reproducing the exact unpicklable structure that triggered the bug."""
        pool = MagicMock()
        real_sem = threading.Semaphore(1)
        pool._gpu_semaphore = real_sem

        # Bind acquire_gpu to a method that uses the real semaphore so that
        # cloudpickle would have to serialise it.
        def _acquire():
            real_sem.acquire()
            return 0

        pool.acquire_gpu = _acquire
        pool.release_gpu = MagicMock()
        pool.gpus = [0]
        return pool

    @pytest.mark.parametrize("tool_name,factory_name,_", SANDBOX_TOOL_FACTORIES)
    def test_run_in_subprocess_is_false(self, tool_name, factory_name, _):
        """Every sandbox tool must have run_in_subprocess=False."""
        import importlib

        mod = importlib.import_module(f"agency.tools.{tool_name}")
        factory = getattr(mod, factory_name)
        tool = factory(self._make_sandbox_mock())
        assert tool.run_in_subprocess is False, (
            f"{factory_name} has run_in_subprocess=True — it will fail cloudpickle "
            f"serialisation after reserve_gpu is called (see test docstring)."
        )

    def test_bash_callable_after_reserve_gpu(self):
        """Calling bash after reserve_gpu must not raise a pickle error.

        Reproduces the exact sequence that failed:
          1. reserve_gpu sets sandbox._gpu_acquire_fn = pool.acquire_gpu
          2. LLM calls bash → agtool.__call__ → must NOT attempt cloudpickle.dumps
        """
        from agency.tools.bash import make_bash
        from agency.tools.resource import make_gpu_reserve

        pool = self._make_pool_with_semaphore()
        sb = MagicMock()
        sb._gpu_virtual = False
        sb._gpu_acquire_fn = None
        sb._gpu_release_fn = None
        sb.exec.return_value = ("hello\n", 0)

        reserve_gpu = make_gpu_reserve(sb, pool)
        bash = make_bash(sb)

        # Step 1: call reserve_gpu — sets _gpu_acquire_fn on the sandbox mock
        reserve_result = reserve_gpu(agdata())
        assert getattr(reserve_result, "error", None) is None

        # Step 2: call bash — must not raise TypeError about _thread.lock
        bash_result = bash(agdata(command="echo hello"))
        assert getattr(bash_result, "error", None) is None

    def test_make_sandboxed_tools_all_run_in_subprocess_false(self):
        """make_sandboxed_tools must return only run_in_subprocess=False tools.

        This is the integration check: even after reserve_gpu has been called and
        sandbox._gpu_acquire_fn points to an unpicklable pool method, no tool in
        the list should attempt to pickle its fn.
        """
        from agency.tools import make_sandboxed_tools
        from agency.agresources import agResourcePool

        pool = agResourcePool()
        sb = MagicMock()
        # Simulate post-reserve_gpu state: sandbox now holds pool method references.
        sb._gpu_virtual = True
        sb._gpu_acquire_fn = pool.acquire_gpu
        sb._gpu_release_fn = pool.release_gpu

        tools = make_sandboxed_tools(sb, pool)
        sandbox_true = [t.name for t in tools if t.run_in_subprocess]
        assert sandbox_true == [], (
            f"These tools have run_in_subprocess=True and will fail cloudpickle after "
            f"reserve_gpu is called: {sandbox_true}"
        )

    def test_sandbox_tools_run_in_calling_thread(self):
        """run_in_subprocess=False tools run in the calling thread, not a worker process.

        This is required for GPU acquisition to update the real agResourcePool
        (a worker process would acquire against a deserialized copy and the
        main process pool semaphore would never be decremented).
        """
        from agency.tools.bash import make_bash

        caller_tid = threading.get_ident()
        tool_tid_box: list[int] = []

        sb = MagicMock()

        def _exec_capture(cmd, workdir="/workspace", timeout=120):
            tool_tid_box.append(threading.get_ident())
            return ("ok\n", 0)

        sb.exec.side_effect = _exec_capture

        bash = make_bash(sb)
        bash(agdata(command="echo ok"))

        assert tool_tid_box, "bash fn was never called"
        assert tool_tid_box[0] == caller_tid, (
            "bash ran in a different thread — GPU pool updates would affect a copy, "
            "not the real pool."
        )
