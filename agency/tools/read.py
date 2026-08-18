from __future__ import annotations

import shlex
from typing import TYPE_CHECKING
from ..agdata import agdata, agerror
from ..agutil import format_exception
from ..agtool import agtool
from ..agtool_pure import paginate_text as _paginate_text_dict, READ_PARAMS as _READ_PARAMS

if TYPE_CHECKING:
    from ..sandbox.agsandbox import agSandbox

_DEFAULT_LIMIT = 2000
_MAX_BYTES = 50 * 1024
_MAX_LINE_LEN = 2000

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
READ_CHECK_TIMEOUT_S = 5  # Timeout in seconds for the shell command that checks whether a path is a file, directory, or missing.
READ_LS_TIMEOUT_S = 10  # Timeout in seconds for the ls command used to list a directory's entries inside the sandbox container.


def _paginate_text(content: str, offset: int, limit: int) -> agdata:
    """Apply offset/limit pagination to text content and return agdata --
    thin agdata wrapper over the shared, dependency-free
    `agtool_pure.paginate_text` (see that module's docstring for why the
    algorithm itself lives there)."""
    return agdata(**_paginate_text_dict(content, offset, limit))


def make_read(sandbox: "agSandbox") -> agtool:
    """Return a read tool that reads files from inside *sandbox*'s container."""

    def _run_sandboxed(arg: agdata) -> agdata:
        file_path = str(arg.file_path)  # type: ignore[arg-type]
        offset: int = int(getattr(arg, "offset", 1) or 1)
        limit: int = int(getattr(arg, "limit", _DEFAULT_LIMIT) or _DEFAULT_LIMIT)

        check_out, _ = sandbox._container_exec(
            f"if [ -d {shlex.quote(file_path)} ]; then echo dir; "
            f"elif [ -f {shlex.quote(file_path)} ]; then echo file; "
            f"else echo notfound; fi",
            timeout=READ_CHECK_TIMEOUT_S,
            shell="sh",
        )
        kind = check_out.strip()

        if kind == "notfound":
            return agerror(f"Not found: {file_path}")

        if kind == "dir":
            ls_out, _ = sandbox._container_exec(
                f"ls -1p {shlex.quote(file_path)}", timeout=READ_LS_TIMEOUT_S, shell="sh"
            )
            entries = sorted(ls_out.splitlines())
            start = offset - 1
            page = entries[start : start + limit]
            return agdata(
                path=file_path,
                type="directory",
                entries=page,
                total=len(entries),
                truncated=(start + len(page) < len(entries)),
            )

        try:
            content = sandbox.read_file(file_path)
        except FileNotFoundError:
            return agerror(f"Not found: {file_path}")
        except Exception as e:
            return agerror(format_exception(e))

        result = _paginate_text(content, offset, limit)
        return agdata(path=file_path, **result._data)

    def _log(tool: agtool, arg: agdata, result: agdata, elapsed_ms: int) -> None:
        if tool._term is None:
            return
        path = str(arg._data.get("file_path", "?"))
        rdata = result._data
        if "error" in rdata:
            tool._term.log("TOOL ✗   ", f"read  {path}  error: {rdata['error']}  ({elapsed_ms}ms)")
            if tool._aglog is not None:
                tool._aglog._tool_call(tool.name, arg.to_dict(), result.to_dict(), elapsed_ms)
            return
        kind = rdata.get("type", "file")
        lines = rdata.get("lines_shown", rdata.get("total", "?"))
        trunc = " [truncated]" if rdata.get("truncated", False) else ""
        tool._term.log(
            "TOOL ✓   ", f"read  {kind}  {path}  ({lines} lines{trunc})  ({elapsed_ms}ms)"
        )
        if tool._aglog is not None:
            tool._aglog._tool_call(tool.name, arg.to_dict(), result.to_dict(), elapsed_ms)

    return agtool(
        name="read",
        fn=_run_sandboxed,
        description="Read a file (with optional offset/limit) or list a directory inside the sandbox.",
        params=_READ_PARAMS,
        log_fn=_log,
        run_in_subprocess=False,
    )
