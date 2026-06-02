from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING
from ..agdata import agdata
from ..agtool import agtool

if TYPE_CHECKING:
    from ..sandbox import agSandbox

_DEFAULT_LIMIT = 2000
_MAX_BYTES = 50 * 1024
_MAX_LINE_LEN = 2000

_READ_PARAMS = {
    "type": "object",
    "properties": {
        "filePath": {"type": "string", "description": "Absolute path to the file or directory"},
        "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)"},
        "limit": {"type": "integer", "description": "Maximum number of lines to read"},
    },
    "required": ["filePath"],
}


def _paginate_text(content: str, offset: int, limit: int) -> agdata:
    """Apply offset/limit pagination to text content and return agdata."""
    all_lines = content.splitlines(keepends=True)
    total = len(all_lines)
    start = offset - 1
    page_lines = all_lines[start: start + limit]

    raw: list[str] = []
    bytes_used = 0
    cut = False
    for i, line in enumerate(page_lines):
        text = line.rstrip("\n")
        if len(text) > _MAX_LINE_LEN:
            text = text[:_MAX_LINE_LEN] + "... (truncated)"
        size = len(text.encode()) + 1
        if bytes_used + size > _MAX_BYTES:
            cut = True
            break
        raw.append(f"{start + i + 1}: {text}")
        bytes_used += size

    more = cut or (start + len(page_lines) < total)
    return agdata(
        type="file",
        content="\n".join(raw),
        offset=offset,
        lines_shown=len(raw),
        total_lines=total,
        truncated=more,
    )


def _run(arg: agdata) -> agdata:
    file_path = Path(str(arg.filePath))  # type: ignore[arg-type]
    offset: int = int(getattr(arg, "offset", 1) or 1)
    limit: int = int(getattr(arg, "limit", _DEFAULT_LIMIT) or _DEFAULT_LIMIT)

    if not file_path.exists():
        return agdata(error=f"Not found: {file_path}")

    if file_path.is_dir():
        entries = sorted(
            e.name + ("/" if e.is_dir() else "") for e in file_path.iterdir()
        )
        start = offset - 1
        page = entries[start : start + limit]
        truncated = start + len(page) < len(entries)
        return agdata(
            path=str(file_path),
            type="directory",
            entries=page,
            total=len(entries),
            truncated=truncated,
        )

    try:
        with open(file_path, "r", errors="replace") as f:
            all_lines = f.readlines()
    except Exception as e:
        return agdata(error=str(e))

    total = len(all_lines)
    start = offset - 1
    page_lines = all_lines[start : start + limit]

    raw: list[str] = []
    bytes_used = 0
    cut = False
    for i, line in enumerate(page_lines):
        text = line.rstrip("\n")
        if len(text) > _MAX_LINE_LEN:
            text = text[:_MAX_LINE_LEN] + "... (truncated)"
        size = len(text.encode()) + 1
        if bytes_used + size > _MAX_BYTES:
            cut = True
            break
        raw.append(f"{start + i + 1}: {text}")
        bytes_used += size

    more = cut or (start + len(page_lines) < total)
    content = "\n".join(raw)
    return agdata(
        path=str(file_path),
        type="file",
        content=content,
        offset=offset,
        lines_shown=len(raw),
        total_lines=total,
        truncated=more,
    )


read = agtool(
    name="read",
    fn=_run,
    description="Read a file (with optional offset/limit) or list a directory.",
    params=_READ_PARAMS,
)


def make_read(sandbox: "agSandbox") -> agtool:
    """Return a read tool that reads files from inside *sandbox*'s container."""
    def _run_sandboxed(arg: agdata) -> agdata:
        file_path = str(arg.filePath)  # type: ignore[arg-type]
        offset: int = int(getattr(arg, "offset", 1) or 1)
        limit: int = int(getattr(arg, "limit", _DEFAULT_LIMIT) or _DEFAULT_LIMIT)

        # Determine if path is a directory or file
        check_out, _ = sandbox._container_exec(
            f"if [ -d {shlex.quote(file_path)} ]; then echo dir; "
            f"elif [ -f {shlex.quote(file_path)} ]; then echo file; "
            f"else echo notfound; fi",
            timeout=5, shell="sh",
        )
        kind = check_out.strip()

        if kind == "notfound":
            return agdata(error=f"Not found: {file_path}")

        if kind == "dir":
            ls_out, _ = sandbox._container_exec(
                f"ls -1p {shlex.quote(file_path)}", timeout=10, shell="sh"
            )
            entries = sorted(ls_out.splitlines())
            start = offset - 1
            page = entries[start: start + limit]
            return agdata(
                path=file_path,
                type="directory",
                entries=page,
                total=len(entries),
                truncated=(start + len(page) < len(entries)),
            )

        # File: read content and paginate
        try:
            content = sandbox.read_file(file_path)
        except FileNotFoundError:
            return agdata(error=f"Not found: {file_path}")
        except Exception as e:
            return agdata(error=str(e))

        result = _paginate_text(content, offset, limit)
        return agdata(path=file_path, **result._data)

    return agtool(
        name="read",
        fn=_run_sandboxed,
        description="Read a file (with optional offset/limit) or list a directory inside the sandbox.",
        params=_READ_PARAMS,
    )
