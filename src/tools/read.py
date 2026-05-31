import os
from pathlib import Path
from ..agdata import agdata
from ..tool import tool

_DEFAULT_LIMIT = 2000
_MAX_BYTES = 50 * 1024
_MAX_LINE_LEN = 2000


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


read = tool(
    name="read",
    fn=_run,
    description="Read a file (with optional offset/limit) or list a directory.",
    params={
        "type": "object",
        "properties": {
            "filePath": {"type": "string", "description": "Absolute path to the file or directory"},
            "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)"},
            "limit": {"type": "integer", "description": "Maximum number of lines to read"},
        },
        "required": ["filePath"],
    },
)
