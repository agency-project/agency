from __future__ import annotations

import json
import re
import shlex
import subprocess
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from ..agdata import agdata
from ..agtool import agtool

if TYPE_CHECKING:
    from ..agsandbox import agSandbox

_LIMIT = 100
_MAX_LINE_LEN = 2000

_GREP_PARAMS = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Regex pattern to search for"},
        "path": {"type": "string", "description": "File or directory to search (defaults to cwd)"},
        "include": {"type": "string", "description": "File glob filter (e.g. '*.py')"},
    },
    "required": ["pattern"],
}


def _grep_with_rg(pattern: str, path: Path, include: str | None) -> list[dict]:
    cmd = ["rg", "--json", "--no-ignore", pattern, str(path)]
    if include:
        cmd += ["--glob", include]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:
        return []
    matches: list[dict] = []
    for line in result.stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "match":
            continue
        data = obj["data"]
        file_path = data["path"]["text"]
        line_num = data["line_number"]
        text = data["lines"]["text"]
        matches.append({"path": file_path, "line": line_num, "text": text})
    return matches


def _grep_with_re(pattern: str, path: Path, include: str | None) -> list[dict]:
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return [{"path": "", "line": 0, "text": f"Invalid regex: {e}"}]

    matches: list[dict] = []
    if path.is_file():
        files: list[Path] = [path]
    else:
        files = [p for p in path.rglob(include or "*") if p.is_file()]

    for file in files:
        try:
            for i, line in enumerate(file.read_text(errors="replace").splitlines(), 1):
                if regex.search(line):
                    matches.append({"path": str(file), "line": i, "text": line})
        except OSError:
            continue
    return matches


def _run(arg: agdata) -> agdata:
    pattern: str = str(arg.pattern)  # type: ignore[arg-type]
    search_path = Path(str(getattr(arg, "path", ".") or "."))
    include: str | None = getattr(arg, "include", None)

    if not search_path.is_absolute():
        search_path = Path.cwd() / search_path

    if shutil.which("rg"):
        matches = _grep_with_rg(pattern, search_path, include)
    else:
        matches = _grep_with_re(pattern, search_path, include)

    # Sort by file mtime descending
    def mtime(m: dict) -> float:
        try:
            return Path(m["path"]).stat().st_mtime
        except OSError:
            return 0.0

    matches.sort(key=mtime, reverse=True)
    truncated = len(matches) > _LIMIT
    matches = matches[:_LIMIT]

    for m in matches:
        if len(m["text"]) > _MAX_LINE_LEN:
            m["text"] = m["text"][:_MAX_LINE_LEN] + "..."

    return agdata(
        matches=matches,
        count=len(matches),
        truncated=truncated,
    )


grep = agtool(
    name="grep",
    fn=_run,
    description="Search for a regex pattern in file contents.",
    params=_GREP_PARAMS,
)


def make_grep(sandbox: "agSandbox") -> agtool:
    """Return a grep tool that searches file contents inside *sandbox*'s container."""
    def _run_sandboxed(arg: agdata) -> agdata:
        pattern: str = str(arg.pattern)  # type: ignore[arg-type]
        path: str = str(getattr(arg, "path", "/workspace") or "/workspace")
        include: str | None = getattr(arg, "include", None)

        cmd = f"rg --json --no-ignore {shlex.quote(pattern)} {shlex.quote(path)}"
        if include:
            cmd += f" --glob {shlex.quote(include)}"
        cmd += " 2>/dev/null || true"

        output, _ = sandbox.exec(cmd, timeout=30)

        matches: list[dict] = []
        for line in output.splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "match":
                continue
            data = obj["data"]
            matches.append({
                "path": data["path"]["text"],
                "line": data["line_number"],
                "text": data["lines"]["text"],
            })

        truncated = len(matches) > _LIMIT
        matches = matches[:_LIMIT]
        for m in matches:
            if len(m["text"]) > _MAX_LINE_LEN:
                m["text"] = m["text"][:_MAX_LINE_LEN] + "..."

        return agdata(matches=matches, count=len(matches), truncated=truncated)

    def _log(tool: agtool, arg: agdata, result: agdata, elapsed_ms: int) -> None:
        if tool._term is None:
            return
        pattern = str(getattr(arg, "pattern", "?"))
        count   = getattr(result, "count", "?")
        trunc   = " [truncated]" if getattr(result, "truncated", False) else ""
        tool._term.log("TOOL ✓   ", f"grep  {pattern!r}  → {count} matches{trunc}  ({elapsed_ms}ms)")
        if tool._aglog is not None:
            tool._aglog._tool_call(tool.name, arg.to_dict(), result.to_dict(), elapsed_ms)

    return agtool(
        name="grep",
        fn=_run_sandboxed,
        description=(
            "Search for a regex pattern in file contents inside the sandbox. "
            "Defaults to searching /workspace."
        ),
        params=_GREP_PARAMS,
        log_fn=_log,
    )
