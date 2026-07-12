from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING
from ..agdata import agdata
from ..agtool import agtool

if TYPE_CHECKING:
    from ..agsandbox import agSandbox

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GREP_EXEC_TIMEOUT_SECS = (
    30  # Maximum seconds to wait for the rg command to complete inside the sandbox container.
)

_LIMIT = 100
_MAX_LINE_LEN = 2000

_GREP_PARAMS = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Regex pattern to search for"},
        "path": {
            "type": "string",
            "description": "File or directory to search (defaults to /workspace)",
        },
        "include": {"type": "string", "description": "File glob filter (e.g. '*.py')"},
    },
    "required": ["pattern"],
}


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

        output, _ = sandbox.exec(cmd, timeout=GREP_EXEC_TIMEOUT_SECS)

        matches: list[dict] = []
        for line in output.splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "match":
                continue
            data = obj["data"]
            matches.append(
                {
                    "path": data["path"]["text"],
                    "line": data["line_number"],
                    "text": data["lines"]["text"],
                }
            )

        truncated = len(matches) > _LIMIT
        matches = matches[:_LIMIT]
        for m in matches:
            if len(m["text"]) > _MAX_LINE_LEN:
                m["text"] = m["text"][:_MAX_LINE_LEN] + "..."

        return agdata(matches=matches, count=len(matches), truncated=truncated)

    def _log(tool: agtool, arg: agdata, result: agdata, elapsed_ms: int) -> None:
        if tool._term is None:
            return
        pattern = str(arg._data.get("pattern", "?"))
        rdata = result._data
        count = rdata.get("count", "?")
        trunc = " [truncated]" if rdata.get("truncated", False) else ""
        tool._term.log(
            "TOOL ✓   ", f"grep  {pattern!r}  → {count} matches{trunc}  ({elapsed_ms}ms)"
        )
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
        run_in_subprocess=False,
    )
