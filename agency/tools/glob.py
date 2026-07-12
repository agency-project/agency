from __future__ import annotations

import shlex
from typing import TYPE_CHECKING
from ..agdata import agdata
from ..agtool import agtool

if TYPE_CHECKING:
    from ..agsandbox import agSandbox

_LIMIT = 100

_GLOB_PARAMS = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Glob pattern to match files against"},
        "path": {"type": "string", "description": "Directory to search (defaults to /workspace)"},
    },
    "required": ["pattern"],
}


def make_glob(sandbox: "agSandbox") -> agtool:
    """Return a glob tool that searches for files inside *sandbox*'s container."""

    def _run_sandboxed(arg: agdata) -> agdata:
        pattern: str = str(arg.pattern)  # type: ignore[arg-type]
        path: str = str(getattr(arg, "path", "/workspace") or "/workspace")

        output, _ = sandbox.exec(
            f"rg --files --glob {shlex.quote(pattern)} {shlex.quote(path)} 2>/dev/null "
            f"|| find {shlex.quote(path)} -name {shlex.quote(pattern)} -type f 2>/dev/null",
            timeout=30,
        )
        files = [line.strip() for line in output.splitlines() if line.strip()]
        truncated = len(files) > _LIMIT
        files = sorted(files[:_LIMIT])
        return agdata(files=files, count=len(files), truncated=truncated)

    def _log(tool: agtool, arg: agdata, result: agdata, elapsed_ms: int) -> None:
        if tool._term is None:
            return
        pattern = str(arg._data.get("pattern", "?"))
        rdata = result._data
        count = rdata.get("count", "?")
        trunc = " [truncated]" if rdata.get("truncated", False) else ""
        tool._term.log("TOOL ✓   ", f"glob  {pattern!r}  → {count} files{trunc}  ({elapsed_ms}ms)")
        if tool._aglog is not None:
            tool._aglog._tool_call(tool.name, arg.to_dict(), result.to_dict(), elapsed_ms)

    return agtool(
        name="glob",
        fn=_run_sandboxed,
        description=(
            "Find files matching a glob pattern inside the sandbox. "
            "Defaults to searching /workspace."
        ),
        params=_GLOB_PARAMS,
        log_fn=_log,
        run_in_subprocess=False,
    )
