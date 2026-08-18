from __future__ import annotations

from typing import TYPE_CHECKING
from ..agdata import agdata
from ..agtool import agtool
from ..agtool_pure import grep_command, parse_grep_json_output, GREP_PARAMS as _GREP_PARAMS

if TYPE_CHECKING:
    from ..sandbox.agsandbox import agSandbox

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GREP_EXEC_TIMEOUT_SECS = (
    30  # Maximum seconds to wait for the rg command to complete inside the sandbox container.
)


def make_grep(sandbox: "agSandbox") -> agtool:
    """Return a grep tool that searches file contents inside *sandbox*'s container."""

    def _run_sandboxed(arg: agdata) -> agdata:
        pattern: str = str(arg.pattern)  # type: ignore[arg-type]
        path: str = str(getattr(arg, "path", "/workspace") or "/workspace")
        include: str | None = getattr(arg, "include", None)

        output, _ = sandbox.exec(
            grep_command(pattern, path, include), timeout=GREP_EXEC_TIMEOUT_SECS
        )
        return agdata(**parse_grep_json_output(output))

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
