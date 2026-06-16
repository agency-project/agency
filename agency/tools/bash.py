from __future__ import annotations

from typing import TYPE_CHECKING
from ..agdata import agdata
from ..agtool import agtool

if TYPE_CHECKING:
    from ..agsandbox import agSandbox

_MAX_BYTES = 50 * 1024

_BASH_PARAMS = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "The shell command to execute"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)"},
        "workdir": {"type": "string", "description": "Working directory (optional)"},
    },
    "required": ["command"],
}


def make_bash(sandbox: "agSandbox") -> agtool:
    """Return a bash tool that executes commands inside *sandbox*'s container.

    All spawned processes are tracked via /proc diff.  The agent's skill future
    will not resolve until all such processes exit or are released as daemons.
    The default working directory is ``/workspace`` inside the container.
    """
    def _run_sandboxed(arg: agdata) -> agdata:
        command: str = arg.command  # type: ignore[assignment]
        timeout: int = getattr(arg, "timeout", 120)
        workdir: str = getattr(arg, "workdir", "/workspace") or "/workspace"

        output, rc = sandbox.exec(command, workdir=workdir, timeout=timeout)

        truncated = False
        if len(output.encode()) > _MAX_BYTES:
            output = "...output truncated...\n\n" + output[-_MAX_BYTES:]
            truncated = True
        return agdata(output=output, exit_code=rc, truncated=truncated)

    def _log(tool: agtool, arg: agdata, result: agdata, elapsed_ms: int) -> None:
        if tool._term is None:
            return
        cmd   = str(arg._data.get("command", ""))[:100]
        rdata = result._data
        rc    = rdata.get("exit_code", "?")
        trunc = " [truncated]" if rdata.get("truncated", False) else ""
        tool._term.log("TOOL ✓   ", f"bash  rc={rc}  ({elapsed_ms}ms){trunc}  $ {cmd}")
        if tool._aglog is not None:
            tool._aglog._tool_call(tool.name, arg.to_dict(), result.to_dict(), elapsed_ms)

    return agtool(
        name="bash",
        fn=_run_sandboxed,
        description=(
            "Run a shell command inside the agent's sandbox container and return its output. "
            "The default working directory is /workspace. "
            "All spawned processes are tracked; the skill will not finish until they exit "
            "or are released via daemon_release."
        ),
        params=_BASH_PARAMS,
        log_fn=_log,
        need_sandbox=True,
    )
