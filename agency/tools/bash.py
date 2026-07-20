from __future__ import annotations

from typing import TYPE_CHECKING
from ..agdata import agdata
from ..agtool import agtool

if TYPE_CHECKING:
    from ..agsandbox import agSandbox

_MAX_BYTES = 50 * 1024

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASH_DEFAULT_TIMEOUT_SECONDS = (
    120  # Default timeout (seconds) for a bash tool invocation when the caller does not supply one
)
BASH_LOG_CMD_MAX_CHARS = 100  # Maximum characters of the shell command shown in the log line to avoid flooding the terminal

_BASH_PARAMS = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "The shell command to execute"},
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds (default 120). For long-running commands pass this here — do NOT use the shell timeout command, which has no effect on the tool watchdog.",
        },
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
        timeout: int = getattr(arg, "timeout", BASH_DEFAULT_TIMEOUT_SECONDS)
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
        cmd = str(arg._data.get("command", ""))[:BASH_LOG_CMD_MAX_CHARS]
        rdata = result._data
        rc = rdata.get("exit_code", "?")
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
            "or are released via daemon_release. "
            "IMPORTANT: if the command uses a GPU (PyTorch, CUDA, model loading, etc.), "
            "you MUST call reserve_gpu first — otherwise CUDA will not be visible and the "
            "command will fail. The GPU is held automatically for the rest of this "
            "sandbox's lifetime; there is no release step."
        ),
        params=_BASH_PARAMS,
        log_fn=_log,
        run_in_subprocess=False,
    )
