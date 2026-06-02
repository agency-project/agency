from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from ..agdata import agdata
from ..agtool import agtool

if TYPE_CHECKING:
    from ..sandbox import agSandbox

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


def _run(arg: agdata) -> agdata:
    command: str = arg.command  # type: ignore[assignment]
    timeout: int = getattr(arg, "timeout", 120)
    workdir: str | None = getattr(arg, "workdir", None)

    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )
        output = proc.stdout + proc.stderr
        truncated = False
        if len(output.encode()) > _MAX_BYTES:
            output = "...output truncated...\n\n" + output[-_MAX_BYTES:]
            truncated = True
        return agdata(output=output, exit_code=proc.returncode, truncated=truncated)
    except subprocess.TimeoutExpired:
        return agdata(output=f"Command timed out after {timeout}s", exit_code=-1, truncated=False)
    except Exception as e:
        return agdata(output=str(e), exit_code=-1, truncated=False)


bash = agtool(
    name="bash",
    fn=_run,
    description="Run a shell command and return its output.",
    params=_BASH_PARAMS,
)


def make_bash(sandbox: "agSandbox") -> agtool:
    """Return a bash tool that executes commands inside *sandbox*'s container.

    Background processes started with ``&`` are automatically tracked.
    The agent's skill future will not resolve until all such processes exit.
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

    return agtool(
        name="bash",
        fn=_run_sandboxed,
        description=(
            "Run a shell command inside the agent's sandbox container and return its output. "
            "The default working directory is /workspace. "
            "Commands run with & are tracked; the skill will not finish until they exit."
        ),
        params=_BASH_PARAMS,
    )
