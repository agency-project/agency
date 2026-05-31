import subprocess
from ..agdata import agdata
from ..tool import tool

_MAX_BYTES = 50 * 1024


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


bash = tool(
    name="bash",
    fn=_run,
    description="Run a shell command and return its output.",
    params={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to execute"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)"},
            "workdir": {"type": "string", "description": "Working directory (optional)"},
        },
        "required": ["command"],
    },
)
