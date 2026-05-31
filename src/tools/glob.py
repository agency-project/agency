import subprocess
import shutil
from pathlib import Path
from ..agdata import agdata
from ..tool import tool

_LIMIT = 100


def _run(arg: agdata) -> agdata:
    pattern: str = str(arg.pattern)  # type: ignore[arg-type]
    root = Path(str(getattr(arg, "path", ".") or "."))
    if not root.is_absolute():
        root = Path.cwd() / root

    if not root.exists():
        return agdata(error=f"Directory not found: {root}")

    # Prefer ripgrep for speed; fall back to pathlib
    if shutil.which("rg"):
        try:
            result = subprocess.run(
                ["rg", "--files", "--glob", pattern, str(root)],
                capture_output=True, text=True, timeout=30,
            )
            files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except Exception:
            files = []
    else:
        files = [str(p) for p in root.rglob(pattern)]

    # Sort by mtime descending
    def mtime(p: str) -> float:
        try:
            return Path(p).stat().st_mtime
        except OSError:
            return 0.0

    files.sort(key=mtime, reverse=True)
    truncated = len(files) > _LIMIT
    files = files[:_LIMIT]

    return agdata(
        files=files,
        count=len(files),
        truncated=truncated,
    )


glob = tool(
    name="glob",
    fn=_run,
    description="Find files matching a glob pattern in a directory tree.",
    params={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern to match files against"},
            "path": {"type": "string", "description": "Directory to search (defaults to cwd)"},
        },
        "required": ["pattern"],
    },
)
