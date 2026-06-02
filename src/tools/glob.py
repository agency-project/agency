from __future__ import annotations

import shlex
import subprocess
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from ..agdata import agdata
from ..agtool import agtool

if TYPE_CHECKING:
    from ..sandbox import agSandbox

_LIMIT = 100

_GLOB_PARAMS = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Glob pattern to match files against"},
        "path": {"type": "string", "description": "Directory to search (defaults to cwd)"},
    },
    "required": ["pattern"],
}


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


glob = agtool(
    name="glob",
    fn=_run,
    description="Find files matching a glob pattern in a directory tree.",
    params=_GLOB_PARAMS,
)


def make_glob(sandbox: "agSandbox") -> agtool:
    """Return a glob tool that searches for files inside *sandbox*'s container."""
    def _run_sandboxed(arg: agdata) -> agdata:
        pattern: str = str(arg.pattern)  # type: ignore[arg-type]
        path: str = str(getattr(arg, "path", "/workspace") or "/workspace")

        # Use rg --files if available, otherwise find
        output, rc = sandbox.exec(
            f"rg --files --glob {shlex.quote(pattern)} {shlex.quote(path)} 2>/dev/null "
            f"|| find {shlex.quote(path)} -name {shlex.quote(pattern)} -type f 2>/dev/null",
            timeout=30,
        )
        files = [line.strip() for line in output.splitlines() if line.strip()]
        truncated = len(files) > _LIMIT
        files = sorted(files[:_LIMIT])
        return agdata(files=files, count=len(files), truncated=truncated)

    return agtool(
        name="glob",
        fn=_run_sandboxed,
        description=(
            "Find files matching a glob pattern inside the sandbox. "
            "Defaults to searching /workspace."
        ),
        params=_GLOB_PARAMS,
    )
