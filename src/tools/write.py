from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from ..agdata import agdata
from ..agtool import agtool

if TYPE_CHECKING:
    from ..sandbox import agSandbox

_WRITE_PARAMS = {
    "type": "object",
    "properties": {
        "filePath": {"type": "string", "description": "Absolute path to the file to write"},
        "content": {"type": "string", "description": "Content to write"},
    },
    "required": ["filePath", "content"],
}


def _run(arg: agdata) -> agdata:
    file_path = Path(str(arg.filePath))  # type: ignore[arg-type]
    content: str = str(arg.content)  # type: ignore[arg-type]

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        existed = file_path.exists()
        file_path.write_text(content, encoding="utf-8")
        return agdata(
            path=str(file_path),
            created=not existed,
            bytes_written=len(content.encode()),
        )
    except Exception as e:
        return agdata(error=str(e))


write = agtool(
    name="write",
    fn=_run,
    description="Write content to a file, creating parent directories if needed.",
    params=_WRITE_PARAMS,
)


def make_write(sandbox: "agSandbox") -> agtool:
    """Return a write tool that writes files inside *sandbox*'s container."""
    def _run_sandboxed(arg: agdata) -> agdata:
        file_path = str(arg.filePath)  # type: ignore[arg-type]
        content: str = str(arg.content)  # type: ignore[arg-type]

        try:
            sandbox.write_file(file_path, content)
            return agdata(
                path=file_path,
                created=True,
                bytes_written=len(content.encode()),
            )
        except Exception as e:
            return agdata(error=str(e))

    return agtool(
        name="write",
        fn=_run_sandboxed,
        description="Write content to a file inside the sandbox, creating parent directories if needed.",
        params=_WRITE_PARAMS,
    )
