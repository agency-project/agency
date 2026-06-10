from __future__ import annotations

from typing import TYPE_CHECKING
from ..agdata import agdata
from ..agtool import agtool

if TYPE_CHECKING:
    from ..agsandbox import agSandbox

_WRITE_PARAMS = {
    "type": "object",
    "properties": {
        "filePath": {"type": "string", "description": "Absolute path to the file to write"},
        "content": {"type": "string", "description": "Content to write"},
    },
    "required": ["filePath", "content"],
}


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

    def _log(tool: agtool, arg: agdata, result: agdata, elapsed_ms: int) -> None:
        if tool._term is None:
            return
        path   = str(getattr(arg, "filePath", "?"))
        nbytes = getattr(result, "bytes_written", "?")
        tool._term.log("TOOL ✓   ", f"write  {path}  ({nbytes} bytes)  ({elapsed_ms}ms)")
        if tool._aglog is not None:
            tool._aglog._tool_call(tool.name, arg.to_dict(), result.to_dict(), elapsed_ms)

    return agtool(
        name="write",
        fn=_run_sandboxed,
        description="Write content to a file inside the sandbox, creating parent directories if needed.",
        params=_WRITE_PARAMS,
        log_fn=_log,
        need_sandbox=True,
    )
