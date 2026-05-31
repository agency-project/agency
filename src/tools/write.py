from pathlib import Path
from ..agdata import agdata
from ..tool import tool


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


write = tool(
    name="write",
    fn=_run,
    description="Write content to a file, creating parent directories if needed.",
    params={
        "type": "object",
        "properties": {
            "filePath": {"type": "string", "description": "Absolute path to the file to write"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["filePath", "content"],
    },
)
