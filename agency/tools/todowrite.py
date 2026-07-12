import json
from ..agdata import agdata, agerror
from ..agtool import agtool

# Module-level todo store (keyed by a session id or default)
_store: list[dict] = []


def _run(arg: agdata) -> agdata:
    global _store
    todos = getattr(arg, "todos", None)
    if todos is None:
        return agerror("todos field is required")
    if not isinstance(todos, list):
        return agerror("todos must be a list")

    _store = [dict(t) for t in todos]
    pending = sum(1 for t in _store if t.get("status") not in ("completed", "cancelled"))
    return agdata(
        todos=_store,
        count=len(_store),
        pending=pending,
        output=json.dumps(_store, indent=2),
    )


todowrite = agtool(
    name="todowrite",
    fn=_run,
    run_in_subprocess=False,
    description="Update the todo list with a new set of items.",
    params={
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The updated todo list",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Task description"},
                        "status": {
                            "type": "string",
                            "description": "pending | in_progress | completed | cancelled",
                        },
                        "priority": {"type": "string", "description": "high | medium | low"},
                    },
                    "required": ["content", "status", "priority"],
                },
            }
        },
        "required": ["todos"],
    },
)
