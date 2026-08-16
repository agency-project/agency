"""Built-in tools for the standalone native harness: bash, read, write,
edit, glob, grep, webfetch, todowrite.

Ported from the old `_native_in_container_entrypoint.py` (same
implementation, same schemas, sourced from the same `agtool_pure.py` this
package loads by path -- see `pure_loader.py`'s docstring for why that's
reuse, not duplication). Plain `subprocess`/file I/O throughout: this
process already runs inside whatever filesystem it's launched in (a
sandbox container, or a user's own machine for a fully standalone run), so
there is no `sandbox.exec()`/agtool bridge to reuse and no reason to invent
one.

Per E.'s decision (see the conversation this package came out of): these
built-ins ship with the harness itself, always available. Anything beyond
them goes through MCP (`mcp_client.py`), never a second, harness-specific
extension mechanism."""

from __future__ import annotations

import json
import os
import subprocess
import uuid

from .pure_loader import load_agtool_pure

_agtool_pure = load_agtool_pure()
_READ_DEFAULT_LIMIT = _agtool_pure.READ_DEFAULT_LIMIT

_BASH_TIMEOUT_S = 120
_WEBFETCH_MAX_BYTES = 5 * 1024 * 1024
_WEBFETCH_DEFAULT_TIMEOUT = 30
_WEBFETCH_MAX_TIMEOUT = 120

# Same default as agtool.py's `_AgToolFields.output_offload_chars` (the
# host-side dispatch_tools()'s own threshold) -- kept as a plain constant
# since this package can't import agconfig (see pure_loader.py's docstring
# on why it avoids the `agency.*` import chain entirely).
_TOOL_OUTPUT_OFFLOAD_CHARS = 40_000


def _parse_tool_args(arguments_json: str) -> dict:
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except (json.JSONDecodeError, TypeError):
        args = {}
    return args if isinstance(args, dict) else {}


def _run_bash_tool(arguments_json: str) -> str:
    args = _parse_tool_args(arguments_json)
    command = args.get("command", "") if isinstance(args, dict) else ""
    try:
        proc = subprocess.run(
            ["bash", "-c", command], capture_output=True, timeout=_BASH_TIMEOUT_S, text=True
        )
        output = proc.stdout + proc.stderr
        return json.dumps({"output": output, "returncode": proc.returncode})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"command timed out after {_BASH_TIMEOUT_S}s"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _run_read_tool(arguments_json: str) -> str:
    args = _parse_tool_args(arguments_json)
    file_path = str(args.get("file_path", ""))
    offset = int(args.get("offset") or 1)
    limit = int(args.get("limit") or _READ_DEFAULT_LIMIT)

    if os.path.isdir(file_path):
        try:
            entries = sorted(os.listdir(file_path))
        except OSError as e:
            return json.dumps({"error": str(e)})
        start = offset - 1
        page = entries[start : start + limit]
        return json.dumps(
            {
                "path": file_path,
                "type": "directory",
                "entries": page,
                "total": len(entries),
                "truncated": start + len(page) < len(entries),
            }
        )
    if not os.path.isfile(file_path):
        return json.dumps({"error": f"Not found: {file_path}"})
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return json.dumps({"error": str(e)})
    result = _agtool_pure.paginate_text(content, offset, limit)
    result["path"] = file_path
    return json.dumps(result)


def _run_write_tool(arguments_json: str) -> str:
    args = _parse_tool_args(arguments_json)
    file_path = str(args.get("file_path", ""))
    content = str(args.get("content", ""))
    try:
        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return json.dumps(
            {"path": file_path, "created": True, "bytes_written": len(content.encode())}
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


def _run_edit_tool(arguments_json: str) -> str:
    args = _parse_tool_args(arguments_json)
    file_path = str(args.get("file_path", ""))
    old_string = str(args.get("old_string", ""))
    new_string = str(args.get("new_string", ""))
    replace_all = bool(args.get("replace_all", False))

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {file_path}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

    try:
        updated = _agtool_pure.replace(content, old_string, new_string, replace_all)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated)
        return json.dumps({"path": file_path, "success": True})
    except (ValueError, OSError) as e:
        return json.dumps({"error": str(e)})


def _run_glob_tool(arguments_json: str) -> str:
    args = _parse_tool_args(arguments_json)
    pattern = str(args.get("pattern", ""))
    path = str(args.get("path") or ".")
    try:
        proc = subprocess.run(
            ["bash", "-c", _agtool_pure.glob_command(pattern, path)],
            capture_output=True,
            timeout=30,
            text=True,
        )
        return json.dumps(_agtool_pure.parse_glob_output(proc.stdout))
    except Exception as e:
        return json.dumps({"error": str(e)})


def _run_grep_tool(arguments_json: str) -> str:
    args = _parse_tool_args(arguments_json)
    pattern = str(args.get("pattern", ""))
    path = str(args.get("path") or ".")
    include = args.get("include")
    try:
        proc = subprocess.run(
            ["bash", "-c", _agtool_pure.grep_command(pattern, path, include)],
            capture_output=True,
            timeout=30,
            text=True,
        )
        return json.dumps(_agtool_pure.parse_grep_json_output(proc.stdout))
    except Exception as e:
        return json.dumps({"error": str(e)})


def _run_webfetch_tool(arguments_json: str) -> str:
    import httpx
    import html2text

    args = _parse_tool_args(arguments_json)
    url = str(args.get("url", ""))
    fmt = str(args.get("format") or "markdown")
    timeout = min(int(args.get("timeout") or _WEBFETCH_DEFAULT_TIMEOUT), _WEBFETCH_MAX_TIMEOUT)

    if not url.startswith(("http://", "https://")):
        return json.dumps({"error": "URL must start with http:// or https://"})

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; agency-bot/1.0)",
        "Accept": "text/html,text/plain,*/*;q=0.8",
    }
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"HTTP {e.response.status_code}: {url}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

    if len(resp.content) > _WEBFETCH_MAX_BYTES:
        return json.dumps({"error": "Response too large (>5 MB)"})

    content_type = resp.headers.get("content-type", "")
    body = resp.text
    if fmt == "markdown" and "text/html" in content_type:
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.body_width = 0
        output = h.handle(body)
    elif fmt == "text" and "text/html" in content_type:
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.body_width = 0
        output = h.handle(body)
    else:
        output = body
    return json.dumps({"url": url, "content_type": content_type, "output": output})


# Module-level store: this process belongs to exactly one agent/run (unlike
# the host-side todowrite tool's real, pre-existing shared-process bug), so
# a module-level list here is correct, not a bug waiting to bite a second
# concurrent run.
_todo_store: list = []


def _run_todowrite_tool(arguments_json: str) -> str:
    global _todo_store
    args = _parse_tool_args(arguments_json)
    todos = args.get("todos")
    if todos is None:
        return json.dumps({"error": "todos field is required"})
    if not isinstance(todos, list):
        return json.dumps({"error": "todos must be a list"})
    _todo_store = [dict(t) for t in todos]
    pending = sum(1 for t in _todo_store if t.get("status") not in ("completed", "cancelled"))
    return json.dumps(
        {
            "todos": _todo_store,
            "count": len(_todo_store),
            "pending": pending,
            "output": json.dumps(_todo_store, indent=2),
        }
    )


def offload_if_oversized(fn_name: str, tc_id: str, result_content: str, offload_dir: str) -> str:
    """Mirrors `agtool.py`'s `dispatch_tools()` offload behavior, generic
    across every tool (built-in or MCP): a result over the threshold is
    saved whole to a file and replaced with a short note pointing at it, so
    the model can `read` it instead of the full content bloating every
    subsequent turn's context."""
    if len(result_content) <= _TOOL_OUTPUT_OFFLOAD_CHARS:
        return result_content
    try:
        parsed = json.loads(result_content)
        file_body = (
            parsed.get("content", result_content) if isinstance(parsed, dict) else result_content
        )
    except json.JSONDecodeError:
        file_body = result_content
    if not isinstance(file_body, str):
        file_body = json.dumps(file_body)
    safe_id = tc_id.replace("-", "")[:12]
    offload_path = f"{offload_dir}/{fn_name}_{safe_id or uuid.uuid4().hex[:12]}.txt"
    try:
        os.makedirs(os.path.dirname(offload_path), exist_ok=True)
        with open(offload_path, "w", encoding="utf-8") as f:
            f.write(file_body)
        return json.dumps(
            {
                "note": f"Output was too large and has been saved to {offload_path}. "
                "Use the read tool to access it."
            }
        )
    except Exception as e:
        print(
            f"[native_harness] WARNING: failed to offload large tool output to {offload_path}: {e}"
        )
        return result_content


def _tool_schema(name: str, description: str, params: dict) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": params},
    }


TOOL_DISPATCH = {
    "bash": _run_bash_tool,
    "read": _run_read_tool,
    "write": _run_write_tool,
    "edit": _run_edit_tool,
    "glob": _run_glob_tool,
    "grep": _run_grep_tool,
    "webfetch": _run_webfetch_tool,
    "todowrite": _run_todowrite_tool,
}

# Same name/params/description as the host-side sandboxed tools
# (agency/tools/*.py), sourced from the same `agtool_pure` module they
# themselves import their PARAMS dicts from -- one place defines what these
# tools look like to a model, not two copies that could drift.
BUILTIN_TOOL_SCHEMAS = {
    "bash": _tool_schema(
        "bash", "Execute a bash command and return its output.", _agtool_pure.BASH_PARAMS
    ),
    "read": _tool_schema(
        "read",
        "Read a file (with optional offset/limit) or list a directory.",
        _agtool_pure.READ_PARAMS,
    ),
    "write": _tool_schema(
        "write",
        "Write content to a file, creating parent directories if needed.",
        _agtool_pure.WRITE_PARAMS,
    ),
    "edit": _tool_schema(
        "edit",
        "Replace a string in a file. Uses fuzzy matching as fallback.",
        _agtool_pure.EDIT_PARAMS,
    ),
    "glob": _tool_schema(
        "glob",
        "Find files matching a glob pattern. Defaults to the current directory.",
        _agtool_pure.GLOB_PARAMS,
    ),
    "grep": _tool_schema(
        "grep",
        "Search for a regex pattern in file contents. Defaults to the current directory.",
        _agtool_pure.GREP_PARAMS,
    ),
    "webfetch": _tool_schema(
        "webfetch",
        "Fetch a URL and return its content as text, markdown, or raw HTML.",
        _agtool_pure.WEBFETCH_PARAMS,
    ),
    "todowrite": _tool_schema(
        "todowrite", "Update the todo list with a new set of items.", _agtool_pure.TODOWRITE_PARAMS
    ),
}


__all__ = ["TOOL_DISPATCH", "BUILTIN_TOOL_SCHEMAS", "offload_if_oversized"]
