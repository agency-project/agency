"""Built-in tools for the standalone tandem harness: bash, read, write,
edit, glob, grep, webfetch, todowrite.

Plain `subprocess`/file I/O throughout: this process already runs inside
whatever filesystem it's launched in (a sandbox container, or a user's own
machine for a fully standalone run), so there is no `sandbox.exec()`/agtool
bridge to reuse and no reason to invent one.

Per E.'s decision (see the conversation this package came out of): these
built-ins ship with the harness itself, always available. Anything beyond
them goes through MCP (`mcp_client.py`), never a second, harness-specific
extension mechanism."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import uuid
from typing import Generator

_BASH_TIMEOUT_S = 120
_WEBFETCH_MAX_BYTES = 5 * 1024 * 1024
_WEBFETCH_DEFAULT_TIMEOUT = 30
_WEBFETCH_MAX_TIMEOUT = 120

# Same default as agconfig's `output_offload_chars` field (the host-side
# dispatch_tools()'s own threshold) -- kept as a plain constant since this
# package can't import agconfig (see this package's own
# `__init__.py` docstring on why it avoids the `agency.*` import chain
# entirely).
_TOOL_OUTPUT_OFFLOAD_CHARS = 40_000


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling `parameters` shape) -- single
# source of truth for this harness's own tool schemas.
# ---------------------------------------------------------------------------

BASH_PARAMS = {
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

READ_PARAMS = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Absolute path to the file or directory"},
        "offset": {
            "type": "integer",
            "description": "Line number to start reading from (1-indexed)",
        },
        "limit": {"type": "integer", "description": "Maximum number of lines to read"},
    },
    "required": ["file_path"],
}

WRITE_PARAMS = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Absolute path to the file to write"},
        "content": {"type": "string", "description": "Content to write"},
    },
    "required": ["file_path", "content"],
}

EDIT_PARAMS = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Absolute path to the file to edit"},
        "old_string": {"type": "string", "description": "The text to replace"},
        "new_string": {"type": "string", "description": "The replacement text"},
        "replace_all": {
            "type": "boolean",
            "description": "Replace all occurrences (default false)",
        },
    },
    "required": ["file_path", "old_string", "new_string"],
}

GLOB_PARAMS = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Glob pattern to match files against"},
        "path": {"type": "string", "description": "Directory to search (defaults to /workspace)"},
    },
    "required": ["pattern"],
}

GREP_PARAMS = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Regex pattern to search for"},
        "path": {
            "type": "string",
            "description": "File or directory to search (defaults to /workspace)",
        },
        "include": {"type": "string", "description": "File glob filter (e.g. '*.py')"},
    },
    "required": ["pattern"],
}

WEBFETCH_PARAMS = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "URL to fetch (must be http:// or https://)"},
        "format": {
            "type": "string",
            "enum": ["markdown", "text", "html"],
            "description": "Output format (default: markdown)",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds (max 120, default 30)",
        },
    },
    "required": ["url"],
}

TODOWRITE_PARAMS = {
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
}


# ---------------------------------------------------------------------------
# glob / grep: shell command construction + output parsing
# ---------------------------------------------------------------------------

GLOB_LIMIT = 100
GREP_LIMIT = 100
GREP_MAX_LINE_LEN = 2000


def glob_command(pattern: str, path: str) -> str:
    return (
        f"rg --files --glob {shlex.quote(pattern)} {shlex.quote(path)} 2>/dev/null "
        f"|| find {shlex.quote(path)} -name {shlex.quote(pattern)} -type f 2>/dev/null"
    )


def parse_glob_output(output: str) -> dict:
    files = [line.strip() for line in output.splitlines() if line.strip()]
    truncated = len(files) > GLOB_LIMIT
    files = sorted(files[:GLOB_LIMIT])
    return {"files": files, "count": len(files), "truncated": truncated}


def grep_command(pattern: str, path: str, include: "str | None") -> str:
    cmd = f"rg --json --no-ignore {shlex.quote(pattern)} {shlex.quote(path)}"
    if include:
        cmd += f" --glob {shlex.quote(include)}"
    cmd += " 2>/dev/null || true"
    return cmd


def parse_grep_json_output(output: str) -> dict:
    matches: "list[dict]" = []
    for line in output.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "match":
            continue
        data = obj["data"]
        matches.append(
            {
                "path": data["path"]["text"],
                "line": data["line_number"],
                "text": data["lines"]["text"],
            }
        )

    truncated = len(matches) > GREP_LIMIT
    matches = matches[:GREP_LIMIT]
    for m in matches:
        if len(m["text"]) > GREP_MAX_LINE_LEN:
            m["text"] = m["text"][:GREP_MAX_LINE_LEN] + "..."

    return {"matches": matches, "count": len(matches), "truncated": truncated}


# ---------------------------------------------------------------------------
# read: pagination
# ---------------------------------------------------------------------------

READ_DEFAULT_LIMIT = 2000
READ_MAX_BYTES = 50 * 1024
READ_MAX_LINE_LEN = 2000


def paginate_text(content: str, offset: int, limit: int) -> dict:
    """Apply offset/limit pagination to file content. Returns a plain dict
    with the same field names: type, content, offset, lines_shown,
    total_lines, truncated."""
    all_lines = content.splitlines(keepends=True)
    total = len(all_lines)
    start = offset - 1
    page_lines = all_lines[start : start + limit]

    raw: list[str] = []
    bytes_used = 0
    cut = False
    for i, line in enumerate(page_lines):
        text = line.rstrip("\n")
        if len(text) > READ_MAX_LINE_LEN:
            text = text[:READ_MAX_LINE_LEN] + "... (truncated)"
        size = len(text.encode()) + 1
        if bytes_used + size > READ_MAX_BYTES:
            cut = True
            break
        raw.append(f"{start + i + 1}: {text}")
        bytes_used += size

    more = cut or (start + len(page_lines) < total)
    return {
        "type": "file",
        "content": "\n".join(raw),
        "offset": offset,
        "lines_shown": len(raw),
        "total_lines": total,
        "truncated": more,
    }


# ---------------------------------------------------------------------------
# edit: fuzzy-match replace pipeline -- port of opencode's edit.ts replacer,
# 9 strategies tried in order, first unique match wins.
# ---------------------------------------------------------------------------

BLOCK_ANCHOR_MIN_LINES = 3
BLOCK_ANCHOR_SCORE_THRESHOLD = 0.3
CONTEXT_AWARE_MATCH_RATIO = 0.5

Replacer = Generator[str, None, None]


def _simple(content: str, find: str) -> Generator[str, None, None]:
    if find in content:
        yield find


def _line_trimmed(content: str, find: str) -> Generator[str, None, None]:
    orig = content.split("\n")
    search = find.split("\n")
    if search and search[-1] == "":
        search.pop()
    for i in range(len(orig) - len(search) + 1):
        if all(orig[i + j].strip() == search[j].strip() for j in range(len(search))):
            start = sum(len(orig[k]) + 1 for k in range(i))
            end = start + sum(len(orig[i + k]) + 1 for k in range(len(search))) - 1
            yield content[start:end]


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1))
        prev = curr
    return prev[len(b)]


def _block_anchor(content: str, find: str) -> Generator[str, None, None]:
    orig = content.split("\n")
    search = find.split("\n")
    if len(search) and search[-1] == "":
        search.pop()
    if len(search) < BLOCK_ANCHOR_MIN_LINES:
        return
    first, last = search[0].strip(), search[-1].strip()

    candidates: list[tuple[int, int]] = []
    for i, line in enumerate(orig):
        if line.strip() != first:
            continue
        for j in range(i + 2, len(orig)):
            if orig[j].strip() == last:
                candidates.append((i, j))
                break

    def _score(start: int, end: int) -> float:
        block = orig[start : end + 1]
        mid_count = min(len(search) - 2, len(block) - 2)
        if mid_count <= 0:
            return 1.0
        total = 0.0
        for k in range(1, mid_count + 1):
            a, b = orig[start + k].strip(), search[k].strip()
            mx = max(len(a), len(b))
            total += (1 - _levenshtein(a, b) / mx) if mx else 1.0
        return total / mid_count

    if len(candidates) == 1:
        s, e = candidates[0]
        if _score(s, e) >= 0.0:
            start = sum(len(orig[k]) + 1 for k in range(s))
            end = start + sum(len(orig[s + k]) + 1 for k in range(e - s + 1)) - 1
            yield content[start:end]
    else:
        best, best_score = None, -1.0
        for s, e in candidates:
            sc = _score(s, e)
            if sc > best_score:
                best_score, best = sc, (s, e)
        if best and best_score >= BLOCK_ANCHOR_SCORE_THRESHOLD:
            s, e = best
            start = sum(len(orig[k]) + 1 for k in range(s))
            end = start + sum(len(orig[s + k]) + 1 for k in range(e - s + 1)) - 1
            yield content[start:end]


def _whitespace_normalized(content: str, find: str) -> Generator[str, None, None]:
    norm = lambda t: re.sub(r"\s+", " ", t).strip()
    nf = norm(find)
    lines = content.split("\n")
    find_lines = find.split("\n")
    for i, line in enumerate(lines):
        if norm(line) == nf:
            yield line
        elif norm(line).__contains__(nf) and len(find_lines) == 1:
            words = re.escape(find.strip()).replace(r"\ ", r"\s+")
            m = re.search(words, line)
            if m:
                yield m.group(0)
    if len(find_lines) > 1:
        for i in range(len(lines) - len(find_lines) + 1):
            block = lines[i : i + len(find_lines)]
            if norm("\n".join(block)) == nf:
                yield "\n".join(block)


def _indentation_flexible(content: str, find: str) -> Generator[str, None, None]:
    def strip_indent(t: str) -> str:
        ls = t.split("\n")
        non_empty = [l for l in ls if l.strip()]
        if not non_empty:
            return t
        min_ind = min(len(l) - len(l.lstrip()) for l in non_empty)
        return "\n".join(l if not l.strip() else l[min_ind:] for l in ls)

    nf = strip_indent(find)
    find_lines = find.split("\n")
    orig = content.split("\n")
    for i in range(len(orig) - len(find_lines) + 1):
        block = orig[i : i + len(find_lines)]
        if strip_indent("\n".join(block)) == nf:
            yield "\n".join(block)


def _escape_normalized(content: str, find: str) -> Generator[str, None, None]:
    _esc = {
        "n": "\n",
        "t": "\t",
        "r": "\r",
        "'": "'",
        '"': '"',
        "`": "`",
        "\\": "\\",
        "\n": "\n",
        "$": "$",
    }

    def unescape(s: str) -> str:
        return re.sub(r"\\(.)", lambda m: _esc.get(m.group(1), m.group(0)), s)

    uf = unescape(find)
    if uf in content:
        yield uf
    orig = content.split("\n")
    find_lines = uf.split("\n")
    for i in range(len(orig) - len(find_lines) + 1):
        block = "\n".join(orig[i : i + len(find_lines)])
        if unescape(block) == uf:
            yield block


def _trimmed_boundary(content: str, find: str) -> Generator[str, None, None]:
    tf = find.strip()
    if tf == find:
        return
    if tf in content:
        yield tf
    orig = content.split("\n")
    find_lines = find.split("\n")
    for i in range(len(orig) - len(find_lines) + 1):
        block = "\n".join(orig[i : i + len(find_lines)])
        if block.strip() == tf:
            yield block


def _context_aware(content: str, find: str) -> Generator[str, None, None]:
    orig = content.split("\n")
    find_lines = find.split("\n")
    if len(find_lines) and find_lines[-1] == "":
        find_lines.pop()
    if len(find_lines) < BLOCK_ANCHOR_MIN_LINES:
        return
    first, last = find_lines[0].strip(), find_lines[-1].strip()
    for i, line in enumerate(orig):
        if line.strip() != first:
            continue
        for j in range(i + 2, len(orig)):
            if orig[j].strip() == last:
                block = orig[i : j + 1]
                if len(block) == len(find_lines):
                    mid_non_empty = [
                        (block[k].strip(), find_lines[k].strip())
                        for k in range(1, len(block) - 1)
                        if block[k].strip() or find_lines[k].strip()
                    ]
                    if (
                        not mid_non_empty
                        or sum(a == b for a, b in mid_non_empty) / len(mid_non_empty)
                        >= CONTEXT_AWARE_MATCH_RATIO
                    ):
                        yield "\n".join(block)
                break


def _multi_occurrence(content: str, find: str) -> Generator[str, None, None]:
    start = 0
    while True:
        idx = content.find(find, start)
        if idx == -1:
            break
        yield find
        start = idx + len(find)


_STRATEGIES = [
    _simple,
    _line_trimmed,
    _block_anchor,
    _whitespace_normalized,
    _indentation_flexible,
    _escape_normalized,
    _trimmed_boundary,
    _context_aware,
    _multi_occurrence,
]


def replace(content: str, old: str, new: str, replace_all: bool = False) -> str:
    """Apply the fuzzy-match replace pipeline. Raises ValueError on
    identical old/new, no match, or an ambiguous (multiple-match) result --
    same contract as the tool this backs."""
    if old == new:
        raise ValueError("old_string and new_string are identical — no change to apply.")

    not_found = True
    for strategy in _STRATEGIES:
        for candidate in strategy(content, old):
            idx = content.find(candidate)
            if idx == -1:
                continue
            not_found = False
            if replace_all:
                return content.replace(candidate, new)
            last_idx = content.rfind(candidate)
            if idx != last_idx:
                continue
            return content[:idx] + new + content[idx + len(candidate) :]

    if not_found:
        raise ValueError(
            "Could not find old_string in the file. "
            "It must match exactly (including whitespace and indentation)."
        )
    raise ValueError(
        "Found multiple matches for old_string. "
        "Provide more surrounding context to make the match unique."
    )


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def _parse_tool_args(arguments_json: str) -> dict:
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except (json.JSONDecodeError, TypeError):
        args = {}
    return args if isinstance(args, dict) else {}


def _run_bash_tool(arguments_json: str) -> str:
    args = _parse_tool_args(arguments_json)
    command = args.get("command", "") if isinstance(args, dict) else ""
    workdir = args.get("workdir") or None
    # BASH_PARAMS advertises both of these to the model; both were
    # previously ignored here, so a command silently ran in this process's
    # own cwd and with the hardcoded default no matter what was passed.
    try:
        timeout = int(args.get("timeout") or _BASH_TIMEOUT_S)
    except (TypeError, ValueError):
        timeout = _BASH_TIMEOUT_S
    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            timeout=timeout,
            text=True,
            cwd=workdir,
        )
        output = proc.stdout + proc.stderr
        return json.dumps({"output": output, "returncode": proc.returncode})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"command timed out after {timeout}s"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _run_read_tool(arguments_json: str) -> str:
    args = _parse_tool_args(arguments_json)
    file_path = str(args.get("file_path", ""))
    try:
        offset = int(args.get("offset") or 1)
        limit = int(args.get("limit") or READ_DEFAULT_LIMIT)
    except (TypeError, ValueError) as e:
        return json.dumps({"error": f"invalid offset/limit: {e}"})

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
    result = paginate_text(content, offset, limit)
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
        updated = replace(content, old_string, new_string, replace_all)
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
            ["bash", "-c", glob_command(pattern, path)],
            capture_output=True,
            timeout=30,
            text=True,
        )
        return json.dumps(parse_glob_output(proc.stdout))
    except Exception as e:
        return json.dumps({"error": str(e)})


def _run_grep_tool(arguments_json: str) -> str:
    args = _parse_tool_args(arguments_json)
    pattern = str(args.get("pattern", ""))
    path = str(args.get("path") or ".")
    include = args.get("include")
    try:
        proc = subprocess.run(
            ["bash", "-c", grep_command(pattern, path, include)],
            capture_output=True,
            timeout=30,
            text=True,
        )
        return json.dumps(parse_grep_json_output(proc.stdout))
    except Exception as e:
        return json.dumps({"error": str(e)})


def _run_webfetch_tool(arguments_json: str) -> str:
    import httpx
    import html2text

    args = _parse_tool_args(arguments_json)
    url = str(args.get("url", ""))
    fmt = str(args.get("format") or "markdown")
    try:
        timeout = min(int(args.get("timeout") or _WEBFETCH_DEFAULT_TIMEOUT), _WEBFETCH_MAX_TIMEOUT)
    except (TypeError, ValueError) as e:
        return json.dumps({"error": f"invalid timeout: {e}"})

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
    try:
        _todo_store = [dict(t) for t in todos]
    except (TypeError, ValueError) as e:
        return json.dumps({"error": f"invalid todos entry: {e}"})
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
            f"[tandem_harness] WARNING: failed to offload large tool output to {offload_path}: {e}"
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

BUILTIN_TOOL_SCHEMAS = {
    "bash": _tool_schema("bash", "Execute a bash command and return its output.", BASH_PARAMS),
    "read": _tool_schema(
        "read", "Read a file (with optional offset/limit) or list a directory.", READ_PARAMS
    ),
    "write": _tool_schema(
        "write", "Write content to a file, creating parent directories if needed.", WRITE_PARAMS
    ),
    "edit": _tool_schema(
        "edit", "Replace a string in a file. Uses fuzzy matching as fallback.", EDIT_PARAMS
    ),
    "glob": _tool_schema(
        "glob",
        "Find files matching a glob pattern. Defaults to the current directory.",
        GLOB_PARAMS,
    ),
    "grep": _tool_schema(
        "grep",
        "Search for a regex pattern in file contents. Defaults to the current directory.",
        GREP_PARAMS,
    ),
    "webfetch": _tool_schema(
        "webfetch",
        "Fetch a URL and return its content as text, markdown, or raw HTML.",
        WEBFETCH_PARAMS,
    ),
    "todowrite": _tool_schema(
        "todowrite", "Update the todo list with a new set of items.", TODOWRITE_PARAMS
    ),
}


__all__ = ["TOOL_DISPATCH", "BUILTIN_TOOL_SCHEMAS", "offload_if_oversized"]
