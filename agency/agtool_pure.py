"""Pure, dependency-free tool algorithms shared by two very different
callers: `agency/tools/edit.py` and `read.py` (host-side, wrapped in an
`agtool` that talks to a sandbox via `sandbox.read_file`/`write_file`), and
the in-container native entrypoint (`agharness_backends/
_native_in_container_entrypoint.py`), which reads/writes the local
filesystem directly since it already runs inside the container.

Deliberately zero imports beyond stdlib (`re` only) and zero relative
imports -- the whole reason this file exists separately from `tools/edit.py`/
`tools/read.py` rather than being imported from there. Those modules import
`agtool`/`agdata`, which transitively import `agent`/`agllm_backends`
(`import openai` etc. at module level) -- fine for the host process, but
exactly what the in-container entrypoint avoids needing installed just to
reuse this pure logic. This module has no such chain, so the entrypoint
loads it directly by file path (`importlib.util.spec_from_file_location`,
same technique it already uses to avoid importing `agency` as a package at
all) and gets the identical algorithm, not a hand-copied drift risk.

Every function here works on plain strings/dicts -- never `agdata` (that
type itself isn't stdlib-loadable this way either) -- callers on both sides
wrap the return value into whatever shape they need.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Generator

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling `parameters` shape) -- single
# source of truth for both the host-side sandboxed tools (agency/tools/
# *.py, which pass these to `agtool(..., params=...)`) and the in-container
# native entrypoint's own tool schemas. Keeping these here, not duplicated
# in both places, is what makes "same tool name/params/description on both
# sides" a structural guarantee rather than a convention to remember.
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
# glob / grep: shell command construction + output parsing. The command
# string is identical whether the caller runs it via `sandbox.exec()`
# (host bridging into a container) or a local `subprocess.run(["bash",
# "-c", cmd])` (already inside the container) -- same `rg`/`find` binary,
# same flags -- so building it once here means the two call sites can never
# quietly drift into different search semantics.
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
    """Apply offset/limit pagination to file content -- identical behavior
    to the tool the host-side `read` tool exposes. Returns a plain dict
    (not agdata) with the same field names: type, content, offset,
    lines_shown, total_lines, truncated."""
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
