from __future__ import annotations

"""
Edit tool: port of opencode's edit.ts replacer pipeline.

Tries 9 replacement strategies in order, raising on not-found or ambiguity.
"""
from typing import TYPE_CHECKING, Generator
from ..agdata import agdata, agerror
from ..agutil import format_exception
from ..agtool import agtool

if TYPE_CHECKING:
    from ..agsandbox import agSandbox


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum number of lines a search block must have for anchor-based fuzzy
# matching to apply (needs a first, last, and at least one middle line)
BLOCK_ANCHOR_MIN_LINES = 3

# Minimum similarity score (0–1) for accepting the best fuzzy block-anchor
# candidate when multiple anchor pairs exist
BLOCK_ANCHOR_SCORE_THRESHOLD = 0.3

# Minimum fraction of non-empty middle lines that must match exactly for a
# context-aware block candidate to be accepted
CONTEXT_AWARE_MATCH_RATIO = 0.5


# ---------------------------------------------------------------------------
# Replacer strategies (each yields candidate substrings from `content` that
# match `find`; the first strategy to produce a unique match wins)
# ---------------------------------------------------------------------------

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
    import re

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
        import re

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


def _replace(content: str, old: str, new: str, replace_all: bool = False) -> str:
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


_EDIT_PARAMS = {
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


def make_edit(sandbox: "agSandbox") -> agtool:
    """Return an edit tool that edits files inside *sandbox*'s container.

    Reads the file via ``sandbox.read_file``, applies the existing ``_replace``
    fuzzy-match pipeline in Python, then writes back via ``sandbox.write_file``.
    """

    def _run_sandboxed(arg: agdata) -> agdata:
        file_path = str(arg.file_path)  # type: ignore[arg-type]
        old_string: str = str(arg.old_string)  # type: ignore[arg-type]
        new_string: str = str(arg.new_string)  # type: ignore[arg-type]
        replace_all: bool = bool(getattr(arg, "replace_all", False))

        try:
            content = sandbox.read_file(file_path)
        except FileNotFoundError:
            return agerror(f"File not found: {file_path}")
        except Exception as e:
            return agerror(format_exception(e))

        try:
            updated = _replace(content, old_string, new_string, replace_all)
            sandbox.write_file(file_path, updated)
            return agdata(path=file_path, success=True)
        except (ValueError, OSError) as e:
            return agerror(format_exception(e))

    def _log(tool: agtool, arg: agdata, result: agdata, elapsed_ms: int) -> None:
        if tool._term is None:
            return
        path = str(getattr(arg, "file_path", "?"))
        ok = not isinstance(result, agerror)
        status = "✓" if ok else f"✗ {result.error}"
        tool._term.log("TOOL ✓   ", f"edit  {path}  {status}  ({elapsed_ms}ms)")
        if tool._aglog is not None:
            tool._aglog._tool_call(tool.name, arg.to_dict(), result.to_dict(), elapsed_ms)

    return agtool(
        name="edit",
        fn=_run_sandboxed,
        description="Replace a string in a file inside the sandbox. Uses fuzzy matching as fallback.",
        params=_EDIT_PARAMS,
        log_fn=_log,
        run_in_subprocess=False,
    )
