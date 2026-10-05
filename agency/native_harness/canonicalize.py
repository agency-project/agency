"""Canonical compact form for command and test-run output.

    is_run_command(command) -> bool
    canonicalize_run(output) -> (text, stats)
    compact_tables(output) -> (text, stats)

Applied by the bash tool, when enabled, to commands that run a program or a
test suite (not to file reads, searches or queries, whose content the caller
asked for line by line).

Kept lines are byte-identical to the input; the only changes are cleanup that
loses nothing a reader sees (ANSI codes, carriage-return progress, trailing
spaces) and, for large outputs only, omitting lines behind an explicit
"[... N lines omitted ...]" marker. Small outputs are what the caller asked to
see, so they pass through after cleanup.

Large outputs keep: the first and last lines, every error/failure line with its
traceback context, test-runner summaries, and one copy of each repeated block
(a run of lines with the same shape, digits aside, is collapsed to its first
and last period).
"""

from __future__ import annotations

import re

SMALL_LINES = 60
SMALL_CHARS = 4000
HEAD_LINES = 5
TAIL_LINES = 12
MAX_CHARS = 6000

_RUN_RE = re.compile(r"runtests|pytest|unittest|python[0-9.]* +/tmp/|python[0-9.]* +-c|\.py\b")
_READ_RE = re.compile(r"\bsed -n|\bcat |\bhead |\btail |\bgrep |\bnl ")
_QUERY_RE = re.compile(r"duckdb|\bselect\b", re.I)
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_NUM_RE = re.compile(r"[-+]?\d[\d_]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
_ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)|^\s*File \"|Error\b|Exception\b|^FAIL|^ERROR|"
    r"AssertionError|assert |^E\s{2,}|Warning:|^FAILED|panic|Segmentation fault|Killed",
)
_SUMMARY_RE = re.compile(
    r"^Ran \d+ tests? in|^(OK|FAILED)\b|^=+ .*(passed|failed|error|skipped).* =+$|"
    r"^\d+ (passed|failed)|^Tests? (passed|failed)|^-{20,}$|^={20,}$"
)


def is_run_command(command: str) -> bool:
    """A command that runs a program or tests, rather than reading, searching or querying."""
    return (
        bool(_RUN_RE.search(command))
        and not _READ_RE.search(command)
        and not _QUERY_RE.search(command)
    )


def clean(text: str) -> str:
    """Lossless cleanup: ANSI codes, carriage-return overwrites, trailing spaces."""
    text = _ANSI_RE.sub("", text)
    lines = []
    for line in text.split("\n"):
        if "\r" in line:
            line = line.rstrip("\r").rsplit("\r", 1)[-1]
        lines.append(line.rstrip())
    return "\n".join(lines)


def _shape(line: str) -> str:
    return _NUM_RE.sub("#", line.strip())


def _keep_set(lines: list[str]) -> set[int]:
    n = len(lines)
    keep = set(range(min(HEAD_LINES, n))) | set(range(max(0, n - TAIL_LINES), n))
    for i, line in enumerate(lines):
        if _ERROR_RE.search(line):
            # The failing line plus a little context on both sides.
            keep.update(range(max(0, i - 2), min(n, i + 3)))
        if _SUMMARY_RE.search(line):
            keep.add(i)
    return keep


def _repeated_runs(
    lines: list[str], min_repeats: int = 3, max_period: int = 8
) -> list[tuple[int, int, int]]:
    """(start, end, period) of runs where the line-shape sequence repeats at least min_repeats times."""
    shapes = [_shape(line) for line in lines]
    runs = []
    i, n = 0, len(lines)
    while i < n:
        best = None
        for p in range(1, max_period + 1):
            if i + p * min_repeats > n:
                break
            j = i + p
            while j < n and shapes[j] == shapes[j - p]:
                j += 1
            if (j - i) // p >= min_repeats and (best is None or j - i > best[1] - best[0]):
                best = (i, j, p)
        if best:
            runs.append(best)
            i = best[1]
        else:
            i += 1
    return runs


def canonicalize_run(output: str) -> tuple[str, dict]:
    text = clean(output)
    lines = text.split("\n")
    stats = {"lines_in": len(lines), "chars_in": len(output)}
    if len(lines) <= SMALL_LINES and len(text) <= SMALL_CHARS:
        stats.update(lines_out=len(lines), omitted=0)
        return text, stats

    keep = _keep_set(lines)
    drop: set[int] = set()
    for start, end, p in _repeated_runs(lines):
        # Keep the first and last period of a repeated block.
        middle = set(range(start + p, end - p))
        drop |= middle - keep
    selected = [i for i in range(len(lines)) if i not in drop]

    # Still large: keep only the must-keep lines.
    if sum(len(lines[i]) + 1 for i in selected) > MAX_CHARS:
        selected = sorted(keep)
    out, prev = [], -1
    for i in selected:
        if i > prev + 1:
            out.append(f"[... {i - prev - 1} lines omitted ...]")
        out.append(lines[i])
        prev = i
    if prev < len(lines) - 1:
        out.append(f"[... {len(lines) - 1 - prev} lines omitted ...]")
    stats.update(lines_out=len(selected), omitted=len(lines) - len(selected))
    return "\n".join(out), stats


_CODE_READ_RE = re.compile(r"\bsed -n|\bcat |\bhead |\btail |\bnl ")
_LICENSE_RE = re.compile(r"copyright|licen[cs]e|warrant", re.I)


def is_code_read_command(command: str) -> bool:
    """A command that prints file contents (and is not a search, query or program run)."""
    return (
        bool(_CODE_READ_RE.search(command))
        and not is_run_command(command)
        and not _QUERY_RE.search(command)
    )


def compact_code_read(output: str) -> tuple[str, dict]:
    """Drop only content-free lines from a code read, with markers that keep line counts exact.

    Trailing whitespace goes; a leading license/copyright comment block and runs of
    3+ blank lines become markers stating how many lines they replace, so line
    numbers the reader derives from the output stay correct.
    """
    lines = [line.rstrip() for line in output.split("\n")]
    out = []
    i = 0
    header = 0
    while header < len(lines) and lines[header].lstrip().startswith("#"):
        header += 1
    if header >= 5 and any(_LICENSE_RE.search(line) for line in lines[:header]):
        out.append(f"[{header} lines of license header omitted]")
        i = header
    while i < len(lines):
        j = i
        while j < len(lines) and not lines[j]:
            j += 1
        if j - i >= 3:
            out.append(f"[{j - i} blank lines]")
            i = j
            continue
        out.append(lines[i])
        i += 1
    text = "\n".join(out)
    return text, {"chars_in": len(output), "chars_out": len(text)}


_PROGRESS_RE = re.compile(r"^\s*\d{1,3}% ▕[^▏]*▏.*$")
_BOX_TOP_RE = re.compile(r"^\s*┌[─┬]+┐$")
_BOX_RULE_RE = re.compile(r"^\s*├[─┼┬┴]+┤$")
_BOX_BOTTOM_RE = re.compile(r"^\s*└[─┴]+┘$")


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip()[1:-1].split("│")]


def _table_rows(body: list[str]) -> "list[str] | None":
    """A box table's inner lines as `a | b` rows, or None if it isn't one we understand."""
    rules = [k for k, line in enumerate(body) if _BOX_RULE_RE.match(line)]
    if not rules or any(
        not line.strip().startswith("│") for k, line in enumerate(body) if k not in rules
    ):
        return None
    head = [_cells(line) for line in body[: rules[0]]]
    if len(head) == 2 and len(head[0]) == len(head[1]):
        rows = [
            " | ".join(name if name == kind else f"{name} ({kind})" for name, kind in zip(*head))
        ]
    else:
        rows = [" | ".join(cells) for cells in head]
    footer = False
    for k in range(rules[0] + 1, len(body)):
        if k in rules:
            # A rule closing the columns (├──┴──┤) starts a free-text footer.
            footer = footer or "┴" in body[k]
            continue
        row = " ".join(body[k].strip()[1:-1].split()) if footer else " | ".join(_cells(body[k]))
        if not (rows and row == rows[-1] and set(row) <= {"·", "|", " "}):
            rows.append(row)
    return rows


def compact_tables(output: str) -> tuple[str, dict]:
    """Box-drawn result tables (DuckDB) as plain `a | b` rows; progress-bar lines dropped.

    Every cell value is kept verbatim, with its column name and type in the header
    row. Padding, borders and repeated `·` elision rows are the only things removed.
    """
    lines = [line for line in clean(output).split("\n") if not _PROGRESS_RE.match(line)]
    progress = len(clean(output).split("\n")) - len(lines)
    out, tables, i = [], 0, 0
    while i < len(lines):
        if not _BOX_TOP_RE.match(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not _BOX_BOTTOM_RE.match(lines[j]):
            j += 1
        rows = _table_rows(lines[i + 1 : j]) if j < len(lines) else None
        if rows is None:
            out.extend(lines[i : j + 1])
            i = j + 1
            continue
        out.extend(rows)
        tables += 1
        i = j + 1
        # DuckDB prints the row/column counts under the box, padded to its width.
        while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
            out.append(" ".join(lines[i].split()))
            i += 1
    text = "\n".join(out)
    return text, {
        "chars_in": len(output),
        "chars_out": len(text),
        "tables": tables,
        "progress_lines": progress,
    }
