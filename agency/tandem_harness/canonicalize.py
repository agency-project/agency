"""Canonical compact form for command and test-run output.

    is_run_command(command) -> bool
    canonicalize_run(output) -> (text, stats)

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
