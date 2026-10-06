"""Locate-then-read reports: code the supervisor didn't name becomes a one-line outline.

    outline_report(text, paths, request) -> (text, stats)

When a worker's report copies more than CODE_LINE_LIMIT lines of code from
files it read, every Python function or method the request didn't name is
replaced by `[not shown: path:a-b  def signature]`, so the supervisor sees
where things are and asks for the ones it wants by name. Named definitions,
module-level lines and prose stay as the worker wrote them.
"""

from __future__ import annotations

import ast
import json
import os
import re

CODE_LINE_LIMIT = 60
_NUM_PREFIX = re.compile(r"^\s*\d+[:\-\t ]\s?")
_CLASS_RE = re.compile(r"^\s*class (\w+)")
_DEF_RE = re.compile(r"^\s*(?:async\s+)?def (\w+)")
_PATH_RE = re.compile(r"[\w./-]+\.py\b")
_CD_RE = re.compile(r"\bcd\s+([^\s;&|]+)")
NOTE = "Code the request didn't name is listed as [not shown: ...]; ask for any of it by name."


def _norm(line: str) -> "str | None":
    s = _NUM_PREFIX.sub("", line).strip()
    return s if len(s) >= 8 else None


def _leaf_definitions(source: str) -> "list[tuple[str, int, int]]":
    """(qualified name, first line, last line) of functions and methods with no nested definition."""
    out = []

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                nested = any(
                    isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    for c in ast.walk(child)
                    if c is not child
                )
                if not isinstance(child, ast.ClassDef) and not nested:
                    first = min([child.lineno] + [d.lineno for d in child.decorator_list])
                    out.append((name, first, child.end_lineno))
                visit(child, name + ".")

    visit(ast.parse(source), "")
    return out


def paths_from_calls(calls: "list[dict]") -> "list[str]":
    """Python files the worker's tool calls touched, resolved to existing paths."""
    found = []
    for entry in calls:
        try:
            args = json.loads(entry.get("arguments") or "{}")
        except ValueError:
            continue
        text = " ".join(str(v) for v in args.values())
        workdir = [str(args["workdir"])] if args.get("workdir") else []
        dirs = [os.path.join(w, d) for w in workdir or [""] for d in _CD_RE.findall(text)] + workdir + [os.getcwd()]
        for token in _PATH_RE.findall(text):
            for cand in [token] + [os.path.join(d, token) for d in dirs]:
                if os.path.isfile(cand) and cand not in found:
                    found.append(os.path.abspath(cand))
                    break
    return list(dict.fromkeys(found))


def _named(qualname: str, request: str) -> bool:
    parts = qualname.split(".")
    return any(re.search(rf"\b{re.escape(p)}\b", request) for p in parts + [qualname])


def outline_report(text: str, paths: "list[str]", request: str) -> tuple[str, dict]:
    index: "dict[str, set]" = {}
    sources = {}
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                source = f.read()
            leaves = _leaf_definitions(source)
        except (OSError, SyntaxError, ValueError):
            continue
        lines = source.split("\n")
        sources[path] = (lines, leaves)
        for n, line in enumerate(lines, 1):
            k = _norm(line)
            if k:
                owner = next((d for d in leaves if d[1] <= n <= d[2]), None)
                index.setdefault(k, set()).add((path, owner))

    report = text.split("\n")
    owners = [set(index.get(_norm(x) or "", ())) for x in report]
    stats = {"code_lines": sum(1 for o in owners if o), "outlined": 0, "kept": 0}
    if stats["code_lines"] <= CODE_LINE_LIMIT:
        return text, stats

    cls = fn = None
    for i, line in enumerate(report):
        body = _NUM_PREFIX.sub("", line)
        if m := _CLASS_RE.match(body):
            cls, fn = m.group(1), None
        if m := _DEF_RE.match(body):
            fn = m.group(1)
        if len(owners[i]) > 1:
            pick = [o for o in owners[i] if o[1] and cls and fn and o[1][0].endswith(f"{cls}.{fn}")]
            pick = pick or [o for o in owners[i] if o[1] and cls and f"{cls}." in o[1][0]]
            pick = pick or [o for o in owners[i] if o[1] and fn and o[1][0].split(".")[-1] == fn]
            if len(pick) == 1:
                owners[i] = set(pick)
    assigned = [next(iter(o)) if len(o) == 1 else None for o in owners]
    for order in (range(len(report)), range(len(report) - 1, -1, -1)):
        last = None
        for i in order:
            if assigned[i] is not None:
                last = assigned[i]
            elif owners[i] and last in owners[i]:
                assigned[i] = last

    counts: "dict[tuple, int]" = {}
    for a in assigned:
        if a and a[1]:
            counts[a] = counts.get(a, 0) + 1
    hidden = {a for a, n in counts.items() if n >= 3 and not _named(a[1][0], request)}
    stats["outlined"] = len(hidden)
    stats["kept"] = len([a for a in counts if a not in hidden and counts[a] >= 3])
    if not hidden:
        return text, stats

    # Short unmatched lines (else:, a closing paren) between two lines of one hidden function go with it.
    prev = [None] * len(report)
    nxt = [None] * len(report)
    for i in range(len(report)):
        prev[i] = assigned[i] if assigned[i] is not None else (prev[i - 1] if i else None)
    for i in range(len(report) - 1, -1, -1):
        nxt[i] = assigned[i] if assigned[i] is not None else (nxt[i + 1] if i + 1 < len(report) else None)
    for i in range(len(report)):
        if assigned[i] is None and prev[i] in hidden and prev[i] == nxt[i]:
            assigned[i] = prev[i]

    out, emitted = [], set()
    for i, line in enumerate(report):
        a = assigned[i]
        if a in hidden:
            if a not in emitted:
                path, (name, first, last) = a
                lines = sources[path][0]
                sig = next((lines[n - 1].strip() for n in range(first, last + 1) if _DEF_RE.match(lines[n - 1])), name)
                out.append(f"[not shown: {path}:{first}-{last}  {name}: {sig}]")
                emitted.add(a)
            continue
        # The def line and decorators belong to the hidden function they introduce.
        if a is None and i + 1 < len(report) and assigned[i + 1] in hidden and _DEF_RE.match(_NUM_PREFIX.sub("", line)):
            continue
        out.append(line)
    out += ["", NOTE]
    return "\n".join(out), stats


def code_line_count(text: str, paths: "list[str]") -> int:
    """Lines of `text` that match a line of one of the files at `paths`."""
    known = set()
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                known.update(k for k in map(_norm, f.read().split("\n")) if k)
        except OSError:
            continue
    return sum(1 for line in text.split("\n") if _norm(line) in known)
