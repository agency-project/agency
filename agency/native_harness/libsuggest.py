"""Harness-side retrieval over the script library at /library (H3): the harness, not the agent, finds similar scripts.

At task start it suggests the library's shared boilerplate script plus the scripts that best match the task text
(TF-IDF cosine). When
the agent writes a script, it points to a library script whose shingles are largely contained in it (the agent is
re-creating it). Each library script is suggested at most once per run. Stdlib only: runs inside the sandbox.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter

ENABLED = False
LIB_DIR = "/library"
START_K = 3
CONTAIN_MIN = 0.4  # share of a library script's 5-token shingles found in the agent's script (H3 runs: 21% of scripts)
MIN_SHINGLES = 30

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
ROW = re.compile(r"^\|\s*(lib_\d+\.\w+)\s*\|\s*(.*?)\s*\|\s*(\S*)\s*\|\s*(\d+)\s*\|\s*$")
HEREDOC_FILE = re.compile(r"cat\s*>\s*(\S+)\s*<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)\n\s*\2\b", re.S)
HEREDOC_PY = re.compile(r"(?:python[0-9.]*|bash|sh)\s+-\s*<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)\n\s*\1\b", re.S)
STOP = {"self", "none", "true", "false", "return", "import", "from", "print", "with", "this", "that", "should",
        "when", "have", "there", "which", "would", "test", "tests", "script", "file", "class", "def"}

_entries: "list[dict] | None" = None
_helpers: "list[dict] | None" = None
_suggested: set = set()


def helpers() -> list:
    """Hints from /library/helpers.json: [{"pattern": regex over a written script, "note": what to use instead}]."""
    global _helpers
    if _helpers is None:
        try:
            _helpers = [h for h in json.load(open(os.path.join(LIB_DIR, "helpers.json"))) if h.get("pattern")]
        except (OSError, ValueError):
            _helpers = []
    return _helpers


def shingles(body: str, k: int = 5) -> set:
    toks = TOKEN.findall(re.sub(r"#[^\n]*", "", body))
    toks = ["S" if t.startswith(("'", '"')) else t for t in toks]
    return {tuple(toks[i:i + k]) for i in range(max(1, len(toks) - k + 1))}


def _words(text: str) -> Counter:
    return Counter(w.lower() for w in WORD.findall(text) if w.lower() not in STOP)


def load(lib_dir: "str | None" = None) -> list:
    """Library entries from INDEX.md, each with its body, shingles and word counts."""
    global _entries
    if _entries is not None:
        return _entries
    _entries = []
    lib_dir = lib_dir or LIB_DIR
    try:
        index = open(os.path.join(lib_dir, "INDEX.md")).read()
    except OSError:
        return _entries
    for line in index.splitlines():
        m = ROW.match(line)
        if not m:
            continue
        try:
            body = open(os.path.join(lib_dir, m.group(1))).read()
        except OSError:
            continue
        _entries.append({"file": m.group(1), "purpose": m.group(2), "task": m.group(3), "body": body,
                         # purpose counted twice: it is the human-readable summary of what the script is for
                         "shingles": shingles(body), "words": _words(m.group(2) + " " + m.group(2) + " " + body)})
    return _entries


def _centrality(e: dict, entries: list) -> float:
    if len(e["shingles"]) < MIN_SHINGLES:
        return 0.0
    others = [o for o in entries if o is not e]
    return sum(len(e["shingles"] & o["shingles"]) for o in others) / (len(e["shingles"]) * len(others))


def _line(e: dict, why: str) -> str:
    return f"  {LIB_DIR}/{e['file']}: {e['purpose']} (written for {e['task']}; {why})"


def start_note(task_text: str) -> str:
    """Suggestion block for the first turn: the library scripts most related to the task text."""
    entries = load()
    if not entries:
        return ""
    df = Counter(w for e in entries for w in e["words"])
    idf = {w: math.log((1 + len(entries)) / (1 + n)) + 1 for w, n in df.items()}
    q = _words(task_text)

    def vec(c):
        return {w: n * idf.get(w, 0.0) for w, n in c.items()}
    qv = vec(q)
    qn = math.sqrt(sum(v * v for v in qv.values())) or 1.0
    scored = []
    for e in entries:
        ev = vec(e["words"])
        en = math.sqrt(sum(v * v for v in ev.values())) or 1.0
        scored.append((sum(qv.get(w, 0.0) * v for w, v in ev.items()) / (qn * en), e))
    # Slot 1: the most central script (most of it recurs inside other library scripts), i.e. the repo's shared
    # boilerplate, which is what agents re-create most (H3 runs: with TF-IDF slots 2-3, 9/10 re-created scripts hit).
    hub = max(entries, key=lambda e: _centrality(e, entries)) if len(entries) > 1 else None
    picks = [(s, e) for s, e in sorted(scored, key=lambda x: -x[0]) if s > 0 and e is not hub][:START_K - (hub is not None)]
    if hub is None and not picks:
        return ""
    lines = [_line(hub, "common setup boilerplate shared by many scripts")] if hub is not None else []
    if helpers():
        lines.insert(0, "  importable helpers (see the top of /library/INDEX.md): " + "; ".join(h["note"] for h in helpers()))
    lines += [_line(e, f"match {s:.2f}") for s, e in picks]
    for e in ([hub] if hub is not None else []) + [e for _, e in picks]:
        _suggested.add(e["file"])
    return ("[harness: script library] These scripts from earlier tasks in this repository look related to this "
            "task; reading or adapting one may save you writing it from scratch:\n" + "\n".join(lines))


def scripts_in(tool: str, args: dict) -> list:
    """(path, body) of scripts a write or bash call creates."""
    if tool == "write":
        body, path = args.get("content") or "", args.get("file_path") or args.get("path") or ""
        return [(path, body)] if body and not path.startswith(LIB_DIR) else []
    if tool == "bash":
        cmd = args.get("command") or ""
        out = [(m.group(1), m.group(3)) for m in HEREDOC_FILE.finditer(cmd) if not m.group(1).startswith(LIB_DIR)]
        return out + [("<stdin>", m.group(2)) for m in HEREDOC_PY.finditer(cmd)]
    return []


def write_note(tool: str, args: dict) -> str:
    """Note for a tool result when the agent just wrote what a library helper does, or re-created a library script."""
    bodies = [body for _, body in scripts_in(tool, args)]
    for i, h in enumerate(helpers()):
        if f"helper{i}" not in _suggested and any(re.search(h["pattern"], b) for b in bodies):
            _suggested.add(f"helper{i}")
            return "[harness: script library] " + h["note"]
    hits = []
    for body in bodies:
        sh = shingles(body)
        for e in load():
            if e["file"] in _suggested or len(e["shingles"]) < MIN_SHINGLES:
                continue
            c = len(sh & e["shingles"]) / len(e["shingles"])
            if c >= CONTAIN_MIN:
                hits.append((c, e))
    if not hits:
        return ""
    c, e = max(hits, key=lambda x: x[0])
    _suggested.add(e["file"])
    return ("[harness: script library] The script you just wrote largely re-creates an existing library script; you "
            "can reuse or adapt it next time:\n" + _line(e, f"{c:.0%} of it appears in yours"))


def with_note(tool: str, handler):
    """Wrap a write/bash handler so its result carries write_note() when it applies."""
    def call(arguments_json: str) -> str:
        result = handler(arguments_json)
        note = write_note(tool, parse_args(arguments_json))
        if not note:
            return result
        try:
            d = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return f"{result}\n{note}"
        key = "output" if "output" in d else "error"
        d[key] = f"{d.get(key) or ''}\n{note}"
        return json.dumps(d)
    return call


def task_text(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else " ".join(b.get("text", "") for b in c or [] if isinstance(b, dict))
    return ""


def reset() -> None:
    global _entries, _helpers
    _entries = _helpers = None
    _suggested.clear()


def parse_args(arguments_json: str) -> dict:
    try:
        a = json.loads(arguments_json) if arguments_json else {}
    except (json.JSONDecodeError, TypeError):
        return {}
    return a if isinstance(a, dict) else {}
