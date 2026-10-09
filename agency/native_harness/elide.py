"""Tool-output compaction for code search: grep hits grouped by file and licence headers dropped (T43);
large outputs shown as outlines with references the agent expands via show_elided (T44)."""

from __future__ import annotations

import ast
import json
import os
import re
import shlex

ELIDE_MIN_TOKENS = 0  # 0 = elision off
COMPACT_SEARCH = False

_CHARS_PER_TOKEN = 3.3
_BLOCK_MIN_TOKENS = 600  # smaller file blocks stay as they are
_HITS_KEEP = 3  # grep hits kept per file when a file's hits are elided
_HITS_ELIDE_MIN = 8
_OUTLINE_MAX_ENTRIES = 160
_SIG_MAX_CHARS = 160

REGISTRY: "dict[str, dict]" = {}
SELECTOR = None  # T45: callable(prompt) -> dispatch result; picks the lines shown in full
_SELECT_MAX_SHARE = 0.35
_CONTEXT = {"task": "", "step": ""}

_HIT = re.compile(r"^(\.?/?[\w./@+-]*[\w@+-]\.\w+|[\w./@+-]+/[\w.@+-]+):(\d+)([:-])")
_LICENCE = re.compile(
    r"(?i)licen[sc]e|copyright|warrant|apache\.org|SPDX|all rights reserved|without limitation|distributed (under|on an)"
)
_COMMENT = re.compile(r"^\s*(#|//|/\*|\*|\"\"\"|''')")
_FILE_START = re.compile(r"^(// Copyright|# Copyright|/\*$|/\*\*$|#!/|//go:build|// \+build)")
_READ_CMDS = {"cat", "nl", "head", "tail", "sed", "less", "more", "bat"}
_SEP = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")


def tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN)


# --------------------------------------------------------------------------- T43


def group_search_hits(output: str) -> "tuple[str, dict]":
    """Print the path once per run of consecutive hits from the same file."""
    out, prev, saved = [], None, 0
    for line in output.split("\n"):
        m = _HIT.match(line)
        if m:
            path = m.group(1)
            if path != prev:
                out.append(f"{path}:")
                prev = path
            out.append(f"  {m.group(2)}{m.group(3)}{line[m.end():]}")
            saved += len(path) - 1
        else:
            prev = None
            out.append(line)
    text = "\n".join(out)
    return (text, {"chars_in": len(output), "chars_out": len(text)}) if saved > 0 else (output, {"chars_in": len(output), "chars_out": len(output)})


def _licence_lines(lines: "list[str]") -> "set[int]":
    k = 0
    while k < len(lines) and (_COMMENT.match(lines[k]) or not lines[k].strip()):
        k += 1
    if not any(_LICENCE.search(lines[j]) for j in range(k)):
        return set()
    return {j for j in range(k) if _COMMENT.match(lines[j]) and (_LICENCE.search(lines[j]) or len(lines[j].strip()) < 90)}


def strip_licence_headers(output: str) -> "tuple[str, dict]":
    """Drop the licence comment block at the top of each file printed in the output."""
    lines = output.split("\n")
    drop: "set[int]" = set()
    start = 0
    for i in range(len(lines) + 1):
        if i == len(lines) or (i > start and _FILE_START.match(lines[i])):
            drop |= {start + j for j in _licence_lines(lines[start:i])}
            start = i
    if not drop:
        return output, {"lines": 0}
    kept, marked = [], False
    for i, l in enumerate(lines):
        if i in drop:
            if not marked:
                kept.append("[licence header omitted]")
                marked = True
            continue
        marked = False
        kept.append(l)
    return "\n".join(kept), {"lines": len(drop)}


def compact_search(output: str) -> "tuple[str, dict]":
    text, s1 = group_search_hits(output)
    text, s2 = strip_licence_headers(text)
    return text, {"chars_in": len(output), "chars_out": len(text), "licence_lines": s2["lines"]}


# --------------------------------------------------------------------------- outlines


def _python_outline(src: str) -> "list[dict]":
    tree = ast.parse(src)
    lines = src.split("\n")
    out = []
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if imports:
        out.append({"name": "imports", "first": imports[0].lineno, "last": imports[-1].end_lineno, "depth": 0,
                    "text": f"{len(imports)} import statements"})

    def sig(node) -> str:
        first = node.lineno
        text = lines[first - 1].strip()
        j = first
        while not text.rstrip().endswith(":") and j < len(lines) and j - first < 8:
            text += " " + lines[j].strip()
            j += 1
        return text

    def visit(nodes, prefix, depth):
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                first = min([n.lineno] + [d.lineno for d in n.decorator_list])
                out.append({"name": prefix + n.name, "first": first, "last": n.end_lineno, "depth": depth, "text": sig(n)})
                if isinstance(n, ast.ClassDef):
                    visit(n.body, prefix + n.name + ".", depth + 1)
            elif depth == 0 and isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                if names:
                    out.append({"name": names[0], "first": n.lineno, "last": n.end_lineno, "depth": 0,
                                "text": lines[n.lineno - 1].strip()})

    visit(tree.body, "", 0)
    return out


_GO_DECL = re.compile(r"^(func|type|var|const|import)\b")
_GO_FUNC = re.compile(r"^func\s+(?:\((?:\w+\s+)?\*?(\w+)[^)]*\)\s*)?(\w+)")
_GO_NAMED = re.compile(r"^(?:type|var|const)\s+(\w+)")
_JS_DECL = re.compile(
    r"^(\s{0,4})(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?(?:async\s+)?"
    r"(function\*?|class|interface|type|enum|const|let|var|namespace)\s+([\w$]+)"
)
_JS_METHOD = re.compile(
    r"^(\s{2,8})(?:(?:public|private|protected|static|readonly|async|get|set|override)\s+)*([\w$]+)\s*(?:<[^>]*>)?\s*\([^;]*$"
)
_C_LIKE = re.compile(r"^(\s{0,4})(?:[\w<>\[\],.*&:]+\s+)+\**([\w$]+)\s*\([^;]*$")
_C_TYPE = re.compile(r"^(\s{0,4})(?:public\s+|private\s+|protected\s+|static\s+|final\s+|abstract\s+)*(class|struct|interface|enum)\s+(\w+)")
_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "else", "do", "try", "new", "function", "with"}


def _brace_end(lines: "list[str]", i: int) -> int:
    depth, seen = 0, False
    for j in range(i, len(lines)):
        s = re.sub(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`[^`]*`|//.*$", "", lines[j])
        depth += s.count("{") + s.count("(") - s.count("}") - s.count(")")
        if "{" in s or "(" in s:
            seen = True
        if seen and depth <= 0:
            return j
        if not seen and j > i and (s.strip() == "" or s.rstrip().endswith(";")):
            return j - 1 if s.strip() == "" else j
    return len(lines) - 1


def _regex_outline(src: str, ext: str) -> "list[dict]":
    lines = src.split("\n")
    out = []
    i = 0
    while i < len(lines):
        l = lines[i]
        e = None
        if ext == ".go":
            if _GO_DECL.match(l):
                if l.startswith("import"):
                    e = {"name": "imports", "depth": 0, "text": "imports"}
                elif l.startswith("func"):
                    m = _GO_FUNC.match(l)
                    name = f"{m.group(1)}.{m.group(2)}" if m and m.group(1) else (m.group(2) if m else "func")
                    e = {"name": name, "depth": 0}
                else:
                    m = _GO_NAMED.match(l)
                    e = {"name": m.group(1) if m else l.split()[0], "depth": 0}
        elif ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
            m = _JS_DECL.match(l)
            if m:
                e = {"name": m.group(3), "depth": 0 if not m.group(1) else 1}
            else:
                m = _JS_METHOD.match(l)
                if m and m.group(2) not in _KEYWORDS and out and out[-1].get("cls"):
                    e = {"name": f"{out[-1]['cls']}.{m.group(2)}", "depth": 1, "method_of": out[-1]["cls"]}
            if l.startswith("import "):
                e = {"name": "imports", "depth": 0, "text": "imports"}
        else:
            m = _C_TYPE.match(l)
            if m:
                e = {"name": m.group(3), "depth": 0 if not m.group(1) else 1}
            else:
                m = _C_LIKE.match(l)
                if m and m.group(2) not in _KEYWORDS:
                    e = {"name": m.group(2), "depth": 0 if not m.group(1) else 1}
        if e is None:
            i += 1
            continue
        end = _brace_end(lines, i)
        if e["name"] == "imports":
            while end + 1 < len(lines) and (lines[end + 1].startswith("import ") or lines[end + 1].strip() == ""):
                end += 1
            if out and out[-1]["name"] == "imports":
                out[-1]["last"] = end + 1
                i = end + 1
                continue
        e.update(first=i + 1, last=end + 1)
        e.setdefault("text", l.strip())
        kind = _JS_DECL.match(l)
        if kind and kind.group(2) == "class":
            e["cls"] = kind.group(3)
        if e.get("method_of") or e.get("cls"):
            out.append(e)
            if e.get("cls"):
                cls_entry = e
                j = i + 1
                while j < end:
                    mm = _JS_METHOD.match(lines[j])
                    if mm and mm.group(2) not in _KEYWORDS:
                        mend = _brace_end(lines, j)
                        out.append({"name": f"{cls_entry['cls']}.{mm.group(2)}", "first": j + 1, "last": mend + 1,
                                    "depth": 1, "text": lines[j].strip()})
                        j = mend + 1
                    else:
                        j += 1
                i = end + 1
                continue
        else:
            out.append(e)
        i = end + 1 if end >= i else i + 1
    return out


def outline(src: str, path: str) -> "list[dict]":
    ext = os.path.splitext(path)[1].lower()
    if ext in (".py", ".pyi"):
        try:
            return _python_outline(src)
        except SyntaxError:
            pass
    if ext in (".py", ".pyi", ".go", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".java", ".kt", ".c", ".h",
               ".cc", ".cpp", ".hpp", ".cs", ".rs", ".scala", ".swift", ".m"):
        try:
            return _regex_outline(src, ext)
        except Exception:
            return []
    return []


def note_context(sent: list, message: dict) -> None:
    """Remember the task and the agent's current step for the selector."""
    if not _CONTEXT["task"]:
        _CONTEXT["task"] = next((m.get("content") or "" for m in sent if m.get("role") == "user"), "")[:3000]
    calls = " ".join(tc["function"]["arguments"] for tc in message.get("tool_calls") or [])
    _CONTEXT["step"] = ((message.get("content") or "") + " " + calls)[-1500:]


_SELECT_PROMPT = """A coding agent is working on this task (beginning shown):
{task}

Its latest step:
{step}

That step printed {path}, lines {first}-{last}, shown below with line numbers.
{numbered}

Which line ranges does the agent need to read in full for its next steps? Pick only what is likely needed, at most a quarter of the lines. Answer with JSON only: {{"ranges": [[start, end], ...]}}"""


def _select(path: str, first: int, block_lines: "list[str]") -> "list[tuple[int, int]]":
    last = first + len(block_lines) - 1
    numbered = "\n".join(f"{first + k}: {l}" for k, l in enumerate(block_lines))
    try:
        resp = SELECTOR(_SELECT_PROMPT.format(task=_CONTEXT["task"], step=_CONTEXT["step"], path=path, first=first,
                                              last=last, numbered=numbered))
        text = (resp.get("message") or {}).get("content") or ""
        ranges = json.loads(text[text.index("{"):text.rindex("}") + 1])["ranges"]
    except Exception:
        return []
    picked, budget = [], int(_SELECT_MAX_SHARE * len(block_lines))
    for r in sorted((int(a), int(b)) for a, b in ranges if int(a) <= int(b)):
        a, b = max(r[0], first), min(r[1], last)
        if a > b:
            continue
        if picked and a <= picked[-1][1] + 1:
            a = picked[-1][0]
            budget += picked[-1][1] - picked[-1][0] + 1
            picked.pop()
        if b - a + 1 > budget:
            b = a + budget - 1
            if b < a:
                break
        picked.append((a, b))
        budget -= b - a + 1
    return picked


def _render_selected(block_lines: "list[str]", first: int, picked: "list[tuple[int, int]]") -> str:
    if not picked:
        return ""
    parts = [f"--- lines {a}-{b}\n" + "\n".join(block_lines[a - first:b - first + 1]) for a, b in picked]
    return "\n[Shown in full as likely relevant:]\n" + "\n".join(parts)


def _render_outline(ref: str, path: str, first_line: int, n_lines: int, entries: "list[dict]", tok: int) -> str:
    head = (f"[{ref}: {path}, lines {first_line}-{first_line + n_lines - 1} (~{tok} tokens), shown as an outline. "
            f"Get the parts you need in one call: show_elided(parts=[\"{ref}:<name>\", \"{ref}:<start>-<end>\", ...]).]")
    rows = []
    for e in entries[:_OUTLINE_MAX_ENTRIES]:
        text = e["text"] if len(e["text"]) <= _SIG_MAX_CHARS else e["text"][:_SIG_MAX_CHARS] + " ..."
        rows.append(f"{'  ' * e['depth']}L{e['first']}-{e['last']}  {text}")
    if len(entries) > _OUTLINE_MAX_ENTRIES:
        rows.append(f"... {len(entries) - _OUTLINE_MAX_ENTRIES} more definitions (ask by line range)")
    if not entries:
        rows.append("(no outline for this file type; ask by line range)")
    return head + "\n" + "\n".join(rows)


# --------------------------------------------------------------------------- locating files in an output


def _read_targets(command: str, workdir: "str | None") -> "list[tuple[str, int | None, int | None]]":
    """(path, first line, last line) of files the command prints whole or in part."""
    cwd = workdir or os.getcwd()
    found = []
    for seg in _SEP.split(command):
        try:
            argv = shlex.split(seg)
        except ValueError:
            continue
        if not argv:
            continue
        if argv[0] == "cd" and len(argv) > 1:
            cwd = os.path.normpath(os.path.join(cwd, os.path.expanduser(argv[1])))
            continue
        if argv[0] not in _READ_CMDS:
            continue
        rng = (None, None)
        if argv[0] == "sed":
            for a in argv[1:]:
                m = re.fullmatch(r"(\d+),(\d+)p", a)
                if m:
                    rng = (int(m.group(1)), int(m.group(2)))
        files = [a for a in argv[1:] if not a.startswith("-") and not re.fullmatch(r"\d+,\d+p|\d+", a)]
        for f in files:
            p = f if os.path.isabs(f) else os.path.join(cwd, f)
            if os.path.isfile(p):
                found.append((p, *rng))
    return found


def _next_ref() -> str:
    return f"E{len(REGISTRY) + 1}"


def _file_runs(out_lines: "list[str]", file_lines: "list[str]") -> "list[tuple[int, int, int]]":
    """(output start, file start, length) of maximal runs where the output repeats the file's lines."""
    pos: "dict[str, list[int]]" = {}
    for j, l in enumerate(file_lines):
        if len(l.strip()) >= 8:
            pos.setdefault(l, []).append(j)
    runs, k = [], 0
    while k < len(out_lines):
        best = None
        for j in pos.get(out_lines[k], [])[:50]:
            n = 0
            while k + n < len(out_lines) and j + n < len(file_lines) and out_lines[k + n] == file_lines[j + n]:
                n += 1
            if best is None or n > best[2]:
                best = (k, j, n)
        if best and best[2] >= 20:
            k0, j0, n = best
            floor = runs[-1][0] + runs[-1][2] if runs else 0
            while k0 > floor and j0 > 0 and out_lines[k0 - 1] == file_lines[j0 - 1]:
                k0, j0, n = k0 - 1, j0 - 1, n + 1
            runs.append((k0, j0, n))
            k = k0 + n
        else:
            k += 1
    return runs


def _elide_file_blocks(output: str, command: str, workdir: "str | None") -> "tuple[str, int]":
    n = 0
    for path, _, _ in dict.fromkeys(_read_targets(command, workdir)):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                src = f.read()
        except OSError:
            continue
        file_lines = src.split("\n")
        out_lines = output.split("\n")
        entries_all = None
        for k, j, length in reversed(_file_runs(out_lines, file_lines)):
            block = "\n".join(out_lines[k:k + length])
            if tokens(block) < _BLOCK_MIN_TOKENS:
                continue
            if entries_all is None:
                entries_all = outline(src, path)
            first, last = j + 1, j + length
            entries = [e for e in entries_all if e["last"] >= first and e["first"] <= last]
            ref = _next_ref()
            REGISTRY[ref] = {"kind": "file", "path": path, "first": first, "last": last, "text": block}
            shown = _render_outline(ref, path, first, length, entries, tokens(block))
            if SELECTOR is not None:
                shown += _render_selected(out_lines[k:k + length], first, _select(path, first, out_lines[k:k + length]))
            out_lines[k:k + length] = shown.split("\n")
            n += 1
        output = "\n".join(out_lines)
    return output, n


def _elide_hits(output: str) -> "tuple[str, int]":
    lines = output.split("\n")
    by_file: "dict[str, list[int]]" = {}
    grouped_path = None
    for i, l in enumerate(lines):
        m = _HIT.match(l)
        if m:
            by_file.setdefault(m.group(1), []).append(i)
            continue
        g = re.match(r"^  (\d+)[:-]", l)
        if g and grouped_path:
            by_file.setdefault(grouped_path, []).append(i)
            continue
        grouped_path = l[:-1] if l.endswith(":") and "/" in l and " " not in l else None
    drop: "set[int]" = set()
    notes: "dict[int, str]" = {}
    n = 0
    for path, idxs in by_file.items():
        if len(idxs) < _HITS_ELIDE_MIN:
            continue
        ref = _next_ref()
        REGISTRY[ref] = {"kind": "hits", "path": path, "text": "\n".join(lines[i] for i in idxs)}
        drop |= set(idxs[_HITS_KEEP:])
        notes[idxs[_HITS_KEEP - 1]] = (f"  [{ref}: {len(idxs) - _HITS_KEEP} more hits in {path}; "
                                       f"show_elided(parts=[\"{ref}:*\"])]")
        n += 1
    if not n:
        return output, 0
    out = []
    for i, l in enumerate(lines):
        if i in drop:
            continue
        out.append(l)
        if i in notes:
            out.append(notes[i])
    return "\n".join(out), n


def elide_output(command: str, workdir: "str | None", output: str) -> "tuple[str, dict]":
    tok_in = tokens(output)
    if not ELIDE_MIN_TOKENS or tok_in < ELIDE_MIN_TOKENS:
        return output, {"files": 0, "hit_files": 0, "tokens_in": tok_in, "tokens_out": tok_in}
    output, nf = _elide_file_blocks(output, command, workdir)
    nh = 0
    if tokens(output) >= ELIDE_MIN_TOKENS:
        output, nh = _elide_hits(output)
    return output, {"files": nf, "hit_files": nh, "tokens_in": tok_in, "tokens_out": tokens(output)}


def elide_read(path: str, src: str) -> "str | None":
    """Outline for a whole-file read tool call, or None to show it as is."""
    if not ELIDE_MIN_TOKENS or tokens(src) < ELIDE_MIN_TOKENS:
        return None
    lines = src.split("\n")
    ref = _next_ref()
    REGISTRY[ref] = {"kind": "file", "path": path, "first": 1, "last": len(lines), "text": src}
    shown = _render_outline(ref, path, 1, len(lines), outline(src, path), tokens(src))
    if SELECTOR is not None:
        shown += _render_selected(lines, 1, _select(path, 1, lines))
    return shown


# --------------------------------------------------------------------------- show_elided


SHOW_ELIDED_PARAMS = {
    "type": "object",
    "properties": {
        "parts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Pieces to return, each '<ref>:<name>' (a name from the outline, e.g. 'E3:Config.load'), "
            "'<ref>:<start>-<end>' (file line numbers) or '<ref>:*' (all of it). Ask for every piece you need "
            "in one call.",
        }
    },
    "required": ["parts"],
}
SHOW_ELIDED_DESCRIPTION = "Return pieces of tool outputs that were shown as outlines or cut short (references like E3)."


def _numbered(lines: "list[str]", first: int) -> str:
    return "\n".join(f"{first + k}: {l}" for k, l in enumerate(lines))


def show_elided(arguments_json: str) -> str:
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except (json.JSONDecodeError, TypeError):
        args = {}
    parts = args.get("parts") if isinstance(args, dict) else None
    if isinstance(parts, str):
        parts = [parts]
    if not parts:
        return json.dumps({"error": "pass parts, e.g. [\"E3:Config.load\", \"E3:120-180\"]"})
    chunks, errors = [], []
    for part in parts:
        ref, _, sel = str(part).partition(":")
        ref, sel = ref.strip(), sel.strip() or "*"
        item = REGISTRY.get(ref)
        if item is None:
            errors.append(f"{part}: unknown reference {ref}")
            continue
        if item["kind"] == "hits":
            chunks.append(f"== {ref} {item['path']}\n{item['text']}")
            continue
        path = item["path"]
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                src = f.read()
        except OSError:
            src = None
        lines = (src if src is not None else item["text"]).split("\n")
        base = 1 if src is not None else item["first"]
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", sel)
        if sel == "*":
            a, b, label = item["first"], item["last"], "all"
        elif m:
            a, b, label = int(m.group(1)), int(m.group(2)), f"{m.group(1)}-{m.group(2)}"
        else:
            entries = outline(src, path) if src is not None else []
            hit = [e for e in entries if e["name"] == sel] or [e for e in entries if e["name"].endswith("." + sel)] \
                or [e for e in entries if e["name"].split(".")[-1] == sel.split(".")[-1]]
            if not hit:
                close = [e["name"] for e in entries if sel.split(".")[-1].lower() in e["name"].lower()][:8]
                errors.append(f"{part}: no definition named {sel!r} in {path}" + (f"; close: {close}" if close else ""))
                continue
            a, b, label = hit[0]["first"], hit[0]["last"], hit[0]["name"]
        a, b = max(a, base), min(b, base + len(lines) - 1)
        chunks.append(f"== {ref} {path} lines {a}-{b} ({label})\n" + _numbered(lines[a - base:b - base + 1], a))
    result = {"output": "\n\n".join(chunks)}
    if errors:
        result["errors"] = errors
    return json.dumps(result)
