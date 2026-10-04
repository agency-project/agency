"""Flag broad `except` handlers that silently swallow the exception.

    python scripts/check_swallows.py [paths...]   # default: agency/

ruff's S110/S112 only see bodies that are exactly `pass` or `continue`; this
also catches `return`, assignments and the like. A handler counts as handled
if it re-raises, uses the caught exception, or reports it (print, logging,
warnings, stderr). A deliberate swallow says why on its `except` line:
`except Exception:  # swallow-ok: <reason>`.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

MARKER = "swallow-ok:"
BROAD = {"Exception", "BaseException"}
REPORT_FUNCS = {
    "print",
    "warn",
    "warning",
    "error",
    "exception",
    "critical",
    "info",
    "debug",
    "log",
}
REPORT_ROOTS = {"logging", "logger", "log", "_log", "_logger", "LOGGER", "warnings"}


def _is_broad(node: ast.expr | None) -> bool:
    if node is None:
        return True
    if isinstance(node, ast.Name):
        return node.id in BROAD
    if isinstance(node, ast.Attribute):
        return node.attr in BROAD
    if isinstance(node, ast.Tuple):
        return any(_is_broad(e) for e in node.elts)
    return False


def _root_name(node: ast.expr) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _reports(call: ast.Call) -> bool:
    func = call.func
    name = (
        func.id
        if isinstance(func, ast.Name)
        else func.attr
        if isinstance(func, ast.Attribute)
        else ""
    )
    # print_exc/print_exception (traceback) and logging.Handler.handleError report too.
    if (
        name in REPORT_FUNCS
        or "report" in name.lower()
        or name.startswith("print")
        or name == "handleError"
    ):
        return True
    if isinstance(func, ast.Attribute):
        if _root_name(func) in REPORT_ROOTS:
            return True
        # sys.stderr.write(...)
        if (
            name == "write"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "stderr"
        ):
            return True
    return False


def _handled(handler: ast.ExceptHandler) -> bool:
    for stmt in handler.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Raise):
                return True
            if handler.name and isinstance(node, ast.Name) and node.id == handler.name:
                return True
            if isinstance(node, ast.Call) and _reports(node):
                return True
    return False


def _marked(lines: list[str], handler: ast.ExceptHandler) -> bool:
    line = lines[handler.lineno - 1]
    reason = line.split(MARKER, 1)[1].strip() if MARKER in line else ""
    return bool(reason)


def find_swallows(path: Path) -> list[str]:
    text = path.read_text()
    lines = text.splitlines()
    found = []
    for node in ast.walk(ast.parse(text, filename=str(path))):
        if isinstance(node, ast.ExceptHandler) and _is_broad(node.type):
            if not _handled(node) and not _marked(lines, node):
                found.append(f"{path}:{node.lineno}: broad except swallows the exception")
    return found


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv] or [Path("agency")]
    files = [f for r in roots for f in ([r] if r.is_file() else sorted(r.rglob("*.py")))]
    found = [hit for f in files for hit in find_swallows(f)]
    for hit in found:
        print(hit)
    if found:
        print(
            f"{len(found)} silent swallow(s); report or re-raise, or mark with '# {MARKER} <reason>'."
        )
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
