"""A second read of code-heavy reports: the supervisor's model, in a separate call, lists suspicious lines.

    review_report(text, paths, llm, model) -> (note or None, stats)

The supervisor reading a report mid-task misses a wrong line it would catch when
asked only to review that code, so the harness asks that question separately and
appends the answer to the report.
"""

from __future__ import annotations

import sys
import time

from .outline import code_line_count

MIN_CODE_LINES = 20
HEADER = "Lines in the code above that may be wrong (automated review):"
PROMPT = (
    "Below is a report that contains source code read from a Python library.\n\n{report}\n\n"
    "List every line of code in it that looks wrong or differs from the standard upstream implementation of this "
    "library, one per line, as `file or function: the line — reason`. If nothing looks wrong, reply `none`."
)


def review_report(text: str, paths: "list[str]", llm, model: str) -> "tuple[str | None, dict]":
    stats = {"code_lines": code_line_count(text, paths), "input_tokens": 0, "output_tokens": 0}
    if stats["code_lines"] < MIN_CODE_LINES:
        return None, stats
    resp = llm.dispatch(model, [{"role": "user", "content": PROMPT.format(report=text)}])
    if "error" in resp:
        print(
            f"[tandem_harness] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} WARNING: report review failed: {resp['error']}",
            file=sys.stderr,
        )
        return None, stats
    usage = resp.get("usage") or {}
    stats["input_tokens"] = usage.get("prompt_tokens") or 0
    stats["output_tokens"] = usage.get("completion_tokens") or 0
    note = ((resp.get("message") or {}).get("content") or "").strip()
    if not note or note.strip("`. ").lower() == "none":
        return None, stats
    return note, stats
