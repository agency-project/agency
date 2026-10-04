"""Timestamped stderr reports for failures that are handled but must stay visible."""

from __future__ import annotations

import sys
import time


def report_failure(tag: str, what: str, exc: BaseException) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[{tag}] {stamp} {what}: {exc!r}", file=sys.stderr, flush=True)
