"""Run numbered examples in separate processes and report any failures."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="lesson numbers, for example 01 04 10")
    parser.add_argument(
        "--example-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="arguments to pass to a single selected example",
    )
    args = parser.parse_args()

    examples_dir = Path(__file__).resolve().parent
    lessons = sorted(examples_dir.glob("[0-9][0-9]_*.py"))
    if args.only:
        wanted = set(args.only)
        lessons = [lesson for lesson in lessons if lesson.name[:2] in wanted]
    else:
        # SWE-bench needs a prepared task and image, supplied explicitly via --only 12.
        lessons = [lesson for lesson in lessons if lesson.name[:2] != "12"]
    if args.example_args and len(lessons) != 1:
        parser.error("--example-args requires exactly one lesson selected with --only")
    if not lessons:
        raise SystemExit("No matching tutorial lessons")

    env = os.environ.copy()
    env.setdefault("AGENCY_PROFILE", "0")
    started = time.monotonic()
    failed = []
    for index, lesson in enumerate(lessons, 1):
        print(f"\n[{index}/{len(lessons)}] {lesson.name}", flush=True)
        result = subprocess.run(
            [sys.executable, str(lesson), *args.example_args], cwd=examples_dir.parent, env=env
        )
        if result.returncode:
            failed.append(lesson.name)
    print(f"\nRan {len(lessons)} examples in {time.monotonic() - started:.1f}s", flush=True)
    if failed:
        raise SystemExit(f"Failed examples: {', '.join(failed)}")
    print("All examples completed successfully", flush=True)


if __name__ == "__main__":
    main()
