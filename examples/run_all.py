"""Run every numbered tutorial as an isolated acceptance test."""

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
    args = parser.parse_args()

    examples_dir = Path(__file__).resolve().parent
    lessons = sorted(examples_dir.glob("[0-9][0-9]_*.py"))
    if args.only:
        wanted = set(args.only)
        lessons = [lesson for lesson in lessons if lesson.name[:2] in wanted]
    if not lessons:
        raise SystemExit("No matching tutorial lessons")

    env = os.environ.copy()
    env.setdefault("AGENCY_PROFILE", "0")
    started = time.monotonic()
    for index, lesson in enumerate(lessons, 1):
        print(f"\n[{index}/{len(lessons)}] {lesson.name}", flush=True)
        subprocess.run([sys.executable, str(lesson)], cwd=examples_dir.parent, env=env, check=True)
    print(f"\nAll {len(lessons)} tutorials passed in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
