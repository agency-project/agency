"""Lesson 8: save an agent and restore it in a fresh process."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from agency import Agent, agdata, agskill

from _common import close_sandboxes, example_wait_timeout, run_example, tutorial_config


def restore_one(path: Path) -> None:
    cfg, _ = tutorial_config("08_restore_one")
    restored = Agent.load(path, agconfig=cfg)
    try:
        print(f"restored agent: {restored.agname}")
        print(f"restored file: {restored.sandbox.read_file('/workspace/note.txt')}")
        print(f"history messages: {len(restored.history.messages)}")
    finally:
        close_sandboxes([restored])


def main() -> None:
    cfg, run_dir = tutorial_config("08_checkpoints")
    worker = Agent("checkpointed", agconfig=cfg, harness="native")
    skill = agskill(
        name="write_note",
        prompt=(
            "Write hello to /workspace/note.txt and return a short confirmation."
            "\nAfter submitting all output fields, finish with a short confirmation."
        ),
        input_schema=agdata(request=str),
        output_schema=agdata(confirmation=str),
    )
    checkpoint = run_dir / "agent.ckpt"
    try:
        result = worker.run(skill, agdata(request="Save a note."))
        result.wait(timeout=example_wait_timeout())
        print(result.confirmation)
        worker.context.resolve_prev_dependencies()
        worker.save(checkpoint)
        print(f"saved checkpoint: {checkpoint}")

        env = os.environ.copy()
        env["AGENCY_EXAMPLE_RUN_ROOT"] = str(run_dir / "restore")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--restore-one", str(checkpoint)],
            check=True,
            env=env,
        )
        print(f"artifacts: {run_dir}")
    finally:
        close_sandboxes([worker])
        if os.environ.get("AGENCY_KEEP_EXAMPLE_CHECKPOINTS") != "1":
            checkpoint.unlink(missing_ok=True)


def dispatch() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-one", type=Path)
    args = parser.parse_args()
    if args.restore_one:
        restore_one(args.restore_one)
    else:
        main()


if __name__ == "__main__":
    run_example(dispatch)
