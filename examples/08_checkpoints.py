"""Lesson 8: save and restore one agent or the whole live registry."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from agency import Agent, agdata, agskill

from _common import close_sandboxes, run_example, tutorial_config


WRITE_STATE = agskill(
    name="write_checkpoint_state",
    system_prompt=(
        "Use the shell to write the exact text CHECKPOINTED to /workspace/checkpoint.txt, "
        "then return that same word as confirmation."
    ),
    input_schema=agdata(request=str),
    output_schema=agdata(confirmation=str),
)


def restore_one(path: Path) -> None:
    cfg, _ = tutorial_config("08_restore_one")
    restored = Agent.load(path, agconfig=cfg)
    assert restored.sandbox is not None
    assert restored.sandbox.read_file("/workspace/checkpoint.txt") == "CHECKPOINTED"
    print(
        f"restored one: {restored.agname}; history={len(restored.history.messages)}; "
        "filesystem=CHECKPOINTED"
    )
    close_sandboxes([restored])


def save_registry(directory: Path) -> None:
    cfg, _ = tutorial_config("08_save_all")
    agents = [Agent("registry-alpha", agconfig=cfg), Agent("registry-beta", agconfig=cfg)]
    paths = Agent.save_all(directory)
    assert len(paths) == 2
    sizes = {path.name: path.stat().st_size for path in paths}
    print(f"save_all metadata checkpoints: {sizes}")
    close_sandboxes(agents)


def restore_registry(directory: Path) -> None:
    cfg, _ = tutorial_config("08_load_all")
    restored = Agent.load_all(directory, agconfig=cfg)
    assert len(restored) == 2
    assert set(restored).issubset(set(Agent.all()))
    print(f"load_all: {[current.agname for current in restored]}")
    close_sandboxes(restored)


def main() -> None:
    cfg, run_dir = tutorial_config("08_checkpoints")
    # The native harness makes the shell mutation explicit before checkpointing.
    worker = Agent("checkpointed", agconfig=cfg, harness="native")
    result = worker.run(WRITE_STATE, agdata(request="Persist state."))
    assert result.confirmation == "CHECKPOINTED"
    result.wait(timeout=300)
    worker.context.resolve_prev_dependencies()
    assert not worker.context.is_pending()
    print("checkpoint precondition: result and ordered context are fully settled")

    checkpoint = run_dir / "checkpointed.ckpt"
    worker.save(checkpoint)
    assert checkpoint.exists()
    checkpoint_size_mb = checkpoint.stat().st_size / (1024 * 1024)
    print(f"sandbox checkpoint size: {checkpoint_size_mb:.1f} MiB")

    env = os.environ.copy()
    env["AGENCY_EXAMPLE_RUN_ROOT"] = str(run_dir / "child-runs")
    script = Path(__file__).resolve()
    subprocess.run(
        [sys.executable, str(script), "--restore-one", str(checkpoint)], check=True, env=env
    )
    if os.environ.get("AGENCY_KEEP_EXAMPLE_CHECKPOINTS") != "1":
        checkpoint.unlink()
        print(
            "removed temporary sandbox checkpoint after restore; "
            "set AGENCY_KEEP_EXAMPLE_CHECKPOINTS=1 to retain it"
        )

    registry_dir = run_dir / "registry"
    subprocess.run(
        [sys.executable, str(script), "--save-registry", str(registry_dir)], check=True, env=env
    )
    subprocess.run(
        [sys.executable, str(script), "--restore-registry", str(registry_dir)], check=True, env=env
    )
    print(f"live registry contains original: {worker in Agent.all()}")
    print(f"artifacts: {run_dir}")
    close_sandboxes([worker])


def dispatch() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-one", type=Path)
    parser.add_argument("--save-registry", type=Path)
    parser.add_argument("--restore-registry", type=Path)
    args = parser.parse_args()
    if args.restore_one:
        restore_one(args.restore_one)
    elif args.save_registry:
        save_registry(args.save_registry)
    elif args.restore_registry:
        restore_registry(args.restore_registry)
    else:
        main()


if __name__ == "__main__":
    run_example(dispatch)
