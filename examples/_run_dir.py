"""Shared utility: create a timestamped run directory under runs/."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent


def make_run_dir(example_name: str) -> Path:
    """Create and return runs/YYYY-MM-DD_HH-MM-SS_<example_name>/."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = _REPO_ROOT / "runs" / f"{ts}_{example_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
