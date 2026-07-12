"""Tests for every script under examples/.

Two tiers:
  - test_example_imports: always runs. Just imports each file (catches
    syntax errors, broken imports, stale references) without invoking it.
  - test_example_runs_live: only runs when VLLM_BASE_URL and VLLM_API_KEY
    are set, since examples need a live LLM endpoint and a real sandbox
    container to actually execute.
"""

import importlib.util
import os
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.py"))

LIVE_LLM = bool(os.environ.get("VLLM_BASE_URL")) and bool(os.environ.get("VLLM_API_KEY"))
LIVE_SKIP_REASON = (
    "set VLLM_BASE_URL and VLLM_API_KEY to run examples end-to-end against a live LLM"
)


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=[p.stem for p in EXAMPLE_FILES])
def test_example_imports(path):
    spec = importlib.util.spec_from_file_location(f"examples.{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


@pytest.mark.skipif(not LIVE_LLM, reason=LIVE_SKIP_REASON)
@pytest.mark.timeout(600)
@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=[p.stem for p in EXAMPLE_FILES])
def test_example_runs_live(path):
    # Every example wraps its run in agwebui.run(fn, ...), which spawns a
    # dashboard server and then blocks forever waiting for Ctrl+C (linger=True).
    # Bypass the dashboard and just call fn() directly so the test can complete.
    # Also reset sys.argv so examples that read sys.argv[1:] (e.g. for an
    # optional topic/image path) don't pick up pytest's own CLI args.
    with (
        patch("agency.agwebui.agwebui.run", side_effect=lambda fn, *a, **kw: fn()),
        patch.object(sys, "argv", [str(path)]),
    ):
        runpy.run_path(str(path), run_name="__main__")
