"""Static and opt-in end-to-end checks for the numbered tutorial suite."""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

import agency


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
LESSONS = sorted(EXAMPLES_DIR.glob("[0-9][0-9]_*.py"))
SUPPORT_FILES = [EXAMPLES_DIR / "_common.py", EXAMPLES_DIR / "run_all.py"]


@pytest.mark.parametrize("path", [*LESSONS, *SUPPORT_FILES], ids=lambda path: path.stem)
def test_example_imports(path: Path) -> None:
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"agency_tutorial_{path.stem}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(EXAMPLES_DIR))


def test_tutorial_is_numbered_and_documented() -> None:
    assert [path.name[:2] for path in LESSONS] == [f"{number:02d}" for number in range(1, 11)]
    readme = (EXAMPLES_DIR / "README.md").read_text()
    for lesson in LESSONS:
        assert lesson.name in readme


def test_tutorial_imports_every_public_api() -> None:
    imported: set[str] = set()
    for path in [*LESSONS, *SUPPORT_FILES]:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "agency":
                imported.update(alias.name for alias in node.names)
    assert set(agency.__all__) <= imported


@pytest.mark.skipif(
    os.environ.get("AGENCY_RUN_EXAMPLES_LIVE") != "1",
    reason="set AGENCY_RUN_EXAMPLES_LIVE=1 and configure a live LLM to run the tutorial suite",
)
@pytest.mark.timeout(3600)
def test_example_suite_live() -> None:
    subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "run_all.py")],
        cwd=EXAMPLES_DIR.parent,
        env=os.environ.copy(),
        check=True,
    )
