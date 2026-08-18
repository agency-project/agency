"""Loads `agtool_pure.py`/`agllm_pure.py` by raw file path, never via
`import agency...`.

This package (`native_harness`) is meant to run with no dependency on
agency's HOST-side runtime -- `agent.py`, `agconfig.py`, the five old
per-process singletons, `harness`'s backends -- so it can be
launched by agency (as a real harness, bridged to `agmanager_harness`) or
run completely standalone from a bash prompt against a real provider, with
no agency process involved at all. Importing `agency` normally (`import
agency.agtool_pure`) would trigger `agency/__init__.py`'s own import chain,
which pulls in `agllm.py`'s `import openai`/`anthropic`/`boto3` and
eventually `agent.py` -- exactly the host-side weight this package exists
to avoid needing. `agtool_pure.py`/`agllm_pure.py` are already written with
zero relative/agency imports for exactly this reason (see their own
docstrings) -- loading them by file path sidesteps the package `__init__.py`
chain entirely while still reusing the actual, single-source-of-truth
algorithm (fuzzy-match edit, pagination, glob/grep parsing, compaction),
not a hand-copied, driftable duplicate.

This only works because `native_harness` lives INSIDE the `agency` repo,
as a sibling of `agtool_pure.py`/`agllm_pure.py` -- launched by agency (the
whole package directory is already bind-mounted into every container-backed
sandbox) or run from a plain checkout of this repo. It does NOT work if
`native_harness/` is copied out on its own with nothing else -- that's a
deliberate scope boundary, not an oversight: "standalone" here means "no
agency HOST PROCESS required," not "zero sibling files in the repo.\""""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_AGENCY_DIR = Path(__file__).resolve().parent.parent


def _load_by_path(module_name: str, relative_path: str) -> ModuleType:
    path = _AGENCY_DIR / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_agtool_pure() -> ModuleType:
    return _load_by_path("agtool_pure", "agtool_pure.py")


def load_agllm_pure() -> ModuleType:
    return _load_by_path("agllm_pure", "agllm_pure.py")


__all__ = ["load_agtool_pure", "load_agllm_pure"]
