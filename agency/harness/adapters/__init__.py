"""Off-the-shelf coding-agent harness backends -- one file per harness
(claude_code, codex, opencode), selected via `agharness_backend.for_config()`.

Mirrors the `llm`/`sandbox` package shape exactly:
`base.py` holds the abstract base class, the shared config-field class, and
selection logic; each concrete backend lives in its own sibling module and
imports `agharness_backend` from `.base` to subclass it.
"""

from .agharness_backend import agharness_backend, agHarnessConfig, AgHarnessFields

__all__ = ["agharness_backend", "agHarnessConfig", "AgHarnessFields"]
