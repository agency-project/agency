"""Off-the-shelf coding-agent harness backends -- one file per harness
(claude_code, codex, opencode), selected via `agharness_backend.for_config()`.

Mirrors the `llm`/`sandbox` package shape exactly:
`agharness_backend.py` holds the abstract base class and selection logic;
each concrete backend lives in its own sibling module and imports
`agharness_backend` to subclass it.
"""

from .agharness_backend import agharness_backend

__all__ = ["agharness_backend"]
