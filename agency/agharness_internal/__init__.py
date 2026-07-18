"""Implementation details behind agency's `agharness` engine seam --
`agproxy_llm`/`agproxy_ptrace` (the LLM gateway and syscall-level
supervisor) and `agharness_backends/` (one concrete off-the-shelf harness
CLI per file). Nothing in here is part of the public API; `agharness.py`
(one level up) and `agent`/`agskill`'s `engine=` seam are the intended
entry points -- see docs/agharness.md.
"""
