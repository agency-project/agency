"""Standalone native harness -- a self-contained ReAct coding-agent CLI,
independent of agency's host-side runtime.

**Invocation -- deliberately NOT `python3 -m agency.native_harness`:**
`agency/__init__.py` (the parent package) eagerly imports `agent.py`,
`agskill.py`, `agsandbox.py`, ... which pulls in `agllm.py`'s `import
openai`/`anthropic`/`boto3` at module level -- exactly the host-side
dependency weight this package exists to avoid needing. Since Python always
runs a package's `__init__.py` before any of its submodules, importing this
package AS `agency.native_harness` (via `-m agency.native_harness` or
`import agency.native_harness`) would trigger that whole chain regardless
of anything this package's own code does.

The fix: run it with `agency/` itself (not the repo root) on `PYTHONPATH`,
so `native_harness` resolves as its OWN top-level package -- a sibling of
`agency/__init__.py`, never an ancestor of it:

    PYTHONPATH=<repo>/agency python3 -m native_harness.cli -p "<prompt>" --model <name> ...

Confirmed empirically: with `agency/` on `PYTHONPATH` this way, `agency`
never appears in `sys.modules` at all, and neither does `openai`/
`anthropic`/`boto3` -- only `native_harness` and whatever
`llm_client.py`/`mcp_client.py` actually need (`httpx`, `httpx2`, `mcp`).
Every submodule here (`tools.py`, `compaction.py`, ...) is a normal,
ordinary member of this package -- no separate raw-file-path loading is
needed for anything, since nothing outside this package depends on their
contents.

A future agency backend launching this as a real harness (mirroring
`claude_code.py`'s launch of the real `claude` binary) would set
`PYTHONPATH` this same way -- inside a container, already trivial, since
the whole `agency/` directory is already bind-mounted read-only into every
container-backed sandbox (`agutil.agency_package_dir()`/
`AGENCY_PACKAGE_CONTAINER_MOUNT`, the same mechanism `agmanager_harness`
already rides)."""

from __future__ import annotations

__all__: "list[str]" = []
