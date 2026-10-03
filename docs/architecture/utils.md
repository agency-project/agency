# utils — workflows and shared runtime helpers

`agency/utils/` contains the small Python workflow helpers exported by Agency and the lower-level functions shared by runtime components.

## Application workflow helpers

[agmap.py](../../agency/utils/agmap.py) runs ordinary Python functions concurrently on background threads. Each call returns an `agtask`, a pending `agdata` subclass. The default call waits for results; asynchronous mode returns the pending tasks so a workflow can join them later. This lets deterministic work such as validation or sandbox preparation participate alongside agent requests.

[agsync.py](../../agency/utils/agsync.py) joins agents, teams and mapped tasks. Team joining waits for workflow threads before collecting team agents, so agents created dynamically during the workflow are included. Agent joining waits for the context chain to settle. Generic pending `agdata` can instead be joined through its own wait API.

These helpers compose work in application Python. They do not add requests to the orchestrator or share its engine admission limit; agent requests made inside a mapped function still go through normal scheduling. Their threads propagate profiling context so the work appears under its calling workflow.

## Shared runtime support

[agutil.py](../../agency/utils/agutil.py) supplies stream batching, timeout/error helpers, signal cleanup, hardware detection and per-run directory/socket paths. The path helpers give gateways, sandbox state, temporary configuration homes and scratch files a common process-owned layout. Cleanup checks owner/process liveness when handling orphaned state.

The same module locates the installed Agency source for the sandbox's read-only mount and detects available CPU, memory and GPU resources for backend/resource-pool setup. These are shared mechanisms; scheduling policy and runtime lifecycle remain with their owning packages.

The [workflow reference](../api/workflows.md) documents the application-facing helpers and `agteam` composition.
