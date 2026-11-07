# Documentation archive

These documents predate the concise user guides. They are preserved for
historical context and may describe outdated APIs, defaults, or behavior.
Start with the [current README](../../README.md) for installation and usage.
Relative Markdown links in archived snapshots have been adjusted for this location.

- Previous landing pages: [root README](root-README.md) and
  [examples README](examples-README.md).
- API and implementation notes: [invocation API](Invocation_API.md),
  [redirects](Redirect.md), [orchestrator](Design_orchestrator.md),
  [execution loop](Design_execution_loop.md), [PTY harnesses](PTY.md),
  [automatic tracing](Design_profiler_automatic_tracing.md), and
  [profiler coverage](Profiler_coverage.md).
- Historical reports: [PTY/Kimi state](current-state.md),
  [Kimi/checkpoint merge validation](checkpoint-kimi-merge-validation.md), and
  [host setup validation](setup-host-validation.md).

Active operational docs remain outside the archive:
[host setup](../guides/host-setup.md), [checkpoints](../guides/checkpoints.md),
[raw trace viewer](../guides/profiling.md#raw-trace-viewer), and [profiler guides](../guides/profiling/views.md).
Contributor instructions, CI files, benchmark instructions, and figure tooling
have kept their existing locations.

## Stage 1 path mapping

Snapshots preserve the content present before this stage, including prior
uncommitted documentation edits. Only relative Markdown destinations are
adjusted so links work from the archive. Earlier archived versions above
remain intact.

| Original path | Snapshot | Current entry point or replacement |
| --- | --- | --- |
| `README.md` | [Before Stage 1](root-README-stage1.md); [earlier version](root-README.md) | [Root README](../../README.md) and [getting started](../getting-started.md) |
| `examples/README.md` | [Before Stage 1](examples-README-stage1.md); [earlier version](examples-README.md) | [Task-oriented example index](../../examples/README.md) |
| `docs/observability.md` | [Before Stage 1](observability-stage1.md) | [Observability](../guides/profiling.md); [profiler startup](../guides/profiling.md) |
| `docs/old/README.md` | [Index before Stage 1](archive-index-stage1.md) | This index |
| `docs/Invocation_API.md` | [Invocation API](Invocation_API.md) | Unreplaced; accessible from the [documentation index](../index.md) |
| `docs/Redirect.md` | [Redirects](Redirect.md) | Unreplaced reference |
| `docs/Design_execution_loop.md` | [Execution loop](Design_execution_loop.md) | Unreplaced reference |
| `docs/Design_orchestrator.md` | [Orchestrator](Design_orchestrator.md) | Unreplaced reference |
| `docs/PTY.md` | [PTY architecture](PTY.md) | Unreplaced reference |
| `docs/Design_profiler_automatic_tracing.md` | [Automatic tracing](Design_profiler_automatic_tracing.md) | Unreplaced reference |
| `docs/Profiler_coverage.md` | [Profiler coverage](Profiler_coverage.md) | Unreplaced reference |
| `current-state.md` | [Historical state](current-state.md) | Historical report; no replacement |
| `docs/checkpoint-kimi-merge-validation.md` | [Merge validation](checkpoint-kimi-merge-validation.md) | Historical report; no replacement |
| `docs/setup-host-validation.md` | [Host validation](setup-host-validation.md) | Historical report; [host setup](../guides/host-setup.md) remains active |

`docs/WebUI_profiler.md`, `docs/profiler/`, `docs/configuration.md`, and
`docs/development.md` retain their existing locations and content. The compact
[documentation index](../index.md) links the current onboarding and deeper
references; [Stage 1 notes](documentation-refactor.md) record checks and follow-ups.

## Stage 2 path mapping

These snapshots preserve the working tree immediately before Stage 2 edits;
only relative Markdown destinations are relocated. Earlier archives are intact.

| Original | Snapshot | Replacement |
| --- | --- | --- |
| `README.md` | [Pre-Stage 2](stage2-README.md) | [README](../../README.md) |
| `CONTRIBUTING.md` | [Pre-Stage 2](stage2-CONTRIBUTING.md) | [Contributing](../../CONTRIBUTING.md) |
| `docs/index.md` | [Pre-Stage 2](stage2-docs-index.md) | [Docs home](../index.md) |
| `docs/getting-started.md` | [Pre-Stage 2](stage2-docs-getting-started.md) | [Getting started](../getting-started.md), [results](../api/results.md) |
| `docs/configuration.md` | [Pre-Stage 2](stage2-docs-configuration.md) | [Provider guide](../guides/configuration.md), [full configuration reference](../api/configuration.md) |
| `docs/development.md` | [Pre-Stage 2](stage2-docs-development.md) | [Development/docs build](../guides/development.md) |
| `docs/documentation-refactor.md` | [Pre-Stage 2](stage2-docs-documentation-refactor.md) | [Stage records](documentation-refactor.md) |
| `docs/old/README.md` | [Pre-Stage 2](stage2-docs-old-README.md) | This index |
| `docs/Invocation_API.md` | [Original notes](Invocation_API.md) | [Pointer](../api/agents.md), [agents](../api/agents.md), [results](../api/results.md), [workflows](../api/workflows.md) |
| `docs/Redirect.md` | [Original notes](Redirect.md) | [Pointer](../api/results.md), [control timing](../api/results.md#targeting-and-timing) |
| `docs/Design_execution_loop.md` | [Original notes](Design_execution_loop.md) | [Pointer](../architecture/engine.md), [execution contract](../api/agents.md#submission-and-ordering) |
| `docs/Design_orchestrator.md` | [Original notes](Design_orchestrator.md) | [Pointer](../architecture/engine.md), [runtime reference](../api/runtime.md) |
