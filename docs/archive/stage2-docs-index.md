# Agency documentation

Agency runs, controls, and inspects coding agents from Python.

## Start here

- [Getting started](../getting-started.md): installation, credentials, first run, and setup failures.
- [Examples](../../examples/README.md): choose a script for the task you want to try.
- [Profiler startup](../guides/profiling.md): run with the Web UI or reopen a saved run.
- [Contributing](../../CONTRIBUTING.md): development setup and checks.

## Go deeper

- [Configuration](../guides/configuration.md): model providers, external CLI harnesses, and project mounts.
- [Observability](../guides/profiling.md): logs and explicit profiling sessions.
- [Existing profiler reference](../guides/profiling/views.md) and [Perfetto build reference](../guides/profiling.md#raw-trace-viewer).
- [Optional host setup](../guides/host-setup.md) and [checkpoint configuration](../guides/checkpoints.md).

The existing [invocation API](Invocation_API.md),
[execution loop](Design_execution_loop.md),
[orchestrator](Design_orchestrator.md), and [PTY architecture](PTY.md)
notes remain available while later stages replace them. They are earlier
implementation references; check the source when behavior differs.
The [archive index](README.md) maps previous paths to their current locations.
