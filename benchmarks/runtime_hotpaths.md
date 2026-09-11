# Persistent MCP state initialization

Run locally with:

```sh
.venv/bin/python -m benchmarks.runtime_hotpaths
```

The benchmark registers a production host MCP tool callback and invokes it
1,000 times. Its persistent-state factory decodes a JSON array of 10,000
integers. No model, container, network service, or remote machine participates.

Measured on macOS arm64, Python 3.12.13:

| Implementation | Tool calls | Factory calls | Elapsed |
| --- | ---: | ---: | ---: |
| Before, at `3c1fbe1` | 1,000 | 1,000 | 454.540 ms |
| Lazy initialization | 1,000 | 1 | 5.048 ms |

The prior `dict.setdefault(name, factory())` evaluated the factory even when
state already existed, then discarded the new object. Factories may allocate
large state or open resources, so avoiding those calls matters independently
of this synthetic workload's timing. The new initialization lock covers only
state creation and lookup. Tool bodies execute outside it; concurrent MCP
regression tests verify shared state and parallel tool execution.
The sandbox endpoint uses the same initialization boundary, preventing two
MCP worker threads from creating competing initial values for a shared key.

These are single local measurements, not an end-to-end throughput claim.
Factory-call count is the deterministic performance assertion; elapsed time
varies with host load and the caller's factory.
