# Profiler measurement contract

The profiler emits schema version 6 in `summary.json`, alongside `summary.md`,
`agprof.trace.json`, and its own `profile_data.sqlite3` span store. Explicit
spans use runtime timestamps, not transcript gaps. `coverage.engines` describes
the harnesses registered during the session. An unavailable metric is `null`
(`n/a` in reports), not a measured zero.

| Source | Captured information | Limits |
|---|---|---|
| Host spans | Run/engine phases, LLM attempts, wall and thread CPU clocks, explicit parents | Thread counters include all work on that thread; nested CPU totals overlap. Missing scheduler counters leave blocked time unknown. |
| Tool admission/completion | Tool and syscall intervals, completion errors, unfinished calls at attempt retirement | Hook boundaries include transport and harness overhead. Opaque internal tools without callbacks remain unobservable. |
| Native semantic events | Nested turns, compaction, retry backoffs, tool execution timestamps | Container-asserted provenance; clock mapping has round-trip uncertainty. Turn completion alone does not imply task success. |
| Automatic Python | Host and managed native Python calls, optionally dependencies | Python 3.12+, configured roots, depth/duration/event limits; no automatic capture inside arbitrary child processes or non-Python runtimes. |
| Process/resource sampling | Cgroup, process, memory, IO and GPU observations; ptrace process events where enabled | Gauges and PID membership are sampled. Brief peaks may be missed. Process GPU power is apportioned, not directly measured. |

`tool_metrics.latency` and per-tool percentiles include only `timing="exact"`
intervals. `latency_by_timing` keeps hook-boundary and legacy derived intervals
separate. Native exact intervals carry `provenance="container_asserted"`; host
observations carry their own provenance. Summaries containing remote assertions
or derived/boundary intervals are marked `data_source="mixed"`.

Unknown tool outcomes never count as successes. A tool result is finalized on
its completion callback, including a final tool with no later model request.
Unmatched admissions are interrupted, not silently removed. Native parent IDs
are resolved within the authenticated attempt; unknown parents and stale
profile sessions are rejected.

Provider attempt counts are distinct from retry counts. `retries` is unavailable
when attempts cannot be correlated; repeated `llm:attempt[0]` labels do not prove
there were no retries. `retries_reported` counts observed native backoff events,
including an interrupted backoff. Missing token usage leaves the complete total
unknown; `reported_input_tokens` / `reported_output_tokens` retain known partial
sums. Throughput uses attempts with both reported output tokens and generation
time and declares that population in `throughput_measured_attempts`.

`sampling.telemetry_errors` exposes recording, ingest and transport failures.
Native automatic capture uses bounded batches and a bounded teardown flush;
there is no promise of lossless telemetry during crashes or transport outages.
A killed native process may never deliver its buffered automatic calls. Open
semantic spans that reached the host are retained as interrupted.

## Verification

`tests/test_profiler_native.py` covers real `sys.monitoring` capture in a separate
Python interpreter without host imports, nested native spans, measured tool
timing, stale-session/parent rejection, all five HTTP wire adapters through the
real LLM handler/profiler with a deterministic fake provider, and the checked-in
`tests/fixtures/profiler_contract.json` tool summary contract. It does not launch
the actual external CLI binaries. Existing Linux-only ptrace/container/GPU tests
remain necessary before claiming coverage for a particular deployment.
