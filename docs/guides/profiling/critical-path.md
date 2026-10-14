# Historical Critical Path Profiler

Open `/critical-path` on an Agency Web UI server and select a saved execution.
This view analyzes finalized historical trace files. It does not subscribe to
the live trajectory stream. Existing logs and Perfetto-compatible exports remain
the source of the evidence.

Agency already collected model attempts and canonical exchanges, permission-hook
tool timing, orchestration envelopes, ptrace supervision, process/cgroup
measurements, OpenTelemetry, and SQLite/Perfetto history. The earlier causal
projection scoped tool IDs and prompt fingerprints correctly, but its historical
normalizer omitted orchestration/process spans, its global view could not follow
cross-agent producer IDs, and saved process measurements often lacked tool
ownership. The dedicated profiler reuses those collectors and fixes the missing
identity boundaries rather than replacing their measurement pipeline.

The overview separates recorded wall time, the longest known dependency chain,
measured contributions to that chain, unexplained dependency intervals, aggregate
observed activity across agents, and wall time outside measured operations.
Temporal coverage does not establish complete causality: even 100% measured
coverage remains qualified if the recording cannot prove the execution boundary
and all dependencies.

## Using the view

The initial timeline shows the path and its recorded cross-agent dependency
context. LLM, tool, waiting, and system observations have separate colors; path
contributions have a highlighted outline. Agent identities remain visible in
named swimlanes. Unrelated agents retain their lanes and can be opened through
**Show local activity**. **All operation activity** reveals parallel operations.

Focus an agent, search operation identities, zoom with the buttons or Ctrl/Cmd
and the wheel, and drag blank space to pan. Select a span or dependency line to
inspect its timing, evidence classification, identity, and available metrics.
Expand a tool's attributed process overlays to inspect process generations,
ancestry, CPU, memory, and I/O evidence. Process overlays never increase the
tool's wall duration. Recorded agent-owned processes without a known owning tool
remain independently inspectable.

Detailed evidence links open the existing historical trajectory and Systems,
LLM, or Tools profiler. Download the original trace and open it in external
Perfetto for deeper investigation. Agency does not upload the trace to Perfetto.

Missing fields remain unavailable. Read **Coverage and limitations** before
treating a partial or candidate path as an explanation of the entire workload.
Results, commands, and other displayed text are bounded previews; source
transcripts remain in the existing detailed investigator.

## Runtime to historical analysis

1. The scheduler and engine emit agent/request envelopes and phase spans.
   LLM handlers capture request attempts, streaming first-token timing, usage,
   outcomes, tool IDs, and canonical exchanges. Permission/tool handlers capture
   invocation admission and completion identities.
2. OpenTelemetry spans, agprof monotonic timing, SQLite event/exchange storage,
   native reports, ptrace process lifetimes, and the cgroup/procfs sampler retain
   execution evidence. Finalization exports `agprof.trace.json` and
   `summary.json` beside `profile_data.sqlite3` and agent logs.
3. The compact loader streams the Chrome event array, retaining semantic
   operations and bounded counter extrema. It counts and discards raw syscalls
   and automatic Python stacks. It replaces transcripts with canonical block
   fingerprints, recovering complete identity evidence from adjacent SQLite
   exchanges when possible.
4. Analysis correlates operation identities, builds the boundary DAG, computes
   a longest defensible chain, and exposes compact summaries to the dedicated UI.

All trace span/sample positions use the recorded `perf_counter_ns` axis. The
loader preserves the original first-span origin and follows explicit parent
span IDs through dropped harness wrappers to recover request ownership.
SQLite wall-clock event timestamps never create monotonic trace dependencies.
Cross-agent blocking links require a recorded shared clock.

## Correlation rules

| Relationship | Accepted evidence | Meaning |
| --- | --- | --- |
| Model → tool | Tool-use ID scoped to the emitting exchange and agent; explicit model ID when present | Issuance dependency |
| Tool → next model | Canonical prompt tool-result identity scoped to its issuer | Result consumption dependency |
| Model → next model | Canonical response fingerprint/lineage present in the consuming prompt | Continuation; older history is provenance |
| Producer request → consumer request | Persisted Future/result-handle or context-chain producer request ID | Producer completion gates consumer admission or dependency wait completion |
| Workers → result barrier | `wait_all` records every producer request ID | Fan-in; workers remain parallel |
| Span parent → child | Exact parent span identity | Containment/provenance; no automatic blocking claim |
| Tool → process | Exact invocation ID, explicit owning tool ID, or generation-qualified process identity | Resource attribution; no added tool duration |
| Agent → sampled process | Explicit sandbox ownership | Agent attribution only; does not establish a tool owner |
| Process → child process | Recorded parent process generation identity | Ancestry/provenance |

Runtime result and context handles retain private execution identity after
their Future resolves. Those fields do not enter serialized task data. Request
dependencies are recorded on request/phase spans and in `request_dependencies`
events, including dependencies already completed when submitted. Result wait
spans record all awaited producers and the consumer execution, when available.
Synchronous submission spans additionally record launcher serialization within
the same caller execution, thread, and profiling session. A later submission
depends on the preceding submission returning; workers remain concurrent.
This captures cold launch/setup overhead without linking the preceding worker's
completion to the next worker. Serial native tools carry exact predecessor
invocation IDs within a unique response batch; external tools receive no
unrecorded serial ordering.

Native Bash/grep/glob subprocess roots report process lifetimes through the
existing profiling bridge. Ptrace spans capture PID namespace, namespace PID,
start ticks, and parent information. The sampler stores PID/start-tick process
generations and generation-qualified ancestry. A host PID alone never maps a
container process or a reused PID to a tool. Short-lived processes may have a
recorded lifetime but no resource sample.

## Critical path calculation

The graph uses start/end boundary vertices. Measured operation intervals are
weighted edges; explicit/reconstructed blocking dependencies connect their
boundaries. A topological longest-path calculation validates endpoints, finite
timestamps, temporal direction, and weights. Cyclic blocking evidence makes the
path unavailable. Malformed links are reported and excluded, never reversed or
repaired silently.

Request submission and engine execution envelopes act as boundary gates rather
than billable work. A process under a tool supplies inspection evidence rather
than another additive duration. Nested intervals remain competing paths, and
aggregate activity uses interval unions per agent. Independent agents never get
edges from their overlap and their durations never sum into global wall time.

At a recorded fan-in, only the tail after the latest recorded prerequisite is
charged as unexplained elapsed dependency time. Earlier branches are not charged
for another worker's execution. Equal elapsed chains prefer measured model/tool
evidence over a containing wait. Gaps establish timing, not their resource cause.

The latest observed prerequisite of a join is also reported separately as a
**join gate**. If a late-starting worker has no recorded startup dependency, a
different earlier worker can have a longer known chain. The profiler discloses
that discrepancy and identifies the latest recorded completion without
manufacturing the late worker's missing earlier history. A join gate is timing
evidence about the recorded prerequisite set; it is not proof that all required
workers or the entire workload boundary were captured.

The core topological analysis is O(V + E). Identity correlation and presentation
sorting add indexing/sort costs. Per-agent paths use partitioned nodes and edges.
Resource queries use generation-qualified series and binary searches of bounded
recorded sample positions. Whole-process CPU/I/O totals are never allocated to a
tool by interval overlap. Reduced sample extrema are observations, not an exact
integral or a guaranteed peak for every subinterval.
Sandbox pressure and I/O windows match an explicit sandbox identity and retain
their shared scope. They are context for an operation or agent, rather than
resources exclusively owned by a tool.

## API and scale limits

| Endpoint | Result |
| --- | --- |
| `GET /api/critical-path/runs` | Allowlisted historical run catalog; no local trace paths or synthetic demo entries |
| `GET /api/critical-path/{id}` | Compact summary, operations, dependency edges, agents, processes, coverage, and evidence links |
| `GET /api/critical-path/{id}?limit=20000` | Additional compact span summaries; initial limit is 5,000 |
| `GET /api/critical-path/{id}/operations/{operation_id}` | Recorded operation and its relationships/process context |

The loader caps semantic spans at 50,000 and explicit ancestry entries at
100,000, disclosing omissions. Counters retain bounded extrema rather than every
sample, with at most 4,096 series and 250,000 retained sample points. Native
observer JSONL histories are streamed with the same semantic span limit.
Trace and canonical block-reference retention each have a 250,000-reference
budget; incomplete prompt histories cannot create active consumption edges.
Projected unmeasured tool observations share the 50,000-operation graph budget.
A single JSON event over 32 MiB is rejected explicitly. Oversized optional
summaries and canonical blocks remain unavailable with a coverage warning.
Truncated or
malformed trace documents return an analysis error.

Two compact results are cached. Trace, summary, canonical database, and WAL
revisions invalidate that cache. A lock prevents simultaneous requests from
retaining duplicate analyses. The API bounds response operations, edges, agents,
and processes and reports omitted counts. Critical/blocking relationships take
priority. The browser renders at most 240 operation marks and 400 dependency
edges per page, culls offscreen marks, and expands processes on demand.

## Validation and remaining limits

Deterministic tests cover sequential LLM/tool chains, process attribution,
namespace/PID reuse, nested timing, independent agents, fan-out/fan-in, request
admission gates, resource window measurements, missing timing, temporal
contradictions, cycles, loader boundaries, payload limits, and frontend filtering.
Recorded agprof/orchestrator tests save and reload actual concurrent executions
and native process observations. Real SWE-bench Agency traces are also analyzed.

Historical runs cannot acquire dependencies or process ownership that were never
recorded. Agent parentage alone does not establish delegation completion, and
unrecorded agent messages do not establish data dependencies. Retry labels and
errors remain evidence; identical prompts alone do not prove a retry dependency.
No process contention, scheduler causality, syscall bottleneck cause, or
hypothetical speedup is inferred. Sampling may miss short processes, and a
process's sampled bounds are not its exact birth/exit unless lifecycle spans
were captured. Resource pressure/network details are available only where
existing instrumentation recorded them.

An exported trace is a completed recording, not proof that its workload
succeeded. Interrupted spans and absent ends remain incomplete. Where no exact
workload start/end is available, recorded wall time is the trace/session extent.
Large traces still require a full streaming scan on the first request, and very
large prompt histories can make identity correlation more expensive than the
core DAG algorithm. No persisted analysis index is introduced in this version.

## Implementation and validation record

| Changed area | Files |
| --- | --- |
| Historical graph and analysis | `agwebui/critical_path.py`, `agwebui/causal_graph.py` |
| Compact historical storage/API | `agwebui/critical_path_store.py`, `agwebui/critical_path_api.py`, `agwebui/server.py`, `agwebui/investigator.py` |
| Dedicated view | `static/critical-path.html`, `critical-path.css`, `critical-path.js`, `critical-path-model.js`, `critical-path-ui.js`; navigation in `static/index.html` and `static/investigator.js` |
| Requests and synchronization | `orchestrator/orchestrator.py`, `orchestrator/scheduler.py`, `agdata.py`, `agcontext.py` |
| Model/tool reporting | `engine/host_servers/llm_handler_server.py`, `host_interaction_server.py`, `native_harness/react_loop.py` |
| Process/resource reporting | `native_harness/profiling.py`, `native_harness/tools.py`, `harness/ptrace/supervisor.py`, `profiler/agprof.py`, `profiler/presentation.py` |
| Tests | `tests/agwebui/test_critical_path.py`, `test_critical_path_store.py`, `critical-path.test.mjs`, `tests/test_critical_path_instrumentation.py` |
| Documentation | This guide, `guides/profiling.md`, `architecture/observability.md` |

Paths in the table are relative to `agency/observability/` for `agwebui` and
`profiler`, to `agency/` for runtime areas, and to the repository root for tests.

Validation on 2026-10-14 included 362 passing web/backend/recorded-runtime tests
with one environment skip, 255 passing broader runtime/profiler regressions,
and all 17 frontend test files, including 13 dedicated UI cases. The final
dedicated backend/loader/instrumentation suite passed all 78 tests, including
explicit non-finite JSON rejection and optional-source size guards. Ruff and scoped
`git diff --check` passed.

Actual browser QA used the recorded three-agent SWE-bench executions under
`runs/tutorials/20261010T212937...` and
`runs/tutorials/20261014T110057.338608Z_swebench_three_psf__requests-6028/`.
The latter 63.5 MB trace contained 135,160 events and reduced to 139 retained
spans / 142 projected operations. A scan with Python allocation tracing took
3.37 seconds and peaked at 22.9 MB of newly tracked Python allocations after
imports; this is not a total-process RSS measurement. Its 72.69-second candidate
chain in a 100.41-second recording had 51.02 seconds of measured contributions
and disclosed missing historical cross-agent and tool/process ownership.
The larger browser recording exposed 827 operations and 685 processes.

Fresh saved/reloaded Agency sessions validate new cross-agent and launcher
correlations, a forced cold submission delay, native serial tool batches,
generation-qualified real CPU subprocess samples, absent samples, and
namespace/PID reuse. These tests use recorded timings and actual orchestrator
and subprocess execution with deterministic model fixtures; the browser QA
uses historical model workloads. Scan artifacts are retained under
`artifacts/critical-path/implementation-validation/`.
