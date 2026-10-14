# Live metrics and profiler views

The execution profiler (`/profiler`) has one Profiler navigation tab containing Activity, LLM, Tools and Systems
category views. Trajectory and other execution views have a six-slot summary
and a collapsible category panel; the detailed Profiler tab hides this repeated
summary. Category, expansion
and pins are saved in local storage (`agency.metrics.v1`). Diamond controls
unpin/add metrics; when all six slots are occupied, a slot picker opens with six
replacement buttons. Cancel or Escape leaves the pins unchanged. Live updates
keep metric and agent control buttons mounted, and pause/resume closes transient
control menus. Status and receipt updates retain a stable workspace height.
Reset restores the six defaults. Metrics/category controls remain mounted
during live updates, so a telemetry patch cannot interrupt a click. Expanding or
collapsing the panel updates only that panel; it does not rebuild the trajectory.
Compact Systems cards read the latest gauges/counter deltas instead of
reprocessing full resource history. Tables show 100 calls per page and support
text filtering and column sorting. Operation links reveal existing trajectory
cards and the captured evidence inspector. Error cards filter recorded failures;
LLM P95 sorts requests by descending duration.

## Data and measurement semantics

No new polling or collector is started. The existing trajectory stream supplies
updates, lifecycle time, call identities and scoped counters. The existing
120ms update batching is retained. Backend counters replace the contribution of
a changed call rather than incrementing on every event. These totals survive
client-side retention and replay/reconnect snapshots. Old recordings calculate
totals from their available calls. Replay and frozen views use their projected
execution duration, not the browser's current wall clock. The summary wall clock
starts at the first observed request execution, skill execution, or agent call,
includes pauses, and ends when the workload callback returns (see below). Program
startup, sandbox preparation, and the pre-launch Enter prompt are excluded. Timeline
coordinates retain the recording origin so setup and resource samples stay aligned.
Latency cards, tables, and chart ticks show milliseconds and seconds together.
Detailed graphs label both axes and explain the diagnostic meaning below each plot.

Summary totals cover the execution; charts and latency statistics cover retained
calls when the live client bounds history. Full recordings remain accessible via
existing review/export. Recent rates use a trailing 60-second window (shorter
at startup), count starts, and use execution-relative seconds. Percentiles use
nearest rank over completed calls with known boundaries and show sample counts.
Sparse P95 estimates are labeled. Aggregate durations can overlap; wall coverage
is the union of measured intervals. Unclassified time is not called idle time.

Token values retain provider reporting semantics. A partial sum includes only
reported usage and shows reporting coverage; absent usage is unavailable.
Client output throughput divides reported output tokens by corresponding E2E
intervals; it is not server decode throughput. Current normalized actions do
not expose reliable first-token/chunk timestamps, cache accounting, retries,
prices, prefill or decode timing, so these are explicitly unavailable.

Tool errors follow existing projection semantics, including structured nonzero
exit codes and isError. They are not described as infrastructure failures.
Cancellations remain separate. Timeout-specific classification is not uniformly
recorded; a reliable timeout count needs an explicit normalized outcome.

Systems reuses systemMeasurements and recorded history. Each series keeps its
host/workload/sandbox/process/device scope. CPU uses 100% per core. Memory and
I/O collectors divide bytes by 2**20, displayed here as MiB and MiB/s. RSS is
labeled separately from scoped RAM, with the process identity in each card.
The compact Systems panel selects CPU, RAM, one named process RSS, disk rates,
and available GPU measurements rather than the first six arbitrary counter series. Counter resets and missing baselines do not
produce rates. GPU charts appear only for recorded supported series. Per-core
CPU, pressure, scheduling delay, context switches, page faults, I/O latency and
network errors need additional projected instrumentation. No zero-filled charts
are substituted. Restart counts and complete agent lifecycle concurrency are
not available; call concurrency is measured from observed intervals.

Remaining scope: activity gaps cover observed call boundaries, not every possible
agent event. The profiler does not infer dependencies. Detailed system series
cover the displayed execution range; selecting an individual metric does not
yet zoom to a narrower sample range. The global agent/search controls belong to
trajectory navigation; profiler tables have their own explicit filter. Statistical
views reproject the bounded retained calls during the existing update batches;
full-recording analysis is not incrementally indexed.

## Agent scope

The summary dashboard has one **Metrics scope** multi-select. It filters the
existing cards; there is no per-agent dashboard.

- **All agents** (default) keeps execution-wide call totals and includes agents
  as they spawn. Ticking individual agents selects any subset; an explicit
  subset is kept while agents spawn or terminate. Completing the set returns
  to all agents. **All agents** itself switches between every agent and none.
- Selecting no agents is valid: the cards are replaced by a "No agents
  selected" state with a **Select all agents** action, nothing is computed,
  and the selector stays available to choose agents again.
- Selecting a column header scopes the dashboard to that agent; selecting it
  again returns to all agents. Columns, pins and live following are unchanged.

| Metric | Under an agent scope |
| --- | --- |
| Tool calls, LLM requests, tokens, tool errors | Summed over the selected agents from `metrics_by_agent`, the per-actor form of the execution counters |
| Latency percentiles, rates, streaks, active calls | Recomputed from the selected agents' own calls (pooled observations, never an average of per-agent values) |
| `CPU · agent sandboxes`, `RAM · agent sandboxes` | Sum of the latest cgroup sample of each selected agent's own sandbox container |
| Wall clock | Execution-wide; labelled `entire execution` and drawn with a dashed border |
| Host, Workload, process and GPU series | Keep their recorded scope; never filtered or apportioned |

"All agents" is not "entire system": agent sandbox cards cover the agents'
containers only, while Workload and Host series also include the orchestrator,
host services and runtime daemons.

Each column status line shows `CPU n% · tokens`. Tokens are reported input
plus output tokens summed over the agent's model calls. CPU is the cgroup CPU
of every process in the agent's sandbox container (100% = one core), from the
existing `cg:<sandbox>:cpu_pct` samples. The engine records the sandbox label
on the agent's `running_harness` state event; without that record, with a
sandbox shared by an unselected agent, or with no sample in the last 10 s, the
value is shown as unavailable rather than estimated.

## Wall-clock measurement boundary

Agent execution, workload completion and the wall-clock measurement are three
separate things. The measurement starts at the first observed agent execution
and, left alone, ends automatically when the callback given to `agwebui.run`
returns. `agwebui.run` records that instant as a `workload_completed` event
before the profiler context exits, so trace export, summary generation and
telemetry flushing are not counted; the later `done` event stays the terminal
lifecycle event and carries `finalization_seconds` so the boundary can be
recovered if `workload_completed` was lost. An exception or cancellation still
records the boundary, with the callback's outcome as its status. Neither event
says anything about task success. Agents going idle never ends the measurement:
gaps between sequential skill requests are counted. Recordings that only have
`done` (older runs, launchers that record it themselves) end there.

The Wall clock card shows the duration, whether the measurement is running or
ended, and whether its boundary is automatic or manual. Its `⋯` menu, available
on the connected live execution only, changes the measurement and nothing else:
no agent is paused, stopped or signalled, and it keeps working after the
workload and its command relay have finished.

- **End now** ends the measurement at the server's current time, even while
  agents are still executing.
- **Keep running** overrides automatic completion. Time is continuous from the
  original start: choosing it after the workload ended includes the interval
  since then. It lasts until End now or Reset to automatic.
- **End at … s** selects any boundary between the start and the present, given
  as seconds since the measurement started, for example a later end than the
  automatic one without rerunning anything.
- **Reset to automatic** drops the override and restores the automatic boundary,
  which is always kept.

Choices are appended to `measurement.jsonl` in the run's log directory, separate
from the immutable execution events; timestamps are assigned and validated by
the server, and the resulting state reaches the page through the ordinary
trajectory patches (`measurement`, `execution_wall_time`). Reloading the page,
reconnecting, reopening the run with the standalone viewer, live replay, and the
saved review of the same run's profiler trace all use that recorded boundary. A
measurement kept running lasts only while a dashboard server is running: when
it is reopened after that server stopped, it ends at the server's last renewal
(`measurement.lease`) and is labeled accordingly. If the launching process died
without recording completion, the live clock stops at its last relay heartbeat
instead of counting to the present.

The execution timeline marks the selected end on the wall-clock axis (manual
and automatic are styled differently, and a superseded automatic end stays
visible). Calls after the boundary remain on the timeline; the trace is never
altered. A boundary later than the last recorded event is pinned to the trace
edge and says so.

Limits: saved profiler traces use the profiler's own clock, so their timeline
shows no measurement marker and a replay of a saved trace ends at the trace
extent, labeled as not a recorded boundary. Only the current run's saved trace
is matched with its recorded measurement; other saved traces show their trace
duration as before. Rates, percentiles and the trace duration are unchanged by
the measurement boundary.

## Existing three-agent SWE-bench launcher

From `/home/eric/Documents/Codex/agency`, with provider credentials already
exported:

```bash
AGENCY_PROFILE=1 AGENCY_WEBUI_PORT=7862 .venv/bin/python swebench_three_agents.py
```

This is the existing local launcher, not a guessed CLI. It creates investigator,
fixer and validator Codex agents with max_concurrent_engines=3. It prepares each
sandbox, then waits for Enter before launching their first tasks in parallel.
Later refinement and independent patch validation are sequential stages.
The script starts its own profiler server and lingers after execution:
`http://localhost:7862/profiler?run=live`. No separate UI command is needed.

Prerequisites:

- Linux x86-64, the repository's Python 3.12+ virtual environment and dependencies.
- Working Podman (preferred when installed) or Docker.
- Existing `runs/swebench-three/task.json` and `input/`, including its Git checkout
  matching task.json's base_commit.
- Image `localhost/agency-swebench-three:latest` in that engine. The existing
  recipes are `runs/swebench/image/Dockerfile` and
  `runs/swebench-three/image/Dockerfile`; the latter inherits
  `localhost/agency-swebench:latest`. Build them with the selected engine if absent.
- Pinned executable at
  `.venv/codex-0.147.0/package/vendor/x86_64-unknown-linux-musl/bin/codex`.
- With default provider settings, exported `OPENAI_API_KEY`; alternatively the
  existing provider-specific variables accepted by `examples/_common.py`.
  `AGENCY_LLM_PROVIDER` and `AGENCY_LLM_MODEL` override repository defaults.
  The launcher does not load `.env` itself.
- `AGENCY_PROFILE=1` is needed for the environment-enabled system collector;
  `examples/_common.py` otherwise defaults it off. Existing profiler scope and
  permission requirements still apply. Do not silently substitute host data.

For an already recorded execution, the supported separate viewer is:

```bash
.venv/bin/python -m agency.observability.agwebui.server \
  --run-dir /absolute/path/to/execution/logs --host 127.0.0.1 --port 7862
```

Activity shows concurrent outstanding calls, recent rates and recorded states.
LLM shows completed request latency and reported usage. Tools shows calls,
failures and inspectable output. Systems shows the scoped samples actually
collected; unsupported measurements remain explained. The launcher was inspected
but was not run as part of frontend verification.
