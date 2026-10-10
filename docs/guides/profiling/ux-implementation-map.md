# Profiler UX implementation map

Audit: 2026-10-10, `eric/episode-traces` at `fd6033c`. Scope: functional UX
before visual design. Agency is a live debugger: **See. Understand. Control.**
The flagship remains per-agent trajectories, current activity above newest-first
history; the wall-clock timeline complements it. No application code changed.
No applicable `AGENTS.md` exists in the checkout or its ancestor directories.

## Implementation surfaces

Full paths below are repository-relative. Unqualified profiler filenames and
`static/` paths are under `agency/observability/agwebui/`.

| Surface | Source and reuse points |
|---|---|
| Routes / shell | `agency/observability/agwebui/server.py`: `/` serves `static/index.html` + `app.js`/`style.css`; `/profiler` serves `static/investigator.html` + `investigator.js`/`investigator.css`. FastAPI, native ES modules, no frontend build or framework store. |
| Profiler state | `static/investigator.js`: one `state`, `loadRun`, `setView`, shared agent/search/cursor/selection, inspector, export and session controls. Run-load generations reject stale callbacks. Keep this state owner. |
| Trajectory | `renderCallStack`, column controls, `reconcileActivities`, `trajectoryOverlay`; `static/investigator-model.js`: `groupCallStack`, `packActionTracks`. Defaults to two columns; all running groups precede the newest 50 completed groups per agent. Activity pagination separately retains older evidence. |
| Timeline | `static/execution-timeline.js`: `ExecutionTimeline`, shared actions/counters, overlapping calls in subrows, zoom/pan/fit, opt-in Lock to Present. Navigation survives view switches and resets on run changes. |
| Detailed investigation | `renderInspector`, `toolDetails`, `originalPayload`, `showRawEvidence`; resources/waits, context, tandem board and comparison renderers in `investigator.js`. Preserve full payload/copy/source-event access, prompt evidence, evaluator outcomes and raw trace downloads. Current navigation has six views including Timeline; older docs' standalone Task progress view is absent, though obligations remain in data/inspector code. |
| Normalization / fixtures | `investigator.py` reads Chrome traces and native observer JSONL; `trajectory.py` owns live/review/replay reduction. `investigator_demo.py`, `trajectory_scenarios.py` provide labeled synthetic fixtures, including 5,000 calls. |
| Runtime / controls | `agency/agent.py`, `agency/agskill.py`, `agency/orchestrator/orchestrator.py`; relay in `agency/observability/agwebui/__init__.py`; `agency/engine/engine.py`, `agency/engine/clients/harness_interaction_client.py`, `agency/harness/servers/harness_interaction_server.py`, `agency/harness/daemon.py`, `agency/harness/adapters/pty/execution.py`, `agency/harness/ptrace/_tracer_loop.py`. |

Rendered trace/model/tool strings use `esc`/`escape`, escaped previews and
`textContent`; retain this boundary when moving content or controls.

## Capability matrix

“Both” means backend and UI exist, with the limits stated here; it does not
establish successful execution on every harness.

| Capability | Availability / limits |
|---|---|
| Live trajectories, timeline, details, resources, context, comparison | Both; missing evidence stays unavailable. Synthetic demos are labeled. |
| Freeze display; replay play/pause/step/seek | Both; display freeze is a local snapshot, replay commands affect playback only. |
| Pause/resume one or all agents | Both; live profiler requires connected transport, active actor and fresh relay heartbeat. Applied-process acknowledgement is uncertain (below). |
| Stop / force-stop | Both; UI `stop`/`kill` map to `Agent.stop(force=False/True)`, SIGTERM/SIGKILL. Global scope dispatches to `Agent.all()`, not just visible columns. |
| Invocation cancellation | Backend-only `Agent.cancel(handle)`; exact request cancellation differs from agent-wide stop. |
| `queue_message` | Backend-only ordered retained-context submission; no browser endpoint or composer. |
| `redirect` | Backend-only handle-scoped API and daemon `/redirect`; registered interactive PTY adapters support delivery. Other adapters or inactive/paused targets fall back to queued context. No profiler bridge. |
| `run` | Backend-only skill submission, returning result `agdata`; no browser skill/input registry or submit endpoint. |
| Configuration changes | Backend and original dashboard UI via `update_config[_all]`; not in profiler. |
| Parent / model-tool correlation | Both for some saved/native evidence; incomplete live projection and external-harness coverage (below). |
| Input/delegation/handoff signals | Reducer/UI support explicit events; not uniformly emitted. Fixture support does not prove a harness emits them. |
| Verified workspace diff / universal artifact inventory | Unavailable. Native write/edit results report paths/status; recorded artifact dictionaries may include digests. Checkpoint storage/diagnostics are not a profiler change stream. |
| Prices / inferred process exit / model understanding | Unavailable; do not derive these from tokens, quiet transport or accepted input. |

## Message and control semantics

**`run`:** `Agent.run(skill, agdata, max_steps)` delegates to `agskill.run` and
the orchestrator. Publication synchronously inserts a new future context into
the agent's chain; the returned `agdata` is pending execution output, not a
success acknowledgement. Dependencies/capacity govern admission; one engine
per agent is active at a time, while calls inside it may overlap.

**`queue_message`:** validates a nonblank string, returns `None`, and publishes a
host-only `context_message` request at the current end of that same context
chain. Once its predecessor resolves, the scheduler appends a sequenced user
message with `source="queue_message"`. It creates no engine/sandbox/model work.
For submissions **A → message → B**, A does not receive it; B and later retained
contexts do. For **A → B → message → C**, C receives it even if B was still
queued when the message was sent. It does not inject into the active invocation
or retroactively modify already-submitted work. Stateless runs replay retained
context; persistent sessions use message cursors. Request completion means
context was committed, not consumed by a model. See `tests/test_ordered_messages.py`.

**`redirect(handle, message)`:** the handle's private `_execution_agent` and
`_execution_id` survive result resolution. Foreign/unowned handles are rejected
without waiting. The orchestrator tries only that owned, currently running
request's engine; daemon request-ID and adapter completion fences prevent a
late RPC reaching its successor. PTY delivery interrupts the native turn,
clears input, submits a marked prompt and waits for native prompt acknowledgement
before returning `delivered: true`. This is prompt acceptance, not completion
or model comprehension. Successful delivery continues the same Agency invocation;
it is not automatically a separate retained-message request. If the invocation
has not started, has ended, is paused, lacks a handler, or delivery returns
false/raises, `Agent.redirect` calls `queue_message` once at the then-current
chain tail. Thus **A → B submitted → failed redirect(A) → C** queues for C,
not A or B. The public method returns `None` in either path; it does not expose
the delivery/fallback distinction. See `tests/test_redirect.py`.

**Current browser control progression:**

1. The live profiler sends `execution_command` with a client ID, command and
   optional agent name; pending state snapshots target IDs/control timestamps.
2. `server._trajectory_execution_command` checks command/actor/relay availability,
   writes an atomic JSON file, and returns `{id, queued: true}` or an error.
   This acknowledges file enqueue only. The file drops the client ID.
3. Execution-side `_poll_commands` checks every 0.2s, dispatches the agent API,
   then deletes the file even on failure. Heartbeat expiry is 30s; files are
   sorted by random filenames, not submission order. No correlated applied/error
   reply returns from this stage.
4. Agent methods call daemon RPCs. Pause holds known tasks at their next ptrace
   stop; resume schedules continuations on the tracer thread. These are process
   controls, not a scheduler admission pause or guaranteed cancellation of
   host-side model/tool work. `_paused` also applies to a later daemon/attempt.
5. Logger events return through trajectory patches. The client confirms pause
   from newer `control_ts`/status and stop from newer `stop_ts`/force or terminal
   state; disconnect/30s timeout reports missing confirmation.

**Acknowledgement limits:** `pause`/`resume` log `agent_paused`/`agent_resumed`
even when there is no ready daemon or its RPC failed. RPC `{ok:true}` can be a
no-op or only queued process continuation. Therefore these events prove recorded
agent intent/state, not that every process is actually parked/running.
`stop` marks outstanding requests cancelled, clears pause and signals returned
active engines (excluding transactions that already claimed completion).
It does not permanently close submissions. Queued work settles when dependencies
allow dispatch; SIGTERM can be ignored. `agent_stop_requested` is not process
exit; the UI's “SIGTERM/SIGKILL delivered” wording overstates independent proof.
Scheduler terminal request events establish invocation outcome; workload `done`
is separate. Do not add stronger success labels without correlated evidence.

## Identity, ingestion and telemetry

| Identity / clock | Actual scope |
|---|---|
| Execution | Live ID is literal `live` for one configured directory; saved catalog ID is a 16-character hash of resolved trace path. Neither is a durable workload UUID. Live `epoch` is reader-instance recovery identity, not execution identity. |
| Agent / invocation | Agent is allocated `agname`; request is `runN` from the process-wide orchestrator sequence, including context-message requests. Namespace these by execution/source; they are not globally durable IDs. `agency.run_id` span metadata denotes this invocation. |
| Model / tool / source | Model `call_label` is a 12-character UUID fragment, action ID `actor:model:label`; tool admission `call_id` is UUID hex, action ID `actor:call_id`. Explicit `action_id` takes precedence. Source IDs are `source/events-or-exchanges/row-id`. Saved actions use span IDs or stable-for-file-order fallbacks. |
| Ordering / time | Logger row IDs contain a padded logger-instance sequence plus UUID; scheduler records `submission_sequence`. They are not a single global event sequence. Wall timestamps are epoch seconds; profiler spans use monotonic durations and wall origins; trace adapters convert microseconds to seconds. Reader batches sort by `(ts,id)`, with late events retained. Missing timestamps get an explicitly labeled fallback observation time. |

`agDataLogger` buffers WAL SQLite. `agwebui.run` flushes loggers every 0.5s and
records workload lifecycle. `LiveSource` reads global registrations, per-agent
`events`/`exchanges` and optional `profile_data.sqlite3`, at most 500 meaningful
rows/table per refresh, throttled to 0.25s. Streamed tokens/syscalls are not
trajectory cards. `/api/trajectory/live` and `/ws/trajectory` share the locked
reader and 512-patch revision journal. Epoch mismatch/expired cursor yields a
snapshot; client exponential reconnect is 0.5–10s, retains cursor, ignores old
socket frames, and closes old subscriptions on run changes. Missing/unreadable
DBs and catch-up are disclosed. Original dashboard `/ws` uses its own tail poll
and historical `/api/timeline` + `/api/events`, not this journal.

Replay reduces only the selected event prefix, starts paused, batches at most
100 events, and preserves the selected clock on reconnect. Saved spans get
labeled reconstructed boundaries; outputs/usage appear at their recorded end.
Live replay still ingests evidence without following its newest clock. Missing
call starts stay marked; missing ends at final boundaries become incomplete.
Overlapping calls retain separate IDs/tracks rather than one “current call”.

Telemetry comes from `agency/observability/profiler/{agprof.py,agprof_trace.py,presentation.py}`
and host interaction/LLM servers: workload cgroup and watched host/sandbox PIDs,
CPU/RSS/VMS/I/O, registered sandbox cgroup CPU/memory/I/O/network, NVML device
metrics and available process GPU attribution. Host coverage is workload scoped,
not universally the entire machine. GPU process power is an estimate. Sandbox
labels and span agent/sandbox IDs permit attribution only where recorded.
Operation spans can contain wall/thread CPU/runqueue/blocked time; remote CPU
may be absent and nested intervals must not be added as exclusive work.
Model exchanges contain prompt/response chains, usage and metadata; hidden
reasoning is excluded. The UI preserves named counter scope, sampled points,
usage coverage and nearest-sample time/distance; proximity is not operation or
agent attribution. No interpolation across missing samples or invented prices.

## Smallest delivery path and invariants

1. Keep the existing shell/state/reducer. Adjust information priority in
   `investigator.html`/`investigator.js`: trajectory columns and current activity
   first; retain timeline and secondary investigation behind existing navigation,
   details and inspector. No replacement framework or data model required.
2. Strengthen identity/target handling before adding intervention UI. Selection
   must include execution/source identity plus stable event/action ID, surviving
   merge, freeze and reconnect without index-based retargeting. Trajectory ↔
   timeline must keep the execution. `setView` currently silently loads B when
   a comparison-B selection leaves Compare; make that an explicit navigation.
3. Reuse separate display/replay/execution command paths. Display freeze retains
   incoming evidence while controls read current live state. Historical/replay
   contexts must never control live execution. Profiler replay rejects execution
   commands server-side; saved review is client-gated. Original dashboard's
   global Pause All/Resume All handlers lack historical-mode guards; selected-agent
   pause relies on its disabled button. Stop/column handlers guard live mode.
   Audit that seam if navigation/control integration touches it.
4. Preserve unknown and incomplete states, raw evidence, source provenance,
   old trace adapters and reconstructed replay. Add optional fields/fallbacks,
   not required schema migrations. Extend existing pytest/Node fixtures for each
   scoped change; never launch models to test UX.

## Concrete gaps / blockers

- New message/run UI needs an execution-side bridge to owned invocation handles
  and allowed skills/inputs, plus observable delivery versus fallback. None exists
  in the browser API. An agent name alone cannot implement `redirect(handle, …)`.
- Control enqueue has no execution UUID/epoch fence, request ID in the relay file,
  deduplication or correlated application acknowledgement. Reused directories,
  daemon errors and out-of-order command files constrain confident intervention.
- Live reducer ignores `parent_agname` and per-agent model/harness registration
  metadata; saved trace adapter reads `agency.parent_agent_id`. `_exchange` drops
  recorded request metadata, and model-start events lack request IDs. Deterministic
  audit reproduction: `request_started(run0)`, `request_submitted(run1)`, then
  `waiting_llm(m0)` associates m0 with run1 through `actor_request`. Do not treat
  this fallback as proven invocation correlation. External tool/model IDs remain
  absent on some harnesses; timing cannot repair them.
- Reader cursors assume append-only sources. Logger sequence restarts and DB
  replacement/truncation are not handled as new executions; use fresh directories.
  Clock skew remains possible; published episode membership is not reorganized.
- All running call groups render, but completed column groups cap at 50 without
  their own older-call control; activity pagination is separate. Projection memory,
  overlays and full on-demand raw payloads are not comprehensively bounded.
- Workspace-change proof is producer-specific. Native write/edit results and
  referenced paths are not complete diffs; the local-tool example explicitly
  emits an artifact SHA-256, which ordinary harnesses need not emit.

These are code-resolved constraints; no product decision blocks this audit.

## Baseline verification

Before this document, with existing untracked `live_profiler_demo.py`, `runs/`,
`swebench_live.py`, `swebench_three_agents.py` untouched and nothing staged:

- **272 passed:** `.venv/bin/pytest tests/agwebui tests/test_profiler_presentation.py tests/test_agprof.py tests/test_profiler_accuracy.py tests/test_agprof_harness.py tests/test_redirect.py tests/test_ordered_messages.py tests/test_orchestrator_lifecycle.py tests/test_orchestrator_lifecycle_edges.py tests/test_submission_handles.py -q`.
- **50 passed:** Node 24 `--test --test-isolation=none` on
  `tests/agwebui/{investigator-model.test.mjs,trajectory-stream.test.mjs,test_live_follow.cjs}`.
- **Passed:** Ruff check `agency tests`; Ruff format check `agency/observability/agwebui tests/agwebui` (13 files); Node syntax check `investigator.js`, `execution-timeline.js`, `app.js`; `git diff --check`.
- **Passed:** strict MkDocs build to a scratch site directory.
- **Pre-existing failure:** `scripts/check_docs.py` exits 1:
  `Unreviewed public methods: agency.agent.agent: {'stop'}`. API inventory is
  unchanged by this audit. Full pytest/auto-fixing pre-commit were not run.
- **Browser:** deterministic edge-case review/event inspector, trajectory →
  timeline retained execution/selection, paused replay and stepping; demo resource,
  context, tandem and comparison views; empty live source showed unknown state,
  disabled execution controls and independent display freeze; original dashboard
  opened. No console errors/warnings observed. Active real-process control and
  real GPU telemetry were not exercised; mocks cover the tested paths.
- Initial sandbox-only tests/server were interrupted because local sockets were
  blocked. Tests were rerun with socket access. The normal server CLI's external
  bind was rejected by automatic approval review; the browser used a temporary
  loopback-only launcher. No models, provider refinement, deployments or remote
  Git operations ran. This documentation-only audit introduces no runtime changes.
