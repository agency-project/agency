# Live trajectory and deterministic replay

Open `/profiler?run=live&view=trajectory` on the existing Agency web server.
The Trajectory tab shows activity starts, running calls, returned evidence,
attention signals, and a final execution state. Layout switches between the
activity feed and agent trajectories on a shared clock. Other profiler tabs
and the native dashboard at `/` remain available. The embedded Perfetto viewer
has been removed; raw trace downloads remain available.

## Run it locally

This cost-free smoke executes real local reads, subprocess tests, and a file
edit through Agency's canonical `HostInteractionServer` and `agDataLogger`.
It does **not** invoke a model or autonomous Agent. Use a fresh directory for
each run; the example rejects existing databases instead of overwriting them.

Terminal one, from the repository root:

```sh
.venv/bin/python -m agency.observability.agwebui.server \
  --run-dir /tmp/agency-trajectory-example-1/logs --port 7860
```

Open <http://localhost:7860/profiler?run=live&view=trajectory>, then terminal two:

```sh
.venv/bin/python examples/11_live_trajectory.py \
  --run-dir /tmp/agency-trajectory-example-1/logs --delay 8
```

The viewer starts empty, shows a running test before its result, accumulates
three matching failures, records an edit with a file digest, and shows the
same test passing. The run reports completed; the viewer does not independently
declare the broader task solved. Full subprocess output remains accessible
through the source-event buttons in the inspector.

For a normal Agent execution, use the existing tutorial on a configured host:

```sh
AGENCY_WEBUI_LINGER=1 AGENCY_WEBUI_PORT=7861 \
  .venv/bin/python examples/10_harnesses_and_webui.py
```

That example needs the repository's normal model credentials and sandbox
setup. Open <http://localhost:7861/profiler?run=live&view=trajectory>. Other
applications can keep using `agwebui.run(workload, run_dir=..., port=...)`;
the wrapper records workload start and explicit completion/failure/cancellation.
Without the wrapper, start a separate server against the application's log
directory. If no workload-final event exists, execution status remains unknown
after the last recorded request; a quiet connection does not establish a stop.

Server startup requires no viewer download or frontend build.

## Pause and inspect live execution

The session bar places blue **Pause Display** / **Resume Display** beside amber
**Pause Agents** / **Resume Agents**. Pause Display holds the visible cards,
details, timelines, and metrics at a snapshot while agents keep executing and
incoming updates are retained. You can scroll, select boxes, and expand their
details while the display is held. A status label counts calls updated in the
background. Resume Display catches up to the latest evidence and keeps the
selected box and scroll position. Display pause sends no execution or replay
playback command.

With an active `agwebui.run(...)` workload, **Pause Agents** / **Resume Agents**
controls execution of all active agents. Selecting an agent, episode, or action also
exposes **Pause agent** / **Resume agent** for its owning agent. The agent
inspector has the same control. If any active agent is paused, the global
button offers **Resume Agents**. Execution controls continue to reflect the latest
agent state while the display is held.

These commands use the existing webui relay and `agent.pause()` / `resume()`
to stop and continue the harness process. The trajectory connection stays
open, so actions, context, resources, and source evidence remain inspectable.
Recorded pause/resume events supply the authoritative agent status and paused
badges. A queued command stays pending until those events arrive; delivery
errors or missing confirmation appear in the session bar.

Running spans freeze while their agent is paused. Duration labels exclude
recorded execution pauses; the inspector also shows the wall interval, and
trace start/end timestamps and the shared wall-clock axis stay intact. The
run's wall time continues advancing. Controls disable after completion,
during disconnection, or when the execution command relay is unavailable.
The standalone log-viewing server and the cost-free smoke example above have
no live agent relay, so their execution controls stay disabled.

## Replay and investigation

Select a live or saved execution, then **Replay recorded events**. Replay starts
paused; use **Next event**, **Play replay**, speed, or **Restart replay**.
`mode=replay` is preserved in the URL. Actual saved spans supply start/end
boundaries, and results, usage, context, and final metadata arrive only at the
recorded end. Reconstructed boundaries are identified as such in raw detail.
Replay exhaustion is separate from a recorded final execution status.
Replay pause controls playback only; saved/replay runs never expose live
execution controls. Scrubbing the time cursor rebuilds the canonical event prefix,
including paused/resumed agent state and resource samples. Live replay continues
reading new evidence while its selected clock stays fixed. Reconnect preserves
the selected time, including positions between sparse events.

The **Timeline** tab follows Trajectory. Agent rows and sampled resource tracks
share one wall-clock origin; overlapping calls occupy separate subrows. Zoom,
fit, horizontal pan, tooltips, and click-to-inspect work on the same action IDs.
**Lock to Present** defaults off. Enabling it follows the newest activity;
manually scrolling backward releases it. Tab switches preserve zoom and pan.
Metric dots represent observations without interpolation across missing samples.

**Show details** on a call reveals formatted input/output, execution metadata,
and system samples nearest the call start, with sample time and distance shown.
Long content stays bounded until **Show full output** loads immutable source
evidence. Copy controls retain the original data. Columns have no fixed limit;
each has a minimum width and the board scrolls horizontally.

The existing summary includes cumulative input/output/total tokens, model-call
counts, run-average output rate, active agents, and available CPU/RAM/GPU/VRAM
samples. Missing telemetry is shown as unavailable; cost is not estimated.
The profiler persists its existing samples through the buffered data logger in
`profile_data.sqlite3`; the shared live reader ingests them without an extra
sampler or browser polling endpoint. CPU percentages are derived from measured
CPU-time deltas and can exceed 100% across cores.

The catalog includes separate synthetic fixtures:

- **Trajectory edge cases:** matching failures, explicit input request and
  response, a legitimate long check, delegation, handoff, an artifact, and
  cancellation with a missing call end.
- **Trajectory scale check:** 5,000 calls / 10,002 events / five actors. It
  exercises rendering and batching; it measures no Agency workload capability.
- The earlier three-agent research demos retain context/resources/comparison
  examples. Their values and relationships are synthetic.

Use **Add agent** to display agent columns side by side. Running calls stay
pinned above completed history; newest activities and calls appear first.
The **Multi-agent** view preserves expanded transcripts while live updates
arrive and aligns concurrent work with any recorded system metrics.
Expand an activity for its calls. Select a call for arguments, lifecycle,
timing, output, artifacts, model links, and immutable source identifiers.
Signals link to supporting calls or raw events. Agent focus retains explicit
neighboring workflow notes. The overlay preserves actual parallel lanes.

Inspecting live calls and switching browser tabs keep incoming updates
following. Moving the time cursor or choosing earlier activity pages explicitly
holds history; replay selections also hold history. Changed-call
counts and highlighted cards show what arrived meanwhile. The overview stays
at the held point; the session bar still reports current execution/transport
state. **Follow latest** refreshes the overview and resumes following. Activity pages render at most 50 cards; earlier pages keep the full
accumulated model available without filling the DOM with every call.

## What the evidence establishes

| Data | Provenance and limits |
|---|---|
| Tool start/end, arguments, result, admission ID | Recorded canonical per-agent SQLite events. A successful tool return is distinct from task correctness. |
| Actor and request | Recorded actor registration/request IDs. Process ancestry is not converted into delegation. |
| Declared purpose/workstream | Native harness annotations forwarded separately from executable tool arguments. Model-tool IDs permit explicit invocation links; proximity does not. |
| Model wait, reply, usage, prompt | Recorded state events and exchange chains. Tokens stay on model invocations. Hidden thinking/reasoning blocks are excluded from summaries and raw exchange detail. Live prompt differences remain unknown; full visible context is available on demand. |
| Edit/artifact | Reported result references, with a digest when the producer records one. An edit-shaped command alone is not proof of a changed file. |
| Activity grouping | Derived deterministically from actor, request, workstream, category, and call ordering. One primary assignment per call; groups hold at most eight calls. Late calls get stable separate groups. |
| Attention | Exact failures; repeated matching assertion/check fingerprints without an intervening recorded edit; explicit input requests; scheduler/dependency waits; missing coverage. Successful equivalent checks close prior matching-check signals, without a task-success claim. Parallel overlapping checks are not called consecutive failures. |
| Resources, transfers, evaluator outcome | Existing saved views retain their recorded data. Live resource samples retain their recorded workload/process/device scope; proximity to a call is not attribution. Absent transfer manifests remain unavailable. Recorded evaluator evidence stays separate from execution completion. |
| Semantic suggestions | Optional interpretation beside the stable original label, with exact supporting IDs and quoted result evidence. It cannot move calls, merge actors, or change outcomes. |

The declared-purpose label conveys what the agent says it is attempting.
Observed result text conveys what a tool/model returned. Attention explanations
describe a recorded pattern. These are separate kinds of evidence.

## Implementation and recovery

`trajectory.py` supplies one incremental reducer for live, historical review,
and replay. `LiveSource` reads the existing global/per-agent databases in
batches of at most 500 meaningful rows per table. High-volume syscalls and
streamed tokens do not become activity nodes. Producer changes add request
correlation, forward native annotations, flush tool completion, and record
workload lifecycle; interpretation runs outside the execution process.

`/api/trajectory/live` and `/ws/trajectory` use the same revision/epoch journal,
so events committed between snapshot and subscription are recovered. One
locked reader is shared across clients. Per-table durable row cursors,
deduplication, a bounded 512-patch journal, and snapshot recovery handle
reconnect/server restart. Missing/unreadable databases and pending catch-up
are disclosed. Run switches close the previous subscription. Saved projections
and reconstructed replay events are cached in bounded caches and invalidated
when the source file changes.

The client merges stable objects and patches activity cards in place. Duration
ticks do not count as unread activity. Browser focus scrolling or a shorter
layout does not turn off follow mode. Replay is processed in batches of at
most 100 events; its clock cannot run beyond the recording or an unprocessed
batch. Reconnect resumes its prefix paused, without skipping evidence.

### Optional semantic labels

Set `AGENCY_TRAJECTORY_LABEL_CONFIG=/absolute/path/to/config.json` **on the web
server** to opt in. The file uses the existing Agency LLM config structure:

```json
{"llm": {"provider": "openai", "model": "your-configured-model", "base_url": "your-endpoint"}}
```

Use your normal credentials/configuration. The default performs no model
requests. A single background worker handles at most eight pending closed
activities, once per activity, with a ten-second stream timeout. Only bounded
purpose/tool/outcome/result previews and IDs are sent; existing secret
redaction is applied. Unsupported claims are rejected. Invalid configuration,
credentials, access, or response leaves deterministic grouping intact and
exposes fallback status. Real provider refinement was not exercised in this
implementation verification; successful application and failure paths use
controlled test futures. Refinement does not split or merge published groups.

## Verification

The selected backend/regression suite passed **467 tests, five skipped**, with
two existing WebSocket deprecation warnings in the final selected suite.
The final trajectory suite includes canonical producer/SQLite integration,
hidden-reasoning exclusion, lifecycle, actor/request/workstream boundaries,
late/out-of-order delivery, journal expiry, snapshot/subscription races,
missing databases, prefix-only replay, cancellation, skill failure, non-alert
long/parallel checks, optional-refinement fallback, and large fixtures.
The eleven JavaScript tests cover comparison logic, overlap accounting, stable
merges, unread behavior, 5,000-item batching, and reconnect/stale socket frames.

```sh
.venv/bin/pytest tests/agwebui tests/test_profiler_presentation.py \
  tests/engine/host_servers_tests tests/tool_annotation_effect \
  tests/harness/test_host_client_routes.py tests/harness/test_native_lifecycle_boundary.py \
  tests/test_native_harness_tools.py tests/test_profiler_native.py \
  tests/test_orchestrator_lifecycle.py tests/test_orchestrator_lifecycle_edges.py \
  tests/test_agorchestrator.py -q
node --test tests/agwebui/investigator-model.test.mjs tests/agwebui/trajectory-stream.test.mjs
node --check agency/observability/agwebui/static/investigator.js
.venv/bin/ruff check agency/observability/agwebui tests/agwebui
git diff --check
```

Browser checks exercised real local tools while running, actual returned
assertions, edit/pass/completion, full raw evidence, disconnected transport with
retained execution state, and restart recovery without duplicate calls.
Held history retained the same focused ID, expanded ID, scroll position
162.5px, and first-card viewport position 649.5px through later results and
completion. Recorded replay was checked paused, incremental, and exhausted.
The scale replay completed with 5,000 calls in 625 activities, a 50-card
render window, working earlier-page navigation, and actor filtering. All six
profiler views and the original dashboard were checked again. The live feed
was checked at 390px with no horizontal overflow; viewport overrides were
reset. No application console errors were observed in the final browser check.
Screenshots are in the [gallery](screenshots.md).

Failures found and corrected during implementation: a saved demo edge lacked
an ID (now assigned a stable derived ID); an optional absent RPC reason changed
an existing payload contract (now omitted when absent); fast replay overshot
its clock (now clamped); layout/focus scrolling falsely held an empty history
(now requires a reader scroll gesture); an inspector refresh discarded newly loaded raw evidence (now preserved
across stream refreshes); and a large tool result obscured the
meaningful assertion (now summarized from actual output). The intentional
unittest failures in the live smoke are workload evidence, not viewer test
failures. The initial local-tool validation invoked no model. The real Codex
validation below subsequently exercised an autonomous Agent with credentials.
No optional LLM label request was made, and the upstream Perfetto build was not rerun.

## Real Codex / GPT Luna validation — 2026-10-02

Ran SWE-bench Lite `psf__requests-1963` on the EC2 host described in
`AGENT_INSTRUCTIONS/SSH.md`, using the repository `.env`, OpenAI `gpt-6-luna`,
and Codex CLI 0.147.0. This used `Agent.run()` through Agency's interactive
PTY harness, gateway, policy hooks, scheduler and Docker sandbox. The official
instance image retained its original test environment; a separate Python 3.12
environment supplied the Agency daemon. Only the issue statement reached the
model. The runtime mount contained Agency code and pyproject.toml, excluding
the `.env` and evaluator metadata.

The successful execution took 151.1 seconds: 26 model calls, 31 tool calls,
2,816 projected source events, no unfinished calls. All 26 model calls reported
usage: 598,605 input tokens and 4,493 output tokens. These are reported token
totals, not a price estimate. The live viewer showed running calls, reproduction
output, attempted edits, test output, provider wait intervals, and explicit
completion. Both episode and overlay layouts, model/harness filters, and the
source inspector were checked against that execution. Browser console errors
and warnings were empty at the final check.

The model's new regression test passed, and its focused redirect test run
reported 6 passed. Its full test file reported 120 passed and one pytest API
compatibility failure. Separately, the official SWE-bench evaluator applied the
exported patch and reported **resolved=true**, with all 7 FAIL_TO_PASS and 112
PASS_TO_PASS checks successful, no failures, and no infrastructure error.
The viewer's completion badge describes execution; this independent verdict
is saved in the evaluation artifacts, not inferred from the model's report.

Two preceding model-backed attempts failed while diagnosing a real harness
bug (23 and 19 model exchanges respectively). Codex's UserPromptSubmit hook
trims trailing prompt whitespace, while Agency's PTY driver required an exact
match; it timed out even though Codex had received the task and was working.
The Codex driver now compares prompts after trimming their trailing whitespace,
while retaining the unique turn marker. A regression test verifies trimmed
acknowledgments and rejects a stale marker. Another setup attempt failed before
any model call; the final runner pins the daemon's Python environment. All
attempts remain on the remote host rather than being overwritten. The runner
also keeps its terminal event even if failed-workflow patch collection fails.

Live workload metadata now updates the header and model/harness filters.
External Codex tool hooks still omit explicit model-call linkage and often
provide shell output without an exit code; the source output is available in
the inspector, and call completion alone does not establish command or task
correctness.

The reproducible runner is `examples/12_swebench_live.py`. Start the server
separately against the same fresh run directory's `logs/`:

```sh
python examples/12_swebench_live.py \
  --task-file /absolute/path/to/one-official-task.json \
  --env-file /absolute/path/to/.env \
  --run-dir /absolute/path/to/fresh-run \
  --image agency/profiler-swebench-requests-1963:20261002 \
  --codex-binary /absolute/path/to/codex-0.147.0 \
  --model gpt-6-luna
python -m agency.observability.agwebui.server \
  --run-dir /absolute/path/to/fresh-run/logs --port 7865
```

The example image requires `/opt/agency-harness-venv/bin/python` with Agency's
harness dependencies. The actual prepared host setup and evaluation command
are recorded in `artifacts/profiler-live-swebench-20261002/validation.md`.
That directory also contains the generated patch, official report, test output,
portable trajectory projection, and original SQLite logs. Provider-key scans
found no configured provider credentials in the successful run's artifacts.
The live SSH viewer is at `http://localhost:7865/profiler?run=live&view=trajectory`.
This validates one task and one harness/model combination; multi-agent overlay
behavior was separately checked using the existing multi-actor recordings and
explicitly synthetic scenarios.

## Real two-agent workflow — 2026-10-02

The same SWE-bench Lite task was also resolved using two real GPT Luna / Codex
agents in separate Docker workspaces. One implemented the source fix while the
other wrote a regression test and observed the original bug. Their initial
skills overlapped for 174.95 seconds. The workflow transferred the test patch
to the implementation agent, recorded the actual handoff, and reused that
agent to verify the combined patch. Exactly two agents were created.

The successful run completed in 283.57 seconds with 27 model calls, 31 tool
calls, 3,263 source events and no unfinished calls. Its new regression passed;
the independent official evaluator reported **resolved=true**, with all
7 FAIL_TO_PASS and 112 PASS_TO_PASS checks successful and no infrastructure
failure. The overlay showed both real actors on a shared clock, and the
Multi-agent view displayed the recorded patch handoff.

Use `examples/12_swebench_live.py --agents 2 --request-interval 10` with the
other prepared-run arguments above. Shared pacing also covers SDK retries.
Earlier attempts exposed a provider token limit, stripped patch termination,
and a model stopping when a shell edit command was absent. The runner preserves
patch termination, supplies available editing guidance, and can follow up on
the existing test agent. The selected runner/harness/viewer checks passed
107 tests. These retries are documented; this validates integration rather
than a benchmark success rate or a speed improvement.

Artifacts and reproduction details are in
`artifacts/profiler-two-agent-swebench-20261002/validation.md`.
The completed viewer is at
`http://localhost:7866/profiler?run=live&view=trajectory&layout=overlay`.

## Remaining limits and next work

One server observes one configured run directory. History/projection memory
grows with a run even though reading, updates, semantic work, and feed DOM are
bounded. The overlay and other saved profiler views are not virtualized.
Very large raw outputs are loaded whole on demand. Database replacement/truncation
within a run is not treated as a new execution; use fresh run directories.
Raw wall-clock timestamps are retained, but host clock skew can limit temporal
alignment; late events do not reorganize published episode membership.

Native annotations and explicit model IDs are connected; external harnesses
may lack tool/context correlation and therefore display unlinked evidence.
Input/delegation/handoff event types are supported, but all harnesses do not
currently emit them. Live pause/resume reuse the existing process controls;
redirect/cancel controls are outside this profiler change, and cancellation
requires an invocation handle.

The highest-value next step is consistent invocation-scoped input, delegation,
transfer, and artifact/validation events across harnesses. That would turn
currently missing workflow context into recorded evidence and support genuine
interventions tied to a trajectory point.
