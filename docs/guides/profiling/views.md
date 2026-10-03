# Profiler views and interpretation

A working, read-only investigator at `/profiler`, alongside the existing live
run dashboard. Six views share a run, selected agent,
action/episode, inspector, event search, and time cursor. No frontend package
installation or build step is needed.

## Open it

Follow the [profiling guide](../profiling.md#open-a-run) to start the server
or reopen your application's log directory.

Open <http://localhost:7860/profiler>. The catalog discovers saved traces under
`agency_runs/` and `artifacts/`, the E1A.1 experiment catalog, native annotation
observer logs, and the configured profiler directory. `AGENCY_PROFILE_DIR`
continues to select an external profiler directory.

The representative demo is available directly at
<http://localhost:7860/profiler?run=demo-baseline&view=trajectory>.
“Open research demo” also opens it from any view. The matching contention run
is selected automatically in Compare runs. Numbers **1–6** switch views;
**Escape** clears the selected event. Export run downloads the normalized data,
including source labels.

All six investigator views start without a viewer build. The embedded
Perfetto viewer has been removed; recorded Chrome traces remain available
through the raw trace download endpoints. Native observer JSONL remains
available in the investigator.

For incremental execution, replay, provenance, and verification details, see
[Live trajectory](live-trajectory.md). The current run is available at
<http://localhost:7860/profiler?run=live&view=trajectory>.

## What is implemented

- **Trajectory:** live lifecycle updates and saved-trace replay through one
  incremental reducer; declared-purpose/category episodes with fixed actor and
  request boundaries; evidence-backed attention and history/follow controls;
  expandable model/tool actions, commands, results, timing, files, and metadata.
  Episode list includes selectable agent columns, pinned running calls, and
  completed history with newest activity first. The Layout dropdown switches
  between Episode list and Agent overlay. The overlay places each agent's actions on a shared time axis, separates nested
  actions into tracks, and retains the same selection, inspector, agent/search
  filters, and time cursor. `layout=overlay` preserves the layout in shared URLs.
- **Resources & waits:** common time axis with model/tool/queue/dependency/gap
  lanes, CPU/memory samples, longest waits, and interval zoom. Selecting a wait
  exposes its causal action while preserving the wait's selected interval.
- **Context & information:** model-call navigation; source composition; added,
  repeated, and dropped prompt blocks; content previews; supervisor/worker
  information boundary in the demo.
- **Multi-agent:** selectable agent transcript columns with expandable
  activities, token/time summaries, concurrent work and sampled system counters
  on a shared time axis, explicit parent relationships, recorded/synthetic
  handoffs, and identical-command detection.
- **Compare runs:** matching-task defaults; longest-common-subsequence alignment
  of action categories; inserted/deleted actions, first divergence, reconvergence,
  per-action duration changes, resource peaks, models/harnesses, referenced-path
  differences, evaluator outcomes, and navigation into either execution.
- **Task progress:** obligations with evidence links and explicit unvalidated
  work; evaluator result separated from inferred tool completion.

## Data flow and provenance

`server.py` adds `/api/investigator/runs`, `/api/investigator/runs/{id}`, and a
read-only raw trace endpoint. Paths are server-side catalog entries, addressed
by stable hashes; API callers cannot supply arbitrary filesystem paths.

`investigator.py` adapts Chrome/Perfetto JSON and native observer JSONL into one
schema: agents, actions, episodes, intervals, counters, edges, and obligations.
It converts trace microseconds to seconds, retains outcome/timing/provenance
metadata, decodes named resource counters, and groups consecutive actions per
agent. Explicit measured durations remain separate from coverage gaps.
The catalog refreshes every 30 seconds; four normalized saved runs are cached
by file modification time. Counter series are reduced to at most about 800
points for rendering. Full raw traces remain available through the download endpoints.

The frontend is native HTML/CSS/ES modules, matching the repository's existing
frontend architecture. `investigator.js` owns shared selection and rendering;
`investigator-model.js` contains alignment, filtering, and interval-union logic.
The responsive layout moves the inspector below the investigation on narrower
screens. User-supplied trace strings are HTML-escaped.

**Real:** saved tool commands/results and admission IDs; native model-stated
purpose annotations joined by `event_id`; model usage; timed spans; sampled CPU
and memory; explicit agent parents; recorded prompt text; E1A.1 condition,
repetition, and official evaluator outcomes; native experiment verifier results.
Native observer durations have `observer_duration` timing; their start is the
completion timestamp minus reported duration, with observer emission overhead.

**Inferred:** action categories, episodes, command-referenced paths, summary
labels, and workflow obligations. Activity completion means its recorded calls returned successfully;
it is not proof of task correctness. Comparison similarity measures
category ordering, not semantic equivalence. Identical commands across agents
suggest duplication; concurrency alone does not prove resource interference.

**Synthetic:** `investigator_demo.py` contains two explicitly labeled executions.
`trajectory_scenarios.py` separately contains edge-case and 5,000-call stress fixtures.
This fixture includes three agents, delegation, handoff differences, dependency
and scheduler waits, resource pressure, a failed test followed by recovery,
context changes, a trajectory insertion, and an unvalidated GPU obligation.
Recorded runs are never silently supplemented with demo measurements.

**Unavailable:** missing prompt/resource/communication data stays unavailable.
Truncated `llm.messages` attributes are recovered only through complete JSON
message prefixes and marked incomplete; dropped blocks are unknown for partial
prompts. Composition uses character counts; measured token totals are separate.
Model-wait wall time unions overlapping intervals rather than adding nested or
parallel spans. CPU work is not inferred from uninstrumented wall time.

## Instrumentation to add next

1. Full prompt-chain references and content hashes instead of truncated span
   attributes, with explicit compaction/drop boundaries and source/file IDs.
2. Supervisor/worker transfer manifests, message IDs, send/receive timestamps,
   dependency IDs, and queue-release causes.
3. Verified artifact inventories/diffs, task obligation IDs, and independent
   validation/evaluator events for workflows outside the saved experiments.
4. Per-agent/cgroup resource attribution and sampled contention conditions;
   exact CPU/I/O intervals where available. Live trajectory streaming now uses
   canonical event databases; sampled live resource attribution remains future work.

## Verification and screenshots

```sh
.venv/bin/pytest tests/agwebui tests/test_profiler_presentation.py -q
node --test tests/agwebui/investigator-model.test.mjs
.venv/bin/ruff check agency/observability/agwebui tests/agwebui/test_investigator.py
node --check agency/observability/agwebui/static/investigator.js
```

Browser validation covered all six views, episode expansion, inspector links,
wait attribution/zoom, model context blocks, agent filters, divergence and run-B
navigation, obligation evidence, empty searches, real E1A.1 baseline/contention
runs, and the mobile layout. No browser console errors were observed. The mobile
layout was checked at 390px and had no horizontal overflow.

The [screenshot gallery](screenshots.md) includes all six finished views, a
recorded resource view, and the mobile layout.
