# Validation report

Terminal-Bench 4.0 resource-contention benchmark, validated 2026-10-12 on the
experimental host (Ryzen 7 9800X3D, 8 cores / 16 threads, 30 GiB RAM, kernel
7.0.0, rootless Podman 5.7.0, cgroup v2).

**Readiness: READY WITH DOCUMENTED LIMITATIONS.**

The runner, limiter, isolation, profiling, artifact pipeline and grading all
work end to end on real task containers with the real Codex CLI and Agency.
No paid model request was made at any point: every agent run used a local mock
endpoint. What remains unverified needs the live gateway and is listed under
"Requires live validation".

The experimental matrix was not executed.

## What was run

| Layer | What | Result |
|---|---|---|
| Unit | token estimation, budget accounting, rolling window, reconciliation modes, concurrency (threads and processes), cancellation, failures, ledger integrity, config, plan, topology, corpus rules, parsers, timeline analysis, CLI and exit codes | 150 tests pass |
| Integration | Agency backend + OpenAI SDK + local HTTP endpoint through the limiter; Codex request → Agency wire format; systemd slice and cpuset enforcement on this kernel; worker isolation checks; Harbor dataset, collection and grading on real images | 38 tests pass, 1 opt-in |
| Concurrency (mocked workers) | 24 tasks at 6, 12 and 24 agents through the real scheduling loop, real processes and real shared ledger | 8 tests pass |
| Failure injection | worker death, setup failure, timeout, ignored deadline, ignored SIGTERM, operator stop, signal, run time limit, collection failure, grading failure, corrupted ledger, partial artifacts, insufficient CPUs, shared writable state, run lock, low disk | 20 tests pass |
| **Total** | `pytest benchmarks/terminalbench_contention/tests` | **216 pass, 1 opt-in skipped** (run separately: pass) |
| End-to-end smoke, 1 agent | real container, Codex, Agency, mock model, grading | pass |
| End-to-end smoke, 2 agents × 4 tasks, tight budget | shared quota across 4 worker processes | pass |
| End-to-end smoke, 24 agents × all 24 task images | full concurrency with session CPU confinement | pass, 24/24 |
| End-to-end, HTTP 429 from the endpoint | retry, global cooldown, ledger | pass |
| End-to-end, task time cap | censored at the cap, stopped, collected, graded | pass |
| Grading, all 24 tasks | Harbor reference solution → reward 1.0; untouched environment → reward 0.0 | 24/24 |
| Token estimator calibration | 6 809 recorded requests from the previous experiment | see below |

Reproduce:

```sh
.venv/bin/python -m pytest benchmarks/terminalbench_contention/tests -q -p no:cacheprovider
AGENCY_TB_E2E=1 .venv/bin/python -m pytest benchmarks/terminalbench_contention/tests/test_harbor_integration.py -q -p no:cacheprovider
scripts/agency-bench smoke --agents 24 --tasks 24 --no-grade --latency 2 --stream-seconds 1 --set cpu.confine_user_session=true
scripts/agency-bench oracle
scripts/agency-bench calibrate-tokens --source artifacts/cpu-baselines/20261012T160531.255904Z/runs
```

Artifacts are under `artifacts/terminalbench-contention/validation/`:
`unit-tests/junit.xml`, `smoke-tests/*/attempt-NN/`, `oracle/results.json`,
`token-calibration.json`, `preflight.json`.

## Evidence for critical invariants

### End-to-end path (requirement 12 C)

Single-agent smoke on `wal-recovery-ordering`, real container:

1. Isolated environment: container in `agencytb.slice`, cpuset `2-7,10-15`,
   `cpu.max = max`, memory limit 4 GiB, not privileged; the in-container probe
   could not widen its affinity.
2. Model request through Agency: 3 requests, Codex → Agency proxy → Agency
   OpenAI-compatible backend → HTTP endpoint.
3. Shared limiter: 3 admissions and 3 reconciliations in the ledger; ledger
   verification passes.
4. Tool executed: the scripted shell command ran in the container and its
   output file was collected.
5. Profiling: Agency `agprof.trace.json` and summary, 10 Hz container trace,
   host samples, run-level Perfetto trace, per-state CPU attribution.
6. Normal termination: status `completed`, no leftover container or unit.
7. Grading: Harbor verdict, reward 0.0 (the scripted agent does not solve the task).
8. Artifacts: full run directory written.

Multi-agent smoke (4 tasks, 2 concurrent, 30 000 tokens per 10 s): 12 requests
from 4 distinct worker PIDs; peak window occupancy 29 986 of 30 000; 10 of 12
admissions waited; waits appear as `api_wait` in each agent's trajectory.

### Full concurrency on real containers

24 agents over all 24 task images, mock model with 2 s latency:

* 24 of 24 completed and collected; peak in flight 24; no agent errors.
* Every container passed its isolation check; no writable host path shared by
  two agents; no leaked containers or units.
* Pool CPU reached 100%; workload CPU pressure 11 s; pool runqueue delay
  333 s; 6 s of measured CPU stall per agent, mostly during harness start-up
  (13 s versus 4 s for a single agent). The contention signals are visible.
* Workload memory peaked at 11.1 GB of the 22 GB ceiling; no OOM kills.
* Session confinement was applied (`app.slice` etc. → `0-1,8-9`) and restored.
* Graded afterwards with `agency-bench grade`: 24 of 24 verdicts, all reward 0.0
  (the scripted agent solves nothing), no grading failures.

### Scheduling and methodology (mocked workers)

* Each of 6, 12 and 24 agents: peak concurrency exactly the limit, never above.
* All 24 tasks ran exactly once, in the planned order, in every condition.
* The three conditions of a repetition share one seed and one launch order;
  repetitions differ. Rebuilding the plan reproduces it.
* One API policy fingerprint, one CPU pool, one timeout set and one
  cache-control record across conditions.
* Every worker process saw exactly the pool CPUs (`2-7,10-15`).
* Per-run ledgers start at sequence 1 and share nothing.
* State durations sum to each task's wall time; units and denominators are
  stated in `summary.json` and `capabilities.json`.

### API accounting

Offline, against 6 809 recorded requests (154.3 M reported prompt tokens):

| Prompt estimate ÷ reported | mean | p05 | p95 | min | max | below reported |
|---|---|---|---|---|---|---|
| Previous `bytes / 2` | 2.340 | 2.108 | 2.577 | 1.973 | 2.984 | 0% |
| Tokenizer count | 1.001 | 0.981 | 1.021 | 0.954 | 1.310 | 46% |
| Default (×1.05 + 16) | 1.052 | 1.031 | 1.073 | 1.002 | 1.377 | 0% |

Total charge for those requests: 383.4 M tokens under the previous rule,
189.7 M now, against 155.2 M reported. Of the remaining 34.5 M, 27.0 M is the
completion reservation and 7.5 M the prompt margin.

Tests cover: charge = prompt estimate + declared completion cap; window
blocking and expiry; four reconciliation modes; failed-request handling;
429 cooldown across clients; deadline and cancellation while waiting; 8 threads
and 6 processes never exceeding the budget; fail-closed behaviour on corrupted
state; detection of removed, altered and torn ledger entries; identical
admission regardless of agent identity.

### Grading

For each of the 24 tasks, on its real images: an untouched environment grades
to 0.0 and Harbor's reference solution to 1.0, using Harbor's `OracleAgent`,
`ArtifactHandler`, `RegradeTrial` and `Verifier`.

### Failure handling

| Injected failure | Outcome |
|---|---|
| Worker killed mid-task | task `worker_died`, run continues, no retry, exit 5 |
| Setup failure in the first wave | nothing measured, exit 3 |
| Setup failure in a later task | task `setup_failed`, run continues, exit 5 |
| Task reaches its time cap | `timeout`, censored, still collected and graded |
| Worker ignores its deadline / SIGTERM | runner terminates, then kills; recorded |
| `STOP` file, SIGTERM, run time limit | agents stopped, rest `not_started`, exit 4 / 130 |
| HTTP 429, HTTP 500, API timeout, cancelled stream | every attempt settled in the ledger; 429 pauses all agents |
| Collection or grading failure | recorded separately; never scored as zero |
| Corrupted ledger | admission stops, corrupt state kept, nothing rebuilt |
| Torn JSON / JSONL, missing files | analysis completes with what is readable |
| Fewer than 8 physical cores | refuses to plan |

## Problems found and fixed during validation

1. **Two selected tasks could not run under Agency's Codex harness.**
   `gsea-proteomics` and `react-lead-form` have no `python3` on
   `/usr/local/bin:/usr/bin:/bin`. Agency installs its Codex hooks as
   `python3 <script>`, so the prompt was never acknowledged and the task failed
   after 60 s. Found by the 24-task smoke, reproduced with each task alone.
   Both were replaced by eligible tasks of the same category
   (`hof-topology-interpenetration`, `data-anonymization`); the requirement is
   now an eligibility rule and an init check. Category and workload mix are
   unchanged.
2. **Agency stops the container when a skill completes**, which removes its
   cgroup. Task-end counters are now read immediately before Agency's
   checkpoint and stop, and collection resumes the stopped container.
3. **Harbor's regrade refuses a record with an uncollected artifact**, so an
   agent that never wrote its output got an error instead of a score. Absent
   outputs are now proven absent at collection and tolerated at grading.
4. **Harbor's Podman backend needs a Compose frontend** that is not installed.
   Replaced by a small Harbor environment class driving `podman` directly.
5. **Agency mounts its repository directory into the sandbox by default**,
   which would expose `.env` to tasks. The sandbox now mounts a frozen copy of
   the `agency` package only.
6. Process-launcher affinity, systemd `MemorySwapMax` spelling, run-deadline
   enforcement, ledger-verifier robustness and stacked instrumentation hooks.

## Requires live validation

These cannot be checked without real gateway requests and were not attempted:

1. **Gateway rate limits and counting rules.** The 1.2 M TPM budget is the
   historical local value. The deployed LiteLLM version, its limiter settings
   and any upstream limit are unknown. An earlier check in the previous
   experiment's artifacts saw no rate-limit headers from the gateway.
2. **Tokenizer for Luna.** `o200k_base` matches recorded usage from the previous
   experiment to within about 2% for 90% of requests; that is evidence, not confirmation
   for the current model deployment.
3. **Usage reporting on streamed responses** from the live gateway, including
   cached-token fields. The code path is exercised with the mock.
4. **Real 429 behaviour and headers** from the gateway.
5. **Agent behaviour on these tasks with the real model**: wall times, how many
   tasks reach the 30-minute cap, real memory and disk use at 24 agents.
6. **Provider-side caching** across runs.

Suggested first live step: one `scripts/agency-bench run --agents 6 --rep 1`
treated as a pilot, then compare `api-metrics/summary.json`
(`estimated_over_reported_prompt`, `requests_estimated_below_reported`,
`http_429`) with expectations before continuing.

## Limitations that can affect validity

1. **Task time cap.** 1 800 s per task against Terminal-Bench's 28 800 s. Many
   tasks may be censored, more at higher concurrency. Pass rates are not
   comparable with published Terminal-Bench results. Censored tasks are graded
   and reported separately.
2. **Admission budget is local.** If the gateway limits earlier than the local
   budget, 429 cooldowns (60 s, global) will dominate API waiting.
3. **Minimum request spacing** (0.05 s, inherited) caps admissions at 20 per
   second regardless of token budget.
4. **Host page cache is not controlled**; neither are provider caches.
5. **Network access is on** in task containers, as the tasks specify. Package
   downloads vary with the network. `sandbox.network=none` is available but
   changes task conditions.
6. **Not confined to reserved CPUs:** root-owned services, kernel threads,
   interrupts.
7. **Harness overhead is inside task wall time:** about 4 s of harness start-up
   alone, 13 s when 24 start together, plus Agency's end-of-task container
   checkpoint. Both are separate measured states.
8. **Setup of refill tasks** (container creation, about 2–10 s) happens during
   the measured interval for LSB-6 and LSB-12, and before it for the first
   wave. It is recorded per task (`setup_seconds`) and is not part of task wall
   time, but it is part of makespan.
9. **Slot turnaround** (collection and container removal, about 1–10 s) delays
   the next launch and occurs 18 times at LSB-6, 12 at LSB-12, none at LSB-24.
   Recorded per task (`turnaround_seconds`).
10. **Per-agent I/O** comes from `/proc/<pid>/io` sums over live processes
    because the `io` cgroup controller is not delegated: a lower bound.
    Per-thread run delay and context switches are likewise lower bounds. Cgroup
    CPU time and pressure totals are exact.
11. **Per-tool CPU** is the container's CPU over the tool call's wall interval;
    concurrent background work in the same container is included.
12. **Corpus is 24 of 45 eligible tasks**, chosen by hand for coverage, and
    excludes GPU, multi-container and large-memory tasks. Results describe this
    corpus.
13. **Swap is allowed** for the workload. Heavy swapping would slow every agent;
    it is measured, not prevented.
14. **Two tasks were swapped during validation** for harness compatibility
    (above). The frozen manifest is the final corpus.
15. **Grading differs from stock `harbor trial regrade`** in tolerating absent
    outputs (above).
16. **Dirty working tree.** The benchmark is not yet committed; `metadata.json`
    records the commit and a `dirty` flag. Commit before measured runs so each
    run names an exact revision.
