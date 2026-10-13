# Terminal-Bench 4.0 resource-contention benchmark

This benchmark runs a fixed corpus of 24 Terminal-Bench 4.0 tasks through
Agency's Codex harness at three concurrency levels (6, 12 and 24 agents) on one
shared six-core CPU pool, one shared memory ceiling and one shared API
admission budget, and records enough to reconstruct what every agent was doing
and what it was waiting for.

It asks a neutral question: does default Linux scheduling plus naive API
admission coordinate concurrent agents efficiently, or not? Nothing here is a
scheduler, and nothing steers agent behaviour toward a result.

The runner executes one planned run and exits. Monitoring, retries and
experiment management belong to a separate operator.

## Quick reference

```sh
scripts/agency-bench prepare                 # once: tools, dataset, images, init checks
scripts/agency-bench validate                # host + environment + integration checks
scripts/agency-bench plan                    # dry run: the nine runs, nothing launched
scripts/agency-bench run --agents 6 --rep 1  # ONE condition and repetition (paid model calls)
scripts/agency-bench report                  # summarise finished runs
```

`scripts/agency-bench` is `.venv/bin/python -m benchmarks.terminalbench_contention`.

| Exit code | Meaning |
|---|---|
| 0 | success |
| 2 | usage or configuration error |
| 3 | precondition failed; nothing was measured |
| 4 | run aborted, or its evidence is unusable |
| 5 | run completed, but some tasks had infrastructure or grading failures |
| 130 | interrupted by a signal |

## Experiment

| Condition | Max concurrent agents | Physical cores | Repetitions |
|---|---|---|---|
| LSB-6 | 6 | 6 | 3 |
| LSB-12 | 12 | 6 | 3 |
| LSB-24 | 24 | 6 | 3 |

Every run processes the same 24 tasks, each exactly once. The concurrency
number is the maximum number of agents in flight; when one finishes, the next
pending task launches. Conditions differ in nothing else.

* **Benchmark:** `terminal-bench/terminal-bench@4.0.0`, downloaded through
  Harbor 0.24.0. Task digests, image digests and tool lock-file hashes are in
  `manifests/tasks.json` and `manifests/benchmark.json`.
* **Launch order:** a seeded shuffle per repetition. All three conditions of a
  repetition use the same order; repetitions use different orders. Seeds and
  orders are stored in every run's `plan.json` and `metadata.json`.
* **Condition order:** rotated across repetitions (6-12-24, 12-24-6, 24-6-12)
  so no condition always runs first.
* **Model:** `gpt-6-luna` through LiteLLM, using the existing configuration
  (`provider="openai"` OpenAI-compatible backend, `reasoning_effort="none"`,
  `max_completion_tokens=4096`, `LITELLM_API_KEY` / `LITELLM_BASE_URL` from
  `.env`). The path is unchanged: Codex → Agency proxy → Agency LLM backend →
  LiteLLM.
* **Timeout policy:** one wall-clock cap per task (`timeouts.task_seconds`,
  default 1800 s) from the moment the agent is released. A task at the cap is
  stopped, marked `censored` and still graded. Tasks never released before a
  run ended are `not_started`. Nothing is dropped from the results.
  The benchmark retains a cancelled or failed skill's container until artifact
  collection, overriding Agency's normal transaction discard for that container.
  Terminal-Bench's own task timeout is 28 800 s; the shorter uniform cap is a
  deliberate, recorded deviation (see Limitations).

### Task corpus

`manifests/tasks.json` lists the 24 tasks with digests, pinned images, declared
resources, an expected workload label and a rationale. Selection:

1. Eligibility rules applied mechanically to all 66 tasks, from fields Harbor
   parsed: no GPU; single container; no verifier collect hooks, MCP servers,
   agent skills or healthcheck services; declared memory ≤ 8 GB and CPUs ≤ 6;
   separate-mode verifier; digest-pinned prebuilt images; root user; public
   network policy; verifier needs no credentials; and the image provides
   `python3` on `/usr/local/bin:/usr/bin:/bin`, which Agency's Codex hooks
   need. 23 tasks are excluded, each with its reasons in the manifest.
2. 24 of the 43 eligible tasks, fixed to cover all seven categories (Software 8,
   Science 5, ML 4, Operations 3, Security 2, Hardware 1, Media 1) and a mix of
   expected behaviour: 4 inference-heavy, 8 computation-heavy, 12 mixed. The
   workload label comes from reading each task, not from measurement.

Every selected task was started once with the harness runtime mounted, its
verifier image was checked, and Harbor's reference solution was graded to
reward 1.0 through this benchmark's grading path (see the validation report).

## How it works

```
runner (reserved CPUs)
 ├─ agencytb.slice  ── one cgroup: AllowedCPUs = pool, MemoryMax = 22G
 │    ├─ worker unit ×N   Agency host side: engine, model proxy, profiler
 │    └─ container  ×N    task image + Codex CLI + Agency harness daemon
 ├─ shared API ledger     one flock-guarded directory for all workers
 ├─ resource sampler      host, slice and per-agent cgroup counters
 └─ Harbor (pinned venv)  artifact collection, then grading after the run
```

| Module | Role |
|---|---|
| `cli.py` | commands and exit codes |
| `config.py` | defaults, strict overrides, plan and seeds |
| `corpus.py`, `manifests/` | eligibility rules, frozen corpus |
| `prepare.py` | pinned tools, dataset, images, init checks, oracle validation |
| `topology.py`, `placement.py` | CPU pool from sysfs; shared slice; launchers |
| `runner.py` | one run: barrier, scheduling, deadlines, collection, cleanup |
| `worker.py` | one agent: container, isolation checks, Agency Codex run |
| `tokens.py`, `api_budget.py`, `api_instrument.py` | estimator, shared ledger, telemetry |
| `sampler.py`, `analysis.py` | kernel counters; trajectories and metrics |
| `cache_control.py` | per-cache-class strategy and host state |
| `harbor_side/` | code that runs inside the Harbor virtualenv |
| `mock_model.py`, `mock_worker.py` | local mock endpoint and stand-in worker for validation |
| `calibrate.py` | offline estimator calibration against recorded usage |

### Harbor integration

Harbor provides the dataset, the task model, artifact collection and grading.
Agency provides agent execution and instrumentation.

* Tasks are downloaded with `harbor download` and parsed with Harbor's models.
  The published 4.0.0 tasks pin prebuilt environment and verifier images by
  digest; those images are pulled, never rebuilt.
* The agent runs in an Agency sandbox created from the task's environment
  image. Harbor's own Codex integration is not used.
* When an agent stops, Harbor's `ArtifactHandler` collects the task's declared
  artifacts from that container into a standard Harbor trial record.
* Grading uses Harbor's `RegradeTrial`: a fresh verifier container from the
  pinned verifier image, Harbor's `Verifier`, Harbor's reward parsing.
* `harbor_side/agency_podman_env.py` is a small Harbor environment class that
  drives `podman` directly. Harbor's built-in Podman backend needs a Compose
  frontend that is not installed here.
* One deliberate difference from `harbor trial regrade`: a declared output the
  agent never created is tolerated, so the verifier runs and the task scores
  what its tests give (normally 0). Harbor's regrade would refuse the record.
  An output that exists but could not be copied remains a collection failure.

### CPU allocation

The Ryzen 7 9800X3D has eight physical cores with SMT.

* Physical cores are found from `/sys/devices/system/cpu/*/topology`; nothing
  assumes a numbering scheme. On this host the pool is logical CPUs
  `2-7,10-15` and the reserved set is `0-1,8-9`.
* The two lowest-numbered physical cores are reserved for the OS, the runner,
  the sampler, artifact collection and grading.
* **SMT decision:** the pool contains both hardware threads of each of its six
  cores (`cpu.smt_policy="siblings"`), in every condition. Reserving and
  allocating whole cores means no reserved thread shares a core with a pool
  thread. `"primary"` (one thread per core) is available as an override.
* All agents share the pool through one systemd user slice with
  `AllowedCPUs=<pool>`. Each container also gets `--cpuset-cpus=<pool>`.
* There is no per-agent CPU quota, weight or pinning. Agency's own `--cpus`
  limit is disabled (`idle_cpus=None`). Before each task the worker checks the
  container's `cpu.max` is `max`, its effective cpuset is the pool, and that a
  process inside cannot widen its affinity; the task does not start otherwise.
* The scheduler is whatever Linux provides. Nothing is tuned.
* With `cpu.confine_user_session=true` (default) the rest of the user session
  is restricted to the reserved CPUs for the duration of a run and restored
  afterwards, including after a crash (`cleanup`).

### Isolation and shared resources

Isolated per agent: container root filesystem, process tree, workspace, logs,
profile, Agency state, Codex home. Each agent's writable host mounts are checked
to belong to it alone, and `summary.json` reports any path two agents share.
The sandbox imports a frozen copy of the `agency` package, not the repository,
so `.env` is not readable from inside a task.

Shared by all agents: the CPU pool, host memory up to `memory.workload_memory_max`
(22G of 30G), disk, and the API budget.

Memory limits are for safety only: each container is limited to the memory its
task declares, and the slice limit keeps the host alive. Swap is allowed up to
the host's configured swap so memory pressure shows as reclaim and swap before
OOM. The runner aborts a run, keeping artifacts, if host `MemAvailable` stays
below 1 GiB for 10 s or free disk falls below 15 GiB.

### API admission

The policy is the previous benchmark's, unchanged in kind: every request from
any agent polls one shared rolling-window budget and is admitted the first time
it fits. No queue, no priority, no ordering by agent, size or age. Agent
identity is recorded and never read by the admission decision.

What changed is the accounting.

| | Previous | Now |
|---|---|---|
| Prompt charge | `ceil(request bytes / 2)` | tokenizer count of the actual request: messages, tool calls, tool schemas, framing; ×1.05 + 16 margin |
| Completion charge | `max_completion_tokens` | same reservation, tracked as a separate number |
| Reported usage | not used | recorded per request, separate from estimates |
| Reconciliation | none | configurable; see below |
| Ledger | window entries only | every admit, settle, fail, 429 and censored wait |

Measured against 6 809 recorded requests from the previous experiment
(154 M reported prompt tokens), offline:

| Prompt estimate ÷ reported prompt tokens | mean | min | max | under-estimates |
|---|---|---|---|---|
| Previous rule | 2.34 | 1.97 | 2.98 | 0% |
| Tokenizer count, no margin | 1.001 | 0.954 | 1.310 | 46% |
| Default (×1.05 + 16) | 1.052 | 1.002 | 1.377 | 0% |

The total admission charge falls from 2.47× to 1.22× of reported tokens. The
rest is the margin and the completion reservation (4 096 reserved per request
against roughly 130 used).

`api.*` configuration:

* `tokens_per_window` — **1 200 000 per 60 s, the historical local budget, not a
  verified provider limit.** Override it once the gateway's limits are known:
  `--set api.tokens_per_window=...`.
* `requests_per_window` — unset. No request-rate limit is invented.
* `completion_reservation` — `max_completion_tokens` (default) or `none`.
* `reconciliation` — how a reservation is settled when usage is reported:
  * `conservative` (default): `max(admitted, reported total)`. Never refunds.
  * `reported_total`: replace with reported prompt + completion tokens.
  * `litellm_v3`: reported total minus cached tokens, and refund failed
    requests. This mirrors LiteLLM 1.105's proxy limiter, which reserves
    `message characters / 4 + max_tokens` up front and reconciles after the
    call. The deployed gateway's version and settings are unverified.
  * `off`.
* `release_on_failure` — whether a request with no usage (error, 429, cancel)
  gives its charge back. Default no.
* `minimum_request_spacing_seconds` (0.05) and `rate_limit_cooldown_seconds`
  (60): a 429 seen by any agent pauses admission for all, for the longer of the
  cooldown and the provider's `Retry-After` / reset headers.
* `estimator.*` — encoding (`o200k_base`), safety factor and constant. If the
  tokenizer cannot load, a deliberately high bytes-based bound is used and
  every affected record says `estimate_method: fallback_bytes`.

Each HTTP attempt, including the SDK's own retries, is admitted separately.
Reported usage is never assumed to equal what an upstream provider counted.

### Profiling

| Source | What | Where |
|---|---|---|
| API telemetry (worker) | per request: ready, admitted, headers, first chunk, completed; estimate, reservation, charge, reported usage, reconciliation; retries, 429s; budget available and waiters at admission | `agents/*/api-requests.jsonl`, `api-metrics/` |
| Agency profiler (`agprof`) | OpenTelemetry spans, Perfetto trace, tool boundaries, harness lifecycle | `agents/*/profile/` |
| Container trace (worker, 10 Hz) | cgroup CPU time, CPU/memory/I/O stall totals, memory | `agents/*/container-series.jsonl` |
| Task boundary counters (worker) | exact per-task cgroup totals: CPU, stalls, peak memory, reclaim, OOM events | `agents/*/counters-{start,end}.json` |
| Resource sampler (runner, 1 Hz; per-agent 0.5 Hz) | pool and reserved CPU time, PSI, runqueue delay, memory, reclaim, swap, disk, slice counters, per-agent threads (runnable, run delay, context switches) and I/O, API budget occupancy | `resource-metrics/samples.jsonl` |

`resource-metrics/capabilities.json` states which sources this host supports.
Here the cgroup `io` controller is not delegated to the user manager, so
per-agent I/O is summed from `/proc/<pid>/io` and I/O pressure comes from the
cgroup PSI files.

**Measured versus inferred.** Agent states are intervals with measured
timestamps: `api_wait`, `model_wait`, `model_stream`, `tool`, `checkpoint`,
`harness_startup`. The one inferred state is `other`, the remainder of the
task's execution window. Time runnable-but-not-running and time stalled on
memory or I/O are reported as amounts from kernel counters (cgroup pressure
totals, runqueue delay), per task, per state and per sampling interval. They
are never drawn on the timeline by inference from application events.

## Operator instructions

Run from the repository root, on the host, outside any restricted sandbox.

### Once per host

```sh
scripts/agency-bench prepare
scripts/agency-bench validate --require-credentials
scripts/agency-bench plan --write
```

`validate` must print `validation passed`. A `[warn]` line (the `io`
controller) does not block.

Before measuring, set the admission budget to the gateway's verified limits if
they differ from the historical default, and use the same value for all nine
runs:

```sh
scripts/agency-bench run --agents 6 --rep 1 --set api.tokens_per_window=<verified TPM>
```

A JSON file works too: `--config my-overrides.json`.

### The nine runs

Run one at a time, in this order (the plan's order):

```sh
# repetition 1
scripts/agency-bench run --agents 6  --rep 1
scripts/agency-bench run --agents 12 --rep 1
scripts/agency-bench run --agents 24 --rep 1
# repetition 2
scripts/agency-bench run --agents 12 --rep 2
scripts/agency-bench run --agents 24 --rep 2
scripts/agency-bench run --agents 6  --rep 2
# repetition 3
scripts/agency-bench run --agents 24 --rep 3
scripts/agency-bench run --agents 6  --rep 3
scripts/agency-bench run --agents 12 --rep 3
```

The whole matrix in one command, only when that is what you intend:

```sh
scripts/agency-bench matrix --execute
```

`matrix` without `--execute` prints the nine commands and starts nothing. With
`--execute` it runs them in plan order and stops at the first run that exits
with anything other than 0 or 5. It never retries.

### During and after a run

* Progress: `runs/<label>/attempt-NN/status.json` and `events.jsonl`.
* Graceful stop: `touch artifacts/terminalbench-contention/STOP`, or send the
  runner SIGINT/SIGTERM. Agents are stopped, collected and graded; unstarted
  tasks are recorded as `not_started`; exit code 4 or 130. Remove `STOP` before
  the next run.
* A re-run of the same condition and repetition creates `attempt-02`; earlier
  attempts are kept and listed by `report`. Decide explicitly which attempt
  counts.
* Grade later: `run ... --no-grade`, then `scripts/agency-bench grade --run LSB-6-r1`.
* After a crash or kill: `scripts/agency-bench cleanup` removes only this
  benchmark's labelled containers and units and restores session CPU placement.
* Wall time: a run is bounded by `ceil(24 / agents) × (task + setup timeout) + 600 s`:
  about 2 h 50 min for LSB-6, 1 h 30 min for LSB-12, 50 min for LSB-24, plus
  grading (two verifiers at a time, after the measured interval).

## Artifacts

```
artifacts/terminalbench-contention/
  manifests/     benchmark.json  tasks.json  environment.json  plan.json
  env/           pinned tools, dataset copy, instructions, prepared.json
  validation/    preflight.json  validation-report.md  token-calibration.json
                 unit-tests/  smoke-tests/  oracle/
  runs/<condition>-r<rep>/attempt-NN/
    config.json            exact configuration and its fingerprint
    plan.json              this run's seed and launch order
    metadata.json          git, versions, digests, topology, placement, limits,
                           API policy, cache strategy, host state at start
    status.json  events.jsonl  run.json
    summary.json           counts, makespan, per-task rows, API, resources,
                           cross-resource metrics, isolation, exit code
    agents/NN-<task>/      job.json task.json isolation.json execution.log
                           api-requests.jsonl container-series.jsonl
                           counters-start.json counters-end.json
                           logs/ (Agency data logger)  profile/ (agprof)
    traces/                run.trace.json trajectories.jsonl timeline.jsonl
                           tool-calls.jsonl agent-series.jsonl
                           cross-resource.json index.json
    resource-metrics/      samples.jsonl capabilities.json summary.json
    api-metrics/           ledger/{ledger.jsonl,state.json} verification.json summary.json
    grading/               source/ (collected records)  results/ (Harbor trials)
                           verdicts.json
    runtime-source/        the Agency package the sandboxes imported
  reports/       runs.json  matrix-progress.json
```

* **Results:** `summary.json`. `counts` separates completed, censored,
  not-started, infrastructure failures and grading failures. `denominators`
  states what each rate divides by. `rewards.mean_reward_corpus` counts
  ungraded tasks as 0; `mean_reward_graded` does not.
* **Traces:** open `traces/run.trace.json` in Perfetto (ui.perfetto.dev) for one
  track per agent plus counters for pool CPU, pressure, API budget and active
  agents. Each agent's detailed Agency trace is `agents/*/profile/agprof.trace.json`;
  `traces/index.json` maps them and gives the common clock origin.
  `trajectories.jsonl` has one row per state interval with its evidence source.
* **Profiling data:** `resource-metrics/samples.jsonl` (time series),
  `traces/timeline.jsonl` (per interval: utilisation, pressure, agents per
  state), `traces/agent-series.jsonl` and `agents/*/container-series.jsonl`
  (per agent).
* **Grading:** `grading/verdicts.json`; Harbor's own trial directory, with
  `verifier/reward.txt` and the test output, under `grading/results/`.
* **Time:** all timestamps are `CLOCK_MONOTONIC` nanoseconds, shared by every
  process on the host. `metadata.json` → `clock` maps them to wall time.
* **Secrets:** only the names of the credential environment variables are
  stored. Request and response bodies, URLs and headers are not written by the
  benchmark's own telemetry. Agency's data logger does record prompts and model
  output in `agents/*/logs/`, as it always does.

## Cache control

Recorded per run in `metadata.json` → `cache_control`.

| Cache | Control | Fresh per |
|---|---|---|
| Codex session, Agency agent state | new agent, container, log and profile directories; no session resume | task |
| API admission ledger | new directory; creation fails if not empty | run |
| Build outputs, pip/npm/cargo caches | only in the container's writable layer; all shared mounts are read-only | task |
| Writable filesystem | new overlay layer from a digest-pinned image; image ID checked before start | task |
| Image layers | pulled once in `prepare`; read-only, shared | never (immutable input) |
| Host page cache | **not controlled** | — |
| Provider and network caches | **not controlled**; reported cached tokens are recorded | — |

A new container does not mean a cold page cache. The runner does not drop
caches, which needs root and affects the whole host. Rotating the condition
order keeps cache warmth from lining up with one condition, and page-cache size
at run start is recorded. For cold starts, reboot or run
`sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` before every run alike.

## Validation

```sh
.venv/bin/python -m pytest benchmarks/terminalbench_contention/tests -q -p no:cacheprovider
AGENCY_TB_E2E=1 .venv/bin/python -m pytest benchmarks/terminalbench_contention/tests/test_harbor_integration.py -q -p no:cacheprovider
scripts/agency-bench smoke --agents 2 --tasks 4      # real containers, mock model
scripts/agency-bench oracle                          # reference solutions through grading
```

None of these makes a paid model request. `VALIDATION.md` has the results.

## Limitations

See `VALIDATION.md` for the full list with evidence. The ones that matter most
when reading results:

1. Gateway rate-limit rules and limits are unverified; the 1.2 M TPM budget is
   a local default.
2. Luna's tokenizer is assumed to be `o200k_base`, supported by recorded usage,
   not by documentation.
3. The task time cap (30 min) is far below Terminal-Bench's 8 h; many tasks may
   be censored, and more so at higher concurrency.
4. Host page cache and provider caches are not controlled.
5. Agents have network access by default; package downloads add outside
   variation.
6. Root-owned services, kernel threads and interrupts are not confined to the
   reserved CPUs.
7. Task wall time includes Agency's harness start-up (about 4 s) and its
   end-of-task container checkpoint; both are separate states in the
   trajectories.
8. Per-agent I/O and per-thread scheduling numbers are lower bounds where
   marked.
