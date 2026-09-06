# Agency concurrent execution experiment

This benchmark uses `agent(...).run(...)` and retains every `Invocation` before
waiting. It does not patch the scheduler, engine, sandbox, or harness. The default
concurrency list is N=1,2. Every workload run requires `--execute` and a reviewed
`--layout`; omitting `--execute` exits before contacting the runtime.

## Architecture and workload

Reviewed against `e558734` (pulled by clean fast-forward from `ab57d21`):

- `agency/orchestrator/orchestrator.py`: one event loop admits work and records
  `request_*` events and `scheduler_state` snapshots. Each dispatched invocation
  gets a fresh `AgentEngine`. The worker executor has no small implicit Python
  thread-pool ceiling; `max_concurrent_engines=None` leaves admission unlimited.
- `agency/orchestrator/scheduler.py`: predecessor futures enforce same-agent
  ordering; ready work for unrelated agents can skip a busy/blocked agent.
- `agency/engine/engine.py` and its clients/host services: a per-sandbox lock spans
  the transaction, host RPC services connect to the real in-container harness,
  success commits, and failure calls `rm_container()`.
- `agency/sandbox/container.py`: commit retains a checkpoint image; discarding
  the live container preserves that image for the next start. This restores
  container filesystem state, not arbitrary host mounts or external side effects.
- `examples/parallel_exec.py` demonstrates submit-first fanout. Existing
  `tests/test_agorchestrator.py`, submission/order/lifecycle tests and
  `test_orchestrator_transaction_fence.py` test important semantics, but their
  fake engines/sandboxes cannot establish actual sandbox concurrency.
- `agency/llm/mock.py` replays finalized `llm_block` exchanges positionally.
  Replay state is per backend instance, and a fresh invocation builds new host
  services. Both streaming and nonstreaming routes support these fixtures.

The benchmark creates **synthetic**, explicitly labeled replay databases. An
instant response requests the native `bash` tool, which starts a Python child
inside `/workspace` in the real sandbox. That child increments private filesystem
state, sleeps for a configured duration, and emits a JSON receipt with timestamps,
PID, and before/after state. A second response returns an `agfile` receipt. Timing
is in the child, not in the replay backend: nonstreaming mock dispatch does not
apply replay delays. No remote model, credentials, inference server, shared
response queue, or host-side sleeping tool is used.

The receipts are recovered from existing `tool_result` telemetry, including when
the invocation fails. Synthetic final responses alone never count as evidence.
This measures process execution and control-plane concurrency for I/O-wait-like
workloads. It does not measure reasoning quality or CPU-bound/model throughput.

## Run in stages

Use a working Agency sandbox runtime and the repository's documented image
(`images/build.sh`); the benchmark will not install/start a daemon or build/pull
an image implicitly. Run each command in a fresh process and output directory.

```sh
# Small payload/evidence checks only: no Agency concurrency claim.
.venv/bin/python -m pytest -q benchmarks/agent_stress/test_evidence.py benchmarks/agent_stress/test_placement.py

# Preparation only: read host state, then calculate placement without starting Agency.
python3 benchmarks/agent_stress/inspect_host.py > inventory.json
python3 benchmarks/agent_stress/placement.py --inventory inventory.json --out cpu_layout.json

# FUTURE ONLY: smallest real case first, then two agents and correctness scenarios.
.venv/bin/python benchmarks/agent_stress/run.py --execute --layout cpu_layout.json --ns 1 --warmups 0 --repeats 1 --out runs/stress-n1
.venv/bin/python benchmarks/agent_stress/run.py --execute --layout cpu_layout.json --ns 1 2 --warmups 1 --repeats 3 --scenarios --rollback --out runs/stress-small

# LATER, on the experiment machine; not run as part of local validation.
AGENCY_PROFILE=1 AGENCY_PROFILE_DIR=./runs/stress-profile \
  .venv/bin/python benchmarks/agent_stress/run.py \
  --execute --layout cpu_layout.json --ns 1 2 4 8 12 --warmups 1 --repeats 3 --seconds 5 --slow-seconds 30 \
  --scenarios --rollback --out runs/stress-sweep
```

For Podman, pass `--backend podman`. The existing profiler requires Linux,
cgroups v2, profiler dependencies, and the documented systemd permissions; see
`examples/profiler_native_example.py`. Keep the actual image ID, runtime info,
CPU/memory limits, revision, and profiler artifacts with the results. Default
per-sandbox limits are 1 CPU and 512 MB, configurable via `--cpus`/`--memory`.
These are limits, not proof of exclusive resource reservations.

## Measurements and acceptance

`results.json`, `metrics.csv`, and `invocations.csv` contain creation, submission,
and completion durations, successful completions/second, success/failure states,
request start/end timestamps, child intervals, and state receipts. The existing
SQLite databases and exported JSON events retain scheduler running counts,
blocked/ready transitions, LLM exchanges, tool calls, and lifecycle evidence.
`timeline.trace.json` displays engine and child lanes in a Chrome trace viewer;
the existing profiler additionally supplies detailed engine/start/commit/discard
spans and resource samples. Driver CPU and process-lifetime peak RSS are recorded;
they exclude the sandbox and daemon and are not total machine utilization.

Creation finishes before submission timing starts. Every submission finishes
before any invocation handle is inspected or waited on. Total time ends after
all handles settle, before evidence extraction and cleanup. It includes submission,
startup, child work, checkpointing, and teardown. Repeated phases reuse committed
agents, but engine hibernation means they are **not guaranteed hot-container runs**.
A failed sweep phase stops the sweep. Missing receipts fail validation. Timeouts
are failures, not successful completions or measured capacity limits.

- **Concurrency:** engine peak uses admission-to-terminal event intervals; it can
  include queued worker startup and teardown. Child peak measures actual overlapping
  timed child processes. Largest demonstrated N requires all N child intervals
  to overlap and every invocation to succeed. A single run is a demonstration,
  not a confidence interval; retain repeated runs and report variation.
- **Cross-blocking:** the short agent must reach terminal success while the slow
  child's interval is still active, not merely while its request is admitted.
  This is a strict acceptance condition; startup skew may make it fail even if
  the scheduler is correct. Inspect the trace before interpreting failure.
- **Ordering:** three submissions to the same agent must have nonoverlapping,
  ordered engine intervals and committed counters 1,2,3. Another agent must have
  overlapping child execution. Submission sequence/ordering IDs are retained.
- **Rollback:** establish counter=1 on both agents. The target writes dirty state
  and counter=2, then exhausts `max_steps=1` after its tool executes. Require a
  dirty-state receipt and the expected step-limit failure. An independent child
  must overlap and succeed. A subsequent public invocation must read target
  counter=1/clean and independent counter=2/clean. A tool error alone is not the
  deliberate invocation failure, and an absent mutation receipt never passes.

`largest_demonstrated_n` supports only the concurrency clause. Support the full
resume sentence only when ordering, cross-blocking, and rollback also pass on
the same machine/revision. Never substitute the requested maximum N for observed
child overlap. No agent concurrency was demonstrated by the checked-in local run.

## Confounders and attribution protocol

1. Compare submit-to-`request_started`, `engine:execute`, `sandbox:start`,
   `sync:container`, tool/child durations, and `teardown:commit`/`teardown:discard`.
   A rising engine-admission count alone is not proof of running child processes.
2. Container runtime calls have a shared semaphore and backend lifecycle costs;
   image layers/commit/squash, startup limits, and daemon load can limit scaling.
   Separate first and subsequent invocation phases. Compare short and longer
   sleep workloads at the same N to expose fixed lifecycle overhead.
3. Inspect profiler host, daemon and container CPU, runqueue, memory, I/O and
   cgroup sampling coverage. Verify VM allocation separately from host CPU count.
   A saturated VM, OOM, or throttled sandbox is not a scheduler capacity limit.
4. Replay removes LLM service saturation by construction. A later real-model run
   is a separate experiment: record LLM queue/TTFT and service concurrency limits.
5. Logging/profiling also consumes CPU and serializes some writes. Compare matched
   profiled/unprofiled runs if it becomes material; do not silently disable logs
   for a resume number. This driver collects evidence after completion, so it
   does not insert polling, printing, or waits into the submission loop.
6. Epoch clocks must be aligned between host and sandbox (native Linux preferred).
   Children on one kernel share a clock; Docker Desktop VM/remote daemon clock
   offsets can invalidate host-terminal versus child-time comparisons. Inspect
   child monotonic elapsed values and host tool spans before accepting overlap.

Conclude Agency itself is the bottleneck only with increasing scheduler/engine
control time **and** evidence of resource/runtime headroom. If profiling coverage
is unavailable or backend startup dominates, report the cause as unresolved or
backend-limited rather than claim a maximum supported N. The sleeping workload
is not a CPU-saturation test, and no asymptotic scaling limit is inferred from N<=32.


## EC2 preparation and affinity

No benchmark, N=1/N=2 smoke workload, or live affinity probe was run during EC2
preparation. `results/ec2_preparation/` contains the read-only machine audit and
proposed map. Preparation tests use mocked container commands and compile the
child program without executing it.

`placement.py` reads actual CPU/core/socket/NUMA IDs from the audit. It reserves
four complete exposed physical cores: two as OS/runtime headroom and two for the
benchmark driver, scheduler, engine host threads, RPC servers, and profiler.
Each sandbox gets one logical CPU on a different remaining physical core. Worker
SMT siblings are unused by this benchmark. A changed topology or an N larger
than the separate-core budget is rejected. Placement is stable by agent slot,
including warmups and repeated runs.

On the audited g6.8xlarge: OS/runtime headroom is `0,1,16,17`; host Agency control
and monitoring use `2,3,18,19`; worker slots 0–11 use CPUs `4–15`, respectively;
all sandbox memory is bound to NUMA node `0`. N=1/2/4/8/12 use prefixes of those
worker slots. N=16/32 require a separately labeled shared-core/SMT experiment;
they cannot be represented as independent physical worker cores in this layout.

`AgentEngine` is a host thread, not a separate host process. The driver sets its
affinity **before importing Agency**, so scheduler/worker/RPC/profiler threads
inherit the host control CPU allocation. The external Docker daemon creates the
container using `--cpuset-cpus` and `--cpuset-mems`. That cgroup constrains the
harness and its child processes, including later `exec` calls, rather than merely
pinning the process that sends a Docker command. CPU/memory/cpuset flags are now
applied on checkpoint-based recreation as well as first creation. Hibernation
retains the existing container HostConfig; changing placement requires a new
container, so this benchmark never changes an agent's placement mid-lifecycle.

Successful receipts must report the expected process affinity and effective
cgroup CPU and memory sets. Post-completion container inspection retains
HostConfig for cross-checking. Cleanup requests all agent destructions before
waiting and verifies their exact container names are gone. These mechanisms
are implemented and covered by preparation tests; **live enforcement, overlap,
rollback cleanup, and tracing overhead remain unvalidated**.

One warmup and three measured repetitions are the defaults. Warmup rows are
explicitly labeled and excluded from largest-demonstrated-N calculations. The
first measured invocation inherits warmup checkpoint state; this is not a fresh
container baseline. Use `--warmups 0` for a separately labeled cold run.

The two-core host control budget can itself constrain scaling. A result is a
capacity demonstration under that explicit budget, not an intrinsic maximum of
Agency. The OS, Docker daemon, IRQs, and other users are **not** repinned, and
cpusets do not exclude them from worker CPUs. Exclusive use of the machine or
administrator-managed isolation is still needed for strong performance claims.
No existing sessions or containers were stopped during preparation.
