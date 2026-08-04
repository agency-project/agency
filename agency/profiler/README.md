# agency.profiler

Profiling subsystem for the framework. The span annotations live permanently in
the framework's hot paths (`agskill`, `agllm`, `agtool`, `agsandbox`, `agmap`,
`agdata`, `agsync`, `agent`) and are **no-ops until a session is active** — the
off-path cost is one global check, and torch is never imported unless profiling
is turned on.

## Installation

Profiling requires Linux. Other operating systems are currently unsupported
because resource isolation and accounting depend on cgroups v2 and Linux
kernel interfaces under `/proc`.

Install Agency with the optional profiling dependencies:

```bash
uv pip install -e ".[profiler]"
```

The extra installs `torch` for trace collection and `nvidia-ml-py` (imported
as `pynvml`) for NVIDIA GPU sampling. These dependencies are intentionally not
part of the default install because PyTorch is large and profiling is optional.
The profiler runs in the host Python environment, so the copy of `torch`
included in Agency's sandbox image does not satisfy this requirement.

## Usage

```python
from agency import agprof

with agprof.session(run_dir / "tb_trace"):   # owns the torch.profiler lifecycle
    team.run()
```

Or profile an **unmodified, non-Web-UI application** for its full process
lifetime:

```bash
AGENCY_PROFILE=1 AGENCY_PROFILE_SCOPE=process [AGENCY_PROFILE_DIR=path] python app.py
```

When `AGENCY_PROFILE` is enabled, Agency validates the operating system before
the workload or profiler starts. On Linux it relaunches the complete command in
a dedicated transient systemd cgroup. The benchmark harness, its child
processes, and Agency-managed Docker containers are placed beneath the same
slice, so unrelated machine processes are excluded. Creating the transient
scope requires cgroup v2, `systemd-run`, `setpriv`, and non-interactive `sudo`
permission for `systemd-run`; the workload itself is immediately dropped back
to the invoking user and supplementary groups.

On a non-Linux system the command fails immediately with a Linux-only error,
and no trace directory or profiler artifacts are created.

Environment profiling defaults to `AGENCY_PROFILE_SCOPE=workload`. The Web UI
automatically opens that boundary immediately before the function passed to
`agwebui.run(...)` and closes it as soon as the function returns, excluding
dashboard startup and linger time. A non-Web-UI application can use the same
scope by wrapping its entry point in `with agprof.workload():`; because Agency
cannot infer an arbitrary application's workload boundary, an otherwise
unmodified non-Web-UI application must explicitly select
`AGENCY_PROFILE_SCOPE=process`. Unset or invalid scope values use `workload`.

View traces with `tensorboard --logdir <runs dir>` (PYTORCH_PROFILER tab →
Views → Trace; needs `tensorboard` + `torch-tb-profiler`) or drag the
`.pt.trace.json` into <https://ui.perfetto.dev>. After a session,
`prof.key_averages().table(sort_by="cpu_time_total")` prints a per-span summary.

Every completed session with an output directory also writes:

- `summary.json`: a versioned machine-readable document containing run
  outcomes/throughput, LLM and tool metrics, span latency distributions,
  per-process/workload/sandbox/GPU resource statistics, energy/totals,
  interrupted spans, and GPU lease statistics.
- `summary.md`: the same metrics as human-readable Markdown tables.

### Metrics collected

A profiling session collects:

- **End-to-end runs:** session duration; started, completed, succeeded, failed,
  and interrupted counts; completed and successful runs per second; and
  mean/min/p50/p95/p99/max latency.
- **Spans:** wall, on-CPU, run-queue, and blocked time; CPU percentage; outcome
  and error type; call counts; and mean/min/p50/p95/p99/max latency. This covers
  skill runs, LLM calls, tools, sandbox operations, synchronization, and custom
  application spans.
- **LLM calls:** calls, attempts, retries, outcomes, total wait, latency, time to
  first token (TTFT), input/output tokens, generation time, and output tokens
  per second.
- **Tools:** started, completed, succeeded, failed, and interrupted counts plus
  overall and per-tool latency distributions.
- **Whole workload:** average and peak sampled CPU utilization, cumulative CPU
  time, average and peak sampled memory, and disk bytes read and written.
- **Processes:** PID, command line, cgroup and owning sandbox; average and peak
  CPU utilization, cumulative CPU time, average and peak RSS/VMS, and disk bytes
  read and written.
- **Sandboxes:** average and peak CPU utilization, cumulative CPU time, average
  and peak sampled memory, disk bytes read/written, and network bytes
  received/transmitted.
- **NVIDIA GPUs:** device utilization, memory, and power averages/peaks; energy
  in joules; per-process GPU memory and utilization; apportioned per-process
  power estimates; and GPU lease counts and durations.
- **Sampling health:** configured/effective frequency, sampled duration, raw
  sample count, GPU availability, and spans still incomplete when profiling
  stopped.

Resource rows report sample count, mean, minimum, maximum, and last value.
Cumulative CPU, disk, and network counters are converted to utilization or
throughput while their non-negative deltas are also summed into CPU seconds or
MB totals. GPU power samples are trapezoidally integrated into joules. Memory
"peaks" are the maximum observed samples rather than kernel high-water marks.
GPU metrics require NVIDIA NVML. The report shows the effective sampling
frequency alongside the configured rate.

### Process resource tracks

The workload cgroup is recursively scanned on every sampler tick. Every PID
found in any descendant `cgroup.procs` file is sampled independently from
`/proc/<pid>/stat`, `/proc/<pid>/cmdline`, and `/proc/<pid>/io`. TensorBoard
receives a separate process group such as `python (PID 85418)` for each stable
`(PID, start time)` identity, with independent `cpu_percent`, `rss_mb`,
`vms_mb`, `io_read_mb_s`, and `io_write_mb_s` tracks. PID start time prevents
PID reuse from merging two different processes. Processes that exit between
cgroup discovery and `/proc` reads are skipped without failing the run.

The cgroup-wide counters remain available under the explicit
`workload_total` name; they are not labeled as a process. Agency-managed
container cgroups also keep their `sandbox:*` aggregate tracks.

If profiling stops while background work is live, open spans are listed under
`incomplete_spans` with `outcome: "interrupted"` and elapsed time at the stop
boundary; they are not misreported as completed latency samples.
`agprof.summary_metrics()` returns a copy of the JSON document for the most
recently completed session, including sessions started with `out_dir=None`.

Custom app-level phases use the same public API:

```python
with agprof.span("stage3:validate"):
    ...
```

## Span glossary

Two reading rules:

1. **Nesting = parenthood.** A span contains whatever opened inside it on the
   same thread; nobody declares hierarchy explicitly.
2. **Every span measures wall time on its thread.** A long span means "this
   took long", not "this burned CPU" — the `sync:`/`llm:` sub-labels exist to
   say *why* an interval was long. (The summary table's "Self CPU" columns are
   wall-in-span for user annotations; concurrent lanes sum past wall clock.)

### Lane roots (one per thread)

| label | meaning |
|---|---|
| `run{N}:{skill}:{agname}` | One complete skill run on its own daemon thread. `N` is process-wide start order (start order ≠ completion order under concurrency). Covers `_task` end-to-end: dependency wait through history prune. |
| `agmap:{fn}[{i}]` | One `agmap`-mapped function call on its own thread; `i` is submission order. Covers sync and `is_asynchronous=True` alike. |
| *(app root, e.g. `SWETeam.run`)* | Whatever the application wraps. On a team flow the main thread is mostly `agsync:join` — `agteam.run()` executes `_run()` on its own thread and returns a pending result. |

### Skill-run phases (children of `run{N}`, lifecycle order)

| label | meaning |
|---|---|
| `resolve` | Waiting for a predecessor run's context and lazy input futures — sync-idle, not work. |
| `sandbox:provision` | Creating the `agSandbox` object for an agent that lacked one (the container itself starts later, lazily). |
| `input:prepare` | Writing agtype input fields / oversized strings into the sandbox. |
| `turn{i}` | One ReAct iteration: one LLM call + its tool dispatch. `i` restarts at 0 per run. |
| `proc_wait` | Polling for background processes the agent left running in its sandbox (5 s polls, can run minutes). |
| `teardown:commit` | The end-of-run `stop(commit=True)` — final container checkpoint. |
| `prune` | Trimming tool outputs from history before handing context to the next chained run. |

### Inside a turn

| label | meaning |
|---|---|
| `llm:{skill}` | The full LLM call from the ReAct loop's perspective, retries included. |
| `llm:sync` | Waiting to acquire the LLM-call semaphore (256 slots) — queueing behind the framework's own throttle, **not** model time. |
| `llm:attempt[n]` | One streaming attempt against the endpoint, semaphore excluded — the honest "model wait + stream decode" number. `n` > 0 means retries happened. |
| `llm:retry_backoff` | Sleeping between failed attempts — endpoint-caused, kept separate so retry cost is its own line. |
| `llm:compact` | A history-compaction summarization call (separate non-streaming path). |
| `tool_dispatch:{skill}` | Executing one turn's batch of tool calls — strictly sequential. |
| `tool:{name}` | One tool invocation as seen from the skill thread. For subprocess tools this is the full round-trip (pool wait + worker execution + transfer), **not** the tool body — the body runs in a worker process the profiler can't see. |

### Sandbox operations

| label | meaning |
|---|---|
| `sandbox:create` | `agSandbox` constructor: image/mount resolution + backend setup. |
| `runtime:detect` | Once per process: probing `docker info` / `podman info` to pick a runtime (~0.5 s). |
| `sandbox:start` | The lazy `docker/podman run` path — only when a container actually starts, never the reuse check. |
| `sandbox:exec` | Running a command in the container (one CLI round-trip). |
| `sandbox:read_file` / `sandbox:write_file` | File I/O into/out of the container — each a full CLI round-trip (~200–350 ms regardless of content size). |
| `sandbox:commit` | Snapshotting container state to an image. |
| `sandbox:stop` | Stop + optional commit — fires after every sandboxed tool call (the checkpoint-per-tool-call design). |
| `sandbox:fork` | Cloning a checkpoint image into a new private sandbox. |
| `sandbox:destroy` | Removing the container and its images. |

### Synchronization (framework machinery — all sync-idle)

| label | meaning |
|---|---|
| `sync:container` | Acquiring the global docker/podman CLI semaphore (16 slots) — fires before every container command. Long spans here = CLI contention under fan-out. |
| `sync:result_wait` | A thread blocked reading a still-pending result (`agdata` field access). Only emitted when the access actually blocks — resolved reads stay silent. This is how cross-thread joins appear. |
| `agsync:join` | An explicit `agsync(...)` barrier — waiting for all in-flight agents/teams/tasks. |

### Lifecycle

| label | meaning |
|---|---|
| `agent:create` | `agent()` construction: name allocation, terminal, logging, config. |

## Known limits (where torch.profiler ends and agprof begins)

- Profiling is Linux-only. CPU/run-queue timing uses
  `/proc/.../schedstat`, and process resource accounting uses cgroups v2.
- Sandbox metrics require cgroup v2 paths readable by the host process.
- Linux may restrict `/proc/<pid>/io` for processes owned by another UID. Such
  processes still receive CPU, RSS, and VMS tracks; their exact container-level
  I/O remains available through the corresponding `sandbox:*` cgroup track.
- GPU metrics require NVIDIA NVML. GPU work is attributed with device sampling,
  per-process sampling, and explicit lease intervals rather than host-thread
  timing.
- Token counts depend on the backend returning streaming usage. TTFT is the
  first non-empty content, reasoning, or tool-call delta.
- Resource sampling is discrete. Energy integration and counter totals cover
  the sampled interval, which is reported separately from session duration.
- Durations vary with live model load; benchmark-grade numbers need the
  mock endpoint.
