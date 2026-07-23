# agency.profiler

Profiling subsystem for the framework. The span annotations live permanently in
the framework's hot paths (`agskill`, `agllm`, `agtool`, `agsandbox`, `agmap`,
`agdata`, `agsync`, `agent`) and are **no-ops until a session is active** — the
off-path cost is one global check, and torch is never imported unless profiling
is turned on.

## Installation

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

Or profile an **unmodified** application:

```bash
AGENCY_PROFILE=1 [AGENCY_PROFILE_DIR=path] python app.py
```

View traces with `tensorboard --logdir <runs dir>` (PYTORCH_PROFILER tab →
Views → Trace; needs `tensorboard` + `torch-tb-profiler`) or drag the
`.pt.trace.json` into <https://ui.perfetto.dev>. After a session,
`prof.key_averages().table(sort_by="cpu_time_total")` prints a per-span summary.

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

- Spans record wall time only — no CPU-vs-wait split within a span
  (needs a `thread_time_ns`/schedstat session backend).
- Container/daemon CPU, tool-worker subprocesses, and the SSE drain thread are
  outside the profiled process.
- No resource counters (energy, GPU, IO) yet — sampler + counter-track
  injection planned.
- Durations vary with live model load; benchmark-grade numbers need the
  mock endpoint.
