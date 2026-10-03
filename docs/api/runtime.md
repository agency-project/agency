# Advanced runtime and observability

These APIs expose process state and infrastructure. Most applications submit through Agent and inspect a snapshot; direct scheduler/resource/context mutation requires coordination with active work.

```python
from agency import (
    agcontext, GlobalAgentOrchestrator, ExecutionScheduler, OrchestratorSnapshot,
    get_orchestrator, agResourcePool, agDataLogger, agprof,
)
from agency.orchestrator import peek_orchestrator
from agency.observability.agwebui import agwebui
```

## Process orchestrator and scheduler

`get_orchestrator(agconfig=None, *, default_db_path=None)` lazily creates a process singleton with logging, hardware resource detection, execution workers and scheduler thread. The first call fixes initial config/path; subsequent calls ignore arguments. Construction/config replacement retain supplied config. `peek_orchestrator()` returns the existing instance or None without initialization. `GlobalAgentOrchestrator(...)` is the same constructor contract but creates an independent instance whose lifecycle you own.

| API | Behavior |
| --- | --- |
| `snapshot()` | Non-workload-blocking immutable OrchestratorSnapshot with state, engine ceiling, ready/blocked/running counts, submitted/completed/failed totals, agents, db_path, persistence_error, telemetry_error and queue_depth. Nested agents dict is still mutable; treat as a snapshot. |
| `flush(timeout_s=None)` | Blocks flushing SQLite; returns None. **timeout_s is currently ignored.** |
| `shutdown(wait=True, timeout_s=None)` | Stops admission and drains accepted work; returns None. With wait false returns before drain. Wait timeout can raise TimeoutError without undoing shutdown; scheduler callbacks force nonwaiting to avoid joining their own thread. Worker teardown can outlast the supplied timeout; not a hard overall deadline. Singleton remains stopped; later get_orchestrator does not restart it. |
| `change_config(config)` | Retains config, validates engine ceiling, cascades logger/resource settings; returns None. Doesn't resize the original worker executor. |
| `submit(ag, skill, skill_input, max_steps=None)` | Advanced equivalent of Agent.run, pending agdata with scheduler acknowledgement. Outer input must be agdata, positive step budget if supplied; stopped process raises RuntimeError. |
| `submit_context_message(ag, message)` | Advanced Agent.queue_message operation; returns None. |
| `redirect_request(ag, request_id, message)` | Returns bool for request delivery, unlike Agent.redirect this low-level call has no queued-message fallback. |
| `cancel_request(ag, future)` | Marks an owned request and returns its active engine or None; Agent.cancel performs the follow-up best-effort engine cancellation. Use original Agent handle API in applications. |

`ExecutionScheduler(orchestrator)` owns one scheduling-cycle protocol. `execute()` discovers/promotes dependencies then calls its assignable `.schedule` callback, default `default_schedule()`; both return None. Admission is FIFO among ready work within capacity and same-agent predecessor constraints. `materialize_dependencies(value)` blocks resolving nested handles and returns the reconstructed value; don't call it on unfinished dependencies from a scheduler callback. Private request-record methods are excluded from application reference.

```python
from agency import get_orchestrator
from agency.configs.agconfig import agconfig, orchestratorconfig

# Call before constructing any agents if setting process admission is required.
orchestrator = get_orchestrator(agconfig(orchestratorconfig(max_concurrent_engines=2)))
state = orchestrator.snapshot()
print(state.running_count, state.blocked_count)
```

## Resource pool

`agResourcePool(...)` detects omitted GPUs/CPU count/memory from the host, clones config and optionally creates GPU marker processes with `mark_gpus=True`. Explicit totals/IDs override detection. Access the shared pool with `get_orchestrator().agresource_pool`; replacing it during active leases is unsafe. Initial orchestrator construction currently omits its supplied resources config; see [known issues](contracts.md).

| Method | Contract |
| --- | --- |
| `acquire_gpus(sandbox, count, timeout=None, is_cancelled=None, poll_interval=0.5)` | Blocks until grant; returns IDs and attaches them to sandbox. Nonpositive count returns []; count above capacity raises ValueError. Smallest queued count first, FIFO ties. TimeoutError on waiting timeout; GpuWaitCancelled when callback becomes true. |
| `release_gpus(sandbox, gpu_ids)` | Releases leases and wakes waiters, returns None. Owner must avoid releasing another sandbox's resources. |
| `acquire_cpu_mem(sandbox, cpus=None, memory_mb=None)` | Sets/bookkeeps limits and returns None; does not wait for a global CPU/memory quota. Floors actual caps at 1 CPU/1024 MB. |
| `release_cpu_mem(sandbox, cpu=False, memory=False)` | Resets selected limits to idle values, returns None. Neither flag means no selected release. |
| `get_config_copy()` / `change_config(config)` | Clone return / clone replacement returning None. Totals are not redetected by changing config. |

`GpuWaitCancelled` is a RuntimeError subclass from `agency.orchestrator.agresources`. Pool totals and `cpus_acquired`, `memory_acquired_mb` are inspectable bookkeeping; resource tools reject requests above host totals, but CPU/memory reservations are not a fair blocking allocator like GPU leases.

## Conversation context

`agcontext(recent_transcript=None, harness_sessions=None, retained_messages=None, harness_message_cursors=None, _future=None)` holds ordered agent state. Omitted mappings/lists are fresh; supplied ones are retained. `_future` belongs to infrastructure. Prefer `agent.history` and `queue_message` for application changes.

`is_pending()` reports presence of a context future (even if done). `resolve_prev_dependencies()` blocks and merges predecessor state, returns None. `get_resolved_transcript()` blocks and returns a shallow list snapshot; `copy()` blocks and returns a deep independent context. `set_transcript(list)` replaces with a list copy without waiting itself.

`append_retained_message(dict)` deep-copies and validates sequence (positive increasing int, not bool), type `message`, role user/system and string content; returns sequence, invalid data raises ValueError. `pending_retained_messages(harness)` returns a deep copy beyond that harness's cursor. `advance_retained_cursor(harness, sequence)` returns None, rejects backwards movement with ValueError. Coordinate these operations with submitted context chains.

## SQLite data logger

`agDataLogger(config, *, default_name=None, default_object=None)` requires config.data_logger.db_path on first use (`ValueError` if missing), retains config, and does not open until `start()`. `start()` creates directories/SQLite schema and enables WAL, returns None. Own standalone logger's `stop()` in finally; don't stop an agent/orchestrator's active logger.

`db_path` returns str. `change_config(config)` retains it, carrying an omitted db_path from the existing config (mutates supplied config); it doesn't move an open connection. `flush()` blocks committing pending rows; `stop()` atomically flushes/closes; both return None and persistence failures propagate.

`record_event(type, payload, ...)` appends an event, with optional snapshot/terminal/flush flags. `record_stream_delta(type, payload, ...)` appends streamed content. `record_llm_exchange(call_label, *, exchange_type, prompt_chain, response_chain, ...)` stores ordered exchange blocks. `record_span(span_name, start_ts, end_ts, attributes, ...)` stores duration/resources/parent attribution. All return None; respect source signatures for keyword defaults and JSON-safe payloads. Batching controls decide when recording touches disk; flush=True commits immediately. `read_profile_records(profile_session_id)` returns a list of profile tuples.

```python
from agency import agDataLogger
from agency.configs.agconfig import agconfig, dataloggerconfig

logger = agDataLogger(agconfig(dataloggerconfig(db_path="agency_runs/app.sqlite3")))
logger.start()
try:
    logger.record_event("application", {"state": "ready"}, flush=True)
finally:
    logger.stop()
```

Reader helpers from `agency.observability.agdatalogger`: `read_llm_exchanges(db_path, exchange_type=None)` opens/closes a read-only SQLite connection and returns reconstructed exchange dicts; `reconstruct_llm_exchanges(connection, exchange_type=None)` uses the caller-owned connection; `resolve_global_db_path(log_dir)` returns `Path(log_dir) / "global_data.sqlite3"` without I/O. SQL/file errors propagate.

## Profiler callables

`from agency import agprof` and `from agency.observability.profiler import agprof` refer to the same module. Direct function imports such as `from agency.observability.profiler import session, span, workload, enabled` are also supported aliases. [Profiler startup](../guides/profiling.md) covers launch paths; trace interpretation is outside this API reference.

| Callable | Contract |
| --- | --- |
| `start(out_dir=None, ...)` | Starts the one process session, returns a session object; Linux /proc/cgroup v2 required. Existing active session or missing support raises RuntimeError. Optional output directory writes trace and summary artifacts; None retains data in memory. Caller must stop. |
| `session(out_dir=None, ...)` | Context manager yields session; calls stop in finally. Same options/requirements as start. |
| `stop()` | Stops sampler/recording, writes trace/summary when configured; returns the stopped session object when active, None when inactive. Can block on persistence. |
| `workload()` | Context manager owning a session only when environment profiling enabled, scope workload, and no active session. Failed RuntimeError startup warns and work continues unprofiled; stops only its own session. |
| `enabled()` / `profile_scope()` | bool active / configured scope str (`process` only if explicitly set, else `workload`). |
| `span(name, parent_context=None)` / `annotate(**metadata)` | Span context manager (inactive is no-op) / annotate current span returning None. Explicit parent context can join traces. |
| `thread_name(name)` | Records current thread label, returns None. |
| `spawn_traced(fn, *args, daemon=True, **kwargs)` | Returns an **unstarted** Thread carrying trace/execution context; caller starts and joins it. |
| `execution_context(metadata, parent_context=None)` / `bind_execution(fn, metadata, parent_context=None)` | Context manager / wrapped callable propagating attribution and parent context; execution/exception behavior of fn preserved. |
| `current_span_context()` / `current_span_attributes()` | Current context or None / copied metadata dict. |
| `next_index(key="run")` | Per-key monotonic int for trace labels. |
| `summary_metrics()` / `profile_records()` | Last completed run metrics dict or None / list of record tuples. |
| `summary_table(sort_by="wall_ms", row_limit=30)` | Markdown table str from last span summary. No model calls. |

Start/session default options: all_threads=True and worker_name=None retained for compatibility, sample_hz=10.0 (0 disables sampler), sample_gpu=True (optional GPU metrics), auto_functions=True, auto_include_dependencies=False, auto_include=None, auto_exclude=None, auto_min_duration_ms=1.0, auto_max_depth=32, auto_max_events=250000. Automatic Python intervals use sys.monitoring; filters/limits bound capture.

Environment profiling is enabled when AGENCY_PROFILE is unset, or explicitly `1`/`true`; other values disable it. AGENCY_PROFILE_SCOPE defaults workload, process opts into process-lifetime profiling. AGENCY_PROFILE_DIR chooses artifacts; default under the run root's profiler directory. Process-mode initialization can perform host cgroup setup/re-execution: docs tooling therefore never imports Agency.

## Web UI entry point

`agwebui.run(fn, *args, run_dir=None, port=7860, linger=True, **kwargs) -> None` starts its server subprocess and viewer assets, then calls fn synchronously in the caller's thread inside a workload profiling boundary. The fn return value is discarded. run_dir/port/linger are consumed by the UI; remaining args/kwargs go to fn. run_dir selects saved artifacts; default is generated under the run root. The server binds 0.0.0.0. An occupied port raises RuntimeError. Viewer build/download and service-start failures propagate before fn runs.

With linger=True it remains until Ctrl+C; false exits after fn. Ordinary fn Exceptions are logged/printed as failed workload and are **not reraised**; BaseException follows Python cleanup propagation. Don't use the return value or absence of an exception as a success signal. The execution profiler starts without downloading or building viewer assets. A workload should settle its background results before returning so profiling ends at the intended boundary.

```python
from agency.observability.agwebui import agwebui

def workload():
    print("Replace this with a joined application workload")

# Creates server/artifacts and may build viewer assets.
# agwebui.run(workload, linger=False)
```

## Source signatures

[Source: agcontext.py](../../agency/agcontext.py)

::: agency.agcontext.agcontext
    options:
      members: ["__init__", "is_pending", "resolve_prev_dependencies", "get_resolved_transcript", "set_transcript", "append_retained_message", "pending_retained_messages", "advance_retained_cursor", "copy"]

[Source: orchestrator.py](../../agency/orchestrator/orchestrator.py)

::: agency.orchestrator.orchestrator.GlobalAgentOrchestrator
    options:
      members: ["__init__", "change_config", "submit", "submit_context_message", "redirect_request", "cancel_request", "snapshot", "flush", "shutdown"]

[Source: orchestrator.py](../../agency/orchestrator/orchestrator.py)

::: agency.orchestrator.orchestrator.OrchestratorSnapshot
    options:
      members: []

[Source: orchestrator.py](../../agency/orchestrator/orchestrator.py)

::: agency.orchestrator.orchestrator.get_orchestrator

[Source: orchestrator.py](../../agency/orchestrator/orchestrator.py)

::: agency.orchestrator.orchestrator.peek_orchestrator

[Source: scheduler.py](../../agency/orchestrator/scheduler.py)

::: agency.orchestrator.scheduler.ExecutionScheduler
    options:
      members: ["__init__", "execute", "default_schedule", "materialize_dependencies"]

[Source: agresources.py](../../agency/orchestrator/agresources.py)

::: agency.orchestrator.agresources.agResourcePool
    options:
      members: ["__init__", "change_config", "get_config_copy", "acquire_gpus", "release_gpus", "acquire_cpu_mem", "release_cpu_mem"]

[Source: agresources.py](../../agency/orchestrator/agresources.py)

::: agency.orchestrator.agresources.GpuWaitCancelled
    options:
      members: []

[Source: agdatalogger.py](../../agency/observability/agdatalogger.py)

::: agency.observability.agdatalogger.agDataLogger
    options:
      members: ["__init__", "db_path", "change_config", "start", "stop", "flush", "read_profile_records", "record_event", "record_stream_delta", "record_llm_exchange", "record_span"]

[Source: agdatalogger.py](../../agency/observability/agdatalogger.py)

::: agency.observability.agdatalogger.read_llm_exchanges

[Source: agdatalogger.py](../../agency/observability/agdatalogger.py)

::: agency.observability.agdatalogger.reconstruct_llm_exchanges

[Source: agdatalogger.py](../../agency/observability/agdatalogger.py)

::: agency.observability.agdatalogger.resolve_global_db_path

[Source: agprof.py](../../agency/observability/profiler/agprof.py)

::: agency.observability.profiler.agprof
    options:
      members: ["start", "stop", "session", "workload", "profile_scope", "enabled", "span", "annotate", "thread_name", "spawn_traced", "bind_execution", "execution_context", "current_span_context", "current_span_attributes", "next_index", "summary_metrics", "profile_records", "summary_table"]

[Source: __init__.py](../../agency/observability/agwebui/__init__.py)

::: agency.observability.agwebui.agwebui
    options:
      members: ["run"]
