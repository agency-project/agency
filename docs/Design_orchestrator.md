# Global Agent Orchestrator

`GlobalAgentOrchestrator` is the process-wide host-side scheduler for every
`agent.run()` submission. It owns dependency admission, ready ordering,
per-agent exclusion, the optional global engine limit, and the global data
collector. `agskill.run()` remains the public compatibility wrapper, but it no
longer creates a worker thread itself.

The implementation is split by ownership:

- `agency/orchestrator/orchestrator.py` contains `GlobalAgentOrchestrator`,
  request and engine lifecycle, the event thread, telemetry, and singleton
  lifecycle.
- `agency/orchestrator/scheduler.py` contains `ExecutionScheduler`, the wait
  pool dependency resolver, ready heap, cycle detection, dependency
  materialization, and scheduling policy.
The main object exposes its scheduling component as `orchestrator.scheduler`.

## Request lifecycle

Each submission becomes an internal execution request with a stable request
ID, monotonically increasing submission sequence, agent, skill, input,
dependency set, captured profiler context, result future, and lifecycle state:

```text
submitted ──┬── no unresolved inputs ──> ready ──> running ──> completed
            └── pending inputs ────────> blocked ─┘              └─> failed
```

The scheduler has one dedicated thread blocked on a condition-backed event
queue. Submissions, dependency callbacks, engine completion, and shutdown push
events into that queue. There is no completion polling interval.

Pending inputs are discovered recursively through `agdata`, dictionaries,
lists, and tuples. A blocked request has no engine thread and consumes no
engine-capacity slot. A future callback only posts a dependency event; the
scheduler owns all lifecycle transitions.

## Event-driven scheduling cycle

Every submission, dependency completion, engine completion, and shutdown event
causes one `orchestrator.scheduler.execute()` cycle under the scheduler lock:

```text
apply scheduler event
        │
        ▼
resolve_dependency(request) for every submitted/blocked request
        │
        ▼
move every newly resolved request to the ready heap
        │
        ▼
schedule() → default_schedule()
        │
        ▼
launch every eligible ready request
```

Dependency-completion events are wakeups rather than targeted transition
commands. Scanning the complete wait pool before scheduling means an engine
completion can promote all requests it unblocked before the newly available
capacity is assigned. `schedule` is wired to `default_schedule` as the policy
seam; the default drains all eligible ready work subject to the global capacity
and one-active-engine-per-agent constraints.

## Admission and ordering

Ready requests are ordered by submission sequence. The scheduler chooses the
oldest request whose agent is not already running and for which global capacity
is available. A dependency-blocked request is not in the ready heap, so a
later ready request—including one for the same agent—can run first. This is the
ready-first rule that prevents the old registration-order history-chain
deadlock.

Dispatch is atomic under the scheduler lock: the request leaves the ready
queue, capacity is reserved, its agent is marked active, and a fresh daemon
thread and fresh request-owned `AgentEngine` are created. At most one engine
thread is active per agent. When it completes, the scheduler commits the
returned context before resolving the public result and releasing the agent
slot. Consequently, history reflects actual execution order.

`max_concurrent_engines=None` preserves unlimited cross-agent concurrency. A
positive integer caps active engine threads across the process. Resource-aware
admission remains the responsibility of `agResourcePool` because requests do
not declare their resource needs before execution.

## Dependency failures and cycles

Cancelled futures, future exceptions, and futures resolving to `agerror` fail
their consumer without launching its engine or changing its agent context.
Futures produced by this orchestrator are tagged with producer request IDs.
The scheduler runs strongly connected component detection over that known
producer graph and fails every member of a cycle. External futures are opaque:
their completion still wakes the scheduler, but an external future that never
settles waits until explicit shutdown.

`agteam` and `agmap` keep their existing thread models. Their pending `agdata`
objects participate as ordinary external dependencies.

## Global collection and scoped facades

The orchestrator owns one `GlobalDataCollector` and one SQLite writer thread.
Publishers update the bounded in-memory scheduler snapshot, assign a global
sequence, enqueue immutable records, and return without doing disk I/O. The
writer owns one WAL-mode connection and stores:

- append-only lifecycle and harness events;
- harness and scheduler-derived profiler spans;
- latest-value projections for agent, scheduler, token, and resource state;
- the Web UI event stream and reconnect projections.

Each engine execution receives a `ScopedDataCollector` facade. The host
interaction server uses the same methods as before, while the facade
automatically attaches agent name, request ID, skill, and harness source. Its
`stop()` method does not stop the process-wide writer.

The Web UI also acts as a facade. Events emitted during agent construction are
buffered until the first submission freezes orchestrator configuration, then
drained into the global writer. The UI server follows the selected global
database path and continues to use the existing command/reply files.

SQLite and subscriber errors are telemetry failures: they are exposed in the
collector snapshot and logged as warnings but never change an agent result.
`flush()` and normal shutdown drain queued records.

## Configuration and lifecycle

```python
from agency import agOrchestratorConfig, get_orchestrator
from agency.agconfig import agConfig

cfg = agConfig(
    agOrchestratorConfig(
        max_concurrent_engines=8,  # None means unlimited
        db_path="runs/my-run/agency.sqlite3",
        flush_batch_size=500,
        flush_interval_s=1.0,
    )
)

# Pass cfg to the first agent that submits work.
snapshot = get_orchestrator().snapshot()
get_orchestrator().flush()
get_orchestrator().shutdown()
```

Database and capacity configuration freeze when the singleton is first
created. Explicit shutdown rejects new submissions, finishes running and ready
work, fails requests that remain dependency-blocked after no runnable producer
can progress, flushes SQLite, and joins the scheduler and writer threads. A
best-effort `atexit` hook initiates the same shutdown path.
