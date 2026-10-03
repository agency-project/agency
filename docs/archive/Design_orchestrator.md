# Global orchestrator design

`GlobalAgentOrchestrator` is the process-wide authority for skill requests and host-only messages. A skill request exposes only a pending `agdata`; the orchestrator privately maps that result's future to the authoritative internal request used by `Agent.redirect()` and `Agent.cancel()`.

The implementation is split by ownership:

- `agency/orchestrator/orchestrator.py` owns atomic submission, request lifecycle, context settlement, the scheduler event thread, reusable execution workers, telemetry, and singleton shutdown.
- `agency/orchestrator/scheduler.py` owns dependency discovery, the wait pool, managed-cycle detection, the ready heap, context-message routing, and dispatch policy. The orchestrator performs context-message settlement.
- `agency/orchestrator/agresources.py` remains the process-wide CPU, memory, and GPU resource authority.

## One authoritative context chain

Every agent has one context head. A call to `run()` or `queue_message()` takes the orchestrator condition and then the agent submission lock and performs one atomic publication:

1. Verify that the orchestrator and agent still accept work.
2. Allocate the agent's next monotonic ordering ID.
3. Capture the current context head as the predecessor.
4. Create the submission's unresolved output-context placeholder.
5. Publish that placeholder as the new head.
6. Register an orchestrator request and attach its private identity to the pending result.

Registration and chain publication therefore have the same linearization point. The predecessor context is an implicit dependency, so same-agent submissions cannot overtake one another.

Pre-execution failures and controlled terminal paths that did not commit new context pass their predecessor through without adding context, using an event-driven continuation only when that predecessor remains unresolved. An ordinary executed skill failure instead copies its predecessor and appends the canonical rollback notice. Context is always settled before the public result, so result callbacks see the submission's context as complete.

## Requests and event scheduling

The scheduler thread blocks on a condition-backed event queue. Submission, control changes, dependency callbacks, engine completion, context pass-through completion, and shutdown post events. There is no scheduler polling interval and no dependency-wait thread per request.

The important internal states are:

```text
submitted ── unresolved predecessor/input ──> blocked
        │                                      │ dependency event
        └──────────── dependencies ready ◀─────┘
                              │
                              ├─ message ──> context-only completion
                              │
                              ▼
                            ready ──> running ──> terminal
```

Each event runs a complete cycle: resolve every waiting request, detect managed dependency cycles, promote all newly ready requests, then dispatch eligible ready work. Recursive dependency discovery and materialization understand pending `agdata` along with dictionaries, lists, tuples, dataclasses, and supported model objects.

Dispatch requires all of the following:

- predecessor and explicit input dependencies are resolved;
- the request is ready and not terminal;
- no other engine-backed request is active for that agent;
- global engine capacity is available.

`max_concurrent_engines=None` preserves effectively unlimited cross-agent concurrency. A positive integer caps active engine-backed requests across the process. Dependency-blocked requests consume no engine capacity; context-only messages never claim it. Pause is enforced by the harness daemon, so even a request submitted while paused can dispatch and retain its slot. Cancelled queued requests dispatch when dependencies and capacity permit, then return cancellation before sandbox provisioning.

## Reusable workers, fresh engines

The orchestrator owns one lazily populated execution-worker pool. It does not create one thread per request. Sequential requests can reuse a worker thread, but every dispatched request receives a new `AgentEngine` bound to that internal request.

Every worker job runs in a fresh `contextvars.Context`, while the profiler parent captured at submission is passed explicitly to the execution span. Reuse therefore cannot leak request-local context or tracing state.

Submission to the pool is admission-gated. `ThreadPoolExecutor` queues a work item before it may attempt to start a worker; if thread startup fails, the gate marks that queued item rejected. A later healthy worker drains it as a no-op while the scheduler settles the already-failed request exactly once.

Engine completion is posted back to the scheduler. The worker never publishes public futures or releases scheduler capacity directly.

## Host-only messages

`queue_message()` registers a request with kind `context_message` and returns `None`. Once its predecessor resolves, the scheduler copies that context and appends the validated retained message. This path does not create an `AgentEngine`, sandbox, daemon, host server, harness, or model request, and it does not use an execution worker or global engine slot.

Agent pause is not a scheduler dispatch gate. Any unresolved predecessor still blocks it through the ordinary context chain, including a running request whose harness is paused.

## Controls and terminal settlement

Dependency failure and scheduler rejection settle on the scheduler thread without execution infrastructure. Queued cancellation waits for normal dispatch and is then observed before sandbox provisioning. Running cancellation uses the exact engine and execution ID for a daemon kill RPC. Its state change races atomically with the engine completion claim; whichever wins determines whether the working transaction can commit.

Ordinary skill failure discards its working context and appends the canonical retained rollback notice. Cancellation passes through committed predecessor context without that notice.

For every terminal path the scheduler:

1. settles or arranges pass-through of the output context;
2. marks the public submission terminal;
3. releases request, agent, dependency, and capacity bookkeeping;
4. settles the public result;
5. permits result callbacks to run against settled context.

## Telemetry and shutdown

Request lifecycle events go to the orchestrator's shared `agDataLogger`;
spans are a separate, purely opt-in concept. Every span -- request/phase
spans from the scheduler, `engine:execute`, sandbox, tool, and LLM profiler
spans alike -- is recorded only while an `agprof` session is active, and all
of them are persisted to that one session's own `agDataLogger`
(`profile_data.sqlite3` for an on-disk profile, otherwise an in-memory
database) rather than being routed to whichever agent or the orchestrator
happened to be executing. With profiling disabled, no span rows are written
at all; the event log (`request_submitted`, `request_started`, etc.) is the
only durable record of request timing in that mode. Each profiling session
has a correlation ID. At shutdown, agprof flushes its own datalogger and
builds the Perfetto trace and summaries from its persisted span rows.

Profiler spans retain wall, CPU, run-queue, blocked, span-ID, parent-ID, and
request-correlation fields. Logging failures remain observational and never
alter request results. LLM stream deltas remain temporary `agdatalogger.py`
records and are finalized into durable events by the LLM server.

Explicit shutdown closes admission, drains active work and context continuations, fails requests that cannot run, joins the scheduler, and retires the reusable workers. Shutdown is idempotent. A callback executing on the scheduler thread may request non-blocking shutdown; scheduler teardown closes the worker pool after the event loop drains.
