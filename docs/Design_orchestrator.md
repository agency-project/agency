# Global orchestrator design

`GlobalAgentOrchestrator` is the process-wide authority for skill invocations and host-only messages. Public handles do not run their own schedulers and do not mirror lifecycle state: each `Invocation` or `MessageSubmission` is the exact submission referenced by its internal orchestrator request.

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
6. Register an orchestrator request that references the exact public submission.

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

Each event runs a complete cycle: resolve every waiting request, detect managed dependency cycles, promote all newly ready requests, then dispatch eligible ready work. Recursive dependency discovery and materialization understand agdata-compatible pending handles, including `Invocation` and `MessageSubmission`, along with dictionaries, lists, tuples, dataclasses, and supported model objects.

Dispatch requires all of the following:

- predecessor and explicit input dependencies are resolved;
- the agent is not suspended;
- the invocation is not terminal or specifically paused;
- no other engine-backed request is active for that agent;
- global engine capacity is available.

`max_concurrent_engines=None` preserves effectively unlimited cross-agent concurrency. A positive integer caps active engine-backed requests across the process. Dependency-blocked and queued requests held behind agent suspension consume no engine capacity; context-only messages never claim it. An invocation that was already running when it parked at a suspension boundary remains active and retains its slot.

## Reusable workers, fresh engines

The orchestrator owns one lazily populated execution-worker pool. It does not create one thread per invocation. Sequential requests can reuse a worker thread, but every dispatched request receives a new `AgentEngine` bound to that request's exact `Invocation`.

Every worker job runs in a fresh `contextvars.Context`, while the profiler parent captured at submission is passed explicitly to the execution span. Reuse therefore cannot leak invocation-local context or tracing state.

Submission to the pool is admission-gated. `ThreadPoolExecutor` queues a work item before it may attempt to start a worker; if thread startup fails, the gate marks that queued item rejected. A later healthy worker drains it as a no-op while the scheduler settles the already-failed request exactly once.

Engine completion is posted back to the scheduler. The worker never publishes public futures or releases scheduler capacity directly.

## Host-only messages

`queue_message()` registers a request with kind `context_message`. Once its predecessor resolves, the scheduler copies that context, appends the validated retained message, settles the output context, then settles the empty result. This path does not create an `AgentEngine`, sandbox, daemon, host server, harness, or model request, and it does not use an execution worker or global engine slot.

Agent suspension is not a dispatch gate for a ready context message. Any unresolved predecessor still blocks it through the ordinary context chain, including a running invocation parked by suspension.

## Controls and terminal settlement

Queued cancellation, dependency failure, scheduler rejection, and destruction are handled on the scheduler thread without creating execution infrastructure. Running cancellation and destruction are observed by the exact `Invocation` at safe boundaries; the completion claim in the engine transaction prevents a late successful commit from winning after a terminal control.

Ordinary skill failure discards its working context and appends the canonical retained rollback notice. Cancellation and destruction pass through committed predecessor context without that notice.

For every terminal path the scheduler:

1. settles or arranges pass-through of the output context;
2. marks the public submission terminal;
3. releases request, agent, dependency, and capacity bookkeeping;
4. settles the public result;
5. permits result callbacks to run against settled context.

## Telemetry and shutdown

Request lifecycle events and scheduler-derived spans go to the orchestrator's
shared `agDataLogger`. Each execution worker binds `agprof` to that
request's per-agent `agDataLogger`, so completed engine, sandbox, tool, and
LLM profiler spans are persisted with the same schema as other agent data.
The host-server thread and traced child threads inherit that binding.

Profiler spans retain wall, CPU, run-queue, blocked, span-ID, parent-ID, and
request-correlation fields. Scheduler external spans are exported directly
through the shared logger when profiling is enabled; the existing lightweight
timer writes the same scheduler interval when profiling is disabled, so the
database has one row per interval in either mode. Logging failures remain
observational and never alter request results. LLM stream deltas remain
temporary `agdatalogger.py` records and are finalized into durable events by
the LLM server.

Explicit shutdown closes admission, drains active work and context continuations, fails requests that cannot run, joins the scheduler, and retires the reusable workers. Shutdown is idempotent. A callback executing on the scheduler thread may request non-blocking shutdown; scheduler teardown closes the worker pool after the event loop drains.
