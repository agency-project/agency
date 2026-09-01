# Invocation API

Agency exposes the same scheduled request as both a lifecycle handle and a pending data dependency. There is no wrapper with an independent state machine: the global orchestrator, engine, harness, and caller all reference the exact public `Invocation`.

```python
from agency import Agent, CloseHandle, Invocation, MessageSubmission, agent

inv: Invocation = ag.run(skill, skill_input)
```

`Agent` is an alias of `agent`. `agskill.run(ag, input)` uses the same submission path and returns the same `Invocation` type.

## Results and dependencies

`run()` and `prepare()` return immediately. An invocation offers the following result-compatible surface:

```python
pending = inv.result       # pending agdata handle
inv.wait()                # blocks and returns inv
output = await inv        # resolves to agdata
value = inv.answer        # unknown attributes proxy to the output
literal = inv.result.result  # output field literally named "result"
```

An invocation implements Agency's pending-data protocol, so it can be passed directly as another skill input or nested in supported lists, tuples, dictionaries, dataclasses, and model objects. Recursive dependency discovery, materialization, cycle detection, serialization, and `agdata.wait_all()` all understand that protocol. Cancelling one asynchronous waiter is shielded from cancelling the shared invocation future.

Once an accepted submission handle has returned, terminal cancellation, destruction, dependency failure, and later scheduler failure resolve to `agerror` data rather than leaving a pending future. Submission validation, closed admission, or failure of the initial acknowledged scheduler cycle may instead raise synchronously before the caller receives a usable handle. For every returned handle, the submission's output context settles before its public result, so result callbacks can safely read committed history or submit follow-up work.

## Submission order

`run()`, `prepare()`, and `send()` share one authoritative context chain per agent. Each call atomically validates admission, allocates a monotonic ordering ID, captures the current context head, publishes its own output-context placeholder as the new head, and registers the request with the orchestrator.

The predecessor context is an implicit dependency. Later same-agent submissions therefore cannot overtake an earlier one, including an earlier PREPARED request.

```python
first = ag.run(first_skill, agdata(topic="one"))
held = ag.prepare(second_skill, first)
message = ag.send("Retain this after the second skill")
last = ag.run(last_skill, agdata(topic="four"))

assert held.state == "PREPARED"
held.start()
```

Before `held.start()`, both `message` and `last` remain behind its unresolved context position. The prepared request creates no engine, sandbox, daemon, host server, harness, dependency-wait thread, or per-request execution thread and consumes no execution slot.

`inv.start()` opens exactly that invocation's readiness gate without moving it. `ag.start()` opens a snapshot of all invocations already PREPARED at the instant of the call; concurrently prepared work outside that snapshot remains PREPARED. Repeated starts are safe. Neither `ag.resume()` nor `inv.resume()` starts prepared work.

## Ordered messages

```python
receipt: MessageSubmission = ag.send("Use primary sources")
receipt.wait()
```

A message is an orchestrator-owned, host-only request. When its predecessor resolves, the scheduler copies that context, appends a validated retained user message, then settles an empty result. It never creates an engine, sandbox, harness, host service, model request, worker job, or capacity claim. The suspension gate is not consulted once a message is ready, but every unresolved predecessor still blocks it—including PREPARED work or an active invocation parked by suspension.

`MessageSubmission` supports pending result, waiting, awaiting, context chaining, and serialization behavior. It deliberately has no `start()`, `steer()`, `pause()`, `resume()`, or `cancel()` skill controls.

## Invocation controls

While an invocation has not crossed its closing fence, it can be controlled directly:

```python
inv.steer("Prefer the smaller implementation")
inv.pause()
inv.resume()
inv.cancel()
```

Controls are observed only at explicit safe boundaries. Pause never interrupts an in-flight model request or tool execution. Steering is FIFO, is delivered only at protocol-valid boundaries, is replayed consistently when the same request boundary is retried, and is rejected during model or closing phases. Internal compaction calls do not consume ordinary user steering. Cancellation is idempotent in PREPARED, blocked, queued, running, paused, and terminal states; work that has not dispatched is settled without creating an engine.

Public invocation states include `PREPARED`, `QUEUED`, `RUNNING`, `PAUSED`, `SUCCEEDED`, `FAILED`, `CANCELLED`, and `DESTROYED`. A brief `CANCELLING` transition may be observable while a running invocation advances to its next safe boundary.

## Agent-wide lifecycle

Agent suspension and invocation pause are independent gates:

- `ag.suspend()` prevents new engine-backed dispatch and asks active work to park at its next safe boundary unless it has already crossed the closing/completion fence.
- `ag.pause()` is a compatibility alias for `ag.suspend()`.
- `ag.resume()` clears only agent suspension. It does not start PREPARED work or clear `inv.pause()`.
- `inv.pause()` and `inv.resume()` affect only that invocation.
- `ag.cancel()` targets only the currently active invocation; it does not cancel future submissions.

Queued work held by suspension consumes no worker or global execution slot. An invocation that was already running remains active and keeps its slot while parked. `is_suspended()`, `is_paused()`, `is_settled()`, and `lifecycle_state` derive from the shared agent control and orchestrator state.

## Destruction

```python
close: CloseHandle = ag.destroy()
close.wait()
# or: await close
assert ag.destroy() is close
```

Destruction closes admission synchronously, then drains asynchronously. PREPARED, dependency-blocked, queued, running, paused, and message submissions all receive terminal results and contexts. Never-started work creates no engine. Running work stops at a safe boundary, and a completion claim prevents success from committing after destruction has won the race.

The reusable `CloseHandle` settles only after active execution, registered submissions, operation leases, callbacks, collector state, resources, and owned sandbox cleanup have drained. Explicit destruction is the reliable lifecycle operation; `__del__` is best-effort only.

## Context on failure

Cancellation, destruction, dependency failure, scheduler rejection, and other pre-execution failure paths copy through the predecessor context without adding context, waiting asynchronously only when that predecessor remains unresolved. An ordinary executed skill failure also discards the working sandbox and context, but appends the canonical retained rollback notice to a clean copy of the predecessor. Cancellation and destruction do not append that notice.
