# Invocation API

Agency exposes one scheduled request as both a lifecycle handle and a pending data dependency. There is no wrapper with an independent state machine: the global orchestrator, engine, harness, and caller all reference the exact public `Invocation`.

```python
from agency import Agent, CloseHandle, Invocation, MessageSubmission, agent

inv: Invocation = ag.run(skill, skill_input)
```

`Agent` is an alias of `agent`. `agskill.run(ag, input)` uses the same submission path and returns the same `Invocation` type. `run()` submits scheduler-eligible work; it does not eagerly start a harness or create execution infrastructure.

## Results and dependencies

`run()` returns immediately. An invocation offers the following result-compatible surface:

```python
pending = inv.result          # pending agdata handle
inv.wait()                    # blocks and returns inv
output = await inv            # resolves to agdata
value = inv.answer            # unknown attributes proxy to the output
literal = inv.result.result   # output field literally named "result"
```

An invocation implements Agency's pending-data protocol, so it can be passed directly as another skill input or nested in supported lists, tuples, dictionaries, dataclasses, and model objects. Recursive dependency discovery, materialization, cycle detection, serialization, and `agdata.wait_all()` all understand that protocol. Cancelling one asynchronous waiter is shielded from cancelling the shared invocation future.

Once an accepted submission handle has returned, terminal cancellation, destruction, dependency failure, and later scheduler failure resolve to `agerror` data rather than leaving a pending future. Submission validation, closed admission, or failure of the initial acknowledged scheduler cycle may instead raise synchronously before the caller receives a usable handle. For every returned handle, the submission's output context settles before its public result, so result callbacks can safely read committed history or submit follow-up work.

## One ordered context chain

`run()` and `queue_message()` share one authoritative context chain per agent. Each call atomically validates admission, allocates a monotonic ordering ID, captures the current context head, publishes its own output-context placeholder as the new head, and registers the request with the orchestrator. The predecessor context is an implicit dependency, so later same-agent submissions cannot overtake earlier ones.

`Agent.queue_message()` advances that serialized chain:

```python
# An ordered context entry for later submissions.
receipt: MessageSubmission = ag.queue_message("Use the production database")
await receipt

# Submitted after the entry, so this invocation observes it.
inv = ag.run(skill, skill_input)
```

Submission order is authoritative:

```python
inv = ag.run(skill, skill_input)

# Queued after `inv`; this does not retroactively update `inv`.
await ag.queue_message("Use the production database")
later = ag.run(other_skill, other_input)
```

When its predecessor resolves, the scheduler copies that context, appends the validated retained user message, then settles an empty result. This host-only path creates no engine, sandbox, harness, host service, model request, worker job, or capacity claim. Agent suspension is not a gate for a ready context-only message, although an unresolved predecessor can still block it.

`MessageSubmission` supports pending result, waiting, awaiting, context chaining, and serialization behavior. It deliberately has no `send_message()`, `pause()`, `resume()`, or `cancel()` invocation controls.

## Exact-invocation messages and controls

`Invocation.send_message()` targets one already-submitted invocation without creating a context-chain node or changing submission order:

```python
inv.send_message("Do not modify database rows")
inv.pause()
inv.resume()
inv.cancel()
```

An invocation message can be accepted while the invocation is queued or running. Before dispatch it is delivered at the first protocol-valid safe boundary. During execution it is delivered at the next one, in FIFO order. It never interrupts a model request or tool execution, does not resume a paused invocation, and does not clear agent suspension. If it arrives during a model request that otherwise produces a final response, Agency atomically admits it for a follow-up generation before establishing the final-answer fence.

Boundary assignments are retry-stable and sequence-deduplicated. Internal compaction/model-management calls do not consume ordinary invocation messages. A message is rejected after the final-answer, cancellation, destruction, or completion fence.

Pause and cancellation are also cooperative safe-boundary controls. Cancellation is idempotent in blocked, queued, running, paused, and terminal states; work that has not dispatched settles without creating an engine.

Public invocation states are `QUEUED`, `RUNNING`, `PAUSED`, `SUCCEEDED`, `FAILED`, `CANCELLED`, and `DESTROYED`. A brief internal or observable `CANCELLING` transition may occur while running work advances to its next safe boundary.

The concise distinction is:

- `Agent.queue_message()` advances the agent's serialized context chain for later submissions.
- `Invocation.send_message()` delivers an additional instruction to one exact already-submitted invocation.

## Agent-wide lifecycle

Agent suspension and invocation pause are independent gates:

- `ag.suspend()` prevents new engine-backed dispatch and asks active work to park at its next safe boundary unless it has crossed the closing/completion fence.
- `ag.resume()` clears only agent suspension. It does not clear `inv.pause()`.
- `inv.pause()` and `inv.resume()` affect only that invocation.
- `inv.cancel()` cancels only that invocation and does not affect later submissions.

Queued work held by suspension consumes no worker or global execution slot. An invocation that was already running remains active and keeps its slot while parked. `is_suspended()`, `is_paused()`, `is_settled()`, and `lifecycle_state` derive from the shared agent control and orchestrator state.

## Destruction

```python
close: CloseHandle = ag.destroy()
close.wait()
# or: await close
assert ag.destroy() is close
```

Destruction closes admission synchronously, then drains asynchronously. Dependency-blocked, queued, running, paused, and context-message submissions all receive terminal results and contexts. Never-dispatched work creates no engine. Running work stops at a safe boundary, and a completion claim prevents success from committing after destruction has won the race.

The reusable `CloseHandle` settles only after active execution, registered submissions, operation leases, callbacks, collector state, resources, and owned sandbox cleanup have drained. Explicit destruction is the reliable lifecycle operation; `__del__` is best-effort only.

## Context on failure

Cancellation, destruction, dependency failure, scheduler rejection, and other pre-execution failure paths copy through the predecessor context without adding context, waiting asynchronously only when that predecessor remains unresolved. An ordinary executed skill failure also discards the working sandbox and context, but appends the canonical retained rollback notice to a clean copy of the predecessor. Cancellation and destruction do not append that notice.
