# Pending result and lifecycle API

Agency exposes a scheduled skill call as a pending `agdata`. There is no public
invocation wrapper or second lifecycle state machine.

```python
from agency import Agent, agdata

result: agdata = worker.run(skill, skill_input)
```

`Agent` is an alias of `agent`. `agskill.run(worker, input)` uses the same
submission path and returns the same pending result. Submission is lazy: the
global orchestrator creates an engine and sandbox only after dependencies and
the agent's earlier context entries are ready.

## Results and dependencies

`run()` returns immediately. Any operation that needs the result waits for it:

```python
result.is_pending()          # non-blocking status check
result.wait(timeout=300)     # resolve in synchronous code
output = await result        # resolve in asynchronous code
value = result.answer        # field access also resolves
mapping = result.to_dict()   # serialization also resolves
```

A pending result can be passed directly as another skill's input, including
inside supported lists, tuples, dictionaries, dataclasses, and model objects.
The scheduler discovers these dependencies recursively and keeps blocked work
out of execution workers and global engine capacity. `agdata.wait_all()` joins
several pending results.

Once `run()` has returned a result, terminal cancellation, dependency failure,
and scheduler failure resolve to error-shaped `agdata` rather than leaving the
future pending. Submission validation or closed admission can raise before a
result is returned.

## One ordered context chain

`run()` and `queue_message()` share one authoritative context chain per agent.
The predecessor context is an implicit dependency, so later submissions on the
same agent cannot overtake earlier ones.

```python
worker.queue_message("Use the production database")  # returns None
result = worker.run(skill, skill_input)               # sees the message
```

`queue_message()` is a host-only context operation. It creates no engine,
sandbox, harness, host service, model call, or execution-worker job. A message
queued after a result was submitted does not retroactively change that result.

## Exact-result controls

The pending `agdata` privately carries the request identity needed by agent
controls:

```python
result = worker.run(skill, skill_input)
worker.redirect(result, "Do not modify database rows")
worker.cancel(result)
```

`redirect()` addresses the execution that produced that exact result. An
active interactive external harness receives it through its PTY; otherwise it
is queued once as future context. A redirect never resumes a paused agent.

`cancel()` is idempotent. Work cancelled while dependency-blocked or queued is
settled without creating execution infrastructure. Running work observes
cancellation at safe boundaries, and the transaction fence prevents a late
successful sandbox commit from winning after cancellation.

## Agent-wide pause and resume

```python
worker.pause()
assert worker.is_paused()
pending = worker.run(skill, skill_input)
worker.resume()
```

Pause is an agent-wide persistent request. It freezes an active harness process
when possible and keeps later engine-backed submissions from launching until
`resume()`. Queued work consumes no worker or engine-capacity slot. There is no
public `Invocation.pause()`, `agent.suspend()`, or `agent.destroy()` API.
Sandbox cleanup is owned by `agSandbox`; callers that explicitly use a sandbox
can call `sandbox.destroy()`.

## Context on failure

Cancellation and dependency failure copy the committed predecessor context
without adding the failed attempt. An ordinary executed skill failure also
discards the working sandbox transaction, then appends the canonical retained
rollback notice to a clean predecessor copy. Result settlement happens only
after the corresponding context has settled, so callbacks can safely inspect
history or submit follow-up work.
