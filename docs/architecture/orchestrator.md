# orchestrator — ordering and shared capacity

`agency/orchestrator/` decides when a request may run and publishes its final outcome. One process-wide orchestrator coordinates all agents, allowing same-agent conversation order and cross-agent input dependencies to share one scheduling model.

## Internal responsibilities

| Source | Responsibility |
| --- | --- |
| [orchestrator.py](../../agency/orchestrator/orchestrator.py) | Request registry, scheduler event loop, worker dispatch, controls and future settlement. |
| [scheduler.py](../../agency/orchestrator/scheduler.py) | Discover dependencies, detect managed dependency cycles and promote ready requests. |
| [agresources.py](../../agency/orchestrator/agresources.py) | Shared GPU leases and CPU/memory limit bookkeeping for sandboxes. |

Submission holds the orchestrator's event lock and the agent's submission lock while registering work and replacing the agent's context future. The new request keeps its predecessor context. This makes successive submissions to one agent a chain even when application threads submit concurrently.

Pending values inside inputs add dependency edges. Future callbacks post events to the scheduler; they do not run the dependent skill themselves. Ready requests enter a sequence-ordered queue. Context-only messages settle on the scheduler without allocating an engine or starting a sandbox.

## Admission and execution

The scheduler admits ready skill requests up to `max_concurrent_engines`, then dispatches an [engine](engine.md) through a reusable thread executor. Dependency-blocked work holds no engine slot. A slot stays occupied through startup, attempts, output repair and teardown until the scheduler handles completion. Pausing a harness does not release that slot.

Engine capacity and sandbox resources have different lifetimes. GPU acquisition can wait after engine admission; GPU release follows sandbox stop or teardown. CPU and memory reservations update limits and accounting rather than acting as the same globally blocking queue.

## Settlement and controls

Workers return execution outcomes to the event loop. The orchestrator resolves context, removes active request ownership and then resolves the public result. Failed execution can preserve predecessor context for the next same-agent request; a failed input dependency can fail its consumer.

Cancellation and the engine's completion claim share a lock. Cancellation wins before the claim; after a successful claim it cannot undo that outcome. Redirect targets a registered request and delegates delivery to its engine. These are process-local coordination rules: the scheduler does not provide distributed execution or recovery from saved log records.

The [submission tests](../../tests/test_submission_orchestrator.py) and [transaction fence tests](../../tests/test_orchestrator_transaction_fence.py) exercise the ordering and settlement boundaries. Public scheduling objects are covered in the [runtime reference](../api/runtime.md).
