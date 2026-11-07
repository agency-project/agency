# Results, waiting, errors and cancellation

```python
from agency import agdata, agerror, agcanceled, AgError
```

`agdata(**fields)` is a mutable payload. `Agent.run()` returns a future-backed wrapper of this class; `_future` is an infrastructure parameter, not an application submission interface. `agerror(message)` and `agcanceled(message="agent invocation cancelled")` are payload classes, while `AgError` is an exception.

## Waiting and conversion

| API/operation | Behavior |
| --- | --- |
| `result.is_pending()` | Nonblocking bool; false when its future is done even before payload materialization. Not a success indicator. |
| `result.wait(timeout=None)` | Blocks and returns the same wrapper. With a timeout, raises `concurrent.futures.TimeoutError` if the immediate future is still pending; execution remains alive. Nested future resolution is not governed by a single overall deadline. Error payloads do not themselves raise. |
| `await result` | Waits asynchronously and returns the same wrapper. Shields the underlying future; cancelling the waiting task does not cancel execution or other waiters. Future exceptions still propagate. |
| Field read, `to_dict()`, `to_json()`, equality | Implicitly block until resolution. Missing fields on `agdata` raise `AttributeError`. `repr` is nonblocking for unfinished futures but resolves completed ones. |
| `agdata.wait_all(pending)` | Sequentially resolves/waits a list of compatible handles, returns that same list. Invalid entries raise `TypeError`; no timeout argument. |
| `to_dict()` | Recursively serializes nested agdata, lists, tuples (to lists), dataclasses and model-like values. Resolves nested payloads. Type markers become names. |
| `to_json()` | JSON-encodes `to_dict()`; unsupported objects such as binary bytes can raise `TypeError`. |
| `agdata.from_dict(d)` | Shallow field construction; nested dicts remain dicts. |
| `agdata.from_json(s)` | Parses an object and normalizes top-level camelCase keys. JSON parsing errors propagate. Supply a JSON object. |
| `resolve_input_dependencies()` | Blocks recursively resolving nested handles, reconstructing supported structured objects; mutates this payload, returns `None`. Mainly infrastructure use. |

Avoid writing fields on an unresolved wrapper: resolution can overwrite them. Methods can shadow payload field names; use `to_dict()` to access ambiguous keys. Serialization is not a deep-copy guarantee for arbitrary application objects.

## Failure payloads

**A pending wrapper stays `agdata` when its future resolves to `agerror` or `agcanceled`.** Resolution copies the payload only. `isinstance(result, agerror)` and `isinstance(result, agcanceled)` therefore do not detect failed/cancelled agent results. Accessing a missing output field on this wrapper raises `AttributeError`, not `AgError`.

```python
from concurrent.futures import Future
from agency import agdata, agerror

future = Future()
result = agdata(_future=future)  # synthetic illustration, no agent or model
future.set_result(agerror("failed"))
assert result.wait().to_dict() == {"error": "failed"}
assert not isinstance(result, agerror)
```

For current agent/tool composition, resolve and inspect the reserved `error` payload key before output fields. Avoid naming a success field `error`. This distinguishes failure from success but cannot reliably distinguish cancellation from other failure without additional application state. Direct `agerror` instances validate `message` as str (`TypeError` otherwise); `.error` returns that message and missing non-error field access raises `AgError`. Direct `agcanceled` inherits that behavior. A future can also hold an exception: waiting then raises that exception, notably for [team run exceptions](workflows.md).

## Targeting and timing

| Action | Target and timing |
| --- | --- |
| `worker.cancel(handle)` | Uses the owned pending handle's future; request must belong to `worker`. Wrong-agent, unknown or already-finished requests are no-ops. After `wait()` clears the future, cancellation is a no-op. Does not cancel the future itself. |
| Cancellation queued/blocked | Flagged but settlement may wait for dependencies/capacity. Wait for the handle to observe outcome; cancellation is not an immediate join. |
| Cancellation running | Best-effort engine/harness interruption, followed by teardown/rollback. An atomic completion claim wins a race with late cancellation; successful work already committing can finish successfully. |
| `worker.redirect(handle, message)` | Nonempty str required (`TypeError`/`ValueError`). Checks handle owner/request identity (`ValueError` for invalid/wrong-agent handle). Preserved identity allows calls after resolution. Does not wait for the result. |
| Redirect active | Attempts delivery to that exact request. It cannot redirect the successor by mistake. Native/unsupported harnesses or failed/late delivery fall back to `queue_message(message)` once. |
| Redirect early/late | If no active accepting target is available, queues a retained message for a later submission. It does not rewrite inputs/context already captured by submitted successors. |
| `worker.pause()` / `resume()` | Agent-wide state plus best-effort active harness RPC. Applies to subsequent runs as well; not a handle cancellation and not a scheduler capacity release. |

Use original handles for controls; reconstructed `agdata` objects are not control handles. Waiting with timeout, cancelling an asyncio waiter and cancelling execution are separate operations. To interrupt execution after a waiting timeout, call `worker.cancel(handle)` while it is still pending, then wait for settlement.

## Source signatures

[Source: agdata.py](../../agency/agdata.py)

::: agency.agdata.agdata
    options:
      members: ["__init__", "is_pending", "wait", "wait_all", "to_dict", "to_json", "from_dict", "from_json", "resolve_input_dependencies"]

[Source: agdata.py](../../agency/agdata.py)

::: agency.agdata.agerror
    options:
      members: ["__init__"]

[Source: agdata.py](../../agency/agdata.py)

::: agency.agdata.agcanceled
    options:
      members: ["__init__"]

[Source: agdata.py](../../agency/agdata.py)

::: agency.agdata.AgError
    options:
      members: []
