# agmap

`agmap` runs an ordinary (non-agent) function over one or more items concurrently — the deterministic analog of fanning out `agent.run()`. It is how you parallelise plain Python work (forking a sandbox, applying a patch, running a test) without a `ThreadPoolExecutor` or a throwaway `agteam`.

## Import

```python
from agency import agmap, agtask
```

## Signature

```python
agmap(fn, items, *, is_asynchronous=False) -> agtask | list[agtask]
```

- `fn` — a callable, or a list of callables.
- `items` — a single item, or a list of items.
- `is_asynchronous` — `False` (default) blocks and returns resolved results; `True` returns pending `agdata` immediately.

Each call runs on its own daemon thread. Results come back as `agtask` — an `agdata` subclass that behaves identically (lazy field access, `to_dict`, `wait_all`) but is recognised by `agsync` as a joinable target. A raised exception becomes `agerror`, isolated to that item.

## Pairing

Either argument may be a single value or a list:

| fn | items | runs |
|---|---|---|
| one | many | `fn(item)` for each item — the common "map" case |
| many | one | `fn(item)` for each fn |
| many | many | `fns[i](items[i])` (lists must be equal length) |
| one | one | a single `fn(item)` |

Mismatched, non-broadcastable lengths raise `ValueError`.

## Return shape

The result mirrors the input shape:

- A single `fn` **and** a single `item` → one `agtask`.
- Any list argument → a `list[agtask]`, in item order.

A list is always treated as a list of items. To pass a list as a *single* item, wrap it: `agmap(fn, [my_list])`.

## Synchronous vs asynchronous

### Synchronous (default)

Runs every task concurrently, blocks until all finish, and returns the resolved result(s):

```python
results = agmap(_validate, list(enumerate(candidates)))
for r in results:
    print(r.passed, r.output)
```

### Asynchronous

Returns pending `agdata` immediately so you can do other work, then join later:

```python
pending = agmap(_validate, candidates, is_asynchronous=True)
# ... do other work ...
agsync(pending)               # explicit barrier — like agsync(team)
# or: agdata.wait_all(pending)
# or: pending[0].passed       # accessing a field also resolves that task
```

The futures live **only** in the returned agtasks — there is no global registry.
Keep the handle: a dropped handle cannot be joined later, the same discipline as
dropping an `agteam` object.

## Relationship to agsync

`agsync` recognises `agtask` targets exactly as it recognises agents and teams — pass the pending results explicitly:

```python
pending = agmap(job, items, is_asynchronous=True)
agsync(pending)                   # joins these tasks
agsync(team, pending)             # mixes freely with agents and teams
```

- **Synchronous `agmap`** joins internally; there is nothing left for `agsync` to do.
- **Bare `agsync()` never waits on agmap tasks** — only on the targets you pass. This is deliberate: it makes calling `agsync()` from *inside* a mapped function safe (it cannot deadlock by waiting on its own still-running task, which a global registry would allow).
- Plain `agdata` objects are still rejected with `TypeError` — only `agtask` (and agents/teams) are joinable targets.
- A failed task resolves to `agerror` rather than re-raising from `agsync` (unlike a team whose `run()` raised) — inspect `"error" in r.to_dict()`.

## Example: parallel sandbox validation

`agmap` replaces the "one throwaway team per item" pattern. Each call forks its own private sandbox, so the tasks never share — or race on — one image:

```python
def _validate(item):
    i, patch = item
    sb = base.fork(f"val-{i}")            # private clean repo
    try:
        apply_patch(sb, patch)
        out, rc = sb.exec("python test.py", workdir="/workspace/repo")
        return agdata(passed=(rc == 0), output=out)
    finally:
        sb.destroy()

results = agmap(_validate, list(enumerate(candidates)))   # runs in parallel, blocks
```

Container creation inside `fn` is throttled by the sandbox container semaphore, so mapping over a large list never starts unbounded containers even though each task gets its own thread.

## Error behaviour

A raised exception in `fn` never propagates — it becomes an `agerror` for that item, and the other items are unaffected:

```python
results = agmap(risky, items)
for r in results:
    if "error" in r.to_dict():
        print("failed:", r.error)
    else:
        print("ok:", r.result)
```

Because both `agmap` results and skill results are `agdata` / `agerror`, they can be mixed in one `agdata.wait_all`.

## Relationship to `agent.run` and `fork`

| Need | Primitive |
|---|---|
| Run an LLM agent (reasoning + tools) | `agent.run(skill, input)` |
| Run several agents from one starting state | `agent.fork(ag).run(...)` |
| Run a plain function over N items in parallel | `agmap(fn, items)` |

`agmap` is the non-agent sibling of forked `agent.run()`: the same "submit work, get a pending `agdata`, join with `agsync` / `wait_all`" ergonomics, but for deterministic functions with no LLM, history, or logging. It does **not** fork a sandbox for you — the mapped function does that itself (as shown above), which keeps `agmap` a general parallel-map primitive rather than a sandbox-specific one.
