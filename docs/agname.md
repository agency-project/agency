# agname

Allocates unique, human-readable agent names from a process-wide shared registry.

## Why agent names matter

Every agent name doubles as a stable identifier used in log filenames, terminal
output, sandbox container names, and Docker image tags. Names must therefore be:

- Filesystem-safe (no spaces or special characters)
- Lowercase-compatible (base noun plus a base-36 suffix are both lowercase)
- Unique within a process so log files and containers never collide

## Class overview

`agname` subclasses `str`. Every instance is a plain string and can be used
wherever a string is expected — f-strings, dict keys, path segments, equality
comparisons — without any unwrapping.

```python
from agency.agname import agname

name = agname.allocate_agname()
print(name)          # e.g. "alex_0000"
print(type(name))    # <class 'agency.agname.agname'>
assert name == "alex_0000"
```

Class-level state (`_noun_index`, `_noun_counters`, `_allocated`) is shared
across all agents in a process and is protected by a single `threading.Lock`.

---

## Constructors

### `agname.allocate_agname(name: str | None = None) -> agname`

The primary way to create a name. Returns a unique name of the form
`<base>_XXXX` and marks it as in-use in the global registry.

| Argument | Behaviour |
|---|---|
| `None` (default) | Picks the next noun from the built-in pool of ~200 short English words, cycling back after the last entry. |
| A string | Uses that string as the base (e.g. `"Worker"` produces `"Worker_0000"`). |

`XXXX` is a four-character base-36 suffix (`0`–`9`, `a`–`z`), giving
1,679,616 unique values per base noun before any collision is possible.

```python
a = agname.allocate_agname()          # "alex_0000"
b = agname.allocate_agname()          # "andy_0001" (next noun)
c = agname.allocate_agname("Worker")  # "Worker_0000"
d = agname.allocate_agname("Worker")  # "Worker_0001"
```

### `agname.claim_unique_agname(name: str) -> agname`

Claims an exact, fully-formed name (including the suffix) as in-use. Raises
`ValueError` if the name is already allocated. Used when restoring an agent
from saved state where the original name must be preserved exactly.

```python
# Restore a saved agent without changing its name
name = agname.claim_unique_agname("hawk_000a")
```

---

## String representation

Because `agname` is a `str` subclass, `str(name)` and `repr(name)` both behave
like the underlying string value. No special formatting is added.

```python
name = agname.allocate_agname("bot")
print(str(name))    # "bot_0000"
print(repr(name))   # "bot_0000"
f"Agent: {name}"    # "Agent: bot_0000"
```

---

## Name recycling and global tracking

Names are **not** recycled automatically. Once a name is marked in `_allocated`
it stays there for the lifetime of the process. The `_noun_index` counter and
the per-base `_noun_counters` dict advance monotonically, so each call to
`allocate_agname` always produces a name that has never been issued before.

This is intentional: reusing a name mid-run would reuse its associated log file
path and container name, which would corrupt diagnostics.

---

## Common patterns

**Auto-named agent (most common)**

```python
name = agname.allocate_agname()
# Use name as a container tag, log prefix, etc.
```

**Named agent with a descriptive base**

```python
name = agname.allocate_agname("crawler")
# "crawler_0000", "crawler_0001", …
```

**Restoring a named agent from a checkpoint**

```python
name = agname.claim_unique_agname(saved_name)
```

---

## Constraints and gotchas

- The noun pool has ~200 entries. After 200 calls with `name=None` the pool
  wraps around, so the second cycle produces `"alex_0001"`, `"andy_0001"`, etc.
  This is harmless but worth knowing if you run large numbers of agents in a
  single process.
- `claim_unique_agname` raises immediately if the name is already taken; it
  does not increment any counter or fall back to a suffixed variant.
- Because class state is module-level and shared, unit tests that allocate names
  can affect later tests. Reset `agname._noun_index`, `agname._noun_counters`,
  and `agname._allocated` between tests if isolation is required.
