# agutil

Internal utility functions shared across the agency framework.

These helpers are not part of the public API. User code generally does not import `agutil` directly, with two exceptions: `format_exception` is occasionally useful in custom error-handling code, and `sigterm_as_exit` is a context manager scripts should reach for whenever they run agency code directly (not via `agwebui.run()`/`graphui.run()`, which already use it internally) — see below. Both are also re-exported from the top-level package: `agency.sigterm_as_exit`.

---

## `format_exception(e)`

Returns the full traceback and exception message as a single string.

**Signature**

```python
def format_exception(e: BaseException) -> str
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `e` | `BaseException` | The caught exception. |

**Returns** a `str` containing the formatted traceback if one is available, or `"ExceptionType: message"` as a fallback.

**Constraint:** must be called from inside an `except` block. `traceback.format_exc()` captures the live stack frame; calling it after the block has exited returns `"NoneType: NoneType"`.

**Example**

```python
from agency.agutil import format_exception

try:
    result = risky_operation()
except Exception as e:
    error_text = format_exception(e)
    print(error_text)
```

---

## `sigterm_as_exit(label="agency")`

Context manager that installs a `SIGTERM` handler for the duration of the `with` block, converting a plain `kill <pid>` into a normal Python exit (`SystemExit`) instead of the OS's default immediate termination.

**Signature**

```python
@contextmanager
def sigterm_as_exit(label: str = "agency") -> Generator[threading.Event, None, None]
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `label` | `str` | `"agency"` | Used only in the message printed when SIGTERM is caught, e.g. `"[agwebui] Received SIGTERM, shutting down..."`. |

**Yields** a `threading.Event` that is set if and only if SIGTERM was actually received during the block — callers can check this in a `finally` to distinguish a signal-triggered exit from a normal one (e.g. to skip an otherwise-unconditional "wait for user input" step).

**Why this exists:** Python installs no handler for `SIGTERM` by default, so a plain `kill <pid>` terminates the process immediately without ever unwinding the stack — every `finally` block and every `atexit` hook the framework relies on (live sandbox teardown in `agsandbox.py`, the tool worker pool in `agtool.py`, a webui/graphui server subprocess, ...) is skipped, exactly like `SIGKILL`. Installing this handler converts SIGTERM into `SystemExit`, so code inside the `with` block unwinds through its own `finally` blocks and reaches normal interpreter shutdown, where those hooks fire exactly as they would on any other clean exit.

`SIGKILL` itself can never be caught by any process, so there is no equivalent possible for it — recovering from a `SIGKILL`'d run relies on the framework's own self-healing (e.g. the orphaned-container reaper described in [sandbox/container.md](sandbox/container.md#orphaned-container-reaping)), not on anything a context manager can do.

**Main-thread only:** `signal.signal()` only works when called from the main thread. From any other thread, `sigterm_as_exit` is a no-op — it yields an `Event` that is simply never set, since a background thread can't rely on `KeyboardInterrupt`/Ctrl+C working there either.

**Used internally by** `agwebui.run()` and `graphui.run()` (see [agwebui.md](agwebui.md#shutdown-and-signal-handling)) so that killing an agwebui-driven run cleans up its containers, tool worker pool, and server subprocess normally. Any script that constructs and runs agents/teams **without** going through one of those two wrappers gets no such protection unless it wraps itself the same way:

**Example**

```python
from agency import sigterm_as_exit

def main():
    ag = agent(agconfig=..., agname="MyAgent")
    ag.run(my_skill, agdata(topic="..."))

with sigterm_as_exit("my_script"):
    main()
```

**Gotcha:** the previous SIGTERM handler is always restored on exit from the `with` block (normal or via the raised `SystemExit`), so nesting or repeated use is safe — but only one handler is active at a time within a thread, so an inner `sigterm_as_exit` block temporarily shadows an outer one for its duration.

---

## `_looks_like_path(s)`

Heuristic that returns `True` if a string looks like an absolute sandbox file or directory path.

**Signature**

```python
def _looks_like_path(s: str) -> bool
```

The check uses the regex `^(/[\w.\-]+)+$`, which matches strings that start with `/` and consist only of path-safe characters separated by slashes.

**Used by** `agschema` to auto-resolve path-like strings in `str`-typed output fields before passing them to tools. User code does not need to call this directly.

**Examples**

```python
_looks_like_path("/workspace/data/file.txt")  # True
_looks_like_path("just a string")             # False
_looks_like_path("/bad path/with spaces")     # False
```

**Gotcha:** the regex intentionally rejects paths with spaces or special characters (e.g., `@`, `:`). A path that contains these will not be auto-resolved by `agschema`.

---

## `_LLMIdleTimeout`

Exception raised by `_iter_batched` when the LLM stream goes silent beyond the configured timeout.

```python
class _LLMIdleTimeout(Exception): ...
```

Two situations trigger this exception:

- **Pre-first-chunk (idle timeout):** no chunk arrives within `idle_timeout` seconds of starting the stream. This typically means the upstream server is unresponsive.
- **Mid-stream timeout:** after streaming has started, no further chunk arrives within `stream_timeout` seconds. This typically means the model stalled mid-generation.

The label (`"pre-first-chunk"` or `"mid-stream"`) is included in the exception message for diagnostics.

Callers in `agllm` catch `_LLMIdleTimeout` and retry the LLM call.

---

## `_iter_batched(iterable, idle_timeout, stream_timeout)`

Drains a streaming iterable in a background thread and yields accumulated batches to the caller, with configurable timeout enforcement.

**Signature**

```python
def _iter_batched(
    iterable: Iterable[T],
    idle_timeout: float | None = None,
    stream_timeout: float | None = None,
) -> Generator[list[T], None, None]
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iterable` | `Iterable[T]` | required | The upstream stream (e.g., an LLM response iterator). |
| `idle_timeout` | `float \| None` | `None` | Seconds to wait for the **first** chunk before raising `_LLMIdleTimeout`. |
| `stream_timeout` | `float \| None` | `None` | Seconds to wait between chunks **after** streaming has started. `None` means wait indefinitely once tokens are flowing. |

**Yields** `list[T]` — each batch contains all items that accumulated in the queue during a 100 ms sleep interval (`_BATCH_INTERVAL_S`).

**Raises**

- `_LLMIdleTimeout` — when a timeout threshold is exceeded.
- Any exception thrown by the background thread is re-raised in the calling thread when the sentinel is reached.

**How it works**

1. A daemon thread iterates `iterable` and puts each item into a `queue.SimpleQueue`.
2. The main thread wakes every 100 ms, drains everything currently in the queue, and yields it as a batch.
3. Between drains the GIL is fully released, so GIL acquisitions drop from O(items) to O(items / avg\_batch\_size).
4. A separate 1-second check interval (`_IDLE_CHECK_INTERVAL_S`) governs timeout polling without blocking indefinitely on `queue.get`.

**Example (internal usage pattern in agllm)**

```python
from agency.agutil import _iter_batched, _LLMIdleTimeout

try:
    for batch in _iter_batched(stream, idle_timeout=30.0, stream_timeout=None):
        for chunk in batch:
            process(chunk)
except _LLMIdleTimeout as e:
    # retry logic
    raise
```

**Gotcha:** `stream_timeout=None` (the default) does not apply a mid-stream deadline. Once the first token arrives, the iterator waits indefinitely for the next one. Pass an explicit value only if the model is known to stall without eventually resuming.

---

## Internal text helpers

These helpers are used within `agllm` to handle model outputs that include chain-of-thought blocks.

### `_strip_thinking(content)`

Removes `<think>…</think>` and `<thinking>…</thinking>` blocks from a string. The match is case-insensitive and spans newlines.

### `_extract_thinking(content)`

Returns the concatenated text of all thinking blocks found in `content`, joined by double newlines. Returns an empty string if no blocks are present.

---

## Internal name helpers

### `_b36_suffix(n, width=4)`

Encodes an integer as a fixed-width base-36 string (digits `0–9`, then `a–z`). Used by `agname` to generate short unique suffixes for agent IDs (e.g., `"0000"` through `"zzzz"`, giving 1,679,616 unique values per noun at width 4).
