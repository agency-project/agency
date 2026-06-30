# Design: Error Handling

This document covers every try/except/finally block, retry loop, and error emission in the framework, along with how errors propagate from their origin up to the caller.

---

## Error Propagation Overview

The framework has four distinct propagation paths:

```
Tool call fails
  └─► agerror(...)  ──► LLM sees the error as a tool result message
                              and may retry or report failure in its output

LLM connection fails
  └─► agskill retry loop (up to 5 attempts, exponential backoff)
        └─► after 5 failures: agerror(...) returned from agskill.execute_react()
              └─► _task() closure in agskill.run() catches it, sets result_future
                    └─► caller's agdata.resolve_input_dependencies() returns the agerror

Output schema validation fails
  └─► correction message appended to conversation, loop continues
        └─► after max_output_schema_retries: agerror(...) from agskill.execute_react()

Uncaught exception inside _task() closure in agskill.run()
  └─► except Exception: agerror(format_exception(exc))  ←── no re-raise
        └─► result_future resolved with agerror
              └─► caller checks isinstance(result, agerror)

Uncaught exception inside agteam._async_run()
  └─► future.set_exception(exc)  ←── stored, not raised yet
        └─► agsync() calls team._run_future.result()
              └─► single failure: re-raised directly
              └─► multiple failures: raised as ExceptionGroup
```

Errors almost never propagate as Python exceptions between threads. The canonical representation is `agerror(message)`: callers check `isinstance(result, agerror)` and read `result.error` rather than catching exceptions. Constructing an `agerror` immediately emits a log line to stderr (and to the web UI when active) — errors are always visible at the point they are created. The only place Python exceptions cross thread boundaries is in `agteam`, where `future.set_exception()` defers the raise to `agsync()`.

---

## `agency/agskill.py` — LLM retry loop and output validation

### LLM connection retry

**Location:** `_llm_call()` and the ReAct loop in `agskill.execute_react()`

**What is caught:** `_LLMIdleTimeout`, `ssl.SSLError`, `OSError`, `httpx.TransportError` — all transient network failures during streaming.

**Timeout constants** (defined at the top of `_llm_call()`):

| Constant | Value | Role |
|---|---|---|
| `_LLM_MAX_RETRIES` | 5 | Total attempts before giving up |
| `_LLM_IDLE_TIMEOUT` | 60 s | Max wait for the **first chunk** — detects a dead server |
| `_LLM_STREAM_TIMEOUT` | 1800 s | Max gap between chunks **mid-stream** — detects a frozen server after streaming started |

`_LLM_IDLE_TIMEOUT` and `_LLM_STREAM_TIMEOUT` are passed as separate parameters to `_iter_batched()`. The distinction matters: a pre-first-chunk timeout means the server never acknowledged the request (retry makes sense); a mid-stream timeout means the model was actively generating and then stalled (also retried, but much rarer in practice — 1800 s gives generous headroom for long reasoning chains on large models).

**Handler:**
1. `_llm_call()` catches the exception and returns `_LLMCallResult(should_retry=True, conn_error=exc)`.
2. The ReAct loop in `agskill.execute_react()` checks `llm_result.should_retry`:
   - Emits `{"type": "llm_retry", "error": str(exc), "attempt": N}` via `full_history_fn`.
   - Sleeps 2 s (allows SSL teardown to complete before reconnecting — see Known failure modes).
   - Continues to the next attempt.
3. After `_LLM_MAX_RETRIES` consecutive failures, returns `_LLMCallResult(ok=False)`.
4. The ReAct loop emits `{"type": "llm_error", "error": "LLM connection error after 5 attempts: ..."}` and returns `agerror(...)` to the `_task()` closure in `agskill.run()`.

**Propagation:** `agerror(...)` → `_task()` closure in `agskill.run()` → `result_future.set_result(error_result)` → caller checks `isinstance(result, agerror)`.

### Output schema validation retry

**Location:** Output validation logic in the ReAct loop of `agskill.execute_react()` (line ~977–1008, 1199–1206)

**What is caught:** JSON parse errors and schema/validator violations on the LLM's final answer.

**Budget:** `max_output_schema_retries` (default 10, configurable per skill).

**Handler:**
1. On each validation failure a correction message is appended to `messages` and the loop continues.
2. Once `output_schema_retries_left` reaches 0, returns `agerror("output schema error after retries: ...")`.

**Propagation:** Same path as LLM connection error — `agerror(...)` flows back through the `_task()` closure in `agskill.run()`.

---

## `agency/agskill.py` — `_task()` closure (skill execution wrapper)

### Main skill try/except/finally

**Location:** `_task()` closure inside `agskill.run()` (line ~601–718)

**Structure:**
```python
try:
    # resolve input, create sandbox, run agskill
    outer_result = af.execute_react(...)
except Exception as exc:
    # swallow — convert to agerror
    outer_result = agerror(format_exception(exc))
    outer_ctx = prev_ctx
finally:
    # always runs, even on exception:
    _remove_offloaded_fields(...)
    pool.release_gpu(...)     # GPU released even if skill crashed
    ag.sandbox.stop(commit=True)  # checkpoint and tear down container no matter what
    sandbox = None
```

**What is caught:** Any unhandled exception from the skill (note: `agskill.execute_react()` returning an error agdata is NOT an exception — only genuine throws reach here).

**Handler:** Formats the exception with full traceback via `format_exception(exc)`, stores it in `outer_result` as an `agerror`, logs `SKILL ✗` to the terminal, then emits `{"type": "skill_error", "skill": ..., "error": ...}` via `_append_full_history()` after the finally block (line 728).

**Propagation:** `result_future.set_result(outer_result)` — the error is carried in an `agerror`; the future resolves successfully (no exception crossing thread boundary). Callers check `isinstance(result, agerror)`.

### Post-skill logging errors

**Location:** line ~732–756

**What is caught:** Any exception from `log._record()` or token tracking.

**Handler:** Logged to terminal as `[log error]`; does not affect `result_future`.

### History pruning errors

**Location:** line ~766–774

**What is caught:** Any exception from `_prune_tool_outputs()`.

**Handler:** Logged as `PRUNE ✗`; `history_future` still resolves with the un-pruned history.

### WebUI push errors

**Location:** `_push_live_messages()` (line ~547–563)

**What is caught:** Any exception from WebUI state updates.

**Handler:** Swallowed silently — UI sync failures never affect execution.

---

## `agency/agsandbox.py` — container lifecycle

### Docker name-conflict retry

**Location:** `_run_with_conflict_retry()` (line ~244–276)

**What is caught:** `subprocess.CalledProcessError` where stderr contains "already in use" or "Conflict".

**Retries:** Up to 3 attempts; sleeps `0.5 × (attempt + 1)` seconds between tries (0.5 s, 1.0 s, 1.5 s).

**Handler:**
- Removes the stale container (`docker rm -f`) and retries.
- If another process claimed the container first, raises `_ContainerAlreadyRunning`; caller (`_ensure_started()`) catches this and reuses the existing container.
- After 3 failures, raises `RuntimeError`.

### Subprocess errors

**Location:** `exec()` and related helpers (line ~315–351)

**What is caught:**
- `subprocess.CalledProcessError` → re-raised as `RuntimeError` with stderr attached.
- `subprocess.TimeoutExpired` → returns `agerror(...)` with `returncode=-1` without raising; the LLM sees the timeout as a tool result.

**Propagation:** `exec()` is called by `agtool` via the process pool; timeout and runtime errors surface as `agerror(...)` to the ReAct loop, which appends them as tool result messages.

---

## `agency/agtool.py` — tool execution

### Direct function call (no isolation)

**Location:** line ~147–151

**What is caught:** Any `Exception` from the tool function.

**Handler:** Returns `agerror(format_exception(exc))`; execution continues.

### Process-pool execution

**Location:** line ~160–176

**What is caught:**
- `concurrent.futures.TimeoutError` → returns `agerror("tool timed out after {N}s")`.
- `concurrent.futures.process.BrokenProcessPool` → resets `_pool = None` so the next call creates a fresh pool; returns `agerror("tool worker process died unexpectedly")`.

**Propagation:** All tool errors become `agerror(...)` appended to the conversation as a `tool` role message. The LLM reads the error and decides how to proceed.

---

## `agency/agteam.py` — team background thread

### Team run exception capture

**Location:** `_async_run()` → `_task()` (line ~159–170)

**What is caught:** Any `Exception` from the user-defined `run()` method.

**Handler:**
1. Prints full traceback to stderr immediately.
2. Calls `future.set_exception(exc)` — stores the exception without raising.
3. `finally` always resets `_active_team` context variable.

**Propagation:** Exception is deferred inside the `Future`. It is only raised when `agsync()` calls `team._run_future.result()`.

---

## `agency/agsync.py` — multi-team join

**Location:** `agsync()` (line ~63–82)

**What is caught:** `Exception` from each `team._run_future.result()`.

**Handler:** Collects all failures without stopping — every team is joined before any exception is raised. Ensures no team is abandoned mid-run.

**Propagation:**
- Single failure: `raise errors[0]` — the original exception propagates to the user's top-level script.
- Multiple failures: `raise ExceptionGroup("agsync: N team(s) failed", errors)` — all failures bundled together.

---

## `agency/agresources.py` — GPU and resource detection

**Location:** GPU detection (line ~20–37), memory detection (line ~104–120), GPU marker spawn/termination (line ~196–211)

**What is caught:** Any `Exception` from CLI tools (`nvidia-smi`, `rocm-smi`) or `/proc/meminfo` reads.

**Handler:** Swallowed — returns empty GPU list or safe memory fallback (4096 MB). Framework runs without GPU acceleration rather than crashing.

**Double-release guard:** `release_gpu()` wraps `sem.release()` in `try/except ValueError` (line 236) to prevent crashes if a GPU is released twice.

---

## `agency/agwebui/__init__.py` — web server process

**Location:** Health-check polling (line ~99–105), resource emission (line ~112–118), server termination (line ~124–127), user function execution (line ~132–136)

**What is caught:**
- Health check: swallowed per attempt; after 50 retries emits a warning.
- Server termination: falls back to `proc.kill()` if graceful wait times out.
- User function: prints traceback to stderr.

---

## `agency/agwebui/emitter.py` — event file writes

**Location:** `push_messages()` (line ~119–122), reply file cleanup (line ~154–157)

**What is caught:**
- JSON serialization failure: skips emission rather than writing corrupt data.
- `reply_file.unlink()`: swallowed — stale reply files are harmless.

---

## `agency/tools/` — individual tools

### `read.py`
- `FileNotFoundError` → `agerror("Not found: {path}")`
- Generic `Exception` → `agerror(format_exception(e))`

### `edit.py`
- `FileNotFoundError` on pre-read → `agerror(...)`
- `ValueError` / `OSError` on write → `agerror(format_exception(e))`

### `webfetch.py`
- `httpx.HTTPStatusError` → `agerror(f"HTTP {status}: {url}\n{details}")`
- Generic `Exception` → `agerror(format_exception(e))`

### `human.py`
- `EOFError` from `input()` → puts `_TIMEOUT_REPLY` in the reply queue
- `queue.Empty` from `q.get(timeout=timeout_s)` → logs timeout and returns `_TIMEOUT_REPLY`

All tool errors return `agerror(...)`. This is appended to the conversation as a `tool` role message; the LLM decides whether to retry the tool, try a different approach, or report failure in its final answer.

---

## `agency/agllm.py` — context compaction

**Location:** `maybe_compact()`, `compact()`, `_prune_tool_outputs()` — model info retrieval (line ~73–79), vLLM tokenize call (line ~110–127)

**What is caught:** Any exception from Anthropic SDK metadata or vLLM tokenize endpoint.

**Handler:** Swallowed; falls back to character-count-based token estimate. Compaction continues with a less precise estimate rather than crashing.

---

## `agency/agtype.py` — large-value offloading

**Location:** sandbox file write (line ~143–147), sandbox file read (line ~157–161)

**What is caught:** Any `Exception` from writing/reading oversized fields to the sandbox.

**Handler:** Swallowed; returns the original value unchanged. The LLM receives the full string inline instead of a file path.

---

## Known failure modes

### `ssl.SSLError: [SSL: WRONG_VERSION_NUMBER]` on LLM retry

**Root cause:** A race between the LLM drain thread and `client.close()` corrupts process-wide OpenSSL state.

`_iter_batched()` spawns a daemon thread that blocks inside `ssl.read()` consuming the streaming response. When `_LLMIdleTimeout` fires in the main thread, `client.close()` is called to unblock the drain thread. This sends a TLS `close_notify` alert and tears down the socket while the drain thread is still mid-`ssl.read()`. The result is an abrupt SSL teardown while the underlying connection is in active use. OpenSSL's internal session state is corrupted — not just for that connection but process-wide. Any new `ssl.create_default_context()` call in the same process immediately after this may fail with `[X509] PEM lib` (unable to load the CA bundle from the already-corrupted context), and new connections to the server fail with `WRONG_VERSION_NUMBER` (the server's SSL stack receives garbled data and expects a different protocol version).

**Reproduced** with direct vLLM+SSL (no proxy, no nginx):
```
delay=0.0s → ok=2  WRONG_VERSION=0  [X509] PEM lib=3   (5 concurrent abort→retry)
delay=1.0s → ok=5  all clean
delay=2.0s → ok=5  all clean
```

**Fix:** `agskill.py` adds `time.sleep(2)` in the retry loop after returning `should_retry=True`, giving the drain thread time to fully exit and the server's SSL layer time to complete teardown before the next connection is attempted. A 1 s delay is sufficient; 2 s is used for margin.

**Symptom pattern:**
- Error appears on the *second* LLM call in a retry sequence, not the first.
- First call times out (idle timeout) → first `client.close()` → second call gets `WRONG_VERSION_NUMBER` or `[X509] PEM lib`.
- The error is transient: a longer delay between retries eliminates it entirely.

**What it is NOT:**
- Not an nginx/proxy timeout (reproduced without any proxy).
- Not a server-side SSL misconfiguration (valid certificate; clean connections succeed).
- Not a stale keep-alive (httpx recovers from stale connections silently).

### `_LLMIdleTimeout` firing mid-stream on large models

**Root cause:** The idle timer was previously a single value applied both before and after the first chunk. On a 122B parameter model, generation of a complex response can take 280 s+ end-to-end, and inter-chunk gaps of several seconds are normal under concurrent load. With the old uniform 60 s timeout, any quiet period longer than 60 s mid-stream fired the timeout, discarded all tokens received so far, and restarted the entire LLM call from scratch.

**Fix:** `_iter_batched()` now takes two separate timeouts (`idle_timeout` for pre-first-chunk, `stream_timeout` for mid-stream). `_LLM_IDLE_TIMEOUT = 60 s` remains tight to detect dead servers quickly. `_LLM_STREAM_TIMEOUT = 1800 s` gives long-running generations generous headroom. A mid-stream timeout discards partial output and retries the full call, same as before — but this is now far less likely to fire spuriously.

**Diagnosed via mitmproxy:** All flows showed `TTFB ≈ 0 s` (vLLM returns `200 OK` headers immediately before streaming chunks). The timeouts were not caused by network drops or SSL issues — the server was simply generating and the inter-chunk gap exceeded the old threshold.

### `peer closed connection without sending a complete message body` / `incomplete chunked read`

Could not be reproduced with direct vLLM+SSL under normal conditions. This error is distinct from the SSL corruption above. The most likely cause is a vLLM server-side crash during generation (OOM, CUDA error, or process killed), which closes the connection abruptly mid-stream. The client receives an incomplete chunked response and raises the error. No keep-alive or SSL mechanism is involved. Diagnosing further requires server-side logs from a live failure.

---

## Retry Budget Summary

| Site | File | Max retries | Backoff |
|---|---|---|---|
| LLM connection | agskill.py | 5 | Fixed 60 s idle (pre-first-chunk) + 1800 s stream (mid-stream) + 2 s sleep before retry |
| Output schema validation | agskill.py:721 | 10 (configurable) | None — immediate correction message |
| Docker name conflict | agsandbox.py:252 | 3 | Linear: 0.5 s × (attempt+1) |
| Tool execution | agtool.py:160 | 0 (single attempt) | Pool reset on `BrokenProcessPool` |

## Event Emission Summary

| Event type | Emitted from | Meaning |
|---|---|---|
| `skill_start` | agskill.py:299 | Skill began executing |
| `skill_error` | agskill.py:328 | Skill returned an error or threw |
| `llm_retry` | agskill.py:1157 | LLM call failed transiently; retrying |
| `llm_error` | agskill.py:1164 | LLM failed after all retry attempts |

All four events are written via `_append_full_history()`, which appends to the in-memory `_full_history` list and the per-agent JSONL file, then immediately pushes the updated snapshot to the web UI.
