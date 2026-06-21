# Design: Error Handling

This document covers every try/except/finally block, retry loop, and error emission in the framework, along with how errors propagate from their origin up to the caller.

---

## Error Propagation Overview

The framework has four distinct propagation paths:

```
Tool call fails
  └─► agdata(error=...)  ──► LLM sees the error as a tool result message
                                  and may retry or report failure in its output

LLM connection fails
  └─► agskill retry loop (up to 5 attempts, exponential backoff)
        └─► after 5 failures: agdata(error=...) returned from agskill.run()
              └─► agent._task() catches it, sets result_future
                    └─► caller's agdata._resolve() re-raises or returns error agdata

Output schema validation fails
  └─► correction message appended to conversation, loop continues
        └─► after max_output_schema_retries: agdata(error=...) from agskill.run()

Uncaught exception inside agent._task()
  └─► except Exception: agdata(error=fmt_exc(exc))  ←── no re-raise
        └─► result_future resolved with error agdata
              └─► caller sees error in agdata.error field

Uncaught exception inside agteam._async_run()
  └─► future.set_exception(exc)  ←── stored, not raised yet
        └─► agsync() calls team._run_future.result()
              └─► single failure: re-raised directly
              └─► multiple failures: raised as ExceptionGroup
```

Errors almost never propagate as Python exceptions between threads. The canonical representation is `agdata(error=str(...))`: callers inspect the `error` field rather than catching exceptions. The only place Python exceptions cross thread boundaries is in `agteam`, where `future.set_exception()` defers the raise to `agsync()`.

---

## `agency/agskill.py` — LLM retry loop and output validation

### LLM connection retry

**Location:** `_llm_call()` (line ~424–509) and the ReAct loop (line ~1148–1165)

**What is caught:** `_LLMIdleTimeout`, `ssl.SSLError`, `OSError`, `httpx.TransportError` — all transient network failures during streaming.

**Backoff sequence:** `_TIMEOUT_SEQUENCE = [60, 120, 240, 480, 960]` seconds — five attempts with doubling read timeouts.

**Handler:**
1. `_llm_call()` catches the exception and returns `_LLMCallResult(should_retry=True, conn_error=exc)`.
2. The ReAct loop in `agskill.run()` checks `llm_result.should_retry`:
   - Emits `{"type": "llm_retry", "error": str(exc), "attempt": N}` via `_full_history_fn`.
   - Continues to next iteration (no sleep — the next timeout is longer to give the server time).
3. After five consecutive failures, returns `_LLMCallResult(ok=False)`.
4. The ReAct loop emits `{"type": "llm_error", "error": "LLM connection error after 5 attempts: ..."}` and returns `agdata(error=...)` to `agent._task()`.

**Propagation:** `agdata(error=...)` → `agent._task()` → `result_future.set_result(error_agdata)` → caller's `agdata._resolve()`.

### Output schema validation retry

**Location:** `agskill._parse_final_answer()` and the ReAct loop (line ~977–1008, 1199–1206)

**What is caught:** JSON parse errors and schema/validator violations on the LLM's final answer.

**Budget:** `max_output_schema_retries` (default 10, configurable per skill).

**Handler:**
1. On each validation failure a correction message is appended to `messages` and the loop continues.
2. Once `output_schema_retries_left` reaches 0, returns `agdata(error="output schema error after retries: ...")`.

**Propagation:** Same path as LLM error — `agdata(error=...)` flows back through `agent._task()`.

---

## `agency/agent.py` — skill execution wrapper

### Main skill try/except/finally

**Location:** `agent.run()` → `_task()` (line ~601–718)

**Structure:**
```python
try:
    # resolve input, create sandbox, run agskill
    outer_result = af.run(...)
except Exception as exc:
    # swallow — convert to agdata
    outer_result = agdata(error=_fmt_exc(exc))
    outer_history = prev_history
finally:
    # always runs, even on exception:
    _remove_offloaded_fields(...)
    pool.release_gpu(...)     # GPU released even if skill crashed
    sandbox.commit(...)       # checkpoint best-effort
    sandbox.destroy()         # container torn down no matter what
    sandbox = None
```

**What is caught:** Any unhandled exception from the skill (including `agskill.run()` returning an error agdata is NOT an exception — only genuine throws reach here).

**Handler:** Formats the exception with full traceback via `_fmt_exc(exc)`, stores it in `outer_result`, logs `SKILL ✗` to the terminal, then emits `{"type": "skill_error", "skill": ..., "error": ...}` via `_append_full_history()` after the finally block (line 728).

**Propagation:** `result_future.set_result(outer_result)` — the error stays inside `agdata.error`; the future resolves successfully (no exception crossing thread boundary).

### Post-skill logging errors

**Location:** line ~732–756

**What is caught:** Any exception from `log._record()` or token tracking.

**Handler:** Logged to terminal as `[log error]`; does not affect `result_future`.

### History pruning errors

**Location:** line ~766–774

**What is caught:** Any exception from `_prune_tool_outputs()`.

**Handler:** Logged as `PRUNE ✗`; `history_future` still resolves with the un-pruned history.

### WebUI / agui push errors

**Location:** `_push_live_messages()` (line ~547–563)

**What is caught:** Any exception from WebUI or TUI state updates.

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
- `subprocess.TimeoutExpired` → returns `agdata(error=..., returncode=-1)` without raising; the LLM sees the timeout as a tool result.

**Propagation:** `exec()` is called by `agtool` via the process pool; timeout and runtime errors surface as `agdata(error=...)` to the ReAct loop, which appends them as tool result messages.

---

## `agency/agtool.py` — tool execution

### Direct function call (no isolation)

**Location:** line ~147–151

**What is caught:** Any `Exception` from the tool function.

**Handler:** Returns `agdata(error=_fmt_exc(exc))`; execution continues.

### Process-pool execution

**Location:** line ~160–176

**What is caught:**
- `concurrent.futures.TimeoutError` → returns `agdata(error="tool timed out after {N}s")`.
- `concurrent.futures.process.BrokenProcessPool` → resets `_pool = None` so the next call creates a fresh pool; returns `agdata(error="tool worker process died unexpectedly")`.

**Propagation:** All tool errors become `agdata(error=...)` appended to the conversation as a `tool` role message. The LLM reads the error and decides how to proceed.

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
- `FileNotFoundError` → `agdata(error="Not found: {path}")`
- Generic `Exception` → `agdata(error=_fmt_exc(e))`

### `edit.py`
- `FileNotFoundError` on pre-read → `agdata(error=...)`
- `ValueError` / `OSError` on write → `agdata(error=_fmt_exc(e))`

### `webfetch.py`
- `httpx.HTTPStatusError` → `agdata(error=f"HTTP {status}: {url}\n{details}")`
- Generic `Exception` → `agdata(error=_fmt_exc(e))`

### `human.py`
- `EOFError` from `input()` → puts `_TIMEOUT_REPLY` in the reply queue
- `queue.Empty` from `q.get(timeout=timeout_s)` → logs timeout and returns `_TIMEOUT_REPLY`

All tool errors return `agdata(error=...)`. This is appended to the conversation as a `tool` role message; the LLM decides whether to retry the tool, try a different approach, or report failure in its final answer.

---

## `agency/agcompaction.py` — context compaction

**Location:** model info retrieval (line ~73–79), vLLM tokenize call (line ~110–127)

**What is caught:** Any exception from Anthropic SDK metadata or vLLM tokenize endpoint.

**Handler:** Swallowed; falls back to character-count-based token estimate. Compaction continues with a less precise estimate rather than crashing.

---

## `agency/agtype.py` — large-value offloading

**Location:** sandbox file write (line ~143–147), sandbox file read (line ~157–161)

**What is caught:** Any `Exception` from writing/reading oversized fields to the sandbox.

**Handler:** Swallowed; returns the original value unchanged. The LLM receives the full string inline instead of a file path.

---

## `agency/agui.py` — TUI

**Location:** Worker thread (line ~562–581), main app loop (line ~587–590), various rendering helpers

**What is caught:** `Exception` from user function → traceback printed via `ui.add_log()`; `KeyboardInterrupt` in main loop → worker thread joined with 10 s timeout.

**Propagation:** Exceptions in user code are printed and swallowed; the TUI exits cleanly.

---

## Retry Budget Summary

| Site | File | Max retries | Backoff |
|---|---|---|---|
| LLM connection | agskill.py:389 | 5 | Doubling timeout: 60→120→240→480→960 s |
| Output schema validation | agskill.py:721 | 10 (configurable) | None — immediate correction message |
| Docker name conflict | agsandbox.py:252 | 3 | Linear: 0.5 s × (attempt+1) |
| Tool execution | agtool.py:160 | 0 (single attempt) | Pool reset on `BrokenProcessPool` |

## Event Emission Summary

| Event type | Emitted from | Meaning |
|---|---|---|
| `skill_start` | agent.py:667 | Skill began executing |
| `skill_error` | agent.py:728 | Skill returned an error or threw |
| `llm_retry` | agskill.py:1157 | LLM call failed transiently; retrying |
| `llm_error` | agskill.py:1164 | LLM failed after all retry attempts |

All four events are written via `_append_full_history()`, which appends to the in-memory `_full_history` list and the per-agent JSONL file, then immediately pushes the updated snapshot to the web UI.
