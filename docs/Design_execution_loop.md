# Execution Loop

This document traces the complete execution path of a single `agent.run()` call from submission through to future resolution, covering every function called, every data structure touched, and how the ReAct loop and sandbox process monitoring integrate.

---

## 1. `agent.run()` — submission and future construction

**File:** `agency/agent.py` · `agent.run(skill_name, input, max_steps=10)`

The caller receives a pending `agdata` immediately. No LLM call has happened yet.

```python
prev_history = self._history                    # agdata wrapping a Future or resolved agdata
result_future:  Future[agdata] = Future()
history_future: Future[agdata] = Future()
ts_start = _ts()                                # ISO-8601 timestamp for the log

threading.Thread(target=_task, daemon=True).start()  # spawn a daemon thread for this task

self._history = agdata(_future=history_future)  # chain: next run() on this agent blocks here
return agdata(_future=result_future)            # caller gets this — pending until _task resolves it
```

**Serialization invariant.** `self._history` is replaced with a new pending `agdata` wrapping `history_future` before `run()` returns. Any subsequent `run()` on the same agent calls `prev_history._resolve()` inside `_task`, which blocks on the previous `history_future.result()`. This is how sequential calls are serialized without any explicit lock — the history chain is itself the queue.

---

## 2. `_task()` — the task thread

**File:** `agency/agent.py` · `_task()` (closure inside `agent.run()`)

Runs on a dedicated daemon thread spawned by `agent.run()`. Everything below executes on this thread.

### 2a. Input resolution

```python
prev_history._resolve()    # blocks if still pending
_resolve_input(input)      # recursively resolves any pending agdata nested inside input fields
```

`_resolve_input` walks `input._data`:
- Top-level values that are pending `agdata` → `val._resolve()`
- List elements that are pending `agdata` → `item._resolve()`

This allows the caller to pass a prior `run()` result directly as input without blocking themselves:

```python
r1 = ag.run("search", agdata(query="..."))   # pending
r2 = ag.run("summarize", agdata(text=r1))    # r1 resolved here inside _task for r2
```

### 2b. Skill lookup

```python
af = next((f for f in self.agskills if f.name == skill_name), None)
```

On failure, both futures resolve immediately with an error and unchanged history.

### 2c. Skill execution

`_task` calls `af.run()` once — there is no outer retry or monitoring loop:

```python
outer_result, outer_history, outer_delta, _tok = af.run(
    self.llm_config, input, prev_history,
    self.sandbox, pool, max_steps, ...,
    _ping_interval_s=agent.ping_interval_s,
    _poll_interval_s=agent.poll_interval_s,
    _agname=self.agname,
)
outer_input_tokens  = _tok[0]
outer_output_tokens = _tok[1]
```

`af.run()` returns a 4-tuple: `(result, updated_history, history_delta, (input_tokens, output_tokens))`. Background process monitoring is handled inside `af.run()` — see Section 3.

### 2d. Exception handling

The entire body of `_task` from input resolution through skill execution is wrapped in a single `try / except / finally`:

```python
try:
    prev_history._resolve()
    _resolve_input(input)
    ...
    outer_result, outer_history, outer_delta, _tok = af.run(...)

except Exception as exc:
    outer_result  = agdata(error=_fmt_exc(exc))
    outer_history = prev_history
    outer_delta   = []

finally:
    self._set_ui_state("finished")
    if self.sandbox is not None:
        _remove_offloaded_fields(...)
        if self.sandbox._gpu_id is not None:
            pool.release_gpu(self.sandbox._gpu_id)
        # commit checkpoint, then destroy container
        try:
            if self.sandbox.commit(_ckpt_tag):
                self._checkpoint = _ckpt_tag
        except Exception:
            pass
        self.sandbox.destroy()
        self.sandbox = None
```

The `finally` block always runs regardless of success or failure. It releases any GPU held by the sandbox, commits the container as a checkpoint image (so the next run can resume from this state), and destroys the container. This guarantees no container is left running even if an unexpected exception escapes `af.run()`.

---

## 3. ReAct loop — `agskill.run()`

**File:** `agency/agskill.py`

Returns `(result: agdata, updated_history: agdata, history_delta: list[dict], token_counts: tuple[int, int])`.

`history_delta` contains only the new messages added during this skill (system + messages after the previous history). `token_counts` is `(total_input_tokens, total_output_tokens)` accumulated across all LLM calls in this skill run.

### 3a. Tool list resolution

```python
active_tools = self.tools if self.tools is not None else agent_tools
tool_map = {t.name: t for t in active_tools}
openai_tools = [t.to_openai_tool() for t in active_tools] or None
```

### 3b. Input validation

```python
if self.input_schema is not None and not _is_continuation:
    errors = self._check_schema(input, self.input_schema)
    if errors:
        return agdata(error=f"input schema error: {errors}"), history, [sys_msg], (0, 0)
```

Skipped when `_is_continuation=True`.

### 3c. Message list construction

```python
history_msgs = list(history._data.get("messages", []))
n_before = len(history_msgs)

messages = (
    [{"role": "system", "content": self._build_system_prompt()}]
    + history_msgs
    + [{"role": "user", "content": input.to_json()}]
)
```

The system message is prepended on every call but never stored — `updated_history = agdata(messages=messages[1:])` strips it before returning.

### 3d. Per-step: inbox drain

At the top of each step, the agent drains `agent._inbox` — a `queue.Queue[str]` that `agUI` (or any external caller) can push messages into:

```python
if _inbox_fn:
    while True:
        msg = _inbox_fn()
        if msg is None:
            break
        messages.append({"role": "user", "content": msg})
        had_inbox = True
```

When `had_inbox` is True and the LLM replies with text (no tool calls), output schema validation is skipped — the LLM is in mid-conversation with the user, not producing a final answer.

### 3e. Pre-call compaction

Before each LLM call, a character-based token estimate is compared against `_context_limit`. If the estimate exceeds the threshold, compaction runs and shrinks the message list before the call. See [compaction.md](compaction.md).

### 3f. LLM call and retry

```python
llm_result = _llm_call(kwargs, llm_config, messages, _timeout_attempt, ...)
if llm_result.should_retry:
    _timeout_attempt = llm_result.next_timeout_attempt
    continue
if not llm_result.ok:
    return agdata(error=f"LLM connection error after 5 attempts: {llm_result.conn_error}"), ...
```

See [LLM timeout and retry](#llm-timeout-and-retry) below for the full backoff sequence.

### 3g. Post-call compaction

After the call, the actual `prompt_tokens` from the API usage chunk is compared against `_context_limit`. A second compaction can run here if needed (using the exact count instead of the estimate).

### 3h. Tool-call branch

```python
if msg_dict.get("tool_calls"):
    _dispatch_tools(msg_dict["tool_calls"], tool_map, messages, sandbox, ...)
```

For each tool call in `_dispatch_tools`:
1. If the LLM produced malformed JSON arguments (truncated generation, Python repr, etc.), arguments are silently coerced to `"{}"` to prevent vLLM from crashing on the next history replay.
2. If the tool is unknown, `{"error": "unknown tool: <name>"}` is injected as the tool result — the LLM can recover.
3. If `need_sandbox=True`, the sandbox is committed to a pre-call checkpoint image before the tool executes.
4. The tool function runs in a `ProcessPoolExecutor` worker with a configurable timeout (default `TOOL_TIMEOUT_S = 30 s`, overridable per call via a `"timeout"` key in arguments).
5. If the tool raises or returns an error and a pre-call checkpoint was taken, the sandbox is restored to the checkpoint image and `"workspace_reverted"` is appended to the error message.
6. If the result exceeds `_TOOL_OUTPUT_OFFLOAD_CHARS` (80 000 chars), it is saved to a file inside the sandbox and replaced with a short reference note.

After all tool calls in the response are dispatched, the loop goes back to step 3d.

### 3i. Final-answer branch

```python
else:
    if had_inbox:
        continue   # LLM is mid-conversation with the user — not a final answer yet

    far = self._parse_final_answer(...)   # parse JSON, validate output_schema
    if far.kind == "retry":
        output_schema_retries_left -= 1
        messages.append(far.correction_msg)
        continue
    if far.kind != "error" and sandbox is not None:
        proc_msg = _wait_for_processes(sandbox, ..., _ping_interval_s, _poll_interval_s)
        if proc_msg is not None:
            messages.append({"role": "user", "content": proc_msg})
            continue   # re-enter loop — LLM will act on process status
    return far.return_tuple
```

Output validation retry: if `output_schema` is set and the response fails validation, a correction message is injected and the loop retries up to `max_output_schema_retries` times (default 10). If all retries are exhausted, the skill returns `agdata(error="output schema error after retries: ...")`.

Process monitoring: after a valid final answer, if the sandbox has live background processes, `_wait_for_processes` polls until all PIDs exit (or `_ping_interval_s` elapses) and returns a user-facing message. That message is appended to the conversation and the loop continues — the LLM sees either "processes completed" or "processes still running" and reacts accordingly. The loop only truly exits when the sandbox is clean.

### 3j. `max_steps` exceeded

If the loop runs for `max_steps` (default `AGSKILL_REACT_MAX_STEPS = 4096`) iterations without reaching a clean final answer:

```python
return agdata(error="max_steps exceeded"), updated_history, ..., (input_tokens, output_tokens)
```

### 3k. Return value construction

```python
updated_history = agdata(messages=messages[1:])         # strip system message
history_delta   = [messages[0]] + messages[1:][n_before:]   # system + new messages only
```

---

## LLM timeout and retry

Each LLM call is protected by an idle watchdog that doubles its deadline on every retry:

| Attempt | Deadline |
|---------|----------|
| 0 | 60 s |
| 1 | 120 s |
| 2 | 240 s |
| 3 | 480 s |
| 4 | 960 s |

The watchdog runs in the main thread: it polls a queue fed by the streaming drain thread and raises `_LLMIdleTimeout` if no chunk arrives within the deadline. Unlike `httpx.ReadTimeout`, this approach works even when the underlying `ssl.read()` is blocked indefinitely (CLOSE-WAIT state). On timeout, `client.close()` is called best-effort to unblock the drain thread.

`ssl.SSLError`, `OSError`, and `httpx.TransportError` raised during streaming are caught by the same retry block and follow the same exponential backoff. All four exception types indicate a dead or dropped connection and are treated identically.

On attempt N < 4: log `LLM ✗`, increment `_timeout_attempt`, `continue` the ReAct loop (the next iteration re-creates the client and retries with a longer deadline).

On attempt 4 (all exhausted): return `agdata(error="LLM connection error after 5 attempts: ...")` — the skill fails without raising.

---

## 4. Post-loop: resource release, logging, future resolution

```python
finally:
    ...  # sandbox commit + destroy (see Section 2d)

# after finally:
self.log._record(skill_name, ts_start, ts_end, input_dict, result_dict,
                 history_len, history_before=..., history_delta=...,
                 input_tokens=..., output_tokens=...)
agent._add_global_tokens(outer_input_tokens, outer_output_tokens)

self._snapshot_messages = list(outer_history._data.get("messages", []))
result_future.set_result(outer_result)    # unblocks caller's field access

# Post-skill history pruning — trims old oversized tool outputs before
# unblocking the next run() so history stays bounded.
pruned_msgs = _prune_tool_outputs(outer_history._data.get("messages", []))
if pruned_msgs is not outer_history._data.get("messages", []):
    outer_history = agdata(messages=pruned_msgs)

history_future.set_result(outer_history)  # unblocks next run() on same agent
```

Note: `result_future` is resolved *before* `history_future`. The caller unblocks as soon as the result is ready; the next `run()` on the same agent blocks a little longer while pruning completes, ensuring history is always clean when the next skill begins.

---

## 5. Complete call graph

```
caller thread                        task thread (daemon)
──────────────────────────────────────────────────────────────────────────────
agent.run(skill, input)
  ├─ prev_history = self._history
  ├─ result_future, history_future = Future(), Future()
  ├─ self._history = agdata(_future=history_future)
  ├─ Thread(target=_task, daemon=True).start() ──────────────────▶ _task()
  └─ return agdata(_future=result_future)                           │
                                                                   prev_history._resolve()
caller.field ────────────── blocks ────────────────────────────┐   _resolve_input(input)
                                                               │   │
                                                               │   skill lookup
                                                               │   │
                                                               │   agskill.run()
                                                               │   ├─ input validation
                                                               │   ├─ message construction
                                                               │   └─ for _ in range(max_steps)
                                                               │        inbox drain
                                                               │        pre-call compaction
                                                               │        LLM call  ──► retry on timeout/SSL/OS error
                                                               │        post-call compaction
                                                               │        ├─ tool calls?
                                                               │        │   malformed args → coerce "{}"
                                                               │        │   unknown tool → error msg
                                                               │        │   need_sandbox? → commit checkpoint
                                                               │        │   tool.fn(agdata)
                                                               │        │   → sandbox.exec()
                                                               │        │   error? → restore checkpoint
                                                               │        │   large output? → offload to file
                                                               │        │   loop back
                                                               │        └─ final answer?
                                                               │            had_inbox? → continue
                                                               │            parse JSON
                                                               │            output schema check
                                                               │            retry? → inject correction, continue
                                                               │            error? → return
                                                               │            _wait_for_processes()
                                                               │            processes running? → inject msg, continue
                                                               │            sandbox clean → return
                                                               │   │
                                                               │   finally: GPU release, checkpoint commit, destroy
                                                               │   log._record()
                                                               │   result_future.set_result()
                                                               │   history pruning
                                                               │   history_future.set_result()
                                                               │   │
caller.field ◀─────────────── unblocks ────────────────────────┘   │
next run()   ◀─────────────── unblocks ────────────────────────────┘
```

---

## 6. Error propagation

| Where it occurs | How it surfaces |
|---|---|
| Skill not found | Both futures resolved immediately with error / unchanged history |
| Input schema failure | `agskill.run()` returns `agdata(error=...)` before any LLM call |
| Exception in `_task` | Caught by outer `except`; `outer_result = agdata(error=str(exc))`; `finally` still runs |
| Unknown tool name | `{"error": "unknown tool: <name>"}` injected as tool result; LLM recovers |
| Malformed tool arguments | Arguments silently coerced to `"{}"` before calling the tool |
| Tool `fn` raises | Per-tool catch; `{"error": str(e)}` (+ `"workspace_reverted"` if checkpoint taken) injected; LLM recovers |
| Tool timeout | Same as tool `fn` raises — timeout counts as a tool error |
| LLM timeout / SSL / OS error | `_llm_call` retries with exponential backoff (5 attempts); on exhaustion returns `agdata(error=...)` |
| Output schema failure after retries | `agskill.run()` returns `agdata(error="output schema error after retries: ...")` |
| `max_steps` exceeded | `agskill.run()` returns `agdata(error="max_steps exceeded")` |

Accessing any field on an error `agdata` via `__getattr__` raises `AgError`. `.error` and `.is_error()` are safe accessors.
