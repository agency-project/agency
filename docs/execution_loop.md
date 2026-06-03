# Execution Loop

This document traces the complete execution path of a single `agent.run()` call from submission through to future resolution, covering every function called, every data structure touched, and how the inner ReAct loop and outer monitoring loop integrate.

---

## 1. `agent.run()` — submission and future construction

**File:** `agency/agent.py` · `agent.run(skill_name, input, max_steps=10)`

The caller receives a pending `agdata` immediately. No LLM call has happened yet.

```python
prev_history = self._history                    # agdata wrapping a Future or resolved agdata
result_future:  Future[agdata] = Future()
history_future: Future[agdata] = Future()
ts_start = _ts()                                # ISO-8601 timestamp for the log

agent._pool.submit(_task)                       # hand _task to shared ThreadPoolExecutor

self._history = agdata(_future=history_future)  # chain: next run() on this agent blocks here
return agdata(_future=result_future)            # caller gets this — pending until _task resolves it
```

**Serialization invariant.** `self._history` is replaced with a new pending `agdata` wrapping `history_future` before `run()` returns. Any subsequent `run()` on the same agent calls `prev_history._resolve()` inside `_task`, which blocks on the previous `history_future.result()`. This is how sequential calls are serialized without any explicit lock — the history chain is itself the queue.

---

## 2. `_task()` — the task thread

**File:** `agency/agent.py` · `_task()` (closure inside `agent.run()`)

Runs on a thread from `agent._pool`. Everything below executes on this thread.

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

### 2c. Outer loop state initialization

```python
current_input   = input
current_history = prev_history
outer_result    = None
outer_history   = prev_history
outer_delta     = []          # accumulates history_delta across all outer iterations
is_continuation = False       # suppresses input schema validation on re-entries
```

---

## 3. Outer monitoring loop

**File:** `agency/agent.py` · `for _outer_iter in range(agent.max_outer_iters)`

Each iteration runs the inner ReAct loop once, then checks for background processes.

```
┌─── outer iteration N ────────────────────────────────────────────────┐
│                                                                      │
│  agskill.run(current_input, current_history, tools, max_steps,      │
│              _is_continuation, _context_limit, ...)                  │
│  → result, new_history, history_delta                                │
│                                                                      │
│  outer_result  = result                                              │
│  outer_history = new_history                                         │
│  outer_delta  += history_delta                                       │
│                                                                      │
│  pids_at_end = set(sandbox._watched_pids)   ← snapshot after ReAct  │
│                                                                      │
│  if not pids_at_end:  break          ← no background work, done     │
│                                                                      │
│  poll get_live_pids() every poll_interval_s                          │
│  for up to ping_interval_s total                                     │
│  break as soon as all PIDs are gone                                  │
│                                                                      │
│  if not live_now:                    ← all processes finished        │
│    current_input = agdata(_event="process_completed", ...)           │
│    is_continuation = True                                            │
│    continue  ────────────────────────────────────────────────────────┤
│                                                                      │
│  else:                               ← still running after full wait │
│    current_input = agdata(_event="process_update", ...)              │
│    is_continuation = True                                            │
│    continue  ────────────────────────────────────────────────────────┘
```

### History threading across outer iterations

`new_history` from each `agskill.run()` becomes `current_history` for the next iteration, so the LLM carries the full conversation context forward — including tool calls from background-process re-entries.

### `process_completed` vs `process_update`

| `pids_at_end` | `live_now` after poll window | Action |
|---|---|---|
| empty | — | `break` — skill done |
| non-empty | empty (job finished during poll) | `continue` with `_event="process_completed"` |
| non-empty | non-empty (still alive after `ping_interval_s`) | `continue` with `_event="process_update"` + summary |

---

## 4. Inner ReAct loop — `agskill.run()`

**File:** `agency/agskill.py`

Returns `(result: agdata, updated_history: agdata, history_delta: list[dict])`.

### 4a. Tool list resolution

```python
active_tools = self.tools if self.tools is not None else agent_tools
tool_map = {t.name: t for t in active_tools}
openai_tools = [t.to_openai_tool() for t in active_tools] or None
```

### 4b. Input validation

```python
if self.input_schema is not None and not _is_continuation:
    errors = self._check_schema(input, self.input_schema)
    if errors:
        return agdata(error=f"input schema error: {errors}"), history, [sys_msg]
```

Skipped on continuation entries — process-status ping messages will not match the skill's declared input schema.

### 4c. Message list construction

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

### 4d. Per-step: inbox drain

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

### 4e. LLM call

```python
resp = client.chat.completions.create(
    model=llm_config.get("model", "gpt-4o"),
    messages=messages,
    tools=openai_tools,    # omitted entirely if no tools
)
msg = resp.choices[0].message
```

### 4f. Auto-compaction check

Immediately after each LLM response, `resp.usage.prompt_tokens` is checked against `_context_limit`. If over threshold, the context is compacted before proceeding:

```python
if _context_limit is not None and resp.usage is not None:
    if should_compact(resp.usage.prompt_tokens, _context_limit):
        messages, _compaction_summary = compact(
            messages, llm_config,
            previous_summary=_compaction_summary,
        )
```

See [compaction.md](compaction.md) for the full algorithm.

### 4g. Tool-call branch

```python
if msg.tool_calls:
    for tc in msg.tool_calls:
        result_content = tool_map[tc.function.name](agdata.from_json(tc.function.arguments)).to_json()
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_content})
    # loop back to LLM call
```

Errors from tools (`Exception` raised or unknown tool name) are caught and injected as tool-result messages so the LLM can recover.

### 4h. Final-answer branch

```python
else:
    if had_inbox:
        continue   # LLM is answering user message, not producing final output

    content = msg.content or "{}"
    # strip markdown code fences that some models add
    result = agdata.from_json(content)
    # output schema validation + retry...
    return result, updated_history, history_delta
```

### 4i. Return value construction

```python
updated_history = agdata(messages=messages[1:])         # strip system message
history_delta   = [messages[0]] + messages[1:][n_before:]   # system + new messages only
```

---

## 5. Post-loop: resource release, logging, future resolution

```python
finally:
    self.sandbox.release_resources(pool)

# after finally:
self.log._record(skill_name, ts_start, ts_end, input, outer_result, outer_history, outer_delta)
result_future.set_result(outer_result)    # unblocks caller's field access
history_future.set_result(outer_history)  # unblocks next run() on same agent
```

---

## 6. Complete call graph

```
caller thread                        task thread (ThreadPoolExecutor)
──────────────────────────────────────────────────────────────────────────────
agent.run(skill, input)
  ├─ prev_history = self._history
  ├─ result_future, history_future = Future(), Future()
  ├─ self._history = agdata(_future=history_future)
  ├─ agent._pool.submit(_task) ──────────────────────────────────▶ _task()
  └─ return agdata(_future=result_future)                           │
                                                                   prev_history._resolve()
caller.field ────────────── blocks ────────────────────────────┐   _resolve_input(input)
                                                               │   │
                                                               │   skill lookup
                                                               │   │
                                                               │   ┌── outer loop ─────────────────┐
                                                               │   │                               │
                                                               │   │  agskill.run()                │
                                                               │   │  ├─ input validation          │
                                                               │   │  ├─ message construction      │
                                                               │   │  └─ for _ in range(max_steps) │
                                                               │   │       inbox drain             │
                                                               │   │       LLM call                │
                                                               │   │       compaction check        │
                                                               │   │       ├─ tool calls?          │
                                                               │   │       │   tool.fn(agdata)     │
                                                               │   │       │   → sandbox.exec()    │
                                                               │   │       │   loop back           │
                                                               │   │       └─ final answer?        │
                                                               │   │           had_inbox? continue │
                                                               │   │           parse JSON          │
                                                               │   │           schema check        │
                                                               │   │           retry? loop back    │
                                                               │   │           return              │
                                                               │   │                               │
                                                               │   │  pids_at_end snapshot         │
                                                               │   │  poll get_live_pids()         │
                                                               │   │  ├─ break (no PIDs)           │
                                                               │   │  ├─ continue (completed)      │
                                                               │   │  └─ sleep + continue (update) │
                                                               │   └───────────────────────────────┘
                                                               │   │
                                                               │   finally: release_resources()
                                                               │   aglog._record()
                                                               │   result_future.set_result()
                                                               │   history_future.set_result()
                                                               │   │
caller.field ◀─────────────── unblocks ────────────────────────┘   │
next run()   ◀─────────────── unblocks ────────────────────────────┘
```

---

## 7. Error propagation

| Where it occurs | How it surfaces |
|---|---|
| Skill not found | Both futures resolved immediately with error / unchanged history |
| Input schema failure | `agskill.run()` returns `agdata(error=...)` before any LLM call |
| Exception in `_task` | Caught; `outer_result = agdata(error=str(exc))`; `finally` still runs |
| Unknown tool name | `{"error": "unknown tool"}` injected as tool result; LLM recovers |
| Tool `fn` raises | Per-tool catch; `{"error": str(e)}` injected; LLM recovers |
| Output schema failure after retries | `agskill.run()` returns `agdata(error=...)` |
| `max_steps` exceeded | `agskill.run()` returns `agdata(error="max_steps exceeded")` |
| `max_outer_iters` exhausted | Loop exits; last `outer_result` used as-is |

Accessing any field on an error `agdata` via `__getattr__` raises `AgError`. `.error` and `.is_error()` are safe accessors.
