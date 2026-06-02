# Execution Loop

This document traces the complete execution path of a single `agent.run()` call from submission through to future resolution, covering every function called, every data structure touched, and how the inner ReAct loop and outer monitoring loop integrate.

---

## 1. `agent.run()` — submission and future construction

**File:** `src/agent.py` · `agent.run(skill_name, input, max_steps=10)`

The caller receives a pending `agdata` immediately. No LLM call has happened yet.

```python
# Runs synchronously on the caller's thread
prev_history = self._history                    # agdata wrapping a Future or a resolved agdata
result_future:  Future[agdata] = Future()
history_future: Future[agdata] = Future()
ts_start = _ts()                                # ISO-8601 timestamp captured now for the log
pool = agent.agresource_pool                    # snapshot class-level pool reference

agent._pool.submit(_task)                       # hand _task to shared ThreadPoolExecutor

self._history = agdata(_future=history_future)  # chain: next run() on this agent blocks here
return agdata(_future=result_future)            # caller gets this — pending until _task resolves it
```

**Serialization invariant.** `self._history` is replaced with a new pending `agdata` wrapping `history_future` before `run()` returns. Any subsequent `run()` on the same agent calls `prev_history._resolve()` inside `_task`, which blocks on the previous `history_future.result()`. This is how sequential calls are serialized without any explicit lock — the history chain is itself the queue.

---

## 2. `_task()` — the task thread

**File:** `src/agent.py` · `_task()` (closure inside `agent.run()`)

Runs on a thread from `agent._pool`. Everything below executes on this thread.

### 2a. Input resolution

```python
prev_history._resolve()    # agdata._resolve() → _future.result() → blocks if still pending
_resolve_input(input)      # recursively resolves any pending agdata nested inside input fields
```

`_resolve_input` (module-level in `agent.py`) walks `input._data`:
- Top-level values that are pending `agdata` → `val._resolve()`
- List elements that are pending `agdata` → `item._resolve()`

This allows the caller to pass a prior `run()` result directly as input without blocking themselves:

```python
r1 = ag.run("search", agdata(query="..."))   # pending
r2 = ag.run("summarize", agdata(text=r1))    # r1 resolved here inside _task for r2, not by caller
```

### 2b. Skill lookup

```python
af = next((f for f in self.agskills if f.name == skill_name), None)
```

Linear scan over `self.agskills`. On failure:

```python
err = agdata(error=f"agskill not found: {skill_name!r}")
result_future.set_result(err)
history_future.set_result(prev_history)
return
```

Both futures resolve immediately. The caller's pending `agdata` unblocks with an error; the history chain is unblocked with the unchanged history.

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

**File:** `src/agent.py` · `for _outer_iter in range(agent.max_outer_iters)`

Each iteration runs the inner ReAct loop once, then checks for background processes.

```
┌─── outer iteration N ────────────────────────────────────────────────┐
│                                                                      │
│  agskill.run(current_input, current_history, tools, max_steps,      │
│              _is_continuation=is_continuation)                       │
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
│  live_now = get_live_pids()                                          │
│                                                                      │
│  if not live_now:                    ← all processes finished        │
│    current_input = agdata(                                           │
│        _event="process_completed",                                   │
│        message="Background processes have completed. ..."            │
│    )                                                                 │
│    current_history = new_history                                     │
│    is_continuation = True                                            │
│    continue  ────────────────────────────────────────────────────────┤
│                                                                      │
│  else:                               ← still running after full wait │
│    summary = sandbox.pid_status_summary()                            │
│    current_input = agdata(                                           │
│        _event="process_update",                                      │
│        message=f"...still running: {summary}..."                     │
│    )                                                                 │
│    current_history = new_history                                     │
│    is_continuation = True                                            │
│    continue  ────────────────────────────────────────────────────────┘
```

### History threading across outer iterations

`new_history` from each `agskill.run()` becomes `current_history` for the next iteration, so the LLM carries the full conversation context forward — including tool calls from background-process re-entries. `outer_delta` grows by concatenation across all iterations; the final log entry includes the complete delta for the entire skill invocation.

### PID snapshot timing

`pids_at_end = set(sandbox._watched_pids)` is taken **after** `agskill.run()` returns. By this point, every `bash` tool call that ran inside the ReAct loop has already written its background PIDs into `sandbox._watched_pids` — the `exec` wrapper extracts PIDs synchronously before returning output to the LLM.

### Polling loop

After detecting background PIDs, the loop polls `get_live_pids()` every `poll_interval_s` (default 5s) for up to `ping_interval_s` total (default 300s). It breaks as soon as all PIDs are gone — whether that takes 2 seconds or 5 minutes. This means a fast job fires `process_completed` promptly rather than waiting for a full ping interval.

### `process_completed` vs `process_update`

| `pids_at_end` | `live_now` after poll window | Action |
|---|---|---|
| empty | — | `break` — skill done |
| non-empty | empty (job finished during poll) | `continue` with `_event="process_completed"` immediately |
| non-empty | non-empty (still alive after `ping_interval_s`) | `continue` with `_event="process_update"` + summary |

---

## 4. Inner ReAct loop — `agskill.run()`

**File:** `src/agskill.py` · `agskill.run(llm_config, input, history, agent_tools, max_steps, term, _is_continuation)`

Returns `(result: agdata, updated_history: agdata, history_delta: list[dict])`.

### 4a. Tool list resolution

```python
active_tools = self.tools if self.tools is not None else agent_tools
tool_map = {t.name: t for t in active_tools}
openai_tools = [t.to_openai_tool() for t in active_tools] or None
```

If the skill has its own `tools` override, those are used exclusively. Otherwise, `agent_tools` (the sandboxed tool list built in `agent.__init__`) is used. `agtool.to_openai_tool()` returns the OpenAI function-calling dict (`{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}`).

### 4b. Input validation

```python
if self.input_schema is not None and not _is_continuation:
    errors = self._check_schema(input, self.input_schema)
    if errors:
        return agdata(error=f"input schema error: {errors}"), history, [sys_msg]
```

`_check_schema` verifies:
1. Every key in `input_schema._data` is present in `input._data`
2. If the value in the schema is a recognized type name (`"str"`, `"int"`, `"float"`, `"bool"`, `"list"`, `"dict"`), the actual value's `type()` must match

`_is_continuation=True` bypasses this check entirely — `process_completed` and `process_update` agdata will not match the skill's declared input schema and must not be rejected.

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

`_build_system_prompt()` concatenates:
1. `self.system_prompt`
2. `"\nInput JSON format:\n" + input_schema.to_json()` (if defined)
3. `"\nOutput JSON format (respond ONLY with this JSON):\n" + output_schema.to_json()` (if defined)

`input.to_json()` → `agdata.to_dict()` → `json.dumps()`, recursively resolving nested agdata. The system message is **prepended on every call but never stored** — `updated_history = agdata(messages=messages[1:])` strips it before returning.

### 4d. LLM call

```python
client = openai.OpenAI(
    api_key=llm_config.get("api_key", ""),
    base_url=llm_config.get("base_url", None),
)
resp = client.chat.completions.create(
    model=llm_config.get("model", "gpt-4o"),
    messages=messages,
    tools=openai_tools,    # omitted entirely (not passed as None) if no tools
)
msg = resp.choices[0].message
```

A new `openai.OpenAI` client is constructed on every `agskill.run()` call. Any OpenAI-compatible endpoint is supported via `base_url`.

### 4e. Tool-call branch

```python
if msg.tool_calls:
    msg_dict = {
        "role": "assistant",
        "tool_calls": [{"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name,
                                      "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls]
    }
    messages.append(msg_dict)

    for tc in msg.tool_calls:
        t = tool_map.get(tc.function.name)
        if t is None:
            result_content = json.dumps({"error": f"unknown tool: {tc.function.name}"})
        else:
            try:
                arg = agdata.from_json(tc.function.arguments)   # JSON string → agdata
                result_content = t(arg).to_json()               # agdata → fn() → agdata → JSON
            except Exception as e:
                result_content = json.dumps({"error": str(e)})
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_content})

    # loop back to LLM call
```

For the sandboxed `bash` tool, `t(arg)` calls `_run_sandboxed(arg)` which calls `sandbox.exec(arg.command, ...)`:
- Wraps the command in the `exec 2>&1 / set -m / __BGPIDS__` shell script
- Runs `docker exec -w <workdir> sandbox-<uuid> bash -c <wrapped_cmd>`
- Parses the `__BGPIDS__` annotation, writes PIDs to `sandbox._watched_pids`
- Strips the annotation, returns `(clean_output, returncode)`
- Wraps in `agdata(output=..., exit_code=..., truncated=...)`

### 4f. Final-answer branch

```python
else:
    content = msg.content or "{}"
    # Strip markdown code fences that models add despite instructions
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped[stripped.find("\n")+1:] if "\n" in stripped else stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        content = stripped.strip()
    try:
        result = agdata.from_json(content)        # JSON string → agdata
    except (json.JSONDecodeError, TypeError):
        result = agdata(result=content)           # fallback: wrap raw string
```

### 4g. Output validation and retry

```python
if self.output_schema is not None:
    errors = self._check_schema(result, self.output_schema)
    if not errors and self.output_validator is not None:
        errors = self.output_validator(result)   # custom fn: agdata → list[str]

    if errors:
        if retries_left > 0:
            retries_left -= 1
            messages.append({
                "role": "user",
                "content": (
                    f"Output schema errors: {errors}. "
                    f"Respond ONLY with valid JSON matching exactly: "
                    f"{self.output_schema.to_json()}"
                ),
            })
            continue    # retry in the same for loop, same messages list, new LLM call
        # retries exhausted
        updated_history = agdata(messages=messages[1:])
        return agdata(error=f"output schema error after retries: {errors}"), updated_history, delta
```

Retries are **in-loop** — the correction message is appended to the existing `messages` list and the LLM is called again immediately, without starting a new outer iteration or creating a new `agskill.run()` call.

### 4h. Return value construction

```python
updated_history = agdata(messages=messages[1:])         # strip system message
history_delta   = [messages[0]] + messages[1:][n_before:]   # system + new messages only
return result, updated_history, history_delta
```

`messages[1:][n_before:]` is the slice of messages added during this particular `agskill.run()` call — new user message, tool calls/results, and final assistant message. The system prompt is prepended to this delta so the log shows which prompt was active, but it is not persisted in `updated_history`.

---

## 5. Post-loop: resource release, logging, future resolution

**File:** `src/agent.py` · after the outer `for` loop, still inside `_task()`

```python
# finally block — always runs
finally:
    self.sandbox.release_resources(pool)
    # → pool.release_gpu(sandbox._gpu_id) if held
    # → sandbox.update_limits(cpus=pool.idle_cpus, memory=pool.idle_memory)
```

Then outside `try/finally`:

```python
ts_end = _ts()
self.log._record(
    skill_name, ts_start, ts_end,
    input.to_dict(),           # original input, fully resolved
    outer_result.to_dict(),    # final result from last outer iteration
    len(outer_history._data.get("messages", [])),
    history_before=history_before,
    history_delta=outer_delta,  # full delta across ALL outer iterations
)
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
                                                               │   │       LLM call                │
                                                               │   │       ├─ tool calls?          │
                                                               │   │       │   agdata.from_json()  │
                                                               │   │       │   tool.fn(agdata)     │
                                                               │   │       │   → sandbox.exec()    │
                                                               │   │       │     docker exec       │
                                                               │   │       │     PID extraction    │
                                                               │   │       │   result.to_json()    │
                                                               │   │       │   loop back           │
                                                               │   │       └─ final answer?        │
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
| Skill not found | `result_future.set_result(agdata(error=...))` immediately; `history_future` set to unchanged `prev_history` |
| Input schema failure | `agskill.run()` returns `agdata(error=...)` before any LLM call; outer loop sees it as a result |
| Exception in `_task` | Caught by `except Exception as exc`; `outer_result = agdata(error=str(exc))`; `finally` still runs |
| Unknown tool name | `{"error": "unknown tool: <name>"}` injected as tool result; LLM sees it and continues |
| Tool `fn` raises | Caught per-tool; `{"error": str(e)}` injected as tool result; LLM continues |
| Output schema failure after retries | `agskill.run()` returns `agdata(error=...)`; outer loop propagates it as `outer_result` |
| `max_steps` exceeded | `agskill.run()` returns `agdata(error="max_steps exceeded")` |
| `max_outer_iters` exhausted | Loop exits; last `outer_result` is used as-is (may be a valid partial result) |

Accessing any field on an error `agdata` via `__getattr__` raises `AgError`. `.error` (a property) and `.is_error()` are safe accessors that do not raise.

---

## 8. `agdata` flow through the execution

```
caller constructs:    agdata(question="What is X?")
                          │
                          │  input.to_json()
                          ▼
LLM user message:     '{"question": "What is X?"}'
                          │
              ┌───────────┴───────────────────────────┐
              │ tool call branch                      │ final answer branch
              │                                       │
              │  tc.function.arguments                │  msg.content
              │  '{"command": "ls /workspace"}'       │  '{"answer": "X is..."}'
              │          │                            │          │
              │  agdata.from_json()                   │  agdata.from_json()
              │  agdata(command="ls /workspace")      │  agdata(answer="X is...")
              │          │                            │          │
              │  tool.fn(agdata)                      │  _check_schema()
              │  → sandbox.exec("ls /workspace")      │  output_validator()
              │  → agdata(output="...", exit_code=0)  │          │
              │          │                            │  return agdata(answer="X is...")
              │  result.to_json()                     │          │
              │  '{"output":"...","exit_code":0}'     │          ▼
              │  → tool message in messages list      │  outer_result = agdata(answer="X is...")
              │          │                            │
              └──────────┴────────────────────────────┘
                          │
                  result_future.set_result(outer_result)
                          │
caller reads:     result.answer
                  → agdata.__getattr__("answer")
                  → _resolve() → future.result() → "X is..."
```

If the resolved agdata contains an `"error"` key, `__getattr__` raises `AgError` on any field access other than `.error`. This propagates skill failures as Python exceptions to the caller without any special handling at the `run()` call site.
