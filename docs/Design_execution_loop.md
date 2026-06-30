# Execution Loop

This document traces the complete execution path of a single `agent.run()` call from submission through to future resolution, covering every function called, every data structure touched, and how the ReAct loop and sandbox process monitoring integrate.

---

## 1. `agent.run()` — submission and future construction

**File:** `agency/agent.py` · `agent.run(skill, skill_input, max_steps=AGSKILL_REACT_MAX_STEPS)`

The caller receives a pending `agdata` immediately. No LLM call has happened yet. `agent.run()` contains no execution logic — it delegates entirely to the skill:

```python
def run(self, skill, skill_input, max_steps=...):
    return skill.run(self, skill_input, max_steps=max_steps)
```

---

## 2. `agskill.run()` — scheduling wrapper

**File:** `agency/agskill.py` · `agskill.run(ag, skill_input, max_steps)`

Non-blocking. Captures state, installs a pending `agcontext` placeholder, spawns a daemon thread, and returns immediately.

```python
prev_ctx = ag.ctx                               # capture before swap
result_future: Future[agdata]    = Future()
ctx_future:    Future[agcontext] = Future()

threading.Thread(target=_task, daemon=True).start()

ag.ctx = agcontext(_future=ctx_future)          # install placeholder on agent
return agdata(_future=result_future)            # caller gets this — pending
```

**Serialization invariant.** `ag.ctx` is replaced with a pending `agcontext` wrapping `ctx_future` before `run()` returns. Any subsequent `run()` on the same agent captures this placeholder as its own `prev_ctx`. Inside `_task()`, `prev_ctx.resolve_prev_dependencies()` blocks on the previous `ctx_future.result()`. This is how sequential calls are serialized without any explicit lock — the context chain is itself the queue.

---

## 3. `_task()` — the task thread

**File:** `agency/agskill.py` · `_task()` (closure inside `agskill.run()`)

Runs on a dedicated daemon thread. Everything below executes on this thread.

### 3a. Input resolution

```python
prev_ctx.resolve_prev_dependencies()       # blocks if prev skill not yet done
skill_input.resolve_input_dependencies()   # resolves any pending agdata nested in input fields
```

`resolve_input_dependencies()` walks `skill_input._data` and resolves any top-level or list-element values that are pending `agdata`. This allows passing a prior `run()` result directly as input without blocking the caller:

```python
r1 = ag.run(search_skill, agdata(query="..."))   # pending
r2 = ag.run(summarize_skill, agdata(text=r1))    # r1 resolved here inside _task for r2
```

### 3b. Sandbox provisioning

```python
if not ag.is_external_sandbox and ag.sandbox is None:
    _out = Path(type(ag).output_dir) / ag.agname if type(ag).output_dir else None
    ag.sandbox = agSandbox(ag.agname, output_dir=_out)
```

Sandbox creation is lazy — only on the first skill run that needs tools. If `is_external_sandbox=True`, the framework skips creation and uses the externally-provided container.

### 3c. Skill execution

```python
outer_result, updated_ctx, outer_delta = self.execute_react(
    ag, prev_ctx, skill_input, max_steps,
)
```

`execute_react()` returns a 3-tuple: `(result: agdata, updated_ctx: agcontext, ctx_delta: list[dict])`. Token counts accumulate inside `prev_ctx` during the run and are committed back via `prev_ctx.total_input_tokens` / `prev_ctx.total_output_tokens`.

### 3d. Exception handling

The entire body of `_task` is wrapped in `try / except / finally`:

```python
try:
    prev_ctx.resolve_prev_dependencies()
    skill_input.resolve_input_dependencies()
    ...
    outer_result, updated_ctx, outer_delta = self.execute_react(...)

except Exception as exc:
    outer_result  = agerror(format_exception(exc))
    updated_ctx   = prev_ctx
    outer_delta   = []

finally:
    _had_error = outer_result._data.get("error")
    ag._set_ui_state("error" if _had_error else "finished")
    if ag.sandbox is not None and ag.sandbox._gpu_id is not None:
        resource_pool.release_gpu(ag.sandbox._gpu_id)
    if not ag.is_external_sandbox and ag.sandbox is not None:
        ag.sandbox.stop(commit=True)
```

The `finally` block always runs. It releases any GPU held by the sandbox and commits + stops the container (`stop(commit=True)` snapshots the container to the lifecycle image and removes it). The next skill run will restore from that image.

### 3e. Logging, token accounting, future resolution

```python
ag.terminal.log("SKILL ✓  ", ...)
ag.log._skill(...)
agent._add_global_tokens(...)
ag.push_token_count_update_to_ui(...)
ag._append_full_history(...)

# Post-skill history pruning — trim oversized tool outputs before
# unblocking the next run() so context stays bounded.
pruned_msgs = agllm._prune_tool_outputs(updated_ctx.messages)
if pruned_msgs is not updated_ctx.messages:
    updated_ctx.messages = pruned_msgs

result_future.set_result(outer_result)    # unblocks caller's field access
ctx_future.set_result(updated_ctx)        # unblocks next run() on same agent
```

`result_future` resolves before `ctx_future`. The caller unblocks as soon as the result is available; the next `run()` on the same agent blocks a little longer while pruning completes, ensuring context is always clean when the next skill begins.

---

## 4. ReAct loop — `agskill.execute_react()`

**File:** `agency/agskill.py` · `agskill.execute_react(ag, prev_ctx, skill_input, max_steps)`

Receives `prev_ctx` explicitly because `ag.ctx` has already been replaced with the new pending placeholder by the time the thread runs.

Returns `(result: agdata, prev_ctx: agcontext, ctx_delta: list[dict])`. `ctx_delta` contains only the new messages added during this skill (system prompt + messages after the previous context). Token totals accumulate into `prev_ctx.total_input_tokens` / `prev_ctx.total_output_tokens` throughout the run.

### 4a. Input validation

```python
if self.input_schema is not None:
    error = self.input_schema.validate_input(skill_input)
    if error:
        return agerror(error), prev_ctx, [sys_msg]
```

### 4b. Input preparation and toolkit construction

```python
_offloaded_paths, auto_fields = self.input_schema.prepare_inputs_in_sandbox(
    skill_input, ag.sandbox, self.name, ...
)
toolkit, _collected_outputs, _required_fields = self._build_toolkit(
    ag.sandbox, type(ag).agresource_pool, ag.terminal, ag.log,
    _ensure_read=bool(_offloaded_paths),
)
_use_return_output = bool(_required_fields)
```

`_build_toolkit()` returns a 3-tuple. `_collected_outputs` and `_required_fields` are mutable containers that the `return_<field>` tools write into as they are called during the loop. `_use_return_output=True` when the skill has a structured `output_schema` (not `agrawstring`).

### 4c. Message list construction

```python
messages = (
    [{"role": "system", "content": self._build_system_prompt(_extra_system)}]
    + list(prev_ctx.messages)
    + [{"role": "user", "content": self._build_user_content(skill_input)}]
)
n_before = len(prev_ctx.messages)
```

The system message is prepended on every call but never stored — `prev_ctx.messages = messages[1:]` strips it before returning.

### 4d. Per-step: inbox drain

At the start of each step, `ag._drain_inbox(messages)` drains the agent's `inbox: Queue[str]` — any string pushed by an external orchestrator is appended as a user message. When `had_inbox=True` and the LLM produces text (no tool calls), the loop continues rather than treating it as a final answer.

### 4e. Pre-call compaction

```python
messages, _pre_estimate = ag.llm.maybe_compact(
    prev_ctx, messages, None, term=ag.terminal, log=ag.log, ...
)
ag.push_token_count_update_to_ui(
    prev_ctx.total_input_tokens - _skill_tokens_in_start + _pre_estimate,
    prev_ctx.total_output_tokens - _skill_tokens_out_start,
)
```

A character-based token estimate is compared against the context limit. If the estimate exceeds the compaction threshold, `maybe_compact()` runs a summarization call, replaces old messages with the summary, and stores the summary in `prev_ctx.compaction_summary` for injection into future system prompts.

### 4f. LLM call

```python
llm_result = ag.llm.call(
    kwargs, messages,
    ag.terminal, ag._set_ui_state, ag._push_live_messages,
    ag.push_token_count_update_to_ui,
    prev_ctx.total_input_tokens, prev_ctx.total_output_tokens,
    self.name, full_history_fn=ag._append_full_history,
)
```

`agllm.call()` streams a single LLM call and returns an `LLMCallResult`. It retries on transient connection errors (`ssl.SSLError`, `OSError`, `httpx.TransportError`, `openai.APIConnectionError`) up to `LLM_MAX_RETRIES` times with a fixed sleep between attempts. On `openai.BadRequestError` matching context length keywords, it returns `context_exceeded=True` without retrying.

```python
if llm_result.context_exceeded:
    messages, _ = ag.llm.maybe_compact(..., force=True)
    continue
if not llm_result.ok:
    return agerror(f"LLM connection error after retries: {llm_result.conn_error}"), ...
prev_ctx.total_input_tokens  = llm_result.total_input_tokens
prev_ctx.total_output_tokens = llm_result.total_output_tokens
```

### 4g. Post-call compaction

```python
messages, _ = ag.llm.maybe_compact(
    prev_ctx, messages, llm_result.prompt_tokens, ...
)
```

After the call, the exact `prompt_tokens` from the API usage chunk triggers a second compaction check using the real count rather than the estimate.

### 4h. Tool-call branch

```python
msg_dict = agllm.build_assistant_msg(
    llm_result.content_parts, llm_result.reasoning_parts, llm_result.tool_calls_raw
)
messages.append(msg_dict)
...
if msg_dict.get("tool_calls"):
    had_inbox = ag._drain_inbox(messages)
    result_msg, _read_injected = dispatch_tools(
        toolkit, tool_calls_raw, ag.sandbox, ...
    )
```

For each tool call in `dispatch_tools`:

1. Unknown tool → `{"error": "unknown tool: <name>"}` injected; LLM recovers.
2. `run_in_subprocess=True` → sandbox committed to pre-tool checkpoint before execution; restored on error.
3. Tool executes in a `ProcessPoolExecutor` worker with configurable timeout.
4. **Large output offloading.** If the result exceeds `TOOL_OUTPUT_OFFLOAD_CHARS` characters, the content is written to `/workspace/long_tool_call_outputs/<tool>_<id>.txt` inside the sandbox and the tool result is replaced with a short note telling the LLM to use `read` to access it. `read` is injected into the toolkit if not already present.
5. **`return_<field>` tools** (structured output): write into `_collected_outputs`. When `_required_fields ⊆ _collected_outputs`, all output fields are collected.

After dispatch, check if output is complete:

```python
if _use_return_output:
    missing = _required_fields - set(_collected_outputs)
    if not missing:
        # all fields collected — wait for background processes, then return
        result = agdata(**_collected_outputs)
        ...
        return result, prev_ctx, delta
    elif output_schema_retries_left > 0:
        # inject reprompt for missing fields, continue
    else:
        return agerror("output schema error: missing fields after retries: ..."), ...
```

### 4i. Text output path

When the LLM produces text with no tool calls:

```python
if had_inbox:
    continue   # mid-conversation with user — not a final answer yet
```

If `_use_return_output=True` and we reach here, the model ignored the return tools — reprompt up to `max_output_schema_retries` times, then error.

If `output_schema` is `None` or has a `raw_key()` (`agrawstring`):
```python
out_key = self.output_schema.raw_key() if self.output_schema else "result"
result  = agdata(**{out_key: msg_dict.get("content") or ""})
```

Raw text is returned as-is — no JSON parsing.

**Process monitoring:** before returning, `agSandbox.wait_for_processes()` polls until any background PIDs tracked by the sandbox exit. If processes are still running, a status message is injected and the loop continues. The loop only exits when the sandbox is clean.

### 4j. `max_steps` exceeded

```python
return agerror("max_steps exceeded"), prev_ctx, [messages[0]] + messages[1:][n_before:]
```

---

## 5. Complete call graph

```
caller thread                         task thread (daemon)
──────────────────────────────────────────────────────────────────────────────
agent.run(skill, input)
  └─ skill.run(ag, input)             # agskill.run()
       ├─ prev_ctx = ag.ctx           # capture agcontext before swap
       ├─ result_future  = Future()
       ├─ ctx_future     = Future()
       ├─ ag.ctx = agcontext(         # install pending placeholder
       │       _future=ctx_future)
       ├─ Thread(_task).start() ─────────────────────────────▶ _task()
       └─ return agdata(              │
               _future=result_future) │  prev_ctx.resolve_prev_dependencies()
                                      │  skill_input.resolve_input_dependencies()
caller.field ───── blocks ────────────┐  agSandbox(...) if needed
                                      │  │
                                      │  agskill.execute_react(ag, prev_ctx, skill_input)
                                      │  ├─ input_schema.validate_input()
                                      │  ├─ input_schema.prepare_inputs_in_sandbox()
                                      │  ├─ _build_toolkit()
                                      │  │   → (toolkit, _collected_outputs, _required_fields)
                                      │  ├─ _build_initial_messages()
                                      │  └─ for _ in range(max_steps):
                                      │       ag._drain_inbox(messages)
                                      │       ag.llm.maybe_compact() [pre-call]
                                      │       ag.push_token_count_update_to_ui()
                                      │       ag.llm.call(kwargs, messages, ...)
                                      │       │  openai streaming call
                                      │       │  retry on SSL/OS/connection error
                                      │       │  return LLMCallResult
                                      │       if context_exceeded → compact, continue
                                      │       if not ok → return agerror
                                      │       prev_ctx.total_*_tokens updated
                                      │       ag.llm.maybe_compact() [post-call]
                                      │       build_assistant_msg() → append
                                      │       ag._drain_inbox(messages)
                                      │       ├─ tool calls?
                                      │       │   dispatch_tools(toolkit, ...)
                                      │       │   ├─ unknown tool → error msg
                                      │       │   ├─ run_in_subprocess? → commit checkpoint
                                      │       │   ├─ tool.fn(agdata) in worker process
                                      │       │   │   → sandbox.exec() inside container
                                      │       │   ├─ error? → restore checkpoint
                                      │       │   ├─ large output? → offload to file, inject read
                                      │       │   └─ return_<field>? → collect into _collected_outputs
                                      │       │   all fields collected?
                                      │       │   → wait_for_processes, return result
                                      │       │   missing fields? → reprompt or error
                                      │       │   loop back
                                      │       └─ text output (no tool calls)?
                                      │           had_inbox? → continue
                                      │           _use_return_output? → reprompt or error
                                      │           agrawstring / no schema → agdata(result=content)
                                      │           wait_for_processes
                                      │           processes running? → inject msg, continue
                                      │           sandbox clean → return
                                      │  │
                                      │  finally:
                                      │    GPU release
                                      │    ag.sandbox.stop(commit=True)
                                      │  log / token accounting
                                      │  _prune_tool_outputs(updated_ctx.messages)
                                      │  result_future.set_result(outer_result)
                                      │  ctx_future.set_result(updated_ctx)
                                      │  │
caller.field ◀──── unblocks ──────────┘  │
next run()   ◀──── unblocks ─────────────┘
```

---

## 6. Error propagation

| Where it occurs | How it surfaces |
|---|---|
| Input schema failure | `execute_react()` returns `agerror(...)` before any LLM call |
| Exception in `_task` | Caught by outer `except`; `outer_result = agerror(format_exception(exc))`; `finally` still runs |
| Unknown tool name | `{"error": "unknown tool: <name>"}` injected as tool result; LLM recovers |
| Tool `fn` raises | Per-tool catch; `{"error": str(e)}` (+ `"workspace_reverted"` if checkpoint taken) injected; LLM recovers |
| Tool timeout | Same as tool `fn` raises |
| LLM connection error | `ag.llm.call()` retries up to `LLM_MAX_RETRIES`; on exhaustion returns `conn_error` in `LLMCallResult` → skill returns `agerror(...)` |
| Context length exceeded | `ag.llm.call()` returns `context_exceeded=True` → forced compaction, loop continues |
| Output schema failure after retries | `execute_react()` returns `agerror("output schema error: missing fields after retries: ...")` |
| `max_steps` exceeded | `execute_react()` returns `agerror("max_steps exceeded")` |

Accessing any field on an error `agdata` via `__getattr__` raises `AgError`. `.error` and `.is_error()` are safe accessors.
