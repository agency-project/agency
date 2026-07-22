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

Immediately after — before touching the sandbox — `ag._check_pause(self.name)` runs once. This honors a `pause()` requested before this run even started, so an agent paused while idle never provisions a sandbox or makes an LLM call. See `agent.md`'s "Pause and resume" section for the other checkpoint (once per ReAct-loop iteration, 4d below).

### 3b. Sandbox provisioning and locking

```python
if ag.sandbox is None:
    _out_dir = (
        ag.agconfig.get("agent", "output_dir", type(ag).output_dir)
        if ag.agconfig is not None else type(ag).output_dir
    )
    _out = Path(_out_dir) / ag.agname if _out_dir else None
    sb_cfg = ag.agconfig
    if _out is not None:
        sb_cfg = sb_cfg.clone() if sb_cfg else agConfig()
        agSandboxConfig(sb_cfg).add_mount("agent_output", _out, "/agent_output")
    ag.sandbox = agSandbox(ag.agname, agconfig=sb_cfg)

# Hold the sandbox's lock for the rest of the skill run so a sandbox
# shared across agents is never driven by more than one skill run
# at a time — released in the teardown below.
sandbox_lock = ag.sandbox._lock
sandbox_lock.acquire()
```

Sandbox creation is lazy — only on the first skill run that needs tools, and only if `ag.sandbox` isn't already set (either from a prior run on this agent, or a caller-provided sandbox passed to `agent(sandbox=...)`). There's no "external sandbox" special case anymore — whoever created the `agSandbox`, this skill run provisions it if missing and manages its lifecycle identically.

Immediately after provisioning, `_task()` acquires `ag.sandbox._lock` (a `threading.RLock`, one per `agSandbox` instance) and holds it until the teardown in 3d releases it. This matters because a single `agSandbox` object can now be shared across more than one agent (e.g. handed from one agent to another, as in `examples/sandbox_handoff.py`); the lock ensures two skill runs never interleave `exec()`/`stop()`/`_ensure_started()`/`commit()`/`rm_container()` calls against the same container. Everything the ReAct loop does to the sandbox — each tool call's own `stop()`/`_ensure_started()` (hibernate/resume only — see `Design_sandbox_lifecycle.md`), `wait_for_processes()` polling, and the single `commit()`/`rm_container()` call at teardown (3d below) — happens on this same thread, so those calls reentrantly reuse the lock this thread already holds at no cost.

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
    # Teardown — commit or discard the sandbox; this is now the only
    # rollback boundary (per-tool-call rollback no longer exists — see
    # agtool.py's dispatch_tools() / Design_sandbox_lifecycle.md).
    _had_error = outer_result is not None and bool(outer_result._data.get("error"))
    ag._set_ui_state("error" if _had_error else "finished")
    if ag.sandbox is not None:
        if _had_error:
            # Discard everything since the last successful skill's
            # commit(). The notice can't go into this skill's own result
            # (already final by this point) -- it goes on the inbox
            # instead, so the NEXT skill call's execute_react loop (via
            # ag._drain_inbox(), run before its first LLM call) surfaces
            # it right as the agent resumes sandbox work.
            ag.sandbox.rm_container()
            ag.inbox.put(
                "Note: the previous skill call failed. Its sandbox "
                "workspace changes have been discarded and the "
                "workspace has been reverted to the last "
                "successful checkpoint."
            )
        else:
            # commit() squashes automatically once the layer chain's
            # actual depth crosses checkpoint_squash_max_depth -- see
            # its docstring for why that's a depth-triggered check, not
            # a fixed commit count.
            ag.sandbox.commit()
    if sandbox_lock is not None:
        sandbox_lock.release()
```

The `finally` block always runs, and is where the sandbox lifecycle decision that used to happen after *every tool call* now happens exactly *once*, at the end of the whole skill. On success, `commit()` checkpoints the container's current filesystem into its lifecycle image **without removing or even stopping the container** — the next skill run, on this agent or whichever one next holds this `agSandbox`, resumes directly from the very same container object. On failure, `rm_container()` force-removes the container outright, discarding everything since the last successful skill's `commit()`, and a revert notice is queued onto `ag.inbox` (see "Sandbox rollback and the inbox notice" below) since the failing result is already final by this point. Neither branch calls `stop()` — a container that's still running or hibernating from the ReAct loop's own per-tool-call `stop()` calls is left exactly as it was; only `commit()`/`rm_container()` change anything durable. There is no separate GPU-release step in this `finally` block: `commit()` never touches the GPU (the container isn't stopped or removed, so there's nothing to release), and `rm_container()` releases it as part of removal, same as `destroy()` does. The GPU's actual release cadence is set one level up, in the per-tool-call `stop()` calls this `finally` block doesn't touch at all: every hibernate between tool calls now releases the GPU too (re-acquired, possibly a *different* physical GPU, on the next `exec()`) — see `agsandbox_backends/container.md`'s "GPU device access" for why attaching every GPU up front to a GPU-reserving container makes that safe. Finally, the per-sandbox lock acquired in 3b is released last, so nothing else can touch this sandbox until this skill run's own teardown has fully finished.

### Sandbox rollback and the inbox notice

Rollback now happens at exactly one point in the whole system: this `finally` block, once per skill call — replacing an older design where every individual tool call inside `dispatch_tools()` decided for itself whether to checkpoint or discard the sandbox. A single tool call failing partway through an otherwise-successful skill no longer discards anything by itself; only the skill's own final outcome (an `"error"` key in `outer_result`, or an exception that escaped `execute_react()`) decides, and it decides for the *entire* skill run's worth of tool calls at once.

Because `rm_container()` runs from inside `finally` — after `outer_result` is already computed and about to be returned to the caller — there is no way to attach a note to that same result the way an inline `workspace_reverted` key once could. Instead, the notice is pushed onto `ag.inbox` (a plain `queue.Queue[str]`), and surfaces at the very start of the *next* skill call: step 4d below shows `ag._drain_inbox(messages)` running every ReAct-loop iteration, including the first — before that skill's first LLM call — turning any queued string into a `{"role": "user", ...}` conversation turn. The agent therefore learns about the revert as it resumes sandbox work on the next skill, not inside the failed skill's own output.

`"finished"`/`"error"` are both leaf states `agent.is_settled()` treats as trivially settled (alongside `"inactive"` and `"paused"`) — a `wait_all_paused()` call covering this agent won't block once `_task()` reaches this line, regardless of whether `pause()` was ever called.

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

### 4d. Per-step: pause checkpoint and inbox drain

Before anything else in the loop body — before tool schemas are even built — `ag._check_pause(self.name)` runs. This is the only place (besides once before the loop starts, in 3a) that a `pause()` request can take effect: it's between steps, never mid-LLM-call or mid-tool-call, so whatever the previous step was doing always finishes first. If `pause()` was called, this blocks until `resume()`; see `agent.md`'s "Pause and resume" section.

Then `ag._drain_inbox(messages)` drains the agent's `inbox: Queue[str]` — any string pushed by an external orchestrator (or by this same skill's *own* teardown from a previous run — see "Sandbox rollback and the inbox notice" under 3d below) is appended as a user message. When `had_inbox=True` and the LLM produces text (no tool calls), the loop continues rather than treating it as a final answer.

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
2. Tool executes — in a `ProcessPoolExecutor` worker with configurable timeout if `run_in_subprocess=True` (the default for custom tools), or synchronously in the calling thread if `run_in_subprocess=False` (every built-in sandboxed tool — bash, read, write, edit, …). This flag no longer has any checkpointing effect: there is no per-tool-call commit or restore anymore, regardless of which path a tool takes (see step 4 below and `Design_sandbox_lifecycle.md`'s "Per-tool-call container lifecycle").
3. **Large output offloading.** If the result exceeds `TOOL_OUTPUT_OFFLOAD_CHARS` characters, the content is written to `/workspace/long_tool_call_outputs/<tool>_<id>.txt` inside the sandbox and the tool result is replaced with a short note telling the LLM to use `read` to access it. `read` is injected into the toolkit if not already present.
4. **Sandbox hibernation.** Whether the tool succeeded or raised, `dispatch_tools()` calls `sandbox.stop()` — a `docker/podman stop` that hibernates the container without removing it — unless `sandbox._has_pending_background_work()` is true, in which case `stop()` is skipped entirely for this call so a still-running background job isn't killed prematurely. A tool call's own success or failure has no bearing on this decision anymore, and no bearing on whether anything gets checkpointed or discarded — that decision moves to skill-exit (3d above).
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
                                      │       │   ├─ tool.fn(agdata) — worker process (run_in_subprocess=True)
                                      │       │   │   or calling thread (run_in_subprocess=False)
                                      │       │   │   → sandbox.exec() inside container
                                      │       │   ├─ large output? → offload to file, inject read
                                      │       │   ├─ pending background work? → stop() deferred
                                      │       │   ├─ otherwise → sandbox.stop() (hibernate only,
                                      │       │   │     success or failure alike — no commit/restore here)
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
                                      │    success → ag.sandbox.commit()  (checkpoint; container untouched)
                                      │    error   → ag.sandbox.rm_container()  (discard) + ag.inbox.put(notice)
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
| Tool `fn` raises | Per-tool catch; `{"error": str(e)}` injected; LLM recovers. No per-tool sandbox rollback anymore — the sandbox is only hibernated (`stop()`), same as on success (see 4h above) |
| Tool timeout | Same as tool `fn` raises |
| LLM connection error | `ag.llm.call()` retries up to `LLM_MAX_RETRIES`; on exhaustion returns `conn_error` in `LLMCallResult` → skill returns `agerror(...)` |
| Context length exceeded | `ag.llm.call()` returns `context_exceeded=True` → forced compaction, loop continues |
| Output schema failure after retries | `execute_react()` returns `agerror("output schema error: missing fields after retries: ...")` |
| `max_steps` exceeded | `execute_react()` returns `agerror("max_steps exceeded")` |
| **Skill itself fails** (any of the above surfaces as `outer_result._data["error"]`, or an exception escapes `execute_react()`) | 3d's `finally` block calls `ag.sandbox.rm_container()` — discarding every tool call's state since the last successful skill's `commit()` — and queues a revert notice onto `ag.inbox`, surfaced as a `{"role": "user", ...}` turn at the start of the *next* skill call (see "Sandbox rollback and the inbox notice" under 3d) |

Accessing any field on an error `agdata` via `__getattr__` raises `AgError`. `.error` and `.is_error()` are safe accessors.
