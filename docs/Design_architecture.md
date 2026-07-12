# Architecture Design

## Core Principle

`agent` is a **pure state container**. `agskill` is the **execution engine**.

This inversion from the original design eliminates the circular import between agent and agskill, removes the `agrun_hooks` indirection layer, and makes the execution logic entirely testable without constructing a real agent.

---

## Module Dependency Order

```
agdata / agtype / agutil / agcontext / agname / agpause
        ↓
agterm / aglog / agresources
        ↓
agllm / agsandbox
        ↓
agtool / agschema
        ↓
agent          ← pure state container; run() just calls skill.run(self, ...)
        ↓
agskill        ← takes agent as an arg; owns scheduling wrapper + ReAct loop
        ↓
agteam / agsync
```

`agent.py` does **not** import `agskill`. It accepts any duck-typed object with a `.run(agent, skill_input)` method and delegates to it. This keeps the dependency graph acyclic.

---

## Ownership Map

### `agent` — state container

Holds all runtime state. No execution logic.

| Field | Type | Purpose |
|---|---|---|
| `llm` | `agllm` | LLM endpoint config and call interface |
| `ctx` | `agcontext` | Current/pending conversation context |
| `sandbox` | `agSandbox \| None` | Container sandbox; created lazily on first skill run, or supplied directly by the caller. Plain attribute — no property, no ownership flag. |
| `terminal` | `agterm` | Structured terminal output for this agent |
| `log` | `aglog` | Persistent event log |
| `agname` | `agname` | Unique allocated agent name |
| `inbox` | `Queue[str]` | Mid-loop message injection from orchestrators |
| `_state` | `agent_state` | Current display state (`.state`/`.skill`/`.tool`, pushed to webui) *and* the pause-synchronization source of truth (`.run_allowed`/`.paused_ack` Events, `.blocked_on`) — see `agent.md`'s "Pause and resume" |
| `_full_history` | `list[dict]` | Append-only transcript of all messages |

Methods on `agent` are limited to state manipulation and thin delegation:
- `run(skill, skill_input)` → `return skill.run(self, skill_input)`
- `fork(source)` → deep-copy an existing agent's state into a new instance
- `_drain_inbox(messages)` — drain queued inbox messages into the conversation
- `push_token_count_update_to_ui(...)` — push live token counts to webui
- `_set_ui_state(...)`, `_push_live_messages(...)`, `_append_full_history(...)` — UI/log callbacks called by agskill; `_set_ui_state` is a thin wrapper around `self._state.update_state(...)`
- `pause()`, `resume()`, `is_paused()`, `is_settled()`, `_check_pause()` — pause/resume coordination; see `agent.md` and `agency/agpause.py` (cross-agent wait/dependency tracking lives in `agpause.py`, not here)

### `agskill` — execution engine

Owns both the non-blocking scheduling wrapper and the synchronous ReAct loop.

#### `agskill.run(ag, skill_input, max_steps)` — scheduling wrapper

Non-blocking. Called from `agent.run()`. Does not perform any LLM calls itself.

1. Captures `prev_ctx = ag.ctx` before the thread starts
2. Creates `result_future: Future[agdata]` and `ctx_future: Future[agcontext]`
3. Spawns a daemon thread running `_task()`
4. Sets `ag.ctx = agcontext(_future=ctx_future)` — replaces the agent's context with a pending placeholder
5. Returns `agdata(_future=result_future)` immediately

The pending `ag.ctx` placeholder is how sequential skill calls on the same agent serialize without explicit locking: each call captures the current (possibly pending) `ag.ctx` as its own `prev_ctx`, then replaces `ag.ctx` with a new placeholder. Each thread blocks on `prev_ctx.resolve_prev_dependencies()` before proceeding, creating an implicit dependency chain.

#### `_task()` — thread closure inside `run()`

Runs in the background thread spawned by `run()`. Sequence:

1. `prev_ctx.resolve_prev_dependencies()` — block until the previous skill's context future resolves
2. `skill_input.resolve_input_dependencies()` — block until any pending agdata inputs resolve
3. Provision sandbox if needed
4. Call `self.execute_react(ag, prev_ctx, skill_input, max_steps)`
5. Log result, update token counts, prune history
6. `result_future.set_result(result)` — unblock any caller awaiting the return value
7. `ctx_future.set_result(updated_ctx)` — unblock the next skill's `resolve_prev_dependencies()`

#### `agskill.execute_react(ag, prev_ctx, skill_input, max_steps)` — synchronous ReAct loop

The actual execution. Receives `prev_ctx` explicitly because `ag.ctx` has already been replaced with the new pending placeholder by the time the thread runs.

Loop structure:
```
for _ in range(max_steps):
    drain inbox messages
    maybe_compact history
    call agllm.call(...)           → LLMCallResult
    if context_exceeded: compact and continue
    if not ok: return error
    update token counts
    post-response compact
    append assistant message
    drain inbox messages
    if tool calls:
        dispatch_tools(...)
        if return_* tool called: collect output field
        if all fields collected: wait_for_processes, return result
        continue
    else:
        if no inbox: proceed to output path
    if _use_return_output: check for missing fields, reprompt or error
    else: return raw text as agdata(result=content)
```

### `agcontext` — conversation state

Accumulates across skill calls. Fields:

| Field | Purpose |
|---|---|
| `messages` | Conversation history (excludes system prompt) |
| `total_input_tokens` | Cumulative input tokens across all LLM calls |
| `total_output_tokens` | Cumulative output tokens across all LLM calls |
| `compaction_summary` | Rolling summary from compaction; injected into next system prompt |
| `_future` | Set when this context is a pending placeholder |

`resolve_prev_dependencies()` blocks until `_future` resolves, then merges the resolved context's state into `self` in-place — so any existing reference to this context object (e.g. `ag.ctx`) automatically sees the resolved state without needing reassignment.

### `agpause` — cross-agent pause/dependency coordination

A leaf module (no agency-internal imports) sitting alongside `agdata`/`agcontext` in the dependency order above — both of those, plus `agent` and `agskill`, import it. Owns:
- A thread-local mapping the current OS thread to "the agent whose worker thread this is" (set for the lifetime of an `agskill._task()` run), and `tag_producer`/`producer_of` to associate a `concurrent.futures.Future` with the agent that will resolve it.
- `note_blocked_on()` — a context manager wrapped around every blocking `future.result()` call in `agdata._resolve()` and `agcontext.resolve_prev_dependencies()`, so a cross-agent dependency wait shows up as an observable `agent._state.blocked_on` link rather than an opaque parked thread.
- `wait_all_paused()` / `wait_all_resumed()` — barriers that recurse through `blocked_on` chains via `agent.is_settled()`, so waiting for a whole dependency graph to pause can't hang on an agent that will never reach its own checkpoint.

`agent_state` itself (the `.state`/`.skill`/`.tool`/`.run_allowed`/`.paused_ack`/`.blocked_on` container, one instance per agent as `ag._state`) is defined in `agent.py`, not here — see the Ownership Map above and `agent.md`'s "Pause and resume" section for the full mechanism.

### `agschema` — skill I/O contracts

Users write `agdata(field=type)` at call sites. `agskill.__init__` converts these to `agschema` internally. Responsibilities:
- Validate input/output data against declared types
- Generate system prompt fragments describing the I/O contract
- Create `return_<field>` tools for structured output collection
- Prepare `agtype` fields before the loop (e.g. write agfile contents to sandbox)
- Recover `agtype` output values after the loop

### `agllm` — LLM client

Single responsibility: make one streaming LLM call and return a `LLMCallResult`. No conversation management, no agent state. Callers (agskill) decide what to do with the result.

Also owns compaction logic:
- `maybe_compact(prev_ctx, messages, prompt_tokens)` — check threshold, compact if needed
- `compact(messages, ...)` — run a summarization call and replace old messages with the summary
- `_prune_tool_outputs(messages)` — trim oversized tool output content from history

### `agtool` — tool execution

A named callable with a JSON Schema parameter spec. `run_in_subprocess=True` means the tool's function runs in a cloudpickle-serialized worker process (safe isolation). `run_in_subprocess=False` means it runs in the calling thread — used by all sandbox-backed tools, which close over the sandbox object directly and cannot be pickled.

`dispatch_tools(toolkit, tool_calls_raw, sandbox, ...)` — execute all tool calls from one LLM turn, handle output offloading, sandbox commit/restore.

---

## Key Design Decisions

### Why agent doesn't import agskill

`agent.run(skill, input)` calls `skill.run(self, input)` via duck typing. Any object with a `.run(agent, agdata)` method works. This keeps `agent.py` free of upward dependencies and makes it trivially testable without loading the full execution stack.

### Why prev_ctx is a separate argument to execute_react

By the time the background thread calls `execute_react`, `ag.ctx` has already been replaced with the new pending placeholder for this skill run. The actual previous context was captured before the swap and must be passed explicitly. Reading `ag.ctx` inside `execute_react` would return the placeholder for the current call — which would deadlock if `resolve_prev_dependencies()` were called on it.

### Future-based serialization without locks

```
ag.ctx = real_ctx          ← start

skill A starts:
  prev_A = ag.ctx          ← captures real_ctx
  ag.ctx = pending(future_A)

skill B starts before A finishes:
  prev_B = ag.ctx          ← captures pending(future_A)
  ag.ctx = pending(future_B)

A's thread: prev_A.resolve_prev_dependencies() → no-op (real_ctx has no future)
            ... executes ...
            future_A.set_result(ctx_A)

B's thread: prev_B.resolve_prev_dependencies() → blocks on future_A
            ← unblocks when A finishes
            ... executes using ctx_A as base ...
            future_B.set_result(ctx_B)
```

No mutex, no explicit ordering code. The future chain enforces sequential history accumulation automatically, while multiple agents (forks) run fully in parallel because they have independent `ag.ctx` chains.

### No-schema output

When a skill has no `output_schema`, the model's raw text response is returned as `agdata(result=content)` — no JSON parsing. Skills that need structured output must declare an `output_schema`, which causes the framework to inject `return_<field>` tools and enforce the contract.

### Sandbox lifecycle

The sandbox is created lazily on the first skill run that needs one. After each successful tool call (`run_in_subprocess=True` tools), the sandbox is checkpointed via `stop(commit=True)` + the next `_ensure_started()` restoring from that image. After the skill run completes, `agskill` commits + stops the sandbox again in its teardown. `_lifecycle_tag()` always lowercases the Docker image name (Docker requires lowercase repository names).

**Sandbox ownership — no flag, plain attribute**

`agent.sandbox` is a plain instance attribute — no property, no getter/setter, no ownership flag:

- `agskill` always provisions a sandbox when `ag.sandbox is None`, and always calls `sandbox.stop(commit=...)` in its teardown — regardless of whether the sandbox was created by agskill or handed in via `agent(sandbox=...)` / direct assignment (`ag.sandbox = sb`). `stop()` is non-destructive to the Python object: it commits + removes the *container*, and the next access transparently restarts it from that checkpoint.
- `agent.__del__` has no sandbox-specific logic at all. Once nothing references an `agSandbox` instance (the agent that held it is gone, and no one else kept a reference), Python's refcounting collects it and `agSandbox.__del__` (which calls `destroy()`) runs — see `agsandbox.md`. Sharing a sandbox across agents (e.g. a harness handing the same `agSandbox` to two agents in turn) works by simply assigning `ag.sandbox = sb` on each; whoever drops the last reference triggers the real cleanup.

**Per-sandbox mutex — serializing concurrent skill runs on a shared sandbox**

Because agskill no longer special-cases "externally owned" sandboxes, a single `agSandbox` object can legitimately be driven by more than one agent's skill run (e.g. a harness pattern like `examples/sandbox_handoff.py`, or two agents constructed with the same `sandbox=` object). Without serialization, two skill runs racing on the same container could interleave `exec()`/`stop()`/`_ensure_started()` calls and corrupt container state (e.g. one run committing+removing the container while another is mid-`exec`).

`agSandbox.__init__` allocates `self._lock = threading.RLock()` for this purpose. `agskill.py`'s `_task()` acquires it right after provisioning (`sandbox_lock = ag.sandbox._lock; sandbox_lock.acquire()`) and releases it only after the final teardown `stop()` — the lock is held for the *entire* skill run, not just for individual tool calls. Everything the skill run does to the sandbox in between (each tool call's own `stop()`/`_ensure_started()` from `agtool.py`, `wait_for_processes()` polling) runs on that same thread and reentrantly re-acquires the same `RLock` for free.

This lock is **not** self-enforcing on `agSandbox` — calling `sandbox.exec()` (or any other method) directly does not itself acquire the lock. It's `agskill` that establishes the "one skill run owns this sandbox at a time" invariant by holding the lock around the whole run; code that drives a shared sandbox outside of an agskill run (harness scripts, `subgraph_owner.py`-style direct access) is responsible for its own coordination if it needs the same guarantee. See `Design_sandbox_lifecycle.md`'s "Concurrency controls" section and `agsandbox.md` for details.

The lock is intentionally excluded from `agSandbox.__getstate__`/`__setstate__` — `threading.RLock` isn't picklable, and custom tools with `run_in_subprocess=True` (the default) get `cloudpickle`d to a worker process. A fresh lock is created on unpickling; it has no relationship to the original process's lock (locks are process-local by nature, so there was never real cross-process mutual exclusion to preserve).

> **WARNING:** `agSandbox` wraps a live Docker container. Cleanup depends on `agSandbox.__del__` and an `atexit` handler. These do not run on SIGKILL or during interpreter shutdown when `sys.meta_path` has already been nulled. In long-running processes, call `sandbox.destroy()` explicitly when done with the container.

---

## Call Flow: `agent.run(skill, input)`

```
agent.run(skill, input)
  └─ skill.run(ag, input)                         # agskill.run()
       ├─ prev_ctx = ag.ctx                        # capture before swap
       ├─ ag.ctx = agcontext(_future=ctx_future)   # install placeholder
       ├─ thread: _task()
       │    ├─ prev_ctx.resolve_prev_dependencies()
       │    ├─ skill_input.resolve_input_dependencies()
       │    ├─ agSandbox(...) if needed
       │    ├─ skill.execute_react(ag, prev_ctx, skill_input)
       │    │    ├─ agschema.prepare_inputs_in_sandbox(...)
       │    │    ├─ agskill._build_toolkit(ag, ...)
       │    │    ├─ agskill._build_initial_messages(...)
       │    │    └─ for each step:
       │    │         ├─ ag._drain_inbox(messages)
       │    │         ├─ agllm.maybe_compact(prev_ctx, messages, ...)
       │    │         ├─ ag.llm.call(kwargs, messages, ...)  → LLMCallResult
       │    │         │    ├─ kwargs["stream"] = True
       │    │         │    ├─ openai.OpenAI(base_url, api_key).chat.completions.create(**kwargs)
       │    │         │    ├─ _iter_batched(stream, idle_timeout, stream_timeout)
       │    │         │    │    └─ yields chunks; raises _LLMIdleTimeout if silent too long
       │    │         │    ├─ per chunk: accumulate content_parts, reasoning_parts, tool_calls_raw
       │    │         │    │    └─ push partial_msg to live UI via live_messages_fn
       │    │         │    ├─ on BadRequestError (context_length_exceeded): return context_exceeded=True
       │    │         │    ├─ on connection error: retry up to LLM_MAX_RETRIES, then return conn_error
       │    │         │    ├─ call update_ui_token_count_fn(total_in, total_out)
       │    │         │    └─ return LLMCallResult(content_parts, reasoning_parts, tool_calls_raw,
       │    │         │                            prompt_tokens, total_input_tokens,
       │    │         │                            total_output_tokens, elapsed_ms)
       │    │         ├─ (compact if context_exceeded)
       │    │         ├─ agllm.build_assistant_msg(...)
       │    │         ├─ dispatch_tools(toolkit, tool_calls, sandbox, ...)
       │    │         └─ (return if output complete)
       │    ├─ ctx_future.set_result(updated_ctx)
       │    └─ result_future.set_result(result)
       └─ return agdata(_future=result_future)     # non-blocking return
```
