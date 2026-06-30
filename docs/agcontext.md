# agcontext

`agcontext` holds persistent conversation state for an agent: the message history, cumulative token counts, and compaction summary. It is passed through every skill run and accumulates across the agent's lifetime.

## When to use / when not to use

Use `agcontext` when you need to inspect or carry forward conversation history and token totals across sequential skill runs on the same agent. Do not construct one manually in most cases — the agent creates and manages the context object; skill code receives it already populated.

## Fields

| Field | Type | Description |
|---|---|---|
| `messages` | `list[dict]` | Conversation history (excludes the per-skill system prompt). Grows with each skill run. |
| `total_input_tokens` | `int` | Cumulative input tokens across all LLM calls on this agent so far. |
| `total_output_tokens` | `int` | Cumulative output tokens across all LLM calls on this agent so far. |
| `compaction_summary` | `str or None` | Rolling summary produced by conversation compaction. `None` if compaction has not yet run. |
| `_future` | `Future[agcontext] or None` | Private. Present when this context is a pending placeholder backed by an in-progress skill run. Resolves to `updated_ctx` — the current skill's finished context. |

## Constructor

```python
ctx = agcontext(
    messages=[],              # optional; defaults to empty list
    total_input_tokens=0,     # optional
    total_output_tokens=0,    # optional
    compaction_summary=None,  # optional
)
```

All parameters are optional. Constructing a bare `agcontext()` gives you an empty, ready-to-use context.

## Methods

### `is_pending() -> bool`

Returns `True` if this context is a placeholder backed by a `Future` that has not yet resolved. A pending context cannot be safely read; call `resolve_prev_dependencies()` first.

```python
if ctx.is_pending():
    ctx.resolve_prev_dependencies()
```

### `resolve_prev_dependencies() -> None`

Blocks until the pending future resolves, then merges the completed context's state (messages, token counts, compaction summary) into `self` and clears `_future`. If the context is not pending, this is a no-op.

This method is the mechanism that serializes concurrent skill runs on the same agent. When `run()` is called, the agent captures the current context as `prev_ctx`, replaces `ag.ctx` with a new pending placeholder that holds a future pointing at `prev_ctx`, and starts the skill's `execute_react` thread. At the start of that thread, `resolve_prev_dependencies()` is called on `prev_ctx` inside the thread — `prev_ctx` is the context captured before `ag.ctx` was replaced — which blocks until the previous skill run's context is fully written and then pulls that state in. This ensures that message history and token totals are always applied in submission order even when skills are dispatched in parallel.

```python
# Inside execute_react (called automatically by the framework):
ctx.resolve_prev_dependencies()  # wait for any prior skill to finish writing state
# ... run ReAct loop, append messages, update token counts ...
```

### `get_resolved_messages() -> list[dict]`

Blocks until any pending future resolves, then returns a snapshot of the message list. Used by `agent.history` to expose the conversation to callers without requiring them to know about the future mechanism.

```python
messages = ag.ctx.get_resolved_messages()  # safe — blocks until in-flight skill finishes
```

### `set_messages(messages: list[dict]) -> None`

Replace the context's message list directly. Used by `agent.history.setter`.

```python
ag.ctx.set_messages([{"role": "user", "content": "reset"}])
```

### `copy() -> agcontext`

Returns a deep copy of this context. If the context is pending, `copy()` blocks by calling `resolve_prev_dependencies()` first, then deep-copies messages and copies the scalar fields into a fresh, non-pending `agcontext`.

```python
snapshot = ag.ctx.copy()
# snapshot.messages is independent of ag.ctx.messages
```

## How agcontext flows through a skill run

1. `ag.run(skill, data)` is called. The agent captures the current `ag.ctx` as `prev_ctx`.
2. A new pending `agcontext` is created with `_future` pointing at a `Future` whose result will be `updated_ctx` (the current skill's finished context) once it completes. This pending object replaces `ag.ctx`.
3. The new skill's `execute_react` thread starts. Its first act is `resolve_prev_dependencies()` called on `prev_ctx` inside the thread — `prev_ctx` is the context captured before `ag.ctx` was replaced — which blocks until the previous skill's final context is available, then merges it in.
4. The ReAct loop runs: messages are appended and token counters are incremented directly on `ctx`.
5. When the skill finishes, `ctx` (now fully populated) is stored back into `ag.ctx`, resolving any future that may be waiting on it.

This design allows `run()` to return immediately while still guaranteeing that all context mutations are applied in order.

## Common patterns

### Reading token usage after a skill run

```python
result = ag.run(skill, data)
_ = result.output          # block until the skill finishes
print(ag.ctx.total_input_tokens, ag.ctx.total_output_tokens)
```

### Taking a snapshot before a branching run

```python
baseline = ag.ctx.copy()   # safe deep copy, blocks if pending
ag.run(skill_a, data_a)
ag.run(skill_b, data_b)
# baseline.messages is unaffected by either run
```

### Checking whether compaction has occurred

```python
if ag.ctx.compaction_summary:
    print("History was compacted:", ag.ctx.compaction_summary[:120])
```

## Constraints and gotchas

- Never read `messages`, `total_input_tokens`, `total_output_tokens`, or `compaction_summary` on a pending context directly. Always call `resolve_prev_dependencies()` first, or use `copy()` which does this for you.
- `_future` is a private implementation detail. Do not set or clear it from outside the framework.
- `copy()` performs a `deepcopy` of `messages`, which can be expensive for long conversations. Use it only when you genuinely need an independent snapshot.
- Token counters are cumulative across the agent's lifetime, not per-skill. To measure the cost of a single skill run, record the counters before and after.
