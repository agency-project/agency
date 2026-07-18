# Auto-Compaction

Long-running skills accumulate tokens across many tool calls and ReAct iterations. Auto-compaction monitors token usage after each LLM response and — when consumption crosses a threshold — replaces old messages with an LLM-generated summary, freeing headroom for further work without interrupting the skill.

## Pre-call token estimation

Before each LLM call, `agskill` computes a prompt size estimate used for two purposes:

1. **Web UI token display** — the running token count is updated immediately, so the UI shows growth even if the call fails.
2. **`max_tokens` clamping** — `max_tokens` is reduced so that `prompt + max_tokens` never exceeds `context_limit`.

The estimate uses the fast chars/4 heuristic (`estimate_messages_tokens`) over the full message list. No tokenize endpoint call is made pre-call.

```python
_pre_estimate = estimate_messages_tokens(messages)  # sum((len(content) + len(tool_call_args)) // 4) for all messages
```

### `max_tokens` clamping

```python
_headroom = max(1, context_limit - _pre_estimate)
if kwargs.get("max_tokens", _headroom) > _headroom:
    kwargs["max_tokens"] = _headroom
```

The configured `max_tokens` is only ever reduced, never increased. If the chars/4 estimate is optimistic (underestimates token-dense content), the API may still return a 400 context-exceeded error — this is handled reactively (see below).

## Reactive compaction on context-exceeded errors

The chars/4 estimate can underestimate token-dense content (code, JSON, base64). When this causes the proactive compaction check to miss, the API returns a 400 `BadRequestError` with a context-length message. `agskill` detects this and triggers forced compaction before retrying:

```python
# In ag.llm.call():
except openai.BadRequestError as e:
    if any(kw in str(e).lower() for kw in ("context_length_exceeded", "maximum context length", "context length", "too long", "reduce the length")):
        return LLMCallResult(context_exceeded=True)

# In the ReAct loop (agskill.execute_react):
if llm_result.context_exceeded:
    messages, _ = ag.llm.maybe_compact(prev_ctx, messages, None, ..., force=True)
    continue  # retry the same step
```

`force=True` bypasses the `should_compact` threshold check so compaction always runs regardless of the estimated token count. The step counter is not incremented — the same ReAct iteration is retried after compaction.

---

## Trigger

After each LLM response in the ReAct loop, `agskill` checks:

```python
# _AgLLMFields.COMPACT_THRESHOLD = 0.85 (hardcoded, no agconfig override)
if _AgLLMFields.should_compact(prompt_tokens, context_limit):
    ag.llm.maybe_compact(...)
```

The trigger fires when the prompt reaches 70% of the context limit, leaving 30% headroom for the summary injection and continued work.

Compaction runs at most once per ReAct step (after the LLM fires, before tool dispatch or output collection), so the messages list is always current before the next call.

## Context limit detection

The context limit is determined once at agent creation, in this order:

1. `cfg.agllm_backend.context_limit` — explicit user override
2. `GET /v1/models` → `max_model_len` — vLLM exposes this on the model list response. `list()` is always used instead of `retrieve()` because model names containing `/` would produce a malformed URL with `retrieve()`.
3. `DEFAULT_CONTEXT_LIMIT` (128 000) — safe fallback so compaction always runs even when the API is unreachable at agent creation time.

```python
# explicit override (useful for non-vLLM backends)
cfg.agllm_backend.context_limit = 131072
ag = agent(agconfig=cfg)
```

The detected limit is logged at agent creation:
```
10:00:00  [agent_smith]  [CREATED  ]  ...  context=131072
```

## Algorithm

**File:** `agency/agllm.py` · `ag.llm.compact(messages, *, context_limit, tail_turns=2, previous_summary=None)`

Compaction runs in three sequential phases before the summary LLM call.

### Phase 1 — Separate the task input

`messages[0]` (system prompt) and `conv[0]` (the skill's original user task input) are always preserved verbatim and are never summarised. The task input is also included in the summarisation prompt so the LLM knows the original goal when writing the summary.

```
messages = [system] + [user: task input] + [... ReAct turns ...]
                ↑              ↑
          always kept     always kept, also passed to summariser
```

### Phase 2 — Select the verbatim tail

The most recent assistant turns are kept verbatim. The tail is sized by token budget rather than a fixed turn count:

```
tail_budget = clamp(usable * 0.25, min=2_000, max=8_000)   # tokens
usable = int(context_limit * 0.85)   # matches _AgLLMFields.COMPACT_THRESHOLD
```

Working backwards, turns (one assistant message + its immediately following tool results) are added to the tail until either:
- `tail_turns` (default 2) turns are accumulated, or
- the next turn would exceed `tail_budget`

A single oversized turn is always accepted — the budget only blocks adding a *second* turn, so the tail is never empty.

This is equivalent to opencode's token-budget tail retention.

### Phase 3 — Prune large tool outputs in the head

Before summarising, tool results in the head that exceed 2 000 characters are truncated:

```
[tool result]: xxxxxxxxxxxxxxx...  [truncated]
```

Pruning only activates when the total potential savings would reach 20 000 tokens (roughly 80 KB of tool output). This avoids unnecessary mutation for typical-sized outputs and matches opencode's pruning threshold. Non-tool messages (assistant, user) are never pruned.

### Summary generation

The head (everything between the task input and the tail) is sent to the LLM with a structured prompt asking for a seven-section markdown summary:

```markdown
## Goal
<one sentence describing the overall task>

## Constraints & Preferences
<coding style, output format, naming conventions, user instructions>

## Progress
- Done: ...
- In progress: ...
- Blocked: ...

## Key Decisions
<decisions made and the reasons>

## Next Steps
<ordered list of what remains>

## Critical Context
<variable values, flags, API responses, invariants>

## Relevant Files
<every file path created, read, or modified>
```

All seven sections are required in the output even if empty, so the LLM cannot silently omit a category.

### Incremental updates

When compaction fires more than once in a single skill run, the previous summary is passed as an anchor:

> *"Previous summary (update it — keep true facts, remove stale ones, add new ones): …"*

The LLM merges the new head into the existing summary rather than starting from scratch. This prevents critical context from being silently dropped across multiple compactions on the same skill run.

### Re-injection

The compacted head is replaced by a two-message exchange injected into the message list:

```
[system]
[user: task input]          ← preserved from original
[user: "[HARNESS SYSTEM] [Conversation history summary — treat as established context, do not ask to re-confirm]\n<summary>"]   ← injection
[assistant: "[HARNESS SYSTEM] Understood. I'll continue from this context."]
[tail turns verbatim]
```

The task input always appears before the summary so the LLM sees the original goal, then the accumulated context, then the recent turns.

## Effect on context

Compaction modifies the **in-flight `messages` list** inside `agskill.execute_react()`. `agent.ctx` (the shared cross-skill context) is not affected mid-run. When the skill finishes, the compacted list (with the summary injection) is persisted into `updated_ctx` and passed to the next skill via the `ctx_future`.

The web UI reflects the compacted list immediately via the `_live_messages_fn` callback passed to `ag.llm.maybe_compact()`.

Note: unlike opencode, which keeps old messages hidden behind a filtered view, agency replaces them in-place. Pre-compaction messages are not recoverable from the running state, but they are preserved in the JSONL log via the `history_before` field of the skill entry logged by `ag.log._record()`.

## Post-skill history pruning

In addition to the in-flight pruning described above, `agskill.run()` runs a second pruning pass on `updated_ctx` after each skill completes. This pass uses the same `agllm._prune_tool_outputs()` function with the same thresholds.

**Timing** — the pass runs inside the `_task()` closure in `agskill.run()`, between the two futures that gate the dependency chain:

```
result_future.set_result(outer_result)   # caller unblocks immediately

pruned_msgs = agllm._prune_tool_outputs(updated_ctx.messages)
# update updated_ctx.messages if any messages were trimmed

ctx_future.set_result(updated_ctx)       # next skill in chain unblocks with clean ctx
```

This means:
- The caller receives the skill result as soon as it is ready.
- The next skill that reads `agent.ctx` waits until pruning is done and then sees a context with oversized tool outputs already trimmed.
- Pruning never blocks the result path — only the context hand-off.

When pruning fires, a terminal log line is emitted:

```
10:00:05  [agent_smith]  [PRUNE    ]  long_task  history pruned to 18 msgs
```

This log is emitted via `ag.terminal.log("PRUNE    ", ...)` inside `_task()`.

The pruning threshold is the same as in-flight pruning (`_AgLLMFields.PRUNE_MIN_FREE_TOKENS = 20_000` tokens of potential savings). If the history does not contain enough large tool outputs to cross that threshold, the pass is a no-op.

## Tuning

All defined on `_AgLLMFields` in `agllm.py`. Only `tail_turns` is actually adjustable per-agent (it's a `DynamicConfigParam`, settable via `agconfig.set("agllm", "tail_turns", ...)` or `cfg.agllm.tail_turns = ...`); the other five are plain hardcoded class constants with no agconfig override — despite the section title, changing them means editing `agllm.py` itself, not passing a different `agconfig`.

| Constant | Default | Meaning |
|---|---|---|
| `_AgLLMFields.tail_turns` (`DynamicConfigParam`, live-configurable) | `2` | Max recent assistant turns to keep verbatim |
| `_AgLLMFields.TAIL_FRACTION` (hardcoded) | `0.25` | Fraction of usable context budgeted for the tail |
| `_AgLLMFields.TAIL_MIN_TOKENS` (hardcoded) | `2_000` | Lower bound on tail token budget |
| `_AgLLMFields.TAIL_MAX_TOKENS` (hardcoded) | `8_000` | Upper bound on tail token budget |
| `_AgLLMFields.TOOL_OUTPUT_MAX_CHARS` (hardcoded) | `2_000` | Characters above which a tool result is truncated |
| `_AgLLMFields.PRUNE_MIN_FREE_TOKENS` (hardcoded) | `20_000` | Pruning only runs if it would free at least this many tokens |

## Logging

Each compaction fires a terminal log line and a structured lifecycle event:

```
10:00:03  [agent_smith]  [COMPACT  ]  skill=long_task  tokens=108200/128000  msgs=42
```

```json
{
  "type": "lifecycle",
  "event": "compacted",
  "agname": "agent_smith",
  "skill": "long_task",
  "prompt_tokens": 108200,
  "context_limit": 128000,
  "msgs_before": 42,
  "msgs_after": 12
}
```

`msgs_before - msgs_after` is the number of messages replaced by the summary injection (typically a large number; the injection itself adds 2 messages).
