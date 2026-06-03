# Auto-Compaction

Long-running skills accumulate tokens across many tool calls and ReAct iterations. Auto-compaction monitors token usage after each LLM response and — when consumption crosses a threshold — replaces old messages with an LLM-generated summary, freeing headroom for further work without interrupting the skill.

## Trigger

After each LLM response in the ReAct loop, `agskill` checks:

```python
threshold = max(context_limit - 20_000, context_limit // 2)
if prompt_tokens >= threshold:
    compact(...)
```

The trigger fires as late as possible — only when the prompt is within 20K tokens of the context limit. The `// 2` floor prevents nonsensical thresholds on small models (below 40K tokens). This matches opencode's approach of waiting until nearly full rather than compacting at an arbitrary percentage.

Compaction runs at most once per ReAct step (after the LLM fires, before tool dispatch or final-answer processing), so the messages list is always current before the next call.

## Context limit detection

The context limit is determined once at agent creation, in this order:

1. `llm_config["context_limit"]` — explicit user override
2. `GET /v1/models/{model}` → `max_model_len` — vLLM exposes this automatically
3. `None` — compaction is silently disabled

```python
# explicit override (useful for non-vLLM backends)
ag = agent(llm_config={..., "context_limit": 131072}, ...)
```

The detected limit is logged at agent creation:
```
10:00:00  [agent_smith]  [CREATED  ]  ...  context=131072
```
`context=unknown` means compaction is disabled for that agent.

## Algorithm

**File:** `agency/agcompaction.py` · `compact(messages, llm_config, *, context_limit, tail_turns=2, previous_summary=None)`

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
usable = max(context_limit - 20_000, context_limit // 2)
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
[user: "[Conversation history summary…]\n<summary>"]   ← injection
[assistant: "Understood. I'll continue from this context."]
[tail turns verbatim]
```

The task input always appears before the summary so the LLM sees the original goal, then the accumulated context, then the recent turns.

## Effect on history

Compaction modifies the **in-flight `messages` list** inside `agskill.run()`. `agent.history` (the shared cross-skill history) is not affected mid-run. When the skill finishes, `updated_history = agdata(messages=messages[1:])` persists the compacted list (with the summary injection) as the new history for future skill calls.

`agUI`'s interaction pane reflects the compacted list immediately via `_live_messages_fn`.

Note: unlike opencode, which keeps old messages hidden behind a filtered view, agency replaces them in-place. Pre-compaction messages are not recoverable from the running state, but they are preserved in the JSONL log via the `history_before` field of the skill entry.

## Post-skill history pruning

In addition to the in-flight pruning described above, `agent` runs a second pruning pass on `agent.history` after each skill completes. This pass uses the same `_prune_tool_outputs` function with the same thresholds.

**Timing** — the pass runs inside `agent._task()`, between the two futures that gate the dependency chain:

```
result_future.set_result(outer_result)   # caller unblocks immediately

pruned_msgs = _prune_tool_outputs(agent.history.messages)
# rebuild outer_history if any messages were trimmed

history_future.set_result(outer_history) # next skill in chain unblocks with clean history
```

This means:
- The caller receives the skill result as soon as it is ready.
- The next skill that reads `agent.history` waits until pruning is done and then sees a history with oversized tool outputs already trimmed.
- Pruning never blocks the result path — only the history hand-off.

When pruning fires, a terminal log line is emitted:

```
10:00:05  [agent_smith]  [PRUNE    ]  long_task  history pruned to 18 msgs
```

The pruning threshold is the same as in-flight pruning (`_PRUNE_MIN_FREE_TOKENS = 20_000` tokens of potential savings). If the history does not contain enough large tool outputs to cross that threshold, the pass is a no-op.

## Tuning

Constants in `agcompaction.py`:

| Constant | Default | Meaning |
|---|---|---|
| `_RESERVED` | `20_000` | Tokens before the limit at which compaction fires |
| `TAIL_TURNS` | `2` | Max recent assistant turns to keep verbatim |
| `_TAIL_FRACTION` | `0.25` | Fraction of usable context budgeted for the tail |
| `_TAIL_MIN_TOKENS` | `2_000` | Lower bound on tail token budget |
| `_TAIL_MAX_TOKENS` | `8_000` | Upper bound on tail token budget |
| `_TOOL_OUTPUT_MAX_CHARS` | `2_000` | Characters above which a tool result is truncated |
| `_PRUNE_MIN_FREE_TOKENS` | `20_000` | Pruning only runs if it would free at least this many tokens |

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
