# Logging

Every agent has an `aglog` instance that records a structured timeline of all skill calls, tool invocations, and lifecycle events. Logging is automatic — no manual calls are needed.

## Three views

| Property | Contains |
|---|---|
| `ag.log.entries` | Skill calls only |
| `ag.log.events` | Full timeline: lifecycle + tool calls + skill calls in chronological order |

## Enabling file logging

```python
from pathlib import Path
from src.agent import agent

agent.log_dir = Path("runs/logs")   # set before creating agents
ag = agent(llm_config, agskills=[...])
# writes to runs/logs/<uuid>.jsonl
```

Each agent writes its own JSONL file. Lines are appended atomically under a lock so concurrent skill calls on the same agent are safe.

## Entry types

### Skill call

```json
{
  "type": "skill",
  "ts_start": "2026-06-02T10:00:00.000+00:00",
  "ts_end":   "2026-06-02T10:00:05.123+00:00",
  "skill": "summarize",
  "input": {"text": "..."},
  "output": {"summary": "...", "word_count": 42},
  "history_len": 12,
  "history_before": [...],
  "history_delta": [
    {"role": "system",    "content": "You are a summarizer..."},
    {"role": "user",      "content": "{\"text\": \"...\"}"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool",      "tool_call_id": "...", "content": "..."},
    {"role": "assistant", "content": "{\"summary\": \"...\"}"}
  ]
}
```

`history_delta` is the slice of the message list added during this skill's execution (system prompt + new messages only). `history_before` is the full message list at the moment the skill started.

### Tool call

Each individual tool invocation is logged as a separate entry in `events` (but not `entries`):

```json
{
  "type": "tool",
  "ts": "2026-06-02T10:00:01.042+00:00",
  "tool": "bash",
  "input": {"command": "wc -w file.txt"},
  "output": {"output": "42 file.txt\n", "exit_code": 0, "truncated": false},
  "elapsed_ms": 312
}
```

### Lifecycle events

```json
{"type": "lifecycle", "event": "created",   "ts": "...", "uuid": "..."}
{"type": "lifecycle", "event": "forked",    "ts": "...", "uuid": "...", "parent_uuid": "..."}
{"type": "lifecycle", "event": "destroyed", "ts": "...", "uuid": "..."}
```

Outer monitoring loop events also appear as lifecycle entries:

```json
{"type": "lifecycle", "event": "procs_started",   "uuid": "...", "skill": "train", "pids": [1234], "summary": "..."}
{"type": "lifecycle", "event": "procs_ping",       "uuid": "...", "skill": "train", "pids": [1234], "summary": "..."}
{"type": "lifecycle", "event": "procs_completed",  "uuid": "...", "skill": "train"}
```

## Reading the log

```python
# In-memory access
for entry in ag.log.entries:          # skill calls only
    print(entry["skill"], entry["ts_start"], entry["output"])

for event in ag.log.events:           # full timeline including tool calls
    print(event["type"], event.get("event") or event.get("skill") or event.get("tool"))

# Human-readable dump
print(ag.log.dump())

# Number of completed skill calls
print(len(ag.log))
```

## Terminal output (`agterm`)

In addition to `aglog`, each agent writes colour-coded single-line status messages to stderr via `agterm`. These are for interactive monitoring and are not persisted:

```
10:00:00  [ec041840]  [CREATED  ]  skills=['summarize']  model=claude-opus-4-8
10:00:00  [ec041840]  [SKILL ▶  ]  summarize  input=['text']
10:00:00  [ec041840]  [LLM      ]  model=claude-opus-4-8  messages=3
10:00:01  [ec041840]  [TOOL ✓   ]  bash  rc=0  (312ms)  $ wc -w file.txt
10:00:05  [ec041840]  [SKILL ✓  ]  summarize  output=['summary', 'word_count']
```

Process monitoring events:

```
10:00:05  [ec041840]  [PROCS ▶  ]  train  monitoring: PID 1234 (running 0m 0s)
10:05:05  [ec041840]  [PROCS ⏳  ]  train  still running: PID 1234 (running 5m 0s)
10:07:30  [ec041840]  [PROCS ✓  ]  train  all processes completed, re-entering agent
```
