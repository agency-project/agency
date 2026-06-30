# Logging

Every agent has a `log` instance that records a structured timeline of all skill calls, tool invocations, lifecycle events, and compaction events. Logging is automatic — no manual calls are needed.

## Enabling file logging

```python
from pathlib import Path
from agency import agent

agent.log_dir = Path("runs/logs")   # set before creating agents
ag = agent(llm_config, agskills=[...])
# writes to runs/logs/<agname>_timeline.jsonl  (structured event log)
# writes to runs/logs/<agname>_history.jsonl   (raw LLM message transcript)
```

Each agent writes two JSONL files:

| File | Contents |
|---|---|
| `<agname>_timeline.jsonl` | Structured event log: lifecycle events, tool calls, skill start/end, compaction |
| `<agname>_history.jsonl` | Raw LLM message transcript: every user/assistant/tool message in full |

Lines are appended atomically under a lock so concurrent skill calls on the same agent are safe.

## Reading the log

```python
for entry in ag.log.entries:   # skill calls only
    print(entry["skill"], entry["ts_start"], entry["output"])

for event in ag.log.events:    # full timeline: lifecycle + tools + skills + compaction
    print(event["type"], event.get("event") or event.get("skill") or event.get("tool"))

print(ag.log.dump())   # human-readable
print(len(ag.log))     # number of completed skill calls
```

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

`history_delta` is the slice of the message list added during this skill's execution (system prompt + new messages). `history_before` is the full message list at the moment the skill started.

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
{"type": "lifecycle", "event": "created",   "ts": "...", "agname": "...", "context_limit": 131072}
{"type": "lifecycle", "event": "forked",    "ts": "...", "agname": "...", "parent_agname": "..."}
{"type": "lifecycle", "event": "destroyed", "ts": "...", "agname": "..."}
```

`context_limit` is included in `created` when a context window size was successfully determined at startup (from `llm_config` or the endpoint).

Outer monitoring loop events appear as lifecycle entries:

```json
{"type": "lifecycle", "event": "procs_started",   "agname": "...", "skill": "train", "pids": [1234], "summary": "..."}
{"type": "lifecycle", "event": "procs_ping",       "agname": "...", "skill": "train", "pids": [1234], "summary": "..."}
{"type": "lifecycle", "event": "procs_completed",  "agname": "...", "skill": "train"}
```

### Compaction events

Written whenever the ReAct loop compacts the context window:

```json
{
  "type": "lifecycle",
  "event": "compacted",
  "ts": "...",
  "agname": "...",
  "skill": "long_running_skill",
  "prompt_tokens": 7820,
  "context_limit": 10000,
  "msgs_before": 42,
  "msgs_after": 12
}
```

`msgs_before` / `msgs_after` show how many messages were in the list before and after compaction. The difference (`msgs_before - msgs_after`) is the number of messages replaced by the summary injection. See [compaction.md](compaction.md).

## Terminal output (`terminal`)

In addition to `log`, each agent writes colour-coded single-line status messages to stderr via `terminal`. These are for interactive monitoring and are not persisted:

```
10:00:00  [agent_smith]  [CREATED  ]  skills=['summarize']  model=kimi-k2  context=131072
10:00:00  [agent_smith]  [SKILL ▶  ]  summarize  input=['text']
10:00:00  [agent_smith]  [LLM      ]  model=kimi-k2  messages=3
10:00:01  [agent_smith]  [TOOL ✓   ]  bash  rc=0  (312ms)  $ wc -w file.txt
10:00:03  [agent_smith]  [COMPACT  ]  skill=summarize  tokens=7820/10000  msgs=42
10:00:05  [agent_smith]  [SKILL ✓  ]  summarize  output=['summary', 'word_count']
```

Process monitoring events:

```
10:00:05  [agent_smith]  [PROCS ▶  ]  train  monitoring: PID 1234 (running 0m 0s)
10:05:05  [agent_smith]  [PROCS ⏳  ]  train  still running: PID 1234 (running 5m 0s)
10:07:30  [agent_smith]  [PROCS ✓  ]  train  all processes completed, re-entering agent
```

When `agwebui` is active, `terminal` output is routed to the web UI instead of stderr. See [agwebui.md](agwebui.md).
