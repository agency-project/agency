# agwebui — web-based UI for agency runs

`agwebui` is a browser-based monitoring dashboard for agency runs. It runs in a completely separate process, avoiding any asyncio/multiprocessing conflict with the execution process.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Execution process (main thread, no asyncio)    │
│                                                 │
│  agwebui.run(fn)                                │
│    ├── starts server subprocess                 │
│    ├── sets agwebui._active emitter             │
│    └── calls fn() directly                     │
│                                                 │
│  agterm.log()  ──────────────────────────────┐  │
│  agent._set_ui_state()  ─────────────────────┤  │
│  agent._push_live_messages()  ───────────────┤  │
│  agteam.__init__()  ─────────────────────────┤  │
│  ask_human tool  ────────────────────────────┤  │
│                           emit to SQLite DB ◄─┘  │
│               agency/runs/webui_*/               │
│               ui_events.db  ◄───────────────┐   │
└─────────────────────────────────────────────────┘
                                             │ poll (50 ms)
┌─────────────────────────────────────────────────┐
│  Web server process (FastAPI + uvicorn)         │
│                                                 │
│  GET  /            → index.html                 │
│  GET  /static/*    → app.js, style.css          │
│  GET  /health      → {"ok": true}               │
│  GET  /api/timeline → timeline metadata         │
│  GET  /api/events  → historical event range     │
│  WS   /ws          → live event stream          │
│                                                 │
│  WebSocket clients receive all events           │
│  New clients get full history on connect        │
│                                                 │
│  WS message {type:"human_reply"} →             │
│    writes  ui_replies/<ask_id>.txt  ────────────┼──►  unblocks ask_human()
└─────────────────────────────────────────────────┘
```

## Usage

```python
from agency.agwebui import agwebui

def main():
    ag = agent(llm_config=..., agname="MyAgent")
    result = ag.run(my_skill, agdata(topic="..."))
    print(result.output)

agwebui.run(main)          # http://localhost:7860
agwebui.run(main, port=8080)
agwebui.run(main, linger=False)   # exit immediately when done
```

`agwebui.run(fn)` blocks until `fn` completes and (if `linger=True`) until Ctrl+C. The web server subprocess is started and stopped automatically.

## Layout

```
┌────────────────────────────────────┬──────────────────────┐
│  Shared Log                        │  Agents              │
│  (all agent events, scrolling)     │                      │
│  14:22:01  [StoryManager]          │  StoryManager        │
│    SKILL ▶   design input=[theme]  │  ● design            │
│  14:22:04  [ChapterAgent_1]        │    LLM Wait          │
│    TOOL ▶   bash  $ ls             │                      │
│                                    │  WriterTeam          │
│                                    │    PlannerAgent      │
│                                    │    ● plan            │
│                                    │      tool: write     │
├────────────────────────────────────│    WriterAgent       │
│  ChapterAgent_1  [2/5]  ← →        │    ○ idle            │
│                                    │                      │
│  ─── sys: Create a detailed...     │                      │
│  ▶ user  Based on the theme...     │                      │
│  💭 thinking  The story should...  │                      │
│  ⚙ bash  $ ls /workspace           │                      │
│  ← exit code 0                     │                      │
│  ◆ asst  Here is the chapter...    │                      │
│  ▶ LLM Thinking…                   │                      │
│                                    │                      │
│  [←] [1/5] [→]                     │                      │
│  > _                               │                      │
└────────────────────────────────────┴──────────────────────┘
```

**Left top — Shared Log**: All agent events from all agents, with ANSI colour preserved. Auto-scrolls; pauses when you scroll up.

**Left bottom — Interaction pane**: The focused agent's full message history — system prompt, user turns, assistant thinking, tool calls, tool results. Shows a running indicator while the agent is active. Shows pending `ask_human` questions.

**Right — Agent list**: All live agents grouped by team, with state indicators. Click an agent to focus it.

**Input bar**: Type a reply and press Enter to respond to an `ask_human` request from the focused agent.

## Keyboard shortcuts

| Key | Action |
|---|---|
| `]` or `Tab` | Focus next agent |
| `[` or `Shift+Tab` | Focus previous agent |
| `← →` buttons | Same as `[` / `]` |
| `Enter` (in input) | Submit reply to focused agent |

Tab/Shift-Tab are captured only when the input bar is not focused.

## Agent state indicators

| Symbol | Colour | Meaning |
|---|---|---|
| `○` | dim | idle — no active skill |
| `●` | agent colour | skill running |
| `● LLM Wait` | cyan | waiting for LLM response |
| `● tool-name` | yellow | tool call in progress |
| `● Shell Wait` | dim | waiting for shell processes |
| `● ?` | yellow | `ask_human` waiting for reply |

---

## Event log — SQLite database

All events are written to `ui_events.db` (SQLite, WAL mode) in the run directory.
The execution process is the sole writer; the web server is a read-only consumer.

### Schema

```sql
CREATE TABLE events (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    type   TEXT    NOT NULL,
    agname TEXT,            -- NULL for non-agent events
    ts     REAL    NOT NULL, -- Unix timestamp (float)
    data   TEXT    NOT NULL  -- full JSON payload
);
CREATE INDEX idx_events_ts     ON events(ts);
CREATE INDEX idx_events_type   ON events(type);
CREATE INDEX idx_events_agname ON events(agname);

-- Latest state per agent (for cold-start reconnects)
CREATE TABLE agent_state (
    agname   TEXT PRIMARY KEY,
    tokens   TEXT,   -- most recent token_update JSON
    messages TEXT    -- most recent messages_snapshot JSON
);

-- Latest resource pool state (single row)
CREATE TABLE resource_state (
    id   INTEGER PRIMARY KEY CHECK (id = 1),
    data TEXT NOT NULL
);
```

WAL mode (`PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL`) allows the server to read concurrently while the execution process inserts, with no write blocking.

### Event types

Every event has `type` and `ts`. `agname` is present for per-agent events.

| `type` | `agname` | Key payload fields | Emitted by |
|---|---|---|---|
| `log` | yes | `line: str` | `agterm.log()` |
| `agent_registered` | yes | `color: str` (hex) | `agterm.__init__()` |
| `agent_state` | yes | `state, skill, tool` | `agent._set_ui_state()` |
| `team_registered` | no | `team_name, agents: list[str]` | `agteam.__init__()` |
| `messages_snapshot` | yes | `messages: list[dict]` | `agent._push_live_messages()` |
| `token_update` | yes | `agent_input, agent_output, global_input, global_output` | `agskill` via `agent.push_token_count_update_to_ui()` after each LLM call completes |
| `resource_update` | no | `gpus_acquired/total, cpus_acquired/total, memory_acquired/total_mb` | `agresources` on acquire/release |
| `ask_human` | yes | `ask_id, question` | `ask_human` tool |
| `human_reply` | yes | `ask_id, reply` | emitter after reply file is read |
| `done` | no | — | `agwebui.run()` on completion |

### What each event type drives in the UI

| `type` | UI effect |
|---|---|
| `log` | Appended to the Shared Log with timestamp and agent colour |
| `agent_registered` | Adds agent to the agent list; establishes colour used for all its events |
| `agent_state` | Updates the state indicator (●/○) and skill/tool label next to the agent |
| `team_registered` | Creates a team group in the agent list; agents are indented under it |
| `messages_snapshot` | Replaces the full message history shown in the Interaction pane |
| `token_update` | Updates the token counter badge in the header |
| `resource_update` | Updates the GPU/CPU/memory badge (e.g. "GPU 2/4 (Used/Total)") |
| `ask_human` | Highlights the agent in the list; shows the question in the Interaction pane |
| `human_reply` | Clears the pending question indicator |
| `done` | Marks the run as complete in the header |

### Pruning — keeping the database small

`token_update` and `messages_snapshot` are emitted at high frequency (every LLM token chunk and every LLM call respectively). `resource_update` fires on every GPU/CPU acquire and release. Storing every row would produce gigabytes for long multi-agent runs.

**Pruning strategy**: every 500 inserts into the `events` table, the emitter deletes redundant rows for these three types, keeping the last event per `(type, agname, time bucket)`:

```sql
DELETE FROM events
WHERE type IN ('token_update', 'messages_snapshot', 'resource_update')
  AND id NOT IN (
    SELECT MAX(id) FROM events
    WHERE type IN ('token_update', 'messages_snapshot', 'resource_update')
    GROUP BY type, agname, CAST(ts / 60.0 AS INTEGER)
  )
```

The default bucket size is **60 seconds** (`agwebui_emitter._PRUNE_BUCKET_S`). Within each 60-second window, only the last row per agent per type survives. Events of all other types (`log`, `agent_state`, `agent_registered`, etc.) are **never pruned**.

Additionally, the latest value for each pruned type is always upserted into `agent_state` / `resource_state` regardless of pruning, so reconnecting clients always receive the current state.

**Storage budget (60 s buckets, 2-hour run, 100 agents)**:

| Type | Rows kept | Typical row size | Total |
|---|---|---|---|
| `token_update` | 2 h × 60 s⁻¹ × 100 agents = 12 000 | ~200 B | ~2.4 MB |
| `messages_snapshot` | 2 h × 60 s⁻¹ × 100 agents = 12 000 | ~5–50 KB (grows with history) | 60 MB–600 MB |
| `resource_update` | 2 h × 60 s⁻¹ × 1 = 120 | ~200 B | ~24 KB |

`messages_snapshot` dominates. Increase `_PRUNE_BUCKET_S` (e.g. to 300 s) to reduce it at the cost of coarser scrubbing fidelity for message history.

---

## Timeline scrubbing

The browser shows a timeline slider covering the full run duration. Dragging it replays a historical snapshot of agent state at the chosen moment.

### How it works

**Server side** (`GET /api/timeline`): returns `{first_ts, last_ts, index_len, samples}` where `samples` is a list of `[i, ts]` pairs sampled evenly across the event stream (one per ~500 events). The browser uses these to map slider position → timestamp.

**Browser side** (`enterHistoricalMode`):
1. Maps slider position to a timestamp `end_ts` using linear interpolation over `samples`.
2. Calls `GET /api/events?start_ts=<first_ts>&end_ts=<end_ts>`.
3. Clears current UI state and replays every returned event through the normal event handler.

This gives an accurate snapshot of all agent states, log lines, and token counts as they existed at `end_ts`.

**Scrubbing fidelity for pruned types**: because `token_update`, `messages_snapshot`, and `resource_update` are pruned to one sample per 60-second bucket, a scrub to time T will show the token counts / message history from the last sample at or before T — at most 60 seconds stale. The `WHERE ts BETWEEN first_ts AND end_ts` query naturally picks up the surviving sample as long as `end_ts` is past the bucket boundary.

**Live mode**: clicking the "Live" button or dragging the slider to maximum reloads the page, reconnecting the WebSocket and resuming the live event stream. The server replays the last 500 events on connect so the client catches up instantly.

---

## ask_human path

When an agent calls `ask_human` while `agwebui` is active:

1. The tool emits an `ask_human` event with a unique `ask_id`.
2. The tool then polls `<run_dir>/ui_replies/<ask_id>.txt` (200 ms sleep loop).
3. The web server pushes the event to connected browsers.
4. The browser switches focus to the asking agent and highlights the question.
5. The user types a reply in the input bar and presses Enter.
6. The browser sends `{"type": "human_reply", "ask_id": "...", "text": "..."}` over WebSocket.
7. The server writes `<run_dir>/ui_replies/<ask_id>.txt`.
8. The polling loop wakes up, reads the reply, deletes the file, and returns the text.

This path is entirely file-based — no shared memory between the execution process and the web server.

## Framework hooks

`agwebui` is wired into the framework at four points. All hooks are guarded with `try/except` so they never affect execution if the web UI is not active.

| Hook location | Event emitted |
|---|---|
| `agterm.__init__()` | `agent_registered` |
| `agterm.log()` | `log` (routes instead of stderr) |
| `agent._set_ui_state()` | `agent_state` |
| `agent._push_live_messages()` | `messages_snapshot` |
| `agteam.__init__()` (after `setup()`) | `team_registered` |
| `ask_human` tool `fn()` | `ask_human`, file-based reply |

## Dependencies

`agwebui` adds two dependencies to the project:

- `fastapi >= 0.100.0`
- `uvicorn[standard] >= 0.20.0`

The server process uses only these and the Python standard library — it has no imports from `agency`.

## Running the server standalone

The server can be started independently against a completed run directory to browse the event history:

```bash
python -m agency.agwebui.server --run-dir runs/webui_20260612_143000 --port 7860
```
