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
│    ├── starts _poll_commands() thread           │
│    └── calls fn() directly                     │
│                                                 │
│  agterm.log()  ──────────────────────────────┐  │
│  agent._set_ui_state()  ─────────────────────┤  │
│  agent._push_live_messages()  ───────────────┤  │
│  agent._emit_config()  ──────────────────────┤  │
│  agteam.__init__()  ─────────────────────────┤  │
│  ask_human tool  ────────────────────────────┤  │
│                           emit to SQLite DB ◄─┘  │
│               agency/runs/webui_*/               │
│               ui_events.db  ◄───────────────┐   │
│                                              │   │
│  _poll_commands() thread:                        │
│    reads ui_commands/*.json, calls               │
│    agent.pause()/resume()/change_config()        │
└─────────────────────────────────────────────────┘
        │ poll (50 ms)              ▲ poll (200 ms)
        ▼                           │
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
│  WS {"type":"human_reply", ...} →                │
│    writes ui_replies/<ask_id>.txt (unblocks       │
│    ask_human())                                  │
│                                                 │
│  WS {"type":"pause"|"resume"|"pause_all"|         │
│    "resume_all"|"update_config"|                 │
│    "update_config_all", ...} →                    │
│    writes ui_commands/<uuid>.json                │
└─────────────────────────────────────────────────┘
```

The `ui_commands/` channel is the mirror image of `ui_events.db`: events flow execution → server (poll every 50 ms, pushed to browsers); commands flow server → execution (poll every 200 ms in `_poll_commands()`, applied via real `agent`/`agteam` calls). Both are plain files/a SQLite DB in the run directory — there is still no shared memory or direct import between the two processes.

## Usage

```python
from agency.agwebui import agwebui

def main():
    ag = agent(agconfig=..., agname="MyAgent")
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
│  > _                               │ ┌──────────────────┐ │
│                                     │ │ Pause             │ │
│                                     │ │ Pause All          │ │
│                                     │ │ Resume All         │ │
│                                     │ │ Update Config      │ │
│                                     │ └──────────────────┘ │
└────────────────────────────────────┴──────────────────────┘
```

**Left top — Shared Log**: All agent events from all agents, with ANSI colour preserved. Auto-scrolls; pauses when you scroll up.

**Left bottom — Interaction pane**: The focused agent's full message history — system prompt, user turns, assistant thinking, tool calls, tool results. Shows a running indicator while the agent is active. Shows pending `ask_human` questions.

**Right — Agent list**: All live agents grouped by team, with state indicators. Click an agent to focus it. A button bar pinned to the bottom of this column has four actions:
- **Pause** — pauses the currently selected agent (relabels to **Resume** once that agent's state is `paused`)
- **Pause All** / **Resume All** — see [Pause / resume / config commands](#pause--resume--config-commands) below for exactly what "all" reaches
- **Update Config** — opens a modal showing the selected agent's current dynamic (live-editable) config fields; Cancel discards, Update applies to just that agent, Update All applies the same edited values everywhere "all" reaches

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
| `⏸ paused` | red | stopped at its ReAct-loop checkpoint (`agent.is_paused()` is true) |

An agent's real backend state can also be `blocked_on_dependency` (blocked resolving another agent's pending result — see `agent.md`'s "Pause and resume"), but the dashboard has no dedicated indicator for it yet: it currently falls through to the generic `●  &lt;skill&gt; - running` row and counts as "Live", which is misleading since such an agent is making zero forward progress. Known gap, not yet fixed.

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
| `agent_state` | yes | `state, skill, tool` — `state` includes `paused`/`blocked_on_dependency` alongside the original `inactive/skill/llm/tool/proc_wait/human/finished/error` | `agent._set_ui_state()` (thin wrapper around `agent._state.update_state(...)`) |
| `agent_config` | yes | `config: {owner: {field: value}}` — `agConfig.dynamic_snapshot()`, Dynamic-tier fields only | `agent._emit_config()`, called from `__init__`/`fork()`/`load()`/`change_config()` |
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
| `agent_state` | Updates the state indicator (●/○) and skill/tool label next to the agent; also drives the Pause/Resume button's label whenever the currently-selected agent's state changes |
| `agent_config` | Cached client-side per agent (`state.agents.get(agname).config`); read when the Update Config modal opens — no round trip to the execution process needed |
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

## Pause / resume / config commands

The Pause/Pause All/Resume All/Update Config buttons need the reverse flow: server process → execution process. Since the server has no agency imports and can't call `agent.pause()` directly, it uses the same file-drop idiom as `ask_human`, just in the opposite direction:

1. The browser sends one of `{"type": "pause"|"resume", "agname": "..."}`, `{"type": "pause_all"|"resume_all"}`, or `{"type": "update_config"|"update_config_all", "agname": ..., "config": {...}}` over the existing WebSocket.
2. The server writes it, unmodified, to `<run_dir>/ui_commands/<uuid>.json`.
3. A background thread in the execution process (`agwebui._poll_commands()`, started alongside the server subprocess in `agwebui.run()`) polls that directory every 200 ms.
4. For each file, `agwebui._dispatch_command()` parses it and applies it via real objects: `agent.pause()`/`resume()` (looked up by `agname` via `agent.all()`), `agent.change_config()`, or for `update_config_all`, all four targets described below. The file is then deleted.

`update_config_all` specifically reaches four separate objects, each sitting behind its own `agConfig.clone()` boundary (cloning is a one-time snapshot, so a mutation after the fact only reaches whoever hasn't cloned yet):
- **Existing agents** (`agent.all()`) — direct `change_config()`.
- **Existing team instances** (`agteam.all()`) — `agteam.change_config()`, which replaces the team's own live `agconfig` *and* cascades to every agent it tracks.
- **Every `agteam` subclass's class-level `agconfig`** — found via a recursive walk of `agteam.__subclasses__()` (Python's own subclass tracking, no framework registry needed) and mutated in place field-by-field, so a team constructed *after* the click clones fresh data. This is what reaches a user script's own `agconfig = LLM_CONFIG`-style class attribute without the framework needing to know that variable exists.
- **`agent.default_agconfig`** — mutated in place if set, for a bare `agent()` call made with no active team context.

`update_config` (single-agent) only touches that one agent — it does not reach team classes or `default_agconfig`, so you can deliberately run mixed backends across agents.

## Framework hooks

`agwebui` is wired into the framework at these points. All emit-side hooks are guarded with `try/except` so they never affect execution if the web UI is not active. The command-side hooks (`agent.pause()` etc.) don't need guarding — they're plain methods on `agent`/`agteam` called directly by `_dispatch_command()`, not callbacks invoked from inside those classes.

| Hook location | Event emitted / effect |
|---|---|
| `agterm.__init__()` | `agent_registered` |
| `agterm.log()` | `log` (routes instead of stderr) |
| `agent._set_ui_state()` | `agent_state` |
| `agent._push_live_messages()` | `messages_snapshot` |
| `agent._emit_config()` (called from `__init__`/`fork()`/`load()`/`change_config()`) | `agent_config` |
| `agteam.__init__()` (after `setup()`) | `team_registered` |
| `ask_human` tool `fn()` | `ask_human`, file-based reply |
| `agwebui._dispatch_command()` | applies `pause`/`resume`/`pause_all`/`resume_all`/`update_config`/`update_config_all` commands via `agent.pause()`/`resume()`/`change_config()`, `agteam.change_config()` |

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
