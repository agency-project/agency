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

## Shutdown and signal handling

`agwebui.run()` wraps the call to `fn()` in [`agutil.sigterm_as_exit("agwebui")`](agutil.md#sigterm_as_exitlabelagency) (re-exported as `agency.sigterm_as_exit`). Without this, a plain `kill <pid>` (`SIGTERM`) would terminate the process immediately, skipping every `atexit` hook the framework relies on — live sandbox teardown, the tool worker pool, and the `_kill_server()` hook that stops this module's own server subprocess — exactly like `SIGKILL` does. With it, SIGTERM is converted into `SystemExit`, so the same `finally` block that runs on normal completion also runs on `kill`:

- `command_stop.set()` and `ui.emitter.done()` — stop the command-poll thread and mark the run as finished in the UI.
- **Lingering is skipped even if `linger=True`** — `sigterm_as_exit` yields an `Event` that is set when the signal fires; the `finally` block checks it and, if set, goes straight to server teardown instead of blocking on Ctrl+C. A `kill` means "exit now," not "keep serving the dashboard until a second signal arrives."
- The server subprocess is terminated (`proc.terminate()`, waited up to 5 s) and its log file closed.

`SIGINT` (Ctrl+C) needs no special handling here — Python's default handler already raises `KeyboardInterrupt`, which the existing `try/except KeyboardInterrupt` around the linger loop (and the framework's own `finally`/`atexit` hooks elsewhere) handle normally.

**`SIGKILL` cannot be caught by any process**, including this one — there is no handler that can run cleanup in response to it. Recovery from a `SIGKILL`'d run relies on the framework's own self-healing at the *next* run's startup, notably the orphaned-container reaper in `agsandbox_backends/container.py` (see [container.md](agsandbox_backends/container.md#orphaned-container-reaping)), not on anything `agwebui.run()` does.

**The web server subprocess does not propagate signals to it.** Sending `kill <pid>` to the execution process's PID only affects that process; the server subprocess (started via `subprocess.Popen`) is a distinct PID with its own default signal handling (uvicorn's own SIGINT/SIGTERM handling) and is stopped only because the execution process's own cleanup code explicitly calls `proc.terminate()` on it — not through any signal relay from the OS.

**Scripts that don't use `agwebui.run()`** (or `graphui.run()`, its project-specific analog) get none of this protection automatically — a bare script that builds and runs agents directly should wrap its own entry point in `sigterm_as_exit()` the same way:

```python
from agency import sigterm_as_exit

with sigterm_as_exit("my_script"):
    main()
```

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

-- Latest-value tables: one row per key, upserted on every emit() for the
-- corresponding event type (see agwebui_emitter._UPSERT_TABLES). Every
-- event type with a meaningful "current value" gets one of these, in
-- ADDITION to its normal append-only row in `events` above -- the
-- append-only row is what an already-connected client sees live; the
-- upsert is what a client connecting or reconnecting LATER recovers,
-- regardless of how long the run has been going. None of these are ever
-- pruned or subject to TAIL_EVENTS' replay-window cap, since each only
-- ever holds one row per key.
CREATE TABLE agent_tokens (
    agname TEXT PRIMARY KEY,
    data   TEXT NOT NULL   -- most recent token_update JSON
);
CREATE TABLE agent_messages (
    agname TEXT PRIMARY KEY,
    data   TEXT NOT NULL   -- most recent messages_snapshot JSON
);
CREATE TABLE agent_state (
    agname TEXT PRIMARY KEY,
    data   TEXT NOT NULL   -- most recent agent_state (status/skill/tool) JSON
);
CREATE TABLE agent_config_state (
    agname TEXT PRIMARY KEY,
    data   TEXT NOT NULL   -- most recent agent_config JSON
);
CREATE TABLE agent_registry (
    agname TEXT PRIMARY KEY,
    data   TEXT NOT NULL   -- agent_registered JSON
);
CREATE TABLE team_registry (
    team_name TEXT PRIMARY KEY,
    data      TEXT NOT NULL  -- team_registered JSON
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

| `type` | `agname` | Key payload fields | Emitted by | Latest-value table |
|---|---|---|---|---|
| `log` | yes | `line: str` | `agterm.log()` | — |
| `agent_registered` | yes | `color: str` (hex) | `agterm.__init__()` | `agent_registry` |
| `agent_state` | yes | `state, skill, tool` — `state` includes `paused`/`blocked_on_dependency` alongside the original `inactive/skill/llm/tool/proc_wait/human/finished/error` | `agent._set_ui_state()` (thin wrapper around `agent._state.update_state(...)`) | `agent_state` |
| `agent_config` | yes | `config: {owner: {field: value}}` — `agConfig.dynamic_snapshot()`, Dynamic-tier fields only | `agent._emit_config()`, called from `__init__`/`fork()`/`load()`/`change_config()` | `agent_config_state` |
| `team_registered` | no | `team_name, agents: list[str]` | `agteam.__init__()` | `team_registry` (keyed by `team_name`) |
| `messages_snapshot` | yes | `messages: list[dict]` | `agent._push_live_messages()` | `agent_messages` |
| `token_update` | yes | `agent_input, agent_output, global_input, global_output` | `agskill` via `agent.push_token_count_update_to_ui()` after each LLM call completes | `agent_tokens` |
| `resource_update` | no | `gpus_acquired/total, cpus_acquired/total, memory_acquired/total_mb` | `agresources` on acquire/release | `resource_state` |
| `ask_human` | yes | `ask_id, question` | `ask_human` tool | — |
| `human_reply` | yes | `ask_id, reply` | emitter after reply file is read | — |
| `done` | no | — | `agwebui.run()` on completion | — |

Every type with a latest-value table gets its current value upserted there on **every** `emit()`, in addition to its normal append-only row in `events` — see [Latest-value tables](#latest-value-tables-surviving-reconnects) below for why, and what this replaced.

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

### Latest-value tables — surviving reconnects

Six event types (marked in the table above) have a meaningful "current value" — a client only ever cares about the *latest* `agent_config`, not a history of every config it's ever had. Each of these gets its own table (one row per key, upserted in place on every `emit()`) in addition to its normal append-only row in `events`. `_fetch_state_preamble()` reads every one of these tables in full on every client connect — so a client joining or reconnecting at any point in a run recovers the true current value for every agent/team/resource gauge, regardless of how long the run has been going.

This matters because the append-only `events` table alone is not durable enough for that purpose:
- **`TAIL_EVENTS`' replay window is fixed-size** (see [Timeline scrubbing](#timeline-scrubbing) below) — a type that fires rarely (like `agent_config`, once per agent at construction) can fall outside the last-N-events window long before the run ends, once enough higher-frequency events accumulate after it.
- **Pruning downsamples high-frequency types** (see below) — without its own durable table, a type subject to pruning would only ever be recoverable at whatever coarse bucket resolution survived, not its true latest value.

Before this table existed for `agent_config`, `agent_state` (the real status/skill/tool), `agent_registered`, and `team_registered`, a client reconnecting to a long-running batch could see a stale or empty config editor, or an agent list that never fully rebuilt after a server restart — the registration types happened to work most of the time only because they're never pruned and re-scanning the full `events` log for them was cheap, but that was incidental, not a designed guarantee, and it still didn't help `agent_config`, which has no such special-cased scan. One mechanism now covers all six types uniformly.

### Pruning — keeping the database small

`token_update`, `messages_snapshot`, `agent_state`, and `resource_update` (`agwebui_emitter._PRUNE_TYPES`) are emitted at high frequency (every LLM token chunk, every LLM call, every ReAct-loop state transition, every GPU/CPU acquire-release respectively). Storing every row would produce gigabytes for long multi-agent runs.

**Pruning strategy**: every `_PRUNE_EVERY` (default **500**) inserts into the `events` table, the emitter deletes redundant rows for these four types, keeping the last event per `(type, agname, time bucket)` — but only among rows older than the most recent `_PRUNE_KEEP_RAW` (default **500**) by `id`:

```sql
DELETE FROM events
WHERE type IN ('token_update', 'messages_snapshot', 'resource_update', 'agent_state')
  AND id <= (SELECT MAX(id) FROM events) - 500   -- _PRUNE_KEEP_RAW cutoff
  AND id NOT IN (
    SELECT MAX(id) FROM events
    WHERE type IN ('token_update', 'messages_snapshot', 'resource_update', 'agent_state')
      AND id <= (SELECT MAX(id) FROM events) - 500
    GROUP BY type, agname, CAST(ts / 60.0 AS INTEGER)
  )
```

The default bucket size is **60 seconds** (`agwebui_emitter._PRUNE_BUCKET_S`). Within each 60-second window, only the last row per agent per type survives — but a sweep never even considers the most recent `_PRUNE_KEEP_RAW` rows, regardless of their type or bucket. Events of all other types (`log`, `agent_registered`, `agent_config`, `ask_human`, etc.) are **never pruned** at all.

**Why the raw-tail cutoff exists**: `TAIL_EVENTS` (server.py, default **1000**) replays the most recent N rows verbatim to a newly connecting client. Without `_PRUNE_KEEP_RAW`, a sweep firing at just the wrong moment could bucket-compact some rows inside that replay window while leaving others untouched, handing a connecting client a tail that's an inconsistent mix of raw and compacted history for the same type/agent. `_PRUNE_KEEP_RAW` guarantees the newest `_PRUNE_KEEP_RAW` rows are always fully raw. Combined with `_PRUNE_EVERY`'s own accumulation before the next sweep, the raw (never-yet-eligible-for-pruning) tail at any moment is between `_PRUNE_KEEP_RAW` and `_PRUNE_KEEP_RAW + _PRUNE_EVERY` rows — with both at 500, that's **500–1000 rows**, which is exactly why `TAIL_EVENTS` is set to 1000: it's sized to the worst case of that range, so a connecting client's replay window is never partially compacted.

Additionally, the latest value for every type in [Latest-value tables](#latest-value-tables-surviving-reconnects) above is always upserted regardless of pruning, so reconnecting clients always receive the true current state for those types independent of whatever the raw/pruned tail happens to contain — the raw-tail guarantee above matters mainly for narrative continuity (recent `log` lines interleaved with recent `token_update`/`agent_state` history), not for correctness of "what is the current value."

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

**Scrubbing fidelity for pruned types**: because `token_update`, `messages_snapshot`, `agent_state`, and `resource_update` are pruned to one sample per 60-second bucket (outside the always-raw `_PRUNE_KEEP_RAW` tail — see [Pruning](#pruning--keeping-the-database-small) above), a scrub to time T will show the token counts / message history from the last sample at or before T — at most 60 seconds stale. The `WHERE ts BETWEEN first_ts AND end_ts` query naturally picks up the surviving sample as long as `end_ts` is past the bucket boundary.

**Live mode**: clicking the "Live" button or dragging the slider to maximum reloads the page, reconnecting the WebSocket and resuming the live event stream. The server replays the last `TAIL_EVENTS` (default 1000) events on connect, followed by the full latest-value state preamble (see above) — so the client catches up instantly regardless of how long the run has been going.

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
- **Existing agents** (`agent.all()`) — merge the editor payload into each agent's current `agconfig`, then `change_config()` the merged result.
- **Existing team instances** (`agteam.all()`) — same merge-then-`change_config()` pattern; `agteam.change_config()` also cascades to every agent the team tracks.
- **Every `agteam` subclass's class-level `agconfig`** — found via a recursive walk of `agteam.__subclasses__()` (Python's own subclass tracking, no framework registry needed) and mutated in place field-by-field, so a team constructed *after* the click clones fresh data. This is what reaches a user script's own `agconfig = LLM_CONFIG`-style class attribute without the framework needing to know that variable exists.
- **`agent.default_agconfig`** — mutated in place if set, for a bare `agent()` call made with no active team context.

The editor payload is a `dynamic_snapshot()` (LLM knobs and other live fields). Live agents/teams therefore **merge** those fields into their existing `agconfig` rather than replacing it outright — a replace would drop static fields the editor never sends (notably `agSandbox.mounts` / `base_image`), and later `agent.fork()` sandboxes would come up without shared bind mounts.

`update_config` (single-agent) only touches that one agent — it does not reach team classes or `default_agconfig`, so you can deliberately run mixed backends across agents. Same merge rule as above.

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
| `agwebui._dispatch_command()` | applies `pause`/`resume`/`pause_all`/`resume_all`/`update_config`/`update_config_all` commands via `agent.pause()`/`resume()`/`change_config()` (config updates merge into the existing agconfig), `agteam.change_config()` |

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
