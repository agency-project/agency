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
│                               emit to file ◄─┘  │
│                   agency/runs/webui_*/           │
│                   ui_events.jsonl  ◄────────┐   │
└────────────────────────────────────────────────┘
                                             │ tail
┌─────────────────────────────────────────────────┐
│  Web server process (FastAPI + uvicorn)         │
│                                                 │
│  GET  /            → index.html                 │
│  GET  /static/*    → app.js, style.css          │
│  GET  /health      → {"ok": true}               │
│  WS   /ws          → event stream               │
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

## Event types

`agwebui` uses a JSONL event stream (`ui_events.jsonl` in the run directory). Every event has a `type` and `ts` (Unix timestamp).

| `type` | Key fields | Emitted by |
|---|---|---|
| `log` | `line: str` | `agterm.log()` |
| `agent_registered` | `agname: str`, `color: str` (hex) | `agterm.__init__()` |
| `agent_state` | `agname`, `state`, `skill`, `tool` | `agent._set_ui_state()` |
| `team_registered` | `team_name: str`, `agents: list[str]` | `agteam.__init__()` |
| `messages_snapshot` | `agname`, `messages: list[dict]` | `agent._push_live_messages()` |
| `ask_human` | `agname`, `ask_id`, `question` | `ask_human` tool |
| `done` | — | `agwebui.run()` on completion |

The event file is append-only and survives server restarts. New browser tabs receive a full replay of all historical events on connect.

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
