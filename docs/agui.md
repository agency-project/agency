# agUI — Terminal User Interface

`agUI` is an optional split-screen TUI for monitoring and interacting with running agents in real time. It is built on [Textual](https://textual.textualize.io/) and falls back gracefully to plain stdout/stdin when not used.

## Layout

```
┌─────────────────────────────────────────┬────────────────────┐
│  Shared Log                             │ Agent List         │
│                                         │                    │
│  10:00:00  [swift_hawk]  [CREATED  ]    │ ● swift_hawk       │
│  10:00:01  [swift_hawk]  [SKILL ▶  ]    │   summarize: bash  │
│  10:00:02  [swift_hawk]  [TOOL ✓   ]    │                    │
│                                         │ ○ calm_mole        │
│                                         │   idle             │
├─────────────────────────────────────────┤                    │
│  swift_hawk  [1/2]  Tab/S-Tab           │                    │
│                                         │                    │
│  ▶ user  {"topic": "KV cache"}          │                    │
│  ⚙ bash                                 │                    │
│    command: ls /workspace               │                    │
│  ← {"output": "report.md\n", ...}       │                    │
│  ▶ LLM Thinking..                       │                    │
│  > _                                    │                    │
└─────────────────────────────────────────┴────────────────────┘
```

- **Top-left — Shared Log**: all `agterm` output from all agents, equivalent to what would go to stderr without the UI.
- **Bottom-left — Interaction pane**: the selected agent's live chat context history. Refreshes every second while the agent is active. Cycles with Tab/Shift-Tab.
- **Right — Agent List**: all live agents with their name and current state, color-coded by agent identity.

## Usage

`agUI.run(fn)` is the only entry point. Textual must run in the main thread (it installs terminal signal handlers); the user script runs in a daemon worker thread.

```python
from agency import agent, agskill, agdata, agUI

def main():
    ag = agent(llm_config=LLM_CONFIG, agskills=[my_skill])
    result = ag.run("my_skill", agdata(task="..."))
    print(result.answer)

if __name__ == "__main__":
    agUI.run(main)
```

`linger=True` (default) keeps the TUI open after the script finishes and shows a "Done — press q" banner. Set `linger=False` to exit immediately when the script returns.

Without `agUI.run()`, the framework works identically: `agterm` output goes to stderr, `ask_human` falls back to `input()`, and `print()` goes to stdout.

## Keyboard bindings

| Key | Action |
|---|---|
| `Tab` | Cycle forward through agents in the interaction pane |
| `Shift-Tab` | Cycle backward |
| `q` | Quit |
| `Ctrl-C` | Quit |
| `Enter` | Submit a message in the input box |

## Interaction pane

### Viewing agent history

The bottom-left pane shows the live `_snapshot_messages` list of the selected agent — the same message list the LLM sees on the next call. It is updated after every LLM response and every tool result. Message types:

| Prefix | Meaning |
|---|---|
| `─── sys: ...` | System prompt (first line, dimmed) |
| `▶ user` | User input or skill input JSON |
| `⚙ tool_name` | LLM tool call with arguments |
| `← result` | Tool result (dimmed) |
| `◆ asst` | LLM text response |
| `▶ LLM Thinking...` | Agent is waiting for LLM (cycling dots) |
| `▶ Tool Running: bash...` | Agent is executing a tool (cycling dots) |
| `▶ Waiting for processes...` | Agent is in the outer monitoring loop (cycling dots) |

### Sending messages to an agent

Type in the input box and press Enter to send a message to the currently selected agent.

**Solicited replies** (`ask_human` tool): when an agent calls `ask_human`, the UI auto-switches to that agent, displays the question, and blocks the agent thread until the user replies. The reply is returned as the tool result.

**Unsolicited messages**: messages typed when the agent has not called `ask_human` are injected into `agent._inbox`. They are drained and appended as user turns at the top of the next ReAct iteration, before the LLM fires. The LLM responds to the message in-context; output schema validation is skipped for that step.

## Agent List panel

Each entry shows the agent's name (color-coded) on the first line and `<skill>: <status>` on the second:

| State | Display |
|---|---|
| `inactive` | `○ name` / `  idle` (dimmed) |
| `llm` | `● name` / `  skill: Waiting LLM` (cyan) |
| `tool` | `● name` / `  skill: tool_name` (yellow) |
| `proc_wait` | `● name` / `  skill: Waiting shell` (dimmed) |
| `human` | `● name ?` / `  skill: Waiting human` (yellow) |
| `skill` | `● name` / `  skill - running` |

Agent colors are derived from the `agterm` color assignment (each agent gets a distinct ANSI color at creation; the UI maps these to Rich color names). Destroyed agents are removed from the list automatically on the next 1-second refresh tick.

## stdout routing

While `agUI.run()` is active, `sys.stdout` is replaced with `_UIWriter`, which buffers output and routes complete lines to the shared log pane. `print()` in user scripts therefore appears in the TUI rather than the terminal. `sys.stdout` is restored when the worker thread exits. Tracebacks from unhandled exceptions in the worker are caught and written to the shared log before restoring stdout.

## `agterm` routing

`agterm.log()` checks for an active `agUI` instance and calls `agui._active.add_log(line)` instead of writing to stderr. This means all `[SKILL ▶]`, `[TOOL ✓]`, `[COMPACT]`, and similar lines appear in the shared log pane.

## Architecture

```
main thread                          worker thread
──────────────────────────────────────────────────────────
agUI.run(fn)
  ├─ create _AgencyApp
  ├─ start worker thread (daemon) ──────────────────▶ ready.wait()
  ├─ app.run()  ← Textual owns terminal              _active = ui
  │   installs signal handlers                       sys.stdout = _UIWriter
  │   starts 1s interval timer                       fn()  ← user script
  │   sets ready_event                               │
  │                                                  │  agent threads running
  │   timer fires every 1s:                          │  _snapshot_messages updated
  │     _refresh_agent_list()                        │  _ui_state updated
  │     _render_history()                            │  _inbox drained
  │                                                  │
  │   on Input.Submitted:                            │
  │     solicited  → reply_q.put(text)               │
  │     unsolicited → agent._inbox.put(text)         │
  │                                                  fn() returns
  │   mark_done() banner                             _active = None
  │                                                  app.exit() / linger
  └─ join worker thread
```

Textual's `call_from_thread()` is used for all cross-thread UI updates — `add_log`, `_add_agent_msg`, `_post_question`. Direct widget access from non-main threads is not allowed by Textual.
