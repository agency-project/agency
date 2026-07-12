# agterm

`agterm` is the per-agent color-coded real-time terminal logger. Every agent is automatically assigned a distinct ANSI color at creation. All output goes to stderr (or the `agwebui` log panel when the web UI is active) so it never interferes with structured stdout output.

## Output format

```
HH:MM:SS.mmm  [agname]  [EVENT   ]  message          (file.py:lineno)
```

Example output from a running agent:

```
10:00:00.123  [swift_hawk_000]  [CREATED  ]  model=llama-3.1  context=131072
10:00:00.130  [swift_hawk_000]  [SKILL ▶  ]  summarize  input=['text']
10:00:00.400  [swift_hawk_000]  [LLM      ]  model=llama-3.1  messages=2
10:00:01.210  [swift_hawk_000]  [TOOL ✓   ]  bash  rc=0  (42ms)  $ wc -w file.txt
10:00:03.050  [swift_hawk_000]  [COMPACT  ]  skill=summarize  tokens=7820/10000  msgs=42
10:00:05.800  [swift_hawk_000]  [SKILL ✓  ]  summarize  output=['summary', 'word_count']
```

- **Timestamp** — wall-clock time in `HH:MM:SS.mmm`, dimmed
- **Agent tag** — `[agname]` in the agent's assigned ANSI color, bold
- **Event tag** — fixed-width 9-char event label in a greyscale style
- **Message** — event-specific content; other agent names appearing in the message are colorized with their own colors
- **Source** — `(filename:lineno)`, dimmed

## Event labels

| Label | Style | Meaning |
|---|---|---|
| `CREATED  ` | bold | Agent constructed; sandbox started |
| `FORKED   ` | bold | Agent forked from a parent |
| `DESTROYED` | dim | Agent sandbox destroyed |
| `SKILL ▶  ` | bold | Skill started |
| `SKILL ✓  ` | normal | Skill completed successfully |
| `SKILL ✗  ` | reverse | Skill failed (exception or schema error) |
| `PAUSE ▶  ` | — | `agent.pause()` called — request only, not yet honored |
| `PAUSE ✓  ` | — | `agent.resume()` called |
| `ERROR ✗  ` | reverse | `agerror(...)` constructed — emitted immediately at creation |
| `LLM ▶    ` | normal | LLM call dispatched |
| `LLM ✓    ` | normal | LLM call completed |
| `LLM ✗    ` | — | LLM call failed (retrying or exhausted) |
| `TOOL ✓   ` | normal | Tool call completed successfully |
| `TOOL ✗   ` | — | Tool call failed or returned an error |
| `PROCS ▶  ` | — | Background processes detected; `wait_for_processes` polling started |
| `PROCS ⏳  ` | — | Ping timeout elapsed; processes still running — injecting status message |
| `PROCS ✓  ` | — | All processes completed; injecting completion message |
| `COMPACT  ` | — | Context window compacted |
| `PRUNE    ` | — | History pruned |
| `CKPT     ` | — | Checkpoint saved or loaded |

## Enabling and disabling

```python
from agency.agterm import agterm

agterm.enabled = False   # silence all terminal output
agterm.enabled = True    # re-enable (default)
```

This is a class-level flag; it affects all agents in the process.

## Color palette

`agterm` samples the xterm-256 6×6×6 color cube at component levels `{0, 2, 4, 5}`, discarding near-black and grey-diagonal entries to produce 54 visually distinct colors. The palette is shuffled once at process startup so consecutive agents get varied assignments. Colors are assigned in creation order via a shared counter.

Agent names appearing inside log messages from *other* agents are automatically colorized with their own color, making cross-agent references easy to follow in multi-agent runs.

## agwebui integration

When `agwebui.run()` is active, `agterm.log()` routes output based on whether the event label contains `✗`:

- **Error events** (label contains `✗`, e.g. `SKILL ✗  `, `ERROR ✗  `): sent to **both** the web UI emitter and stderr. Errors are always visible regardless of which interface is open.
- **Non-error events**: sent to the web UI emitter only; stderr is silent so it stays uncluttered during interactive sessions.

When the web UI is not active, all events go to stderr.

The emitter appends a `{"type": "log", "line": ...}` event to `ui_events.jsonl`; the standalone server broadcasts it to all connected browsers. Additionally, `agterm.__init__()` emits an `agent_registered` event with the agent's hex color so the browser can display the agent in its assigned color. See [agwebui.md](agwebui.md).
