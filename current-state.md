# PTY harness unification + Kimi Code — current state

Branch: `eric/kimi-code` (off `refactor-master`)

Two pieces of work landed here, and both are finished: every interactive CLI
harness now runs through one shared PTY runner, and Kimi Code is integrated
end-to-end with portable session resume.

---

## 1. PTY unification (done)

### What it was

`PtyExecution` was the shared runner for Codex, Grok Build and OpenCode, and it
worked by asking "which harness am I?" seventeen separate times — `if
self.name == "codex"`, `elif self.name == "grok"`, and so on, scattered through
the run loop, the submit path, the interrupt path and the transcript scanner.
Claude Code didn't use it at all. It had its own 280-line `_ClaudePtyExecution`
inside `claude_code.py`, a near-copy of the runner with Claude's differences
baked in.

That meant adding a harness was a diff across the runner, and Claude's copy
silently drifted from the original.

### What it is now

```
PtyExecution   — the runner. Knows about PTYs, deadlines, submission,
                 interruption, snapshots. Knows nothing about any CLI.
PtyDriver      — the dialect. One subclass per harness.
```

`PtyExecution` has zero harness names in it. Everything a CLI does differently
is a named hook on the driver, so adding a harness is a new file, not a diff
through shared code.

The subclass tree:

```
PtyDriver
├── _HookPtyDriver          (hook files + session filtering + transcript scan)
│   ├── CodexDriver
│   ├── GrokDriver
│   └── OpencodeDriver      (plugin, not lifecycle hooks)
├── ClaudeDriver
└── KimiDriver
```

`driver_for()` reads `adapter._PTY_DRIVER`, so a driver lives next to its own
adapter and there is no central registry to edit.

### The two axes that turned out to matter

Writing the hooks forced the real differences into the open. There are only
two, and both are now explicit:

**Who owns the turn identity.** Codex, Grok and Kimi mint their own turn IDs
and report them. Claude does not, so Agency mints one itself and writes it
where Claude will echo it back — that is `begin_turn(prompt)`, which returns a
turn ID or `None` for "the CLI will tell us."

**What counts as proof a turn finished.** OpenCode's stop event is enough on
its own. Codex, Grok, Claude and Kimi all need the transcript to agree before a
stop is believed — that is `completed(event)`. Kimi is the extreme case: its
`Stop` hook names no turn at all, so it is stamped with the turn last started
and only honoured once `wire.jsonl` shows a matching `turn.ended`.

Everything else — interrupt key, whether the interrupt needs confirming, the
submission marker format, the timeouts, whether activity extends the deadline —
is a class attribute or a small hook with a working default.

### Two deliberate behaviour changes

Both are tested.

1. **Claude's redirect waits for an empty composer before resubmitting.** It
   used to send the replacement prompt immediately after Escape. If Claude had
   restored a draft into the composer, the new prompt appended to the old one
   and the model saw both. `ClaudeDriver` now sends a second Escape and waits
   for the composer to actually be empty.

2. **A startup failure fails fast instead of waiting out the deadline.** The
   runner ignores events that carry no turn ID, because nothing can complete
   before a turn exists. But a CLI that dies during startup emits exactly that
   — a turn-less error — and the old code swallowed it and sat there until the
   timeout. Now a turn-less `error` before any turn has started fails the
   attempt immediately. After a turn has run, a turn-less event is stale and is
   still ignored, so a late error can't kill a healthy replacement turn.

---

## 2. Kimi Code (done)

Added as `agency/harness/adapters/kimi.py` — `KimiDriver` plus `KimiAdapter`.
All of its values were derived empirically against the real CLI, version
0.42.0.

- Provider is `type = "openai"`, i.e. plain Chat Completions. The formatting
  code that OpenCode already had was extracted into
  `openai_chat_completions.ChatCompletionsProtocol` and both harnesses now inherit it,
  rather than being copied.
- Lifecycle hooks (`SessionStart`, `TurnStarted`, `Stop`, `Interrupt`,
  `StopFailure`, `SessionEnd`) plus the shared permission hook.
- Step budget goes in `[loop_control] max_steps_per_turn`.
- `KIMI_CODE_HOME` relocates config, sessions and credentials together, so an
  attempt's entire footprint stays inside its isolated root.
- The "Trust this folder?" prompt is answered once, guarded by a one-shot flag
  — readiness is polled every 25ms and surplus Enters land in the composer as
  empty submissions.

### Bugs this shook out of shared infrastructure

Integrating a fifth harness found three problems that were not Kimi's.

- **`turn_id` 0 is falsy.** Kimi numbers turns from zero, and the runner's
  adoption check was `if event.get("turn_id")`. Turn 0 was never adopted. All
  turn IDs are now normalized to `str` in the driver.
- **pyte died on a private DSR.** Kimi emits `\x1b[?996n`; pyte 0.8.2's
  `report_device_status` doesn't accept `private=`, so it raised `TypeError`
  and killed the tracer's reader thread — taking terminal observation down for
  whatever harness happened to trigger it. Fixed with a tolerant `Screen`
  subclass in `_tracer_loop.py`.
- **PostToolUse telemetry was silently dropped.** The shared permission hook
  only looked for `tool_use_id` / `tool_output`. Kimi sends `tool_call_id` /
  `tool_response`, so its tool spans never closed. The hook now accepts both
  spellings.

### Verified working

- `examples/01_basic_agent.py` runs green end-to-end on Linux/x86_64 with the
  real Kimi 0.42.0 CLI and Arena's `kimi-k3` model.
- Single-attempt Kimi runs complete in ~1.5s against a stub LLM.
- A real Kimi 0.42.0 process completes a turn, restores only its portable
  session blob into a fresh config home, and completes a second turn under the
  same native session ID.
- The full harness test directory passes on macOS arm64: 374 passed, 65
  optional/platform tests skipped, and the one x86_64-only ptrace test
  deselected.

---

## 3. Kimi resume (fixed)

### Root cause

The absolute paths in `session_index.jsonl` and session JSON needed relocation,
but they were not what cleared the model. The decisive missing state was
Kimi's `workspace-trust/wd_*` record.

Agency accepted Kimi's "Trust this folder?" dialog automatically. In Kimi
0.42.0, accepting that dialog during TUI startup can race with initial model
binding and leave even a cold session with no active model. The same bug is
deterministic when the dialog appears after loading `--session <id>`. Kimi then
displays `LLM not set`, ignores the prompt, and eventually looks like an Agency
input-acknowledgment timeout.

This was isolated by changing one piece of state at a time: removing only
`workspace-trust` broke a same-home resume, while preserving only that record
made a fully relocated fresh-home resume succeed. Changing model aliases,
copying caches, and relocating additional absolute paths did not affect the
failure.

### Fix

Immediately before launch, the Kimi driver writes the exact narrowly scoped
`workspace-trust/wd_<slug>_<hash>` record Kimi would write after Agency accepts
the dialog. This happens after the final working directory is known, avoids the
startup race on cold attempts, and does not make a new trust decision: Agency
already accepted the same dialog unconditionally.

Kimi session bundles also carry these records into resumed attempts. The normal
bundle identity, path traversal, symlink, file-count, and size checks still
apply.

`_relocate_session_index()` continues to repoint Kimi's session directory and
stored home paths at the fresh attempt root.

### Regression coverage

- A unit test snapshots a Kimi session containing the trust marker, restores it
  into a different root, and verifies the marker survives byte-for-byte.
- A unit test verifies the pre-launch marker uses Kimi's exact workspace slug,
  SHA-256 suffix, canonical root, and JSON record shape.
- The real-CLI opt-in test now performs two turns with a fresh config home for
  the second turn and asserts that both use the same Kimi session ID.
- An exact Kimi 0.42.0 diagnostic run against a synthetic OpenAI-compatible
  server completed both attempts, produced the expected answer twice, and sent
  two upstream chat-completion requests.
- The real `examples/01_basic_agent.py` completed through Agency's Podman
  sandbox, ptrace PTY, Kimi Code, Agency's OpenAI-compatible proxy, and Arena
  K3, returning the typed `summary` result and destroying its sandbox cleanly.

---

## 4. Kimi resumed-turn completion (fixed)

### Root cause

Kimi can durably finish a resumed turn without delivering its best-effort
`Stop` lifecycle hook. The completed `turn.ended` row is present in
`wire.jsonl`, including the correct turn ID and completion reason, but the PTY
runner previously consulted that transcript only after receiving `Stop`.
Agency therefore waited until its deadline even though Kimi had already
finished successfully.

This was reproduced against the real Kimi 0.42.0 CLI and Arena K3 during the
multi-turn lifecycle example. The transcript contained `turn.ended` for the
active resumed turn while no matching hook file arrived.

### Fix

`KimiDriver` now exposes every durable `turn.ended` row as a candidate stop
event. The shared PTY controller still accepts only the candidate whose
normalized turn ID matches the currently acknowledged turn, and
`KimiDriver.completed()` still verifies that the transcript reason is
`completed` before reconstructing output and usage. This keeps stale and
cancelled turns from completing the active request while removing the optional
hook as a single point of failure.

The fallback intentionally does not depend on Kimi's last observed hook turn
ID: that bookkeeping can itself be missing in the same failure mode.

### Regression coverage

- A unit test writes multiple ended turns without a `Stop` hook and verifies
  that the driver reports their normalized transcript candidates.
- The real multi-turn lifecycle example now completes queued context,
  dependency fan-through, async work, redirect, pause/resume, and cancellation
  without hanging.

---

## 5. Full example acceptance (passed)

All ten tutorials passed on the EC2 Linux/x86_64 host against Arena's
`kimi-k3`. Harness-specific lessons kept their intentional native or Codex
harness overrides; every LLM-backed path still used K3 through the Arena
OpenAI-compatible endpoint.

1. Basic agent: typed result and history.
2. Context and lifecycle: queued context, dependencies, async, redirect,
   pause/resume, and cancellation.
3. Tools and policy: direct tools, allow/deny policy, host MCP, sandbox MCP,
   and process boundaries.
4. Files, images, and types: typed text/binary/path/raw/custom values and image
   interpretation.
5. Parallel workflows: `agmap`, `wait_all`, concurrent fork/merge, and teams.
6. Configuration and resources: namespaces, redaction, clone isolation,
   mounts, limits, and output paths.
7. Sandbox API: file and process APIs, limits, checkpoint restore, fork, and
   stop.
8. Checkpoints: full save/load and registry save/load.
9. Observability: counters, Perfetto trace, and trace summary.
10. Harnesses and web UI: Codex/native comparison plus a temporary embedded
    Perfetto UI that shut down cleanly.

For lesson 8 only, the sandbox image was changed to the documented lightweight
`python:3.12-slim` example setup. Serializing the global 24 GB Kimi/CUDA image
was spending minutes compressing unrelated image contents; the 79.4 MiB slim
checkpoint exercised the same checkpoint APIs and completed successfully.

The examples retain their five-minute default wait. For slow remote models,
`AGENCY_EXAMPLE_WAIT_TIMEOUT_SECONDS` can now raise that boundary without
editing source; the K3 acceptance run used 900 seconds.

The complete remote acceptance artifacts are under
`/home/eric/agency-kimi-e2e-20260914/runs/all-kimi-k3`. A post-run scan found no
Arena or project API-key patterns in those artifacts or the Perfetto build
tree, and no example process, web server, or Podman container was left running.
