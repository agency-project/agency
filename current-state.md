# PTY harness unification + Kimi Code — current state

Branch: `eric/kimi-code` (off `refactor-master`)

Two pieces of work landed here. The first is finished: every interactive CLI
harness now runs through one shared PTY runner. The second is mostly finished:
Kimi Code is integrated end-to-end, but **its resume path is broken** and that
is the one open problem.

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

## 2. Kimi Code (integrated; resume broken)

Added as `agency/harness/adapters/kimi.py` — `KimiDriver` plus `_KimiBackend`.
All of its values were derived empirically against the real CLI, version
0.42.0.

- Provider is `type = "openai"`, i.e. plain Chat Completions. The formatting
  code that OpenCode already had was extracted into
  `openai_protocol.ChatCompletionsBackend` and both harnesses now inherit it,
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

- `examples/01_basic_agent.py` runs green.
- Single-attempt Kimi runs complete in ~1.5s against a stub LLM.
- 407 harness tests pass, 28 skipped.

---

## 3. The open problem: Kimi resume

**One line: Kimi bakes absolute config-home paths into its session state, so
restoring a session into Agency's fresh per-attempt home leaves it with no
model bound — and it fails silently, as a generic acknowledgment timeout rather
than an error.**

### How it presents

Attempt 1 succeeds in ~1.5s. The engine's second attempt restores the session
blob into a *new* config home and passes `--session <id>`. Kimi comes up
showing:

```
Error: LLM not set, send "/login" to login
Model: stub-model
```

It then ignores the pasted prompt entirely — no `TurnStarted` hook, no error
event, nothing. The runner has no evidence anything is wrong, so it waits the
full 60s input timeout and reports "timed out waiting for native prompt
acknowledgment." The message is true and useless.

### Why it happens

Agency's session model is: snapshot the portable state out of one config home,
restore it into a fresh one next attempt. That contract requires the state to
be relocatable. Kimi's isn't. It stores the absolute path of the home that
wrote it in at least:

- `sessionDir` in `session_index.jsonl`
- `agents.<name>.homedir` in each session's `state.json`
- a `profile.bind` row in `wire.jsonl` recording `modelAlias`

The paths point at a directory that no longer exists, and the model binding
names an alias the fresh config didn't define.

### What has been tried (neither fixed it)

1. **Make the alias be the model name.** The config used to key the model entry
   under a synthetic `agency-proxy` alias. A resumed session restores its
   binding *by name*, so an alias the new config doesn't define leaves nothing
   bound. `config.toml` now keys the model under its own name and `--model`
   passes the same. Correct on its own merits; did not fix resume.
2. **Relocate every absolute path on restore.** `_relocate_session_index()`
   rewrites `sessionDir` in the index and replaces the old root prefix
   throughout every `sessions/**/*.json`. Also correct; also did not fix it.

Still failing, which points at a stored *binding* rather than a stale path —
the next step is to dump the `config.toml` attempt 2 actually generates and
confirm the model name resolves at all inside the resumed session.

### Scope: is this Kimi-only?

Unverified, and worth checking before assuming it is. The class of bug — a CLI
storing absolute paths in state that Agency relocates between attempts — is not
Kimi-specific, and Codex, Grok and OpenCode all resume through the same
snapshot/restore contract. Two cheap checks:

- grep their session bundles for absolute paths
- run the resume reproduction against each of them

If they're clean it's a Kimi quirk. If they aren't, session-state relocation
should become an explicit `PtyDriver` hook rather than something Kimi does ad
hoc in `_restore`.

### Reproducing it

`scratchpad/resume_repro.py` runs attempt 1 fresh, then attempt 2 resuming,
against a stub LLM, in about 30 seconds. It needs the Kimi binary and runs
inside the `kimidbg` podman container. Set `DUMP_CONFIG=1` to print each
attempt's generated `config.toml` and argv.

---

## Notes for whoever picks this up

- Temporary debug instrumentation was added to `pty_session.py` and
  `_pty_hook.py` during this investigation and has been **removed**; neither
  file carries anything from it. Don't re-commit it if you re-add it.
- Podman is installed rootless under `~/.local/podman`. Two packages
  (`golang-github-containers-common` among them) are still unconfigured because
  an `&&` short-circuited the original install; `sudo apt-get -f install` would
  finish the job. Workarounds are in place in the meantime: a user-level
  `~/.config/containers/policy.json`, a symlinked `catatonit`, and
  `CUDA_VISIBLE_DEVICES=-1` to dodge a CDI GPU error.
