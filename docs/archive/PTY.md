# External harness PTYs

Claude Code, Codex, Grok Build, Kimi Code, and OpenCode all run their interactive
terminal interfaces under the daemon's existing ptrace supervisor, through one runner.
Native uses its standalone process, pipe transport, model loop, and session
format; it never enters the PTY runner.

## Ownership and lifecycle

The public agent and orchestrator still select an engine. The engine owns host
services and sends a typed attempt over the existing Unix socket. Inside the
sandbox, `HarnessManager` selects the adapter, which prepares isolated CLI state
and starts `PtyExecution`. With image-commit or filesystem-only COW checkpoints,
that runner owns a fresh PTY process for one attempt. With COW fast resume enabled,
the manager retains the successful attempt's idle CLI and PTY for the next turn
of the same agent. It retires the process on failure or incompatible model,
step-limit, tool, or syscall-policy configuration changes. Agents never share a CLI.

Hibernation first detaches Agency's ptrace supervisor, then snapshots the mutable
filesystem and saves the live process tree through the runtime's CRIU support.
Restore reattaches ptrace before accepting another prompt through the same PTY.
The ZFS snapshot remains the durable checkpoint: if live restore fails, the
engine starts a fresh harness and resumes its persisted native session.

Fast-resume MCP endpoints use stateless JSON responses and decline optional GET
event streams with HTTP 405. An idle CLI can otherwise hold a stream open across
turns and prevent the host's attempt leases from draining before checkpointing.

The shared runner handles bracketed-paste input, native acknowledgment deadlines,
pause-aware timeouts, redirects, completion, and process retirement. It contains
no per-harness branches. `PtyDriver` is the dialect contract, with one subclass
per harness; `driver_for()` reads the class off the adapter's `_PTY_DRIVER`, so a
new harness is a new driver beside its adapter rather than an edit to shared code.
`_HookPtyDriver` carries what the hook-and-rollout CLIs (Codex, Grok) share. Hooks
and the OpenCode plugin only observe lifecycle events and enforce tool policy;
prompts and interrupts travel through the PTY, not an alternate headless server
or prompt-injection HTTP endpoint.

Two axes vary enough to be explicit driver hooks. **Turn identity**: most CLIs
assign it and report it on submission, so the runner learns it from the
acknowledgment; Claude assigns none, so `begin_turn()` commits one to
`agency-turn.json` for its lifecycle hook to stamp events with. **Completion
evidence**: a stop event is sufficient for OpenCode, while Codex, Grok, Kimi, and
Claude must also reconcile it against the persisted transcript before the turn
counts as finished. Kimi is the extreme case -- its `Stop` names no turn at all,
so the transcript is the only thing that can attribute a completion.

The terminal screen establishes whether the composer is ready. It is not the
source of final answers. Submissions include an Agency marker, and native
acknowledgments must match the complete prompt. Completion and redirection share
a lock. Delayed events retain their original native turn IDs; they cannot finish
or acknowledge a replacement turn. If native completion wins an interrupt race,
the result is preserved and the redirect falls back to queued context. A failure
reported before the first prompt still fails the attempt immediately rather than
waiting out the startup deadline; after any turn has run, a turn-less event is
stale by definition and is discarded.

| Harness | Submission and completion | Interrupt | Persisted state |
| --- | --- | --- | --- |
| Claude Code | Agency-assigned turn ID in `agency-turn.json`; `UserPromptSubmit`/`Stop` hooks, then a transcript showing the prompt and the final assistant message (or a `turn_duration` marker when the turn ends after tool output) | Escape; a `[Request interrupted by user]` transcript row confirms, or a restored draft is cleared with a second Escape pair | Native session transcript JSONL |
| Kimi Code | `TurnStarted` carries the turn ID and prompt; `Stop` carries neither and is stamped with the turn last started, then only counts once `wire.jsonl` records a matching `turn.ended` with reason `completed`; the answer and usage come from that turn's `content.part`/`step.end` events | Escape, confirmed by the `Interrupt` hook, which does report its turn | `session_index.jsonl` plus the session directory's JSON/JSONL |
| Codex | `UserPromptSubmit`/`Stop` hooks, then matching rollout `task_complete` | Escape, confirmed by rollout `turn_aborted` | Native session rollout files |
| Grok Build | `UserPromptSubmit`/`Stop` hooks, then matching `turn_completed` record; full answer and usage come from native updates | Ctrl+C, confirmed by native cancellation; restored drafts are cleared before submission | Native session JSON/JSONL files |
| OpenCode | `chat.message` identifies the user message; its persisted text part acknowledges acceptance; `session.idle` reads committed assistant messages with that parent ID | Escape, wait for confirmation prompt, Escape; native `MessageAbortedError` confirms | Consistent SQLite backup, including committed WAL pages |

The tested CLI versions are Claude Code 2.1.251, Codex 0.147.0, Grok Build 1.0.0,
Kimi Code 0.42.0, and OpenCode 1.18.15.
These are version-sensitive integrations: readiness or lifecycle changes must
pass the live tests before claiming support. Unknown or unacknowledged behavior
fails the attempt; it is never converted from arbitrary terminal text into a
successful answer. Grok/OpenCode retain their native step-limit settings, and
Kimi's goes in `loop_control.max_steps_per_turn`. Codex has no equivalent CLI
step-limit flag, as with its previous adapter.

Kimi Code also gates startup behind a "Trust this folder?" prompt, which its
driver accepts the same way Codex's is, and emits `CSI ? 996 n` at startup --
a private-mode query pyte 0.8.2 cannot dispatch, so the tracer's screen ignores
the private form rather than letting a `TypeError` kill the reader thread.

## Sessions, policy, and cancellation

Each new CLI gets a fresh HOME/config/XDG namespace and only its Agency gateway
credential. A retained CLI keeps this namespace across fast restores. Versioned
session bundles carry the harness name and native session
ID. Restoration rejects mismatched identities, absolute/traversing paths,
non-session files, and oversized data. Auth files, generated gateway config,
plugins, hooks, caches, and lock files are not session state. Kimi's bundle also
preserves its approved workspace-trust decision so a fresh resumed CLI does not
lose its model binding behind a second trust prompt.

The existing engine session-blob protocol stages the snapshot and commits it
after the sandbox checkpoint. Fresh-process resumption uses native CLI resume flags. A failed
attempt cannot publish a partially written session as a successful result.

Model traffic still passes through the existing Responses or Chat Completions
adapter and Agency's host services. Disconnecting a TUI stream closes its
upstream model request. Native tool hooks still check Agency policy, and all CLI
processes and children remain under ptrace. A partial redirect is reaped before
the runner reports failure to the queue fallback. Retirement closes the PTY and
cleans its isolated config, including startup and snapshot failures. A successful
fast-resume attempt leaves both alive until hibernation or retirement.

The shared tracer also handles a kill racing with thread creation: it drains
all wait events owned by its tracer thread, even when the parent's clone
notification was lost. Thread-scoped waits prevent one harness from consuming
another harness's exit status. This fixes process retirement without changing
Native's execution or introducing a PTY into Native.

## Tests

`tests/harness/test_external_pty.py` covers interactive configuration, exact
acknowledgments, stale events, completion races, malformed input, process cleanup,
and session isolation. Existing traced-terminal tests cover controlling-terminal
semantics, resize, bounded output draining, and process control on Linux.

`AGENCY_TEST_EXTERNAL_PTY=1 pytest tests/harness/test_external_pty_live.py` runs
real installed CLIs through ptrace, HTTP model routing, and the daemon's Unix
socket. A synthetic model tests resumption and redirects during both blocked
generation and a running shell tool, without API credentials or model charges.
Executables default to `~/.cache/agency_harness_bin`; set
`AGENCY_TEST_HARNESS_BIN_DIR` to test another installation.

`tests/sandbox/test_harness_checkpoint_matrix_linux.py` is an opt-in EC2 matrix
for all six harnesses. It uses real CLIs and deterministic upstream model replies,
checking three turns with Podman COW fast resume and Docker COW fresh restart.
It checks filesystem and conversation continuity, retained process identity for
fast resume, and new daemon identity for fresh restart. The CRIU regression suite
also verifies ptrace/PTY handoff and filesystem fallback after a failed restore.
