# Claude Code interactive validation — 2026-11-06

This repeats the [Codex comparison](inherited-terminal-validation.md) with
unmodified Claude Code **2.1.299**, using the same implementation in
`/home/eric/Documents/Codex/agency-native-runtime` on baseline `37c28cb8`.
All live inference used only **gpt-6-luna** at `https://router.js-park.info`.
The existing local router credential was loaded without printing it;
no fallback model/provider or native subscription authentication was used.

## Three real execution paths

Each path completed two turns with actual Bash commands: write and reread
`parity.txt`, then read it and produce a second output marker. We checked
filesystem contents and tool outputs, not only the final model text.

| Path | Result |
| --- | --- |
| Interactive Agency + Claude | Two turns; three successful Bash calls/results; six successful native Messages exchanges; `/compact`; exact SQLite/request reconstruction; cgroup freeze/thaw; clean `/exit` |
| Base managed Agency + Claude | Public `Agent` + `agskill`, real Docker `python:3.12-slim`; two tasks; two Bash calls/results; four Messages exchanges; profiling; test sandbox removed |
| Pure Claude | Same two terminal tasks and file results; direct router; `/compact`; clean `/exit` |

The test selected Bash as an allowed native tool and low effort for the
interactive/pure pair. These are controlled test arguments, not product
permission defaults. The terminal test driver supplies a user's terminal;
Agency adds no additional PTY. All three paths used the same CLI version.

The interactive run verified identical inherited fd0/fd1/fd2, active ptrace,
real systemd re-exec, a CLI workload cgroup and a separate collector/runtime
cgroup. Freezing and thawing the CLI cgroup left Agency running. It recorded
176 syscall admission/completion pairs; its profiler contained 155 events
and 304 spans. All six request bodies matched their archive hashes and
reconstructed from the current MessageStore tables. All native response
model IDs were `gpt-6-luna`; model action and exchange counts matched without
duplicate OTel records. Native subagent lifecycle events were also retained;
the HTTP gateway continues to attribute model traffic to the parent session
rather than guessing a subagent identity.

After compaction, both interactive Agency and pure Claude resumed their
original conversation UUID and completed another real Bash turn. The Agency
resume produced two further reconstructed exchanges and exited cleanly.
The test driver waits for asynchronous exchange persistence after a reply
appears on the terminal; an initial premature assertion is retained in the
diagnostic log, separate from the successful final resume.

## Native compatibility fixes

Claude now owns its conversation ID. The old unconditional `--session-id`
injection made real `--resume`/`--continue` fail unless `--fork-session` was
also supplied. Agency's own launch identity remains independent, while
native resume, continue, fork and explicit user IDs pass through unchanged.

Claude accepts one effective `--settings` option; a later user option
previously replaced Agency's hook file. Agency now composes that explicit
file or JSON object with its hooks in one private settings file. Other
settings, permission rules, unknown fields, duplicate user hooks and native
last-occurrence behavior are preserved. The source settings file is not
modified. A real live test confirmed the user hook and environment survived,
while Agency's denial still blocked Bash even when the user hook allowed it.
No forbidden command/file was executed or created.
That run also retained an auxiliary request cancelled during exit, including
its original request and partial-response metadata.

Native hook opt-outs remain opt-outs. `--bare`, `--safe-mode`, their known
environment equivalents and explicit `disableAllHooks` mark tool/turn hook
coverage unavailable. Combining those known opt-outs with Agency tool policy
fails clearly before launch. Model capture, process/resource profiling and
independent syscall policy remain available. Other native user/project/managed
layers can also disable hooks; Agency does not duplicate Claude's complete
configuration resolver or claim those gates are an OS security boundary.

## Terminal and scope controls

Separate real Bash tests compared pure Claude with the actual scoped
`agency claude` entrypoint, without submitting model prompts. Both passed
resize, keyboard Ctrl-Z, `fg`, resumed input, `bg` with native terminal-input
stopping, a usable parent shell, clean double-Ctrl-C exit and exact shell
termios restoration. Agency retained the original standard descriptors and
kept its runtime outside the CLI workload scope.

Shell suspension pauses the Agency wrapper too, as required for Bash job
control. Dashboard/cgroup pause affects the workload while keeping the
collector/control runtime alive. The scope covers newly launched local
descendants; remote/cloud execution or attachment to an existing external
process is not covered. Root-exit cleanup retires this launch's descendants,
so this attended launcher does not provide detached `--background` service
lifetime parity.

## What still differs

| Feature | Interactive Agency + Claude | Base managed Agency + Claude | Pure Claude |
| --- | --- | --- | --- |
| User interaction | Original terminal | Agency driver/PTY and skill API | Original terminal |
| Settings/state | Native settings plus composed observation hooks | Isolated native state, selected settings sources, Agency MCP setup | Native settings/state |
| Permissions | Native permissions plus optional Agency denial | Managed bypass mode plus Agency policy | Native permissions |
| Filesystem | Host cwd | Sandbox `/workspace` and checkpoint lifecycle | Host cwd |
| Models | Native route with capture off; native HTTP passthrough with capture on | Native Messages preserved on matching Anthropic backend; configured backend parameters apply | Native provider directly |
| Logging/context | Agency collectors/profiler, optional wire archives | Agency collectors/profiler and exchange records | Native session logs |
| Process controls | Scope/tracer controls plus native keys | Managed controls/tracing | Native keys/OS controls |
| Agent tools | Native tools and subagents | Agency MCP tools; native Agent subtool disabled by default | Native tools/subagents |

These checks establish task and interactive behavior, not identical prompts,
timings or model prose. Managed mode adds instructions and lifecycle work.
Claude warns that the custom Luna model name defaults to a 200k context
window; that native warning appears in both pure and Agency sessions.
Compaction remains CLI-owned. Encrypted provider state and absent server
history remain opaque, and capture does not implement context editing or
subscription-preserving capture authentication.

## Verification and evidence

The complete sessions suite passed **160 tests**, including both real CLI
browser terminals with scripted native responses, tools, cancellation and
compaction. The focused Claude/settings/policy/capture suite passed **107**.
These counts overlap. The additional entrypoint/lifecycle suite passed
**14 tests**, covering clean configuration errors and scope controls. Ruff,
pre-commit and diff checks passed and are recorded with the final artifacts.

Private ignored artifacts are under
`artifacts/claude-interactive-validation/`: `interactive/`, `pure/`,
`managed/`, `resume/`, `policy-settings/`, `terminal/`, native request/response
archives, SQLite/profile data, repeatable drivers, validation logs and the
current UI screenshot. Credential/configuration/native-state files are
omitted from the copied runs, and the router credential is scanned for
absence. Stored references retain original test paths. No commit, push,
merge or HANDOFF.md execution was performed.
