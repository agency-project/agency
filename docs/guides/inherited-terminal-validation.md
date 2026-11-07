# Inherited terminal validation — 2026-11-06

This extends the [earlier capture validation](interactive-capture-validation.md).
The default interactive launch now inherits the user's existing terminal;
the browser PTY remains an explicit alternate mode. The implementation
reuses Agency's existing ptrace supervisor, interaction service, policy,
SQLite logger, profiler, control relay and dashboard. See the
[runtime guide and three-way comparison](interactive-runtime.md).

The review worktree is `/home/eric/Documents/Codex/agency-native-runtime`,
based on the already-merged local `eric/interactive-sessions` tip
`37c28cb8`. An initial inspector used the older requested `e6a42a9d`
capture checkout. The UI investigation identified that source mismatch;
the implementation was reapplied to the complete current baseline,
including its preservation fixes and MessageStore schema. No new merge
was performed. The modern profiler, Critical Path and Compare Runs assets
are served directly from this repository; no frontend build is required.

## Real model comparison

All three paths used the same unmodified Codex 0.162.1 installation and only
`gpt-6-luna` at `https://router.js-park.info`. The existing router key was
loaded locally without printing it or adding it to source. No fallback
provider/model was used.

Each path completed two turns: a real shell tool wrote and reread
`parity.txt`, then another shell tool read it and produced a second marker.
We verified actual file contents and tool results, rather than treating
model text alone as execution evidence.

| Path | Verified result |
| --- | --- |
| Interactive Agency + Codex | Two native terminal turns; two tool admissions/results; native `/compact`; six captured model exchanges with nonempty context; native Ctrl-C exit without Python traceback |
| Base managed Agency + Codex | Public `Agent` + `agskill(output_schema=None)` API; Docker `python:3.12-slim`; two successful tasks; two tool uses/results; six reconstructed exchanges (one cancelled auxiliary call); profiler records; test sandbox removed |
| Pure Codex | Two native terminal turns and matching file/tool results; direct Luna route; native Ctrl-C exit |

The terminal test driver supplied the equivalent of a user's terminal.
Agency itself allocated no additional PTY in native mode. The CLI's three
standard descriptors matched the inherited terminal. `/proc` confirmed
active ptrace, a workload scope containing Codex, and a separate scope
containing Agency. Actual CLI entry exercised systemd re-exec with capture
options intact. The inherited terminal test used explicit native
`workspace-write`/`never` permissions for deterministic commands; those
are test settings, not new product defaults.

The interactive SQLite database contained two user prompts, two tool calls,
two tool results, and 596 syscall admission/completion pairs. Its profiler
database contained 141 events and 793 spans. All six native request bodies
reconstructed exactly from SQLite metadata and matched their wire archive
hashes. The dashboard's context projection exposed all six exchanges;
model action and exchange counts matched, with no duplicate OTel models.

These are behavior checks, not a performance benchmark. Managed execution
has extra instructions, sandbox setup and checkpoint work. On the matching
Responses backend it now preserves native envelopes and events; other
backend combinations can require translation. Explicit Agency model
options and the managed lifecycle still apply.
Exact prompts, token counts, timings and prose consequently differ.

## Process, terminal and policy checks

Regression coverage includes inherited descriptor identity, foreground
input, Ctrl-C, resize, terminal restoration, failed launch cleanup, and
background descendants. Real systemd tests cover both traced and untraced
launches: the root enters its workload scope before exec, descendants that
start a new session remain in the scope, freeze stops progress, resume
restarts it, and force-stop while frozen retires both root and descendants.

Real Bash tests cover Ctrl-Z, `fg`, `bg`, background terminal input
(`SIGTTIN`), exact raw termios restoration, nonzero CLI exit status, and
the actual scoped Agency CLI entry. Shell suspension also suspends the
Agency wrapper until the shell resumes it. Dashboard pause remains a
workload-only control. Repeated Ctrl-C after CLI exit cannot interrupt final
logger/profiler cleanup. Reopened recordings retain the execution clock and
terminal agent state even when global completion arrives before actor rows.

Tool policy uses the existing interaction service. Denials are native hook
responses; an allow leaves the CLI's own permissions in charge. Explicit
tool-policy transport failures fail closed through the command relay,
including Claude's PreToolUse hook. Observation-only telemetry can fail
without blocking execution. Syscall policy requires tracing.

The shared gateway still preserves unknown fields, tool schemas, opaque
provider state, original streaming bytes, errors and partial cancellation.
The prior live compaction and `previous_response_id` evidence remains
applicable: an encrypted compaction item is retained but cannot be decoded,
and a server-history reference does not contain the referenced history.
The context inspector marks those requests incomplete. Capture does not
implement context editing or native subscription-preserving authentication.

## Final checks

| Check | Result |
| --- | --- |
| Review worktree sessions, tracing, policy, dashboard and profiler regression | 544 passed, 4 skipped |
| Dashboard JavaScript | 318 passed |
| Both real CLI browser terminals with scripted native upstream, tools, cancellation and compaction | 2 passed |
| Repository-wide run | 4,108 passed, 94 skipped; 3 benchmark import failures |
| All affected benchmark tests in the fresh review environment | 94 passed; all three failing cases passed |
| Repository pre-commit checks, Ruff and diff check | Passed |

The repository-wide run used the earlier capture worktree's environment
with the current Agency source selected through `PYTHONPATH`. Its editable
installation still resolved the `benchmarks` namespace to the old checkout,
causing three Codex annotation-schema assertions to fail. The review worktree
has its own fresh `uv sync --locked --extra dev` environment. We verified
both Agency and benchmark import paths, then reran every test importing
benchmark code and the complete annotation directory there: 94 passed,
without namespace overrides or product fixes. The full run was not repeated
after that environment correction; the original failure log is retained.

## Local evidence

Ignored private artifacts are under
`artifacts/inherited-terminal-validation/`:

- `interactive/summary.json`, `interactive/reconstruction.json`,
  `interactive/runs/`: successful terminal/cgroup/context checks and native
  request/response archives, agent/global databases and resource profiling.
- `pure/summary.json`, `pure/screen.txt`: pure Codex comparison.
- `managed/summary.json`, `managed/exchanges.json`, `managed/logs/`,
  `managed/verification.json`, `managed/profile/`: public managed execution,
  reconstructed data and trace. The verification records the validation
  driver's correction from old normalized tool blocks to preserved native
  Responses items; the original failed assertion is retained in its log.
- `ui-source.json`, `native-context-ui.png`: current static-asset hashes and
  browser inspection of native captured context.
- `run_terminal_live.py`, `run_base_live.py`, `verify_base_current.py`:
  repeatable local live drivers and current-schema verification.
- Regression logs and final check results are copied alongside these runs.

Session credentials/configuration and native CLI state directories are
omitted from copied evidence. Raw captured model content is private;
the upstream credential is checked to be absent. Recorded archive/database
references retain their original test paths. No commit, push, merge,
or HANDOFF.md execution is part of this work.
