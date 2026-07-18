# Harness Integration Design

> **Status:** implemented (all six build phases). `agharness`/`agharness_internal/agharness_backends/`,
> `agproxy_llm`, `agproxy_ptrace`/`agharness_internal/agproxy_ptrace_internal/`, and `agpolicy` all exist in the
> codebase, per this document's design — see [agharness.md](agharness.md),
> [agproxy_llm.md](agharness_internal/agproxy_llm.md), [agproxy_ptrace.md](agharness_internal/agproxy_ptrace.md), and
> [agpolicy.md](agpolicy.md) for the concrete implementation, and the plan referenced in that work
> for the phase-by-phase build log. This document remains the source of truth for *why* things are
> shaped this way; where implementation surfaced a correction to an assumption made here (e.g. the
> container-capability requirement below), the correction is recorded in place rather than left
> stale. Off-the-shelf coding agent harnesses (Claude Code, Codex CLI, opencode) run as agents
> inside the agency framework **without modifying the harnesses themselves, with minimal
> interference in how they already operate, and with total, ground-truth visibility into every
> file and process operation they perform.** Linux/POSIX is the primary target; see
> "Platform scope" below.

## Core Principle

The harness keeps its own operation intact: its own system prompt and scaffolding, its own
built-in tools (Bash, Edit, Read, ...), its own compaction. Agency does not inject a system prompt,
does not disable or replace the harness's default toolkit, and does not force its own tool
definitions into the harness's tool list. Agency occupies exactly two seams:

1. **The LLM endpoint** — every one of these harnesses is explicitly designed to have this
   swapped (`ANTHROPIC_BASE_URL`, `model_providers.base_url`, a custom `provider` block).
2. **The syscall boundary of the harness's own process** — every file it opens and every process
   it spawns passes through the kernel, which means it can be observed and mediated *without the
   harness's cooperation or awareness*, via `ptrace`/`seccomp`, the same mechanism debuggers and
   sandboxes use. This is the default mediation path, not an escalation from something weaker.

Everything else — which built-in tool the model chooses, how the harness phrases its own system
prompt, when it compacts — is left entirely to the harness. What agency captures is **every
observable effect** of those choices at the OS level: every `execve`, every file open, every
process the harness's tree creates — routed through the same `agpolicy`/`aglog`/`agsandbox`
abstractions the native ReAct loop uses, with the harness's own tool implementation still doing
all the actual work.

### Isolation: one generic supervisor, no harness-specific interception code

The syscall-interception layer (`agproxy_ptrace`, Component 3) is a **single process, harness-agnostic by
construction**. It does not parse Claude Code's hook JSON, Codex's hook JSON, or opencode's plugin
API — it sees `execve(2)`/`openat(2)`/etc. and resolved arguments, which look identical regardless
of which binary produced them. One supervisor implementation covers all three harnesses, and any
future one, with zero per-harness branching. This is what makes "minimal changes to agency" and
"total capture" compatible: the interception logic lives once, outside every harness-specific code
path, and the existing agency abstractions (`agpolicy`, `aglog`, `agsandbox`, a profiler hook) run
underneath it unmodified. The per-harness backend files (`claude_code.py`/`codex.py`/`opencode.py`)
shrink to "how do I launch this binary headlessly and parse its own turn-level event stream for
logging/webui purposes" — they carry no execution-capture logic at all.

---

## Why This Is Possible

| Capability | Claude Code | Codex CLI (0.144.x) | opencode |
|---|---|---|---|
| Custom LLM endpoint | `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN`; wire = Anthropic Messages API | `model_providers` in `config.toml`; wire = OpenAI Responses API only | `provider` block naming an ai-sdk package → any wire format |
| Headless run + turn-level event stream | `claude -p --output-format stream-json` | `codex exec --json` | `opencode serve` (HTTP+SSE) / `opencode run --format json` |
| Execution-level mediation | **Not needed via the harness's own hooks** — superseded by syscall interception (Component 3), which requires no harness support at all | same | same |
| Supplemental tools (opt-in, additive only) | MCP | MCP | MCP + native `.opencode/tools/*.ts` |

The key shift from earlier drafts of this design: tool-call capture no longer depends on each
harness's own `PreToolUse`/`tool.execute.before` hook system firing correctly, or on hook JSON
schemas that have already changed shape at least once per harness in the last year. It depends on
the kernel delivering `execve`/`open` syscalls, which is a stable, decades-old ABI. Harness-native
hooks are still functionally available in all three and are kept as a documented **fallback** for
environments where `ptrace` isn't usable (see Platform scope / Design Tensions), but they are no
longer the default mechanism.

---

## New Modules

```
agdata / agtype / agutil / agcontext / agname / agpause
        ↓
agterm / aglog / agresources
        ↓
agllm / agsandbox
        ↓
agtool / agschema
        ↓
agent
        ↓
agskill  ───────────────────────────┐
        ↓                           ↓
agteam / agsync              agharness  (new)
                                     ↓
                          agharness_internal/agharness_backends/  (new: claude_code.py, codex.py, opencode.py)
                                     ↓
                    ┌────────────────┴────────────────┐
                    ↓                                  ↓
              agproxy_llm (new)                     agproxy_ptrace (new)
              LLM routing                          syscall-level supervisor
                    ↓                                  ↓
               agllm (existing)          agpolicy (new) · aglog · agsandbox PID tracking
```

`agharness` is selected instead of `execute_react` for a given agent, never in combination with
it. `agproxy_ptrace` and `agproxy_llm` are independent of each other and of `agskill`/`agteam`/`agsync`,
which gain no new dependencies at all.

---

## Component 1: `agproxy_llm` — LLM routing (unchanged from prior draft)

Agency's LLM client (`agency/agllm.py`) is server-less today; wiring a harness's LLM traffic to an
agent's `agllm` requires a small local HTTP server translating each harness's wire format
(Anthropic Messages / OpenAI Responses / chat-completions) to and from agllm's internal
chat-completions format, routed per-run by a bearer token minted into the harness's isolated
config/env at launch (`token → agent → agent.llm`). Passthrough mode (skip translation, only
observe) applies when the configured backend already speaks the harness's native format, to avoid
losing prompt-cache breakpoints or thinking-block signatures to a round-trip translation. See the
original rationale — this component is unaffected by the interception redesign below.

---

## Component 2: `agharness` + `agharness_internal/agharness_backends/` — thin, turn-level glue only

Same responsibilities as before, with execution-capture logic removed (it now lives entirely in
`agproxy_ptrace`):

1. Materializes an isolated, ephemeral config home per run, containing only the model/endpoint env
   from Component 1 — no hook registration, since mediation no longer depends on the harness's
   hook system.
2. Delivers the skill's task as a plain **user-turn prompt** (`system_prompt` + input JSON +, if
   `output_schema` is declared, a text instruction describing the required response shape) — never
   as `--append-system-prompt`, never as a tool.
3. **Launches the harness through `agproxy_ptrace` instead of a plain `sandbox.exec()`** — the one change
   to how the backend starts the process. From the backend's point of view this is a drop-in
   replacement for an exec call; it still gets the harness's stdout/stderr/exit code back.
4. Parses the harness's own turn-level event stream (`stream-json` / `codex exec --json` / opencode
   SSE) purely for **semantic, turn-level** information — which the OS-level view cannot
   reconstruct on its own: which model turn triggered a given execution, the model's assistant
   text/reasoning, token usage for cross-checking against `agproxy_llm`'s counts, and the final
   answer text for output-schema recovery. This is the harness's-eye view of "what tool did I call
   and why"; `agproxy_ptrace` supplies the OS's-eye view of "what actually happened." Both feed `aglog`,
   correlated by timestamp/PID, so the webui can show "Bash tool call (turn 4)" *and* the exact
   `execve` argv `agproxy_ptrace` observed for it.
5. Collects output by validating the harness's final response text against `output_schema` via the
   existing `agschema` path, reprompting as an ordinary user turn on failure.
6. Stores the harness's session id on `ag.ctx` for resume/fork.

---

## Component 3: `agproxy_ptrace` — syscall-level supervisor (the default interception mechanism)

### What it is

A single external process, launched by `agharness_internal/agharness_backends/*` in place of a direct `sandbox.exec`,
that starts the harness binary as its **own traced child** and observes/mediates its entire
process tree at the syscall boundary — no changes to the harness binary, no dependence on its hook
system, no per-harness code in the supervisor itself.

### Launch sequence (standard debugger-launch pattern — the harness is never modified)

```
agproxy_ptrace supervisor process
  └─ fork()
       └─ child: ptrace(PTRACE_TRACEME)
          child: install seccomp filter — SECCOMP_RET_TRACE on
                 {execve, execveat, openat, open, connect, unlink, unlinkat, rename, renameat2}
          child: execve(harness_binary, argv, envp)   ← unmodified harness binary
  supervisor: waitpid() loop; PTRACE_O_TRACEFORK|TRACEVFORK|TRACECLONE|TRACEEXEC
              → every new thread/process the harness tree creates is auto-attached
```

The child installs its own seccomp filter *before* exec — seccomp filters are inherited across
`execve` by design, so the filter applies to the harness binary and everything it forks/execs
afterward without the harness's knowledge. `SECCOMP_RET_TRACE` (rather than plain `PTRACE_SYSCALL`
single-stepping every syscall) means the supervisor only receives a `PTRACE_EVENT_SECCOMP` stop for
the handful of syscalls in the filter list — hot-loop syscalls like `read`/`write`/`futex` never
generate a stop, keeping overhead bounded to the operations actually worth mediating.

### What gets intercepted, and how

- **Process creation (`execve`/`execveat`)** — the supervisor reads `argv`/`envp` from the
  tracee's memory (`process_vm_readv`, falling back to `PTRACE_PEEKDATA`) at the syscall-entry
  stop, and calls `agpolicy.check(ag, event)` before letting the syscall proceed. This is real
  before-execution mediation on *every* process the harness's tree spawns, regardless of which
  internal built-in tool (or internal helper never exposed as a "tool" at all) triggered it —
  strictly more coverage than a hook that only fires for calls the harness's own tool-dispatch
  chooses to report. A `deny` decision rewrites the syscall to fail (e.g. force `-EPERM` at
  syscall-exit); an `allow`/log-only decision resumes it unmodified. The same "insert our own
  wrapper as the effective target" technique from the earlier hook-based draft still works here,
  now applied by rewriting `argv[0]`/the binary path directly in tracee memory before resuming —
  useful for attaching a profiler around a specific command without touching policy semantics.
- **File opens (`openat`/`open`)** — path argument resolved from tracee memory, checked against the
  agent's sandbox workspace scope, and logged. **Path redirection is deliberately left to
  `agsandbox`'s existing mount/bind-mount mechanism, not to raw pointer rewriting in tracee
  memory** — rewriting a string argument in place requires scratch space and is fragile across
  architectures; agsandbox already has a real, well-tested primitive for "this path resolves
  somewhere else" (`agSandboxConfig.add_mount`). `agproxy_ptrace`'s job here is policy (allow/deny) and
  ground-truth logging of every file actually touched — a strictly finer-grained audit trail than
  what even native `agtool`'s Read/Edit/Write report, since it also sees files the harness's own
  runtime opens without going through any tool-call abstraction at all (config files, dependency
  resolution, anything).
- **Process lifetime (`fork`/`vfork`/`clone`/exit)** — delivered as synchronous ptrace events, not
  inferred by diffing `/proc` before and after a shell command the way
  `agsandbox_backend.exec()` (`base.py:298`) does for the native path today. This is strictly more
  precise: every child, grandchild, and orphaned background process the harness's tree creates is
  visible the instant it's created and the instant it exits, with no polling and no race window.
  These events populate the **same** `agSandbox` bookkeeping that backs `get_live_pids()`,
  `pid_status_summary()`, and `wait_for_processes()` — those methods' external contract is
  unchanged; only their internal population source differs for harness-driven agents (ptrace event
  stream) versus native ones (`/proc` diff + `__BGPIDS__` marker).
- **Network (`connect`/`bind`) and destructive filesystem ops (`unlink`/`rename`)** — optional
  extensions using the identical mechanism; useful if a policy wants to gate outbound connections
  or flag/deny destructive operations distinct from ordinary reads/writes.

### Profiling

Because `agproxy_ptrace` already holds a `ptrace`-attached (or `PTRACE_SEIZE`d) handle on the harness's
entire process tree, attaching a profiler is a configuration choice, not new plumbing: dump a
syscall trace (the `strace`-equivalent falls out of the interception log itself, for free),
sample stacks (`perf record -p <pid>` targeting the same PIDs `agproxy_ptrace` is tracking), or run a
custom instrumentation layer — all driven by one `profiler:` config knob on the harness's
`agConfig` view, applied uniformly regardless of which harness is running.

### `agpolicy` — unchanged shape, different input

```python
class agpolicy:
    def check(self, ag: agent, event: agsyscallevent) -> agdecision
        # agsyscallevent: syscall name, pid, resolved argv/envp or path, timestamp
        # agdecision: allow() | deny(reason) | rewrite(new_args)
```

Same interface family as the earlier hook-based draft, now operating on resolved syscall
arguments instead of a harness-reported JSON tool call — one `agpolicy` implementation can back
both the native `dispatch_tools` retrofit (future work, unchanged from before) and `agproxy_ptrace`.

### Prerequisites and required (small) changes elsewhere

- **`agsandbox_backends/container.py` (docker/podman): NO extra capability or seccomp profile is
  required** — corrected by direct empirical testing (a real docker container, default seccomp
  profile, no `--cap-add`), which is the opposite of what an earlier draft of this section assumed.
  `agproxy_ptrace` always traces its own forked descendants (it never `PTRACE_ATTACH`/`SEIZE`s an
  already-running, unrelated process) — the kernel permits a process to trace its own children
  without `CAP_SYS_PTRACE` at all (Yama's default `ptrace_scope=1` explicitly allows this), and
  Docker's default seccomp profile does not block the `ptrace`/`seccomp` syscalls for that case
  either. Verified end-to-end inside a plain `docker run` container (no added flags): the full
  fork → `PTRACE_TRACEME` → seccomp-filter-install → `execve` → `PTRACE_EVENT_SECCOMP` cycle, plus
  the `deny` (register rewrite) and `rewrite` (scratch-memory injection) decision paths, all work
  identically to the bare-host case.
- **Chroot backend**: no container daemon or default seccomp profile stands in the way; the
  supervisor should perform the `unshare`/chroot setup itself (as the direct ancestor process) and
  fork the harness as its traced child from inside that same namespace, so credential/namespace
  checks resolve the same way they already do for the backend's own `_container_exec`.
- **Cross-namespace tracing — real prerequisite still open, now narrowed.** The capability finding
  above only covers a supervisor whose `fork()` happens *inside* the same PID namespace the traced
  harness will run in. That's automatically true for a bare host-level launch and for the chroot
  backend (shares the host namespace) — but a docker/podman-backed `agSandbox` runs its container
  in its own separate PID namespace, and `agproxy_ptrace`'s Python code today forks from wherever
  the *calling* process runs (typically the host-side agency process, not inside that container).
  For a harness that must execute inside an existing container-backed sandbox's namespace (so its
  filesystem writes land in the same workspace the rest of that agent's tools see), the supervisor
  itself needs to run inside that namespace too — e.g. via `docker exec` invoking a small
  in-container entrypoint that does the fork/trace loop, with intercepted events relayed back to
  the host-side `agpolicy` over that exec'd process's stdio. This bidirectional IPC bridge is
  **not implemented** — `agproxy_ptrace.launch()` as built forks directly from the calling
  process's own namespace, which is correct and sufficient for a bare host-level harness launch
  (what Phase 3's `agharness_backends` actually exercise, since none of the target harness CLIs
  were runnable inside a container in this environment either) but not yet for "run the harness
  inside this docker/podman-backed agent's own sandbox." Building and testing that bridge is
  explicitly deferred, not silently assumed solved.
- **Interaction with a harness's own internal OS-level sandboxing** (Codex's bwrap/seatbelt/landlock
  Bash sandbox, Claude Code's `sandbox.enabled`) has not been separately verified — those run as
  descendants of the traced harness process, so the same ancestor-of-descendant permission argument
  should apply, but this is a real interaction to check per-harness, not an assumption to ship
  untested. **Recommendation stands: disable the harness's own OS-level sandbox flags** when running
  under `agproxy_ptrace` + `agsandbox` containment — redundant once agency owns both boundaries, and
  removing it avoids debugging denials that could originate from either layer.

### Platform scope

`seccomp` and `PTRACE_O_TRACE*`/`PTRACE_EVENT_SECCOMP` are Linux-specific — there is no equivalent
combination on macOS or BSD (macOS has a different, much more restricted ptrace and no seccomp;
the closest analog is the Endpoint Security framework, which is a separate, unwritten backend).
Given the stated target is Linux/POSIX, `agproxy_ptrace` is a Linux-only component, matching
`agsandbox`'s existing docker/podman/chroot backends, which are already Linux-centric. Non-Linux
environments (or Linux environments where `SYS_PTRACE` genuinely cannot be granted — some hardened
multi-tenant runtimes) fall back to the harness-native hook mechanism from the earlier draft
(`PreToolUse`/`tool.execute.before`), which remains documented but demoted to a fallback path, with
reduced coverage (misses non-tool-call I/O, depends on the harness's hook schema staying stable).

---

## Component 4: The `engine` seam on `agent` (unchanged)

```python
cfg = agConfig(
    agVLLMBackendConfig(base_url=..., model=...),      # any existing agllm backend, unchanged
    agClaudeCodeConfig(                                # new config view, owner "agharness"
        gateway_mode="translate",
    ),
    agPtraceConfig(                                     # new config view, owner "agproxy_ptrace"
        syscalls=["execve", "openat", "connect"],
        profiler="perf",
        disable_harness_native_sandbox=True,
    ),
)
ag = agent(agconfig=cfg, engine="claude_code")
result = ag.run(my_existing_skill, agdata(task=...))    # completely unchanged call site
```

`agskill.run`'s `_task()` gains the same single branch as before: `engine == "native"` →
`execute_react` (untouched); any other value → `agharness_backend.for_config(ag.agconfig).execute(...)`,
which now launches its process through `agproxy_ptrace` rather than a direct `sandbox.exec`. Everything
above `agent.run()` — teams, `agsync`, dataflow, pause/fork/checkpointing, inbox injection, webui —
is unaffected, exactly as in the prior draft.

---

## Design Tensions

**ptrace/seccomp overhead is real but bounded.** `SECCOMP_RET_TRACE` only traps the filtered
syscall list, not every syscall — this keeps overhead far below naive full-syscall ptrace
stepping, but every trapped `execve`/`openat` still costs at least one extra context switch to the
supervisor. For workloads that `openat` heavily (large repo scans, package installs), expect
measurable but not prohibitive slowdown; this should be benchmarked before defaulting `openat` into
the filter list for every agent, versus only trapping it when file-level audit logging is actually
wanted (`execve`-only tracing is cheap and covers the highest-value case — process creation).

**Two complementary but distinct data sources.** `agproxy_ptrace` gives ground-truth OS-level
"what happened"; the harness's own turn-level stream (Component 2) gives semantic
"why/which-model-turn." Neither replaces the other — `aglog` should record and correlate both
rather than treating one as a superset of the other.

**No container hardening tradeoff, but a real namespace-boundary gap remains.** An earlier draft of
this section assumed granting `SYS_PTRACE` plus a custom seccomp profile inside a docker/podman
sandbox was required, and treated that as a meaningful loosening of the container's default
confinement worth calling out. Empirical testing corrected this: since `agproxy_ptrace` only ever
traces its own forked descendants, no extra capability or profile is needed, so that specific
hardening tradeoff doesn't exist. What *does* remain open is the namespace-boundary question this
masked — a docker/podman container has its own separate PID namespace, so a supervisor forked on
the host (today's implementation) cannot usefully trace a process that's meant to run *inside* that
container's namespace; bridging that (an in-container supervisor entrypoint via `docker exec` plus
an IPC channel back to the host-side `agpolicy`) is unimplemented, see "Prerequisites" above.

**Nested harness-native sandboxing interactions need per-harness verification**, not just the
Yama-default-allows-descendants argument above — seccomp filters *stack* (all installed filters
are evaluated per syscall, most-restrictive-wins), so if a harness's own internal sandbox (bwrap,
seatbelt-equivalent) installs a *stricter* filter than agproxy_ptrace's on some syscall, agproxy_ptrace may never
see it get to trace-worthy state, or a denial may appear to come from the wrong layer during
debugging. Recommending the harness's own OS sandbox be disabled (above) sidesteps this rather
than resolving it in general.

**Linux/POSIX-only.** No macOS/BSD backend exists in this design; see Platform scope.

**MCP visibility stays opt-in and additive-only**, unchanged from the prior draft — reserved
strictly for `skill.add_tools`, never for replacing the harness's default toolkit.

---

## Build Order (as executed)

All six phases are implemented and tested; this section is kept as the historical build log.

1. **`agproxy_ptrace` core: launch-and-trace + `execve` interception, against trivial test binaries
   (not a real harness yet).** Built the fork/`PTRACE_TRACEME`/seccomp-install/exec sequence,
   `PTRACE_EVENT_SECCOMP` handling, argv resolution from tracee memory, and the `agpolicy`
   allow/deny/rewrite loop. Several real bugs surfaced and were fixed during this step (wrong
   ctypes `restype` truncating 64-bit memory reads, a `SIGSTOP` synchronization race with the
   seccomp filter, ptrace's per-thread tracer identity, and `waitpid(-1, ...)` racing with
   unrelated subprocesses elsewhere in the process) — see `agproxy_ptrace.md`'s implementation
   notes for the specifics.
2. **Extended `agproxy_ptrace` with `openat` interception + `agSandbox` PID-tracking integration**
   (`agsandbox_backend.ingest_ptrace_pids()`, feeding the same `_watched_pids` dict `get_live_pids()`/
   `wait_for_processes()` already use). **The docker/podman `SYS_PTRACE` capability requirement
   assumed above turned out to be unnecessary** — verified empirically inside a real container
   with the default seccomp profile and no added capability: `agproxy_ptrace` always traces its
   own forked descendants, which the kernel permits without `CAP_SYS_PTRACE`. See "Prerequisites"
   above for the corrected finding and the real gap it surfaced instead (a docker/podman
   container's separate PID namespace vs. a host-forked supervisor).
3. **`agent.engine` seam** (`agent.py`/`agskill.py`) + **`agproxy_llm` (chat-completions route) +
   opencode backend** — the smallest real-harness integration slice. No `opencode` binary was
   installable in the development environment (no Node/Bun), so this backend is verified by
   mocked tests only.
4. **Claude Code backend**, verified against the real, authenticated `claude` CLI (v2.1.212)
   end-to-end — both raw-text and structured-`output_schema` output recovery, real
   `agproxy_ptrace` tracing of the real process. LLM routing through `agproxy_llm` is **not**
   implemented for this backend (Claude Code speaks the Anthropic Messages API, which
   `agproxy_llm` doesn't adapt to yet) — the harness authenticates with whatever credentials it
   already has on the host, same as running it by hand. **Codex backend** built structurally
   (same shape, mocked tests only) — no `codex` binary was available to verify against.
5. **Harness-native hook fallback** (`agharness_internal/agharness_backends/_native_hooks.py`) — built at the reduced
   scope this phase called for: the hook-JSON ↔ `agsyscallevent`/`agdecision` translation logic is
   real and tested, but it is not wired into any concrete backend's `execute()` as an actual
   alternate mediation path (that would be a second full implementation of Component 3's
   mediation, out of scope for a documented fallback).
6. **`agpolicy` retrofit into native `dispatch_tools`.** `agsyscallevent` gained `tool_name`/
   `tool_args` fields (defaulted to `None`) so one event type serves both mediation paths;
   `dispatch_tools()` gained optional `policy=`/`ag=` parameters, `None` by default so every
   existing native call site is unaffected. `rewrite` is not meaningful for a native tool call
   (no single argv-shaped string to rewrite) and is treated as `allow` if returned here.
