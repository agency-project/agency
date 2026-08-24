# Harness Integration Design

> **Status:** implemented (all six build phases). `engine/`, `harness/daemon.py`,
> `harness/adapters/`, `harness/ptrace/`, and `agpolicy` contain the current implementation
> codebase, per this document's design — see [agharness.md](agharness.md),
> [agproxy_llm.md](harness/agproxy_llm.md), [agproxy_ptrace.md](harness/agproxy_ptrace.md), and
> [agpolicy.md](agpolicy.md) for the concrete implementation, and the plan referenced in that work
> for the phase-by-phase build log. This document remains the source of truth for *why* things are
> shaped this way; where implementation surfaced a correction to an assumption made here (e.g. the
> container-capability requirement below), the correction is recorded in place rather than left
> stale. Off-the-shelf coding agent harnesses (Claude Code, Codex CLI, opencode, Grok Build) run as agents
> inside the agency framework **without modifying the harnesses themselves, with minimal
> interference in how they already operate, and with total, ground-truth visibility into every
> file and process operation they perform.** Linux/POSIX is the primary target; see
> "Platform scope" below.
>
> **One superseded extension, one active:** [Design_harness_filesystem.md](Design_harness_filesystem.md)
> proposed closing the docker/podman filesystem-visibility gap *without* running the harness inside
> the container (a FUSE-backed filesystem around a host-resident harness process) — **superseded**;
> the project decided to run the harness inside the container after all, which is this document's
> original direction and makes that gap moot (see "Prerequisites" and "Design Tensions" below, which
> briefly documented the FUSE direction before being corrected back). The active in-container
> supervisor bridge described there is now being implemented directly.
> [Design_harness_history.md](Design_harness_history.md) — cross-invocation history/continuity for
> harness-driven agents, decoupled from any specific sandbox instance, matching the portability
> `agcontext` already gives native agents — unaffected by the filesystem-direction reversal, still
> active.

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
construction**. It does not parse Claude Code's hook JSON, Codex's hook JSON, opencode's plugin
API, or Grok Build's (Claude-Code-compatible) hook JSON — it sees `execve(2)`/`openat(2)`/etc. and
resolved arguments, which look identical regardless of which binary produced them. One supervisor
implementation covers all four harnesses, and any future one, with zero per-harness branching.
This is what makes "minimal changes to agency" and "total capture" compatible: the interception
logic lives once, outside every harness-specific code path, and the existing agency abstractions
(`agpolicy`, `aglog`, `agsandbox`, a profiler hook) run underneath it unmodified. The per-harness
backend files (`claude_code.py`/`codex.py`/`opencode.py`/`grok.py`) shrink to "how do I launch
this binary headlessly and parse its own turn-level event stream for logging/webui purposes" —
they carry no execution-capture logic at all.

---

## Why This Is Possible

| Capability | Claude Code | Codex CLI (0.144.x) | opencode | Grok Build (xAI) |
|---|---|---|---|---|
| Custom LLM endpoint | `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN`; wire = Anthropic Messages API | `model_providers` in `config.toml`; wire = OpenAI Responses API only | `provider` block naming an ai-sdk package → any wire format | `[model.<name>]` in `config.toml`, `base_url`/`api_key`/`env_key`; `api_backend` selects `chat_completions` \| `responses` \| `messages` — any of the three wire formats |
| Headless run + turn-level event stream | `claude -p --output-format stream-json` | `codex exec --json` | `opencode serve` (HTTP+SSE) / `opencode run --format json` | `grok -p "..." --output-format json` (single JSON result) / `streaming-json` (NDJSON) |
| Execution-level mediation | **Not needed via the harness's own hooks** — superseded by syscall interception (Component 3), which requires no harness support at all | same | same | same |
| Supplemental tools (opt-in, additive only) | MCP | MCP | MCP + native `.opencode/tools/*.ts` | MCP (stdio + HTTP/SSE), plus auto-imports Claude Code's/Cursor's MCP config |

The key shift from earlier drafts of this design: tool-call capture no longer depends on each
harness's own `PreToolUse`/`tool.execute.before` hook system firing correctly, or on hook JSON
schemas that have already changed shape at least once per harness in the last year. It depends on
the kernel delivering `execve`/`open` syscalls, which is a stable, decades-old ABI. Harness-native
hooks are still functionally available in all four and are kept as a documented **fallback** for
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
                          harness/agharness_backends/  (new: claude_code.py, codex.py, opencode.py, grok.py)
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

## Component 1: `agproxy_llm` — LLM routing (both modes now implemented)

Agency's LLM client (`agency/agllm.py`) is server-less today; wiring a harness's LLM traffic to an
agent's `agllm` requires a small local HTTP server translating each harness's wire format
(Anthropic Messages / OpenAI Responses / chat-completions) to and from agllm's internal
chat-completions format, routed per-run by a bearer token minted into the harness's isolated
config/env at launch (`token → agent → agent.llm`). Passthrough mode (`/v1/chat/completions`, no
reshaping) applies when the configured backend already speaks the harness's native format —
opencode and Grok Build. Translate mode (`/v1/messages` for Claude Code, `/v1/responses` for
Codex, conversion functions in `harness/agproxy_llm_adapters.py`) reshapes the request
into the uniform `client.chat.completions.create()` call every `agllm_backend` exposes and reshapes
the response back — implemented as unconditional translation for every request on those two
routes, not a conditional "reuse the native format when it happens to match" optimization. This is
a real fidelity cost, not a free lunch: extended-thinking blocks, prompt-cache breakpoints
(`cache_control`), and image content blocks have no chat-completions equivalent and are silently
dropped on translation rather than erroring. A future optimization could detect when the
configured backend's *native* format already matches the harness's wire format (e.g. an
`agAnthropicBackendConfig`-backed agent talking to Claude Code) and skip translation entirely to
preserve that fidelity — not built, since the immediate goal was closing the "harness bypasses
agency's backend choice entirely" gap, not maximizing streaming fidelity for an already-matched
case.

---

## Component 2: `agharness` + `harness/agharness_backends/` — thin, turn-level glue only

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

A single external process, launched by `harness/agharness_backends/*` in place of a direct `sandbox.exec`,
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

- **`sandbox/container.py` (docker/podman): NO extra capability or seccomp profile is
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
- **Cross-namespace tracing — run the supervisor inside the sandbox.** A docker/podman-backed
  `agSandbox` runs its container in its own separate PID namespace. The Harness Manager daemon and
  `harness/ptrace/` supervisor run inside that sandbox, so the tracer forks from the same namespace
  in which the harness must execute. The harness therefore executes *inside* the container's namespace
  — so its filesystem writes land in the same workspace the rest of that agent's tools see, and so
  its own built-in tools resolve real container-native paths with no interception layer standing
  in for the kernel — which means the supervisor itself needs to run inside that namespace too.

  An intermediate draft of this section considered the opposite shape — keep the harness on the
  bare host, in a private namespace of its own, with a FUSE-backed filesystem making the container's
  files visible without the harness ever entering it (see
  [Design_harness_filesystem.md](Design_harness_filesystem.md), now superseded) — specifically to
  avoid building this bridge. That direction was reconsidered: running the harness inside the
  container is the adopted design after all. `ensure_harness_daemon()` launches the daemon through
  the sandbox backend; the daemon then invokes the local tracer directly. Trapped syscall events
  cross the existing host-services UDS to `agpolicy.check()`, and the allow/deny response returns on
  that same request. There is no second ptrace implementation or ptrace-specific relay.
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
ag = agent(agconfig=cfg, harness="claude_code")
result = ag.run(my_existing_skill, agdata(task=...))    # completely unchanged call site
```

**Update: this branch is gone.** `execute_react()` (the old host-process ReAct loop this section
originally described as staying "untouched") has since been retired entirely -- `agskill.run`'s
`_task()` now unconditionally calls `agharness_backend.for_config(ag.agconfig).execute(...)` for
every engine, `"native"` included (see `agharness_backends/native.py`: a persistent in-container
process, not a host-process loop). Everything
above `agent.run()` — teams, `agsync`, dataflow, pause/fork/checkpointing, inbox injection, webui —
is unaffected, exactly as in the prior draft.

---

## Component 5: Cross-cutting concerns for harness-driven agents (logging, webui, schema retry, GPU)

**Status: partially built.** Resource-control and output-submission are no longer proposed --
`harness/agmcp_server.py`'s shared `agMCPServer` (Phase 4 of the container-unification
plan; see the plan's own doc/PR for the full design) exposes `reserve_cpu`/`cpu_release`/
`daemon_release`/`submit_output` as real MCP tools, reached by `claude_code.py` today via
`--mcp-config`/`--strict-mcp-config` (reached through the Harness Manager's HTTP proxy over the
host-services UDS) and by `native.py`'s
in-container react loop via a real `mcp` client. `submit_output` replaces the schema-reprompting
approach described below FOR THOSE TWO ENGINES: structured output is now collected via tool calls
and read back through `agmcp_server.collected_output(token)`, not parsed post-hoc from the harness's
final text. `codex.py`/`opencode.py`/`grok.py` are NOT wired to this server yet (no container support
at all currently -- a separate, larger task) and still use the free-text JSON + `validate_and_recover`
path described below unchanged. **`reserve_gpu`/`gpu_release` remain unimplemented** -- see
`agmcp_server.py`'s own module docstring for the specific gap (GPU env-var injection only reaches
`sandbox/base.py`'s `exec()`, a path neither an in-container harness's own tool execution
nor `native.py`'s bash tool goes through) -- the paragraph below describing the intended MCP-based
GPU design is still accurate as a target, just not yet built.

**Update: `execute_react()` itself is gone (see the retirement note above); the five concerns below
are now split between `execute_harness()` (input/output schema validation+recovery, sandbox
lifecycle) and `native.py`'s `_NativeBackend.execute()` (`aglog`/webui push, now reconstructed via a
background thread polling `agllm_terminus`'s live per-token transcript rather than driven inline by
an in-process loop -- see that module's own docstring).** Originally, the native ReAct loop
(`agskill.py:execute_react`, now retired) drove
five things inline as it ran: `aglog` tool-call/turn logging, webui push (`_push_live_messages`/
`_set_ui_state`/`token_update`), input/output schema validation with reprompt-on-failure, sandbox
lifecycle (lazy-start/hibernate), and GPU/resource acquisition. A harness-driven run needs all five
too, but can't hook into a loop it doesn't control. Each one maps onto a different existing seam:

- **Sandbox lifecycle has no gap at the skill-run boundary.** Container create/`commit()`/
  `rm_container()` already wraps *either* engine identically, at the `agskill.py:run()` boundary
  outside both `execute_react` and `execute_harness` — nothing harness-specific to add here. The
  finer-grained per-call lazy-start/hibernate (`_ensure_started()`/`stop()`) is a different
  question, and running the harness inside the container genuinely coarsens it: native's per-tool-
  call hibernation (`agtool.py:455-470`, `container.py:1521-1573`, explicitly to release "the
  runtime slot ... AND the GPU") works because nothing lives inside the container between tool
  calls — only the tool's own transient exec needs it up. Once the harness's own reasoning process
  is what's alive inside the container, there's no safe point to fully stop it without killing that
  process, so container liveness for a harness-driven call coarsens from per-tool-call to
  per-skill-call — up for the duration of one harness invocation, torn down after, same boundary
  `agskill.py:400-422` already uses. `docker pause`/`unpause` doesn't recover this: a frozen cgroup
  still holds the GPU context/memory, so it only reclaims CPU scheduling, not the resource native's
  `stop()` actually releases. Accept this as the real cost of this design rather than building
  speculative pause-point detection.
- **Logging and webui should be driven by tool-call boundaries parsed at the LLM proxy, not a
  separate stream-json parser.** `agproxy_llm_adapters.py` already parses `tool_use`/`tool_result`
  (Claude Code) and `function_call`/`function_call_output` (Codex) content blocks out of every
  request/response crossing the gateway, as a byproduct of wire-format translation — real,
  protocol-level, unambiguous tool-call visibility, not yet logged or acted on. Extending each route
  handler to open a "tool call in flight" window on the outgoing call and close it on the matching
  result gives (a) `aglog._tool_call` the real tool name/args/result instead of today's coarse
  ptrace-argv-only logging (`agharness.py:77-89`), and (b) webui pushes at the same granularity
  native gets. Keep `agproxy_ptrace`'s syscall stream as a second, parallel ground-truth feed into
  `aglog` — it answers "what actually happened at the OS level," the proxy-level window answers
  "what tool, semantically" — neither replaces the other. Note this window is a logging/webui signal
  only now, not a sandbox lazy-start/hibernate trigger — the sandbox is already up for the whole
  invocation per the bullet above, so there's nothing left for it to trigger on that axis.
- **Output schema reprompting** can't inject a mid-loop correction message the way native does,
  since the harness's internal loop is opaque. Coarsen the retry unit instead: on validation
  failure, issue the correction as a new top-level turn against the harness, bounded by the same
  `output_schema_retries_left` counter every engine (native included, via its own bounded reprompt
  loop in `_NativeBackend.execute()`) shares. How that turn reaches the
  harness with the right context depends on the continuity mechanism —
  see [Design_harness_history.md](Design_harness_history.md).
- **GPU/resource control has no native-loop precedent to break**, since it's already tool-mediated
  and opt-in even for native agents (`gpu_reserve` calling `agResourcePool.acquire_gpu`, not a fixed
  once-per-run acquire). The right channel for a harness-driven agent is MCP — already the one
  sanctioned, additive-only path into a harness's own tool list (reserved for `skill.add_tools`,
  never for replacing built-ins). Expose `gpu_reserve`/`gpu_release` as an MCP tool when a skill
  needs the model to request GPU access dynamically from within a harness-driven turn; for a fixed
  budget, pre-allocate before launch at the sandbox-creation boundary instead and skip the dynamic
  path entirely.

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

**No container hardening tradeoff, but a real namespace-boundary gap remains, and it's being
bridged, not designed around.** An earlier draft of this section assumed granting `SYS_PTRACE` plus
a custom seccomp profile inside a docker/podman sandbox was required, and treated that as a
meaningful loosening of the container's default confinement worth calling out. Empirical testing
corrected this: since `agproxy_ptrace` only ever traces its own forked descendants, no extra
capability or profile is needed, so that specific hardening tradeoff doesn't exist. What *does*
remain open is the namespace-boundary question this masked — a docker/podman container has its own
separate PID namespace, so a supervisor forked on the host (today's implementation) cannot usefully
trace a process that's meant to run *inside* that container's namespace. An intermediate draft of
this document proposed avoiding this by keeping the harness on the host entirely (see
[Design_harness_filesystem.md](Design_harness_filesystem.md), now superseded) — that direction was
reconsidered in favor of the original plan: the harness runs inside the container, and the
namespace boundary is bridged with an in-container supervisor entrypoint via `docker exec` plus an
IPC channel back to the host-side `agpolicy`, per "Prerequisites" above. This is real, nontrivial
engineering, not a gap that dissolves on its own — it's the current implementation focus.

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
3. **`agent.harness` seam** (`agent.py`/`agskill.py`) + **`agproxy_llm` (chat-completions route) +
   opencode backend** — the smallest real-harness integration slice. No `opencode` binary was
   installable in the development environment (no Node/Bun), so this backend is verified by
   mocked tests only.
4. **Claude Code backend**, verified against the real, authenticated `claude` CLI (v2.1.212)
   end-to-end — both raw-text and structured-`output_schema` output recovery, real
   `agproxy_ptrace` tracing of the real process. LLM routing through `agproxy_llm` was **not**
   implemented for this backend initially (Claude Code speaks the Anthropic Messages API, a
   different wire format from the chat-completions-only gateway that existed at the time) — closed
   in a later pass, see item 8 below. **Codex backend** built structurally (same shape, mocked
   tests only) — no `codex` binary was available to verify against, then or since.
5. **Harness-native hook fallback** (`harness/agharness_backends/_native_hooks.py`) — built at the reduced
   scope this phase called for: the hook-JSON ↔ `agsyscallevent`/`agdecision` translation logic is
   real and tested, but it is not wired into any concrete backend's `execute()` as an actual
   alternate mediation path (that would be a second full implementation of Component 3's
   mediation, out of scope for a documented fallback).
6. **`agpolicy` retrofit into native `dispatch_tools`.** `agsyscallevent` gained `tool_name`/
   `tool_args` fields (defaulted to `None`) so one event type serves both mediation paths;
   `dispatch_tools()` gained optional `policy=`/`ag=` parameters, `None` by default so every
   existing native call site is unaffected. `rewrite` is not meaningful for a native tool call
   (no single argv-shaped string to rewrite) and is treated as `allow` if returned here.
7. **`grok.py` backend added afterward**, following the exact same shape as the other three (no
   architecture changes required — this is the point of the design: adding a harness is "one more
   file in `agharness_backends/`"). Notable because it's the second backend (after opencode) whose
   `[model.*]` config genuinely supports `api_backend = "chat_completions"`, so it routes through
   `agproxy_llm`'s existing passthrough gateway rather than leaving the endpoint untouched.
   Structural + mocked tests only (see `agharness.md`'s per-backend table) — no `grok` binary was
   installed, since doing so means running xAI's `curl | bash` install script, deliberately not
   done without being asked first.
8. **Closed the LLM-routing gap for Claude Code and Codex.** Built the two translate-mode routes
   Component 1 originally deferred: `/v1/messages` (Anthropic Messages API) and `/v1/responses`
   (OpenAI Responses API), conversion logic in the new `harness/agproxy_llm_adapters.py`
   module. `claude_code.py` now mints a gateway token and points `ANTHROPIC_BASE_URL`/
   `ANTHROPIC_AUTH_TOKEN` at the gateway instead of carrying over the host's real Anthropic
   credentials (API key, OAuth login, Bedrock env) — verified against the real `claude` CLI
   end-to-end, with the request genuinely reaching a real configured backend (Amazon Bedrock)
   *through* the gateway's translation, not around it. `codex.py` writes a
   `[model_providers.agency-proxy]` block into its isolated `CODEX_HOME/config.toml` pointing at
   the gateway with `wire_api = "responses"`; unverified against a live binary, same caveat as
   before. Every backend now genuinely routes its LLM traffic through `agproxy_llm` — no harness is
   left free to use its own host credentials/endpoint.
