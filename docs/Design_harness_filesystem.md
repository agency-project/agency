# Harness Filesystem Design

> **Status: SUPERSEDED.** The project reconsidered and decided to run the harness process inside
> the sandbox container after all — [Design_harness_integration.md](Design_harness_integration.md)'s
> "Prerequisites" and "Design Tensions" sections now describe the adopted in-container supervisor
> bridge (a `docker exec`-launched entrypoint plus an IPC relay to the host-side `agpolicy`), which
> this document's constraint #1 explicitly argued against building. That argument is preserved
> below as a record of why the FUSE approach was attempted and what it would have cost, not because
> it's still the plan. The filesystem-visibility gap this document exists to close doesn't need
> closing under the adopted direction: a harness running inside the container sees the container's
> real filesystem the same way any process inside it does — no interception layer required.
>
> Extends [Design_harness_integration.md](Design_harness_integration.md) — read that first. This document
> covers exactly one gap in the shipped design: **a harness-driven agent's filesystem and tool-call
> visibility today only work for the chroot sandbox backend**, because the harness process (Claude
> Code, Codex, opencode, Grok Build) launches as a bare host process with `cwd` set to an ephemeral
> host tempdir (`agharness.materialize_config_home()`), never inside the sandbox's own namespace.
> For docker/podman-backed agents this means a skill telling the harness to read/write
> `/workspace/foo` fails or hangs, since `/workspace` doesn't exist on the host at all. This
> document proposes closing that gap **without** running the harness process inside the
> container, and **without** replacing the harness's own built-in toolkit with MCP — see "Rejected
> Alternatives" for why those two more obvious-looking fixes don't fit this framework's constraints.

## Constraints this design must satisfy

Established over the course of scoping this (see conversation history for the full back-and-forth
that ruled out simpler-looking alternatives):

1. **The harness process must never run inside the sandbox container/jail.** Placing it there
   (e.g. `docker exec <container> claude ...`) would tie the harness's process lifetime to the
   sandbox's, and — more importantly — would require a second, in-container tracer entrypoint
   with an IPC relay back to the host `agpolicy` to recover syscall-level visibility, since a
   host-forked ptrace supervisor cannot see into a container's separate PID namespace. Rejected.
2. **The harness's own built-in toolkit (Bash, Edit, Read, ...) must stay intact and actually
   execute.** MCP-based tool replacement (registering agency's own tools as MCP tools and denying
   the harness's built-ins) would give clean per-tool-call visibility "for free," but breaks this
   constraint outright, and isn't uniformly supported for *denying* built-ins across all four
   harnesses. Rejected.
3. **Individual tool calls (and their start/end boundaries) must be detectable**, so that sandbox
   lazy-start/hibernate can key off the same event granularity the native ReAct loop already uses
   — not "eagerly start the sandbox before every harness launch just in case."
4. **All actual file/process execution must go through the existing `agsandbox_backend`
   abstraction** (`read_file`/`write_file`/`_container_exec`/etc.) — the same code native tool
   dispatch already uses — not a parallel redirection mechanism.
5. Minimal changes to the harnesses themselves (Core Principle of the parent doc, unchanged).

## Why syscall interception (the parent doc's Component 3) can't get you here alone

`agproxy_ptrace` observes and can allow/deny/rewrite `argv[0]`, but rewriting an arbitrary path
argument to point somewhere else does not — and structurally cannot — relocate *where that path
resolves*, because the tracee is still living in whatever mount namespace it was `execve`'d into.
Making a denied-and-faked syscall actually behave like a real one (return a working fd that
subsequent `read`/`write`/`close` calls keep using correctly) means building a virtual
per-process file-descriptor table in userspace and forging every subsequent syscall's registers —
real, buildable (it's what gVisor's ptrace platform does), but a different-order engineering
project than this codebase's "one more file per backend" pattern, and it duplicates work the
kernel already does correctly for a real filesystem. See `Design_harness_integration.md`'s "File
opens" section, which reached the same conclusion for the narrower single-path-rewrite case and
punted to `agsandbox`'s bind-mount mechanism instead — this document generalizes that same
instinct to a case (docker/podman, no plain host directory to bind-mount from) where a plain bind
mount isn't available either.

Syscall tracing also can't reconstruct tool-call *boundaries*: many of a harness's built-in tools
(a Grep-tool doing an in-process regex scan, an Edit-tool doing read-modify-write) never
`execve` anything distinguishable, and even where they do, there's no syscall that says "this
`openat` is tool call #12" versus the harness's own internal bookkeeping opening the same kind of
path. Reconstructing tool-call semantics from syscalls is exactly the fragile inference the native
ReAct loop avoids by having Python's own dispatch code *be* the decision point.

## The two mechanisms this design combines

### 1. Tool-call detection at the LLM proxy, not the syscall layer

`agproxy_llm_adapters.py` already parses `tool_use`/`tool_result` content blocks out of every
request/response crossing the gateway, for both translate routes:

- **Anthropic Messages** (`/v1/messages`, Claude Code): `anthropic_messages_to_openai()`
  (`agproxy_llm_adapters.py:94-178`) already detects `block["type"] == "tool_use"` (line 140) and
  `"tool_result"` (line 119) on the way in; `openai_response_to_anthropic_message()`
  (`agproxy_llm_adapters.py:193-224`) already builds `tool_use` blocks from the configured
  backend's `tool_calls` on the way out (lines 202-209).
- **OpenAI Responses** (`/v1/responses`, Codex): `responses_request_to_openai()`
  (`agproxy_llm_adapters.py:393-456`) detects `item["type"] == "function_call"` (line 410) /
  `"function_call_output"` (line 427); `openai_response_to_responses_api()`
  (`agproxy_llm_adapters.py:459-506`) emits `function_call` items (lines 479-488).
- **chat-completions** (`/v1/chat/completions`, opencode/Grok, passthrough): no parsing today —
  this route forwards the body unmodified (`agproxy_llm.py:138-159`). Passthrough mode means the
  wire format already matches chat-completions' own `tool_calls`/`role: "tool"` shape, so adding
  the same shallow inspection here (without reshaping anything) is a small, additive change.

This is real, protocol-level, unambiguous tool-call visibility, already half-built as a byproduct
of wire-format translation — it just isn't logged or acted on yet. Each of the three route
handlers (`agproxy_llm.py:138-159`, `161-203`, `228-258`) gets a small addition: after parsing
`tool_use`/`function_call` items out of a response about to be relayed back to the harness, record
`{turn_id, tool_call_id, tool_name, tool_args}` and open a "tool call in flight" window for this
agent; when the *next* request from the harness carries the matching `tool_result`/
`function_call_output`, close the window and log the result. This mirrors exactly the
`tool_name`/`tool_args` fields already added to `agsyscallevent`
(`harness/agproxy_ptrace.py:99-123`, added specifically so `agtool.dispatch_tools()`'s
native-tool retrofit and this could share one event type — see `agpolicy.py:45-49`, `agdecision`
at `agpolicy.py:26-42`) rather than inventing a parallel policy interface.

**This window is the lazy-start/hibernate trigger**: `ag.sandbox._backend._ensure_started()` fires
when a tool-call window opens (a real "the model is about to use a tool" signal, arriving
*before* the harness's own local execution happens), and hibernate/commit fires at task end exactly
as it does today — no change to `_ensure_started()`'s idempotent inspect-and-reuse logic
(`container.py:809-928`) or to task-boundary teardown.

### 2. A FUSE-backed filesystem, mounted in a private namespace around the harness process

The harness process still runs on the bare host, in its own **private mount namespace**
(`unshare --user --map-root-user --mount`) — exactly the primitive `_ChrootBackend` already uses
for every native tool call (`sandbox/chroot.py:812-821`,
`_chroot_unshare_prefix()`/`_run_unshared()`, `chroot.py:276-283`/`786-861`). Inside that private
namespace, a FUSE filesystem is mounted at the paths that should transparently resolve into the
sandbox; every `open`/`read`/`write`/`readdir`/`stat` the harness (or any child process it spawns —
its own real Bash tool, `grep`, the dynamic linker) issues against those paths is trapped by the
kernel and handed to a userspace callback server, which dispatches into the **same
`agsandbox_backend` primitives native tool dispatch already uses**:
`read_file`/`write_file`/`write_file_bytes`/`remove_files`/`_container_exec` (signatures at
`sandbox/base.py:532-613, 855-863` and `container.py:1119-1159`). No virtual
fd table, no register forging — the kernel's real VFS layer does all the POSIX bookkeeping
(offsets, partial reads, `stat` fields); your code only answers "what's the content" and "what's
in this directory," the same two questions `read_file`/`_container_exec`-driven `ls` already
answer for native dispatch.

Because children inherit their parent's mount namespace, this "just works" for the harness's own
Bash tool spawning `ls`/`grep`/`cat`/whatever with zero extra plumbing — those really are the
sandbox's files, resolved by the kernel, not approximated.

Empirically confirmed on this host (not assumed): a bare, unprivileged `mount(2)` for `-t fuse`
fails outside any namespace (`EPERM`, confirmed via direct `ctypes` call as the plain host user);
the identical call **succeeds** inside `unshare --user --map-root-user --mount` (verified via
`/proc/self/mountinfo` showing the resulting mount), for the same reason `chroot.py`'s bind-mounts
already work unprivileged there — the mapped "root" inside the userns has `CAP_SYS_ADMIN`
*within that namespace* (confirmed via `/proc/self/status`'s `CapEff` inside the unshared
process), no host-level privilege escalation involved. `allow_other` is unavailable on this host
(`/etc/fuse.conf` doesn't have `user_allow_other` uncommented) but also unneeded — nothing outside
the mapped-root process inside this private namespace needs access to the mount.

#### Scope: whole rootfs, tiered by mutability — not just `/workspace`

Restricting the FUSE mount to `/workspace` leaves every other path (`/usr`, `/etc`, wherever `pip
install`/`apt install` would write) resolving against the **real host filesystem**, unsandboxed —
a live containment hole, not just a completeness gap. The fix is covering the whole rootfs the
harness process sees, tiered by how each region actually behaves, so the common case (reading
static binaries/libraries) doesn't pay a container round-trip per syscall:

| Region | Handling | Why |
|---|---|---|
| Static image layers (`/usr`, `/bin`, `/lib`, image-baked `/etc`) | Local read cache, synced once per launch (`docker export`/tar extraction, or lazy on-first-read with FUSE kernel attribute/entry caching enabled) | Read-mostly; a per-syscall `docker exec` round-trip for every `ld.so`/libc page-in would make even `ls` unusably slow |
| Mutable state (`/workspace`, anything a tool call writes outside it — e.g. `site-packages` after `pip install`) | Live-proxied through `agsandbox_backend`, no caching | Must stay authoritative in the container; a write here needs to actually land there |
| Pseudo-filesystems (`/proc`, `/dev`, `/sys`) | Excluded from the FUSE mount entirely — left as the host's own | These are kernel views of *whichever* namespace's processes/devices; since the harness process itself lives on the host, the host's own `/proc` is the semantically correct one for it, not a gap |

A write landing outside the initially-mutable set (a mid-task `pip install` writing into
`/usr/lib/python3.x/site-packages`, still nominally under the "static" tier) needs detecting and
promoting that subtree into the live-proxied set for the remainder of the task — open item, see
below.

For the live-proxied paths, avoid spawning a fresh `docker exec` per syscall: hold one persistent
`docker exec -i <container> <small-rpc-loop>` session per launch, sending file-op requests over
its stdin and reading results off stdout, amortizing exec-setup cost into one long-lived process.

#### Where the mount setup runs

`TracerLoop._child_exec()` (`agproxy_ptrace_internal/_tracer_loop.py:257-289`) is the traced
child's pre-`execve` code, run in exactly the process that needs the new mount namespace (mount
namespace changes must happen in the process that will use them — a parent can't set this up on a
child's behalf after the fact). Confirmed order today: dup stdio → `chdir(cwd)` (line 268) →
`PTRACE_TRACEME` (line 269) → `SIGSTOP` (line 274, synchronization point so the parent can set
`PTRACE_O_TRACESECCOMP` before the filter is live) → install seccomp filter (line 275) →
`execve` (line 277). There is currently **no `preexec_fn`-style hook** here — `_child_exec`'s only
customization points are its fixed `argv`/`envp`/`cwd` args — so wiring in the
`unshare`+FUSE-mount sequence means directly editing `_child_exec`, inserting the namespace/mount
setup between `chdir` and `PTRACE_TRACEME` (before the seccomp filter is live, so the mount
syscalls themselves aren't subject to it).

## Per-backend launch changes

None of the four harness backends' own generated configs reference `/workspace` today — confirmed
by inspection: `claude_code.py`'s `argv`/`envp` (lines 81-126), `codex.py`'s `config.toml` (lines
129-140), `opencode.py`'s `opencode.json` (lines 121-133), and `grok.py`'s `config.toml` (lines
143-158) all use `cwd=str(config_home)` (an ephemeral tempdir) and never mention `/workspace`
anywhere in argv, envp, or written config. This is good news for this design: `/workspace` (or
whatever the FUSE mount root is) is a **new** convention this design introduces, not an existing
behavior to preserve — each backend's `cwd` simply changes from `str(config_home)` to the
FUSE-mounted workspace root, and `config_home`'s own files (the per-launch isolated config each
backend already writes) can live alongside it, outside the tiered/proxied regions, exactly as
today.

## What reuses existing code unchanged

- `agsandbox_backend.read_file`/`write_file`/`write_file_bytes`/`remove_files`/`_container_exec`
  (`base.py:532-613, 855-863`, `container.py:1119-1159`) — the FUSE callback server's entire
  backend, no new sandbox-side primitives needed.
- `_ensure_started()`'s idempotent inspect-and-reuse logic (`container.py:809-928`) — called once
  when a tool-call window opens; unchanged internals.
- `agpolicy.check(ag, event)` / `agdecision` (`agpolicy.py:26-58`) — a FUSE callback populates a
  new `agsyscallevent` (e.g. `syscall="fuse_open"`, `path` set, `argv`/`envp` left `None`) and
  calls the exact same policy hook, following the precedent already set by the
  `tool_name`/`tool_args` fields added for the native `dispatch_tools()` retrofit
  (`agproxy_ptrace.py:99-123`) — no parallel policy interface.
- Chroot's own `_run_unshared`/`_chroot_unshare_prefix` primitive (`chroot.py:276-283, 786-861`) —
  directly reused for the private-namespace setup, not reimplemented.
- `agproxy_llm_adapters.py`'s existing `tool_use`/`tool_result`/`function_call` parsing
  (`agproxy_llm_adapters.py:94-178, 393-456`) — extended with logging/window-tracking, not
  rewritten.

## Does the chroot backend need any of this?

No. Chroot's jail is a real host directory (`_CHROOT_JAILS_DIR/<name>/workspace`); `chroot(2)`
just relabels root, and every file op already resolves through the kernel against real inodes with
no interception at all. The FUSE layer exists specifically to give docker/podman-backed sandboxes
the same "genuinely see the sandbox filesystem" property chroot already has for free, without
requiring the harness process to physically enter the container. A chroot-backed agent's harness
launch can skip the FUSE mount and bind-mount the jail's real workspace directory directly, exactly
like `_ChrootBackend`'s existing tool-call path does.

## Open items — not yet verified or resolved

- **No Python FUSE binding is a dependency today.** `fusepy`/`pyfuse3`/`llfuse` are all absent from
  `pyproject.toml`/`uv.lock` and not installed in this environment. Implementing the callback
  server means adding one as a new dependency (or hand-writing a minimal raw `/dev/fuse` protocol
  handler via `ctypes`, mirroring how `chroot.py` already hand-rolls its own `unshare`/`mount`
  calls rather than depending on a wrapper library — worth weighing against just taking the
  `fusepy` dependency).
- **`_child_exec` needs a direct edit**, not an added hook — there's no `preexec_fn`-equivalent
  today. Whether to add one (so this stays additive rather than modifying shared tracer code) or
  edit `_child_exec` in place is an implementation decision, not resolved here.
- **Mid-task promotion of a write outside the initial mutable-path set** (e.g. a `pip install`
  landing in a "static" region) is identified but not designed — needs either a copy-on-write
  promotion rule or treating a wider region as live-proxied from the start at some cache-locality
  cost.
- **Verified only for the bare-host/chroot-style unshare path.** Whether the same unprivileged
  userns FUSE mount works identically when the *calling* process (the tracer/supervisor) is itself
  already running inside constrained environments (e.g. nested inside another container) was not
  tested — only a plain host shell was used for the empirical `mount(2)` check above.
- **The tool-call-window ↔ FUSE-mount-lifetime relationship isn't fully specified.** Does the FUSE
  mount live for the whole harness launch (simplest — one mount, one private namespace, spanning
  every tool-call window in that turn) or per-tool-call-window (tighter scoping, more setup/teardown
  churn)? This document assumes the former (one mount per harness launch, matching one
  `agProxyPtrace.launch()` call) but doesn't argue for it over the alternative.
- **chat-completions passthrough route's tool-call detection** (opencode/Grok) is described above
  as "a small additive change" but the exact shape of that addition isn't drafted here.
