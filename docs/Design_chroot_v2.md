# Chroot Sandbox v2: Namespace-Isolated Jail Design

> **Status:** proposed, not yet implemented. Extends `agsandbox_backends/chroot.py`'s existing
> `_ChrootBackend` in place — same public contract (`agsandbox_backend`'s interface), same
> `agSandboxConfig(backend="chroot")` selector, same checkpoint/restore-as-directory-copy model.
> This document proposes closing the three gaps the current implementation's own module docstring
> explicitly documents as out of scope (chroot.py:15-32) — no network namespace, no PID namespace,
> no cgroup limits — plus replacing `chroot(2)` itself with Landlock, which incidentally fixes a
> fourth, previously-unfixable gap (GPU access). Every claim below was verified live on this host
> (Ubuntu, kernel 5.15.0-1095-nvidia) before being written down; verification commands are recorded
> inline so they can be re-run on a target deployment host before relying on them there.

## Why this, and why now

Two motivations converged on the same target:

1. **The container backend's isolation properties — network namespace, PID namespace, cgroup
   limits — are the reason a harness-driven agent (Claude Code, Codex, opencode, Grok Build) can't
   just use chroot today**, per the discussion in
   [Design_harness_filesystem.md](Design_harness_filesystem.md): those docker/podman-specific
   properties are exactly what motivated the FUSE-based filesystem bridge, since chroot lacked
   them. If chroot can be extended to provide the same properties, the FUSE design's entire
   reason for existing — bridging the harness into a container's namespace without moving the
   harness process into it — goes away, and a harness backend can launch directly into an extended
   chroot jail with no bridge needed at all (a real host directory the harness natively sees,
   exactly like today's chroot-backed native tool calls).
2. **Every gap has an independently-available, unprivileged Linux primitive**, confirmed live on
   this host rather than assumed from documentation — see "Verified capabilities" below.

## What the current implementation already gets right (unchanged by this design)

- Filesystem containment: `chroot(2)` into a per-agent directory with the host's own
  interpreters/libraries bind-mounted read-only (`_setup_lines()`, chroot.py:712-772).
- No root/sudo/setcap needed: `unshare --user --map-root-user --mount` (or the `rootlesskit
  --net=none unshare` fallback for AppArmor-restricted hosts, chroot.py:223-240).
- Checkpoint/restore as a real point-in-time directory copy (`cp -a --reflink=auto`), not an
  image layer — this document does not change the checkpoint model at all.
- PGID-based background-process tracking (`_live_pgid_matched_pids()`, chroot.py:91-109) — reused
  as-is; see "PID namespace" below for what changes and what doesn't.

None of this needs to be rebuilt. The changes below are additive to the same `unshare` invocation
and the same jail-script generation, not a rewrite of the backend's architecture.

## The four changes

### 1. Network namespace — add `--net` to the existing `unshare` call

**Today:** the jailed process shares the host's real network stack entirely (chroot.py:17-18) —
no boundary at all. The `rootlesskit --net=none` fallback (used only on hosts where bare
`unshare` is AppArmor-denied) explicitly *disables* networking in `rootlesskit`'s own setup today,
specifically because the jail never asked for a network namespace of its own.

**Change:** add `--net` to the unshare invocation. Verified live on this host:

```
$ unshare --user --map-root-user --net --mount -- ip addr
1: lo: <LOOPBACK> mtu 65536 qdisc noop state DOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
```

A fresh, empty network namespace — only loopback exists, down by default. `ip link set lo up`
inside it brings up a fully functional loopback (confirmed: `127.0.0.1`/`::1` both live), with
zero visibility into the host's real interfaces. This is the same namespace primitive
docker/podman use for their own default network isolation — just applied to the existing
`unshare` call instead of a separate container runtime.

**Open item:** a jail with `--net` and no further setup has no route to the outside world at
all (correct for pure filesystem-containment workloads, wrong for anything needing outbound
network — e.g. a harness's own LLM calls, or `pip install` reaching PyPI). Giving the jail
controlled connectivity (a veth pair to a host bridge, or simply not adding `--net` when the
workload needs real network access and relying on Landlock/other layers instead) is a per-workload
policy decision this document doesn't resolve — see "Open items."

### 2. PID namespace — add `--pid --fork`, which *now* also fixes the procfs restriction

**Today:** deliberately not created. The module docstring (chroot.py:49-60) explains the
specific reason: mounting a *fresh* procfs instance inside the jail requires the caller's user
namespace to own the target PID namespace, and this jail's `unshare` only ever created a
user+mount namespace — no PID namespace of its own — so there was no PID namespace for the
user namespace to own, and the mount was refused (`EINVAL`, confirmed empirically per the
docstring). The workaround was reading the *host's* real `/proc` from outside the chroot, with
correctness for background-job tracking then handled entirely via PGID matching instead of PID
namespace isolation.

**Change:** add `--pid --fork` to the same `unshare` call. Because the namespace being created
and the namespace doing the mount are now created together, the earlier restriction's own
precondition is satisfied. Verified live on this host:

```
$ unshare --user --map-root-user --mount --pid --fork -- sh -c \
    "mkdir -p /tmp/newproc && mount -t proc proc /tmp/newproc && ls /tmp/newproc | head -5"
1
4
5
acpi
bootconfig
```

A genuinely fresh procfs, mountable and readable, showing only the new PID namespace's own
processes (PID 1 being the jailed process itself) — not the host's.

**What this changes for PID tracking:** `_live_pgid_matched_pids()`'s PGID-matching approach
(chroot.py:91-109) can stay as the *within-jail* tracking mechanism — it still works fine, and its
resilience to `setsid()`-detached daemonizing processes escaping tracking is unrelated to which
PID namespace is in use. What changes is *scope*: `_read_proc_table()`'s current "read the whole
host `/proc`, from outside the chroot" (chroot.py:56-60) is no longer necessary — with a fresh,
owned PID namespace, the jail's own `/proc` can be mounted and read directly, containing *only*
this jail's processes. This also closes the "can see every host process" visibility leak the
current docstring documents as accepted (chroot.py:20-22) — with a real PID namespace, there is
nothing outside this jail's own process tree to see at all.

**Resolved via a lightweight per-jail daemon** — see "The per-jail daemon" section below. A real
PID namespace can only ever be read from inside itself or by a descendant of the process that
created it (chroot.py:71-80's `nsenter`-from-outside rejection still applies to any later,
unrelated process, including a later call from the very same orchestrator). The fix is not to
avoid this restriction but to never need to re-enter after the fact: spawn one small relay process
*inside* the jail's namespaces at jail-creation time, before anything else, and have every later
`exec()`/`get_live_pids()`/etc. call talk to it over a socket instead of trying to join the
namespace again.

### 3. cgroup resource limits — real delegation confirmed present on this host, not assumed

**Today:** `update_limits()` is a literal no-op (chroot.py:960, docstring at chroot.py:22-23) —
no CPU or memory ceiling exists for a chroot-backed jail at all.

**Change:** create a per-jail cgroup under this user's own delegated systemd slice, move the
jailed process's PID into it, and write real `memory.max`/`cpu.max` controls. This is *not* tied
to mount/PID/network namespaces at all — cgroups are an orthogonal kernel mechanism; any process
can be moved into any cgroup its owner has write access to, regardless of what namespaces that
process is in.

**Verified present on this host** (this is genuinely host/distro/systemd-version dependent, and
must be re-probed on any deployment target — see below):

```
$ cat /proc/self/cgroup
0::/user.slice/user-200682.slice/session-211335.scope
$ find /sys/fs/cgroup/user.slice/user-200682.slice/ -maxdepth 2 -writable
/sys/fs/cgroup/user.slice/user-200682.slice/user@200682.service
/sys/fs/cgroup/user.slice/user-200682.slice/user@200682.service/cgroup.procs
/sys/fs/cgroup/user.slice/user-200682.slice/user@200682.service/cgroup.subtree_control
```

`user@<uid>.service`'s `cgroup.procs`/`cgroup.subtree_control` are writable by this user —
standard systemd user-session delegation (the same primitive `systemd --user` services and
rootless Podman rely on). A per-jail sub-cgroup created under this path, with the jailed PID moved
into it and `memory.max`/`cpu.max` written, gets real kernel-enforced resource limits, unprivileged.

**Required addition to `chroot_available()`'s capability probe:** unlike the network/PID namespace
changes (which only need `unshare` flags that either work or don't, uniformly), cgroup delegation
depth genuinely varies by distro, systemd version, and PAM/login-manager configuration — a host
without `systemd-logind` managing sessions, or with delegation disabled, may not have any writable
cgroup subtree at all. This needs its own live probe (mirroring `_probe_chroot_available()`'s own
pattern of "don't trust documentation, do a real check," chroot.py:243-273) — e.g. attempting to
create a test subdirectory under the user service slice and write to its `cgroup.procs` — with
`update_limits()` falling back to today's no-op (with a logged warning) rather than failing hard
when delegation isn't available. Cgroup support should be treated as a **best-effort, probed
capability**, not an assumed-present one, exactly like the existing `chroot_available()`/
`ptrace_available()` pattern elsewhere in this codebase.

### 4. Replace `chroot(2)` with Landlock — fixes GPU access as a side effect

**Today:** `chroot(2)` itself is confirmed (chroot.py:34-44, empirically on real H100 hardware) to
trigger the NVIDIA driver's own chroot-detection, which unconditionally refuses GPU access — even
with every device node correctly bind-mounted and reachable. The docstring is explicit that
nothing short of not chrooting at all fixes this, since it defeats the point of the backend as
currently built.

**Change:** replace the `exec chroot <root> <shell> -c <cmd>` step (chroot.py:783) with a Landlock
ruleset that restricts the process to the jail's own directory tree, applied via
`landlock_restrict_self` instead of the `chroot(2)` syscall. Verified live on this host — Landlock
is available, ABI version 1, and works fully unprivileged:

```c
// landlock_create_ruleset(&attr, sizeof(attr), 0) → succeeds, returns a ruleset fd, no privilege needed
// prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) → required before restrict_self
// landlock_restrict_self(ruleset_fd, 0) → succeeds
// fopen("/etc/hostname") afterward, with no rule granting it → fails (Operation not permitted / EACCES)
```

Confirmed: with only `LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_WRITE_FILE` handled and no
rule added for `/etc/hostname`, opening it after `restrict_self` was denied — a real, enforced,
per-syscall filesystem restriction, not merely advisory. And critically: `nvidia-smi` inside
`unshare --user --net --pid --mount --fork` (all three new namespaces from items 1-2, still with
no `chroot(2)` call anywhere) **succeeds** — confirmed live, listing all GPUs on this host
normally. This isolates the NVIDIA driver's refusal to the `chroot(2)` syscall specifically, not
to namespaces in general, confirming that swapping the confinement mechanism (Landlock instead of
`chroot`) is what unblocks GPU access, not anything about the namespace changes above.

**What changes in practice:** the jail's directory-materialization step
(`_setup_lines()`/`_build_jail_script()`) stays almost identical — the bind-mount setup for
`/bin`, `/usr`, per-file `/dev` nodes, per-jail `/workspace`, and user-declared `self._mounts`
all remain necessary, since Landlock restricts *access*, it doesn't relabel root the way `chroot`
does. What changes is the final exec step: instead of `chroot <root> <shell> -c <cmd>`, the
jailed process (still inside the same `unshare`'d mount namespace, so bind mounts still apply)
builds a Landlock ruleset scoped to `<root>` (and nothing else) and calls `restrict_self` on
itself, then execs the shell directly — no `chroot(2)` call at all.

**Real tradeoff, not a strict improvement — must be verified before relying on it:** Landlock
denies *access* to paths outside the ruleset; it does not make those paths *not exist* or
resolve elsewhere the way `chroot` does. A process inside a Landlock-restricted (not chroot'd)
jail sees real absolute host paths — `ls /` shows the real host root, `/home/other-agent/...` is
a path that exists and is merely access-denied, not nonexistent. Landlock also does not restrict
the view a jailed process has of `/proc`'s own path-based metadata the way an actual pivoted root
would, and it's a filesystem-access-only mechanism — it does no path virtualization, so anything
depending on chroot's "your `/workspace` really is absolute path `/workspace`, not
`/real/host/path/workspace`" property (bind-mount targets, tool-reported paths, `agpath`-typed
skill inputs/outputs) needs checking against Landlock's real-path behavior before this is treated
as a drop-in replacement. This is the single largest open risk in this whole design and needs
concrete verification against this codebase's actual path-handling (`agtype.agpath`, `agschema`'s
output-schema recovery) before implementation, not assumed compatible.

## The per-jail daemon

### Why the daemon is needed at all

Today, `_run_unshared()` spawns a brand-new `unshare` subprocess per call (chroot.py:786-861) and
lets it exit when the command finishes — there is no state to preserve between calls, so nothing
persists and nothing needs to. Adding a real PID namespace (`--pid --fork`, above) breaks this:
once that one-shot process exits, its PID namespace is gone, and a *later* call — even from the
same orchestrating process — cannot rejoin it. This isn't a missing feature to work around; it's
a kernel-enforced rule already confirmed empirically in this exact codebase (chroot.py:71-80):
joining an existing user namespace from outside it is refused unless the joining process is a
*descendant* of the process that created it, full stop, even for that namespace's own creator.

So the fix isn't "find a way back in" — there isn't one — it's "never leave": spawn one small
process inside the jail's namespaces once, when the jail is first created, and keep it running for
the jail's whole lifetime. Every later operation talks to that process instead of trying to
re-enter anything.

### Addressed by a well-known path, not a Python handle

The daemon listens on a Unix domain socket at a path deterministically derived from the jail's own
name — `_CHROOT_JAILS_DIR / <name> / "daemon.sock"` — created once, at the same time as the jail's
root directory. Crucially, **the daemon's handle is never stored as a live Python attribute**
(no `Popen` object, no open socket kept around in `_ChrootBackend.__dict__`). Every caller —
whether the original orchestrating process or a `ProcessPoolExecutor` worker that received a fresh
cloudpickled copy of `_ChrootBackend` — reconnects by recomputing the same path from `self._name`,
a plain string field that already survives pickling today with no special handling. This is the
same trick a real docker daemon relies on: the `docker` CLI never holds a live reference either, it
just connects to `/var/run/docker.sock` fresh, every single invocation. It sidesteps the
cloudpickle-boundary problem entirely, rather than solving it — there's no live handle to lose in
the first place.

This is also the direct answer to whether `run_in_subprocess=True` custom tools can keep touching
the sandbox: yes, unchanged. A pool worker's copy of `_ChrootBackend` has `self._name`; it derives
the same socket path and connects, exactly like the orchestrator would. No new restriction on
`agtool`/`agsandbox` is needed.

### Protocol and concurrency — deliberately minimal, no new dependency

A small length-prefixed JSON request/response protocol over the socket, stdlib-only
(`socket`/`json`/`threading`), covering exactly the operations `_ChrootBackend` already exposes:
`exec(cmd, workdir, timeout, stdin)`, `read_file(path)`, `write_file(path, content)`,
`get_live_pids()`, `kill_all()`. Concurrent tool calls from one ReAct turn (or from several
different workers at once) can arrive in parallel, so the daemon accepts multiple simultaneous
connections via a small thread pool — one thread per connection, each running the requested
operation and writing back a response — rather than serializing every request through one loop.
This is meaningfully *less* machinery than the container backend already depends on (a real docker
daemon, its own supervised RPC surface); it only needs to speak a private protocol to
`_ChrootBackend` itself, not a general one.

### Lifecycle — the one genuinely new piece of bookkeeping

Unlike today's self-cleaning one-shot subprocess, a long-lived daemon needs explicit lifecycle
handling, mirroring patterns already used elsewhere in this codebase rather than inventing new
ones:

- **Start-once, race-free.** Two callers discovering the jail's daemon isn't running yet (e.g. two
  concurrent workers on a cold jail) must not both try to spawn it. A pidfile + advisory lock at a
  well-known path, checked the same "don't trust an in-memory flag, check ground truth on disk"
  way `_ensure_started()` already treats `self._workspace.is_dir()` (chroot.py:141), with the
  race's loser simply connecting to the socket the winner just created instead of erroring.
- **Staleness detection.** Before trusting a socket path exists, confirm the recorded PID is both
  alive *and* actually answering (connect + a trivial ping), the same live-check discipline
  `_inspect_container_state()` already applies to a container's `docker inspect`-reported state
  rather than a cached flag — an orchestrator crash could leave a dead socket file behind, and a
  stale file must not be mistaken for a live daemon.
- **Teardown.** An explicit shutdown message (or killing the recorded PID directly) at
  `stop()`/`destroy()` — nothing kills this daemon on its own the way today's one-shot subprocess
  self-terminates when its command finishes.

## What the extended jail looks like end to end

```
unshare --user --map-root-user --mount --pid --net --fork -- bash -c '
    { <bind-mount setup, unchanged from _setup_lines()> } >/dev/null 2>&1
    mount -t proc proc <root>/proc                      # NEW — fresh procfs, now mountable
    ip link set lo up                                    # NEW — bring up loopback in the fresh netns
    mkdir -p /sys/fs/cgroup/user.slice/.../agency-jail-<id>   # NEW — per-jail cgroup
    echo $$ > .../agency-jail-<id>/cgroup.procs                # NEW — move self into it
    echo "<memory limit>" > .../agency-jail-<id>/memory.max    # NEW
    echo "<cpu limit>"    > .../agency-jail-<id>/cpu.max       # NEW
    exec <landlock-restrict-self-then-exec-helper> <root> <shell> -c "<cmd>"   # CHANGED — no chroot(2)
'
```

Every line marked NEW/CHANGED is additive to the existing `unshare`/`_setup_lines()`/
`_build_jail_script()` machinery — nothing about the checkpoint model, the `agsandbox_backend`
public interface, or `for_config()`'s selection logic needs to change.

## Isolation properties summary, before vs. after

| Property | Today | This design |
|---|---|---|
| Filesystem containment | `chroot(2)`, real path relabeling | Landlock ruleset, real-path-visible but access-denied |
| Network | None — shares host stack | Own network namespace, isolated unless explicitly bridged |
| PID visibility | Host's real `/proc`, every host process visible | Own PID namespace, only this jail's processes exist |
| Resource limits | None (`update_limits()` is a no-op) | Real cgroup `memory.max`/`cpu.max`, best-effort/probed |
| GPU access | Broken (NVIDIA chroot-detection) | Expected to work (no `chroot(2)` call) — needs live confirmation with the actual Landlock-restricted exec path, not just the namespace-only probe done so far |
| Background-job tracking | PGID matching, reads host `/proc` from outside jail | PGID matching, read via a per-jail daemon's own `/proc`, relayed over a socket |
| Persistent state | None — one-shot `unshare` subprocess per call | One lightweight per-jail daemon, addressed by a well-known socket path, for the jail's whole lifetime |

## Open items — not yet resolved

- **Network namespace connectivity policy** — an isolated netns with no bridge/veth has no
  outbound path at all; whether/how to give jails controlled outbound access (needed for a
  harness's own LLM calls, `pip install`, etc.) is unresolved.
- **cgroup delegation must be probed per-host, not assumed** — verified present here via standard
  systemd session delegation, but this is exactly the kind of environment-dependent capability
  `chroot_available()` already treats as "probe live, don't trust documentation"; the same
  discipline applies here, with a soft-fail (no-op + warning) fallback rather than a hard error.
- **Landlock's real-path-visible-but-denied semantics vs. `agtype.agpath`/`agschema` path
  handling** — flagged above as the largest correctness risk; not verified against this
  codebase's actual path-typed I/O yet.
- **GPU access with the FULL new exec path (netns + pidns + cgroup + Landlock all together, not
  just namespaces alone) was not re-verified** — the `nvidia-smi` success shown above used
  `unshare --user --net --pid --mount --fork` with no `chroot(2)` call, but did not go through an
  actual Landlock-restricted exec; confirming GPU access survives the complete new jail
  construction (not just the namespace subset) is a needed follow-up check before implementation.
- **Whether this design still needs `Design_harness_filesystem.md`'s FUSE bridge at all** — if
  this chroot v2 design lands and harness backends launch directly into it, the FUSE-based
  filesystem work for docker/podman-backed harness agents may become unnecessary for any workload
  that can tolerate chroot v2's isolation profile instead of a real container's. Not decided here;
  worth revisiting once this design's open items above are resolved.
- **The daemon's RPC protocol is specified only at the level of "which operations, roughly what
  shape"** — exact wire format (message framing, error propagation, timeout behavior mid-request)
  isn't drafted. Low-risk (stdlib-only, small surface) but not yet concrete enough to implement
  from directly.
- **Daemon crash mid-jail-lifetime** isn't addressed — if the daemon process dies unexpectedly
  (OOM, bug) while the jail is otherwise still considered live, callers need a clear "reconnect
  failed, is this jail dead or just needs a daemon restart" story; today's design only covers
  orderly start/teardown and orchestrator-crash staleness, not daemon-crash-while-orchestrator-lives.
