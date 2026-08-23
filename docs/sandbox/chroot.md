# Chroot backend (`sandbox/chroot.py`)

> `_ChrootBackend` subclasses `agsandbox_backend` directly ([base.md](base.md)), not `_ContainerBackendBase` — it has no daemon, no image format, and shares none of the docker/podman-specific mechanics in [container.md](container.md)/[docker.md](docker.md)/[podman.md](podman.md).

Trades Docker/Podman's full isolation for a much lighter mechanism, for the case where you just want each agent to see only its own files and its own installed packages, with no access to the host filesystem, and don't need network/process/resource isolation.

## Availability probe

`chroot_available()` (cached for the process lifetime) checks three things, since any one alone can give a false positive:

1. `/proc/sys/kernel/unprivileged_userns_clone` isn't explicitly disabled (the file doesn't exist on distros that ship it enabled upstream — Debian/Ubuntu-specific).
2. A live `unshare --user --map-root-user --mount true` actually succeeds.
3. If (2) fails and `rootlesskit` is on `PATH`, a `rootlesskit --net=none`-wrapped retry of the same command.

The sysctl alone can false-positive: some distros (Ubuntu 24.04+ with `kernel.apparmor_restrict_unprivileged_userns=1`) deny `CLONE_NEWUSER` to unconfined binaries like a bare `unshare` call even when the sysctl says it's allowed. Those same distros still allow it for `rootlesskit`, which ships its own AppArmor profile explicitly granting `userns` — the same exemption rootless Docker/Podman rely on to keep working under that policy. `_unshare_prefix_candidates()` tries bare `unshare` first (so hosts where it already works don't gain a dependency on `rootlesskit`) and only falls back to the wrapped form if that fails.

Whichever candidate actually works is cached (`_chroot_unshare_prefix()`) and reused as the prefix for every real `_ChrootBackend` exec, not just the availability probe.

## Mechanism

Every `exec()` call runs inside a fresh `unshare --user --map-root-user --mount` (an unprivileged user + mount namespace — no root, sudo, or `setcap` needed; wrapped in `rootlesskit --net=none` on hosts where bare `unshare` is denied, see above) followed by `chroot` into a per-agent directory. The mount namespace (and everything bind-mounted into it) is torn down automatically when that one process exits — there's no long-lived daemon to exec into the way `docker exec`/`podman exec` has a container to attach to.

## Directory layout

`<tmp>/agency-chroot-sandboxes/jails/<sandbox-name>/` is the jail root. `workspace/` inside it is a plain host directory (no bind mount needed, since chroot just repoints `/` — `/workspace` inside the jail *is* that directory) and is the only thing that persists across execs. `bin`, `sbin`, `lib`, `lib32`, `lib64`, `usr`, `etc` are bind-mounted read-only from the host on every exec (so the jail gets the host's own interpreters/system libraries without needing a separate image).

`/dev` is individual **per-file** bind mounts (`null`, `zero`, `random`, `urandom`, `tty`, `full`, plus any leased GPU device — see below), not a whole-directory bind and not a tmpfs. Both alternatives were tried and rejected: bind-mounting `/dev` as a whole directory silently fails to make any device usable inside the jail (confirmed empirically — `/dev/null` becomes unwritable), and mounting a fresh tmpfs to hold the device files instead forces the `nodev` flag in a nested unprivileged user namespace, which blocks device-node access the same way. Binding each real device file directly onto a plain (never separately mounted) directory sidesteps both restrictions.

**No procfs is mounted inside the jail at all**, and none is needed. Mounting a fresh `procfs` instance (`mount -t proc`) requires the caller's user namespace to own the target PID namespace, and bind-mounting the host's existing `/proc` is refused for the identical reason (confirmed empirically via strace: `EINVAL` on `mount(2)` either way) — this jail's `unshare` only creates a user+mount namespace, never a PID namespace of its own, so the process stays in the host's real PID namespace regardless. Background-process tracking instead reads the real host `/proc` directly from *outside* the chroot (see below) — no new mount is needed for that at all, since it's the same PID namespace either way.

## Background-process tracking

Tracking is **PGID-only** — there is no before/after PID-diffing the way the container backends use (an earlier version of this backend had one; it was removed). Each `exec()` call's underlying `unshare` invocation is started as its own new process-group leader (`start_new_session=True`), and that group id is recorded into `_invocation_pgids`. `_live_pgid_matched_pids()` is the single scan (reading `pid`/`ppid`/`pgid`/`state` for every entry in `/proc`, using pure shell builtins — no `awk`/`cat` forked per entry, confirmed empirically to matter: forking per entry across thousands of `/proc` entries on a busy host took 8+ seconds) that backs all three of:

- `get_live_pids()` — every PID whose pgid is currently tracked, excluding zombies and daemon-released descendants.
- `_has_pending_background_work()` — `bool(get_live_pids())`. Since both read the same scan, they can never disagree with each other.
- `_kill_all_sandbox_processes()` — `os.killpg()` on every tracked group, called by `stop()`/`destroy()`/`restore()` before tearing down or overwriting the workspace. Every group is attempted and a non-race failure is surfaced after those attempts, so teardown cannot silently report success while a tracked process survives. There is no container-removal-style implicit kill here (chroot has no cgroup/namespace boundary whose teardown guarantees termination), so this explicit step is the only thing that stops a background process from outliving the workspace being deleted or rewritten out from under it.

Why PGID instead of a one-time "baseline" diff (tried first, then abandoned): reading another process's PGID needs no ptrace permission, and PGID membership is a real, kernel-tracked relationship established once at spawn time — immune to however much unrelated churn a busy shared host generates, and confirmed to survive a child outliving its exited parent's reparenting, as well as to catch a child spawned well *after* the `exec()` call that backgrounded its parent already returned (something a before/after diff limited to that one call's own window could never see). A baseline-diff scan was tried and rejected for the opposite reason: on a host with any background churn it could see literally everything as "not yet clear," and it could never verify a candidate PID's ownership well enough to justify `os.killpg()`-ing it.

**Cost of dropping per-PID diff tracking:** there is no longer a per-PID capture timestamp, so `pid_status_summary()` here lists matched PIDs (`"PID 1234"`) without an elapsed-time figure, unlike the container backends' `"PID 1234 (running 2m 30s)"`.

**Accepted gap:** a process that calls `setsid()`/`setpgid()` to detach into its own new process group — the same mechanism `nohup`/`disown`/many daemonizing library patterns use — escapes `_invocation_pgids` tracking entirely (confirmed empirically: `setsid sleep & echo $!` produces a child with its own distinct pgid, not the invocation's). Accepted deliberately, same reasoning as above: reverting to a baseline scan to also catch this case would trade it for the much worse "never resolves quickly on a busy host" problem.

**Known, separate limitation:** `_invocation_pgids` is a plain in-memory attribute, so it does not survive the cloudpickle/worker-process boundary that `run_in_subprocess=True` tool calls (the default) create — a worker's own copy records a PGID into its own memory, discarded when that worker process exits. `agSandbox.wait_for_processes()` is always called from the orchestrating process, whose own copy never independently ran `exec()`, so a worker-spawned background job is invisible to `_has_pending_background_work()` when checked from the orchestrator. Not yet fixed — would need persisting recorded PGIDs to disk, mirroring how `_ensure_started()` (below) already treats the workspace directory's existence as cross-process ground truth rather than anything in-memory.

## GPU device access

`CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` are set exactly like the container backends (see [container.md](container.md)) — a physical GPU is claimed lazily on the first real `exec()` and its index exported (and made `readonly`, so a command can't hijack a different GPU by reassigning the variable inline).

`_chroot_gpu_dev_paths(gpu_id)` bind-mounts the device nodes needed to actually reach that leased GPU: for NVIDIA, the one matching `/dev/nvidia<gpu_id>` compute node plus the shared control devices every GPU needs regardless of index (`nvidiactl`, `nvidia-uvm`, etc.); for AMD/ROCm, `/dev/kfd` (shared) plus the one matching `/dev/dri/renderD*` node (untested against real hardware). `gpu_id=None` (no GPU leased) mounts nothing. This mirrors what docker's `--gpus device=N`/podman's CDI actually expose to a container, rather than every GPU on the host.

There is still no per-GPU cgroup device filter here, though (unlike docker's `--gpus`/podman's CDI, which are backed by one) — this restricts what the *default* jail setup exposes, not what a process inside could reach if it somehow discovered and opened another leased jail's device node directly by path.

**NVIDIA GPUs may not be usable inside the jail at all regardless of the above**: confirmed empirically (H100, recent driver) that `nvidia-smi` fails with "GPU access blocked by the operating system" the moment the calling process has `chroot`'d — even with every relevant device node correctly bind-mounted in and reachable, and even though the exact same process calling `nvidia-smi` from a bare `unshare --user --mount` (same namespaces, no chroot) works fine. This is the NVIDIA driver's own chroot-detection refusing access, not a mount/permission/namespace issue this backend's code can route around — nothing short of not chrooting at all (defeating the entire point of this backend) fixes it. Untested against AMD/ROCm, which may not carry the same restriction.

**GPU release**: there is no `_gpu_is_clear()`/polling mechanism on this backend (or any backend — the concept was removed entirely). `agResourcePool.release_gpu()` releases the semaphore immediately, with no wait. Safety comes from ordering: `stop()`/`rm_container()`/`destroy()` all call `_kill_all_sandbox_processes()` synchronously *before* releasing the GPU, so by the time release happens, everything this backend could see and terminate already has been. All three release the GPU on every call, same as the container backends' `stop()`/`rm_container()`/`destroy()` (see [container.md](container.md)'s "GPU device access") — though for a different underlying reason here: there's no persistent container object with device flags baked in at creation at all, since GPU visibility is granted per-`exec()` via env vars (re-derived fresh on every call, not attached once), so a later resume can trivially be handed a different physical GPU. There is also no agent-facing `gpu_release` tool — once `reserve_gpu` triggers a real GPU acquisition, it's freed automatically by `stop()`/`rm_container()`/`destroy()`, not by an explicit release call.

## Checkpointing

`commit`/`restore`/`stop`/`rm_container`/`fork`/`tag_image`/`export_image`/`import_image` all operate on the `workspace/` directory instead of a container filesystem. A commit is `cp -a --reflink=auto <workspace> <tmp>/agency-chroot-sandboxes/snapshots/<sanitized-tag>` — a true point-in-time copy (using a filesystem reflink where available, a full copy otherwise), not a hardlink clone that a later in-place write to the live workspace would silently corrupt. `export_image`/`import_image` tar/untar that snapshot directory.

These are three separate, orthogonal operations (mirroring the container backends' `stop()`/`rm_container()`/`commit()` split — see [container.md](container.md)'s "Container lifecycle"):

- **`stop()`** — hibernate: kills tracked processes, releases the GPU, and stops. **Leaves the live workspace in place untouched** — unlike the container backends, there's no container process that must be torn down to "stop," so the next explicit or defensive readiness check reuses the directory directly, with no destroy-and-recreate round trip and no snapshot taken.
- **`commit(tag=None)`** — snapshots the workspace to `tag` (default: this jail's own lifecycle tag) the way described above, **without touching the live workspace** — it keeps existing exactly as before, so the sandbox resumes directly from it, no different from a container backend's `commit()` not removing the container.
- **`rm_container()`** — kills tracked processes, releases the GPU, and deletes the live workspace outright; it does **not** eagerly restore it from the last checkpoint. The next transaction's public `ensure_started()` delegates to `_ensure_started()`, which materializes from `_checkpoint_image` whenever it finds the workspace missing (see below) — doing the same copy at `rm_container()` time too would pay that cost even when the sandbox is never used again (e.g. right before a `destroy()` that deletes the whole jail root moments later), and would introduce a real failure surface (a `cp -a` that can raise) into a step that otherwise can't fail. `destroy()` calls this internally before removing the jail root.

## Cross-process safety

Ground truth for "has this jail's workspace already been set up" is the workspace directory's existence on disk (`_ensure_started()`), not any in-memory flag — there is no `self._started` cache at all on this backend (or the container backends, as of the same redesign). Tool calls with `run_in_subprocess=True` (the default) each get a *fresh* cloudpickled copy of the backend dispatched to a `ProcessPoolExecutor` worker, so a per-process flag would only ever reflect what the object looked like at the *original* process's last pickle, never what a different worker already did. `_ensure_started()`/`commit()`/`stop()` check the workspace directory directly instead, exactly analogous to how `_ContainerBackendBase` falls back to `_container_running()` (querying the docker/podman daemon — see [container.md](container.md)) rather than trusting a flag in the same situation — there's no daemon here, so the workspace directory itself is the cross-process source of truth. Getting this wrong previously caused a real bug: a file written by one worker was wiped by the very next worker's `_ensure_started()` re-materializing from the last checkpoint, because that worker's own (stale) in-memory flag said nothing had started yet.

`_ensure_started()` itself does nothing else once the workspace exists — earlier it also captured a one-time baseline PID snapshot to support diff-based tracking, but that mechanism has been removed (see "Background-process tracking" above), so there is no longer anything to capture at startup.

## What it does *not* isolate

By design (matches the scope this backend was built for — filesystem containment + independent per-agent installs, not containment against adversarial code):

- **Network** — the jailed process shares the host's network stack; there is no network namespace.
- **Processes** — there is no PID namespace. Background-process tracking reads the *host's* real process table directly (see "Background-process tracking" above), so a command running inside the jail can *see* every host process (though signalling/killing them still goes through the kernel's normal permission checks against the real, unprivileged host uid the mapped "root" resolves to — it can't act on processes it doesn't own).
- **CPU/memory** — no cgroup of its own; `update_limits()` is a no-op.
- **GPU device scoping** — bind-mounted device nodes are scoped to the leased GPU by default (see above), but there's no cgroup device filter enforcing it, so a process that discovers another device path directly isn't blocked from opening it.

## Cross-process checkpoints

`agent.save()` records which backend produced a checkpoint (`state["sandbox_image_kind"]`, either `"container"` or `"chroot"`) and `agent.load()` routes `import_image`/`tag_image`/`delete_image` to the matching backend class via `agSandbox.backend_for_image_kind(kind)` (see [base.md](base.md)), forcing the reconstructed sandbox's `backend` config to `"chroot"` when needed — auto-detection (which prefers podman/docker when available) would otherwise pick a backend that can't make sense of a chroot snapshot's tag.
