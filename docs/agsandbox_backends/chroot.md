# Chroot backend (`agsandbox_backends/chroot.py`)

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

`<tmp>/agency-chroot-sandboxes/jails/<sandbox-name>/` is the jail root. `workspace/` inside it is a plain host directory (no bind mount needed, since chroot just repoints `/` — `/workspace` inside the jail *is* that directory) and is the only thing that persists across execs. `bin`, `sbin`, `lib`, `lib32`, `lib64`, `usr`, `etc` are bind-mounted read-only from the host on every exec (so the jail gets the host's own interpreters/system libraries without needing a separate image), and `dev` is bind-mounted read-write (unscoped — see "What it does not isolate" below) since most programs assume `/dev/null`, `/dev/urandom`, etc. exist and are writable. A fresh `procfs` is also mounted so PID tracking keeps working.

## Checkpointing

`commit`/`restore`/`stop(commit=...)`/`fork`/`tag_image`/`export_image`/`import_image` all operate on the `workspace/` directory instead of a container filesystem. A commit is `cp -a --reflink=auto <workspace> <tmp>/agency-chroot-sandboxes/snapshots/<sanitized-tag>` — a true point-in-time copy (using a filesystem reflink where available, a full copy otherwise), not a hardlink clone that a later in-place write to the live workspace would silently corrupt. `export_image`/`import_image` tar/untar that snapshot directory.

## Cross-process safety

Tool calls with `run_in_subprocess=True` (the default) each get a *fresh* cloudpickled copy of the backend dispatched to a `ProcessPoolExecutor` worker, so a worker's own `self._started` is unreliable — it reflects whatever the object looked like at the *original* process's last pickle, not what a different worker already did. `_ensure_started()`/`commit()`/`stop()` therefore check the workspace directory's existence on disk as their ground truth instead of trusting `self._started`, exactly analogous to how `_ContainerBackendBase` falls back to `_container_running()` (querying the docker/podman daemon — see [container.md](container.md)) instead of trusting its own `_started` in the same situation — there's no daemon here, so the workspace directory itself is the cross-process source of truth. Getting this wrong previously caused a real bug: a file written by one worker was wiped by the very next worker's `_ensure_started()` re-materializing from the last checkpoint, because that worker's own (stale) view said nothing had started yet.

## What it does *not* isolate

By design (matches the scope this backend was built for — filesystem containment + independent per-agent installs, not containment against adversarial code):

- **Network** — the jailed process shares the host's network stack; there is no network namespace.
- **Processes** — there is no PID namespace. The fresh `procfs` mounted into the jail reflects the *host's* real process table, so a command running inside the jail can *see* every host process (though signalling/killing them still goes through the kernel's normal permission checks against the real, unprivileged host uid the mapped "root" resolves to — it can't act on processes it doesn't own).
- **CPU/memory** — no cgroup of its own; `update_limits()` is a no-op.
- **GPU device scoping** — `CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` are still exported the same way as the container backends, but nothing stops a process from seeing every `/dev` entry the host user can.

## Cross-process checkpoints

`agent.save()` records which backend produced a checkpoint (`state["sandbox_image_kind"]`, either `"container"` or `"chroot"`) and `agent.load()` routes `import_image`/`tag_image`/`delete_image` to the matching backend class via `agSandbox.backend_for_image_kind(kind)` (see [base.md](base.md)), forcing the reconstructed sandbox's `backend` config to `"chroot"` when needed — auto-detection (which prefers podman/docker when available) would otherwise pick a backend that can't make sense of a chroot snapshot's tag.
