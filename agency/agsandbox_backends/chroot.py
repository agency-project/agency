"""Chroot backend -- no daemon, no image format: isolation is a per-agent
directory chrooted into via an unprivileged user+mount namespace
(`unshare --user --map-root-user --mount`, transparently wrapped in
`rootlesskit --net=none` on hosts that deny it to bare `unshare` -- see
`_unshare_prefix_candidates()`), needing no root/sudo/setcap.

What this gets you: each agent sees only its own writable workspace plus a
read-only view of the host's own interpreters/system libraries (bin, lib,
usr, ...) bind-mounted in -- it cannot read or write anything on the host
outside that. "Committing"/"restoring" is a plain directory copy
(`cp -a --reflink=auto`, so it's a true point-in-time copy, not a hardlink
clone that a later in-place write would silently corrupt) instead of an
image layer.

What this does NOT get you, matching the scope this backend was built for
(filesystem containment + independent per-agent installs, not defense
against adversarial code): no network namespace (the jailed process shares
the host's network stack), no PID namespace (background-process tracking
works against the REAL host /proc rather than a jailed one -- see below --
which means the jailed process can see -- though not touch, since real
permission checks still key off the unprivileged host uid the mapped
"root" resolves to -- every host process), no cgroup CPU/memory limits
(update_limits() is a no-op), and no GPU device scoping (a leased GPU's
CUDA_VISIBLE_DEVICES env var is still set, same as the container backend,
but nothing stops a process from seeing every /dev entry the host user can).

Background-process tracking (`_snapshot_pids()`/`get_live_pids()`/
`exec()`'s before/after PID diffing, all defined on the shared
`agsandbox_backend` base class) fundamentally cannot read a jailed `/proc`
here: mounting a FRESH procfs instance inside the jail (`mount -t proc`)
requires the caller's user namespace to own the target PID namespace, and
BIND-mounting the host's existing `/proc` is refused for the identical
reason (confirmed empirically via strace: `EINVAL` on `mount(2)` either
way) -- since this jail's `unshare` only creates a user+mount namespace,
never a PID namespace of its own, the process stays in the HOST's PID
namespace, which is owned by the *initial* user namespace, not this jail's
freshly created one. Adding `--pid` would fix the mount but would also
require a persistent per-jail "init" process for that namespace to survive
across calls (this backend deliberately has none -- see `_container_exec()`
and `_ensure_started()`), and would turn `_watched_pids` into jail-local
PIDs needing host-PID translation, undoing the very "PIDs are already
host-native, no translation needed" property `_own_host_pids()` relies on.
Instead, `_read_proc_table()` and `_exec_with_pid_tracking()` below read
`/proc` directly against the real host mount, from OUTSIDE the chroot
(reading an already-mounted procfs needs no new mount() call at all, so the
same-userns-must-own-the-pidns restriction never applies), while still
running the user's own command chrooted for filesystem containment.

`/dev` has the identical restriction (devtmpfs bind-mounting is refused for
the same reason procfs is), so it is populated with a plain `tmpfs` plus
individual per-file bind mounts of the handful of device files most
programs assume exist (`null`, `zero`, `random`, `urandom`, `tty`, `full`)
-- bind-mounting a single FILE doesn't cross the whole-filesystem-instance
boundary the restriction applies to, confirmed empirically. This mirrors
how rootless Docker/Podman populate their own `/dev`.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import threading
import uuid as _uuid
from pathlib import Path

from ..agconfig import agConfig
from .base import (
    _BGPIDS_MARKER,
    AgSandboxBackendFields,
    agsandbox_backend,
    run_with_unkillable_child_grace,
)

_chroot_available_cache: "bool | None" = None
_chroot_available_lock = threading.Lock()
# The invocation prefix chroot_available() found actually creates a user+
# mount namespace on this host -- plain ["unshare"] on most hosts, or
# ["rootlesskit", "--net=none", "unshare"] on ones where bare unshare is
# denied. Populated as a side effect of _probe_chroot_available() (see there
# for why rootlesskit is needed at all), under the same lock.
_chroot_unshare_prefix_cache: "list[str] | None" = None


def chroot_available() -> bool:
    """Return True if unprivileged user namespaces + chroot are usable on this
    host, cached for the process lifetime.

    Three checks, since any one alone can give a false positive:
    ``/proc/sys/kernel/unprivileged_userns_clone`` (when present -- it's
    Debian/Ubuntu-specific; distros that ship it enabled by default in the
    upstream kernel don't have the file at all) must not be explicitly
    disabled, and a live ``unshare --user --map-root-user`` -- either bare or
    wrapped in ``rootlesskit`` (see ``_probe_chroot_available``) -- must
    actually succeed, since AppArmor/seccomp policies can block unprivileged
    user namespaces even when the sysctl allows them.
    """
    global _chroot_available_cache
    if _chroot_available_cache is not None:
        return _chroot_available_cache
    with _chroot_available_lock:
        if _chroot_available_cache is None:
            _chroot_available_cache = _probe_chroot_available()
        return _chroot_available_cache


def _unshare_prefix_candidates() -> "list[list[str]]":
    """Candidate prefixes for creating an unprivileged user+mount namespace,
    tried in order.

    Plain ``unshare`` works on most hosts and is tried first so hosts where
    it already works don't gain a dependency on ``rootlesskit``. Some
    distros (Ubuntu with ``kernel.apparmor_restrict_unprivileged_userns=1``,
    the default since 24.04) deny ``CLONE_NEWUSER`` to unconfined binaries
    like a bare ``unshare`` call, but still allow it for ``rootlesskit`` --
    which ships its own AppArmor profile explicitly granting ``userns``, the
    same exemption rootless Docker/Podman rely on to keep working under that
    policy. Wrapping ``unshare`` in ``rootlesskit --net=none`` reuses that
    same exemption for the chroot backend, with no root/sudo/setcap needed.
    """
    candidates = [["unshare"]]
    if shutil.which("rootlesskit") is not None:
        candidates.append(["rootlesskit", "--net=none", "unshare"])
    return candidates


def _probe_chroot_available() -> bool:
    global _chroot_unshare_prefix_cache
    if shutil.which("unshare") is None or shutil.which("chroot") is None:
        return False
    try:
        sysctl_path = Path("/proc/sys/kernel/unprivileged_userns_clone")
        if sysctl_path.exists() and sysctl_path.read_text().strip() == "0":
            return False
    except OSError:
        # Ignore sysctl read errors and fall back to the live probe(s)
        # below, which are the authoritative capability check.
        pass
    for prefix in _unshare_prefix_candidates():
        try:
            proc = subprocess.run(
                [*prefix, "--user", "--map-root-user", "--mount", "true"],
                capture_output=True,
                timeout=AgSandboxBackendFields().inspect_timeout_s,
            )
        except Exception:
            continue
        if proc.returncode == 0:
            _chroot_unshare_prefix_cache = prefix
            return True
    return False


def _chroot_unshare_prefix() -> "list[str]":
    """The unshare invocation prefix _ChrootBackend should use to actually
    create its jail's namespace -- whichever candidate chroot_available()
    found working (bare ``unshare``, or ``rootlesskit``-wrapped). Calling
    chroot_available() first guarantees the cache is populated regardless of
    call order, since it's the only thing that runs the probe."""
    chroot_available()
    return _chroot_unshare_prefix_cache or ["unshare"]


_CHROOT_STATE_ROOT = Path(tempfile.gettempdir()) / "agency-chroot-sandboxes"
_CHROOT_JAILS_DIR = _CHROOT_STATE_ROOT / "jails"
_CHROOT_SNAPSHOTS_DIR = _CHROOT_STATE_ROOT / "snapshots"

# Host directories bind-mounted read-only into every jail so common
# interpreters/tools (python, bash, coreutils, shared libs) are usable
# without needing a separate root filesystem image.
_CHROOT_RO_BASE_DIRS = ("bin", "sbin", "lib", "lib32", "lib64", "usr", "etc")

# /dev cannot be bind-mounted as a whole directory (devtmpfs bind-mounting
# is refused from a nested unprivileged user namespace -- see module
# docstring), so the jail gets a plain tmpfs at /dev instead, populated with
# individual per-file bind mounts of just these device files -- bind-mounting
# a single FILE doesn't cross the whole-filesystem-instance boundary the
# devtmpfs restriction applies to. Read-write, unscoped (see module
# docstring), matching what most programs assume /dev/null, /dev/urandom
# etc. to be.
_CHROOT_DEV_FILES = ("null", "zero", "random", "urandom", "tty", "full")


def _sanitize_tag(tag: str) -> str:
    """Turn a checkpoint tag (e.g. "agency/lifecycle-name") into a single
    filesystem-safe path segment for use under _CHROOT_SNAPSHOTS_DIR."""
    return tag.replace("/", "__").replace(":", "__")


class _ChrootBackend(agsandbox_backend):
    """Manages a single chroot jail for one agent.

    All filesystem operations run inside a fresh
    ``unshare --user --map-root-user --mount`` + ``chroot`` invocation per
    call (see ``_chroot_unshare_prefix()`` for the ``rootlesskit`` fallback
    some hosts need to make that ``unshare`` call succeed at all) -- there is
    no long-lived daemon process to exec into (unlike docker/podman), so
    "starting" a sandbox is just making sure its jail
    directory exists on disk; the directory itself is the persistent state.
    """

    IMAGE_KIND = "chroot"

    # Host /proc is shared with every other process on the machine. Adopting
    # newly discovered non-baseline PIDs in get_live_pids() would let an
    # unrelated long-lived host process latch into _watched_pids and stall
    # skill wait / GPU release. Only BGPIDS from our own exec() may watch.
    _adopt_unwatched_live_pids = False

    def __init__(
        self,
        agname: str,
        *,
        name: str,
        checkpoint_image: "str | None",
        mounts: "dict[str, tuple[str, str, str]]",
        agconfig: "agConfig | None",
    ) -> None:
        self._agname = agname
        self._gpu_id: int | None = None
        self._gpu_virtual: bool = False
        self._gpu_acquire_fn = None
        self._gpu_release_fn = None
        self._cpu_acquired: float = 0.0
        self._memory_acquired_mb: int = 0
        self._watched_pids: dict[int, float] = {}
        self._baseline_pids: set[int] = set()
        self._daemon_pids: set[int] = set()
        self._started = False
        self._destroyed = False
        self._checkpoint_image: str | None = checkpoint_image
        self._agconfig = agconfig
        self._name = name
        self._mounts = mounts
        self._root = _CHROOT_JAILS_DIR / name
        self._workspace = self._root / "workspace"

    def change_config(self, agconfig: "agConfig | None") -> None:
        self._agconfig = agconfig

    def get_config_copy(self) -> "agConfig | None":
        return self._agconfig.clone() if self._agconfig is not None else None

    def _lifecycle_tag(self) -> str:
        return f"agency/lifecycle-{self._name}".lower()

    def _own_host_pids(self) -> "set[int]":
        """Chroot processes run directly on the host (no PID namespace) —
        _watched_pids is already host-native."""
        return set(self._watched_pids)

    def _gpu_is_clear(self) -> bool:
        """True once no watched sandbox processes remain alive on the host
        (chroot has no container-exit signal; this is the equivalent idle
        condition for release_gpu())."""
        for pid in list(self._watched_pids):
            if Path(f"/proc/{pid}").exists():
                return False
        return True

    def _ensure_started(self) -> None:
        """Create the jail's workspace directory on first use, restoring it
        from ``_checkpoint_image`` if one was given at construction time.

        Ground truth is the workspace directory's existence on disk, not
        ``self._started``. Tool calls with ``run_in_subprocess=True`` (the
        default) get a fresh cloudpickled copy of this backend per call, so
        a worker process's own ``_started`` is unreliable — exactly the
        problem ``_ContainerBackendBase._ensure_started()`` documents and
        solves by querying the docker daemon (``_container_running()``)
        instead of trusting its own ``_started``. There's no daemon here, so
        the workspace directory itself is the cross-process source of truth:
        materializing from a checkpoint every time some worker's copy
        happens to see ``_started=False`` would wipe out whatever a
        *different* worker already wrote to it.
        """
        if self._started:
            return
        if not self._workspace.is_dir():
            self._materialize_workspace(self._checkpoint_image)
        self._started = True  # set before _snapshot_pids() to prevent re-entry via _container_exec
        self._baseline_pids = self._snapshot_pids()

    def _materialize_workspace(self, tag: "str | None") -> None:
        """Replace the current workspace contents with a copy of *tag*'s
        snapshot, or an empty workspace if *tag* is None or has no snapshot."""
        self._root.mkdir(parents=True, exist_ok=True)  # cp -a needs the parent to exist
        if self._workspace.exists():
            shutil.rmtree(self._workspace, ignore_errors=True)
        snapshot_dir = _CHROOT_SNAPSHOTS_DIR / _sanitize_tag(tag) if tag else None
        if snapshot_dir is not None and snapshot_dir.is_dir():
            subprocess.run(
                ["cp", "-a", "--reflink=auto", str(snapshot_dir), str(self._workspace)],
                check=True,
            )
        else:
            self._workspace.mkdir(parents=True, exist_ok=True)

    def _setup_lines(self) -> "list[str]":
        """Shell lines that materialize the jail's directory structure and
        bind mounts, run unprivileged inside the `unshare --user
        --map-root-user --mount` namespace before anything else (chroot,
        /proc reads) happens. Shared by `_build_jail_script()` and
        `_exec_with_pid_tracking()` so the two entry points into a jail
        agree on what it looks like.

        Does not attempt to mount /proc inside the jail -- see module
        docstring for why that's refused by the kernel for both a fresh
        instance and a bind mount, and how PID tracking works around it
        instead (`_read_proc_table()`/`_exec_with_pid_tracking()`).
        """
        root = str(self._root)
        lines = [f"mkdir -p {shlex.quote(root)}"]
        for d in _CHROOT_RO_BASE_DIRS:
            host_path = f"/{d}"
            if not os.path.isdir(host_path):
                continue
            jail_path = f"{root}/{d}"
            lines.append(f"mkdir -p {shlex.quote(jail_path)}")
            lines.append(f"mount --bind {shlex.quote(host_path)} {shlex.quote(jail_path)}")
            lines.append(f"mount -o remount,bind,ro {shlex.quote(jail_path)} 2>/dev/null || true")
        # /dev: individual per-file bind mounts of the device files most
        # programs assume exist, directly on a plain mkdir -- NOT inside an
        # intermediate tmpfs. A freshly created mount (tmpfs included) gets
        # `nodev` forced on it in a nested unprivileged user namespace, and
        # that restriction is enforced per mountpoint-of-access: a device
        # special file bind-mounted onto a path under that nodev mount is
        # blocked from ever being opened as a device (confirmed empirically
        # -- world-writable /dev/null still EACCES'd through a tmpfs, but
        # opened fine through a bind directly onto a plain directory, which
        # isn't itself a fresh mount and so was never nodev-flagged).
        dev_jail_path = f"{root}/dev"
        lines.append(f"mkdir -p {shlex.quote(dev_jail_path)}")
        for name in _CHROOT_DEV_FILES:
            host_dev = f"/dev/{name}"
            if not os.path.exists(host_dev):
                continue
            jail_dev = f"{dev_jail_path}/{name}"
            lines.append(f"touch {shlex.quote(jail_dev)}")
            lines.append(f"mount --bind {shlex.quote(host_dev)} {shlex.quote(jail_dev)}")
        for host, container, mode in self._mounts.values():
            jail_path = f"{root}{container}"
            lines.append(f"mkdir -p {shlex.quote(jail_path)}")
            lines.append(f"mkdir -p {shlex.quote(host)}")
            lines.append(f"mount --bind {shlex.quote(host)} {shlex.quote(jail_path)}")
            if mode == "ro":
                lines.append(
                    f"mount -o remount,bind,ro {shlex.quote(jail_path)} 2>/dev/null || true"
                )
        lines.append(f"mkdir -p {shlex.quote(root + '/workspace')}")
        lines.append(f"mkdir -p {shlex.quote(root + '/proc')}")  # left empty, see module docstring
        lines.append(f"mkdir -p {shlex.quote(root + '/tmp')}")
        return lines

    def _build_jail_script(self, sh_cmd: str, *, workdir: str, shell: str) -> str:
        root = str(self._root)
        lines = self._setup_lines()
        # Setup (mkdir/mount) output must never reach the caller -- it isn't
        # part of the command's own stdout/stderr, and read_file()/write_file()
        # (inherited from agsandbox_backend) parse the captured output as raw
        # base64, which a stray "mount: permission denied" line would corrupt.
        setup = "{ " + "; ".join(lines) + "; } >/dev/null 2>&1"
        inner_cmd = f"cd {shlex.quote(workdir)} 2>/dev/null; {sh_cmd}"
        exec_line = f"exec chroot {shlex.quote(root)} {shell} -c {shlex.quote(inner_cmd)}"
        return f"{setup}\n{exec_line}"

    def _run_unshared(
        self, script: str, *, stdin: "bytes | None" = None, timeout: int
    ) -> "tuple[str, int]":
        """Run *script* as ``bash -c script`` inside a fresh ``unshare
        --user --map-root-user --mount`` namespace (see
        `_chroot_unshare_prefix()`), returning ``(output, rc)``.

        The lowest-level primitive shared by `_container_exec()` (which
        further wraps *script* in `_build_jail_script()`'s chroot) and
        `_exec_with_pid_tracking()` (which builds its own script that only
        chroots the user's command, not the surrounding /proc reads).
        """
        args = [
            *_chroot_unshare_prefix(),
            "--user",
            "--map-root-user",
            "--mount",
            "--",
            "bash",
            "-c",
            script,
        ]
        try:
            # run_with_unkillable_child_grace() bounds this call even against
            # a child stuck in uninterruptible kernel sleep (e.g. a wedged
            # mount/overlayfs syscall) that a plain subprocess.run(timeout=...)
            # would hang on forever -- see its docstring.
            proc = run_with_unkillable_child_grace(
                lambda: subprocess.run(args, input=stdin, capture_output=True, timeout=timeout),
                args=args,
                timeout=timeout,
                grace_s=self.unkillable_child_grace_s,
            )
            output = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
            return output, proc.returncode
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s", -1
        except Exception as e:
            return str(e), -1

    def _container_exec(
        self,
        sh_cmd: str,
        workdir: str = "/workspace",
        timeout: int = AgSandboxBackendFields.DEFAULT_EXEC_TIMEOUT_S,
        stdin: bytes | None = None,
        shell: str = "bash",
    ) -> tuple[str, int]:
        """Run a raw shell command inside the chroot jail."""
        self._ensure_started()
        script = self._build_jail_script(sh_cmd, workdir=workdir, shell=shell)
        return self._run_unshared(script, stdin=stdin, timeout=timeout)

    def _read_proc_table(self, script: str, timeout: int) -> "tuple[str, int]":
        """Run a pure /proc-reading *script* directly against the real host
        /proc, unchrooted -- see module docstring for why the jail has no
        procfs of its own to exec into, and why this needs no `unshare`/
        `chroot` at all: it's the same host PID namespace either way, no
        namespace boundary to cross, no new mount() call needed to read it.
        """
        self._ensure_started()
        try:
            proc = run_with_unkillable_child_grace(
                lambda: subprocess.run(["sh", "-c", script], capture_output=True, timeout=timeout),
                args=["sh", "-c", script],
                timeout=timeout,
                grace_s=self.unkillable_child_grace_s,
            )
            output = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
            return output, proc.returncode
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s", -1
        except Exception as e:
            return str(e), -1

    def _exec_with_pid_tracking(
        self, env_export: str, cmd: str, workdir: str, timeout: int
    ) -> "tuple[str, int]":
        """Run *cmd* chrooted for filesystem containment, while snapshotting
        /proc before and after it against the REAL host /proc (outside the
        jail) rather than a jailed one -- see module docstring for why the
        jail has no working procfs of its own.

        Structured as one shell script so the before-snapshot and the
        after-diff share state (``__AGENCY_BEFORE``) the way
        `agsandbox_backend._exec_with_pid_tracking()`'s single-exec-session
        version does for docker/podman -- but only the *middle* `chroot`
        sub-invocation (a plain forked subprocess, not `exec`-replacing this
        script) touches the jailed filesystem; the surrounding scans never
        leave the un-chrooted, real-/proc-visible outer shell. A command
        that backgrounds a process (``cmd &``) still detaches and survives
        past the `chroot` sub-invocation's own exit exactly as it does
        today, since the orphan is reparented on the shared host PID
        namespace regardless of which process was its immediate parent.
        """
        self._ensure_started()
        root = str(self._root)
        setup = "{ " + "; ".join(self._setup_lines()) + "; } >/dev/null 2>&1"
        inner_cmd = f"cd {shlex.quote(workdir)} 2>/dev/null; {env_export}{cmd}"
        script = (
            f"{setup}\n"
            # Snapshot every live host PID before the command runs. Filtering
            # on /proc/<N>/status avoids races with short-lived kernel threads.
            f"__AGENCY_BEFORE=$(for __d in /proc/[0-9]*; do"
            f" [ -f \"$__d/status\" ] && echo \"${{__d##*/}}\"; done | tr '\\n' ' ')\n"
            f"__AGENCY_SHELL=$$\n"
            # The chroot sub-invocation is a plain forked child (not `exec`
            # here), so this outer shell -- and its real /proc view -- is
            # still here to run the after-diff once it returns.
            f"__AGENCY_CHROOT_OUT=$(chroot {shlex.quote(root)} bash -c {shlex.quote(inner_cmd)} 2>&1)\n"
            f"__AGENCY_RC=$?\n"
            f"printf '%s' \"$__AGENCY_CHROOT_OUT\"\n"
            # Diff /proc after the command: any PID not in the before-snapshot
            # and not this shell itself was spawned by the command.
            f"__AGENCY_BGPIDS=''\n"
            f"for __d in /proc/[0-9]*; do\n"
            f'  [ -f "$__d/status" ] || continue\n'
            f"  __p=${{__d##*/}}\n"
            f'  case " $__AGENCY_BEFORE $__AGENCY_SHELL " in\n'
            f'    *" $__p "*) ;;\n'
            f'    *) __AGENCY_BGPIDS="$__AGENCY_BGPIDS $__p" ;;\n'
            f"  esac\n"
            f"done\n"
            f"printf '\\n{_BGPIDS_MARKER}%s' \"$__AGENCY_BGPIDS\"\n"
            f"exit $__AGENCY_RC"
        )
        return self._run_unshared(script, timeout=timeout)

    def update_limits(self, *, cpus: float | None = None, memory: str | None = None) -> None:
        """No-op -- chroot jails have no cgroup of their own to update."""
        return

    def commit(self, tag: str) -> bool:
        """Snapshot the current workspace to *tag*. Returns False if the
        jail was never started (nothing to snapshot).

        Checks the workspace directory's existence on disk rather than
        ``self._started`` -- this may be called from the orchestrating
        process on a sandbox whose actual workspace was written to entirely
        by worker-process tool calls (see _ensure_started()'s docstring),
        which never touch this process's own ``_started`` flag.
        """
        if not self._workspace.is_dir():
            return False
        snapshot_dir = _CHROOT_SNAPSHOTS_DIR / _sanitize_tag(tag)
        snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = snapshot_dir.with_name(snapshot_dir.name + f".tmp-{_uuid.uuid4().hex[:8]}")
        subprocess.run(
            ["cp", "-a", "--reflink=auto", str(self._workspace), str(tmp_dir)],
            check=True,
        )
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir, ignore_errors=True)
        tmp_dir.rename(snapshot_dir)
        return True

    def stop(self, *, commit: bool = False) -> None:
        """ "Stop" the jail: clear PID tracking and either snapshot the
        workspace (commit=True) or revert it to the last checkpoint
        (commit=False), mirroring the container backend's semantics --
        there is no running process to actually tear down.

        Checks the workspace directory's existence on disk rather than
        ``self._started`` -- called from the orchestrating process, which
        may never have run a single exec() of its own (every tool call ran
        in a worker process, each with its own cloudpickled copy of this
        backend). Trusting ``self._started`` here would silently skip
        committing real work just because *this* process's flag never
        flipped to True."""
        gpu_id_to_release = (
            self._gpu_id if (self._gpu_virtual and self._gpu_id is not None) else None
        )
        # Wait for watched processes to exit before clearing tracking / releasing.
        if gpu_id_to_release is not None and self._gpu_release_fn is not None:
            self._gpu_release_fn(gpu_id_to_release, is_clear=self._gpu_is_clear)
            self._gpu_id = None
        self._watched_pids = {}
        self._baseline_pids = set()
        if not self._started and not self._workspace.is_dir():
            return
        if commit:
            tag = self._lifecycle_tag()
            if self.commit(tag):
                self._checkpoint_image = tag
        else:
            self._materialize_workspace(self._checkpoint_image)
        self._started = False

    def restore(self, tag: str) -> None:
        """Restore the jail's workspace from a previously committed snapshot."""
        self._watched_pids = {}
        self._baseline_pids = set()
        self._checkpoint_image = tag
        # Materialize explicitly rather than resetting _started and calling
        # _ensure_started() -- that only restores when the workspace doesn't
        # exist yet (see its docstring), which would make an explicit
        # restore() onto an already-materialized workspace a silent no-op.
        self._materialize_workspace(tag)
        self._started = True
        self._baseline_pids = self._snapshot_pids()

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        if self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)
        if self._checkpoint_image:
            self.delete_image(self._checkpoint_image, force=True)
            self._checkpoint_image = None
        # Pretool snapshots created during this sandbox's lifetime (mirrors
        # _ContainerBackendBase.destroy()'s dangling-image cleanup).
        prefix = _sanitize_tag(f"agency/pretool-{self._name}-")
        if _CHROOT_SNAPSHOTS_DIR.is_dir():
            for entry in _CHROOT_SNAPSHOTS_DIR.iterdir():
                if entry.name.startswith(prefix):
                    shutil.rmtree(entry, ignore_errors=True)

    # ------------------------------------------------------------------
    # Static helpers — snapshot-directory-level operations, the chroot
    # equivalent of _ContainerBackendBase's image-tag helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def tag_image(source: str, dest: str) -> None:
        src_dir = _CHROOT_SNAPSHOTS_DIR / _sanitize_tag(source)
        dest_dir = _CHROOT_SNAPSHOTS_DIR / _sanitize_tag(dest)
        if not src_dir.is_dir():
            return
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        if dest_dir.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)
        subprocess.run(["cp", "-a", "--reflink=auto", str(src_dir), str(dest_dir)], check=True)

    @staticmethod
    def delete_image(tag: str, *, force: bool = False) -> None:
        snapshot_dir = _CHROOT_SNAPSHOTS_DIR / _sanitize_tag(tag)
        shutil.rmtree(snapshot_dir, ignore_errors=True)

    @staticmethod
    def export_image(tag: str, timeout: int) -> bytes:
        """Tar up *tag*'s snapshot directory, storing it under the sanitized
        tag name so import_image() can restore the same tag."""
        snapshot_dir = _CHROOT_SNAPSHOTS_DIR / _sanitize_tag(tag)
        if not snapshot_dir.is_dir():
            raise FileNotFoundError(f"No chroot snapshot for tag {tag!r}")
        buf = tempfile.NamedTemporaryFile(delete=False)
        try:
            with tarfile.open(buf.name, "w:gz") as tar:
                tar.add(snapshot_dir, arcname=_sanitize_tag(tag))
            return Path(buf.name).read_bytes()
        finally:
            buf.close()
            os.unlink(buf.name)

    @staticmethod
    def import_image(image_bytes: bytes, timeout: int) -> None:
        buf = tempfile.NamedTemporaryFile(delete=False)
        try:
            buf.write(image_bytes)
            buf.close()
            _CHROOT_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            with tarfile.open(buf.name, "r:gz") as tar:
                tar.extractall(_CHROOT_SNAPSHOTS_DIR, filter="data")
        finally:
            os.unlink(buf.name)
