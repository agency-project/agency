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
(update_limits() is a no-op), and only best-effort GPU device scoping: a
leased GPU's CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES env vars are set
exactly like the container backend, and `_chroot_gpu_dev_paths(gpu_id)`
bind-mounts only that GPU's own compute node (plus the shared control
devices every GPU needs regardless of index) into the jail -- mirroring
what docker's `--gpus device=N`/podman's CDI actually expose, rather than
every GPU on the host. There is still no per-GPU cgroup device filter here
though, so this restricts what the *default* jail setup exposes, not what a
process inside could reach if it somehow discovered and opened another
leased jail's device node directly by path.

NVIDIA GPUs specifically may not be usable inside the jail AT ALL regardless
of the above: confirmed empirically (H100, recent driver) that `nvidia-smi`
fails with "GPU access blocked by the operating system" the moment the
calling process has `chroot`'d -- even with every relevant device node
correctly bind-mounted in and reachable, and even though the exact same
process calling nvidia-smi from a bare `unshare --user --mount` (same
namespaces, no chroot) works fine. This is the NVIDIA driver's own
chroot-detection refusing access, not a mount/permission/namespace issue
this backend's code can route around -- nothing short of not chrooting at
all (defeating the entire point of this backend) fixes it. Untested against
AMD/ROCm, which may not carry the same restriction.

Background-process tracking is PGID-only here -- there is no baseline-diff
or before/after BGPIDS mechanism the way docker/podman (and an earlier
version of this backend) use. Reading a jailed `/proc` at all has its own
problem first: mounting a FRESH procfs instance inside the jail
(`mount -t proc`) requires the caller's user namespace to own the target
PID namespace, and BIND-mounting the host's existing `/proc` is refused for
the identical reason (confirmed empirically via strace: `EINVAL` on
`mount(2)` either way) -- since this jail's `unshare` only creates a
user+mount namespace, never a PID namespace of its own, the process stays
in the HOST's PID namespace, which is owned by the *initial* user
namespace, not this jail's freshly created one. Instead, `_read_proc_table()`
reads `/proc` directly against the real host mount, from OUTSIDE the
chroot (reading an already-mounted procfs needs no new mount() call at all,
so the same-userns-must-own-the-pidns restriction never applies), while the
user's own command still runs chrooted for filesystem containment.

That fixes VISIBILITY (background jobs are detected at all, where before
this backend's tracking was permanently empty) but reading the WHOLE host
/proc raises a different problem: distinguishing this sandbox's own work
from every other process on a busy shared host. A per-PID ownership check
via `/proc/<pid>/root` (does this PID's root match our jail's?) was tried
and rejected: reading another process's `/proc/<pid>/root` requires being
its ptrace-eligible ancestor (Yama `ptrace_scope=1`, the Ubuntu/Debian
default), and a backgrounded job gets reparented away from any ancestor
relationship to whichever later, separate call tries to check on it --
confirmed empirically. A real per-process ownership check would need a
persistent per-jail process that never leaves its own PID namespace and
relays later commands to it over IPC (roughly what Podman's `conmon` does)
-- also confirmed empirically NOT achievable as a lighter-weight
`nsenter`-from-a-separate-process trick: joining an existing unprivileged
user namespace from outside it is refused by the kernel (`nsenter --user`
from an unrelated process fails with EPERM even for the namespace's own
creator) unless that later process is a descendant of the one that created
the namespace, which -- given this backend deliberately runs each call as
its own fresh `unshare` invocation, no persistent daemon -- it never is.

A "one-time startup baseline, anything not in it is ours" scan (the
approach docker/podman use safely, scoped to their own isolated PID
namespace) was tried here too and abandoned: on a host with any background
churn, that scan could see literally everything as "not yet clear," making
`_has_pending_background_work()` never resolve to "done" quickly in the
common case -- confirmed empirically on at least one real dev host. It also
could never verify a candidate pid's ownership well enough to justify
`_kill_all_sandbox_processes()` SIGKILLing it.

Fixed via process-group (PGID) matching, and PGID matching ALONE: each
`exec()` call's underlying `unshare` invocation is started as its own new
process-group leader (see `_run_unshared()`), and that group id is recorded
into `_invocation_pgids`. `_live_pgid_matched_pids()` is the one shared
scan-and-match this all runs through -- it backs `get_live_pids()`,
`_has_pending_background_work()`, and (via `_kill_all_sandbox_processes()`)
actual teardown, all from the exact same /proc read. Reading another
process's process-group id (unlike its `/proc/<pid>/root`) needs no ptrace
permission at all, and -- unlike the abandoned baseline scan -- PGID
membership is a real, kernel-tracked relationship established once at spawn
time, immune to however much unrelated churn a busy host generates, and
confirmed empirically to survive reparenting (a child that outlives its
exited parent keeps the same pgid) as well as to catch a child spawned
*after* the `exec()` call that backgrounded its parent already returned --
something a before/after diff limited to that one call's own window could
never see. It's also a relationship `_kill_all_sandbox_processes()` can
safely act on destructively via `os.killpg()` -- unlike the abandoned
baseline scan, which could never verify a candidate pid's ownership well
enough to justify SIGKILLing it.

One real cost of dropping per-PID diff tracking: `get_live_pids()` can no
longer report how long an individual PID has been running (there is no
longer a per-PID capture timestamp) -- `pid_status_summary()` here lists
matched PIDs without an elapsed-time figure, unlike docker/podman's version.
Given the diff mechanism's own reporting was already only approximately
attributable to a specific PID's true spawn time for any child discovered
late, this is a small, honest precision loss in exchange for a mechanism
that no longer needs to exist in two different forms for two different
purposes (fine-grained-but-blind-to-later-children vs. coarse-but-complete).

The accepted trade-off: a process that calls `setsid()`/`setpgid()` to
detach into its own new group -- a common, legitimate daemonizing idiom
(`nohup`, `setsid`, `disown`, many "daemonize" library patterns) --
escapes `_invocation_pgids` tracking entirely. Confirmed empirically:
`setsid sleep & echo $!` inside a tracked invocation produces a child with
its OWN distinct pgid, not the invocation's. Chroot sandboxes that spawn
genuinely daemonizing background work are simply not tracked past that
point -- accepted as out of scope for what this backend is for (filesystem
containment + independent per-agent installs, not adversarial isolation),
same as the broader "no persistent daemon" limitation above.

KNOWN, SEPARATE LIMITATION -- `_invocation_pgids` is a plain in-memory
attribute, so it does NOT survive across the cloudpickle/worker-process
boundary tool calls with `run_in_subprocess=True` (the default) create:
a worker's own copy of this backend records a real PGID in ITS OWN
`_invocation_pgids`, which is discarded when that worker process exits.
`agSandbox.wait_for_processes()` is always called from the orchestrating
process (see agskill.py), whose OWN copy never independently ran `exec()`
-- confirmed empirically: a worker-spawned background job is invisible to
`_has_pending_background_work()` when checked from the orchestrator's
copy, exactly the scenario `_ensure_started()`'s own docstring already
documents needing disk-based (not in-memory) ground truth for. Not yet
fixed -- would need persisting recorded PGIDs to disk, mirroring how
`_ensure_started()` treats `self._workspace.is_dir()` as ground truth,
rather than anything in this attribute.

GPU release itself does NOT need any of this: `stop()`/`destroy()` always
run their own kill step (`_kill_all_sandbox_processes()`) synchronously
BEFORE calling `release_gpu()`, and that ordering -- not a separate
"is it actually clear yet" check -- is the confirmation the sandbox has
exited (see `agresources.agResourcePool.release_gpu()`'s docstring). An
earlier version of this threaded an `is_clear` predicate through release_gpu()
for exactly that re-check; removed as redundant with the kill step that
already ran, for both this backend and the container one.

`/dev` has the identical mount restriction (devtmpfs bind-mounting is
refused for the same reason procfs is), so it is populated with individual
per-file bind mounts of the handful of device files most programs assume
exist (`null`, `zero`, `random`, `urandom`, `tty`, `full`), directly onto a
plain `mkdir`ed directory -- NOT an intermediate `tmpfs`. A tmpfs mount
inside a nested unprivileged user namespace gets `nodev` forced on it,
which then blocks opening ANY device special file bind-mounted underneath
it (confirmed empirically: a world-writable `/dev/null` bind-mounted into
a tmpfs still EACCES'd on open, but opened fine bind-mounted directly onto
a plain directory, which was never itself a fresh mount and so was never
nodev-flagged). Bind-mounting a single FILE doesn't cross the
whole-filesystem-instance boundary the devtmpfs restriction applies to.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import threading
import uuid as _uuid
from pathlib import Path

from ..agconfig import agConfig
from ..agresources import amd_render_node_paths_by_pci_bus
from .base import (
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
        except Exception as _e:
            # This candidate prefix isn't usable (missing binary, exec
            # failure, timeout) -- try the next one rather than treating a
            # single candidate's failure as fatal to capability detection.
            # Runs at most once per candidate, cached after the first
            # success, so this print is not a per-call cost.
            print(f"[agsandbox_backend] chroot capability probe {prefix} failed: {_e}")
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
# docstring), so the jail gets individual per-file bind mounts of just these
# device files instead, directly on a plain mkdir'd directory (NOT a tmpfs --
# see module docstring for why that would break device access). Read-write,
# unscoped (see module docstring), matching what most programs assume
# /dev/null, /dev/urandom etc. to be.
_CHROOT_DEV_FILES = ("null", "zero", "random", "urandom", "tty", "full")

_gpu_dev_paths_cache: "list[str] | None" = None
_gpu_dev_paths_lock = threading.Lock()

# Matches only a per-GPU indexed NVIDIA compute node ("nvidia0", "nvidia7"),
# never a control/shared device -- "nvidiactl", "nvidia-uvm", "nvidia-fs0",
# "nvidia-nvswitch0" etc. all have non-digit characters right after "nvidia"
# and so never match. Confirmed against a real 8-GPU host's /dev listing.
_NVIDIA_INDEXED_DEV_RE = re.compile(r"^nvidia(\d+)$")


def _chroot_gpu_dev_paths(gpu_id: "int | None") -> "list[str]":
    """Return absolute host device paths to bind-mount into a jail for the
    specific *gpu_id* leased to it (or [] if none is leased), cached for
    the process lifetime -- mirrors container.py's _gpu_flags()' nvidia-vs-amd
    detection (nvidia-smi on PATH => NVIDIA, else AMD/ROCm).

    Scoped to just the leased GPU's own compute node plus the shared control
    devices every GPU needs regardless of index (nvidiactl/nvidia-uvm for
    NVIDIA, /dev/kfd for AMD) -- mirrors what docker's `--gpus device=N`/
    podman's CDI actually bind into a container, rather than every compute
    node on the host. See this module's docstring for the residual caveat:
    there is still no per-GPU cgroup device filter here, so this only
    prevents the *default* jail setup from exposing every GPU -- a process
    that discovers and opens another leased jail's device node directly
    (if it somehow knows the path) is not blocked by anything at the kernel
    level the way a real cgroup device filter would.

    Without this, a GPU *lease* (CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES,
    handled identically to the container backend -- see module docstring)
    is scheduled successfully but the device itself is unreachable inside
    the jail, same root cause as /dev/null above: the old whole-directory
    /dev bind silently failed for every device file, GPU included, so this
    was ALREADY true before the /dev fix above, not something it changed.

    Returns [] on a GPU-less host, and [] when gpu_id is None (no GPU
    leased) -- CPU-only hosts and un-leased jails pay nothing extra here.
    """
    if gpu_id is None:
        return []
    all_paths = _all_chroot_gpu_dev_paths()
    if not all_paths:
        return []
    if shutil.which("nvidia-smi"):
        control = [p for p in all_paths if not _NVIDIA_INDEXED_DEV_RE.match(os.path.basename(p))]
        indexed = sorted(
            (p for p in all_paths if _NVIDIA_INDEXED_DEV_RE.match(os.path.basename(p))),
            key=lambda p: int(_NVIDIA_INDEXED_DEV_RE.match(os.path.basename(p)).group(1)),
        )
        return control if gpu_id >= len(indexed) else control + [indexed[gpu_id]]
    # AMD/ROCm: /dev/kfd is the one shared control device; each GPU's own
    # compute node is /dev/dri/renderD<128+N>, matched to gpu_id by PCI bus
    # (agresources.amd_render_node_paths_by_pci_bus()) rather than assumed
    # host enumeration order -- confirmed on real 8x MI350X hardware that
    # naive sorted order does NOT correspond to GPU index (each GPU exposes
    # itself plus 7 XCD/compute-partition sibling render nodes), falling
    # back to naive order only if the PCI-bus mapping can't be built.
    control = [p for p in all_paths if os.path.basename(p) == "kfd"]
    naive_render_nodes = sorted(p for p in all_paths if os.path.basename(p).startswith("render"))
    render_nodes = (
        amd_render_node_paths_by_pci_bus(naive_render_nodes) if naive_render_nodes else None
    )
    if render_nodes is None:
        render_nodes = naive_render_nodes
    return control if gpu_id >= len(render_nodes) else control + [render_nodes[gpu_id]]


def _all_chroot_gpu_dev_paths() -> "list[str]":
    """Return every GPU-related device path present on the host (shared
    control devices plus every per-GPU compute node), cached for the
    process lifetime. Not scoped to any one lease -- _chroot_gpu_dev_paths()
    is what callers should use; this is only the shared detection step it
    further restricts."""
    global _gpu_dev_paths_cache
    if _gpu_dev_paths_cache is not None:
        return _gpu_dev_paths_cache
    with _gpu_dev_paths_lock:
        if _gpu_dev_paths_cache is None:
            _gpu_dev_paths_cache = _detect_chroot_gpu_dev_paths()
        return _gpu_dev_paths_cache


def _detect_chroot_gpu_dev_paths() -> "list[str]":
    from ..agresources import detect_gpus

    if not detect_gpus():
        return []
    if shutil.which("nvidia-smi"):
        return sorted(str(p) for p in Path("/dev").glob("nvidia*") if p.is_char_device())
    paths = []
    if os.path.exists("/dev/kfd"):
        paths.append("/dev/kfd")
    dri = Path("/dev/dri")
    if dri.is_dir():
        paths.extend(str(p) for p in sorted(dri.iterdir()) if p.is_char_device())
    return paths


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
        # Never populated -- chroot tracks background work purely via
        # _invocation_pgids (see module docstring). Kept only because
        # agsandbox_backend.exec()'s shared BGPIDS-marker handling and
        # release_daemon() reference self._watched_pids unconditionally.
        self._watched_pids: dict[int, float] = {}
        self._daemon_pids: set[int] = set()
        # Process-group ids captured from _exec_with_pid_tracking()'s own
        # invocations (see _run_unshared()) -- backs _has_pending_background_work()
        # and _kill_all_sandbox_processes(). See this module's docstring for
        # the full rationale, the accepted setsid/daemonizing blind spot,
        # and the separate cross-process (worker) limitation.
        self._invocation_pgids: set[int] = set()
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
        """Chroot processes run directly on the host (no PID namespace) --
        same underlying scan as get_live_pids()."""
        return self.get_live_pids()

    def get_live_pids(self) -> "set[int]":
        """Every PID whose process group is one of this jail's tracked
        `_invocation_pgids` -- the sole tracking mechanism for this backend
        (see module docstring). Same scan-and-match as
        `_has_pending_background_work()`; both call `_live_pgid_matched_pids()`
        so they can never disagree with each other the way a separate
        diff-based `get_live_pids()` and PGID-based `_has_pending_background_work()`
        could."""
        return self._live_pgid_matched_pids()

    def pid_status_summary(self) -> str:
        """Same purpose as `agsandbox_backend.pid_status_summary()`, but
        without a per-PID elapsed-time figure -- there is no longer a
        per-PID capture timestamp to report one from (see module docstring
        for why dropping per-PID diff tracking cost that precision). Also
        never needs the base version's "not individually trackable" fallback:
        get_live_pids() and _has_pending_background_work() are both backed
        by the exact same scan here, so they can never disagree about
        whether anything is running.
        """
        live = self.get_live_pids()
        if not live:
            return "no background processes running"
        return ", ".join(f"PID {pid}" for pid in sorted(live))

    def _has_pending_background_work(self) -> bool:
        """See `agsandbox_backend._has_pending_background_work()`'s
        docstring for why the default (just checking `_watched_pids`) isn't
        safe here -- gates whether `agSandbox.wait_for_processes()` bothers
        waiting at all before a skill's result is delivered."""
        return bool(self._live_pgid_matched_pids())

    def _live_pgid_matched_pids(self) -> "set[int]":
        """Every PID sharing one of this jail's tracked `_invocation_pgids`
        (see `_run_unshared()`/`_exec_with_pid_tracking()`) that's still
        alive, excluding explicitly-released daemons and their descendants
        (see `release_daemon()`) -- the one scan backing `get_live_pids()`,
        `_has_pending_background_work()`, and (via
        `_kill_all_sandbox_processes()`) actual teardown.

        Replaces the host-wide "anything not in the startup baseline"
        scan this module used before: PGID membership is a real,
        kernel-tracked relationship established once at spawn time and
        never renumbered, so it can't be confused with unrelated churn
        elsewhere on a busy shared host regardless of how much of it there
        is -- confirmed empirically as a real (not just theoretical)
        problem with the baseline-based scan, which could see a busy host
        as perpetually non-clear. It also matters for
        `_kill_all_sandbox_processes()`: unlike the old baseline scan
        (which could never be safely used to decide what to SIGKILL, since
        it couldn't verify a candidate pid's ownership -- see that
        method's docstring history), a PGID match IS a verified,
        kernel-checked relationship, safe to act on destructively via
        `os.killpg()`.

        The accepted trade-off (see this module's docstring for the full
        rationale): a process that calls `setsid()`/`setpgid()` to detach
        into its own new group -- a common, legitimate daemonizing idiom
        (`nohup`, `setsid`, `disown`, many "daemonize" library patterns) --
        escapes this check entirely. Chosen deliberately over reverting to
        the baseline scan, which traded that same blind spot for a *worse*
        one: systematically treating a busy host as never-clear, which
        would defeat the whole point of `_has_pending_background_work()`
        resolving quickly in the common case.

        Also prunes `_invocation_pgids` of any group confirmed (by this
        same scan) to have no surviving members -- bounding it across a
        long sandbox lifetime. Safe to prune here: removing an entry we've
        just confirmed is dead cannot mistakenly adopt anything new, it
        only forgets something that's genuinely, permanently gone (a
        process group ceases to exist once its last member exits).
        """
        if not self._invocation_pgids:
            return set()
        # Pure shell builtins only (read/parameter-expansion/case) -- NOT
        # awk/cat per entry. Forking 2-3 subprocesses per /proc entry across
        # thousands of entries on a busy host was confirmed empirically to
        # take 8+ seconds, which defeats the purpose of a "quick safety
        # check": a short-lived process could exit before the scan even
        # reaches it. Reading /proc/<pid>/status line-by-line via the `read`
        # builtin and /proc/<pid>/stat via a single `read` (no subprocess at
        # all) is dramatically faster since nothing forks per entry.
        script = (
            "for __d in /proc/[0-9]*; do\n"
            '  [ -f "$__d/status" ] || continue\n'
            "  __p=${__d##*/}\n"
            "  __ppid=''\n"
            "  __st=''\n"
            "  while IFS= read -r __line; do\n"
            '    case "$__line" in\n'
            "      PPid:*) set -- ${__line#PPid:}; __ppid=$1 ;;\n"
            "      State:*) set -- ${__line#State:}; __st=$1 ;;\n"
            "    esac\n"
            '  done < "$__d/status"\n'
            '  IFS= read -r __statline < "$__d/stat" 2>/dev/null\n'
            "  __rest=${__statline##*\\)}\n"
            "  set -- $__rest\n"
            "  __pgid=$3\n"
            '  echo "$__p $__ppid $__pgid $__st"\n'
            "done"
        )
        output, _ = self._read_proc_table(script, timeout=self.inspect_timeout_s)

        proc_info: "dict[int, tuple[int, int, str]]" = {}
        for line in output.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                pid, ppid, pgid, state = int(parts[0]), int(parts[1]), int(parts[2]), parts[3]
            except ValueError:
                continue
            proc_info[pid] = (ppid, pgid, state)

        # Snapshot-local daemon-descendant propagation (never written back to
        # self._daemon_pids) so an explicitly-released daemon's later
        # children are also excluded.
        daemon_pids = set(self._daemon_pids)
        changed = True
        while changed:
            changed = False
            for pid, (ppid, _, _) in proc_info.items():
                if pid not in daemon_pids and ppid in daemon_pids:
                    daemon_pids.add(pid)
                    changed = True

        alive_pids: "set[int]" = set()
        alive_pgids: "set[int]" = set()
        for pid, (_, pgid, state) in proc_info.items():
            if pid in daemon_pids or state == "Z":
                continue
            if pgid in self._invocation_pgids:
                alive_pids.add(pid)
                alive_pgids.add(pgid)

        self._invocation_pgids = alive_pgids
        return alive_pids

    def _kill_all_sandbox_processes(self) -> None:
        """SIGKILL every process group in `_invocation_pgids` (see
        `_live_pgid_matched_pids()`'s docstring for why this is safe to act
        on destructively). Best-effort: called from `stop()`/`destroy()`/
        `restore()` before tearing down or overwriting the workspace.

        `os.killpg()` on a tracked invocation group is a verified, kernel
        -checked operation -- not a guess the way the old baseline scan's
        candidates were -- so it's safe to use here: a delayed child never
        individually captured by a before/after diff still shares its
        invocation's PGID, and so still gets killed here. The accepted
        residual gap: a `setsid`-detached descendant (see
        `_live_pgid_matched_pids()`'s docstring) has its own PGID and is
        not reachable through this mechanism.

        There is a separate, low-probability residual risk inherent to any
        PID-based system: if the invocation's original group leader has
        already exited and the kernel has since recycled that exact PID
        number for an unrelated new process that also happens to become a
        group leader, `killpg()` would signal that unrelated group instead.
        Considered acceptable given how narrow the window is (recorded
        PGIDs are only ever acted on within this same sandbox's lifetime,
        typically seconds to minutes) and that avoiding it entirely would
        require PID-file-descriptor-based process handles, a materially
        larger change.

        Unlike `_ContainerBackendBase` (whose `kill {pids}` must go through
        `docker/podman exec`, since its processes live in the container's
        own namespace, then gets a hard guarantee for free from container
        removal itself regardless of what was tracked), chroot's tracked
        process groups are already host-native, so a plain `os.killpg()`
        reaches them directly -- no shell/unshare round-trip needed. There
        is also no container-removal-style implicit kill here: chroot has
        no cgroup/namespace boundary whose teardown could guarantee
        termination, so this explicit step is the only thing that stops a
        jail's background work from outliving `destroy()` (which would
        otherwise delete the jail root out from under a still-running
        process) or surviving `stop()`/`restore()` rewriting the workspace
        underneath it.
        """
        import signal

        for pgid in list(self._invocation_pgids):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as _e:
                print(
                    f"[agsandbox_backend] WARNING: failed to kill process group {pgid} in {self._name}: {_e}"
                )

    def _ensure_started(self) -> None:
        """Create the jail's workspace directory on first use, restoring it
        from ``_checkpoint_image`` if one was given at construction time.

        Ground truth is always the workspace directory's existence on disk
        -- there is no ``self._started`` cache. Tool calls with
        ``run_in_subprocess=True`` (the default) get a fresh cloudpickled
        copy of this backend per call, so a per-process flag would be
        unreliable — exactly the problem ``_ContainerBackendBase``'s
        equivalent method documents and solves by querying the docker daemon
        (``_container_running()``) rather than trusting a flag. There's no
        daemon here, so the workspace directory itself is the cross-process
        source of truth: materializing from a checkpoint every time some
        worker's copy finds the workspace missing (when it's actually
        present, just not yet observed by this copy) would wipe out whatever
        a *different* worker already wrote to it -- but that can only happen
        if we skip the ``is_dir()`` check, which we never do.

        No baseline PID snapshot is captured here: unlike the container
        backend (whose shared ``get_live_pids()`` needs one to exclude its
        own init/runtime-helper processes), this backend's tracking is
        PGID-only (see module docstring) and needs no startup snapshot at
        all -- there is nothing else for this method to do once the
        workspace exists.
        """
        if not self._workspace.is_dir():
            self._materialize_workspace(self._checkpoint_image)

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
        dev_host_paths = [f"/dev/{name}" for name in _CHROOT_DEV_FILES] + _chroot_gpu_dev_paths(
            self._gpu_id
        )
        for host_dev in dev_host_paths:
            if not os.path.exists(host_dev):
                continue
            rel = os.path.relpath(host_dev, "/dev")  # e.g. "nvidia0" or "dri/card0"
            jail_dev = f"{dev_jail_path}/{rel}"
            jail_dev_dir = os.path.dirname(jail_dev)
            if jail_dev_dir != dev_jail_path:
                lines.append(f"mkdir -p {shlex.quote(jail_dev_dir)}")
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
    ) -> "tuple[str, int, int | None]":
        """Run *script* as ``bash -c script`` inside a fresh ``unshare
        --user --map-root-user --mount`` namespace (see
        `_chroot_unshare_prefix()`), returning ``(output, rc, pgid)``.

        The lowest-level primitive shared by `_container_exec()` (which
        further wraps *script* in `_build_jail_script()`'s chroot, and
        discards *pgid*) and `_exec_with_pid_tracking()` (which builds its
        own script that only chroots the user's command, not the
        surrounding /proc reads, and records *pgid* into
        `_invocation_pgids`).

        *pgid* is the top-level `unshare` process's own process group id:
        it's started via `start_new_session=True` (`setsid()` before exec),
        making it a session AND process-group leader of a brand new group
        -- so its pgid equals its own pid, known synchronously at spawn
        time, before waiting for it to finish. Anything it or its
        descendants spawn inherits this same pgid by default (confirmed
        empirically, including after the original leader has exited and a
        descendant has been reparented to host init), UNLESS that
        descendant explicitly calls `setsid()`/`setpgid()` itself -- see
        `_live_pgid_matched_pids()`'s docstring for why that
        (deliberately) escapes the tracking built on top of this.
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
        pgid_holder: "list[int]" = []

        def _call() -> "subprocess.CompletedProcess[bytes]":
            proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE if stdin is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            pgid_holder.append(proc.pid)
            try:
                out, err = proc.communicate(input=stdin, timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
                raise
            return subprocess.CompletedProcess(args, proc.returncode, out, err)

        try:
            # run_with_unkillable_child_grace() bounds this call even against
            # a child stuck in uninterruptible kernel sleep (e.g. a wedged
            # mount/overlayfs syscall) that a plain subprocess.run(timeout=...)
            # would hang on forever -- see its docstring.
            proc = run_with_unkillable_child_grace(
                _call,
                args=args,
                timeout=timeout,
                grace_s=self.unkillable_child_grace_s,
            )
            output = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
            return output, proc.returncode, (pgid_holder[0] if pgid_holder else None)
        except subprocess.TimeoutExpired:
            return (
                f"Command timed out after {timeout}s",
                -1,
                (pgid_holder[0] if pgid_holder else None),
            )
        except Exception as e:
            return str(e), -1, (pgid_holder[0] if pgid_holder else None)

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
        output, rc, _pgid = self._run_unshared(script, stdin=stdin, timeout=timeout)
        return output, rc

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
        """Run *cmd* chrooted for filesystem containment. No before/after
        /proc diff here -- unlike `agsandbox_backend._exec_with_pid_tracking()`'s
        docker/podman version, this backend tracks background work purely
        via the invocation's own process group (see module docstring), so
        there is nothing for a diff to capture that PGID matching doesn't
        already cover, including a child spawned well after this call
        already returned (which a before/after diff limited to this one
        call's own window could never see in the first place).

        The chroot sub-invocation's own stdout/stderr is captured via a
        temp FILE, not ``$(...)`` command substitution -- confirmed
        empirically that command substitution reads a real OS pipe, whose
        EOF (and therefore ``$(...)``'s own return) is held open until
        EVERY process that inherited the write end closes it, including a
        backgrounded descendant of the chrooted command. That made this
        method block for the backgrounded job's ENTIRE runtime before ever
        returning -- exactly the case this method exists to avoid waiting
        on -- silently defeating background-process tracking for chroot in
        a way distinct from (and found only after fixing) the
        /proc-visibility bug this module's docstring describes. A file's
        readers don't block on other processes' open write handles the way
        a pipe's do, so `cat`-ing it after the foreground chroot invocation
        returns picks up whatever was written by then and no more, which is
        exactly the "don't wait on background work" behavior needed here.

        The whole script runs via `_run_unshared()`, whose top-level
        `unshare` process is its own new process group leader (see that
        method's docstring) -- its pgid is recorded into
        `_invocation_pgids`, letting `_live_pgid_matched_pids()` find this
        call's own descendants later, at any point, not just within this
        one call's own before/after window. A command that backgrounds a
        process (``cmd &``) still detaches and survives past the `chroot`
        sub-invocation's own exit exactly as it does today, since the
        orphan is reparented on the shared host PID namespace regardless of
        which process was its immediate parent -- but it keeps the same
        pgid either way, which is all this mechanism needs.
        """
        self._ensure_started()
        root = str(self._root)
        setup = "{ " + "; ".join(self._setup_lines()) + "; } >/dev/null 2>&1"
        inner_cmd = f"cd {shlex.quote(workdir)} 2>/dev/null; {env_export}{cmd}"
        script = (
            f"{setup}\n"
            f"__AGENCY_OUT_FILE=$(mktemp)\n"
            # Output goes to a file, not a pipe -- see this method's
            # docstring for why.
            f"chroot {shlex.quote(root)} bash -c {shlex.quote(inner_cmd)} "
            f'> "$__AGENCY_OUT_FILE" 2>&1\n'
            f"__AGENCY_RC=$?\n"
            f'cat "$__AGENCY_OUT_FILE"\n'
            f'rm -f "$__AGENCY_OUT_FILE"\n'
            f"exit $__AGENCY_RC"
        )
        output, rc, pgid = self._run_unshared(script, timeout=timeout)
        if pgid is not None:
            self._invocation_pgids.add(pgid)
        return output, rc

    def update_limits(self, *, cpus: float | None = None, memory: str | None = None) -> None:
        """No-op -- chroot jails have no cgroup of their own to update."""
        return

    def commit(self, tag: "str | None" = None) -> bool:
        """Snapshot the current workspace to *tag* (default: this jail's own
        lifecycle tag) -- the chroot equivalent of a container `commit`.
        Leaves the live workspace untouched (unlike the old commit-then-
        discard stop()): the jail keeps running from it exactly as before,
        no different from a container backend's commit() not removing the
        container. Returns False if the jail was never started (nothing to
        snapshot).

        Checks the workspace directory's existence on disk directly -- this
        may be called from the orchestrating process on a sandbox whose
        actual workspace was written to entirely by worker-process tool
        calls (see _ensure_started()'s docstring).
        """
        if not self._workspace.is_dir():
            return False
        tag = tag if tag is not None else self._lifecycle_tag()
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
        self._checkpoint_image = tag
        return True

    def stop(self) -> None:
        """Hibernate the jail: kill tracked processes and release the GPU --
        the chroot equivalent of releasing a container's resources. There
        is no keyring/runtime slot to release here (chroot processes run
        directly on the host, with no docker/podman container involved at
        all -- see the module docstring), so this is simpler than the
        container backend's stop(): unlike that one, releasing the GPU
        here is safe, since chroot has no persistent container object with
        GPU device flags baked in at creation -- visibility is granted
        per-exec via env vars, so a later resume can safely be handed a
        different physical GPU.

        Deliberately leaves self._workspace untouched -- hibernating must
        preserve state across calls the same way the container backend's
        stop() does; use rm_container() to discard it, and commit() to
        checkpoint it.
        """
        gpu_id_to_release = (
            self._gpu_id if (self._gpu_virtual and self._gpu_id is not None) else None
        )
        # Kill first: wait_for_processes() already gave background work its
        # fair chance to finish naturally before a skill's teardown ever
        # reaches stop() (see _kill_all_sandbox_processes()'s docstring for
        # why nothing does this implicitly here, unlike container removal).
        # Release happens right after -- this kill attempt IS the
        # confirmation the sandbox has exited; there is no separate
        # "is it actually clear yet" wait (see release_gpu()'s docstring for
        # why re-checking the same tracked state the kill just acted on
        # would be redundant).
        self._kill_all_sandbox_processes()
        if gpu_id_to_release is not None and self._gpu_release_fn is not None:
            self._gpu_release_fn(gpu_id_to_release)
            self._gpu_id = None
        self._watched_pids = {}
        self._invocation_pgids = set()

    def rm_container(self) -> None:
        """Discard the jail's current workspace outright -- kill tracked
        processes, release the GPU, and delete the live workspace directory
        under /tmp. The chroot equivalent of force-removing a container.

        Does not touch any committed snapshot: the next _ensure_started()
        materializes fresh from self._checkpoint_image (or starts empty if
        none exists yet), which is what makes this the discard/revert
        primitive -- call it without a preceding commit() to throw away
        everything since the last checkpoint.
        """
        gpu_id_to_release = (
            self._gpu_id if (self._gpu_virtual and self._gpu_id is not None) else None
        )
        self._kill_all_sandbox_processes()
        if gpu_id_to_release is not None and self._gpu_release_fn is not None:
            self._gpu_release_fn(gpu_id_to_release)
            self._gpu_id = None
        self._watched_pids = {}
        self._invocation_pgids = set()
        if self._workspace.exists():
            shutil.rmtree(self._workspace, ignore_errors=True)

    def restore(self, tag: str) -> None:
        """Restore the jail's workspace from a previously committed snapshot."""
        # Kill before overwriting the workspace underneath any still-running
        # process from the state being replaced (see
        # _kill_all_sandbox_processes()'s docstring).
        self._kill_all_sandbox_processes()
        self._watched_pids = {}
        self._invocation_pgids = set()
        self._checkpoint_image = tag
        # Materialize explicitly rather than calling _ensure_started() --
        # that only restores when the workspace doesn't exist yet (see its
        # docstring), which would make an explicit restore() onto an
        # already-materialized workspace a silent no-op.
        self._materialize_workspace(tag)

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        # rm_container() kills tracked processes, releases the GPU, and
        # deletes the workspace (all idempotent -- a no-op if a prior
        # stop()/rm_container() already did them) before the root rmtree
        # below removes anything left over; safe to call unconditionally
        # even though it duplicates that one step.
        self.rm_container()
        if self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)
        if self._checkpoint_image:
            self.delete_image(self._checkpoint_image, force=True)
            self._checkpoint_image = None

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

    @staticmethod
    def relabel_owner_pid(tag: str, owner_pid: "int | None", timeout: int) -> None:
        """No-op: a chroot snapshot is a plain directory copy with no
        label/metadata concept at all (see _ContainerBackendBase's
        version, which this mirrors for API parity so agent.py's
        save()/load() can call it generically regardless of backend)."""
