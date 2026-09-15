"""Shared docker/podman plumbing.

Runtime detection (`get_container_runtime()`), the subprocess-call throttle
(`_get_docker_semaphore()`), and `_ContainerBackendBase` -- the base class
`.docker._DockerBackend` and `.podman._PodmanBackend` both subclass for
everything that doesn't differ between the two runtimes, which is nearly
everything, including the session-keyring-quota machinery below.

Linux charges each `docker run`/`podman run` invocation's session keyring
against a quota (`/proc/sys/kernel/keys/maxkeys`) keyed by the *real host
UID*, not by user namespace -- rootless Podman's per-container user
namespaces do not exempt it: `runc` joins/creates the session keyring before
the container process finishes transitioning into its remapped identity, so
the charge lands on the same `key_user` bucket a rootless Docker container
run by the same host user would hit. (Confirmed both empirically -- watching
`/proc/keys` gain a `_ses.*` entry owned by the real UID across a plain
`podman run`/`rm` cycle -- and upstream: see containers/podman#13363,
kubernetes-sigs/kind#3806.) `_keyring_container_limit()`/`keyring_quota()`/
`_semaphore_held_count()` and the concrete `_is_quota_exhaustion_error()`/
`_wait_for_quota_slot()`/`_quota_diagnostics()`/`_acquire_runtime_slot()`/
`_release_runtime_slot()` implementations on `_ContainerBackendBase` below
therefore apply to both runtimes identically; neither `.docker` nor `.podman`
overrides any of them. See `.docker` and `.podman` for the handful of things
that still differ (mainly `_resolve_image`).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid as _uuid
from contextlib import contextmanager as _contextmanager
from pathlib import Path

from ..observability.profiler import agprof
from ..configs.agconfig import DOCKER_SEMAPHORE_LIMIT, agconfig as agconfig_cls
from ..utils.agutil import (
    agency_run_id,
    agency_run_scratch_dir,
    amd_render_node_paths_by_pci_bus,
    detect_gpus,
)
from .base import AgSandboxBackendFields, agsandbox_backend, run_with_unkillable_child_grace
from ._layer_squash import merge_layer_tars, overlay_diff_to_tar, sha256_file, build_save_archive

# _RUN_ID is never read within this module itself -- it's defined here and
# imported by sandbox/__init__.py (which re-exports it for
# agsandbox.py's container-naming and a battery of tests/test_sandbox_naming.py
# cases). Declared here explicitly so static analysis recognizes it as an
# intentional export rather than a dead global.
__all__ = ["_RUN_ID"]


class _ContainerAlreadyRunning(Exception):
    """Raised by _run_with_conflict_retry when another process has already
    started the same container — the caller should reuse it."""


_RUNTIME: str | None = None

# Per-run ID so concurrent and successive runs never share container/image names.
# The same process-wide id every other run-scoped path uses (the LLM gateway/
# UDS socket directory, the default log directory) -- avoids PID-reuse
# collisions and prevents stale lifecycle images from crashed runs being
# accidentally picked up by a new run that happens to get the same PID.
_RUN_ID = agency_run_id()

# Limit the number of containers starting simultaneously.  Each concrete
# container backend construction acquires one slot for the duration of its
# startup sequence (docker/podman run + first exec). Without this, a burst of
# hundreds of parallel agent tasks overwhelms the daemon.  All docker/podman
# calls go through _run(), which holds this semaphore for the duration of each
# subprocess call.  Caps concurrent daemon calls at 16: the daemon serialises
# most operations internally (GPU init, overlay diff, container teardown), so
# more than ~16 concurrent calls increase contention without reducing
# wall-clock time.
# Process-wide, not per-agent config (agency.configs.agconfig.DOCKER_SEMAPHORE_LIMIT)
# -- lazily created on first use. Locked once actually read, matching a real
# semaphore's can't-resize-after-creation semantics.
_docker_semaphore: threading.Semaphore | None = None
_docker_semaphore_init_lock = threading.Lock()


def _get_docker_semaphore() -> threading.Semaphore:
    global _docker_semaphore
    if _docker_semaphore is None:
        with _docker_semaphore_init_lock:
            if _docker_semaphore is None:
                _docker_semaphore = threading.Semaphore(DOCKER_SEMAPHORE_LIMIT)
    return _docker_semaphore


@_contextmanager
def _docker_semaphore_slot():
    """Profile acquisition separately from the grouped docker/podman slot hold."""
    sem = _get_docker_semaphore()
    with agprof.span("sync:container"):
        sem.acquire()
    try:
        with agprof.span("runtime:container_slot_hold"):
            yield
    finally:
        sem.release()


def _runtime_works(runtime: str) -> bool:
    try:
        proc = subprocess.run(
            [runtime, "info"],
            capture_output=True,
            timeout=agconfig_cls().sandbox.inspect_timeout_s,
        )
        return proc.returncode == 0
    except Exception:
        return False


def get_container_runtime() -> str:
    """Return ``docker`` or ``podman``, preferring podman when both are usable."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    with agprof.span("runtime:detect"):
        has_docker = shutil.which("docker") is not None
        has_podman = shutil.which("podman") is not None
        docker_ok = has_docker and _runtime_works("docker")
        podman_ok = has_podman and _runtime_works("podman")

    if podman_ok:
        _RUNTIME = "podman"
    elif docker_ok:
        _RUNTIME = "docker"
    elif has_docker or has_podman:
        parts = []
        if has_docker and not docker_ok:
            parts.append("docker is installed but not reachable (is the daemon running?)")
        if has_podman and not podman_ok:
            parts.append("podman is installed but not reachable")
        raise RuntimeError("; ".join(parts))
    else:
        raise RuntimeError(
            "Neither docker nor podman is installed. Install one of them to use sandboxed agents."
        )
    return _RUNTIME


# Every container _ContainerBackendBase starts is labeled with the PID of the
# process that created it, so reap_orphaned_containers() below can tell a
# genuinely dead run's containers apart from a different, still-live agency
# process's -- container names alone can't do this: _RUN_ID is deliberately
# randomized per process (see its own comment) specifically so a new run
# never collides with a crashed run's names, which also means a new run
# never naturally reclaims them either.
_AGENCY_OWNER_PID_LABEL = "agency.owner_pid"


def _pid_alive(pid: int) -> bool:
    """Return True if *pid* is running on this host -- delegates to
    `agutil.pid_alive` so every reaper shares one definition."""
    from ..utils.agutil import pid_alive

    return pid_alive(pid)


_reap_lock = threading.Lock()
_reap_done = False


def reap_orphaned_containers() -> None:
    """Force-remove containers (and their lifecycle images) left by a
    SIGKILL'd previous process, plus any lifecycle image whose owning
    container is already gone. Runs at most once per process, triggered by
    `_ContainerBackendBase.__init__`; best-effort throughout."""
    global _reap_done
    if _reap_done:
        return
    with _reap_lock:
        if _reap_done:
            return
        _reap_done = True
        try:
            _do_reap_orphaned_containers()
        except Exception as _e:
            # DATACOLLECTOR: append -- host-level, no single agname (runs once per process at startup).
            print(f"[agsandbox_backend] WARNING: startup container reap failed: {_e}")


def _do_reap_orphaned_containers() -> None:
    try:
        runtime = get_container_runtime()
    except RuntimeError:
        return  # no container runtime usable -- nothing to reap against
    fmt = '{{.ID}}\t{{.Label "%s"}}\t{{.Names}}' % _AGENCY_OWNER_PID_LABEL
    result = subprocess.run(
        [runtime, "ps", "-a", "--filter", f"label={_AGENCY_OWNER_PID_LABEL}", "--format", fmt],
        capture_output=True,
        timeout=agconfig_cls().sandbox.inspect_timeout_s,
    )
    if result.returncode != 0:
        return
    own_pid = os.getpid()
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        container_id, pid_str, name = parts
        try:
            owner_pid = int(pid_str)
        except ValueError:
            continue
        if owner_pid == own_pid or _pid_alive(owner_pid):
            continue  # still owned by a live process (or ourselves) -- leave it
        # DATACOLLECTOR: append -- informational (routine cleanup action), host-level; the
        # orphaned agent's own process is already dead, so this can't reach its data_collector.
        print(
            f"[agsandbox_backend] Reaping container {name!r} ({container_id[:12]}), "
            f"orphaned by dead process {owner_pid} (likely SIGKILL'd)",
            flush=True,
        )
        subprocess.run([runtime, "rm", "-f", container_id], capture_output=True)
        lifecycle_tag = f"agency/lifecycle-{name}".lower()
        subprocess.run([runtime, "rmi", "-f", lifecycle_tag], capture_output=True)

    _reap_orphaned_lifecycle_images(runtime, own_pid)


def _reap_orphaned_lifecycle_images(runtime: str, own_pid: int) -> None:
    """Remove `agency/lifecycle-*` images whose owning process is confirmed
    dead, independent of whether their container still exists.

    `docker commit` propagates a container's labels onto its image
    automatically, so every lifecycle image already carries
    _AGENCY_OWNER_PID_LABEL for free -- EXCEPT one produced by
    `_squash_commit()`'s export/import fallback, which needs (and, as of
    this fix, has) an explicit `--change` to re-apply it, since export/
    import doesn't preserve container config at all. Images still missing
    the label (e.g. ones committed before this fix existed) simply don't
    match the `--filter label=...` query below and are silently skipped --
    unreapable by this mechanism until a future commit refreshes them, not
    a correctness problem, just a known gap for pre-existing images.

    Safety: a lifecycle tag can carry a STALE label from an unrelated,
    long-dead process -- most notably after `agent.load()`'s checkpoint
    restore (`docker load` DOES preserve the original label, from
    whatever process originally called `agent.save()`, possibly on a
    different host entirely) -- while a brand-new, genuinely-live process
    is right now running a container from that same tag, simply because it
    hasn't committed under its OWN pid yet. Checking the label alone would
    risk deleting an image a live container depends on. So even after the
    label says "dead," this also confirms no container (running or not --
    matches stop()'s own old-image-cleanup check) currently exists with
    this image as its ancestor before ever calling `rmi`.
    """
    # Unlike `docker ps`, `docker images`'s --format context has no
    # .Label/.Labels accessor at all (confirmed empirically: both raise a
    # template-parsing error, even {{json .}} omits labels entirely) --
    # the `--filter label=...` half still works fine, it's only reading
    # the value back in the same command that's unsupported. So this
    # filters for candidate tags first, then reads each one's actual
    # label value via a separate `docker inspect` (cheap: this list is
    # small and this only runs once per process).
    result = subprocess.run(
        [
            runtime,
            "images",
            "--filter",
            f"label={_AGENCY_OWNER_PID_LABEL}",
            "--format",
            "{{.Repository}}:{{.Tag}}",
        ],
        capture_output=True,
        timeout=agconfig_cls().sandbox.inspect_timeout_s,
    )
    if result.returncode != 0:
        return
    for tag in result.stdout.decode("utf-8", errors="replace").splitlines():
        tag = tag.strip()
        if not tag or not tag.startswith("agency/lifecycle-"):
            continue
        try:
            label_result = subprocess.run(
                [
                    runtime,
                    "inspect",
                    "--format",
                    f'{{{{index .Config.Labels "{_AGENCY_OWNER_PID_LABEL}"}}}}',
                    tag,
                ],
                capture_output=True,
                timeout=agconfig_cls().sandbox.inspect_timeout_s,
            )
            if label_result.returncode != 0:
                continue
            owner_pid = int(label_result.stdout.decode("utf-8", errors="replace").strip())
        except Exception as _e:
            # The images list above was already filtered to label=..., so
            # reaching here means the tag vanished (e.g. removed by a
            # concurrent process) or its label value was somehow
            # unparseable -- rare. Unknown either way, never treat as dead.
            # DATACOLLECTOR: append -- host-level, no single agname.
            print(
                f"[agsandbox_backend] WARNING: could not read owner_pid label for "
                f"lifecycle image {tag!r}, skipping: {_e}",
                file=__import__("sys").stderr,
                flush=True,
            )
            continue
        if owner_pid == own_pid or _pid_alive(owner_pid):
            continue  # still owned by a live process (or ourselves) -- leave it
        try:
            in_use = subprocess.run(
                [runtime, "ps", "-a", "--filter", f"ancestor={tag}", "--format", "{{.ID}}"],
                capture_output=True,
                timeout=agconfig_cls().sandbox.stop_ps_check_timeout_s,
            )
            if in_use.returncode != 0:
                continue  # couldn't confirm safety -- skip rather than risk it
            if in_use.stdout.strip():
                continue  # a container -- possibly a fresh owner reusing this tag -- is still running from it
        except Exception as _e:
            # DATACOLLECTOR: append -- host-level, no single agname.
            print(
                f"[agsandbox_backend] WARNING: could not confirm lifecycle image {tag!r} "
                f"is unused, skipping rather than risk deleting a live image: {_e}",
                file=__import__("sys").stderr,
                flush=True,
            )
            continue
        # DATACOLLECTOR: append -- informational (routine cleanup action), host-level; the
        # orphaned agent's own process is already dead, so this can't reach its data_collector.
        print(
            f"[agsandbox_backend] Reaping lifecycle image {tag!r}, "
            f"orphaned by dead process {owner_pid} (likely SIGKILL'd)",
            flush=True,
        )
        subprocess.run([runtime, "rmi", "-f", tag], capture_output=True)


def seed_cache_from_image(
    host_dir,
    container_path: str,
    image: str,
    timeout: int = AgSandboxBackendFields.SEED_CACHE_TIMEOUT_S,
) -> None:
    """Copy *container_path* out of *image* into *host_dir*, once, if *host_dir* is empty.

    For a shared host mount (see ``agconfig.add_mount``) that would
    otherwise shadow pre-baked, gated content already inside the image (e.g.
    HuggingFace model weights only fetchable with credentials available at
    build time, not at container-run time).
    """
    host_dir = Path(host_dir).resolve()
    host_dir.mkdir(parents=True, exist_ok=True)
    if any(host_dir.iterdir()):
        return
    runtime = get_container_runtime()
    import shlex

    subprocess.run(
        [
            runtime,
            "run",
            "--rm",
            "-v",
            f"{host_dir}:/__seed_out",
            image,
            "sh",
            "-c",
            f"cp -a {shlex.quote(container_path)}/. /__seed_out/ 2>/dev/null || true",
        ],
        check=False,
        timeout=timeout,
    )


_gpu_kind_cache: "dict[str, str]" = {}
_gpu_flags_lock = threading.Lock()
_amd_render_node_paths_cache: "list[str] | None" = None

# Matches only a per-GPU render node ("renderD128", "renderD129"), never the
# shared /dev/dri/card* control nodes -- mirrors chroot.py's
# _NVIDIA_INDEXED_DEV_RE, same rationale.
_AMD_RENDER_NODE_RE = re.compile(r"^renderD(\d+)$")


def _gpu_kind(runtime: str) -> str:
    """Return "nvidia", "amd", or "none" for *runtime*, cached per runtime
    for the process lifetime -- the expensive part of GPU flag selection
    (detect_gpus()/nvidia-smi presence) doesn't depend on which specific
    GPU a given sandbox leases, so it's cached separately from the
    per-lease flags themselves (see _gpu_flags())."""
    if runtime in _gpu_kind_cache:
        return _gpu_kind_cache[runtime]
    with _gpu_flags_lock:
        if runtime not in _gpu_kind_cache:
            if not detect_gpus():
                kind = "none"
            elif shutil.which("nvidia-smi"):
                kind = "nvidia"
            else:
                kind = "amd"
            _gpu_kind_cache[runtime] = kind
        return _gpu_kind_cache[runtime]


def _amd_render_node_paths() -> "list[str]":
    """Host /dev/dri/renderD* paths ordered so index N is the render node
    for rocm-smi's GPU N, cached for the process lifetime -- mirrors
    chroot.py's AMD branch of _chroot_gpu_dev_paths() so both backends
    enumerate GPUs the same way.

    Ordered via amd_render_node_paths_by_pci_bus() (matches each GPU's
    rocm-smi PCI bus against its render node's own resolved PCI bus),
    falling back to naive sorted order only if that mapping can't be built
    (e.g. rocm-smi --showbus unavailable) -- confirmed on real 8x MI350X
    hardware that sorted order does NOT correspond to GPU index (see
    agresources.amd_render_node_paths_by_pci_bus's docstring)."""
    global _amd_render_node_paths_cache
    if _amd_render_node_paths_cache is not None:
        return _amd_render_node_paths_cache
    with _gpu_flags_lock:
        if _amd_render_node_paths_cache is None:
            dri = Path("/dev/dri")
            naive = (
                sorted(str(p) for p in dri.iterdir() if _AMD_RENDER_NODE_RE.match(p.name))
                if dri.is_dir()
                else []
            )
            resolved = amd_render_node_paths_by_pci_bus(naive) if naive else None
            _amd_render_node_paths_cache = resolved if resolved is not None else naive
        return _amd_render_node_paths_cache


def _gpu_flags(runtime: str) -> list[str]:
    """Return GPU passthrough flags for *runtime* ("docker" or "podman").
    Attaches every GPU unconditionally (neither runtime supports attaching
    a device after container creation); access is actually gated later,
    in-container, via CUDA/HIP_VISIBLE_DEVICES. Podman needs CDI
    (`--device nvidia.com/gpu=all`), not Docker's `--gpus all`."""
    kind = _gpu_kind(runtime)
    if kind == "none":
        return []
    if kind == "nvidia":
        return ["--device", "nvidia.com/gpu=all"] if runtime == "podman" else ["--gpus", "all"]
    flags = ["--device", "/dev/kfd"]
    for node in _amd_render_node_paths():
        flags += ["--device", node]
    return flags


# Hard cap on the number of simultaneously running containers (docker and
# podman share this one cap, not one each -- see this module's docstring for
# why they're both subject to the same kernel session-keyring quota).
# multiprocessing.Semaphore is backed by a POSIX IPC semaphore so the limit
# remains process-wide if multiple processes use the backend, rather than
# applying independently to threads in each process.
def _keyring_container_limit() -> int:
    """Return the concurrent-container cap derived from the kernel keyring quota."""
    _fields = agconfig_cls().sandbox
    try:
        maxkeys = int(Path("/proc/sys/kernel/keys/maxkeys").read_text().strip())
        return max(_fields.container_limit_floor, maxkeys - _fields.container_limit_buffer)
    except OSError:
        return _fields.container_limit_fallback - _fields.container_limit_buffer


def keyring_quota() -> dict[str, int]:
    """Return the current Linux session-keyring quota for diagnostics.

    Returns a dict with ``used``, ``max``, and ``free`` key counts.
    ``used`` is -1 when /proc/keys is not readable (non-root on some kernels).
    """
    try:
        maxkeys = int(Path("/proc/sys/kernel/keys/maxkeys").read_text().strip())
    except OSError:
        maxkeys = -1
    try:
        used = sum(1 for ln in Path("/proc/keys").read_text().splitlines() if ln.strip())
    except OSError:
        used = -1
    free = (maxkeys - used) if (maxkeys >= 0 and used >= 0) else -1
    return {"used": used, "max": maxkeys, "free": free}


def _semaphore_held_count() -> str:
    """Return 'held/limit' for the container-concurrency semaphore, or
    '?/limit' if unreadable. Reflects only this process's own containers --
    compare against keyring_quota()['used'] to see external processes'
    share."""
    limit = _keyring_container_limit()
    try:
        available = _container_semaphore._semlock._get_value()
        held = limit - available
    except Exception:
        held = "?"
    return f"{held}/{limit}"


_container_semaphore: "multiprocessing.BoundedSemaphore" = multiprocessing.BoundedSemaphore(
    _keyring_container_limit()
)


class _ContainerBackendBase(agsandbox_backend):
    """Manages a single container for one agent via docker or podman.

    Subclassed by `.docker._DockerBackend` and `.podman._PodmanBackend`,
    which each hardcode their own `_runtime` string and override only the
    handful of things that genuinely differ between the two: bare image
    names need a `localhost/` prefix for Podman only (`_resolve_image` —
    Podman requires fully-qualified names when no unqualified-search
    registries are configured in /etc/containers/registries.conf, Docker
    accepts bare names fine). The session-keyring-derived concurrency
    handling below (`_acquire_runtime_slot`/`_release_runtime_slot`/
    `_is_quota_exhaustion_error`/`_wait_for_quota_slot`/`_quota_diagnostics`)
    is NOT one of those differences: both runtimes are subject to the exact
    same kernel quota (see this module's docstring), so both use the same
    concrete implementations here rather than one of them overriding a
    no-op. Everything else — command building, retries, checkpointing,
    cleanup — is identical regardless of which binary is actually being
    shelled out to, and lives here once.

    All filesystem operations route through ``docker exec`` / ``podman exec``.
    Background PIDs spawned by bash calls are tracked in ``_watched_pids`` and
    the outer monitoring loop in ``agent._task()`` waits for them before
    resolving the skill future.
    """

    IMAGE_KIND = "container"
    _runtime: str = ""  # set by _DockerBackend/_PodmanBackend as a class attribute

    def _resolve_image(self, name: str) -> str:
        """Identity by default (Docker accepts bare image names fine).
        Overridden by _PodmanBackend, which requires a fully-qualified name."""
        return name

    def _acquire_runtime_slot(self) -> None:
        """Acquire the session-keyring-derived container-concurrency
        semaphore before starting a new container."""
        _container_semaphore.acquire()

    def _release_runtime_slot(self) -> None:
        """Release the slot _acquire_runtime_slot() acquired."""
        try:
            _container_semaphore.release()
        except ValueError as _e:
            # DATACOLLECTOR: append, agname=self._agname -- real invariant-violation signal.
            print(
                f"[agsandbox_backend] WARNING: container-slot semaphore double-release: {_e}",
                file=__import__("sys").stderr,
                flush=True,
            )

    def _is_quota_exhaustion_error(self, stderr: str) -> bool:
        """True if *stderr* indicates the runtime hit the Linux
        session-keyring quota (matches both docker's and podman's/runc's wording)."""
        return "session key" in stderr or ("disk quota exceeded" in stderr and "keyring" in stderr)

    def _wait_for_quota_slot(self) -> None:
        """Poll /proc's actual keyring free count until a slot opens (or
        give up after keyring_wait_timeout_s) -- catches quota exhaustion
        from processes outside our own semaphore's accounting."""
        deadline = time.monotonic() + self._agconfig.sandbox.keyring_wait_timeout_s
        while time.monotonic() < deadline:
            if keyring_quota().get("free", 0) > 0:
                return
            time.sleep(self._agconfig.sandbox.keyring_poll_interval_s)

    def _quota_diagnostics(self) -> str:
        """Return a short diagnostic string describing quota state, appended
        to _run_with_conflict_retry()'s final "retries exhausted" error
        message."""
        quota = keyring_quota()
        return (
            f"[keyring: {quota['used']}/{quota['max']} used, "
            f"framework semaphore: {_semaphore_held_count()} held]"
        )

    def __init__(
        self,
        agname: str,
        *,
        name: str,
        checkpoint_image: "str | None",
        base_image: str,
        mounts: "dict[str, tuple[str, str, str]]",
        agconfig: "agconfig_cls | None",
    ) -> None:
        reap_orphaned_containers()
        # Captured at construction (not lazily) so a serialized/forked copy
        # still reports the real owning process.
        self._owner_pid = os.getpid()
        self._agname = agname
        self._gpu_ids: list[int] = []
        self._gpu_count_requested: int = 0
        self._gpu_acquire_fn = None
        self._gpu_release_fn = None
        self._cpu_acquired: float = 0.0
        self._memory_acquired_mb: int = 0
        self._watched_pids: dict[int, float] = {}
        # None means "not yet captured", distinct from an empty baseline.
        self._baseline_pids: "set[int] | None" = None
        self._daemon_pids: set[int] = set()
        self._ptrace_managed_pids: set[int] = set()
        self._started = False
        self._destroyed = False
        self._checkpoint_image: str | None = checkpoint_image
        # Squash-time accumulator state (see _build_accumulator_for_squash()
        # / _squash_commit()); cleared after each squash attempt.
        self._accumulated_diff_path: "Path | None" = None
        self._accumulated_layer_count: int = 0
        self._accumulator_dir: "Path | None" = None
        self._squash_base_diff_ids: "list[str] | None" = None
        self._agconfig = agconfig if agconfig is not None else agconfig_cls()
        self._validate_config(self._agconfig)
        self._name = name
        self._base_image = base_image
        self._checkpoint_mounts = tuple(mounts.values())
        from .checkpoint import checkpoint_backend, ZfsRuntimeStorage, CheckpointCapabilityError

        self._checkpointer = checkpoint_backend(self._agconfig.sandbox.checkpoint_backend)
        self._checkpoint_storage = None
        if self._agconfig.sandbox.checkpoint_zfs_parent:
            if {"_agharness_llm_gateway", "_agency_logs"} & self._agconfig.sandbox.mounts.keys():
                raise CheckpointCapabilityError(
                    "Private ZFS storage cannot override Agency's control/log mounts"
                )
            if checkpoint_image is not None:
                raise CheckpointCapabilityError(
                    "Private ZFS storage cannot import lifecycle image tags"
                )
            self._checkpoint_storage = ZfsRuntimeStorage(self)
        self._vol_flags: list[str] = []
        for _mount_name, (host, container, mode) in mounts.items():
            self._vol_flags += ["-v", f"{host}:{container}:{mode}"]

    def change_config(self, agconfig: "agconfig_cls | None") -> None:
        updated = agconfig if agconfig is not None else agconfig_cls()
        if (
            updated.sandbox.checkpoint_backend != self._agconfig.sandbox.checkpoint_backend
            or updated.sandbox.checkpoint_zfs_parent != self._agconfig.sandbox.checkpoint_zfs_parent
            or updated.sandbox.checkpoint_fast_resume
            != self._agconfig.sandbox.checkpoint_fast_resume
        ):
            raise ValueError("Checkpoint backend/storage is fixed at sandbox construction")
        self._validate_config(updated)
        self._agconfig = updated

    def get_config_copy(self) -> "agconfig_cls":
        return self._agconfig.clone()

    def _container_running(self) -> bool:
        """Return True if the named container is currently running in Docker/Podman."""
        result = self._run(
            [self._runtime, "inspect", "--format", "{{.State.Running}}", self._name],
            check=False,
            timeout=self._agconfig.sandbox.inspect_timeout_s,
        )
        return result.returncode == 0 and result.stdout.strip() == b"true"

    def _container_status(self) -> str:
        """Return the container state string: 'running', 'exited', 'created', etc., or '' if not found."""
        result = self._run(
            [self._runtime, "inspect", "--format", "{{.State.Status}}", self._name],
            check=False,
            timeout=self._agconfig.sandbox.inspect_timeout_s,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.decode("utf-8", errors="replace").strip()

    def _own_host_pids(self) -> "set[int]":
        """Return the host PIDs of every process currently running inside
        this container, or an empty set if the container doesn't exist / the
        query fails.

        Uses `docker/podman top` (not a /proc walk from {{.State.Pid}}):
        `exec` sessions are siblings under the runtime supervisor, not
        descendants of init. Runtime syntax differs and is NOT
        interchangeable:
          docker: `docker top <name> -eo pid` — host PIDs.
          podman: `podman top <name> hpid` — host PIDs (plain `pid` is
            namespace-local; ps(1)-style flags make podman run ps inside
            the container).
        """
        if self._runtime == "podman":
            cmd = [self._runtime, "top", self._container_name(), "hpid"]
        else:
            cmd = [self._runtime, "top", self._container_name(), "-eo", "pid"]
        result = self._run(cmd, check=False, timeout=self._agconfig.sandbox.inspect_timeout_s)
        if result.returncode != 0:
            return set()
        pids: set[int] = set()
        lines = result.stdout.decode("utf-8", errors="replace").splitlines()
        for line in lines[1:]:  # first line is the descriptor header (PID/HPID)
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
        return pids

    def _inspect_container_state(self) -> "tuple[bool, str]":
        """Return (running, status) from a SINGLE docker/podman inspect call
        -- merges what _container_running()/_container_status() would
        otherwise need two separate inspect round-trips for, since
        _ensure_started() (its only caller that needs both) always wants to
        know both facts together. status is '' if the container doesn't
        exist at all, same convention as _container_status(). Ground truth,
        same as those two -- no caching across calls (see _ensure_started()'s
        docstring for why)."""
        result = self._run(
            [
                self._runtime,
                "inspect",
                "--format",
                "{{.State.Running}}|{{.State.Status}}",
                self._name,
            ],
            check=False,
            timeout=self._agconfig.sandbox.inspect_timeout_s,
        )
        if result.returncode != 0:
            return (False, "")
        running_str, _, status = (
            result.stdout.decode("utf-8", errors="replace").strip().partition("|")
        )
        return (running_str == "true", status)

    def _ensure_workspace_dir(self) -> None:
        """Guarantee /workspace exists in a container that is about to be used."""
        self._run(
            [self._runtime, "exec", self._name, "mkdir", "-p", "/workspace"],
            check=True,
        )

    def _ensure_started(self) -> None:
        storage = getattr(self, "_checkpoint_storage", None)
        if storage is not None:
            storage.prepare(self)
        from .checkpoint import CowZfsCheckpoint

        checkpointer = getattr(self, "_checkpointer", None)
        restoring = isinstance(checkpointer, CowZfsCheckpoint) and checkpointer.hibernated
        if isinstance(checkpointer, CowZfsCheckpoint):
            if checkpointer.hibernated:
                checkpointer.restore(self, checkpointer.latest)
        phase = "runtime.container_start" if restoring else "runtime.container_create"
        with agprof.span("sandbox:start"), agprof.span(phase):
            self._ensure_started_profiled()
        self._register_prof_container()

    def _ensure_started_profiled(self) -> None:
        """Start the Docker/Podman container on first use."""
        name = self._name
        storage = getattr(self, "_checkpoint_storage", None)
        running, status = self._inspect_container_state()
        if running:
            # Reuse an already-running container — it already holds whatever
            # slot _acquire_runtime_slot() would take, so we must NOT acquire
            # it again here.
            if self._baseline_pids is None:
                self._baseline_pids = self._snapshot_pids_started()
            return
        if status:
            # Hibernating (stopped(), not removed) -- resume it in place.
            # No image, no `docker/podman run`: the container's writable
            # layer already holds everything from before it was stopped.
            self._acquire_runtime_slot()
            try:
                self._start_with_quota_retry(name)
            except Exception:
                self._release_runtime_slot()
                raise
            if storage is not None:
                storage.restore_runtime_files_after_start()
            self._ensure_workspace_dir()
            if self._baseline_pids is None:
                self._baseline_pids = self._snapshot_pids_started()
            return
        # Acquire the physical GPU(s) (if reserve_resource(gpu=N) was called)
        # before the container is created, not just in exec() -- this method
        # can be reached first via read_file()/write_file() rather than
        # exec(), so exec()'s own lazy acquire (base.py) can't be relied on
        # to have already run. Guarded by `not self._gpu_ids` the same way
        # exec()'s does, so whichever entry point gets here first acquires it
        # exactly once. Note this only affects which physical GPU(s)
        # CUDA_VISIBLE_DEVICES points at -- _gpu_flags() below attaches every
        # GPU device to the container unconditionally, so it no longer
        # matters whether this runs before or after reserve_resource().
        if self._gpu_count_requested > 0 and not self._agconfig.sandbox.gpu_passthrough:
            raise RuntimeError("GPU reservation is forbidden by this sandbox's CPU-only policy")
        if self._gpu_count_requested > 0 and not self._gpu_ids and self._gpu_acquire_fn is not None:
            self._gpu_ids = self._gpu_acquire_fn(self._gpu_count_requested)
        gpu_flags = _gpu_flags(self._runtime) if self._agconfig.sandbox.gpu_passthrough else []
        if (
            storage is not None
            and self._agconfig.sandbox.checkpoint_fast_resume
            and storage.criu_ready
        ):
            gpu_flags += ["--cap-add=SYS_PTRACE"]
            if self._runtime == "podman":
                gpu_flags += [f"--init-path={storage.init_path}"]
            gpu_flags += ["--annotation", f"org.criu.config={storage.criu_config}"]
        cgroup_flags = []
        cgroup_parent = agprof.container_cgroup_parent()
        if cgroup_parent is not None and self._runtime == "docker":
            cgroup_flags = [f"--cgroup-parent={cgroup_parent}"]
        self._acquire_runtime_slot()
        try:
            # A restored image does not carry HostConfig resource constraints.
            # Apply the same limits on fresh creation and rollback recreation.
            limit_flags = []
            if self._agconfig.resources.idle_memory is not None:
                limit_flags.append(f"--memory={self._agconfig.resources.idle_memory}")
            if self._cfs_supported():
                limit_flags.append(f"--cpus={self._agconfig.resources.idle_cpus}")
            if self._agconfig.sandbox.cpuset_cpus is not None:
                limit_flags.append(f"--cpuset-cpus={self._agconfig.sandbox.cpuset_cpus}")
            if self._agconfig.sandbox.cpuset_mems is not None:
                limit_flags.append(f"--cpuset-mems={self._agconfig.sandbox.cpuset_mems}")
            if self._checkpoint_image is not None:
                # Restart from last committed checkpoint (set by commit()).
                # /workspace and all state from the previous tool call are preserved.
                image = self._checkpoint_image
                run_cmd = (
                    [self._runtime, "run", "-d", "--init", "--name", name]
                    + ["--label", f"{_AGENCY_OWNER_PID_LABEL}={self._owner_pid}"]
                    + limit_flags
                    + cgroup_flags
                    + gpu_flags
                    + self._vol_flags
                    + list(self._agconfig.sandbox.flags)
                    + [image, "tail", "-f", "/dev/null"]
                )
                self._run_with_conflict_retry(run_cmd, name)
                self._ensure_workspace_dir()
                # Keep _checkpoint_image — not a one-shot restore, needed for future restarts.
            else:
                image = self._resolve_image(self._base_image)
                run_cmd = (
                    [self._runtime, "run", "-d", "--init", "--name", name]
                    + ["--label", f"{_AGENCY_OWNER_PID_LABEL}={self._owner_pid}"]
                    + limit_flags
                    + cgroup_flags
                    + gpu_flags
                    + self._vol_flags
                    + list(self._agconfig.sandbox.flags)
                    + [image, "tail", "-f", "/dev/null"]
                )
                self._run_with_conflict_retry(run_cmd, name)
                self._ensure_workspace_dir()
        except _ContainerAlreadyRunning:
            # Another process started the container while we were retrying;
            # that process owns the slot — release ours.
            self._release_runtime_slot()
            if self._baseline_pids is None:
                self._baseline_pids = self._snapshot_pids_started()
            return
        except Exception:
            self._release_runtime_slot()
            raise
        if self._baseline_pids is None:
            self._baseline_pids = self._snapshot_pids_started()

    def _prof_container_label(self) -> str:
        """Keep the sandbox object's unique identity in every resource series."""
        return str(self._agname)

    def _register_prof_container(self) -> None:
        """Register this running container's kernel cgroup with agprof.

        Resolve the cgroup from the runtime-reported host init PID rather
        than assuming a Docker or Podman cgroup naming scheme. Profiling is
        intentionally strict here: an active session on anything other than
        cgroup v2 fails loudly instead of producing incomplete resource data.
        """
        if not agprof.enabled():
            return
        label = self._prof_container_label()
        if agprof.container_registered(label):
            return
        result = self._run(
            [
                self._runtime,
                "inspect",
                "--format",
                "{{.Id}}|{{.State.Pid}}",
                self._name,
            ],
            check=False,
            timeout=self._agconfig.sandbox.inspect_timeout_s,
        )
        raw_inspect = result.stdout.decode("utf-8", errors="replace").strip()
        container_id, separator, pid_text = raw_inspect.partition("|")
        try:
            pid = int(pid_text)
        except (TypeError, ValueError):
            pid = 0
        if (
            result.returncode != 0
            or not separator
            or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
            or pid <= 0
        ):
            raise RuntimeError(
                f"agprof: cannot resolve ID and init PID for container {self._name!r} — "
                "only docker/podman on cgroup v2 are supported"
            )
        try:
            cg_text = Path(f"/proc/{pid}/cgroup").read_text()
        except OSError as e:
            raise RuntimeError(f"agprof: cannot read cgroup of container PID {pid}: {e}") from e
        rel = next(
            (
                line.split("::", 1)[1].strip()
                for line in cg_text.splitlines()
                if line.startswith("0::")
            ),
            None,
        )
        if rel is None:
            raise RuntimeError(
                f"agprof: container PID {pid} has no cgroup v2 entry — "
                "cgroup v1 hosts are not supported"
            )
        cgroup_dir = "/sys/fs/cgroup" + rel
        # Podman's systemd driver can place the payload in a `container`
        # child of its libpod scope. Match the exact ID-bearing component
        # anywhere in the path, retaining the remote-daemon identity check.
        expected_names = {
            container_id,
            f"docker-{container_id}.scope",
            f"libpod-{container_id}.scope",
        }
        if expected_names.isdisjoint(os.path.normpath(cgroup_dir).split(os.sep)):
            raise RuntimeError(
                f"agprof: local cgroup for runtime PID {pid} does not match container "
                f"{container_id[:12]} — the docker/podman daemon may be remote or VM-backed"
            )

        daemon_dir = None
        daemon_kind = "conmon"
        scope = cgroup_dir
        while scope and not scope.endswith(".scope") and scope != "/sys/fs/cgroup":
            scope = os.path.dirname(scope)
        base = os.path.basename(scope)
        if base.startswith("libpod-"):
            candidate = os.path.join(
                os.path.dirname(scope), f"libpod-conmon-{base[len('libpod-') :]}"
            )
            if os.path.isdir(candidate):
                daemon_dir = candidate
        elif base.startswith("docker-"):
            daemon_kind = "dockerd"
            for service in ("docker.service", "containerd.service"):
                candidate = f"/sys/fs/cgroup/system.slice/{service}"
                if os.path.isdir(candidate):
                    agprof.container_started(label, cgroup_dir, candidate, daemon_kind)
            daemon_dir = None
        agprof.container_started(label, cgroup_dir, daemon_dir, daemon_kind)

    def _snapshot_pids_started(self) -> set[int]:
        """List live PIDs through `_container_exec_started()` while starting."""
        out, _ = self._container_exec_started(
            "__SELF=$$\n"
            "for __d in /proc/[0-9]*; do\n"
            '  [ -f "$__d/status" ] || continue\n'
            "  __p=${__d##*/}\n"
            '  [ "$__p" != "$__SELF" ] && echo "$__p"\n'
            "done",
            timeout=self._agconfig.sandbox.inspect_timeout_s,
            shell="sh",
        )
        pids: set[int] = set()
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
        return pids

    def _run_with_conflict_retry(self, run_cmd: list[str], name: str) -> None:
        """Run a docker/podman run command, retrying on name-conflict/quota
        errors up to conflict_retry_max_attempts times.

        A "Conflict / already in use" error can arise when a previous run call
        failed mid-way (e.g. GPU allocation timeout) and left a container
        object in "Created" state without ever starting.  We force-remove the
        stale entry and retry rather than surfacing an opaque error to the agent.

        Quota exhaustion (the Linux session-keyring quota -- see
        _is_quota_exhaustion_error's docstring; both docker and podman are
        subject to it) is handled through the _is_quota_exhaustion_error()/
        _wait_for_quota_slot()/_quota_diagnostics() hooks rather than directly
        here.
        """
        _last_stderr = ""
        for attempt in range(self._agconfig.sandbox.conflict_retry_max_attempts):
            result = self._run(run_cmd, timeout=self._agconfig.sandbox.docker_run_timeout_s)
            if result.returncode == 0:
                return
            stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            _last_stderr = stderr
            conflict = "already in use" in stderr or "Conflict" in stderr
            if self._is_quota_exhaustion_error(stderr):
                self._wait_for_quota_slot()
                # run can partially succeed before failing on quota: it creates
                # the container object (reserving the name) but fails before
                # starting processes.  Remove any such "Created" artifact so
                # the next attempt does not see a spurious name conflict.
                if self._container_status():
                    self._rm_container(name)
            elif conflict:
                if self._container_running():
                    raise _ContainerAlreadyRunning()
                # Leftover container in a non-running state — remove it.
                # Wait until it's actually gone before retrying run.
                self._rm_container(name)
                deadline = time.monotonic() + self._agconfig.sandbox.container_removal_wait_s
                while time.monotonic() < deadline:
                    if not self._container_status():
                        break
                    time.sleep(self._agconfig.sandbox.container_removal_poll_interval_s)
                # If a quota is also exhausted (e.g. the object got created
                # then hit the limit), wait for a slot before retrying —
                # otherwise we'll create another "Created" container and loop
                # on conflicts. No-op for runtimes with no quota to wait on.
                self._wait_for_quota_slot()
                time.sleep(self._agconfig.sandbox.conflict_retry_backoff_base_s * (attempt + 1))
            else:
                msg = f"{' '.join(run_cmd[:3])} failed (exit {result.returncode})"
                if stderr:
                    msg += f": {stderr}"
                raise RuntimeError(msg)
        # Final attempt after retries exhausted.
        msg = (
            f"{self._runtime} run --name {name} failed after retries "
            f"(container name conflict or runtime quota)"
        )
        diagnostics = self._quota_diagnostics()
        if diagnostics:
            msg += f" {diagnostics}"
        if _last_stderr:
            msg += f": {_last_stderr}"
        raise RuntimeError(msg)

    def _start_with_quota_retry(self, name: str) -> None:
        """Run `docker/podman start` on an existing, stopped container,
        retrying on session-keyring quota exhaustion -- resuming a
        hibernating container re-acquires a keyring slot exactly the same
        way `run` does (see _run_with_conflict_retry()'s docstring). Unlike
        that method, there's no name-conflict case to handle here: the
        container already exists under this exact name, so there's nothing
        to remove-and-retry on -- only the quota-wait loop applies.
        """
        _last_stderr = ""
        for attempt in range(self._agconfig.sandbox.conflict_retry_max_attempts):
            result = self._run(
                [self._runtime, "start", name],
                timeout=self._agconfig.sandbox.docker_start_timeout_s,
            )
            if result.returncode == 0:
                return
            stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            _last_stderr = stderr
            if self._is_quota_exhaustion_error(stderr):
                self._wait_for_quota_slot()
                time.sleep(self._agconfig.sandbox.conflict_retry_backoff_base_s * (attempt + 1))
            else:
                msg = f"{self._runtime} start {name} failed (exit {result.returncode})"
                if stderr:
                    msg += f": {stderr}"
                raise RuntimeError(msg)
        # Final attempt after retries exhausted.
        msg = f"{self._runtime} start {name} failed after retries (runtime quota)"
        diagnostics = self._quota_diagnostics()
        if diagnostics:
            msg += f" {diagnostics}"
        if _last_stderr:
            msg += f": {_last_stderr}"
        raise RuntimeError(msg)

    def _container_name(self) -> str:
        return self._name

    @staticmethod
    def _cfs_supported() -> bool:
        """Return True if the kernel supports CFS CPU quota enforcement."""
        import os

        return os.path.exists("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")

    def _run(
        self,
        args: list[str],
        *,
        check: bool = False,
        input: bytes | None = None,
        timeout: int = AgSandboxBackendFields.DEFAULT_EXEC_TIMEOUT_S,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run *args*, blocking for at most timeout + unkillable_child_grace_s
        even against a child stuck in uninterruptible kernel sleep that a
        plain subprocess.run(timeout=...) would hang on forever (see
        run_with_unkillable_child_grace()'s docstring for why).

        _get_docker_semaphore()'s slot is held for the call's duration and
        released as soon as we give up waiting -- a wedged call must not also
        starve every OTHER sandbox's ability to make a docker/podman call
        (see this module's docstring for the semaphore's shared-pool
        rationale).
        """
        operation = args[1] if len(args) > 1 else "unknown"
        storage = getattr(self, "_checkpoint_storage", None)
        if storage is not None and args[0] == self._runtime:
            if not storage.prepared:
                # Never let cleanup or a pre-start inspect contact the shared
                # Podman store under a private sandbox identity.
                return subprocess.CompletedProcess(args, 1, b"", b"Private storage not provisioned")
            args = storage.prefix + args[1:]
        sem = _get_docker_semaphore()
        with agprof.span("sync:container"):
            sem.acquire()
        released = False

        def _release_once() -> None:
            nonlocal released
            if not released:
                released = True
                sem.release()

        try:
            # Excludes semaphore acquisition; includes CLI/daemon response time.
            # Record only the operation verb, never command payloads or credentials.
            with agprof.span(f"runtime:container_call:{operation}"):
                return run_with_unkillable_child_grace(
                    lambda: subprocess.run(
                        args, input=input, capture_output=True, timeout=timeout, check=check
                    ),
                    args=args,
                    timeout=timeout,
                    grace_s=self._agconfig.sandbox.unkillable_child_grace_s,
                    on_give_up=_release_once,
                )
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode("utf-8", errors="replace").strip()
            msg = f"{' '.join(args)} failed (exit {e.returncode})"
            if err:
                msg += f": {err}"
            raise RuntimeError(msg) from e
        finally:
            _release_once()

    def _rm_container(self, name: str) -> None:
        """Force-remove a container by name. Raises on failure."""
        self._run(
            [self._runtime, "rm", "-f", name],
            check=True,
            timeout=self._agconfig.sandbox.docker_rm_timeout_s,
        )

    def _rmi(self, image_ref: str, *, force: bool = False) -> None:
        """Remove an image by ID or tag. Raises on failure."""
        cmd = [self._runtime, "rmi"]
        if force:
            cmd.append("-f")
        cmd.append(image_ref)
        self._run(cmd, check=True, timeout=self._agconfig.sandbox.image_timeout_s)

    def _container_exec(
        self,
        sh_cmd: str,
        workdir: str = "/workspace",
        timeout: int = AgSandboxBackendFields.DEFAULT_EXEC_TIMEOUT_S,
        stdin: bytes | None = None,
        shell: str = "bash",
    ) -> tuple[str, int]:
        """Run a raw shell command inside the container."""
        self._ensure_started()
        return self._container_exec_started(
            sh_cmd, workdir=workdir, timeout=timeout, stdin=stdin, shell=shell
        )

    def _container_exec_started(
        self,
        sh_cmd: str,
        workdir: str = "/workspace",
        timeout: int = AgSandboxBackendFields.DEFAULT_EXEC_TIMEOUT_S,
        stdin: bytes | None = None,
        shell: str = "bash",
    ) -> tuple[str, int]:
        """Same as _container_exec() but skips _ensure_started() -- only for
        use by _ensure_started() itself (via _snapshot_pids_started()) while
        it establishes the baseline PID snapshot on a container it has
        already just confirmed or just created. Going through the public
        _container_exec() there would call _ensure_started() again and
        recurse without end, since there is no self._started flag to
        short-circuit that re-entry."""
        args = [self._runtime, "exec"]
        if stdin is not None:
            args.append("-i")
        args += ["-w", workdir, self._container_name(), shell, "-c", sh_cmd]
        try:
            proc = self._run(args, input=stdin, timeout=timeout)
            output = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
            return output, proc.returncode
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s", -1
        except Exception as e:
            return str(e), -1

    def _container_exec_detached(
        self,
        sh_cmd: str,
        workdir: str = "/workspace",
        shell: str = "bash",
    ) -> None:
        """Launch sh_cmd inside the container (`docker/podman exec -d`) and
        return as soon as it's registered, without waiting for it to finish
        -- for starting a long-lived in-container process (an
        adapters/native.py react-loop entrypoint, or a
        container-relocated agproxy_llm) that the caller reaches afterward
        over its own bridge (a bind-mounted UDS -- see agsandbox.py's
        `_agharness_llm_gateway` mount), not via this call's stdout/exit
        code, unlike every other `_container_exec*` method here.

        Raises if the container itself couldn't be reached (a real launch
        failure), but has no way to know if sh_cmd's own process later
        crashes -- detecting that is the caller's job (e.g. a heartbeat over
        the bridge), not this method's, exactly as `docker exec -d` itself
        offers no such feedback once the process is handed off.
        """
        self._ensure_started()
        if self._agconfig.sandbox.checkpoint_fast_resume:
            # The short exec parent exits and container init adopts the
            # daemon. CRIU then sees no live runtime exec session or host pipe.
            args = [
                self._runtime,
                "exec",
                "-w",
                workdir,
                self._container_name(),
                shell,
                "-c",
                f"nohup {shell} -c {shlex.quote(sh_cmd)} </dev/null >/dev/null 2>&1 &",
            ]
            self._run(args, check=True, timeout=self._agconfig.sandbox.exec_quick_timeout_s)
            return
        args = [
            self._runtime,
            "exec",
            "-d",
            "-w",
            workdir,
            self._container_name(),
            shell,
            "-c",
            sh_cmd,
        ]
        self._run(args, check=True, timeout=self._agconfig.sandbox.exec_quick_timeout_s)

    def update_limits(
        self,
        *,
        cpus: float | None = None,
        memory: str | None = None,
    ) -> None:
        """Live-update container CPU/memory limits."""
        if not self._container_running():
            return
        cmd = [self._runtime, "update"]
        if cpus is not None and self._cfs_supported():
            cmd.append(f"--cpus={cpus}")
        if memory is not None:
            cmd.append(f"--memory={memory}")
        if len(cmd) == 2:
            return  # nothing to update
        cmd.append(self._container_name())
        self._run(cmd, timeout=self._agconfig.sandbox.inspect_timeout_s)

    def _image_diff_ids(self, image_ref: str) -> "list[str]":
        """The image's uncompressed layer content digests, in order --
        matches `docker save`'s config.json `rootfs.diff_ids` exactly for
        an uncompressed-layer image (verified empirically during
        development: the digest docker inspect reports here for a plain
        `docker commit`-produced layer IS the blob's own filename under
        `docker save`'s `blobs/sha256/<digest>`, not a separately-computed
        compressed-layer digest)."""
        result = self._run(
            [self._runtime, "inspect", "--format={{json .RootFS.Layers}}", image_ref],
            check=True,
            timeout=self._agconfig.sandbox.stop_inspect_timeout_s,
        )
        return json.loads(result.stdout.decode("utf-8", errors="replace"))

    def _locate_layer_diff_dir(
        self, diff_id: str, *, diff_ids: "list[str] | None" = None
    ) -> "Path | None":
        """Find the on-disk diff directory backing *diff_id*, reused for
        free instead of re-deriving it via `docker diff`/`docker save`.
        None by default (runtime-specific; overridden by
        `_DockerBackend`/`_PodmanBackend`) falls back to `_squash_commit()`."""
        return None

    def _host_to_container_id(self, uid: int, gid: int) -> "tuple[int, int]":
        """Identity by default -- host and container ownership match except
        under rootless Docker/Podman, which override this."""
        return (uid, gid)

    def _reset_accumulator(self) -> None:
        if self._accumulator_dir is not None:
            shutil.rmtree(self._accumulator_dir, ignore_errors=True)
        self._accumulator_dir = None
        self._accumulated_diff_path = None
        self._accumulated_layer_count = 0

    def _fold_overlay_diff_into_accumulator(self, diff_dir: Path) -> None:
        """Append one overlay upper directory into the squash-time
        accumulator tar (creating the TMPDIR scratch dir on first use).
        Raises on failure -- caller is the lazy squash builder, which
        treats any error as "fall back to export/import".
        """
        if self._accumulator_dir is None:
            # A unique subdir of scratch/, not scratch/ itself --
            # _reset_accumulator() rmtree's this whole path, and scratch/
            # is shared by every sandbox in the run.
            self._accumulator_dir = Path(
                tempfile.mkdtemp(prefix="agency-accum-", dir=str(agency_run_scratch_dir()))
            )
        nonce = _uuid.uuid4().hex
        cycle_tar = self._accumulator_dir / f"cycle-{nonce}.tar"
        overlay_diff_to_tar(diff_dir, cycle_tar, uid_gid_translate=self._host_to_container_id)

        if self._accumulated_diff_path is None:
            self._accumulated_diff_path = cycle_tar
        else:
            new_accumulated = self._accumulator_dir / f"accum-{nonce}.tar"
            merge_layer_tars([self._accumulated_diff_path, cycle_tar], new_accumulated)
            self._accumulated_diff_path.unlink(missing_ok=True)
            cycle_tar.unlink(missing_ok=True)
            self._accumulated_diff_path = new_accumulated
        self._accumulated_layer_count += 1

    def _build_accumulator_for_squash(self, tag: str) -> None:
        """Lazily build the squash accumulator from every layer between
        the reference chain and *tag*'s current chain.

        Called only when a squash is due -- ordinary plain commits leave
        the lifecycle image alone and do not copy overlay uppers into
        TMPDIR. Under the hibernate model that image usually has exactly
        one tip layer past the reference (sibling commits replace each
        other); after skill-failure recreates the chain can be deeper,
        and each of those layers is folded here in one shot.

        Raises on any failure (missing reference prefix, unlocatable
        overlay diff, tar/merge error) so `commit()` falls back to
        `_squash_commit()` export/import. Temp files under
        `agency-accum-*` must be removed afterward via
        `_reset_accumulator()` (commit()'s squash block uses try/finally).
        """
        self._reset_accumulator()
        current_diff_ids = self._image_diff_ids(tag)
        if self._squash_base_diff_ids is not None:
            base_diff_ids = self._squash_base_diff_ids
        else:
            base_diff_ids = self._image_diff_ids(self._resolve_image(self._base_image))
        if (
            len(current_diff_ids) < len(base_diff_ids)
            or current_diff_ids[: len(base_diff_ids)] != base_diff_ids
        ):
            raise RuntimeError(
                f"reference chain is not a prefix of {tag}; cannot build squash accumulator"
            )
        expected_new_layers = len(current_diff_ids) - len(base_diff_ids)
        if expected_new_layers <= 0:
            raise RuntimeError(f"no layers beyond the reference chain to squash for {tag}")

        for i in range(len(base_diff_ids), len(current_diff_ids)):
            prefix = current_diff_ids[: i + 1]
            layer_digest = prefix[-1]
            diff_dir = self._locate_layer_diff_dir(layer_digest, diff_ids=prefix)
            if diff_dir is None:
                raise RuntimeError(
                    f"could not locate on-disk diff directory for layer {layer_digest} (tag {tag})"
                )
            self._fold_overlay_diff_into_accumulator(diff_dir)

        if (
            self._accumulated_diff_path is None
            or self._accumulated_layer_count != expected_new_layers
        ):
            raise RuntimeError(
                f"squash accumulator incomplete after build: tracked "
                f"{self._accumulated_layer_count} layers, expected {expected_new_layers}"
            )

    def _accumulator_squash_commit(self, tag: str) -> None:
        """Fast path: apply the lazily-built accumulator diff-tar
        directly onto the reference chain's own layers (referenced by
        digest only -- see `_layer_squash.build_save_archive()`, never
        touched) to produce the new squashed HEAD image. Confirmed
        empirically to complete in well under a second regardless of base
        image size.

        The accumulator must already have been built by
        `_build_accumulator_for_squash()` for this same *tag*. The
        reference chain is `self._squash_base_diff_ids` if this backend
        has already squashed successfully at least once (fast or
        fallback -- see below), else `self._base_image`'s own digests.
        This is what lets a sandbox recover fast-path eligibility after a
        `_squash_commit()` fallback: without it, the base image's digests
        stop being a prefix of the chain FOREVER the moment a fallback
        ever runs (export/import produces a parentless image, permanently
        disconnected from `self._base_image`'s lineage), forcing every
        future squash to fall back too -- confirmed as a real production
        issue (repeated tens-of-GB export/imports for a long-running
        sandbox whose per-commit diffs were themselves large enough that
        a single lookup hiccup was plausible). Re-baselining against
        whatever the last squash actually produced sidesteps that: the
        very next squash gets a fresh reference and a fresh accumulator,
        so a one-time hiccup doesn't compound into permanent degradation.

        Raises if the accumulator can't be trusted for this squash --
        e.g. `_accumulated_layer_count` doesn't match the real gap
        between the current checkpoint and the reference chain, or the
        reference chain doesn't prefix the current chain. Callers must
        catch and fall back to `_squash_commit()` -- this method never
        silently produces a possibly-wrong image. Does NOT clear the
        accumulator temp dir; `commit()`'s squash finally-block always
        does that so a failed fast path can't leave multi-GB tars behind.
        """
        if self._accumulated_diff_path is None:
            raise RuntimeError("no checkpoint diff accumulator available")

        if self._squash_base_diff_ids is not None:
            base_diff_ids = self._squash_base_diff_ids
        else:
            resolved_base = self._resolve_image(self._base_image)
            base_diff_ids = self._image_diff_ids(resolved_base)
        current_diff_ids = self._image_diff_ids(tag)
        expected_new_layers = len(current_diff_ids) - len(base_diff_ids)
        if (
            expected_new_layers < 0
            or current_diff_ids[: len(base_diff_ids)] != base_diff_ids
            or self._accumulated_layer_count != expected_new_layers
        ):
            raise RuntimeError(
                f"checkpoint diff accumulator tracks {self._accumulated_layer_count} layers, "
                f"expected {expected_new_layers} -- refusing to trust it"
            )

        image_info = json.loads(
            self._run(
                [self._runtime, "inspect", tag],
                check=True,
                timeout=self._agconfig.sandbox.stop_inspect_timeout_s,
            ).stdout.decode("utf-8", errors="replace")
        )[0]

        merged_digest = sha256_file(self._accumulated_diff_path)
        new_diff_ids = base_diff_ids + [f"sha256:{merged_digest}"]
        new_config = {
            "config": image_info.get("Config", {}),
            "architecture": image_info.get("Architecture", "amd64"),
            "os": image_info.get("Os", "linux"),
            "rootfs": {"type": "layers", "diff_ids": new_diff_ids},
            "history": [
                {"created": "1970-01-01T00:00:00Z", "comment": "agency checkpoint"}
                for _ in new_diff_ids
            ],
        }
        new_config_bytes = json.dumps(new_config).encode()
        new_config_digest = hashlib.sha256(new_config_bytes).hexdigest()

        tmp_dir = Path(tempfile.mkdtemp(prefix="agency-squash-", dir=str(agency_run_scratch_dir())))
        try:
            out_tar_path = tmp_dir / "out.tar"
            build_save_archive(
                out_tar_path,
                base_layer_digests=base_diff_ids,
                merged_blob_path=self._accumulated_diff_path,
                merged_blob_digest=merged_digest,
                config_bytes=new_config_bytes,
                config_digest=new_config_digest,
                tag=tag,
            )
            self._run(
                [self._runtime, "load", "-i", str(out_tar_path)],
                check=True,
                timeout=self._agconfig.sandbox.squash_timeout_s,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Re-baseline: the NEXT squash validates/builds against this
        # squash's own resulting chain, not self._base_image -- see this
        # method's docstring. Accumulator temp cleanup is left to
        # commit()'s squash finally-block so a failed fast path can't
        # leave multi-GB tars behind either.
        self._squash_base_diff_ids = new_diff_ids

    def _squash_commit(self, tag: str) -> None:
        """Flatten the container's filesystem into a brand-new single-layer
        image, resetting layer depth to 1 before the chain crosses the
        runtime's hard layer-depth cap. Re-applies
        `_AGENCY_OWNER_PID_LABEL` via `--change` since export/import drops
        all container labels/config. Re-baselines `self._squash_base_diff_ids`
        on success so the next squash can use the faster diff-only path."""
        export_result = self._run(
            [self._runtime, "export", self._container_name()],
            check=True,
            timeout=self._agconfig.sandbox.squash_timeout_s,
        )
        self._run(
            [
                self._runtime,
                "import",
                "--change",
                f"LABEL {_AGENCY_OWNER_PID_LABEL}={self._owner_pid}",
                "-",
                tag,
            ],
            input=export_result.stdout,
            check=True,
            timeout=self._agconfig.sandbox.squash_timeout_s,
        )
        try:
            self._squash_base_diff_ids = self._image_diff_ids(tag)
        except Exception as _e:
            # DATACOLLECTOR: append, agname=self._agname -- best-effort failure, low priority.
            print(
                f"[agsandbox_backend] WARNING: could not record post-squash chain "
                f"for tag {tag}, next squash will fall back to export/import again: {_e}",
                file=__import__("sys").stderr,
                flush=True,
            )
            self._squash_base_diff_ids = None
        self._reset_accumulator()

    def stop(self) -> None:
        from .checkpoint import CowZfsCheckpoint

        checkpointer = getattr(self, "_checkpointer", None)
        if isinstance(checkpointer, CowZfsCheckpoint):
            checkpointer.checkpoint(self)
            return
        self._stop_image()

    def _stop_image(self) -> None:
        """Hibernate the container: `docker/podman stop` it WITHOUT removing
        it, releasing the runtime slot (the session-keyring-derived
        concurrency semaphore) AND the GPU. The container object and its
        writable layer stay intact, so the next _ensure_started() can
        `docker/podman start` it straight back into the exact same state:
        no commit, no image, no `run` involved at all.

        Releasing the GPU here (rather than holding it for the container's
        whole life) is safe ONLY because `_gpu_flags()` attaches EVERY GPU
        on the host to every container at `run` time, not just the one
        currently assigned -- so a resumed container can be handed a
        *different* physical GPU than it had before hibernating without
        ever needing to be recreated (`docker/podman start` can't change a
        container's device attachment; neither runtime supports hot-
        attaching one). The actual restriction to one GPU at a time is
        therefore purely `CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`
        (base.py's exec()) -- see `_gpu_flags()`'s docstring for the
        isolation trade-off this accepts.

        Use rm_container() instead when the container's current state must
        be discarded outright, and commit() to checkpoint the current state
        into a lifecycle image -- either way, without removing the
        container itself.
        """
        gpu_ids_to_release = list(self._gpu_ids) if self._gpu_count_requested > 0 else []
        if not self._container_running():
            agprof.container_stopped(self._prof_container_label())
            return
        # Clear PID tracking — stopping kills every process inside, same as rm.
        self._watched_pids = {}
        self._baseline_pids = None  # force a fresh capture on the next _ensure_started()
        self._ptrace_managed_pids = set()
        self._daemon_pids = set()  # PIDs do not identify daemons across a restart.
        self._infrastructure_pids = {}
        name = self._container_name()
        stop_exc: Exception | None = None
        try:
            self._run(
                [
                    self._runtime,
                    "stop",
                    "-t",
                    str(self._agconfig.sandbox.docker_stop_grace_s),
                    name,
                ],
                check=True,
                timeout=self._agconfig.sandbox.docker_stop_timeout_s,
            )
        except Exception as _e:
            stop_exc = _e
        # Only release once actually confirmed not running -- a failed stop
        # may still leave the container (and the slot/GPU it holds) alive.
        if not self._container_running():
            agprof.container_stopped(self._prof_container_label())
            self._release_runtime_slot()
            if gpu_ids_to_release and self._gpu_release_fn is not None:
                self._gpu_release_fn(gpu_ids_to_release)
                self._gpu_ids = []
        if stop_exc is not None:
            raise stop_exc

    def _mark_runtime_checkpoint_stopped(self) -> None:
        """Release host bookkeeping after CRIU stopped the container."""
        from .checkpoint import CheckpointCapabilityError

        if self._container_running():
            raise CheckpointCapabilityError("CRIU checkpoint left the container running")
        gpu_ids_to_release = list(self._gpu_ids) if self._gpu_count_requested > 0 else []
        self._watched_pids = {}
        self._baseline_pids = None
        self._ptrace_managed_pids = set()
        self._daemon_pids = set()
        self._infrastructure_pids = {}
        agprof.container_stopped(self._prof_container_label())
        self._release_runtime_slot()
        if gpu_ids_to_release and self._gpu_release_fn is not None:
            self._gpu_release_fn(gpu_ids_to_release)
            self._gpu_ids = []

    def rm_container(self) -> None:
        """Force-remove the container outright, discarding all of its
        current state and releasing both the runtime slot and the GPU (see
        stop()'s docstring for why the GPU is only released here, never on
        a mere hibernate). Raises if removal fails after retrying -- an
        unconfirmed removal means the container, and whatever resources it
        holds, may still be alive, which the caller must not silently
        ignore.

        Does not touch self._checkpoint_image: the next _ensure_started()
        recreates fresh from it (or from self._base_image if none exists
        yet), which is exactly what makes this the discard/revert
        primitive -- call it without a preceding commit() to throw away
        everything since the last checkpoint.

        Safe to call when there is no container at all (never started, or
        already removed) -- a no-op in that case, aside from a GPU release
        if one was still held.
        """
        from .checkpoint import CowZfsCheckpoint

        checkpointer = getattr(self, "_checkpointer", None)
        if (
            isinstance(checkpointer, CowZfsCheckpoint)
            and checkpointer.latest is not None
            and not getattr(self, "_checkpoint_destroying", False)
        ):
            # For local COW storage, discard means roll back on next use.
            # Keep the runtime object because its metadata is part of the
            # same private dataset as the writable filesystem and CRIU cache.
            if self._container_running():
                self._stop_image()
            checkpointer.hibernated = True
            return
        gpu_ids_to_release = list(self._gpu_ids) if self._gpu_count_requested > 0 else []
        if not self._container_status():
            agprof.container_stopped(self._prof_container_label())
            if gpu_ids_to_release and self._gpu_release_fn is not None:
                self._gpu_release_fn(gpu_ids_to_release)
                self._gpu_ids = []
            return
        # Captured BEFORE the rm attempt: this is what tells us whether the
        # runtime slot is actually ours to release below. Called on an
        # already-hibernating container (stop() already ran and already
        # released the slot -- the normal case for a skill-failure teardown,
        # since every prior tool call's stop() already hibernated it), this
        # is False, and the release check below correctly no-ops instead of
        # crediting the semaphore a second time. Without this gate, a
        # multiprocessing.Semaphore silently over-releases past its true
        # capacity (confirmed empirically: release() raises no ValueError on
        # over-release, unlike threading.BoundedSemaphore), letting more
        # containers run concurrently than the kernel keyring quota actually
        # supports -- the same class of bug the quota exists to prevent.
        had_container = self._container_running()
        self._watched_pids = {}
        self._baseline_pids = None
        self._ptrace_managed_pids = set()
        self._daemon_pids = set()
        self._infrastructure_pids = {}
        name = self._container_name()
        # rm_exc is raised at the end, after the release checks below still
        # run -- an unconfirmed removal must reach the caller, but shouldn't
        # cut short a release that's still legitimately possible to check.
        rm_exc: Exception | None = None
        for _attempt in range(self._agconfig.sandbox.rm_retry_attempts):
            try:
                self._rm_container(name)
                rm_exc = None
                break
            except Exception as _e:
                rm_exc = _e
                if _attempt != self._agconfig.sandbox.rm_retry_attempts - 1:
                    time.sleep(self._agconfig.sandbox.rm_retry_backoff_s)
        if had_container and not self._container_running():
            self._release_runtime_slot()
        if (
            gpu_ids_to_release
            and self._gpu_release_fn is not None
            and not self._container_running()
        ):
            self._gpu_release_fn(gpu_ids_to_release)
            self._gpu_ids = []
        # A runtime client can raise even when the daemon completed removal;
        # drop stale attribution whenever removal succeeded or the container
        # is independently confirmed no longer running. Keep it registered if
        # a failed rm left the workload alive.
        if rm_exc is None or not self._container_running():
            agprof.container_stopped(self._prof_container_label())
        if rm_exc is not None:
            raise rm_exc

    def commit(self, tag: "str | None" = None) -> bool:
        return self.checkpoint(tag) is not None

    def checkpoint(self, tag: "str | None" = None):
        from .checkpoint import ImageCommitCheckpoint

        checkpointer = getattr(self, "_checkpointer", None) or ImageCommitCheckpoint()
        return checkpointer.checkpoint(self, tag)

    def _normalized_checkpoint_manifest(self) -> dict:
        image = json.loads(
            self._run(
                [self._runtime, "image", "inspect", self._resolve_image(self._base_image)],
                check=True,
            ).stdout
        )[0]
        return {
            "schema_version": 1,
            "logical_sandbox_id": self._name,
            "runtime": self._runtime,
            "base_image_reference": self._base_image,
            "base_image_id": image["Id"],
            "platform": {
                "architecture": image.get("Architecture"),
                "os": image.get("Os"),
            },
            # Kept only in memory. Flags may contain configured environment
            # secrets and must never be copied into profiler annotations.
            "launch_flags": tuple(self._agconfig.sandbox.flags),
            "mounts": self._checkpoint_mounts,
            "resources": {
                "cpus": self._agconfig.resources.idle_cpus,
                "memory": self._agconfig.resources.idle_memory,
                "cpuset_cpus": self._agconfig.sandbox.cpuset_cpus,
                "cpuset_mems": self._agconfig.sandbox.cpuset_mems,
            },
            "command": ("tail", "-f", "/dev/null"),
        }

    def delete_checkpoint(self, checkpoint) -> None:
        self._checkpointer.delete_checkpoint(self, checkpoint)

    def _commit_image(self, tag: "str | None" = None) -> bool:
        """Checkpoint the container's current filesystem into a lifecycle
        image WITHOUT removing the container -- it keeps running (or stays
        hibernating) so the next tool call or skill resumes directly from
        it, with no `docker/podman run` needed either way.

        Squashing is fully automatic: triggered purely by the chain's
        actual current depth (checkpoint_squash_max_depth), checked right
        after this commit succeeds -- never by a fixed commit count, and
        never forced. A count can't account for how many layers the base
        image itself already consumes (a real base image was observed at
        80 layers on its own), so a count-based interval could let the real
        depth cross the runtime's actual cap before the interval ever
        fired -- exactly what caused a real "max depth exceeded" failure on
        an ordinary plain commit. `_image_diff_ids()` is a cheap inspect
        (cost independent of image size), not a second commit.

        Returns False if there's no container to commit at all (e.g. a
        skill that never touched the sandbox) -- otherwise True once the
        plain commit (step 1 below) has succeeded, regardless of whether a
        squash was due or how it went. Raises if the plain commit itself
        fails after retrying; a squash failure only warns, since the
        checkpoint from step 1 already succeeded either way.
        """
        if not self._container_status():
            return False
        tag = tag if tag is not None else self._lifecycle_tag()

        # 0. Capture whatever image *tag* currently points to, before the
        #    plain commit below moves it -- under the hibernate model this
        #    container is never recreated between successful commits, so
        #    `docker/podman commit` always diffs against the container's
        #    fixed run-time ancestor, not against the previous commit's
        #    image (confirmed empirically: two consecutive commits with no
        #    recreate in between produce SIBLING images of identical depth,
        #    each containing the full cumulative diff -- never a growing
        #    chain). That means the previous commit's image is never a
        #    parent of this one, so once the tag moves off it, it's
        #    immediately, safely deletable. Best-effort: a
        #    failure to even look this up just means one image doesn't get
        #    cleaned up this cycle, not that the commit itself is at risk.
        previous_image_id: str | None = None
        try:
            _prev_result = self._run(
                [self._runtime, "inspect", "--format={{.Id}}", tag],
                check=False,
                timeout=self._agconfig.sandbox.stop_inspect_timeout_s,
            )
            if _prev_result and _prev_result.returncode == 0:
                previous_image_id = (
                    _prev_result.stdout.decode("utf-8", errors="replace").strip() or None
                )
        except Exception as _e:
            # DATACOLLECTOR: append, agname=self._agname -- best-effort cleanup lookup failure.
            print(
                f"[agsandbox_backend] WARNING: could not inspect existing image for tag "
                f"{tag} before commit: {_e}",
                file=__import__("sys").stderr,
                flush=True,
            )

        # 1. Always do the normal, fast plain commit first -- unconditionally,
        #    regardless of whether a squash turns out to be due. Squashing
        #    (below) is a separate, additional step layered on top when
        #    due, never a replacement for it, so an ordinary commit's cost
        #    never regresses.
        commit_exc: Exception | None = None
        diagnostic = None
        if self._agconfig.sandbox.checkpoint_diagnostics:
            from .checkpoint_diagnostics import collect_before, write_after

            diagnostic = collect_before(self)
        commit_started = time.perf_counter() if diagnostic is not None else None
        for _attempt in range(self._agconfig.sandbox.commit_retry_attempts):
            try:
                self._run(
                    [self._runtime, "commit", self._container_name(), tag],
                    check=True,
                    timeout=self._agconfig.sandbox.commit_timeout_s,
                )
                self._checkpoint_image = tag
                commit_exc = None
                break
            except Exception as _e:
                commit_exc = _e
                if _attempt != self._agconfig.sandbox.commit_retry_attempts - 1:
                    time.sleep(self._agconfig.sandbox.commit_retry_backoff_s)
        if diagnostic is not None:
            commit_seconds = time.perf_counter() - commit_started
            write_after(self, diagnostic, tag, commit_seconds, commit_exc)
        if commit_exc is not None:
            raise commit_exc

        # 1a. Clean up the image *tag* pointed at before this commit moved
        #     it -- see step 0's docstring for why this is always safe now
        #     (never a parent of the new commit), unlike squash cleanup
        #     below which only ever applied on a squash cycle. Guarded by
        #     an ancestor check the same way: a fork may have tagged this
        #     exact image and already be running a container from it, in
        #     which case it's left alone. Best-effort -- a stray dangling
        #     image costs disk space, not correctness.
        if previous_image_id:
            try:
                _in_use = self._run(
                    [
                        self._runtime,
                        "ps",
                        "-a",
                        "--filter",
                        f"ancestor={previous_image_id}",
                        "--format",
                        "{{.ID}}",
                    ],
                    check=False,
                    timeout=self._agconfig.sandbox.stop_ps_check_timeout_s,
                )
                if _in_use and _in_use.stdout.strip():
                    pass  # a container (e.g. a fork) still runs from this image — leave it
                else:
                    self._rmi(previous_image_id)
            except Exception as _e:
                # DATACOLLECTOR: append, agname=self._agname -- best-effort cleanup failure, low priority.
                print(
                    f"[agsandbox_backend] WARNING: could not check/delete previous image "
                    f"{previous_image_id} for tag {tag}: {_e}",
                    file=__import__("sys").stderr,
                    flush=True,
                )

        # 1b. Squash is triggered purely by the chain's actual current depth
        #     (checkpoint_squash_max_depth) -- not by a fixed commit count:
        #     a count can't account for how many layers the base image
        #     itself already consumes (a real base image was observed at 80
        #     layers on its own), so a count-based interval could let the
        #     real depth cross the runtime's actual cap before the interval
        #     ever fired. `_image_diff_ids()` is a cheap inspect (cost
        #     independent of image size), not a second commit.
        should_squash = False
        try:
            should_squash = (
                len(self._image_diff_ids(tag)) >= self._agconfig.sandbox.checkpoint_squash_max_depth
            )
        except Exception as _e:
            # DATACOLLECTOR: append, agname=self._agname -- degrade warning (squash may run more often than intended).
            print(
                f"[agsandbox_backend] WARNING: could not check chain depth for tag "
                f"{tag}, skipping this cycle's squash check: {_e}",
                file=__import__("sys").stderr,
                flush=True,
            )

        # 2. Ordinary commits keep only the lifecycle image (step 1 / 1a) --
        #    no TMPDIR overlay-diff copy. The accumulator is built lazily
        #    below, only when a squash is actually due.
        if should_squash:
            # 3. A squash is due -- perform it as an ADDITIONAL step now, on
            #    top of the commit that just succeeded above. old_image_id
            #    is the FULL chain's own image ID (base + every commit
            #    including the one just above) -- everything the new
            #    squashed image is about to make obsolete. A plain commit's
            #    result is ALWAYS a child layer of whatever it replaces, so
            #    the runtime would refuse to delete it ("has dependent child
            #    images"); only a squash's result -- rebuilt directly on the
            #    shared base, whether via the accumulator fast path or
            #    _squash_commit()'s parentless export/import fallback --
            #    makes the old chain's deletion possible, which is why this
            #    lookup only happens on a squash cycle.
            old_image_id: str | None = None
            try:
                result = self._run(
                    [self._runtime, "inspect", "--format={{.Id}}", tag],
                    check=False,
                    timeout=self._agconfig.sandbox.stop_inspect_timeout_s,
                )
                if result and result.returncode == 0:
                    old_image_id = result.stdout.decode("utf-8", errors="replace").strip() or None
            except Exception as _e:
                # DATACOLLECTOR: append, agname=self._agname -- best-effort cleanup lookup failure.
                print(
                    f"[agsandbox_backend] WARNING: could not inspect existing image for tag {tag}: {_e}",
                    file=__import__("sys").stderr,
                    flush=True,
                )
            try:
                # Lazy: materialize the overlay-diff tar(s) for layers
                # since the reference chain, fast-squash from that, and
                # always delete the temp dir afterward -- even when the
                # fast path fails and we fall back to export/import.
                try:
                    self._build_accumulator_for_squash(tag)
                    self._accumulator_squash_commit(tag)
                except Exception as _fast_e:
                    # DATACOLLECTOR: append, agname=self._agname -- degrade warning, falling back to slower squash path.
                    print(
                        f"[agsandbox_backend] WARNING: fast squash path failed for "
                        f"tag {tag}, falling back to export/import: {_fast_e}",
                        file=__import__("sys").stderr,
                        flush=True,
                    )
                    try:
                        self._squash_commit(tag)
                    except Exception as _e:
                        # Best-effort: the checkpoint itself (step 1) already
                        # succeeded -- a squash failure just means the layer
                        # chain keeps growing until the next attempt, not that
                        # this cycle's checkpoint is lost.
                        # DATACOLLECTOR: append, agname=self._agname -- ongoing degradation signal (layer chain growing).
                        print(
                            f"[agsandbox_backend] WARNING: squash failed for tag {tag}, "
                            f"layer chain will keep growing until the next attempt: {_e}",
                            file=__import__("sys").stderr,
                            flush=True,
                        )
            finally:
                self._reset_accumulator()
            # Delete the previous image now that the tag points to the new
            # one -- a plain commit's result can never actually free its own
            # parent, only a squash's result can. Only delete if no
            # containers are currently using it -- a fork may still be
            # running from the same image (the fork's own commit()/
            # rm_container() will delete it once its container is gone).
            # This backend's own container is virtually always still alive
            # at this point too, but its `run`-time ancestor is whatever it
            # was originally created from, not this just-superseded
            # intermediate commit -- so it never matches here. Best-effort:
            # a stray dangling image costs disk space, not correctness.
            if old_image_id and self._checkpoint_image == tag:
                try:
                    in_use = self._run(
                        [
                            self._runtime,
                            "ps",
                            "-a",
                            "--filter",
                            f"ancestor={old_image_id}",
                            "--format",
                            "{{.ID}}",
                        ],
                        check=False,
                        timeout=self._agconfig.sandbox.stop_ps_check_timeout_s,
                    )
                    if in_use and in_use.stdout.strip():
                        pass  # containers still running from this image — leave it
                    else:
                        self._rmi(old_image_id)
                except Exception as _e:
                    # DATACOLLECTOR: append, agname=self._agname -- best-effort cleanup failure, low priority.
                    print(
                        f"[agsandbox_backend] WARNING: could not check/delete old image {old_image_id}: {_e}",
                        file=__import__("sys").stderr,
                        flush=True,
                    )
        return True

    def restore(self, tag: str) -> None:
        from .checkpoint import CheckpointHandle, ImageCommitCheckpoint

        checkpointer = getattr(self, "_checkpointer", None) or ImageCommitCheckpoint()
        if isinstance(tag, CheckpointHandle):
            if tag.backend != self._agconfig.sandbox.checkpoint_backend:
                raise ValueError("Checkpoint backend mismatch")
            checkpointer.restore(self, tag)
        else:
            if self._agconfig.sandbox.checkpoint_backend != "image_commit":
                raise ValueError("cow_zfs restore requires a CheckpointHandle")
            checkpointer.restore(
                self, CheckpointHandle("image_commit", self._runtime, self._name, tag)
            )

    def _restore_image(self, tag: str) -> None:
        """Restore the sandbox to a previously committed image snapshot.

        Stops the running container (killing watched pids first), then restarts
        from *tag*.  The image is kept so it can be reused on subsequent
        failures during the same skill run.
        """
        if self._container_running():
            if self._watched_pids:
                pids = " ".join(str(p) for p in self._watched_pids)
                try:
                    self._container_exec(
                        f"kill {pids} 2>/dev/null; true",
                        timeout=self._agconfig.sandbox.exec_quick_timeout_s,
                        shell="sh",
                    )
                except Exception as _e:
                    # DATACOLLECTOR: append, agname=self._name -- best-effort courtesy signal, low priority.
                    print(
                        f"[agsandbox_backend] WARNING: failed to kill PIDs {pids} in {self._name} during restore: {_e}"
                    )
            self.rm_container()
        self._checkpoint_image = tag
        self._ensure_started()

    def destroy(self) -> None:
        if self._destroyed:
            return
        if getattr(self, "_checkpoint_storage", None) is not None:
            self._checkpoint_destroying = True
            try:
                if self._runtime == "docker":
                    with agprof.span("checkpoint.cleanup"):
                        self._checkpoint_storage.destroy()
                    self.rm_container()
                    self._checkpoint_storage.prepared = False
                    self._checkpoint_storage.dataset = None
                else:
                    self.rm_container()
                    with agprof.span("checkpoint.cleanup"):
                        self._checkpoint_storage.destroy()
                self._destroyed = True
            finally:
                self._checkpoint_destroying = False
            return
        container_name = self._container_name()

        # Best-effort courtesy signal before rm_container() forces the issue
        # -- rm -f kills everything inside the container regardless of
        # whether this succeeds, so a failure here doesn't change what
        # actually happens, only whether tracked processes got a chance to
        # react first. Warn, don't raise: nothing depends on this
        # succeeding. Kept as destroy()'s own step (not something
        # rm_container() does) since destroy() is called from
        # atexit/__del__ on sandboxes that may never have gone through a
        # normal stop() first -- rm_container() alone can't assume anything
        # already gave tracked processes this courtesy.
        if self._watched_pids:
            pids = " ".join(str(p) for p in self._watched_pids)
            try:
                self._container_exec(
                    f"kill {pids} 2>/dev/null; true",
                    timeout=self._agconfig.sandbox.exec_quick_timeout_s,
                    shell="sh",
                )
            except Exception as _e:
                # DATACOLLECTOR: append, agname via container_name -- best-effort courtesy signal, low priority.
                print(
                    f"[agsandbox_backend] WARNING: failed to kill PIDs {pids} in {container_name}: {_e}"
                )

        # rm_container() already retries and only releases the runtime
        # slot/GPU once actually confirmed gone (see its docstring) --
        # destroy() reuses that logic rather than keeping its own copy,
        # since two copies of this logic risk drifting out of sync when
        # only one gets a fix.
        # rm_exc is raised at the end, after the image/accumulator cleanup
        # below has still been attempted -- an unconfirmed removal must
        # reach the caller, but shouldn't cut short cleanup that doesn't
        # depend on it.
        rm_exc: Exception | None = None
        try:
            self.rm_container()
        except Exception as _e:
            rm_exc = _e

        # Remove the checkpoint image created during this sandbox's
        # lifetime. Best-effort: a stray dangling image costs disk space,
        # not correctness, so this warns rather than raising.
        if self._checkpoint_image:
            try:
                self._rmi(self._checkpoint_image, force=True)
            except Exception as _e:
                # DATACOLLECTOR: append, agname via container_name -- best-effort cleanup failure, low priority.
                print(
                    f"[agsandbox_backend] WARNING: checkpoint image cleanup failed for {container_name}: {_e}"
                )
            self._checkpoint_image = None

        if self._accumulator_dir is not None:
            shutil.rmtree(self._accumulator_dir, ignore_errors=True)
            self._accumulator_dir = None
            self._accumulated_diff_path = None

        if rm_exc is not None:
            raise rm_exc
        self._destroyed = True

    def _lifecycle_tag(self) -> str:
        return f"agency/lifecycle-{self._name}".lower()

    # ------------------------------------------------------------------
    # Static helpers — image-level operations used for checkpointing.
    # These operate on image tags, not containers, so they don't need
    # a backend instance.
    # ------------------------------------------------------------------

    @staticmethod
    def tag_image(source: str, dest: str) -> None:
        """Retag an image from *source* to *dest* (docker/podman tag)."""
        runtime = get_container_runtime()
        with _docker_semaphore_slot():
            subprocess.run(
                [runtime, "tag", source, dest],
                capture_output=True,
                check=True,
            )

    @staticmethod
    def delete_image(tag: str, *, force: bool = False) -> None:
        """Remove an image by tag (docker/podman rmi).  Never raises."""
        runtime = get_container_runtime()
        cmd = [runtime, "rmi"]
        if force:
            cmd.append("-f")
        cmd.append(tag)
        with _docker_semaphore_slot():
            subprocess.run(cmd, capture_output=True)

    @staticmethod
    def export_image(tag: str, timeout: int) -> bytes:
        """Export an image to a tar byte-string (docker/podman save).

        The image must already exist.  Returns the raw tar bytes suitable
        for writing to a file or embedding in a larger archive.
        Raises ``subprocess.CalledProcessError`` on failure.
        """
        runtime = get_container_runtime()
        with _docker_semaphore_slot():
            result = subprocess.run(
                [runtime, "save", tag],
                capture_output=True,
                check=True,
                timeout=timeout,
            )
        return result.stdout

    @staticmethod
    def import_image(image_bytes: bytes, timeout: int) -> None:
        """Load an image from a tar byte-string (docker/podman load).

        Raises ``subprocess.CalledProcessError`` on failure.
        """
        runtime = get_container_runtime()
        with _docker_semaphore_slot():
            subprocess.run(
                [runtime, "load"],
                input=image_bytes,
                capture_output=True,
                check=True,
                timeout=timeout,
            )

    @staticmethod
    def relabel_owner_pid(tag: str, owner_pid: "int | None", timeout: int) -> None:
        """Overwrite *tag*'s _AGENCY_OWNER_PID_LABEL (clearing it if
        *owner_pid* is None).

        Used by agent.py's save()/load(): a checkpoint's embedded image
        was originally committed by (or, if it went through
        `_squash_commit()`'s export/import fallback, explicitly
        re-labelled with) the PID of whatever process happened to be
        running the sandbox at checkpoint time -- meaningless, and
        potentially misleading, once embedded in a portable .ckpt file
        that might be restored by an entirely different process, on a
        different host, at an arbitrary later time (that foreign PID
        could even coincidentally collide with a real, live, unrelated
        process on the restoring host). save() scrubs it (passing
        owner_pid=None) before embedding the image; load() re-stamps it
        with the actually-current restoring process's own PID afterward,
        so `reap_orphaned_containers()`'s image scan (which trusts this
        label to decide "is this image's owner still alive") never acts
        on stale, foreign evidence.

        `docker create` registers a container without ever starting or
        running it -- sufficient as `docker commit --change`'s source
        here, since the only thing being changed is metadata, not
        anything that requires the image to actually run. The temporary
        container is always removed, even on failure.
        """
        runtime = get_container_runtime()
        value = "" if owner_pid is None else str(owner_pid)
        with _docker_semaphore_slot():
            created = subprocess.run(
                [runtime, "create", tag],
                capture_output=True,
                check=True,
                timeout=timeout,
            )
            container_id = created.stdout.decode("utf-8", errors="replace").strip()
            try:
                subprocess.run(
                    [
                        runtime,
                        "commit",
                        "--change",
                        f"LABEL {_AGENCY_OWNER_PID_LABEL}={value}",
                        container_id,
                        tag,
                    ],
                    capture_output=True,
                    check=True,
                    timeout=timeout,
                )
            finally:
                subprocess.run([runtime, "rm", "-f", container_id], capture_output=True)
