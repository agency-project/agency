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
import shutil
import subprocess
import tempfile
import threading
import time
import uuid as _uuid
from pathlib import Path

from ..profiler import agprof
from ..agconfig import agConfig
from ..agresources import amd_render_node_paths_by_pci_bus, detect_gpus, _AgResourcePoolFields
from .base import AgSandboxBackendFields, agsandbox_backend, run_with_unkillable_child_grace
from ._layer_squash import merge_layer_tars, overlay_diff_to_tar, sha256_file, build_save_archive

# _RUN_ID is never read within this module itself -- it's defined here and
# imported by agsandbox_backends/__init__.py (which re-exports it for
# agsandbox.py's container-naming and a battery of tests/test_agterm.py
# cases). Declared here explicitly so static analysis recognizes it as an
# intentional export rather than a dead global.
__all__ = ["_RUN_ID"]


class _ContainerAlreadyRunning(Exception):
    """Raised by _run_with_conflict_retry when another process has already
    started the same container — the caller should reuse it."""


_RUNTIME: str | None = None

# Per-run ID so concurrent and successive runs never share container/image names.
# UUID avoids PID-reuse collisions and prevents stale lifecycle images from crashed
# runs being accidentally picked up by a new run that happens to get the same PID.
_RUN_ID = f"r{_uuid.uuid4().hex[:8]}"

# Limit the number of containers starting simultaneously.  Each concrete
# container backend construction acquires one slot for the duration of its
# startup sequence (docker/podman run + first exec). Without this, a burst of
# hundreds of parallel agent tasks overwhelms the daemon.  All docker/podman
# calls go through _run(), which holds this semaphore for the duration of each
# subprocess call.  Caps concurrent daemon calls at 16: the daemon serialises
# most operations internally (GPU init, overlay diff, container teardown), so
# more than ~16 concurrent calls increase contention without reducing
# wall-clock time.
# Tier-1 (global class) config: lazily created on first use so a caller can
# override the limit via agsandbox_backend.docker_semaphore_limit = N (or
# cfg.agsandbox_backend.docker_semaphore_limit = N before any backend exists)
# before the first docker/podman call in the process. Locked once actually
# read, matching a real semaphore's can't-resize-after-creation semantics.
_docker_semaphore: threading.Semaphore | None = None
_docker_semaphore_init_lock = threading.Lock()


def _get_docker_semaphore() -> threading.Semaphore:
    global _docker_semaphore
    if _docker_semaphore is None:
        with _docker_semaphore_init_lock:
            if _docker_semaphore is None:
                limit = AgSandboxBackendFields().docker_semaphore_limit
                _docker_semaphore = threading.Semaphore(limit)
    return _docker_semaphore


from contextlib import contextmanager as _contextmanager


@_contextmanager
def _docker_semaphore_slot():
    """Hold a docker/podman CLI slot; the span covers only the acquire wait."""
    sem = _get_docker_semaphore()
    with agprof.span("sync:container"):
        sem.acquire()
    try:
        yield
    finally:
        sem.release()


def _runtime_works(runtime: str) -> bool:
    try:
        proc = subprocess.run(
            [runtime, "info"],
            capture_output=True,
            timeout=AgSandboxBackendFields().inspect_timeout_s,
        )
        return proc.returncode == 0
    except Exception:
        return False


def get_container_runtime() -> str:
    """Return ``docker`` or ``podman``, preferring podman when both are usable."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    # Span covers only the uncached probe: one `docker info`/`podman info`
    # subprocess each, a one-time per-process cost of several hundred ms.
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
    """Return True if *pid* refers to a currently-running process on this host."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, just not signalable by us -- not expected for our own
        # sandbox containers' owner PIDs (always this same user), but
        # "exists" is the correct answer either way.
        return True
    return True


_reap_lock = threading.Lock()
_reap_done = False


def reap_orphaned_containers() -> None:
    """Force-remove containers (and their lifecycle images) left behind by a
    SIGKILL'd -- or otherwise uncleanly terminated -- previous process.
    Also sweeps lifecycle IMAGES whose owning container is already gone
    (see _reap_orphaned_lifecycle_images()) -- the container-based sweep
    above only ever helps if the owning container still exists (carrying
    its agency.owner_pid label) when this runs; once that container's
    already been removed by anything other than this reaper (a crash's
    container surviving to a later run, or e.g. external tooling deleting
    containers directly without ever invoking Python), its lifecycle image
    would otherwise be unreapable forever, since nothing else ever revisits it.

    SIGKILL can never be caught (see agwebui.run()'s SIGTERM handler for what
    *can* be done about a plain `kill`), so a SIGKILL'd process's containers
    just keep running in the daemon forever: nothing in that process's own
    lifecycle ever gets a chance to call destroy(), and -- per
    _AGENCY_OWNER_PID_LABEL's docstring above -- a later run doesn't
    naturally collide with (and thereby reclaim) their names either. This is
    the other half of that gap: actively look for containers whose owning
    PID is no longer alive and remove them.

    Runs at most once per process (guarded by _reap_lock/_reap_done) --
    called automatically the first time any _ContainerBackendBase is
    constructed (see its __init__), so a real run reaps stale state near its
    own start without every caller needing to remember to invoke this
    directly. Best-effort throughout: any failure is logged and swallowed,
    never allowed to block the actual work.
    """
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
        timeout=AgSandboxBackendFields().inspect_timeout_s,
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
        timeout=AgSandboxBackendFields().inspect_timeout_s,
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
                timeout=AgSandboxBackendFields().inspect_timeout_s,
            )
            if label_result.returncode != 0:
                continue
            owner_pid = int(label_result.stdout.decode("utf-8", errors="replace").strip())
        except Exception as _e:
            # The images list above was already filtered to label=..., so
            # reaching here means the tag vanished (e.g. removed by a
            # concurrent process) or its label value was somehow
            # unparseable -- rare. Unknown either way, never treat as dead.
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
                timeout=AgSandboxBackendFields().stop_ps_check_timeout_s,
            )
            if in_use.returncode != 0:
                continue  # couldn't confirm safety -- skip rather than risk it
            if in_use.stdout.strip():
                continue  # a container -- possibly a fresh owner reusing this tag -- is still running from it
        except Exception as _e:
            print(
                f"[agsandbox_backend] WARNING: could not confirm lifecycle image {tag!r} "
                f"is unused, skipping rather than risk deleting a live image: {_e}",
                file=__import__("sys").stderr,
                flush=True,
            )
            continue
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

    For a shared host mount (see ``agSandboxConfig.add_mount``) that would
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

    Attaches EVERY GPU on the host to EVERY container, regardless of
    whether reserve_gpu() has been called on the owning sandbox --
    `docker/podman run` is the only point these flags are ever set
    (neither runtime supports hot-attaching a device to an
    already-created container), so gating on `_gpu_virtual` at creation
    time meant a container first created (e.g. via a bash/read_file/
    write_file call) before reserve_gpu() ran was permanently stuck
    without device access for its entire lifetime -- reserve_gpu() could
    flip the sandbox's own flag, but nothing could retroactively attach
    the device to the already-running container. Attaching unconditionally
    removes that ordering dependency entirely: it no longer matters
    whether reserve_gpu() is called before or after the first exec().

    This is safe because CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES (set in
    base.py's exec(), readonly-exported so a command can't hijack a
    different GPU by reassigning the variable inline) is the ONLY access
    control point by design -- a sandbox that hasn't called reserve_gpu()
    gets "NoDevFiles" and sees no GPU regardless of what's attached at the
    container level. Nothing stops code running inside a container from
    opening another GPU's device node directly and ignoring the env var
    entirely; this was already true for any sandbox that HAD reserved a
    GPU even under the old gpu_reserved-gated design, so unconditional
    attachment does not weaken isolation for the trusted-code case this
    is meant for.

    NVIDIA: Docker ``--gpus all``; Podman ``--device nvidia.com/gpu=all``
      (CDI). Podman does not understand Docker's ``--gpus`` flag: it
      accepts it silently (no error) but never mounts the NVIDIA
      driver/devices, so a container started that way has zero GPU access
      despite `podman run` appearing to succeed -- `nvidia-smi` inside
      prints "WARNING: The NVIDIA Driver was not detected" and isn't even
      on PATH. This mirrors the identical fix applied to images/build.sh's
      own smoke tests.
    AMD: ``--device /dev/kfd`` (shared control device) plus every
      ``/dev/dri/renderD*`` node found -- there's no single "all" flag for
      ROCm the way ``--gpus all``/CDI ``=all`` covers NVIDIA, so each
      render node is attached explicitly.
    CPU-only hosts get no flags at all (no GPUs to attach).
    """
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
# is enforced across all worker processes (which run _ensure_started) and
# the main process (which calls stop/destroy), not just threads within one
# process.
def _keyring_container_limit() -> int:
    """Return the concurrent-container cap derived from the kernel keyring quota."""
    _fields = AgSandboxBackendFields()
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
    '?/limit' if unreadable.

    Uses sem_getvalue() via the internal _semlock on POSIX (Linux).  The count
    reflects this process's view only — other unrelated processes (including
    a different runtime, or a container started outside this framework
    entirely) are not tracked by our semaphore but do consume system keyring
    slots, so comparing this number with keyring_quota()['used'] reveals how
    many slots belong to external processes.
    """
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
            print(
                f"[agsandbox_backend] WARNING: container-slot semaphore double-release: {_e}",
                file=__import__("sys").stderr,
                flush=True,
            )

    def _is_quota_exhaustion_error(self, stderr: str) -> bool:
        """Return True if *stderr* (from a failed run_cmd) indicates the
        runtime hit the Linux session-keyring quota -- a concurrency quota
        that waiting (for another container to exit and free its keyring)
        can resolve. Matches both docker's and podman's (via runc) wording."""
        return "session key" in stderr or ("disk quota exceeded" in stderr and "keyring" in stderr)

    def _wait_for_quota_slot(self) -> None:
        """Poll the actual keyring free count from /proc until a slot opens
        up (or give up after keyring_wait_timeout_s).

        The runtime-slot semaphore (_container_semaphore) prevents our own
        containers from exceeding the limit, but external processes --
        including the *other* container runtime, or anything else run as
        this same host user -- can consume slots outside our accounting;
        polling /proc catches that case too. Also called unconditionally
        after a name-conflict is resolved in _run_with_conflict_retry(), in
        case a quota was *also* exhausted (e.g. the container object got
        created then hit the limit)."""
        deadline = time.monotonic() + self.keyring_wait_timeout_s
        while time.monotonic() < deadline:
            if keyring_quota().get("free", 0) > 0:
                return
            time.sleep(self.keyring_poll_interval_s)

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
        agconfig: "agConfig | None",
    ) -> None:
        reap_orphaned_containers()
        # Captured here, at construction time, rather than read fresh from
        # os.getpid() inside _ensure_started() -- tool calls with
        # run_in_subprocess=True (the default) cloudpickle this backend to a
        # ProcessPoolExecutor worker, so _ensure_started() (and the `docker
        # run` it issues) can run in a short-lived *worker* process distinct
        # from -- and which can exit independently of -- the main process
        # that actually owns this sandbox for its whole lifetime. Labeling
        # the container with a fresh os.getpid() there would tag it with
        # whichever worker happened to create it; once that worker exits
        # (routine pool recycling, not a crash) while the container and its
        # owning main process are both still very much alive, a concurrent
        # reap_orphaned_containers() elsewhere would wrongly see a "dead"
        # owner and delete a container still in active use. self._owner_pid
        # is a plain instance attribute fixed here in whichever process
        # actually constructs this backend (always the main process, never a
        # worker), so it survives that same cloudpickling unchanged.
        self._owner_pid = os.getpid()
        self._agname = agname
        self._gpu_id: int | None = None
        self._gpu_virtual: bool = False  # LLM has called reserve_gpu
        self._gpu_acquire_fn = None  # pool.acquire_gpu, set by make_gpu_reserve
        self._gpu_release_fn = None  # pool.release_gpu, set by make_gpu_reserve
        self._cpu_acquired: float = 0.0
        self._memory_acquired_mb: int = 0
        self._watched_pids: dict[int, float] = {}
        # None means "not captured yet" -- distinct from a legitimately empty
        # baseline. See _ensure_started()'s docstring for why this must be
        # captured exactly once and never recomputed on a later call.
        self._baseline_pids: "set[int] | None" = None
        self._daemon_pids: set[int] = set()
        self._destroyed = False
        self._checkpoint_image: str | None = checkpoint_image
        # Transient squash-time "diff since reference chain" tar, built
        # lazily only when a squash is due (see
        # `_build_accumulator_for_squash()`) by reading each new layer's
        # on-disk overlay diff via `_locate_layer_diff_dir()`. Ordinary
        # plain commits do NOT touch this -- under the hibernate model the
        # lifecycle tag already keeps a single commit image, so eagerly
        # copying that tip into TMPDIR every skill was pure duplicate
        # disk. Cleared in a finally after the squash attempt (success or
        # fallback). None/0 means "not built yet"; a failed build raises
        # so commit() falls back to `_squash_commit()` export/import.
        self._accumulated_diff_path: "Path | None" = None
        self._accumulated_layer_count: int = 0
        self._accumulator_dir: "Path | None" = None
        # The reference chain _accumulator_squash_commit() validates and
        # builds against -- None means "use self._base_image's own
        # digests" (the ordinary case, and always true until this
        # backend's own first successful squash). Set to the JUST-
        # PRODUCED chain's digests after every successful squash (fast or
        # fallback -- see _accumulator_squash_commit()/_squash_commit()),
        # so a squash that fell back to export/import doesn't permanently
        # lock this backend out of the fast path for the rest of its
        # life: the fallback's own single-layer result becomes the new
        # reference point for the NEXT squash instead of the original,
        # now-unrelated base image. Referenced by DIGEST only (never the
        # mutable lifecycle tag string, which gets overwritten by every
        # subsequent commit) -- safe to hold onto indefinitely, since a
        # layer that's an ancestor of the current chain can't be deleted
        # out from under it (the runtime refuses "has dependent child
        # images"). Purely in-memory: lost on process restart or fork
        # (a fresh backend object starts back at None, falling back to
        # self._base_image -- the same one-time-per-object gap as before,
        # not a regression).
        self._squash_base_diff_ids: "list[str] | None" = None
        self._agconfig = agconfig
        self._name = name
        self._base_image = base_image
        self._vol_flags: list[str] = []
        for host, container, mode in mounts.values():
            self._vol_flags += ["-v", f"{host}:{container}:{mode}"]

    def change_config(self, agconfig: "agConfig | None") -> None:
        self._agconfig = agconfig

    def get_config_copy(self) -> "agConfig | None":
        return self._agconfig.clone() if self._agconfig is not None else None

    def _container_running(self) -> bool:
        """Return True if the named container is currently running in Docker/Podman."""
        result = self._run(
            [self._runtime, "inspect", "--format", "{{.State.Running}}", self._name],
            check=False,
            timeout=self.inspect_timeout_s,
        )
        return result.returncode == 0 and result.stdout.strip() == b"true"

    def _container_status(self) -> str:
        """Return the container state string: 'running', 'exited', 'created', etc., or '' if not found."""
        result = self._run(
            [self._runtime, "inspect", "--format", "{{.State.Status}}", self._name],
            check=False,
            timeout=self.inspect_timeout_s,
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
        result = self._run(cmd, check=False, timeout=self.inspect_timeout_s)
        if result.returncode != 0:
            return set()
        pids: set[int] = set()
        lines = result.stdout.decode("utf-8", errors="replace").splitlines()
        for line in lines[1:]:  # first line is the descriptor header (PID/HPID)
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
        return pids

    def _inspect_container_state(self) -> "tuple[bool, str, int | None]":
        """Return (running, status, init_pid) from a SINGLE docker/podman
        inspect call -- merges what _container_running()/_container_status()
        would otherwise need two separate inspect round-trips for, since
        _ensure_started() (its only caller that needs both) always wants to
        know both facts together. status is '' if the container doesn't
        exist at all, same convention as _container_status(). init_pid is the
        container's init process on the host, used by _register_prof_cgroup()
        to resolve the container's cgroup path from /proc/<pid>/cgroup --
        carried in the same inspect so profiling costs no extra round-trip;
        None when the container doesn't exist or isn't running (a stopped
        container reports Pid 0). Ground truth, same as those two -- no
        caching across calls (see _ensure_started()'s docstring for why)."""
        result = self._run(
            [
                self._runtime,
                "inspect",
                "--format",
                "{{.State.Running}}|{{.State.Status}}|{{.State.Pid}}",
                self._name,
            ],
            check=False,
            timeout=self.inspect_timeout_s,
        )
        if result.returncode != 0:
            return (False, "", None)
        parts = result.stdout.decode("utf-8", errors="replace").strip().split("|")
        # running and pid anchor to the ends; only status could absorb an
        # unexpected '|' in the middle.
        running_str, status, pid_str = parts[0], "|".join(parts[1:-1]), parts[-1]
        try:
            pid = int(pid_str) or None  # stopped containers report Pid 0
        except ValueError:
            pid = None
        return (running_str == "true", status, pid)

    def _ensure_started(self) -> None:
        """Start the Docker/Podman container on first use.

        Called lazily by _container_exec() so containers are only created when
        an agent actually needs sandboxed execution (bash, file I/O, etc.).
        Tasks that complete using only host-side tools (webfetch, todowrite,
        find_papers, …) never start a container at all.

        Invariant: after rm_container() the container does not exist; after
        stop() (hibernate) it still does, just not running -- see that
        method's docstring. The three states handled here are therefore:
          - running     → reuse (worker-reuse path, does NOT acquire the runtime slot)
          - hibernating  → docker/podman start (resume in place, acquires the
                           runtime slot; the GPU is re-acquired lazily on the
                           next exec() -- see base.py's exec())
          - absent       → docker/podman run (acquires the runtime slot and GPU)
        A hibernating container is unambiguously our own: container names
        embed _RUN_ID (a fresh uuid4 per process), so no other process could
        have created one under this exact name -- there is no "leftover from
        someone else" case to force-remove here the way there used to be.

        Ground truth is always a real docker/podman inspect -- there is no
        self._started cache. Every call pays that cost (just one inspect
        call now, via _inspect_container_state(), not two), but that's the
        price of never trusting a per-process flag that a different
        worker-process copy of this backend could have made stale.

        _baseline_pids is captured exactly once (None means "not yet") and
        never refreshed after that, even though this method itself now runs
        on every single call (the "reuse" branch below fires on every call
        once the container exists). Recomputing it every time would make it
        track "whatever's running right now" instead of "what was already
        running before this sandbox's own tracked work started" -- silently
        reclassifying a still-running tracked background process as baseline
        noise the moment any later call (e.g. get_live_pids() itself) happens
        to re-enter here.
        """
        name = self._name
        running, status, pid = self._inspect_container_state()
        if running:
            # Reuse an already-running container — it already holds whatever
            # slot _acquire_runtime_slot() would take, so we must NOT acquire
            # it again here.
            self._register_prof_cgroup(pid)
            if self._baseline_pids is None:
                self._baseline_pids = self._snapshot_pids_started()
            return
        with agprof.span("sandbox:start"):
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
                self._register_prof_cgroup(None)
                if self._baseline_pids is None:
                    self._baseline_pids = self._snapshot_pids_started()
                return
            # Acquire the physical GPU (if reserve_gpu was called) before the
            # container is created, not just in exec() -- this method can be
            # reached first via read_file()/write_file() rather than exec(), so
            # exec()'s own lazy acquire (base.py) can't be relied on to have
            # already run. Guarded by `self._gpu_id is None` the same way
            # exec()'s does, so whichever entry point gets here first acquires it
            # exactly once. Note this only affects which physical GPU
            # CUDA_VISIBLE_DEVICES points at -- _gpu_flags() below attaches every
            # GPU device to the container unconditionally, so it no longer
            # matters whether this runs before or after reserve_gpu().
            if self._gpu_virtual and self._gpu_id is None and self._gpu_acquire_fn is not None:
                self._gpu_id = self._gpu_acquire_fn()
            gpu_flags = _gpu_flags(self._runtime)
            cgroup_flags = []
            cgroup_parent = agprof.container_cgroup_parent()
            if cgroup_parent is not None and self._runtime == "docker":
                cgroup_flags = [f"--cgroup-parent={cgroup_parent}"]
            self._acquire_runtime_slot()
            try:
                if self._checkpoint_image is not None:
                    # Restart from last committed checkpoint (set by stop(commit=True)).
                    # /workspace and all state from the previous tool call are preserved.
                    image = self._checkpoint_image
                    run_cmd = (
                        [self._runtime, "run", "-d", "--init", "--name", name]
                        + ["--label", f"{_AGENCY_OWNER_PID_LABEL}={self._owner_pid}"]
                        + cgroup_flags
                        + gpu_flags
                        + self._vol_flags
                        + [image, "tail", "-f", "/dev/null"]
                    )
                    self._run_with_conflict_retry(run_cmd, name)
                    # Keep _checkpoint_image — not a one-shot restore, needed for future restarts.
                else:
                    image = self._resolve_image(self._base_image)
                    _pool_fields = _AgResourcePoolFields(self._agconfig)
                    limit_flags = []
                    if _pool_fields.idle_memory is not None:
                        limit_flags.append(f"--memory={_pool_fields.idle_memory}")
                    if self._cfs_supported():
                        limit_flags.append(f"--cpus={_pool_fields.idle_cpus}")
                    run_cmd = (
                        [self._runtime, "run", "-d", "--init", "--name", name]
                        + ["--label", f"{_AGENCY_OWNER_PID_LABEL}={self._owner_pid}"]
                        + limit_flags
                        + cgroup_flags
                        + gpu_flags
                        + self._vol_flags
                        + [image, "tail", "-f", "/dev/null"]
                    )
                    self._run_with_conflict_retry(run_cmd, name)
                    self._run(
                        [self._runtime, "exec", name, "mkdir", "-p", "/workspace"],
                        check=True,
                    )
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
            self._register_prof_cgroup(None)
            if self._baseline_pids is None:
                self._baseline_pids = self._snapshot_pids_started()

    def _prof_container_label(self) -> str:
        """Profiler label for this container: the plain agname.

        The backend's _agname is agSandbox's allocated 'sandbox_<agname>_<dedup>'
        (see agsandbox.py __init__) — unwrap it so profiler series line up with
        the agent's span lanes.
        """
        name = str(self._agname)
        if name.startswith("sandbox_"):
            name = name[len("sandbox_") :]
            base, _, suffix = name.rpartition("_")
            if base and len(suffix) == 4:
                name = base
        return name

    def _register_prof_cgroup(self, pid: "int | None") -> None:
        """Register this container's cgroup with the profiler.

        The cgroup dir comes from the kernel (/proc/<pid>/cgroup) — exact for
        both docker and podman, no runtime naming assumptions. No-op when
        profiling is off; raises during a profiling session if the layout
        cannot be resolved (only docker/podman on cgroup v2 are supported —
        a profiled run should fail loudly, not silently lack container data).
        """
        if not agprof.enabled():
            return
        label = self._prof_container_label()
        if pid is None:
            _running, _status, pid = self._inspect_container_state()
        if not pid:
            raise RuntimeError(
                f"agprof: cannot resolve init PID for container {self._name!r} — "
                "only docker/podman on cgroup v2 are supported"
            )
        try:
            cg_text = Path(f"/proc/{pid}/cgroup").read_text()
        except OSError as e:
            raise RuntimeError(f"agprof: cannot read cgroup of container PID {pid}: {e}") from e
        rel = next(
            (l.split("::", 1)[1].strip() for l in cg_text.splitlines() if l.startswith("0::")),
            None,
        )
        if rel is None:
            raise RuntimeError(
                f"agprof: container PID {pid} has no cgroup v2 entry — "
                "cgroup v1 hosts are not supported"
            )
        cdir = "/sys/fs/cgroup" + rel
        # Runtime-daemon cgroup for the "daemon cost" aggregate: podman's
        # per-container conmon scope, or docker's system services.
        daemon_dir = None
        daemon_kind = "conmon"
        scope = cdir
        while scope and not scope.endswith(".scope") and scope != "/sys/fs/cgroup":
            scope = os.path.dirname(scope)
        base = os.path.basename(scope)
        if base.startswith("libpod-"):
            cand = os.path.join(os.path.dirname(scope), f"libpod-conmon-{base[len('libpod-') :]}")
            if os.path.isdir(cand):
                daemon_dir = cand
        elif base.startswith("docker-"):
            daemon_kind = "dockerd"
            for svc in ("docker.service", "containerd.service"):
                cand = f"/sys/fs/cgroup/system.slice/{svc}"
                if os.path.isdir(cand):
                    agprof.container_started(label, cdir, cand, daemon_kind)
            daemon_dir = None  # registered above (possibly twice, one per service)
        agprof.container_started(label, cdir, daemon_dir, daemon_kind)

    def _snapshot_pids_started(self) -> set[int]:
        """Same PID listing as base._snapshot_pids(), routed through
        _container_exec_started() instead of _container_exec() -- see that
        method's docstring for why this is required here."""
        out, _ = self._container_exec_started(
            "__SELF=$$\n"
            "for __d in /proc/[0-9]*; do\n"
            '  [ -f "$__d/status" ] || continue\n'
            "  __p=${__d##*/}\n"
            '  [ "$__p" != "$__SELF" ] && echo "$__p"\n'
            "done",
            timeout=self.inspect_timeout_s,
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
        for attempt in range(self.conflict_retry_max_attempts):
            result = self._run(run_cmd, timeout=self.docker_run_timeout_s)
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
                deadline = time.monotonic() + self.container_removal_wait_s
                while time.monotonic() < deadline:
                    if not self._container_status():
                        break
                    time.sleep(self.container_removal_poll_interval_s)
                # If a quota is also exhausted (e.g. the object got created
                # then hit the limit), wait for a slot before retrying —
                # otherwise we'll create another "Created" container and loop
                # on conflicts. No-op for runtimes with no quota to wait on.
                self._wait_for_quota_slot()
                time.sleep(self.conflict_retry_backoff_base_s * (attempt + 1))
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
        for attempt in range(self.conflict_retry_max_attempts):
            result = self._run([self._runtime, "start", name], timeout=self.docker_start_timeout_s)
            if result.returncode == 0:
                return
            stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            _last_stderr = stderr
            if self._is_quota_exhaustion_error(stderr):
                self._wait_for_quota_slot()
                time.sleep(self.conflict_retry_backoff_base_s * (attempt + 1))
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
            return run_with_unkillable_child_grace(
                lambda: subprocess.run(
                    args, input=input, capture_output=True, timeout=timeout, check=check
                ),
                args=args,
                timeout=timeout,
                grace_s=self.unkillable_child_grace_s,
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
        self._run([self._runtime, "rm", "-f", name], check=True, timeout=self.docker_rm_timeout_s)

    def _rmi(self, image_ref: str, *, force: bool = False) -> None:
        """Remove an image by ID or tag. Raises on failure."""
        cmd = [self._runtime, "rmi"]
        if force:
            cmd.append("-f")
        cmd.append(image_ref)
        self._run(cmd, check=True, timeout=self.image_timeout_s)

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
        self._run(cmd, timeout=self.inspect_timeout_s)

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
            timeout=self.stop_inspect_timeout_s,
        )
        return json.loads(result.stdout.decode("utf-8", errors="replace"))

    def _locate_layer_diff_dir(
        self, diff_id: str, *, diff_ids: "list[str] | None" = None
    ) -> "Path | None":
        """Find the raw, on-disk diff directory backing *diff_id* directly
        -- i.e. exactly the same data a `docker/podman commit` producing
        this layer already read to build it, reused here essentially for
        free instead of re-deriving it via `docker diff` (a generic scan
        costing ~9s on a real ~24GB/many-file image regardless of how
        much actually changed) or `docker save` (cost proportional to the
        whole image). See `_build_accumulator_for_squash()`'s docstring
        for how this feeds the fast squash path, and
        docs/agsandbox_backends/container.md's "Fast incremental
        squashing" section for the full rationale.

        *diff_ids*, when provided, is the image's full RootFS.Layers list
        (most-base-first) ending at *diff_id*. Some storage backends
        (Docker's containerd overlayfs snapshotter) key layers by ChainID,
        which is a function of that whole prefix -- not the tip DiffID
        alone -- so the fold path always passes it; backends that only
        need *diff_id* may ignore it.

        None by default -- this means reaching into a runtime's own
        undocumented internal storage layout, which is necessarily
        runtime-specific (Docker's overlay2 / containerd-overlayfs layouts
        and Podman's `containers/storage` layout are unrelated). Overridden
        by `_DockerBackend` (`.docker`) and `_PodmanBackend` (`.podman`);
        returning None here means "no fast lookup available for this
        runtime," which `_build_accumulator_for_squash()` treats as
        "accumulator unavailable," safely falling back to the slower
        but always-correct `_squash_commit()` path -- never as an error.
        """
        return None

    def _host_to_container_id(self, uid: int, gid: int) -> "tuple[int, int]":
        """Translate the HOST-side ownership `_locate_layer_diff_dir()`'s
        files carry into the ownership the CONTAINER itself sees for
        them. Identity by default -- correct for any runtime that
        doesn't remap ownership between its own user namespace and the
        container's (true of non-rootless Docker/Podman, where the
        overlay2/overlay diff directory's on-disk ownership already IS
        the container-visible ownership).

        Overridden by `_DockerBackend` for rootless Docker and
        `_PodmanBackend` for rootless Podman, where the runtime's user
        namespace means a raw `os.lstat()` on the diff directory reports
        HOST-remapped ownership instead (confirmed empirically: a
        root-owned file inside the container showed up as owned by the
        invoking host user via the raw overlay path, not uid 0) --
        passed to `_layer_squash.overlay_diff_to_tar()`'s
        `uid_gid_translate` parameter by `_fold_overlay_diff_into_accumulator()`
        below.
        """
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
            self._accumulator_dir = Path(tempfile.mkdtemp(prefix="agency-accum-"))
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
        image size (verified against the real ~24GB, 80-layer
        `agency-sandbox:latest`).

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
                timeout=self.stop_inspect_timeout_s,
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

        tmp_dir = Path(tempfile.mkdtemp(prefix="agency-squash-"))
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
                timeout=self.squash_timeout_s,
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
        """Flatten the container's current filesystem into a brand-new
        single-layer image tagged *tag*, instead of committing a diff on
        top of the existing chain.

        `docker commit` always creates one more layer on top of whatever
        the container was started from; since _ensure_started() always
        restarts FROM the last checkpoint image, a long-running sandbox's
        layer chain grows by exactly one every checkpoint cycle with
        nothing to bound it, until it crosses the container runtime's hard
        layer-depth cap ("max depth exceeded" on docker/moby, ~125 layers
        observed empirically on this host). `export`+`import` serializes
        the container's FULL filesystem into a new image with no parent
        chain at all, resetting depth back to 1 -- this is the
        always-correct fallback `stop()` uses when
        `_accumulator_squash_commit()`'s faster, diff-only path raises
        (its accumulator can't be trusted for this squash); materially
        slower since it re-serializes the whole merged filesystem rather
        than just the accumulated diff.

        Safe to drop the image metadata `docker commit` would normally
        preserve (env, embedded CMD/ENTRYPOINT): _ensure_started()'s
        restart `run` command always passes an explicit `tail -f
        /dev/null`, never relying on anything baked into the image itself.
        The one label that IS explicitly re-applied via `--change` is
        `_AGENCY_OWNER_PID_LABEL` -- `docker commit` propagates a
        container's labels onto its image automatically, but `export`/
        `import` doesn't preserve container config at all (confirmed
        empirically: an export/import round-trip strips every label), so
        without this the resulting image would carry no owner_pid at
        all, making it permanently unreapable by
        `reap_orphaned_containers()`'s image-scan (see that function's
        docstring) even after its owning process dies.

        Resets the diff accumulator on success (see
        `_accumulator_squash_commit()`): export/import produces a
        PARENTLESS image, disconnected from `self._base_image`'s own
        lineage entirely -- meaning `_accumulator_squash_commit()`'s
        base-is-a-prefix precondition can never hold again relative to
        the ORIGINAL base image. This no longer means every future squash
        is permanently stuck on this slower path, though: this method
        re-baselines `self._squash_base_diff_ids` to the freshly-
        flattened image's own (single-layer) chain right below, so the
        NEXT squash attempt validates/builds against THAT instead --
        recovering fast-path eligibility from here on, for as long as
        this backend object lives (see `self._squash_base_diff_ids`'s
        docstring in __init__ for why this is safe and why it doesn't
        persist across a process restart or fork).
        """
        export_result = self._run(
            [self._runtime, "export", self._container_name()],
            check=True,
            timeout=self.squash_timeout_s,
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
            timeout=self.squash_timeout_s,
        )
        try:
            self._squash_base_diff_ids = self._image_diff_ids(tag)
        except Exception as _e:
            # Best-effort: the flatten itself already succeeded above --
            # this only means the NEXT squash won't benefit from the fast
            # path re-baseline (falls back to self._base_image instead,
            # which will fail its prefix check again and fall back once
            # more, exactly like before this re-baselining existed).
            print(
                f"[agsandbox_backend] WARNING: could not record post-squash chain "
                f"for tag {tag}, next squash will fall back to export/import again: {_e}",
                file=__import__("sys").stderr,
                flush=True,
            )
            self._squash_base_diff_ids = None
        self._reset_accumulator()

    def stop(self) -> None:
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
        gpu_id_to_release = (
            self._gpu_id if (self._gpu_virtual and self._gpu_id is not None) else None
        )
        if not self._container_running():
            return
        # Clear PID tracking — stopping kills every process inside, same as rm.
        self._watched_pids = {}
        self._baseline_pids = None  # force a fresh capture on the next _ensure_started()
        name = self._container_name()
        stop_exc: Exception | None = None
        try:
            self._run(
                [self._runtime, "stop", "-t", str(self.docker_stop_grace_s), name],
                check=True,
                timeout=self.docker_stop_timeout_s,
            )
        except Exception as _e:
            stop_exc = _e
        # Only release once actually confirmed not running -- a failed stop
        # may still leave the container (and the slot/GPU it holds) alive.
        if not self._container_running():
            self._release_runtime_slot()
            if gpu_id_to_release is not None and self._gpu_release_fn is not None:
                self._gpu_release_fn(gpu_id_to_release)
                self._gpu_id = None
        if stop_exc is not None:
            raise stop_exc

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
        # Deregister from the profiler's container sampling (no-op with no
        # profiling session). Deliberately NOT done in stop()/hibernate: the
        # cgroup reappears at the same path on resume, so the registration
        # stays valid across hibernation.
        agprof.container_stopped(self._prof_container_label())
        gpu_id_to_release = (
            self._gpu_id if (self._gpu_virtual and self._gpu_id is not None) else None
        )
        if not self._container_status():
            if gpu_id_to_release is not None and self._gpu_release_fn is not None:
                self._gpu_release_fn(gpu_id_to_release)
                self._gpu_id = None
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
        name = self._container_name()
        # rm_exc is raised at the end, after the release checks below still
        # run -- an unconfirmed removal must reach the caller, but shouldn't
        # cut short a release that's still legitimately possible to check.
        rm_exc: Exception | None = None
        for _attempt in range(self.rm_retry_attempts):
            try:
                self._rm_container(name)
                rm_exc = None
                break
            except Exception as _e:
                rm_exc = _e
                if _attempt != self.rm_retry_attempts - 1:
                    time.sleep(self.rm_retry_backoff_s)
        if had_container and not self._container_running():
            self._release_runtime_slot()
        if (
            gpu_id_to_release is not None
            and self._gpu_release_fn is not None
            and not self._container_running()
        ):
            self._gpu_release_fn(gpu_id_to_release)
            self._gpu_id = None
        if rm_exc is not None:
            raise rm_exc

    def commit(self, tag: "str | None" = None) -> bool:
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
        #    immediately, safely deletable -- unlike the old teardown-every-
        #    cycle design, where a plain commit's result was always a child
        #    of what it replaced and could never free it. Best-effort: a
        #    failure to even look this up just means one image doesn't get
        #    cleaned up this cycle, not that the commit itself is at risk.
        previous_image_id: str | None = None
        try:
            _prev_result = self._run(
                [self._runtime, "inspect", "--format={{.Id}}", tag],
                check=False,
                timeout=self.stop_inspect_timeout_s,
            )
            if _prev_result and _prev_result.returncode == 0:
                previous_image_id = (
                    _prev_result.stdout.decode("utf-8", errors="replace").strip() or None
                )
        except Exception as _e:
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
        for _attempt in range(self.commit_retry_attempts):
            try:
                self._run(
                    [self._runtime, "commit", self._container_name(), tag],
                    check=True,
                    timeout=self.commit_timeout_s,
                )
                self._checkpoint_image = tag
                commit_exc = None
                break
            except Exception as _e:
                commit_exc = _e
                if _attempt != self.commit_retry_attempts - 1:
                    time.sleep(self.commit_retry_backoff_s)
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
                    timeout=self.stop_ps_check_timeout_s,
                )
                if _in_use and _in_use.stdout.strip():
                    pass  # a container (e.g. a fork) still runs from this image — leave it
                else:
                    self._rmi(previous_image_id)
            except Exception as _e:
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
            should_squash = len(self._image_diff_ids(tag)) >= self.checkpoint_squash_max_depth
        except Exception as _e:
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
                    timeout=self.stop_inspect_timeout_s,
                )
                if result and result.returncode == 0:
                    old_image_id = result.stdout.decode("utf-8", errors="replace").strip() or None
            except Exception as _e:
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
                        timeout=self.stop_ps_check_timeout_s,
                    )
                    if in_use and in_use.stdout.strip():
                        pass  # containers still running from this image — leave it
                    else:
                        self._rmi(old_image_id)
                except Exception as _e:
                    print(
                        f"[agsandbox_backend] WARNING: could not check/delete old image {old_image_id}: {_e}",
                        file=__import__("sys").stderr,
                        flush=True,
                    )
        return True

    def restore(self, tag: str) -> None:
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
                        timeout=self.exec_quick_timeout_s,
                        shell="sh",
                    )
                except Exception as _e:
                    print(
                        f"[agsandbox_backend] WARNING: failed to kill PIDs {pids} in {self._name} during restore: {_e}"
                    )
            self.rm_container()
        self._checkpoint_image = tag
        self._ensure_started()

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
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
                    f"kill {pids} 2>/dev/null; true", timeout=self.exec_quick_timeout_s, shell="sh"
                )
            except Exception as _e:
                print(
                    f"[agsandbox_backend] WARNING: failed to kill PIDs {pids} in {container_name}: {_e}"
                )

        # rm_container() already retries and only releases the runtime
        # slot/GPU once actually confirmed gone (see its docstring) --
        # destroy() no longer needs its own copy of that logic (a prior
        # copy here silently drifted out of sync with a fix made to
        # rm_container() itself, which is exactly the risk of keeping two).
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
