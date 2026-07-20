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

import multiprocessing
import os
import re
import shutil
import subprocess
import threading
import time
import uuid as _uuid
from pathlib import Path

from ..agconfig import agConfig
from ..agresources import amd_render_node_paths_by_pci_bus, detect_gpus, _AgResourcePoolFields
from .base import AgSandboxBackendFields, agsandbox_backend, run_with_unkillable_child_grace

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


def _gpu_flags(runtime: str, gpu_id: "int | None") -> list[str]:
    """Return GPU passthrough flags for *runtime* ("docker" or "podman"),
    scoped to the single *gpu_id* leased to this sandbox -- mirrors
    chroot.py's _chroot_gpu_dev_paths(gpu_id): a sandbox that hasn't leased a
    GPU (gpu_id is None) gets zero GPU devices, and one that has leased a
    GPU only gets that one device exposed, not every GPU on the host. This
    is the hardware-level half of GPU isolation; CUDA_VISIBLE_DEVICES /
    HIP_VISIBLE_DEVICES (set in base.py's exec()) is the software half --
    without this, a process could bypass the env var by opening another
    GPU's device node directly, since it was mounted into the container
    regardless of which GPU was actually leased.

    NVIDIA:
      - Docker: ``--gpus device=N`` (nvidia-container-toolkit's Docker-specific
        CLI wrapper hook).
      - Podman: ``--device nvidia.com/gpu=N`` (CDI). Podman does not
        understand Docker's ``--gpus`` flag: it accepts it silently (no
        error) but never mounts the NVIDIA driver/devices, so a container
        started that way has zero GPU access despite `podman run` appearing
        to succeed -- `nvidia-smi` inside prints "WARNING: The NVIDIA Driver
        was not detected" and isn't even on PATH. This mirrors the identical
        fix applied to images/build.sh's own smoke tests.
    AMD:    ``--device /dev/kfd`` (shared control device, every AMD GPU needs
            it regardless of index) plus the one ``/dev/dri/renderD*`` node
            matching *gpu_id*, ordered by PCI bus to line up with rocm-smi's
            own GPU numbering (see _amd_render_node_paths() and
            agresources.amd_render_node_paths_by_pci_bus() -- confirmed on
            real 8x MI350X hardware that naive sorted /dev/dri order does
            NOT correspond to GPU index).
    CPU-only hosts, and sandboxes that haven't leased a GPU, get no flags.
    """
    if gpu_id is None:
        return []
    kind = _gpu_kind(runtime)
    if kind == "none":
        return []
    if kind == "nvidia":
        return (
            ["--device", f"nvidia.com/gpu={gpu_id}"]
            if runtime == "podman"
            else ["--gpus", f"device={gpu_id}"]
        )
    render_nodes = _amd_render_node_paths()
    flags = ["--device", "/dev/kfd"]
    if gpu_id < len(render_nodes):
        flags += ["--device", render_nodes[gpu_id]]
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

    def _ensure_started(self) -> None:
        """Start the Docker/Podman container on first use.

        Called lazily by _container_exec() so containers are only created when
        an agent actually needs sandboxed execution (bash, file I/O, etc.).
        Tasks that complete using only host-side tools (webfetch, todowrite,
        find_papers, …) never start a container at all.

        Invariant: after stop() the container does not exist.  The only two
        states we handle here are therefore:
          - running  → reuse (worker-reuse path, does NOT acquire the runtime slot)
          - absent   → docker/podman run (acquires the runtime slot)
        Any leftover container in another state is force-removed first.

        Ground truth is always _container_running() -- there is no
        self._started cache. Every call pays a real docker/podman inspect,
        but that's the price of never trusting a per-process flag that a
        different worker-process copy of this backend could have made stale.

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
        if self._container_running():
            # Reuse an already-running container — it already holds whatever
            # slot _acquire_runtime_slot() would take, so we must NOT acquire
            # it again here.
            if self._baseline_pids is None:
                self._baseline_pids = self._snapshot_pids_started()
            return
        # Remove any leftover container in a non-running state (created,
        # exited, dead, …) that stop() failed to clean up.
        if self._container_status():
            self._rm_container(name)
        # Acquire the physical GPU (if reserve_gpu was called) before the
        # container is created, not just in exec() -- the container's GPU
        # device flags are fixed at `docker/podman run` time, and this method
        # can be reached first via read_file()/write_file() rather than
        # exec(), so exec()'s own lazy acquire (base.py) can't be relied on
        # to have already run. Guarded by `self._gpu_id is None` the same way
        # exec()'s does, so whichever entry point gets here first acquires it
        # exactly once.
        if self._gpu_virtual and self._gpu_id is None and self._gpu_acquire_fn is not None:
            self._gpu_id = self._gpu_acquire_fn()
        gpu_flags = _gpu_flags(self._runtime, self._gpu_id)
        self._acquire_runtime_slot()
        try:
            if self._checkpoint_image is not None:
                # Restart from last committed checkpoint (set by stop(commit=True)).
                # /workspace and all state from the previous tool call are preserved.
                image = self._checkpoint_image
                run_cmd = (
                    [self._runtime, "run", "-d", "--init", "--name", name]
                    + ["--label", f"{_AGENCY_OWNER_PID_LABEL}={self._owner_pid}"]
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
        if self._baseline_pids is None:
            self._baseline_pids = self._snapshot_pids_started()

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

    def commit(self, tag: str) -> bool:
        """Commit the container filesystem to a new image tag.

        Returns True if the commit succeeded, False if the container doesn't
        exist.  Works on both running and stopped containers (docker commit
        does not require the container to be running).
        """
        if not self._container_running():
            return False
        self._run(
            [self._runtime, "commit", self._container_name(), tag],
            check=True,
            timeout=self.commit_timeout_s,
        )
        return True

    def stop(self, *, commit: bool = False) -> None:
        """Stop and remove the container, releasing its runtime slot and GPU.

        If commit=True, the container filesystem is committed to an image first
        so _ensure_started() can recreate from it on the next tool call.  Pass
        commit=True after a successful sandbox tool call; commit=False after a
        failure to discard the dirty state and revert to the last checkpoint.
        """
        gpu_id_to_release = (
            self._gpu_id if (self._gpu_virtual and self._gpu_id is not None) else None
        )
        if not self._container_running():
            if gpu_id_to_release is not None and self._gpu_release_fn is not None:
                self._gpu_release_fn(gpu_id_to_release)
                self._gpu_id = None
            return
        # Clear PID tracking — remove kills all processes.
        self._watched_pids = {}
        self._baseline_pids = None  # force a fresh capture on the next _ensure_started()
        # commit_exc is raised at the end, after rm and the release checks
        # below still run -- a failed commit doesn't mean the container
        # shouldn't still be torn down and its resources still released,
        # it just means this attempt's state wasn't checkpointed forward.
        commit_exc: Exception | None = None
        if commit:
            tag = self._lifecycle_tag()
            # Capture the current image ID before overwriting the tag so we
            # can delete it afterward — committing to an existing tag leaves
            # the old image dangling (untagged but still on disk). Best
            # effort: a stray dangling image costs disk space, not
            # correctness, so this warns rather than aborting stop() over it.
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
            # Delete the previous image now that the tag points to the new one.
            # Only delete if no containers are currently using it — a fork may still
            # be running from the same image.  The fork's own stop() will delete it
            # once its container is gone. Best-effort, same reasoning as above.
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
        name = self._container_name()
        # rm_exc, like commit_exc, is raised at the end rather than
        # immediately -- but unlike commit_exc, its failure also gates the
        # release checks below: an unconfirmed removal means the container
        # (and whatever it holds) is not known to be gone.
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
        # Only release the runtime slot once the container is actually
        # confirmed gone -- if every rm -f attempt failed, the container (and
        # the physical slot it occupies) is still alive, so releasing here
        # would over-credit the semaphore while destroy() (or a later stop())
        # still has a live container to clean up and would release again once
        # removal genuinely succeeds.
        if not self._container_running():
            self._release_runtime_slot()
        # Free the GPU only once the container is confirmed gone -- same
        # ground-truth gate as the runtime slot above; releasing while rm
        # failed and the container might still be running would let
        # something else acquire the same physical GPU concurrently.
        if (
            gpu_id_to_release is not None
            and self._gpu_release_fn is not None
            and not self._container_running()
        ):
            self._gpu_release_fn(gpu_id_to_release)
            self._gpu_id = None
        # Surface whichever failure is more severe: an unconfirmed removal
        # means real resources may still be held, which matters more than a
        # missed checkpoint.
        if rm_exc is not None:
            raise rm_exc
        if commit_exc is not None:
            raise commit_exc

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
            self._rm_container(self._container_name())
            self._watched_pids = {}
            self._baseline_pids = None  # force a fresh capture in _ensure_started() below
        self._checkpoint_image = tag
        self._ensure_started()

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        # Always attempt cleanup below -- docker rm -f is a no-op when the
        # container doesn't exist, and _container_running() is ground truth
        # regardless of which process (this one or a worker) actually
        # started the container.
        container_name = self._container_name()
        # Release the GPU here too -- destroy() is called from atexit/__del__
        # (see agsandbox.py) on sandboxes that may never have gone through a
        # normal stop() first, so this can't assume stop() already handled it.
        # Not delegated to stop(): stop()'s rm retry loop raises immediately
        # on failure, but destroy() must still attempt every remaining
        # cleanup step (GPU release check, image cleanup) before surfacing
        # that failure, rather than aborting partway through.
        gpu_id_to_release = (
            self._gpu_id if (self._gpu_virtual and self._gpu_id is not None) else None
        )

        # Best-effort courtesy signal before rm -f forces the issue --
        # rm -f kills everything inside the container regardless of whether
        # this succeeds, so a failure here doesn't change what actually
        # happens, only whether tracked processes got a chance to react
        # first. Warn, don't raise: nothing depends on this succeeding.
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

        # Check before rm so we know whether a runtime slot must be released.
        had_container = self._container_running()

        # rm_exc is raised at the end, after every remaining cleanup step
        # below has still been attempted -- an unconfirmed removal must
        # reach the caller, but shouldn't cut short the GPU-release check or
        # the best-effort image cleanup that don't depend on it.
        rm_exc: Exception | None = None
        try:
            if self._container_status():
                self._rm_container(container_name)
        except Exception as _e:
            rm_exc = _e
        finally:
            # Only release if the container is actually confirmed gone now --
            # if had_container was True because a prior stop() already
            # observed removal succeed (and already released), this recheck
            # correctly sees no container and skips a second release; if rm
            # here fails too, the slot is still legitimately held and must
            # not be released.
            if had_container and not self._container_running():
                self._release_runtime_slot()

        # Same ground-truth gate as the runtime slot above -- releasing the
        # GPU while rm failed and the container might still be running would
        # let something else acquire the same physical GPU concurrently.
        if (
            gpu_id_to_release is not None
            and self._gpu_release_fn is not None
            and not self._container_running()
        ):
            self._gpu_release_fn(gpu_id_to_release)
            self._gpu_id = None

        # Remove the checkpoint image and all pre-tool snapshots created
        # during this sandbox's lifetime. Best-effort: a stray dangling
        # image costs disk space, not correctness, so these warn rather
        # than raising.
        if self._checkpoint_image:
            try:
                self._rmi(self._checkpoint_image, force=True)
            except Exception as _e:
                print(
                    f"[agsandbox_backend] WARNING: checkpoint image cleanup failed for {container_name}: {_e}"
                )
            self._checkpoint_image = None
        try:
            result = self._run(
                [self._runtime, "images", "--format", "{{.Repository}}:{{.Tag}}"],
                timeout=self.image_timeout_s,
            )
            prefix = f"agency/pretool-{self._name}-"
            for line in result.stdout.decode("utf-8", errors="replace").splitlines():
                tag = line.strip()
                if tag.startswith(prefix):
                    self._rmi(tag, force=True)
        except Exception as _e:
            print(
                f"[agsandbox_backend] WARNING: pretool image cleanup failed for {container_name}: {_e}"
            )

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
        with _get_docker_semaphore():
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
        with _get_docker_semaphore():
            subprocess.run(cmd, capture_output=True)

    @staticmethod
    def export_image(tag: str, timeout: int) -> bytes:
        """Export an image to a tar byte-string (docker/podman save).

        The image must already exist.  Returns the raw tar bytes suitable
        for writing to a file or embedding in a larger archive.
        Raises ``subprocess.CalledProcessError`` on failure.
        """
        runtime = get_container_runtime()
        with _get_docker_semaphore():
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
        with _get_docker_semaphore():
            subprocess.run(
                [runtime, "load"],
                input=image_bytes,
                capture_output=True,
                check=True,
                timeout=timeout,
            )
