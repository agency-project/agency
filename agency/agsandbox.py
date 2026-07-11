from __future__ import annotations

import atexit
import multiprocessing
import os
import shlex
import shutil
import subprocess
import threading
import time
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .agconfig import agConfig, GlobalConfigParam, StaticConfigParam, _AgConfigViewBase
from .agresources import detect_gpus, _AgResourcePoolFields

if TYPE_CHECKING:
    from .agresources import agResourcePool
    from .agterm import agterm
    from .aglog import aglog


# Exists to register agSandbox's config fields (via __set_name__ at import
# time) and hold their hardcoded defaults as plain class attributes --
# agSandbox inherits from this below, so self.base_image etc. work via the
# inherited ConfigParam descriptors exactly as if declared directly on it.
# All fields here are tier 1 (global): agsandbox.py's subprocess-execution
# layer is shared process-wide (docker/podman daemon calls, the container
# semaphore, the keyring quota), and several of these values are also read
# from bare module-level functions with no agSandbox instance/agconfig at
# hand (e.g. _runtime_works, _docker_container_limit) -- GlobalConfigParam
# lets both `self.xxx` (from an instance) and `_AgSandboxFields().xxx` (a
# throwaway instance, from a module function) resolve identically.
class _AgSandboxFields:
    # Referenced by name elsewhere as def-time default arguments (bare
    # class-attribute reads with no instance at hand) -- must stay plain
    # attributes, not inlined into a ConfigParam.
    DEFAULT_EXEC_TIMEOUT_S = 120  # Fallback for _run/_container_exec/exec when no explicit timeout is passed.
    SEED_CACHE_TIMEOUT_S = 900    # seed_cache_from_image: copying a large pre-baked directory out of an image.
    WAIT_PING_INTERVAL_S = 300    # wait_for_processes() fallback, overridden in practice by every real caller.
    WAIT_POLL_INTERVAL_S = 5      # wait_for_processes() fallback, overridden in practice by every real caller.

    base_image = StaticConfigParam("agSandbox", default="agency-sandbox:latest")
    docker_semaphore_limit = GlobalConfigParam("agSandbox", default=16)

    # Timeouts (seconds) for the various classes of docker/podman subprocess call.
    inspect_timeout_s = GlobalConfigParam("agSandbox", default=120)      # Fast metadata queries: docker inspect, docker ps, nvidia-smi, docker update.
    exec_quick_timeout_s = GlobalConfigParam("agSandbox", default=120)   # Quick in-container exec calls: kill <pids>, test -d, and similar.
    docker_run_timeout_s = GlobalConfigParam("agSandbox", default=120)   # docker run: GPU init via the NVIDIA container runtime can take 60+ s under load.
    docker_rm_timeout_s = GlobalConfigParam("agSandbox", default=120)    # docker rm -f: fast teardown; should complete in a few seconds.
    file_io_timeout_s = GlobalConfigParam("agSandbox", default=120)      # In-container file I/O via docker exec (base64 read/write, mkdir).
    image_timeout_s = GlobalConfigParam("agSandbox", default=120)        # docker images list / docker rmi.
    commit_timeout_s = GlobalConfigParam("agSandbox", default=120)       # docker commit: snapshots a full overlay layer; large workspaces need extra time.
    keyring_wait_timeout_s = GlobalConfigParam("agSandbox", default=120) # Maximum time to wait for a keyring slot before abandoning a docker run retry.

    # _docker_container_limit(): concurrent-container cap derived from the kernel
    # session-keyring quota (see that function's docstring for the full rationale).
    container_limit_floor = GlobalConfigParam("agSandbox", default=4)      # Never cap below this many concurrent containers.
    container_limit_buffer = GlobalConfigParam("agSandbox", default=5)     # Safety margin subtracted from the raw kernel/fallback quota.
    container_limit_fallback = GlobalConfigParam("agSandbox", default=200) # Assumed kernel quota when /proc/sys/kernel/keys/maxkeys isn't readable.

    # _run_with_conflict_retry(): retrying `docker run` on name-conflict/keyring errors.
    conflict_retry_max_attempts = GlobalConfigParam("agSandbox", default=8)
    keyring_poll_interval_s = GlobalConfigParam("agSandbox", default=5)              # Poll interval while waiting for a free keyring slot.
    container_removal_wait_s = GlobalConfigParam("agSandbox", default=10)            # Max time to wait for a conflicting container to finish being removed.
    container_removal_poll_interval_s = GlobalConfigParam("agSandbox", default=0.5)
    conflict_retry_backoff_base_s = GlobalConfigParam("agSandbox", default=0.5)      # Multiplied by attempt number for linear backoff between retries.

    # stop(): commit-then-remove teardown sequence.
    stop_inspect_timeout_s = GlobalConfigParam("agSandbox", default=30)  # docker inspect (pre-commit image-id lookup).
    commit_retry_attempts = GlobalConfigParam("agSandbox", default=3)
    commit_retry_backoff_s = GlobalConfigParam("agSandbox", default=1)
    stop_ps_check_timeout_s = GlobalConfigParam("agSandbox", default=10)  # docker ps (checking whether the old image is still in use).
    rm_retry_attempts = GlobalConfigParam("agSandbox", default=3)
    rm_retry_backoff_s = GlobalConfigParam("agSandbox", default=1)


_BGPIDS_MARKER = "__BGPIDS__:"


class _ContainerAlreadyRunning(Exception):
    """Raised by _run_with_conflict_retry when another process has already
    started the same container — the caller should reuse it."""
_RUNTIME: str | None = None

# Per-run ID so concurrent and successive runs never share container/image names.
# UUID avoids PID-reuse collisions and prevents stale lifecycle images from crashed
# runs being accidentally picked up by a new run that happens to get the same PID.
import uuid as _uuid
_RUN_ID = f"r{_uuid.uuid4().hex[:8]}"

# Global registry of live sandboxes for atexit cleanup.
_live_sandboxes: weakref.WeakSet["agSandbox"] = weakref.WeakSet()

# Limit the number of containers starting simultaneously.  Each agSandbox.__init__
# acquires one slot for the duration of its startup sequence (docker run + first exec).
# Without this, a burst of hundreds of parallel agent tasks overwhelms the Docker
# All Docker/Podman calls go through _run(), which holds this semaphore for the
# duration of each subprocess call.  Caps concurrent daemon calls at 16: the
# daemon serialises most operations internally (GPU init, overlay diff, container
# teardown), so more than ~16 concurrent calls increase contention without
# reducing wall-clock time.
# Tier-1 (global class) config: lazily created on first use so a caller can
# override the limit via agSandbox.docker_semaphore_limit = N (or
# cfg.agSandbox.docker_semaphore_limit = N before any agSandbox exists)
# before the first docker/podman call in the process. Locked once actually
# read, matching a real semaphore's can't-resize-after-creation semantics.
_docker_semaphore: threading.Semaphore | None = None
_docker_semaphore_init_lock = threading.Lock()


def _get_docker_semaphore() -> threading.Semaphore:
    global _docker_semaphore
    if _docker_semaphore is None:
        with _docker_semaphore_init_lock:
            if _docker_semaphore is None:
                limit = _AgSandboxFields().docker_semaphore_limit
                _docker_semaphore = threading.Semaphore(limit)
    return _docker_semaphore

# Hard cap on the number of simultaneously running Docker containers, derived from
# the Linux kernel session-keyring quota.  Each running Docker container holds one
# session keyring against the user that ran `docker run`; when total keys reach
# /proc/sys/kernel/keys/maxkeys the next docker run fails with
# "unable to create session key: disk quota exceeded".
# Podman is exempt: rootless Podman uses user namespaces with independent keyring
# namespaces and is not subject to this quota.
# multiprocessing.Semaphore is backed by a POSIX IPC semaphore so the limit is
# enforced across all worker processes (which run _ensure_started) and the main
# process (which calls stop/destroy).
def _docker_container_limit() -> int:
    """Return the concurrent-Docker-container cap derived from the kernel keyring quota."""
    _fields = _AgSandboxFields()
    try:
        maxkeys = int(Path("/proc/sys/kernel/keys/maxkeys").read_text().strip())
        return max(_fields.container_limit_floor, maxkeys - _fields.container_limit_buffer)
    except OSError:
        return _fields.container_limit_fallback - _fields.container_limit_buffer

_container_semaphore: multiprocessing.Semaphore = multiprocessing.Semaphore(
    _docker_container_limit()
)


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
    """Return 'held/limit' for _container_semaphore, or '?/limit' if unreadable.

    Uses sem_getvalue() via the internal _semlock on POSIX (Linux).  The count
    reflects this process's view only — other unrelated processes are not
    tracked by our semaphore but do consume system keyring slots, so comparing
    this number with keyring_quota()['used'] reveals how many slots belong to
    external processes.
    """
    limit = _docker_container_limit()
    try:
        available = _container_semaphore._semlock._get_value()
        held = limit - available
    except Exception:
        held = "?"
    return f"{held}/{limit}"


def _cleanup_all_sandboxes() -> None:
    """Destroy all live sandbox containers on process exit."""
    for sandbox in list(_live_sandboxes):
        try:
            sandbox.destroy()
        except Exception as _e:
            print(f"[agsandbox] WARNING: atexit destroy failed for {sandbox._name}: {_e}")


atexit.register(_cleanup_all_sandboxes)



def _runtime_works(runtime: str) -> bool:
    try:
        proc = subprocess.run(
            [runtime, "info"],
            capture_output=True,
            timeout=_AgSandboxFields().inspect_timeout_s,
        )
        return proc.returncode == 0
    except Exception:
        return False


def get_container_runtime() -> str:
    """Return ``docker`` or ``podman``, preferring docker when both are usable."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    has_docker = shutil.which("docker") is not None
    has_podman = shutil.which("podman") is not None
    docker_ok = has_docker and _runtime_works("docker")
    podman_ok = has_podman and _runtime_works("podman")

    if docker_ok:
        _RUNTIME = "docker"
    elif podman_ok:
        _RUNTIME = "podman"
    elif has_docker or has_podman:
        parts = []
        if has_docker and not docker_ok:
            parts.append("docker is installed but not reachable (is the daemon running?)")
        if has_podman and not podman_ok:
            parts.append("podman is installed but not reachable")
        raise RuntimeError("; ".join(parts))
    else:
        raise RuntimeError(
            "Neither docker nor podman is installed. "
            "Install one of them to use sandboxed agents."
        )
    return _RUNTIME


def seed_cache_from_image(host_dir, container_path: str, image: str, timeout: int = _AgSandboxFields.SEED_CACHE_TIMEOUT_S) -> None:
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
    subprocess.run(
        [runtime, "run", "--rm", "-v", f"{host_dir}:/__seed_out",
         image, "sh", "-c",
         f"cp -a {shlex.quote(container_path)}/. /__seed_out/ 2>/dev/null || true"],
        check=False, timeout=timeout,
    )


_gpu_flags_cache: "list[str] | None" = None
_gpu_flags_lock = threading.Lock()


def _gpu_flags() -> list[str]:
    """Return GPU passthrough flags for the container runtime, cached for the
    process lifetime.

    NVIDIA: ``--gpus all`` (requires nvidia-container-toolkit).
    AMD:    ``--device /dev/kfd --device /dev/dri`` (ROCm device files).
    CPU-only hosts get no flags so they keep working without GPU drivers.

    GPU presence is delegated to agresources.detect_gpus() — the same probe
    the process-wide agResourcePool singleton uses — instead of running an
    independent nvidia-smi/rocm-smi subprocess here. Every agSandbox()
    construction used to pay its own full subprocess round-trip just to pick
    a CLI flag; on a busy shared GPU host that adds up to real contention.
    Caching the result (rather than only reusing detect_gpus()'s logic)
    means this now runs at most once per process regardless of how many
    sandboxes get created.
    """
    global _gpu_flags_cache
    if _gpu_flags_cache is not None:
        return _gpu_flags_cache
    with _gpu_flags_lock:
        if _gpu_flags_cache is None:
            if not detect_gpus():
                _gpu_flags_cache = []
            elif shutil.which("nvidia-smi"):
                _gpu_flags_cache = ["--gpus", "all"]
            else:
                _gpu_flags_cache = ["--device", "/dev/kfd", "--device", "/dev/dri"]
        return _gpu_flags_cache


class agSandboxConfig(_AgConfigViewBase):
    """View over an ``agConfig`` exposing ``agSandbox``'s own vocabulary
    (image, mounts) scoped to the ``"agSandbox"`` namespace.

    ``base_image`` is a flat registered field, so it's just sugar over the
    inherited ``update()``. ``mounts`` is a ``dict[str, tuple[str, str, str]]``
    built incrementally (``add_mount``/``remove_mount`` read-modify-write it)
    -- that doesn't fit ``update()``'s one-field-at-a-time-overwrite model, so
    it's layered on top here directly via the raw ``agConfig`` get/set calls,
    same as before this class subclassed ``_AgConfigViewBase``.
    """

    _OWNER = "agSandbox"

    @property
    def base_image(self) -> str:
        return self._agconfig.get_static(self._OWNER, "base_image", _AgSandboxFields.base_image.default)

    def set_base_image(self, image: str) -> "agSandboxConfig":
        return self.update(base_image=image)

    @property
    def mounts(self) -> dict[str, tuple[str, str, str]]:
        return self._agconfig.get_static(self._OWNER, "mounts", {})

    def add_mount(self, name: str, host_path, container_path: str, mode: str = "rw") -> "agSandboxConfig":
        # Raw (non-locking) read: this is a builder mutating a not-yet-consumed
        # config, not a consumer resolving it — going through the `mounts`
        # property here would lock the key via get_static() on its own first
        # call and then immediately fail the .set() below.
        current = self._agconfig.get(self._OWNER, "mounts", {})
        mounts = {**current, name: (str(host_path), container_path, mode)}
        self._agconfig.set(self._OWNER, "mounts", mounts)
        return self

    def remove_mount(self, name: str) -> "agSandboxConfig":
        current = self._agconfig.get(self._OWNER, "mounts", {})
        mounts = dict(current)
        mounts.pop(name, None)
        self._agconfig.set(self._OWNER, "mounts", mounts)
        return self


class agSandbox(_AgSandboxFields):
    """Manages a single container for one agent via docker or podman.

    All filesystem operations route through ``docker exec`` / ``podman exec``.
    Background PIDs spawned by bash calls are tracked in ``_watched_pids`` and
    the outer monitoring loop in ``agent._task()`` waits for them before
    resolving the skill future.
    """

    def _resolve_image(self, name: str) -> str:
        """Prefix bare image names with ``localhost/`` for Podman.

        Podman requires fully-qualified names when no unqualified-search
        registries are configured in /etc/containers/registries.conf.
        Docker accepts bare names fine, so the prefix is Podman-only.
        """
        if self._runtime == "podman" and "/" not in name:
            return f"localhost/{name}"
        return name

    def __init__(
        self,
        agname: str,
        checkpoint_image: str | None = None,
        agconfig: "agConfig | None" = None,
    ) -> None:
        self._agname   = agname
        self._runtime  = get_container_runtime()
        # Held by agskill for the full duration of a skill run so a sandbox
        # shared across agents is never driven by more than one skill run
        # at a time. See agskill.py's _task().
        self._lock     = threading.RLock()
        self._gpu_id:  int | None           = None
        self._gpu_virtual: bool             = False   # LLM has called reserve_gpu
        self._gpu_acquire_fn                = None    # pool.acquire_gpu, set by make_gpu_reserve
        self._gpu_release_fn                = None    # pool.release_gpu, set by make_gpu_reserve
        self._cpu_acquired: float           = 0.0
        self._memory_acquired_mb: int       = 0
        self._watched_pids: dict[int, float] = {}
        self._baseline_pids: set[int]        = set()
        self._daemon_pids:   set[int]        = set()
        self._started   = False
        self._destroyed = False
        self._checkpoint_image: str | None = checkpoint_image
        # Cloned so this sandbox's own agconfig is independent of the
        # caller's -- mutating the caller's original agConfig afterward does
        # not affect this sandbox. To change it live, mutate
        # sandbox._agconfig (or one of its owner views) directly.
        self._agconfig: "agConfig | None"  = agconfig.clone() if agconfig is not None else None

        # Container name is fixed at creation time using the main-process PID
        # prefix so that worker processes (with different PIDs) use the correct name.
        self._name = f"sandbox-{_RUN_ID}-{agname}"

        # Resolve image/mounts once, here, rather than lazily in
        # _ensure_started() -- a running container is physically fixed once
        # created, so this is a tier-2 (object-static) read: it locks these
        # keys on *this* agconfig instance against further mutation.
        self._gpu_flags     = _gpu_flags()
        self._base_image = self.base_image
        self._vol_flags: list[str] = []
        for host, container, mode in agSandboxConfig(self._agconfig).mounts.values():
            host_path = Path(host)
            host_path.mkdir(parents=True, exist_ok=True)
            self._vol_flags += ["-v", f"{host_path.resolve()}:{container}:{mode}"]

    def change_config(self, agconfig: "agConfig | None") -> None:
        """Replace this sandbox's agconfig with a clone of the given one.

        Only affects fields read live (DynamicConfigParam) going forward --
        the container's image and mounts were resolved once at construction
        (tier-2, physically fixed once the container exists) and are not
        re-resolved here."""
        self._agconfig = agconfig.clone() if agconfig is not None else None

    def __getstate__(self) -> dict:
        # threading.RLock isn't picklable — custom tools with run_in_subprocess=True
        # (the default) get cloudpickled to a worker process, so this must not crash.
        # A lock is process-local anyway, so there's nothing meaningful to carry over.
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def _container_running(self) -> bool:
        """Return True if the named container is currently running in Docker/Podman."""
        result = self._run(
            [self._runtime, "inspect", "--format", "{{.State.Running}}", self._name],
            check=False, timeout=self.inspect_timeout_s,
        )
        return result.returncode == 0 and result.stdout.strip() == b"true"

    def _container_status(self) -> str:
        """Return the container state string: 'running', 'exited', 'created', etc., or '' if not found."""
        result = self._run(
            [self._runtime, "inspect", "--format", "{{.State.Status}}", self._name],
            check=False, timeout=self.inspect_timeout_s,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.decode("utf-8", errors="replace").strip()

    def _ensure_started(self) -> None:
        """Start the Docker container on first use.

        Called lazily by _container_exec() so containers are only created when
        an agent actually needs sandboxed execution (bash, file I/O, etc.).
        Tasks that complete using only host-side tools (webfetch, todowrite,
        find_papers, …) never start a container at all.

        Invariant: after stop() the container does not exist.  The only two
        states we handle here are therefore:
          - running  → reuse (worker-reuse path, does NOT acquire the semaphore)
          - absent   → docker run (acquires semaphore)
        Any leftover container in another state is force-removed first.
        """
        if self._started:
            return
        name = self._name
        if self._container_running():
            # Reuse an already-running container — it already holds a
            # keyring slot so we must NOT acquire _container_semaphore here.
            _live_sandboxes.add(self)
            self._started = True
            self._baseline_pids = self._snapshot_pids()
            return
        # Remove any leftover container in a non-running state (created,
        # exited, dead, …) that stop() failed to clean up.
        if self._container_status():
            self._rm_container(name)
        if self._runtime == "docker":
            _container_semaphore.acquire()
        try:
            if self._checkpoint_image is not None:
                # Restart from last committed checkpoint (set by stop(commit=True)).
                # /workspace and all state from the previous tool call are preserved.
                image = self._checkpoint_image
                run_cmd = (
                    [self._runtime, "run", "-d", "--init", "--name", name]
                    + self._gpu_flags + self._vol_flags
                    + [image, "tail", "-f", "/dev/null"]
                )
                self._run_with_conflict_retry(run_cmd, name)
                # Keep _checkpoint_image — not a one-shot restore, needed for future restarts.
            else:
                image = self._resolve_image(self._base_image)
                _pool_fields = _AgResourcePoolFields(self._agconfig)
                limit_flags = [f"--memory={_pool_fields.idle_memory}"]
                if self._cfs_supported():
                    limit_flags.append(f"--cpus={_pool_fields.idle_cpus}")
                run_cmd = (
                    [self._runtime, "run", "-d", "--init", "--name", name]
                    + limit_flags + self._gpu_flags + self._vol_flags
                    + [image, "tail", "-f", "/dev/null"]
                )
                self._run_with_conflict_retry(run_cmd, name)
                self._run(
                    [self._runtime, "exec", name, "mkdir", "-p", "/workspace"],
                    check=True,
                )
        except _ContainerAlreadyRunning:
            # Another process started the container while we were retrying;
            # that process owns the keyring slot — release ours.
            if self._runtime == "docker":
                _container_semaphore.release()
            _live_sandboxes.add(self)
            self._started = True
            self._baseline_pids = self._snapshot_pids()
            return
        except Exception:
            if self._runtime == "docker":
                _container_semaphore.release()
            raise
        _live_sandboxes.add(self)
        self._started = True  # set before _snapshot_pids() to prevent re-entry via _container_exec
        self._baseline_pids = self._snapshot_pids()

    def _run_with_conflict_retry(self, run_cmd: list[str], name: str) -> None:
        """Run a docker run command, retrying on name-conflict/keyring errors
        up to conflict_retry_max_attempts times.

        A "Conflict / already in use" error can arise when a previous docker run
        call failed mid-way (e.g. GPU allocation timeout) and left a container
        object in "Created" state without ever starting.  We force-remove the
        stale entry and retry rather than surfacing an opaque error to the agent.
        """
        _last_stderr = ""
        for attempt in range(self.conflict_retry_max_attempts):
            result = self._run(run_cmd, timeout=self.docker_run_timeout_s)
            if result.returncode == 0:
                return
            stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            _last_stderr = stderr
            conflict = "already in use" in stderr or "Conflict" in stderr
            keyring = "session key" in stderr or (
                "disk quota exceeded" in stderr and "keyring" in stderr
            )
            if keyring:
                # Linux session keyring quota exhausted.  The _container_semaphore
                # prevents our own containers from exceeding the limit, but external
                # processes can consume slots outside our accounting.  Poll the
                # actual keyring free count from /proc until a slot opens up.
                deadline = time.monotonic() + self.keyring_wait_timeout_s
                while time.monotonic() < deadline:
                    if keyring_quota().get("free", 0) > 0:
                        break
                    time.sleep(self.keyring_poll_interval_s)
                # docker run can partially succeed before failing with keyring:
                # it creates the container object (reserving the name) but fails
                # before starting processes.  Remove any such "Created" artifact
                # so the next attempt does not see a spurious name conflict.
                if self._container_status():
                    self._rm_container(name)
            elif conflict:
                if self._container_running():
                    raise _ContainerAlreadyRunning()
                # Leftover container in a non-running state — remove it.
                # Wait until it's actually gone before retrying docker run.
                self._rm_container(name)
                deadline = time.monotonic() + self.container_removal_wait_s
                while time.monotonic() < deadline:
                    if not self._container_status():
                        break
                    time.sleep(self.container_removal_poll_interval_s)
                # If keyring is also full (docker created the object then hit
                # the limit), wait for a slot before retrying — otherwise we'll
                # create another "Created" container and loop on conflicts.
                deadline = time.monotonic() + self.keyring_wait_timeout_s
                while time.monotonic() < deadline:
                    if keyring_quota().get("free", 0) > 0:
                        break
                    time.sleep(self.keyring_poll_interval_s)
                time.sleep(self.conflict_retry_backoff_base_s * (attempt + 1))
            else:
                msg = f"{' '.join(run_cmd[:3])} failed (exit {result.returncode})"
                if stderr:
                    msg += f": {stderr}"
                raise RuntimeError(msg)
        # Final attempt after retries exhausted.
        quota = keyring_quota()
        msg = (
            f"docker run --name {name} failed after retries "
            f"(container name conflict or keyring quota) "
            f"[keyring: {quota['used']}/{quota['max']} used, "
            f"framework semaphore: {_semaphore_held_count()} held]"
        )
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

    def _snapshot_pids(self) -> set[int]:
        """Return the set of all live PIDs currently in the container, excluding
        the snapshot shell itself so that monitoring shells are not mistaken
        for user-spawned processes."""
        out, _ = self._container_exec(
            "__SELF=$$\n"
            "for __d in /proc/[0-9]*; do\n"
            "  [ -f \"$__d/status\" ] || continue\n"
            "  __p=${__d##*/}\n"
            "  [ \"$__p\" != \"$__SELF\" ] && echo \"$__p\"\n"
            "done",
            timeout=self.inspect_timeout_s, shell="sh",
        )
        pids: set[int] = set()
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
        return pids

    def _run(
        self,
        args: list[str],
        *,
        check: bool = False,
        input: bytes | None = None,
        timeout: int = _AgSandboxFields.DEFAULT_EXEC_TIMEOUT_S,
    ) -> subprocess.CompletedProcess[bytes]:
        with _get_docker_semaphore():
            try:
                return subprocess.run(
                    args,
                    input=input,
                    capture_output=True,
                    timeout=timeout,
                    check=check,
                )
            except subprocess.CalledProcessError as e:
                err = (e.stderr or b"").decode("utf-8", errors="replace").strip()
                msg = f"{' '.join(args)} failed (exit {e.returncode})"
                if err:
                    msg += f": {err}"
                raise RuntimeError(msg) from e

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
        timeout: int = _AgSandboxFields.DEFAULT_EXEC_TIMEOUT_S,
        stdin: bytes | None = None,
        shell: str = "bash",
    ) -> tuple[str, int]:
        """Run a raw shell command inside the container."""
        self._ensure_started()
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

    def exec(
        self,
        cmd: str,
        workdir: str = "/workspace",
        timeout: int = _AgSandboxFields.DEFAULT_EXEC_TIMEOUT_S,
    ) -> tuple[str, int]:
        """Run a user command inside the container."""
        # Lazily acquire a physical GPU now that we have a bash call to run.
        # Blocks (polling every 0.25 s) until any GPU in the pool is free.
        if self._gpu_virtual and self._gpu_id is None and self._gpu_acquire_fn is not None:
            self._gpu_id = self._gpu_acquire_fn()

        # Restrict GPU access to the acquired GPU ID. Set both CUDA_VISIBLE_DEVICES
        # (NVIDIA/CUDA) and HIP_VISIBLE_DEVICES (AMD/ROCm) so only the leased
        # device is accessible regardless of which runtime is present.
        # "NoDevFiles" hides all GPUs when no GPU has been acquired.
        # An empty string would leave CUDA_VISIBLE_DEVICES unset, making all GPUs visible.
        gpu_id = str(self._gpu_id) if self._gpu_id is not None else "NoDevFiles"
        hf_token = os.environ.get("HF_TOKEN", "")
        hf_export = f"export HF_TOKEN={hf_token}\n" if hf_token else ""
        env_export = f"export CUDA_VISIBLE_DEVICES={gpu_id}\nexport HIP_VISIBLE_DEVICES={gpu_id}\n{hf_export}"

        wrapped = (
            f"exec 2>&1\n"      # merge stderr into stdout so the BGPIDS marker is never split
            # Snapshot every live PID in the container before the command runs.
            # Filtering on /proc/<N>/status avoids races with short-lived kernel threads.
            f"__AGENCY_BEFORE=$(for __d in /proc/[0-9]*; do"
            f" [ -f \"$__d/status\" ] && echo \"${{__d##*/}}\"; done | tr '\\n' ' ')\n"
            f"__AGENCY_SHELL=$$\n"
            f"{env_export}{cmd}\n"
            f"__AGENCY_RC=$?\n"
            # Diff /proc after the command: any PID not in the before-snapshot
            # and not the shell itself was spawned by the command.
            f"__AGENCY_BGPIDS=''\n"
            f"for __d in /proc/[0-9]*; do\n"
            f"  [ -f \"$__d/status\" ] || continue\n"
            f"  __p=${{__d##*/}}\n"
            f"  case \" $__AGENCY_BEFORE $__AGENCY_SHELL \" in\n"
            f"    *\" $__p \"*) ;;\n"
            f"    *) __AGENCY_BGPIDS=\"$__AGENCY_BGPIDS $__p\" ;;\n"
            f"  esac\n"
            f"done\n"
            f"printf '\\n{_BGPIDS_MARKER}%s' \"$__AGENCY_BGPIDS\"\n"
            f"exit $__AGENCY_RC"
        )

        output, rc = self._container_exec(wrapped, workdir=workdir, timeout=timeout)

        if _BGPIDS_MARKER in output:
            parts = output.rsplit(_BGPIDS_MARKER, 1)
            clean_output = parts[0].rstrip("\n")
            pids_str = parts[1].strip()
            if pids_str:
                now = time.monotonic()
                for pid_s in pids_str.split():
                    try:
                        self._watched_pids[int(pid_s)] = now
                    except ValueError:
                        pass
        else:
            clean_output = output

        # Release the physical GPU if no background processes remain.  The
        # /proc diff can catch transient PIDs that already exited by the time
        # we parse __BGPIDS__, so re-verify liveness when _watched_pids is
        # non-empty — get_live_pids() prunes dead entries and releases the GPU
        # if none survive the check.
        if self._watched_pids and self._gpu_virtual and self._gpu_id is not None:
            self.get_live_pids()
        elif not self._watched_pids and self._gpu_virtual and self._gpu_id is not None:
            self._gpu_release_fn(self._gpu_id)
            self._gpu_id = None

        return clean_output, rc

    def read_file(self, path: str) -> str:
        """Read a text file from the container.

        Raises:
            IsADirectoryError: if the path exists but is a directory.
            UnicodeDecodeError: if the file exists but is not valid UTF-8.
            FileNotFoundError: if the path does not exist.
        """
        import base64
        b64, rc = self._container_exec(
            f"base64 {shlex.quote(path)}", timeout=self.file_io_timeout_s, shell="sh"
        )
        if rc != 0:
            _, dir_rc = self._container_exec(
                f"test -d {shlex.quote(path)}", timeout=self.exec_quick_timeout_s, shell="sh"
            )
            if dir_rc == 0:
                raise IsADirectoryError(f"Path is a directory, not a file: {path}")
            raise FileNotFoundError(f"Not found in container: {path}")
        raw = base64.b64decode(b64.strip())
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnicodeDecodeError(
                exc.encoding, exc.object, exc.start, exc.end,
                f"File {path} contains binary data and is not UTF-8 text",
            ) from None

    def read_file_bytes(self, path: str) -> bytes:
        """Read raw bytes from a file in the container.

        Like read_file but returns bytes without any UTF-8 decode attempt.
        Use for binary files (images, audio, compiled artifacts, etc.).

        Raises:
            IsADirectoryError: if the path exists but is a directory.
            FileNotFoundError: if the path does not exist.
        """
        import base64
        b64, rc = self._container_exec(
            f"base64 {shlex.quote(path)}", timeout=self.file_io_timeout_s, shell="sh"
        )
        if rc != 0:
            _, dir_rc = self._container_exec(
                f"test -d {shlex.quote(path)}", timeout=self.exec_quick_timeout_s, shell="sh"
            )
            if dir_rc == 0:
                raise IsADirectoryError(f"Path is a directory, not a file: {path}")
            raise FileNotFoundError(f"Not found in container: {path}")
        return base64.b64decode(b64.strip())

    def write_file_bytes(self, path: str, data: bytes) -> None:
        """Write raw bytes to a file in the container.

        Use for binary files. The data is base64-encoded on the host and
        decoded inside the container, avoiding any shell-quoting issues with
        arbitrary byte sequences.
        """
        import base64
        b64 = base64.b64encode(data).decode("ascii")
        quoted = shlex.quote(path)
        sh_cmd = (
            f"mkdir -p $(dirname {quoted}) && "
            f"printf '%s' {shlex.quote(b64)} | base64 -d > {quoted}"
        )
        _, rc = self._container_exec(sh_cmd, timeout=self.file_io_timeout_s, shell="sh")
        if rc != 0:
            raise OSError(f"Failed to write binary file {path} in container")

    def write_file(self, path: str, content: str) -> None:
        quoted = shlex.quote(path)
        sh_cmd = f"mkdir -p $(dirname {quoted}) && cat > {quoted}"
        _, rc = self._container_exec(
            sh_cmd, stdin=content.encode("utf-8"), timeout=self.file_io_timeout_s, shell="sh"
        )
        if rc != 0:
            raise OSError(f"Failed to write {path} in container")

    def update_limits(
        self,
        *,
        cpus: float | None = None,
        memory: str | None = None,
    ) -> None:
        """Live-update container CPU/memory limits."""
        if not self._started:
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
        does not require the container to be running).  Also handles the case
        where the container was started by a worker process and ``_started``
        is still False in the main process.
        """
        if not self._started:
            if not self._container_running():
                return False
        self._run(
            [self._runtime, "commit", self._container_name(), tag],
            check=True,
            timeout=self.commit_timeout_s,
        )
        return True

    def stop(self, *, commit: bool = False) -> None:
        """Stop and remove the container, releasing its session keyring and GPU.

        If commit=True, the container filesystem is committed to an image first
        so _ensure_started() can recreate from it on the next tool call.  Pass
        commit=True after a successful sandbox tool call; commit=False after a
        failure to discard the dirty state and revert to the last checkpoint.
        """
        if not self._started:
            # Worker-process scenario: _started is False in the calling process
            # even though a worker may have started the container.
            if not self._container_running():
                return
        # Release GPU so other agents can use it while the container is gone.
        if self._gpu_virtual and self._gpu_id is not None:
            self._gpu_release_fn(self._gpu_id)
            self._gpu_id = None
        # Clear PID tracking — remove kills all processes.
        self._watched_pids = {}
        self._baseline_pids = set()
        if commit:
            tag = self._lifecycle_tag()
            # Capture the current image ID before overwriting the tag so we
            # can delete it afterward — committing to an existing tag leaves
            # the old image dangling (untagged but still on disk).
            old_image_id: str | None = None
            try:
                result = self._run(
                    [self._runtime, "inspect", "--format={{.Id}}", tag],
                    check=False, timeout=self.stop_inspect_timeout_s,
                )
                if result and result.returncode == 0:
                    old_image_id = result.stdout.decode("utf-8", errors="replace").strip() or None
            except Exception:
                pass
            for _attempt in range(self.commit_retry_attempts):
                try:
                    self._run(
                        [self._runtime, "commit", self._container_name(), tag],
                        check=True, timeout=self.commit_timeout_s,
                    )
                    self._checkpoint_image = tag
                    break
                except Exception as _e:
                    if _attempt == self.commit_retry_attempts - 1:
                        print(
                            f"[agsandbox] WARNING: docker commit {self._container_name()} → {tag} "
                            f"failed after {self.commit_retry_attempts} attempts: {_e}",
                            file=__import__("sys").stderr, flush=True,
                        )
                    else:
                        time.sleep(self.commit_retry_backoff_s)
            # Delete the previous image now that the tag points to the new one.
            # Only delete if no containers are currently using it — a fork may still
            # be running from the same image.  The fork's own stop() will delete it
            # once its container is gone.
            if old_image_id and self._checkpoint_image == tag:
                try:
                    in_use = self._run(
                        [self._runtime, "ps", "-a",
                         "--filter", f"ancestor={old_image_id}",
                         "--format", "{{.ID}}"],
                        check=False, timeout=self.stop_ps_check_timeout_s,
                    )
                    if in_use and in_use.stdout.strip():
                        pass  # containers still running from this image — leave it
                    else:
                        self._rmi(old_image_id)
                except Exception as _e:
                    print(
                        f"[agsandbox] WARNING: could not check/delete old image {old_image_id}: {_e}",
                        file=__import__("sys").stderr, flush=True,
                    )
        name = self._container_name()
        for _attempt in range(self.rm_retry_attempts):
            try:
                self._rm_container(name)
                break
            except Exception:
                if _attempt == self.rm_retry_attempts - 1:
                    import traceback as _tb
                    print(
                        f"[agsandbox] WARNING: docker rm -f {name} failed after {self.rm_retry_attempts} attempts:\n"
                        f"{_tb.format_exc()}",
                        file=__import__("sys").stderr, flush=True,
                    )
                else:
                    time.sleep(self.rm_retry_backoff_s)
        if self._runtime == "docker":
            _container_semaphore.release()
        self._started = False

    def restore(self, tag: str) -> None:
        """Restore the sandbox to a previously committed image snapshot.

        Stops the running container (killing watched pids first), then restarts
        from *tag*.  The image is kept so it can be reused on subsequent
        failures during the same skill run.
        """
        if self._started:
            if self._watched_pids:
                pids = " ".join(str(p) for p in self._watched_pids)
                try:
                    self._container_exec(
                        f"kill {pids} 2>/dev/null; true", timeout=self.exec_quick_timeout_s, shell="sh"
                    )
                except Exception as _e:
                    print(f"[agsandbox] WARNING: failed to kill PIDs {pids} in {self._name} during restore: {_e}")
            self._rm_container(self._container_name())
            self._started = False
            self._watched_pids = {}
            self._baseline_pids = set()
        self._checkpoint_image = tag
        self._ensure_started()

    def release_daemon(self, pid: int) -> None:
        """Move *pid* out of the monitored set into the daemon set.

        The process and all its future descendants will continue running in the
        container but will never block the outer monitoring loop.
        """
        self._daemon_pids.add(pid)
        self._watched_pids.pop(pid, None)

    def get_live_pids(self) -> set[int]:
        if not self._watched_pids:
            return set()

        # Read pid, ppid, state, and comm name for every entry in /proc,
        # excluding the monitoring shell itself.
        script = (
            "__SELF=$$\n"
            "for __d in /proc/[0-9]*; do\n"
            "  [ -f \"$__d/status\" ] || continue\n"
            "  __p=${__d##*/}\n"
            "  [ \"$__p\" = \"$__SELF\" ] && continue\n"
            "  __ppid=$(awk '/^PPid:/{print $2}' $__d/status 2>/dev/null)\n"
            "  __st=$(awk '/^State:/{print $2}' $__d/status 2>/dev/null)\n"
            "  __nm=$(awk '/^Name:/{print $2}' $__d/status 2>/dev/null)\n"
            "  echo \"$__p $__ppid $__st $__nm\"\n"
            "done"
        )
        output, _ = self._container_exec(script, timeout=self.inspect_timeout_s, shell="sh")

        proc_info: dict[int, tuple[int, str, str]] = {}   # pid → (ppid, state, name)
        for line in output.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                pid  = int(parts[0])
                ppid = int(parts[1])
                state = parts[2] if len(parts) > 2 else "?"
                name  = parts[3] if len(parts) > 3 else ""
            except ValueError:
                continue
            proc_info[pid] = (ppid, state, name)

        # Mark NVIDIA container-runtime helper processes as system PIDs and
        # propagate that status to their children.  The NVIDIA toolkit entrypoint
        # (comm = "nvidia_entrypoi") periodically spawns short-lived GPU check
        # processes (cudaCheck, deviceQuery, …) that are not user processes and
        # must not be counted as live background work.
        #
        # Baseline PIDs are always excluded — PID 1 in GPU containers IS named
        # "nvidia_entrypoi" (it is the init process), so we must not add it to
        # system_pids or every process reparented to it after its parent exits
        # would be incorrectly filtered.
        system_pids: set[int] = {
            pid for pid, (_, _, name) in proc_info.items()
            if name == "nvidia_entrypoi" and pid not in self._baseline_pids
        }
        changed = True
        while changed:
            changed = False
            for pid, (ppid, _, _) in proc_info.items():
                if (pid not in system_pids
                        and pid not in self._baseline_pids
                        and ppid in system_pids):
                    system_pids.add(pid)
                    changed = True

        # Propagate daemon status down the tree: if a process's parent is a
        # daemon, the child inherits that status and is also excluded from
        # monitoring.  Repeat until no new daemons are discovered.
        changed = True
        while changed:
            changed = False
            for pid, (ppid, _, _) in proc_info.items():
                if pid not in self._daemon_pids and ppid in self._daemon_pids:
                    self._daemon_pids.add(pid)
                    self._watched_pids.pop(pid, None)
                    changed = True

        # A PID is alive if it exists in /proc, is not baseline, not a system
        # process (NVIDIA runtime helper), not a daemon, and not a zombie.  Any
        # newly discovered non-baseline PID is added to _watched_pids so the
        # outer loop waits for it.
        alive: set[int] = set()
        now = time.monotonic()
        for pid, (_, state, _) in proc_info.items():
            if (pid in self._baseline_pids or pid in system_pids
                    or pid in self._daemon_pids or state == "Z"):
                continue
            alive.add(pid)
            if pid not in self._watched_pids:
                self._watched_pids[pid] = now

        # Prune _watched_pids entries that are no longer alive.
        for pid in set(self._watched_pids):
            if pid not in alive:
                del self._watched_pids[pid]

        # Release the physical GPU once all watched processes have finished.
        if not alive and self._gpu_virtual and self._gpu_id is not None:
            self._gpu_release_fn(self._gpu_id)
            self._gpu_id = None

        return alive

    def pid_status_summary(self) -> str:
        live = self.get_live_pids()
        if not live:
            return "no background processes running"
        now = time.monotonic()
        parts = []
        for pid in sorted(live):
            elapsed = int(now - self._watched_pids.get(pid, now))
            mins, secs = divmod(elapsed, 60)
            parts.append(f"PID {pid} (running {mins}m {secs}s)")
        return ", ".join(parts)

    def release_resources(self, pool: "agResourcePool | None" = None) -> None:
        self._gpu_virtual = False
        if self._gpu_id is not None:
            if pool is not None:
                pool.release_gpu(self._gpu_id)
            elif self._gpu_release_fn is not None:
                self._gpu_release_fn(self._gpu_id)
            self._gpu_id = None
        if pool is not None and (self._cpu_acquired or self._memory_acquired_mb):
            pool.notify_cpu_released(self._cpu_acquired, self._memory_acquired_mb)
            self._cpu_acquired = 0.0
            self._memory_acquired_mb = 0
        if pool is not None:
            try:
                self.update_limits(cpus=pool.idle_cpus, memory=pool.idle_memory)
            except Exception as _e:
                print(f"[agsandbox] WARNING: update_limits failed for {self._name}: {_e}")

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        _live_sandboxes.discard(self)
        # _started is only set to True in the process that called _ensure_started.
        # When tools run in worker processes the main process always has
        # _started=False, even though a container may be running.  Always attempt
        # cleanup — docker rm -f is a no-op when the container doesn't exist.
        container_name = self._container_name()

        if self._started and self._watched_pids:
            pids = " ".join(str(p) for p in self._watched_pids)
            try:
                self._container_exec(
                    f"kill {pids} 2>/dev/null; true", timeout=self.exec_quick_timeout_s, shell="sh"
                )
            except Exception as _e:
                print(f"[agsandbox] WARNING: failed to kill PIDs {pids} in {container_name}: {_e}")

        # Check before rm so we know whether a keyring slot must be released.
        had_container = self._runtime == "docker" and bool(
            self._started or self._container_running()
        )

        try:
            if self._container_status():
                self._rm_container(container_name)
        finally:
            if had_container:
                _container_semaphore.release()

        # Remove the checkpoint image and all pre-tool snapshots created during
        # this sandbox's lifetime.
        if self._checkpoint_image:
            try:
                self._rmi(self._checkpoint_image, force=True)
            except Exception as _e:
                print(f"[agsandbox] WARNING: checkpoint image cleanup failed for {container_name}: {_e}")
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
            print(f"[agsandbox] WARNING: pretool image cleanup failed for {container_name}: {_e}")

    def remove_files(self, paths: list[str]) -> None:
        """Delete sandbox files previously written by offload/agfile helpers."""
        import shlex as _shlex
        for path in paths:
            try:
                self._container_exec(f"rm -f {_shlex.quote(path)}", shell="sh")
            except Exception as _e:
                print(f"[agsandbox] WARNING: failed to remove offloaded file {path}: {_e}")

    def fork(self, new_agname: str, agconfig: "agConfig | None" = None) -> "agSandbox":
        """Return a new agSandbox for *new_agname* starting from this sandbox's
        current checkpoint image.  If no checkpoint exists the fork starts fresh.

        When *agconfig* is not given, the fork inherits this sandbox's own
        agconfig unchanged (rather than silently re-reading whatever
        base_image happens to be at fork time).

        The caller owns the returned sandbox and is responsible for calling
        destroy() on it when done.
        """
        cfg = agconfig if agconfig is not None else self._agconfig
        fork_sb = agSandbox(new_agname, agconfig=cfg)
        if self._checkpoint_image:
            agSandbox.tag_image(self._checkpoint_image, fork_sb._lifecycle_tag())
            fork_sb._checkpoint_image = fork_sb._lifecycle_tag()
        return fork_sb

    def _lifecycle_tag(self) -> str:
        return f"agency/lifecycle-{self._name}".lower()

    # ------------------------------------------------------------------
    # Static helpers — image-level operations used for checkpointing.
    # These operate on image tags, not containers, so they don't need
    # a sandbox instance.
    # ------------------------------------------------------------------

    @staticmethod
    def tag_image(source: str, dest: str) -> None:
        """Retag an image from *source* to *dest* (docker/podman tag)."""
        runtime = get_container_runtime()
        with _get_docker_semaphore():
            subprocess.run(
                [runtime, "tag", source, dest],
                capture_output=True, check=True,
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
                capture_output=True, check=True, timeout=timeout,
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
                input=image_bytes, capture_output=True, check=True, timeout=timeout,
            )

    def wait_for_processes(
        self,
        skill_name: str,
        term: "agterm | None",
        log: "aglog | None" = None,
        agname: str = "",
        ping_interval_s: float = _AgSandboxFields.WAIT_PING_INTERVAL_S,
        poll_interval_s: float = _AgSandboxFields.WAIT_POLL_INTERVAL_S,
        state_fn: "Callable | None" = None,
    ) -> "str | None":
        """Wait for sandbox background processes after the LLM produces a final answer.

        Returns None if the sandbox is already clean (no action needed).
        Otherwise polls until all PIDs exit or ping_interval_s elapses, then
        returns a user-facing message to inject into the conversation so the
        LLM can act on the outcome.
        """
        _term            = term
        _log             = log
        _agname          = agname
        _ping_interval_s = ping_interval_s
        _poll_interval_s = poll_interval_s
        _state           = state_fn

        watched = getattr(self, "_watched_pids", None)
        if not isinstance(watched, dict) or not watched:
            return None
        get_live = self.get_live_pids
        if not get_live():
            return None

        summary = self.pid_status_summary()
        if _state:
            _state("proc_wait", skill=skill_name)
        if _term:
            _term.log("PROCS ▶  ", f"{skill_name}  monitoring: {summary}")
        if _log:
            _log._lifecycle("procs_started", agname=_agname, skill=skill_name,
                            pids=list(get_live()), summary=summary)

        deadline = time.monotonic() + _ping_interval_s
        while time.monotonic() < deadline:
            time.sleep(_poll_interval_s)
            if not get_live():
                break

        live_now = get_live()

        if not live_now:
            if _term:
                _term.log("PROCS ✓  ", f"{skill_name}  all processes completed, re-entering agent")
            if _log:
                _log._lifecycle("procs_completed", agname=_agname, skill=skill_name)
            return (
                "Background processes have completed. "
                "Read their output and act on the results."
            )

        summary = self.pid_status_summary()
        if _term:
            _term.log("PROCS ⏳  ", f"{skill_name}  still running: {summary}")
        if _log:
            _log._lifecycle("procs_ping", agname=_agname, skill=skill_name,
                            pids=list(live_now), summary=summary)
        return (
            f"Background processes are still running: {summary}. "
            f"You may check their output, wait, or proceed if appropriate. "
            f"If any of these processes are intentional long-running services "
            f"(daemons, servers, monitors) that should not block completion, "
            f"call daemon_release(pid) for each such PID to release it from monitoring."
        )

