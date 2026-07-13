"""Shared docker/podman plumbing.

Runtime detection (`get_container_runtime()`), the subprocess-call throttle
(`_get_docker_semaphore()`), the session-keyring-quota diagnostics
(`keyring_quota()`/`_semaphore_held_count()` -- read by the shared retry
logic below even though only Docker ever actually exhausts that quota, see
`.docker`'s module docstring), and `_ContainerBackendBase` -- the base class
`.docker._DockerBackend` and `.podman._PodmanBackend` both subclass for
everything that doesn't differ between the two runtimes, which is nearly
everything. See those two modules for the handful of things that do.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
import uuid as _uuid
from pathlib import Path

from ..agconfig import agConfig
from ..agresources import detect_gpus, _AgResourcePoolFields
from .base import AgSandboxBackendFields, agsandbox_backend

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


# Hard cap on the number of simultaneously running Docker containers, derived from
# the Linux kernel session-keyring quota.  Each running Docker container holds one
# session keyring against the user that ran `docker run`; when total keys reach
# /proc/sys/kernel/keys/maxkeys the next docker run fails with
# "unable to create session key: disk quota exceeded".
# Podman is exempt: rootless Podman uses user namespaces with independent keyring
# namespaces and is not subject to this quota -- see .docker's module docstring
# for why the semaphore built from this quota is only ever acquired/released by
# _DockerBackend, even though it (and the diagnostics below) live in this shared
# module so _run_with_conflict_retry() -- shared by both runtimes -- can read them.
# multiprocessing.Semaphore is backed by a POSIX IPC semaphore so the limit is
# enforced across all worker processes (which run _ensure_started) and the main
# process (which calls stop/destroy).
def _docker_container_limit() -> int:
    """Return the concurrent-Docker-container cap derived from the kernel keyring quota."""
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
    """Return 'held/limit' for the Docker container-concurrency semaphore, or
    '?/limit' if unreadable.

    Uses sem_getvalue() via the internal _semlock on POSIX (Linux).  The count
    reflects this process's view only — other unrelated processes are not
    tracked by our semaphore but do consume system keyring slots, so comparing
    this number with keyring_quota()['used'] reveals how many slots belong to
    external processes.
    """
    from .docker import _container_semaphore

    limit = _docker_container_limit()
    try:
        available = _container_semaphore._semlock._get_value()
        held = limit - available
    except Exception:
        held = "?"
    return f"{held}/{limit}"


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
    independent nvidia-smi/rocm-smi subprocess here. Every sandbox backend
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


class _ContainerBackendBase(agsandbox_backend):
    """Manages a single container for one agent via docker or podman.

    Subclassed by `.docker._DockerBackend` and `.podman._PodmanBackend`,
    which each hardcode their own `_runtime` string and override only the
    handful of things that genuinely differ between the two: whether bare
    image names need a `localhost/` prefix (`_resolve_image`, Podman-only —
    Podman requires fully-qualified names when no unqualified-search
    registries are configured in /etc/containers/registries.conf, Docker
    accepts bare names fine), and whether starting/stopping a container needs
    to hold the session-keyring-derived concurrency slot
    (`_acquire_runtime_slot`/`_release_runtime_slot`, Docker-only — rootless
    Podman uses independent per-namespace keyrings and isn't subject to that
    quota at all). Everything else — command building, retries, checkpointing,
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
        """No-op by default. Overridden by _DockerBackend to acquire the
        session-keyring-derived container-concurrency semaphore before
        starting a new container."""
        return

    def _release_runtime_slot(self) -> None:
        """Release whatever _acquire_runtime_slot() acquired, if anything."""
        return

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
        self._agname = agname
        self._gpu_id: int | None = None
        self._gpu_virtual: bool = False  # LLM has called reserve_gpu
        self._gpu_acquire_fn = None  # pool.acquire_gpu, set by make_gpu_reserve
        self._gpu_release_fn = None  # pool.release_gpu, set by make_gpu_reserve
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
        self._gpu_flags = _gpu_flags()
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
        """
        if self._started:
            return
        name = self._name
        if self._container_running():
            # Reuse an already-running container — it already holds whatever
            # slot _acquire_runtime_slot() would take, so we must NOT acquire
            # it again here.
            self._started = True
            self._baseline_pids = self._snapshot_pids()
            return
        # Remove any leftover container in a non-running state (created,
        # exited, dead, …) that stop() failed to clean up.
        if self._container_status():
            self._rm_container(name)
        self._acquire_runtime_slot()
        try:
            if self._checkpoint_image is not None:
                # Restart from last committed checkpoint (set by stop(commit=True)).
                # /workspace and all state from the previous tool call are preserved.
                image = self._checkpoint_image
                run_cmd = (
                    [self._runtime, "run", "-d", "--init", "--name", name]
                    + self._gpu_flags
                    + self._vol_flags
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
                    + limit_flags
                    + self._gpu_flags
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
            self._started = True
            self._baseline_pids = self._snapshot_pids()
            return
        except Exception:
            self._release_runtime_slot()
            raise
        self._started = True  # set before _snapshot_pids() to prevent re-entry via _container_exec
        self._baseline_pids = self._snapshot_pids()

    def _run_with_conflict_retry(self, run_cmd: list[str], name: str) -> None:
        """Run a docker/podman run command, retrying on name-conflict/keyring
        errors up to conflict_retry_max_attempts times.

        A "Conflict / already in use" error can arise when a previous run call
        failed mid-way (e.g. GPU allocation timeout) and left a container
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
                # Linux session keyring quota exhausted (Docker-only, see this
                # module's docstring). The runtime-slot semaphore prevents our
                # own containers from exceeding the limit, but external
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

    def _run(
        self,
        args: list[str],
        *,
        check: bool = False,
        input: bytes | None = None,
        timeout: int = AgSandboxBackendFields.DEFAULT_EXEC_TIMEOUT_S,
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
        timeout: int = AgSandboxBackendFields.DEFAULT_EXEC_TIMEOUT_S,
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
        """Stop and remove the container, releasing its runtime slot and GPU.

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
                    break
                except Exception as _e:
                    if _attempt == self.commit_retry_attempts - 1:
                        print(
                            f"[agsandbox_backend] WARNING: docker commit {self._container_name()} → {tag} "
                            f"failed after {self.commit_retry_attempts} attempts: {_e}",
                            file=__import__("sys").stderr,
                            flush=True,
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
        for _attempt in range(self.rm_retry_attempts):
            try:
                self._rm_container(name)
                break
            except Exception:
                if _attempt == self.rm_retry_attempts - 1:
                    import traceback as _tb

                    print(
                        f"[agsandbox_backend] WARNING: docker rm -f {name} failed after {self.rm_retry_attempts} attempts:\n"
                        f"{_tb.format_exc()}",
                        file=__import__("sys").stderr,
                        flush=True,
                    )
                else:
                    time.sleep(self.rm_retry_backoff_s)
        self._release_runtime_slot()
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
                        f"kill {pids} 2>/dev/null; true",
                        timeout=self.exec_quick_timeout_s,
                        shell="sh",
                    )
                except Exception as _e:
                    print(
                        f"[agsandbox_backend] WARNING: failed to kill PIDs {pids} in {self._name} during restore: {_e}"
                    )
            self._rm_container(self._container_name())
            self._started = False
            self._watched_pids = {}
            self._baseline_pids = set()
        self._checkpoint_image = tag
        self._ensure_started()

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
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
                print(
                    f"[agsandbox_backend] WARNING: failed to kill PIDs {pids} in {container_name}: {_e}"
                )

        # Check before rm so we know whether a runtime slot must be released.
        had_container = bool(self._started or self._container_running())

        try:
            if self._container_status():
                self._rm_container(container_name)
        finally:
            if had_container:
                self._release_runtime_slot()

        # Remove the checkpoint image and all pre-tool snapshots created during
        # this sandbox's lifetime.
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
