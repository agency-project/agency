"""Sandbox backend abstraction for agsandbox.

An `agSandbox` instance builds exactly one `agsandbox_backend` from its config
(via `agsandbox_backend.for_config()`) and delegates every sandboxing
operation (exec, file I/O, lifecycle, checkpointing) to it. Backend selection
logic (podman vs. docker vs. a future chroot-based backend, see
`agSandboxBackendConfig.backend`) lives here instead of being hardcoded into
`agSandbox` itself.

Every backend exposes the same surface `agSandbox` uses: `exec()`,
`_container_exec()`, `read_file()`/`write_file()` (+ `_bytes` variants),
`commit()`/`stop()`/`restore()`/`destroy()`, `update_limits()`,
`remove_files()`, `get_live_pids()`/`pid_status_summary()`/`release_daemon()`,
`release_resources()`, `fork()`, and the static image-level helpers
`tag_image()`/`export_image()`/`import_image()`/`delete_image()`.
"""

from __future__ import annotations

import multiprocessing
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid as _uuid
from pathlib import Path
from typing import TYPE_CHECKING

from .agconfig import agConfig, GlobalConfigParam, DynamicConfigParam, _AgConfigViewBase
from .agresources import detect_gpus, _AgResourcePoolFields

if TYPE_CHECKING:
    from .agresources import agResourcePool


# ---------------------------------------------------------------------------
# Backend config -- every tunable used by any backend, as ConfigParam
# descriptors. `backend` (podman/docker/chroot/auto) is the analogue of
# agllm_backend's `provider`. The rest are tier-1 (global) tunables for the
# backend machinery itself (docker/podman daemon calls, the container
# semaphore, the keyring quota) -- unchanged in kind from when they lived on
# agsandbox.py's _AgSandboxFields, just moved to their own owner namespace.
# ---------------------------------------------------------------------------


class AgSandboxBackendFields:
    DEFAULT_EXEC_TIMEOUT_S = (
        120  # Fallback for exec/_container_exec when no explicit timeout is passed.
    )
    SEED_CACHE_TIMEOUT_S = (
        900  # seed_cache_from_image: copying a large pre-baked directory out of an image.
    )
    WAIT_PING_INTERVAL_S = (
        300  # wait_for_processes() fallback, overridden in practice by every real caller.
    )
    WAIT_POLL_INTERVAL_S = (
        5  # wait_for_processes() fallback, overridden in practice by every real caller.
    )

    backend = DynamicConfigParam("agsandbox_backend", default="auto")
    docker_semaphore_limit = GlobalConfigParam("agsandbox_backend", default=16)

    # Timeouts (seconds) for the various classes of docker/podman subprocess call.
    inspect_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # Fast metadata queries: docker inspect, docker ps, nvidia-smi, docker update.
    exec_quick_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # Quick in-container exec calls: kill <pids>, test -d, and similar.
    docker_run_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # docker run: GPU init via the NVIDIA container runtime can take 60+ s under load.
    docker_rm_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # docker rm -f: fast teardown; should complete in a few seconds.
    file_io_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # In-container file I/O via docker exec (base64 read/write, mkdir).
    image_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # docker images list / docker rmi.
    commit_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # docker commit: snapshots a full overlay layer; large workspaces need extra time.
    keyring_wait_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=120
    )  # Maximum time to wait for a keyring slot before abandoning a docker run retry.

    # _docker_container_limit(): concurrent-container cap derived from the kernel
    # session-keyring quota (see that function's docstring for the full rationale).
    container_limit_floor = GlobalConfigParam(
        "agsandbox_backend", default=4
    )  # Never cap below this many concurrent containers.
    container_limit_buffer = GlobalConfigParam(
        "agsandbox_backend", default=5
    )  # Safety margin subtracted from the raw kernel/fallback quota.
    container_limit_fallback = GlobalConfigParam(
        "agsandbox_backend", default=200
    )  # Assumed kernel quota when /proc/sys/kernel/keys/maxkeys isn't readable.

    # _run_with_conflict_retry(): retrying `docker run` on name-conflict/keyring errors.
    conflict_retry_max_attempts = GlobalConfigParam("agsandbox_backend", default=8)
    keyring_poll_interval_s = GlobalConfigParam(
        "agsandbox_backend", default=5
    )  # Poll interval while waiting for a free keyring slot.
    container_removal_wait_s = GlobalConfigParam(
        "agsandbox_backend", default=10
    )  # Max time to wait for a conflicting container to finish being removed.
    container_removal_poll_interval_s = GlobalConfigParam("agsandbox_backend", default=0.5)
    conflict_retry_backoff_base_s = GlobalConfigParam(
        "agsandbox_backend", default=0.5
    )  # Multiplied by attempt number for linear backoff between retries.

    # stop(): commit-then-remove teardown sequence.
    stop_inspect_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=30
    )  # docker inspect (pre-commit image-id lookup).
    commit_retry_attempts = GlobalConfigParam("agsandbox_backend", default=3)
    commit_retry_backoff_s = GlobalConfigParam("agsandbox_backend", default=1)
    stop_ps_check_timeout_s = GlobalConfigParam(
        "agsandbox_backend", default=10
    )  # docker ps (checking whether the old image is still in use).
    rm_retry_attempts = GlobalConfigParam("agsandbox_backend", default=3)
    rm_retry_backoff_s = GlobalConfigParam("agsandbox_backend", default=1)


class agSandboxBackendConfig(_AgConfigViewBase):
    """View over an agConfig for pre-selecting the sandbox backend::

    cfg = agSandboxBackendConfig(backend="docker").agconfig
    ag = agent(agconfig=cfg)
    """

    _OWNER = "agsandbox_backend"


_BGPIDS_MARKER = "__BGPIDS__:"


class _ContainerAlreadyRunning(Exception):
    """Raised by _run_with_conflict_retry when another process has already
    started the same container — the caller should reuse it."""


_RUNTIME: str | None = None

# Per-run ID so concurrent and successive runs never share container/image names.
# UUID avoids PID-reuse collisions and prevents stale lifecycle images from crashed
# runs being accidentally picked up by a new run that happens to get the same PID.
_RUN_ID = f"r{_uuid.uuid4().hex[:8]}"

# Limit the number of containers starting simultaneously.  Each _ContainerBackend
# construction acquires one slot for the duration of its startup sequence (docker
# run + first exec). Without this, a burst of hundreds of parallel agent tasks
# overwhelms the Docker daemon.  All Docker/Podman calls go through _run(), which
# holds this semaphore for the duration of each subprocess call.  Caps concurrent
# daemon calls at 16: the daemon serialises most operations internally (GPU init,
# overlay diff, container teardown), so more than ~16 concurrent calls increase
# contention without reducing wall-clock time.
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
    _fields = AgSandboxBackendFields()
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


_chroot_available_cache: "bool | None" = None
_chroot_available_lock = threading.Lock()


def chroot_available() -> bool:
    """Return True if unprivileged user namespaces + chroot are usable on this
    host, cached for the process lifetime.

    Two checks, since either alone can give a false positive:
    ``/proc/sys/kernel/unprivileged_userns_clone`` (when present -- it's
    Debian/Ubuntu-specific; distros that ship it enabled by default in the
    upstream kernel don't have the file at all) must not be explicitly
    disabled, and a live ``unshare --user --map-root-user`` must actually
    succeed -- AppArmor/seccomp policies can block unprivileged user
    namespaces even when the sysctl allows them.
    """
    global _chroot_available_cache
    if _chroot_available_cache is not None:
        return _chroot_available_cache
    with _chroot_available_lock:
        if _chroot_available_cache is None:
            _chroot_available_cache = _probe_chroot_available()
        return _chroot_available_cache


def _probe_chroot_available() -> bool:
    if shutil.which("unshare") is None or shutil.which("chroot") is None:
        return False
    try:
        sysctl_path = Path("/proc/sys/kernel/unprivileged_userns_clone")
        if sysctl_path.exists() and sysctl_path.read_text().strip() == "0":
            return False
    except OSError:
        pass
    try:
        proc = subprocess.run(
            ["unshare", "--user", "--map-root-user", "--mount", "true"],
            capture_output=True,
            timeout=AgSandboxBackendFields().inspect_timeout_s,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _auto_detect_runtime() -> str:
    """Auto-detect which backend to use: podman, then docker, then chroot."""
    try:
        return get_container_runtime()
    except RuntimeError:
        if chroot_available():
            return "chroot"
        raise RuntimeError(
            "No usable sandbox backend found: neither docker nor podman is "
            "installed/reachable, and unprivileged user namespaces (required "
            "for the chroot backend) are not usable on this host."
        ) from None


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


# ---------------------------------------------------------------------------
# Backend base class + factory
# ---------------------------------------------------------------------------


class agsandbox_backend(AgSandboxBackendFields):
    """One backend instance per agSandbox — owns the actual isolation
    mechanism (a running container, a chroot jail, ...) for that sandbox's
    whole lifetime. Use `agsandbox_backend.for_config()` to get the right
    subclass; don't instantiate a subclass directly.

    Concrete backends implement every method `agSandbox` (the facade in
    agsandbox.py) delegates to: exec/_container_exec, file I/O,
    commit/stop/restore/destroy, update_limits, remove_files, PID tracking,
    fork, and the static image-level helpers.
    """

    # Identifies the image/snapshot format a concrete backend's
    # tag_image/export_image/import_image/delete_image use -- checkpoints
    # produced by one backend kind are meaningless to another (a chroot
    # snapshot directory isn't a docker image tag, and vice versa), so
    # agent.py's save()/load() records this alongside a checkpoint and uses
    # backend_for_image_kind() to route to the matching backend class rather
    # than assuming the container backend unconditionally.
    IMAGE_KIND: "str" = ""

    @staticmethod
    def for_config(
        agconfig: "agConfig | None",
        *,
        agname: str,
        name: str,
        checkpoint_image: "str | None",
        base_image: str,
        mounts: "dict[str, tuple[str, str, str]]",
    ) -> "agsandbox_backend":
        requested = (
            agconfig.get("agsandbox_backend", "backend", "auto") if agconfig else None
        ) or "auto"

        if requested == "auto":
            runtime = _auto_detect_runtime()
        elif requested in ("docker", "podman"):
            if not (shutil.which(requested) and _runtime_works(requested)):
                raise RuntimeError(
                    f"agsandbox_backend.backend={requested!r} was requested but {requested} "
                    f"is not installed or its daemon is not reachable."
                )
            runtime = requested
        elif requested == "chroot":
            if not chroot_available():
                raise RuntimeError(
                    "agsandbox_backend.backend='chroot' was requested but unprivileged user "
                    "namespaces are not usable on this host (checked "
                    "/proc/sys/kernel/unprivileged_userns_clone and a live "
                    "`unshare --user --map-root-user` smoke test)."
                )
            runtime = "chroot"
        else:
            raise ValueError(
                f"Unknown agsandbox_backend.backend {requested!r} "
                f"(expected 'auto', 'docker', 'podman', or 'chroot')"
            )

        if runtime == "chroot":
            return _ChrootBackend(
                agname,
                name=name,
                checkpoint_image=checkpoint_image,
                mounts=mounts,
                agconfig=agconfig,
            )

        return _ContainerBackend(
            agname,
            name=name,
            runtime=runtime,
            checkpoint_image=checkpoint_image,
            base_image=base_image,
            mounts=mounts,
            agconfig=agconfig,
        )

    # ------------------------------------------------------------------
    # Static image-level helpers, forwarded to the container backend — the
    # only backend with an image concept so far. agSandbox (the facade)
    # calls these without an instance at hand (see agent.py's save()/load()),
    # so they're exposed here rather than requiring a live backend instance.
    # ------------------------------------------------------------------

    @staticmethod
    def tag_image(source: str, dest: str) -> None:
        _ContainerBackend.tag_image(source, dest)

    @staticmethod
    def delete_image(tag: str, *, force: bool = False) -> None:
        _ContainerBackend.delete_image(tag, force=force)

    @staticmethod
    def export_image(tag: str, timeout: int) -> bytes:
        return _ContainerBackend.export_image(tag, timeout)

    @staticmethod
    def import_image(image_bytes: bytes, timeout: int) -> None:
        _ContainerBackend.import_image(image_bytes, timeout)

    # ------------------------------------------------------------------
    # Shared implementation -- exec (GPU env injection + background-PID
    # tracking), file I/O, and PID bookkeeping are all implemented purely in
    # terms of the abstract _container_exec() primitive below, so every
    # concrete backend gets them for free rather than reimplementing the
    # same base64/proc-diffing logic per backend.
    # ------------------------------------------------------------------

    def _snapshot_pids(self) -> set[int]:
        """Return the set of all live PIDs currently in the container, excluding
        the snapshot shell itself so that monitoring shells are not mistaken
        for user-spawned processes."""
        out, _ = self._container_exec(
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

    def exec(
        self,
        cmd: str,
        workdir: str = "/workspace",
        timeout: int = AgSandboxBackendFields.DEFAULT_EXEC_TIMEOUT_S,
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
        import os

        gpu_id = str(self._gpu_id) if self._gpu_id is not None else "NoDevFiles"
        hf_token = os.environ.get("HF_TOKEN", "")
        hf_export = f"export HF_TOKEN={hf_token}\n" if hf_token else ""
        env_export = f"export CUDA_VISIBLE_DEVICES={gpu_id}\nexport HIP_VISIBLE_DEVICES={gpu_id}\n{hf_export}"

        wrapped = (
            f"exec 2>&1\n"  # merge stderr into stdout so the BGPIDS marker is never split
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
                exc.encoding,
                exc.object,
                exc.start,
                exc.end,
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
            f"mkdir -p $(dirname {quoted}) && printf '%s' {shlex.quote(b64)} | base64 -d > {quoted}"
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
            '  [ -f "$__d/status" ] || continue\n'
            "  __p=${__d##*/}\n"
            '  [ "$__p" = "$__SELF" ] && continue\n'
            "  __ppid=$(awk '/^PPid:/{print $2}' $__d/status 2>/dev/null)\n"
            "  __st=$(awk '/^State:/{print $2}' $__d/status 2>/dev/null)\n"
            "  __nm=$(awk '/^Name:/{print $2}' $__d/status 2>/dev/null)\n"
            '  echo "$__p $__ppid $__st $__nm"\n'
            "done"
        )
        output, _ = self._container_exec(script, timeout=self.inspect_timeout_s, shell="sh")

        proc_info: dict[int, tuple[int, str, str]] = {}  # pid → (ppid, state, name)
        for line in output.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                ppid = int(parts[1])
                state = parts[2] if len(parts) > 2 else "?"
                name = parts[3] if len(parts) > 3 else ""
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
            pid
            for pid, (_, _, name) in proc_info.items()
            if name == "nvidia_entrypoi" and pid not in self._baseline_pids
        }
        changed = True
        while changed:
            changed = False
            for pid, (ppid, _, _) in proc_info.items():
                if (
                    pid not in system_pids
                    and pid not in self._baseline_pids
                    and ppid in system_pids
                ):
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
            if (
                pid in self._baseline_pids
                or pid in system_pids
                or pid in self._daemon_pids
                or state == "Z"
            ):
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
                print(f"[agsandbox_backend] WARNING: update_limits failed for {self._name}: {_e}")

    def remove_files(self, paths: list[str]) -> None:
        """Delete sandbox files previously written by offload/agfile helpers."""
        import shlex as _shlex

        for path in paths:
            try:
                self._container_exec(f"rm -f {_shlex.quote(path)}", shell="sh")
            except Exception as _e:
                print(f"[agsandbox_backend] WARNING: failed to remove offloaded file {path}: {_e}")


class _ContainerBackend(agsandbox_backend):
    """Manages a single container for one agent via docker or podman.

    All filesystem operations route through ``docker exec`` / ``podman exec``.
    Background PIDs spawned by bash calls are tracked in ``_watched_pids`` and
    the outer monitoring loop in ``agent._task()`` waits for them before
    resolving the skill future.
    """

    IMAGE_KIND = "container"

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
        *,
        name: str,
        runtime: str,
        checkpoint_image: "str | None",
        base_image: str,
        mounts: "dict[str, tuple[str, str, str]]",
        agconfig: "agConfig | None",
    ) -> None:
        self._agname = agname
        self._runtime = runtime
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
            # that process owns the keyring slot — release ours.
            if self._runtime == "docker":
                _container_semaphore.release()
            self._started = True
            self._baseline_pids = self._snapshot_pids()
            return
        except Exception:
            if self._runtime == "docker":
                _container_semaphore.release()
            raise
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
                    check=False,
                    timeout=self.stop_inspect_timeout_s,
                )
                if result and result.returncode == 0:
                    old_image_id = result.stdout.decode("utf-8", errors="replace").strip() or None
            except Exception:
                pass
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


# ---------------------------------------------------------------------------
# Chroot backend -- no daemon, no image format: isolation is a per-agent
# directory chrooted into via an unprivileged user+mount namespace
# (`unshare --user --map-root-user --mount`), needing no root/sudo/setcap.
#
# What this gets you: each agent sees only its own writable workspace plus a
# read-only view of the host's own interpreters/system libraries (bin, lib,
# usr, ...) bind-mounted in -- it cannot read or write anything on the host
# outside that. "Committing"/"restoring" is a plain directory copy
# (`cp -a --reflink=auto`, so it's a true point-in-time copy, not a hardlink
# clone that a later in-place write would silently corrupt) instead of an
# image layer.
#
# What this does NOT get you, matching the scope this backend was built for
# (filesystem containment + independent per-agent installs, not defense
# against adversarial code): no network namespace (the jailed process shares
# the host's network stack), no PID namespace (a fresh procfs is mounted
# inside the jail so background-process tracking keeps working, but that
# means the jailed process can see -- though not touch, since real
# permission checks still key off the unprivileged host uid the mapped
# "root" resolves to -- every host process), no cgroup CPU/memory limits
# (update_limits() is a no-op), and no GPU device scoping (a leased GPU's
# CUDA_VISIBLE_DEVICES env var is still set, same as the container backend,
# but nothing stops a process from seeing every /dev entry the host user can).
# ---------------------------------------------------------------------------

_CHROOT_STATE_ROOT = Path(tempfile.gettempdir()) / "agency-chroot-sandboxes"
_CHROOT_JAILS_DIR = _CHROOT_STATE_ROOT / "jails"
_CHROOT_SNAPSHOTS_DIR = _CHROOT_STATE_ROOT / "snapshots"

# Host directories bind-mounted read-only into every jail so common
# interpreters/tools (python, bash, coreutils, shared libs) are usable
# without needing a separate root filesystem image. /dev is bind-mounted
# read-write (unscoped -- see the module docstring above) since most
# programs assume /dev/null, /dev/urandom etc. exist and are writable.
_CHROOT_RO_BASE_DIRS = ("bin", "sbin", "lib", "lib32", "lib64", "usr", "etc")
_CHROOT_RW_BASE_DIRS = ("dev",)


def _sanitize_tag(tag: str) -> str:
    """Turn a checkpoint tag (e.g. "agency/lifecycle-name") into a single
    filesystem-safe path segment for use under _CHROOT_SNAPSHOTS_DIR."""
    return tag.replace("/", "__").replace(":", "__")


class _ChrootBackend(agsandbox_backend):
    """Manages a single chroot jail for one agent.

    All filesystem operations run inside a fresh
    ``unshare --user --map-root-user --mount`` + ``chroot`` invocation per
    call -- there is no long-lived daemon process to exec into (unlike
    docker/podman), so "starting" a sandbox is just making sure its jail
    directory exists on disk; the directory itself is the persistent state.
    """

    IMAGE_KIND = "chroot"

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

    def _ensure_started(self) -> None:
        """Create the jail's workspace directory on first use, restoring it
        from ``_checkpoint_image`` if one was given at construction time.

        Ground truth is the workspace directory's existence on disk, not
        ``self._started``. Tool calls with ``run_in_subprocess=True`` (the
        default) get a fresh cloudpickled copy of this backend per call, so
        a worker process's own ``_started`` is unreliable — exactly the
        problem ``_ContainerBackend._ensure_started()`` documents and solves
        by querying the docker daemon (``_container_running()``) instead of
        trusting its own ``_started``. There's no daemon here, so the
        workspace directory itself is the cross-process source of truth:
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

    def _build_jail_script(self, sh_cmd: str, *, workdir: str, shell: str) -> str:
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
        for d in _CHROOT_RW_BASE_DIRS:
            host_path = f"/{d}"
            if not os.path.isdir(host_path):
                continue
            jail_path = f"{root}/{d}"
            lines.append(f"mkdir -p {shlex.quote(jail_path)}")
            lines.append(f"mount --bind {shlex.quote(host_path)} {shlex.quote(jail_path)}")
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
        lines.append(f"mkdir -p {shlex.quote(root + '/proc')}")
        lines.append(f"mount -t proc proc {shlex.quote(root + '/proc')} 2>/dev/null || true")
        lines.append(f"mkdir -p {shlex.quote(root + '/tmp')}")
        # Setup (mkdir/mount) output must never reach the caller -- it isn't
        # part of the command's own stdout/stderr, and read_file()/write_file()
        # (inherited from agsandbox_backend) parse the captured output as raw
        # base64, which a stray "mount: permission denied" line would corrupt.
        setup = "{ " + "; ".join(lines) + "; } >/dev/null 2>&1"
        inner_cmd = f"cd {shlex.quote(workdir)} 2>/dev/null; {sh_cmd}"
        exec_line = f"exec chroot {shlex.quote(root)} {shell} -c {shlex.quote(inner_cmd)}"
        return f"{setup}\n{exec_line}"

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
        args = ["unshare", "--user", "--map-root-user", "--mount", "--", "bash", "-c", script]
        try:
            proc = subprocess.run(args, input=stdin, capture_output=True, timeout=timeout)
            output = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
            return output, proc.returncode
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s", -1
        except Exception as e:
            return str(e), -1

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
        if self._gpu_virtual and self._gpu_id is not None:
            self._gpu_release_fn(self._gpu_id)
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
        # _ContainerBackend.destroy()'s dangling-image cleanup).
        prefix = _sanitize_tag(f"agency/pretool-{self._name}-")
        if _CHROOT_SNAPSHOTS_DIR.is_dir():
            for entry in _CHROOT_SNAPSHOTS_DIR.iterdir():
                if entry.name.startswith(prefix):
                    shutil.rmtree(entry, ignore_errors=True)

    # ------------------------------------------------------------------
    # Static helpers — snapshot-directory-level operations, the chroot
    # equivalent of _ContainerBackend's image-tag helpers.
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


_BACKEND_KINDS: "dict[str, type]" = {
    _ContainerBackend.IMAGE_KIND: _ContainerBackend,
    _ChrootBackend.IMAGE_KIND: _ChrootBackend,
}


def backend_for_image_kind(kind: str) -> type:
    """Return the backend class whose tag_image/export_image/import_image/
    delete_image understand a checkpoint of the given IMAGE_KIND.

    Used by agent.py's load() to route a saved checkpoint's image bytes to
    the same kind of backend that produced them in save() -- a chroot
    snapshot directory and a docker/podman image tag are different formats
    entirely, so this can't be assumed to always be the container backend.
    """
    try:
        return _BACKEND_KINDS[kind]
    except KeyError:
        raise ValueError(
            f"Unknown sandbox checkpoint image kind {kind!r} "
            f"(expected one of {sorted(_BACKEND_KINDS)})"
        ) from None
