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
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from .agresources import agResourcePool

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
# reducing wall-clock time.  A single semaphore replaces the former trio of
# _startup_semaphore / _commit_semaphore / _shutdown_semaphore.
_docker_semaphore = threading.Semaphore(16)

# ---------------------------------------------------------------------------
# Timeout constants (seconds)
# ---------------------------------------------------------------------------
# Fast metadata queries: docker inspect, docker ps, nvidia-smi, docker update.
_TIMEOUT_INSPECT    = 120
# Quick in-container exec calls: kill <pids>, test -d, and similar.
_TIMEOUT_EXEC_QUICK = 120
# docker run: GPU initialisation via the NVIDIA container runtime serialises
# across concurrent containers and can take 60+ s under load.
_TIMEOUT_DOCKER_RUN = 120
# docker rm -f: fast teardown; should complete in a few seconds.
_TIMEOUT_DOCKER_RM  = 120
# In-container file I/O via docker exec (base64 read/write, mkdir).
_TIMEOUT_FILE_IO    = 120
# docker images list / docker rmi.
_TIMEOUT_IMAGE      = 120
# docker commit: snapshots a full overlay layer; large workspaces need extra time.
_TIMEOUT_COMMIT     = 120
# Maximum time to wait for a keyring slot before abandoning a docker run retry.
_TIMEOUT_KEYRING_WAIT = 120

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
    try:
        return max(4, int(Path("/proc/sys/kernel/keys/maxkeys").read_text().strip()) - 5)
    except OSError:
        return 200 - 5

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
            timeout=_TIMEOUT_INSPECT,
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


def _gpu_flags() -> list[str]:
    """Return GPU passthrough flags for the container runtime.

    NVIDIA: ``--gpus all`` (requires nvidia-container-toolkit).
    AMD:    ``--device /dev/kfd --device /dev/dri`` (ROCm device files).
    CPU-only hosts get no flags so they keep working without GPU drivers.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, timeout=_TIMEOUT_INSPECT,
        )
        if result.returncode == 0 and result.stdout.strip():
            return ["--gpus", "all"]
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid", "--csv"],
            capture_output=True, timeout=_TIMEOUT_INSPECT,
        )
        if result.returncode == 0 and result.stdout.strip():
            return ["--device", "/dev/kfd", "--device", "/dev/dri"]
    except Exception:
        pass
    return []


class agSandbox:
    """Manages a single container for one agent via docker or podman.

    All filesystem operations route through ``docker exec`` / ``podman exec``.
    Background PIDs spawned by bash calls are tracked in ``_watched_pids`` and
    the outer monitoring loop in ``agent._task()`` waits for them before
    resolving the skill future.
    """

    BASE_IMAGE: ClassVar[str] = "agency-sandbox:latest"

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
        output_dir: Path | None = None,
        lifecycle_image: str | None = None,
    ) -> None:
        self._agname   = agname
        self._runtime  = get_container_runtime()
        self._gpu_id:  int | None           = None
        self._gpu_virtual: bool             = False   # LLM has called reserve_gpu
        self._gpu_acquire_fn                = None    # pool.acquire_gpu, set by make_gpu_reserve
        self._gpu_release_fn                = None    # pool.release_gpu, set by make_gpu_reserve
        self._cpu_acquired: float           = 0.0
        self._memory_acquired_mb: int       = 0
        self._watched_pids: dict[int, float] = {}
        self._baseline_pids: set[int]        = set()
        self._daemon_pids:   set[int]        = set()
        self._started  = False
        self._lifecycle_image: str | None    = lifecycle_image

        # Container name is fixed at creation time using the main-process PID
        # prefix so that worker processes (with different PIDs) use the correct name.
        self._name = f"sandbox-{_RUN_ID}-{agname}"

        # Store startup parameters for _ensure_started().
        self._gpu_flags     = _gpu_flags()
        self._vol_flags: list[str] = []
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self._vol_flags = ["-v", f"{output_dir.resolve()}:/agent_output:rw"]

    def _container_running(self) -> bool:
        """Return True if the named container is currently running in Docker/Podman."""
        result = self._run(
            [self._runtime, "inspect", "--format", "{{.State.Running}}", self._name],
            check=False, timeout=_TIMEOUT_INSPECT,
        )
        return result.returncode == 0 and result.stdout.strip() == b"true"

    def _container_status(self) -> str:
        """Return the container state string: 'running', 'exited', 'created', etc., or '' if not found."""
        result = self._run(
            [self._runtime, "inspect", "--format", "{{.State.Status}}", self._name],
            check=False, timeout=_TIMEOUT_INSPECT,
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
        self._run([self._runtime, "rm", "-f", name], check=False)
        if self._runtime == "docker":
            _container_semaphore.acquire()
        try:
            if self._lifecycle_image is not None:
                # Restart from last committed checkpoint (set by stop(commit=True)).
                # /workspace and all state from the previous tool call are preserved.
                image = self._lifecycle_image
                run_cmd = (
                    [self._runtime, "run", "-d", "--init", "--name", name]
                    + self._gpu_flags + self._vol_flags
                    + [image, "tail", "-f", "/dev/null"]
                )
                self._run_with_conflict_retry(run_cmd, name)
                # Keep _lifecycle_image — not a one-shot restore, needed for future restarts.
            else:
                image = self._resolve_image(self.BASE_IMAGE)
                cpu_flags = ["--cpus=1"] if self._cfs_supported() else []
                run_cmd = (
                    [self._runtime, "run", "-d", "--init", "--name", name]
                    + cpu_flags + self._gpu_flags + self._vol_flags
                    + [image, "tail", "-f", "/dev/null"]
                )
                self._run_with_conflict_retry(run_cmd, name)
                self._run(
                    [self._runtime, "exec", name, "mkdir", "-p", "/workspace"],
                    check=False,
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
        """Run a docker run command, retrying up to 3 times on name-conflict errors.

        A "Conflict / already in use" error can arise when a previous docker run
        call failed mid-way (e.g. GPU allocation timeout) and left a container
        object in "Created" state without ever starting.  We force-remove the
        stale entry and retry rather than surfacing an opaque error to the agent.
        """
        _last_stderr = ""
        for attempt in range(8):
            result = self._run(run_cmd, timeout=_TIMEOUT_DOCKER_RUN)
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
                deadline = time.monotonic() + _TIMEOUT_KEYRING_WAIT
                while time.monotonic() < deadline:
                    if keyring_quota().get("free", 0) > 0:
                        break
                    time.sleep(5)
                # docker run can partially succeed before failing with keyring:
                # it creates the container object (reserving the name) but fails
                # before starting processes.  Remove any such "Created" artifact
                # so the next attempt does not see a spurious name conflict.
                self._run([self._runtime, "rm", "-f", name], check=False)
            elif conflict:
                if self._container_running():
                    raise _ContainerAlreadyRunning()
                # Leftover container in a non-running state — remove it.
                # Wait until it's actually gone before retrying docker run.
                self._run([self._runtime, "rm", "-f", name], check=False)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if not self._container_status():
                        break
                    time.sleep(0.5)
                # If keyring is also full (docker created the object then hit
                # the limit), wait for a slot before retrying — otherwise we'll
                # create another "Created" container and loop on conflicts.
                deadline = time.monotonic() + _TIMEOUT_KEYRING_WAIT
                while time.monotonic() < deadline:
                    if keyring_quota().get("free", 0) > 0:
                        break
                    time.sleep(5)
                time.sleep(0.5 * (attempt + 1))
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
            timeout=_TIMEOUT_INSPECT, shell="sh",
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
        timeout: int = 120,
    ) -> subprocess.CompletedProcess[bytes]:
        with _docker_semaphore:
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

    def _container_exec(
        self,
        sh_cmd: str,
        workdir: str = "/workspace",
        timeout: int = 120,
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
        timeout: int = 120,
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
            f"base64 {shlex.quote(path)}", timeout=_TIMEOUT_FILE_IO, shell="sh"
        )
        if rc != 0:
            _, dir_rc = self._container_exec(
                f"test -d {shlex.quote(path)}", timeout=_TIMEOUT_EXEC_QUICK, shell="sh"
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
            f"base64 {shlex.quote(path)}", timeout=_TIMEOUT_FILE_IO, shell="sh"
        )
        if rc != 0:
            _, dir_rc = self._container_exec(
                f"test -d {shlex.quote(path)}", timeout=_TIMEOUT_EXEC_QUICK, shell="sh"
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
        _, rc = self._container_exec(sh_cmd, timeout=_TIMEOUT_FILE_IO, shell="sh")
        if rc != 0:
            raise OSError(f"Failed to write binary file {path} in container")

    def write_file(self, path: str, content: str) -> None:
        quoted = shlex.quote(path)
        sh_cmd = f"mkdir -p $(dirname {quoted}) && cat > {quoted}"
        _, rc = self._container_exec(
            sh_cmd, stdin=content.encode("utf-8"), timeout=_TIMEOUT_FILE_IO, shell="sh"
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
        self._run(cmd, timeout=_TIMEOUT_INSPECT)

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
            timeout=_TIMEOUT_COMMIT,
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
            tag = f"agency/lifecycle-{self._name}"
            for _attempt in range(3):
                try:
                    self._run(
                        [self._runtime, "commit", self._container_name(), tag],
                        check=True, timeout=_TIMEOUT_COMMIT,
                    )
                    self._lifecycle_image = tag
                    break
                except Exception as _e:
                    if _attempt == 2:
                        print(
                            f"[agsandbox] WARNING: docker commit {self._container_name()} → {tag} "
                            f"failed after 3 attempts: {_e}",
                            file=__import__("sys").stderr, flush=True,
                        )
                    else:
                        time.sleep(1)
        name = self._container_name()
        for _attempt in range(3):
            try:
                self._run([self._runtime, "rm", "-f", name], check=True, timeout=_TIMEOUT_DOCKER_RM)
                break
            except Exception:
                if _attempt == 2:
                    import traceback as _tb
                    print(
                        f"[agsandbox] WARNING: docker rm -f {name} failed after 3 attempts:\n"
                        f"{_tb.format_exc()}",
                        file=__import__("sys").stderr, flush=True,
                    )
                else:
                    time.sleep(1)
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
                        f"kill {pids} 2>/dev/null; true", timeout=_TIMEOUT_EXEC_QUICK, shell="sh"
                    )
                except Exception as _e:
                    print(f"[agsandbox] WARNING: failed to kill PIDs {pids} in {self._name} during restore: {_e}")
            try:
                self._run(
                    [self._runtime, "rm", "-f", self._container_name()],
                    timeout=_TIMEOUT_DOCKER_RM,
                )
            except Exception as _e:
                print(f"[agsandbox] WARNING: docker rm -f {self._container_name()} failed during restore: {_e}")
            self._started = False
            self._watched_pids = {}
            self._baseline_pids = set()
        self._lifecycle_image = tag
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
        output, _ = self._container_exec(script, timeout=_TIMEOUT_INSPECT, shell="sh")

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

    def destroy(self) -> None:
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
                    f"kill {pids} 2>/dev/null; true", timeout=_TIMEOUT_EXEC_QUICK, shell="sh"
                )
            except Exception as _e:
                print(f"[agsandbox] WARNING: failed to kill PIDs {pids} in {container_name}: {_e}")

        # Check before rm so we know whether a keyring slot must be released.
        had_container = self._runtime == "docker" and bool(
            self._started or self._container_running()
        )

        try:
            self._run(
                [self._runtime, "rm", "-f", container_name],
                timeout=_TIMEOUT_DOCKER_RM,
            )
        except Exception as _e:
            print(f"[agsandbox] WARNING: docker rm -f {container_name} failed during destroy: {_e}")
        if had_container:
            _container_semaphore.release()

        # Remove pre-tool checkpoint images created during this sandbox's lifetime.
        try:
            result = self._run(
                [self._runtime, "images", "--format", "{{.Repository}}:{{.Tag}}"],
                timeout=_TIMEOUT_IMAGE,
            )
            prefix = f"agency/pretool-{self._name}-"
            lifecycle = f"agency/lifecycle-{self._name}"
            for line in result.stdout.decode("utf-8", errors="replace").splitlines():
                tag = line.strip()
                # docker images --format {{.Repository}}:{{.Tag}} includes the ":latest"
                # suffix when no explicit tag was given (e.g. "agency/lifecycle-x:latest").
                tag_repo = tag.rsplit(":", 1)[0] if ":" in tag else tag
                if tag.startswith(prefix) or tag_repo == lifecycle:
                    self._run([self._runtime, "rmi", "-f", tag], timeout=_TIMEOUT_IMAGE, check=False)
        except Exception as _e:
            print(f"[agsandbox] WARNING: image cleanup failed for {container_name}: {_e}")

