from __future__ import annotations

import atexit
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

# Per-process prefix so concurrent script invocations never share container names.
_PID_PREFIX = f"p{os.getpid()}"

# Global registry of live sandboxes for atexit cleanup.
_live_sandboxes: weakref.WeakSet["agSandbox"] = weakref.WeakSet()

# Limit the number of containers starting simultaneously.  Each agSandbox.__init__
# acquires one slot for the duration of its startup sequence (docker run + first exec).
# Without this, a burst of hundreds of parallel agent tasks overwhelms the Docker
# daemon.  With --gpus all, the NVIDIA runtime serializes GPU device initialization,
# so more than ~8 concurrent docker-run calls increase contention and leave stale
# "Created" containers without reducing wall-clock time.
_startup_semaphore = threading.Semaphore(8)


def _cleanup_all_sandboxes() -> None:
    """Destroy all live sandbox containers on process exit."""
    for sandbox in list(_live_sandboxes):
        try:
            sandbox.destroy()
        except Exception:
            pass


atexit.register(_cleanup_all_sandboxes)


def _runtime_works(runtime: str) -> bool:
    try:
        proc = subprocess.run(
            [runtime, "info"],
            capture_output=True,
            timeout=10,
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
            capture_output=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return ["--gpus", "all"]
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid", "--csv"],
            capture_output=True, timeout=10,
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
        restore_image: str | None = None,
    ) -> None:
        self._agname   = agname
        self._runtime  = get_container_runtime()
        self._gpu_id:  int | None           = None
        self._cpu_acquired: float           = 0.0
        self._memory_acquired_mb: int       = 0
        self._watched_pids: dict[int, float] = {}
        self._baseline_pids: set[int]        = set()
        self._daemon_pids:   set[int]        = set()
        self._started  = False

        # Container name is fixed at creation time using the main-process PID
        # prefix so that worker processes (with different PIDs) use the correct name.
        self._name = f"sandbox-{_PID_PREFIX}-{agname}"

        # Store startup parameters for _ensure_started().
        self._restore_image = restore_image
        self._gpu_flags     = _gpu_flags()
        self._vol_flags: list[str] = []
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self._vol_flags = ["-v", f"{output_dir.resolve()}:/agent_output:rw"]

    def _container_running(self) -> bool:
        """Return True if the named container is currently running in Docker/Podman."""
        result = self._run(
            [self._runtime, "inspect", "--format", "{{.State.Running}}", self._name],
            check=False, timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip() == b"true"

    def _ensure_started(self) -> None:
        """Start the Docker container on first use.

        Called lazily by _container_exec() so containers are only created when
        an agent actually needs sandboxed execution (bash, file I/O, etc.).
        Tasks that complete using only host-side tools (webfetch, todowrite,
        find_papers, …) never start a container at all.

        When tools run in worker processes the container is started there, not
        in the main process.  On the next tool call (a fresh worker or the same
        one) ``_started`` is False but the container may still be running — we
        check with ``docker inspect`` and reuse it rather than destroying it.
        """
        if self._started:
            return
        name = self._name
        with _startup_semaphore:
            if self._started:   # re-check after acquiring the semaphore
                return
            # Reuse a container that a previous worker process already started,
            # but only when we are NOT restoring a specific checkpoint image.
            if self._restore_image is None and self._container_running():
                _live_sandboxes.add(self)
                self._started = True
                self._baseline_pids = self._snapshot_pids()
                return
            self._run([self._runtime, "rm", "-f", name], check=False)
            try:
                if self._restore_image is not None:
                    image = self._restore_image
                    run_cmd = (
                        [self._runtime, "run", "-d", "--name", name]
                        + self._gpu_flags + self._vol_flags
                        + [image, "tail", "-f", "/dev/null"]
                    )
                    self._run_with_conflict_retry(run_cmd, name)
                    self._run([self._runtime, "rmi", self._restore_image], check=False)
                else:
                    image = self._resolve_image(self.BASE_IMAGE)
                    cpu_flags = ["--cpus=1"] if self._cfs_supported() else []
                    run_cmd = (
                        [self._runtime, "run", "-d", "--name", name]
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
                # reuse it just as we would in the fast-path above.
                _live_sandboxes.add(self)
                self._started = True
                self._baseline_pids = self._snapshot_pids()
                return
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
        for attempt in range(3):
            result = subprocess.run(run_cmd, capture_output=True, timeout=120)
            if result.returncode == 0:
                return
            stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            conflict = "already in use" in stderr or "Conflict" in stderr
            if conflict:
                # If another process just started the container and it is now
                # running, reuse it instead (handled by caller's _container_running).
                if self._container_running():
                    # Signal to the caller that the container is ready.
                    # We re-use the path: set _started + return from _ensure_started.
                    raise _ContainerAlreadyRunning()
                # Stale Created/Exited container; remove and retry.
                self._run([self._runtime, "rm", "-f", name], check=False)
                time.sleep(0.5 * (attempt + 1))
            else:
                msg = f"{' '.join(run_cmd[:3])} failed (exit {result.returncode})"
                if stderr:
                    msg += f": {stderr}"
                raise RuntimeError(msg)
        # Final attempt after retries exhausted.
        raise RuntimeError(
            f"docker run --name {name} failed after retries (container name conflict)"
        )

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
            timeout=10, shell="sh",
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

        return clean_output, rc

    def read_file(self, path: str) -> str:
        output, rc = self._container_exec(
            f"cat {shlex.quote(path)}", timeout=30, shell="sh"
        )
        if rc != 0:
            raise FileNotFoundError(f"Not found in container: {path}")
        return output

    def write_file(self, path: str, content: str) -> None:
        quoted = shlex.quote(path)
        sh_cmd = f"mkdir -p $(dirname {quoted}) && cat > {quoted}"
        _, rc = self._container_exec(
            sh_cmd, stdin=content.encode("utf-8"), timeout=30, shell="sh"
        )
        if rc != 0:
            raise OSError(f"Failed to write {path} in container")

    def read_file(self, path: str) -> str:
        out, rc = self._container_exec(f"cat {shlex.quote(path)}", shell="sh")
        if rc != 0:
            raise FileNotFoundError(f"Failed to read {path} from container")
        return out

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
        self._run(cmd, timeout=10)

    def commit(self, tag: str) -> bool:
        """Commit the container filesystem to a new image tag.

        Returns True if the commit succeeded, False if no container is running
        (nothing to commit).  Also works when the container was started by a
        worker process and ``_started`` is still False in the main process.
        """
        if not self._started and not self._container_running():
            return False
        self._run(
            [self._runtime, "commit", self._container_name(), tag],
            check=True,
            timeout=120,
        )
        return True

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
                        f"kill {pids} 2>/dev/null; true", timeout=5, shell="sh"
                    )
                except Exception:
                    pass
            try:
                self._run(
                    [self._runtime, "rm", "-f", self._container_name()],
                    timeout=30,
                )
            except Exception:
                pass
            self._started = False
            self._watched_pids = {}
            self._baseline_pids = set()
        self._restore_image = tag
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

        # Read pid, ppid, and state for every entry in /proc, excluding the
        # monitoring shell itself so it is never mistaken for a user process.
        script = (
            "__SELF=$$\n"
            "for __d in /proc/[0-9]*; do\n"
            "  [ -f \"$__d/status\" ] || continue\n"
            "  __p=${__d##*/}\n"
            "  [ \"$__p\" = \"$__SELF\" ] && continue\n"
            "  __ppid=$(awk '/^PPid:/{print $2}' $__d/status 2>/dev/null)\n"
            "  __st=$(awk '/^State:/{print $2}' $__d/status 2>/dev/null)\n"
            "  echo \"$__p $__ppid $__st\"\n"
            "done"
        )
        output, _ = self._container_exec(script, timeout=10, shell="sh")

        proc_info: dict[int, tuple[int, str]] = {}   # pid → (ppid, state)
        for line in output.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                pid  = int(parts[0])
                ppid = int(parts[1])
                state = parts[2] if len(parts) > 2 else "?"
            except ValueError:
                continue
            proc_info[pid] = (ppid, state)

        # Propagate daemon status down the tree: if a process's parent is a
        # daemon, the child inherits that status and is also excluded from
        # monitoring.  Repeat until no new daemons are discovered.
        changed = True
        while changed:
            changed = False
            for pid, (ppid, _) in proc_info.items():
                if pid not in self._daemon_pids and ppid in self._daemon_pids:
                    self._daemon_pids.add(pid)
                    self._watched_pids.pop(pid, None)
                    changed = True

        # A PID is alive if it exists in /proc, is not baseline, not a daemon,
        # and not a zombie.  Any newly discovered non-baseline PID is added to
        # _watched_pids so the outer loop waits for it.
        alive: set[int] = set()
        now = time.monotonic()
        for pid, (_, state) in proc_info.items():
            if pid in self._baseline_pids or pid in self._daemon_pids or state == "Z":
                continue
            alive.add(pid)
            if pid not in self._watched_pids:
                self._watched_pids[pid] = now

        # Prune _watched_pids entries that are no longer alive.
        for pid in set(self._watched_pids):
            if pid not in alive:
                del self._watched_pids[pid]

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
        if self._gpu_id is not None and pool is not None:
            pool.release_gpu(self._gpu_id)
            self._gpu_id = None
        if pool is not None and (self._cpu_acquired or self._memory_acquired_mb):
            pool.notify_cpu_released(self._cpu_acquired, self._memory_acquired_mb)
            self._cpu_acquired = 0.0
            self._memory_acquired_mb = 0
        if pool is not None:
            try:
                self.update_limits(cpus=pool.idle_cpus, memory=pool.idle_memory)
            except Exception:
                pass

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
                    f"kill {pids} 2>/dev/null; true", timeout=5, shell="sh"
                )
            except Exception:
                pass

        try:
            self._run(
                [self._runtime, "rm", "-f", container_name],
                timeout=30,
            )
        except Exception:
            pass

        # Remove pre-tool checkpoint images created during this sandbox's lifetime.
        try:
            result = self._run(
                [self._runtime, "images", "--format", "{{.Repository}}:{{.Tag}}"],
                timeout=15,
            )
            prefix = f"agency/pretool-{self._name}-"
            for line in result.stdout.decode("utf-8", errors="replace").splitlines():
                tag = line.strip()
                if tag.startswith(prefix):
                    self._run([self._runtime, "rmi", "-f", tag], timeout=15, check=False)
        except Exception:
            pass

