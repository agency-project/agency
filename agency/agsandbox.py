from __future__ import annotations

import atexit
import shlex
import shutil
import subprocess
import time
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from .agresources import agResourcePool

_BGPIDS_MARKER = "__BGPIDS__:"
_RUNTIME: str | None = None

# Global registry of live sandboxes for atexit cleanup.
_live_sandboxes: weakref.WeakSet["agSandbox"] = weakref.WeakSet()


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
    """Return ``--gpus all`` when the host has NVIDIA GPUs, otherwise ``[]``.

    Without ``--gpus all`` the NVIDIA device files are never mounted and
    CUDA is inaccessible regardless of CUDA_VISIBLE_DEVICES.  We only add
    the flag when GPUs are actually present so CPU-only hosts keep working.
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
    return []


class agSandbox:
    """Manages a single container for one agent via docker or podman.

    All filesystem operations route through ``docker exec`` / ``podman exec``.
    Background PIDs spawned by bash calls are tracked in ``_watched_pids`` and
    the outer monitoring loop in ``agent._task()`` waits for them before
    resolving the skill future.
    """

    BASE_IMAGE: ClassVar[str] = "agency-sandbox:latest"

    def __init__(
        self,
        agname: str,
        parent_agname: str | None = None,
        output_dir: Path | None = None,
        restore_image: str | None = None,
    ) -> None:
        self._agname = agname
        self._runtime = get_container_runtime()
        self._snapshot_name: str | None = None
        self._gpu_id: int | None = None
        self._watched_pids: dict[int, float] = {}
        self._baseline_pids: set[int] = set()   # populated after container starts
        self._daemon_pids:   set[int] = set()   # explicitly released; never waited on

        # Pass --gpus all if GPUs are available so device files are present.
        # CUDA_VISIBLE_DEVICES is set to "" in every exec call when no GPU is
        # held, so idle containers cannot access any GPU even though the
        # device files exist.
        gpu_flags = _gpu_flags()

        # Shared output volume: all agents read and write the same directory.
        vol_flags: list[str] = []
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            vol_flags = ["-v", f"{output_dir.resolve()}:/agent_output:rw"]

        name = self._container_name()
        if restore_image is not None:
            # Start container from a previously saved checkpoint image
            self._run(
                [self._runtime, "run", "-d", "--name", name] + gpu_flags + vol_flags +
                [restore_image, "tail", "-f", "/dev/null"],
                check=True,
            )
            # Remove the image tag now — container holds a reference by digest
            self._run([self._runtime, "rmi", restore_image], check=False)
        elif parent_agname is not None:
            snap = f"snapshot-{agname}"
            self._run([self._runtime, "commit", f"sandbox-{parent_agname}", snap], check=True)
            self._snapshot_name = snap
            self._run(
                [self._runtime, "run", "-d", "--name", name] + gpu_flags + vol_flags +
                [snap, "tail", "-f", "/dev/null"],
                check=True,
            )
        else:
            self._run(
                [self._runtime, "run", "-d", "--name", name] + gpu_flags + vol_flags +
                [self.BASE_IMAGE, "tail", "-f", "/dev/null"],
                check=True,
            )
            self._run(
                [self._runtime, "exec", name, "mkdir", "-p", "/workspace"],
                check=False,
            )

        _live_sandboxes.add(self)

        # Capture the process baseline after the container is fully ready.
        # Any PID not in this set was spawned by user commands and must be
        # monitored by the outer loop until it exits.
        self._baseline_pids = self._snapshot_pids()

    def _container_name(self) -> str:
        return f"sandbox-{self._agname}"

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
        timeout: int | None = None,
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
        # Always export CUDA_VISIBLE_DEVICES so the container cannot access
        # GPUs that were not explicitly acquired via gpu_acquire, even if
        # --gpus all was passed at container startup.
        cuda_id = str(self._gpu_id) if self._gpu_id is not None else ""
        env_export = f"export CUDA_VISIBLE_DEVICES={cuda_id}\n"

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

    def update_limits(
        self,
        *,
        cpus: float | None = None,
        memory: str | None = None,
    ) -> None:
        """Live-update container CPU/memory limits."""
        cmd = [self._runtime, "update"]
        if cpus is not None:
            cmd.append(f"--cpus={cpus}")
        if memory is not None:
            cmd.append(f"--memory={memory}")
        cmd.append(self._container_name())
        self._run(cmd, timeout=10)

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
        if pool is not None:
            try:
                self.update_limits(cpus=pool.idle_cpus, memory=pool.idle_memory)
            except Exception:
                pass

    def destroy(self) -> None:
        _live_sandboxes.discard(self)

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

        if self._snapshot_name:
            try:
                self._run(
                    [self._runtime, "rmi", self._snapshot_name],
                    timeout=30,
                )
            except Exception:
                pass
