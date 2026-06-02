from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from .resources import agResourcePool

_BGPIDS_MARKER = "__BGPIDS__:"
_RUNTIME: str | None = None


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


class agSandbox:
    """Manages a single container for one agent via docker or podman.

    All filesystem operations route through ``docker exec`` / ``podman exec``.
    Background PIDs spawned by bash calls are tracked in ``_watched_pids`` and
    the outer monitoring loop in ``agent._task()`` waits for them before
    resolving the skill future.
    """

    BASE_IMAGE: ClassVar[str] = "python:3.12-slim"

    def __init__(self, uuid: str, parent_uuid: str | None = None) -> None:
        self._uuid = uuid
        self._runtime = get_container_runtime()
        self._snapshot_name: str | None = None
        self._gpu_id: int | None = None
        self._watched_pids: dict[int, float] = {}

        name = self._container_name()
        if parent_uuid is not None:
            snap = f"snapshot-{uuid}"
            self._run([self._runtime, "commit", f"sandbox-{parent_uuid}", snap], check=True)
            self._snapshot_name = snap
            self._run(
                [self._runtime, "run", "-d", "--name", name, snap, "tail", "-f", "/dev/null"],
                check=True,
            )
        else:
            self._run(
                [self._runtime, "run", "-d", "--name", name,
                 self.BASE_IMAGE, "tail", "-f", "/dev/null"],
                check=True,
            )
            self._run(
                [self._runtime, "exec", name, "mkdir", "-p", "/workspace"],
                check=False,
            )

    def _container_name(self) -> str:
        return f"sandbox-{self._uuid}"

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
        env_prefix = ""
        if self._gpu_id is not None:
            env_prefix = f"CUDA_VISIBLE_DEVICES={self._gpu_id} "

        wrapped = (
            f"set -m\n"
            f"{env_prefix}{cmd}\n"
            f"__AGENCY_RC=$?\n"
            f"__AGENCY_BGPIDS=$(jobs -p 2>/dev/null | tr '\\n' ' ')\n"
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

    def get_live_pids(self) -> set[int]:
        if not self._watched_pids:
            return set()

        pid_list = " ".join(str(p) for p in self._watched_pids)
        script = (
            f"for __p in {pid_list}; do\n"
            f"  kill -0 $__p 2>/dev/null && echo \"alive:$__p\"\n"
            f"done"
        )
        output, _ = self._container_exec(script, timeout=10, shell="sh")

        alive: set[int] = set()
        for line in output.splitlines():
            if line.startswith("alive:"):
                try:
                    alive.add(int(line[6:].strip()))
                except ValueError:
                    pass

        for pid in set(self._watched_pids) - alive:
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
