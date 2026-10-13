"""A compose-free Podman environment for Harbor, used by this benchmark only.

Harbor's built-in ``-e podman`` backend needs a Compose frontend. This class
gives Harbor the same operations (start, exec, copy, stop) with plain
``podman`` commands, in two modes:

* managed: Harbor starts a container from the task's digest-pinned prebuilt
  image (the separate verifier environment during grading). Images are never
  built or pulled here; preparation does that ahead of time.
* attach: Harbor operates on a container that already exists and that Agency
  owns (artifact collection, oracle validation). Selected by setting
  ``AGENCY_TB_ATTACH_CONTAINER``; ``stop`` then leaves the container alone.

Loaded by Harbor through ``-e agency_podman_env:AgencyPodmanEnvironment``.
This directory holds only Harbor-side modules so nothing here can shadow a
package Harbor imports.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import NetworkMode

ENGINE = os.environ.get("AGENCY_TB_ENGINE", "podman")


class AgencyPodmanEnvironment(BaseEnvironment):
    def __init__(self, *args, **kwargs):
        self._attach = os.environ.get("AGENCY_TB_ATTACH_CONTAINER") or None
        self._container: "str | None" = self._attach
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return "agency-podman"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    def _validate_definition(self):
        if self._attach is None and not self.task_env_config.docker_image:
            raise FileNotFoundError(
                "AgencyPodmanEnvironment only runs digest-pinned prebuilt images; "
                f"{self.environment_name} declares none"
            )

    # ---------------------------------------------------------------- helpers

    async def _run(self, argv, timeout=None) -> ExecResult:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ExecResult(stdout="", stderr=f"timeout after {timeout}s", return_code=124)
        return ExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode,
        )

    async def _checked(self, argv, timeout=600) -> ExecResult:
        result = await self._run(argv, timeout=timeout)
        if result.return_code != 0:
            raise RuntimeError(f"{' '.join(argv[:3])} failed: {(result.stderr or '').strip()}")
        return result

    def _require_container(self) -> str:
        if self._container is None:
            raise RuntimeError("Environment has not been started")
        return self._container

    # -------------------------------------------------------------- lifecycle

    async def start(self, force_build: bool) -> None:
        if self._attach is not None:
            state = await self._run(
                [ENGINE, "inspect", "-f", "{{.State.Running}}", self._attach], timeout=60
            )
            if state.return_code != 0 or (state.stdout or "").strip() != "true":
                raise RuntimeError(f"Container {self._attach} is not running: {state.stderr}")
            return
        image = self.task_env_config.docker_image
        name = "agency-tb-" + re.sub(r"[^a-zA-Z0-9_.-]", "-", self.session_id).lower()[:80]
        exists = await self._run([ENGINE, "container", "exists", name], timeout=60)
        if exists.return_code == 0:
            raise RuntimeError(f"Container {name} already exists; refusing to reuse it")
        present = await self._run([ENGINE, "image", "exists", image], timeout=60)
        if present.return_code != 0:
            raise RuntimeError(
                f"Image {image} is not present locally; run the benchmark's prepare step"
            )
        argv = [ENGINE, "run", "-d", "--pull=never", "--init", "--name", name]
        label = os.environ.get("AGENCY_TB_LABEL")
        if label:
            argv += ["--label", label]
        if self.network_policy.network_mode == NetworkMode.NO_NETWORK:
            argv += ["--network=none"]
        elif self.network_policy.network_mode != NetworkMode.PUBLIC:
            raise ValueError("Allowlist network policies are not supported by this environment")
        memory_mb = self._effective_memory_mb
        if memory_mb:
            argv += [f"--memory={int(memory_mb)}m"]
        cpus = self._effective_cpus
        if cpus:
            argv += [f"--cpus={cpus}"]
        cpuset = os.environ.get("AGENCY_TB_CPUSET")
        if cpuset:
            argv += [f"--cpuset-cpus={cpuset}"]
        for key, value in self._startup_env().items():
            argv += ["--env", f"{key}={value}"]
        argv += [image, "sh", "-c", "sleep infinity"]
        await self._checked(argv, timeout=300)
        self._container = name

    async def stop(self, delete: bool):
        if self._attach is not None or self._container is None:
            return
        await self._run([ENGINE, "rm", "--force", "--time", "0", self._container], timeout=120)
        self._container = None

    # ------------------------------------------------------------- operations

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        argv = [ENGINE, "exec"]
        user = self._resolve_user(user)
        if user is not None:
            argv += ["--user", str(user)]
        workdir = cwd or self.task_env_config.workdir
        if workdir:
            argv += ["--workdir", workdir]
        for key, value in (self._merge_env(env) or {}).items():
            argv += ["--env", f"{key}={value}"]
        argv += [self._require_container(), "bash", "-c", command]
        return await self._run(argv, timeout=timeout_sec)

    async def upload_file(self, source_path, target_path):
        parent = str(Path(str(target_path)).parent)
        await self.exec(f"mkdir -p {shlex.quote(parent)}", user="root")
        await self._checked(
            [ENGINE, "cp", str(source_path), f"{self._require_container()}:{target_path}"]
        )

    async def upload_dir(self, source_dir, target_dir):
        await self.exec(f"mkdir -p {shlex.quote(str(target_dir))}", user="root")
        await self._checked(
            [ENGINE, "cp", f"{source_dir}/.", f"{self._require_container()}:{target_dir}"]
        )

    async def download_file(self, source_path, target_path):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        await self._checked(
            [ENGINE, "cp", f"{self._require_container()}:{source_path}", str(target_path)]
        )

    async def download_dir(self, source_dir, target_dir):
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        await self._checked(
            [ENGINE, "cp", f"{self._require_container()}:{source_dir}/.", str(target_dir)]
        )
