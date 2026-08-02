from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

from agency.agsandbox import get_container_runtime

from ..base import BenchmarkProvider, BenchmarkResult, BenchmarkTask


class TerminalBenchProvider(BenchmarkProvider):
    """Loads Terminal-Bench tasks in the real Harbor task format:

        <task_dir>/
        ├── instruction.md   # task instructions (plain markdown text)
        ├── task.toml        # [task]/[environment]/[agent]/[verifier]/... config
        ├── environment/     # Dockerfile defining the task's own container
        │                    #   (or a bare `docker_image` reference in task.toml,
        │                    #   if there's no Dockerfile to build)
        └── tests/           # copied into the container and run at grading time

    Environment fidelity is intentionally scoped to the common case: a single
    container per task (no docker-compose multi-service environments) and the
    default `environment_mode = "shared"` verifier (tests run in the same
    container the agent worked in, not an isolated grading container). Other
    task.toml settings (resource limits, network_mode, [solution], [metadata])
    are read but not enforced.
    """

    def __init__(self, tasks_dir: Path, image_prefix: str = "benchmark-tb"):
        self.tasks_dir = Path(tasks_dir)
        self.image_prefix = image_prefix

    @property
    def name(self) -> str:
        return "terminal-bench"

    def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        task_dirs = sorted(p.parent for p in self.tasks_dir.rglob("task.toml"))
        if limit is not None:
            task_dirs = task_dirs[:limit]
        return [self._task_from_dir(d) for d in task_dirs]

    def evaluate(self, task: BenchmarkTask, result: BenchmarkResult) -> dict:
        if task.metadata.get("environment_mode") != "shared":
            return {
                "passed": None,
                "error": "only environment_mode='shared' verifiers are supported",
            }

        checkpoint_image = result.metadata.get("checkpoint_image")
        if not checkpoint_image:
            return {
                "passed": None,
                "reward": None,
                "error": "agent run produced no checkpoint image to grade",
            }

        task_dir = Path(task.metadata["task_dir"])
        container = f"tb-eval-{task.task_id}".lower()
        runtime = get_container_runtime()
        try:
            subprocess.run(
                [
                    runtime,
                    "run",
                    "-d",
                    "--name",
                    container,
                    checkpoint_image,
                    "tail",
                    "-f",
                    "/dev/null",
                ],
                check=True,
            )
            subprocess.run(
                [runtime, "exec", container, "mkdir", "-p", "/tests", "/logs/verifier"], check=True
            )
            subprocess.run(
                [runtime, "cp", f"{task_dir / 'tests'}/.", f"{container}:/tests"], check=True
            )
            test_run = subprocess.run(
                [runtime, "exec", container, "bash", "/tests/test.sh"],
                capture_output=True,
                text=True,
            )
            return {
                "passed": test_run.returncode == 0,
                "reward": self._read_reward(runtime, container),
                "test_exit_code": test_run.returncode,
                "test_output": test_run.stdout + test_run.stderr,
            }
        finally:
            subprocess.run([runtime, "rm", "-f", container], check=False)
            subprocess.run([runtime, "rmi", "-f", checkpoint_image], check=False)

    # ------------------------------------------------------------------
    # task.toml / instruction.md parsing, image build, and workdir discovery
    # ------------------------------------------------------------------

    def _task_from_dir(self, task_dir: Path) -> BenchmarkTask:
        task_id = "__".join(task_dir.relative_to(self.tasks_dir).parts)
        config = tomllib.loads((task_dir / "task.toml").read_text())
        instructions = (task_dir / "instruction.md").read_text()

        image = self._ensure_image(task_dir, task_id, config.get("environment", {}))
        return BenchmarkTask(
            task_id=task_id,
            benchmark=self.name,
            instructions=instructions,
            workspace=None,
            metadata={
                "task_dir": str(task_dir),
                "image": image,
                "workdir": self._image_workdir(image),
                "environment_mode": config.get("verifier", {}).get("environment_mode", "shared"),
            },
        )

    def _ensure_image(self, task_dir: Path, task_id: str, env_config: dict) -> str:
        runtime = get_container_runtime()
        environment_dir = task_dir / "environment"
        if (environment_dir / "Dockerfile").exists():
            tag = f"{self.image_prefix}-{task_id}".lower()
            subprocess.run([runtime, "build", "-t", tag, str(environment_dir)], check=True)
            return tag

        docker_image = env_config.get("docker_image")
        if not docker_image:
            raise ValueError(
                f"task {task_id!r} has neither environment/Dockerfile nor "
                f"[environment].docker_image in task.toml"
            )
        subprocess.run([runtime, "pull", docker_image], check=True)
        return docker_image

    @staticmethod
    def _image_workdir(image: str) -> str:
        inspect = subprocess.run(
            [get_container_runtime(), "inspect", "--format", "{{.Config.WorkingDir}}", image],
            check=True,
            capture_output=True,
            text=True,
        )
        return inspect.stdout.strip() or "/"

    @staticmethod
    def _read_reward(runtime: str, container: str) -> "float | dict | None":
        as_json = subprocess.run(
            [runtime, "exec", container, "cat", "/logs/verifier/reward.json"],
            capture_output=True,
            text=True,
        )
        if as_json.returncode == 0 and as_json.stdout.strip():
            return json.loads(as_json.stdout)

        as_txt = subprocess.run(
            [runtime, "exec", container, "cat", "/logs/verifier/reward.txt"],
            capture_output=True,
            text=True,
        )
        if as_txt.returncode == 0 and as_txt.stdout.strip():
            return float(as_txt.stdout.strip())
        return None
