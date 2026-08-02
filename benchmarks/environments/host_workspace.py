from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agency.agsandbox import agSandboxConfig

from ..base import BenchmarkTask, ExecutionEnvironment, PreparedEnvironment


class HostWorkspaceEnvironment(ExecutionEnvironment):
    """Bind-mounts a temp copy of `task.workspace` into the sandbox at
    /workspace — the original is never mutated. Used by SWE-bench: the
    task's workspace is a host git checkout, and the patch is collected by
    diffing that host-side copy after the run, since the container's own
    overlay is discarded rather than committed for extraction.
    """

    def prepare(self, cfg: Any, task: BenchmarkTask) -> PreparedEnvironment:
        if task.workspace is None:
            raise ValueError(f"task {task.task_id!r} has no workspace to run in")

        tmp = tempfile.mkdtemp(prefix=f"benchmark-{task.task_id}-")
        workdir = Path(tmp) / "workspace"
        shutil.copytree(task.workspace, workdir)

        agSandboxConfig(cfg).add_mount("workspace", workdir, "/workspace")

        return PreparedEnvironment(
            prompt_hint=(
                "Your working copy of the repository is mounted at /workspace — "
                "make all your edits there. You do not need to install "
                "dependencies or run the test suite; evaluation happens "
                "separately."
            ),
            context={"tmp": tmp, "workdir": workdir},
        )

    def collect_artifacts(self, prepared: PreparedEnvironment, sandbox: Any) -> dict:
        diff = subprocess.run(
            ["git", "diff"],
            cwd=prepared.context["workdir"],
            capture_output=True,
            text=True,
            check=False,
        )
        return {"patch": diff.stdout or None}

    def cleanup(self, prepared: PreparedEnvironment) -> None:
        shutil.rmtree(prepared.context["tmp"], ignore_errors=True)
