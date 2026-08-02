from __future__ import annotations

from typing import Any

from agency.agsandbox import agSandboxConfig

from ..base import BenchmarkTask, ExecutionEnvironment, PreparedEnvironment


class ContainerImageEnvironment(ExecutionEnvironment):
    """Points the sandbox at a container image the task itself defines,
    instead of a generic sandbox image. Used by Terminal-Bench: each task
    supplies its own `task.metadata["image"]` (built/pulled by the provider
    at load_tasks() time — this environment never builds anything itself)
    and its own working directory.

    Grading needs the exact filesystem state the agent left behind, but the
    sandbox's per-tool-call container is torn down after the run and only
    its commit — a lifecycle image tag — survives that teardown. So before
    the sandbox is destroyed, collect_artifacts() retags that commit to a
    stable, predictable name and hands it back via
    `BenchmarkResult.metadata["checkpoint_image"]` for
    `BenchmarkProvider.evaluate()` to spin up, grade, and clean up.
    """

    def __init__(self, image_prefix: str = "benchmark-result"):
        self._image_prefix = image_prefix

    def prepare(self, cfg: Any, task: BenchmarkTask) -> PreparedEnvironment:
        image = task.metadata.get("image")
        if not image:
            raise ValueError(f"task {task.task_id!r} has no metadata['image'] to run against")
        workdir = task.metadata.get("workdir") or "/"

        agSandboxConfig(cfg).set_base_image(image)

        return PreparedEnvironment(
            prompt_hint=(
                f"You do not start in /workspace — your files are in this "
                f"container's own working directory, {workdir!r}. Pass "
                f"workdir={workdir!r} on bash calls (or `cd` there first)."
            ),
            context={"task_id": task.task_id},
        )

    def collect_artifacts(self, prepared: PreparedEnvironment, sandbox: Any) -> dict:
        checkpoint_image = None
        if sandbox is not None and sandbox._checkpoint_image is not None:
            checkpoint_image = f"{self._image_prefix}-{prepared.context['task_id']}".lower()
            # type(sandbox._backend), not a bare container-only forwarder --
            # matches the dispatch agSandbox.fork() uses for the same reason
            # (a chroot-backed sandbox's checkpoint isn't a container image).
            type(sandbox._backend).tag_image(sandbox._checkpoint_image, checkpoint_image)
        return {"metadata": {"checkpoint_image": checkpoint_image}}
