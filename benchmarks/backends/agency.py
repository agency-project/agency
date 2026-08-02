from __future__ import annotations

import time

from agency import agdata, agent, agskill
from agency.agconfig import agConfig

from ..base import AgentBackend, BenchmarkResult, BenchmarkTask, ExecutionEnvironment


class AgencyBackend(AgentBackend):
    """Runs a BenchmarkTask through a fresh Agency agent inside its sandbox.

    How the task's files reach the sandbox — and how artifacts are pulled
    back out afterward — is delegated entirely to an `ExecutionEnvironment`
    (e.g. a bind-mounted host workspace for SWE-bench, or a task-owned
    container image for Terminal-Bench). This backend only knows how to run
    an Agency agent; it has no opinion on the shape of the environment.

    A fresh agent/sandbox is created per task and destroyed afterward,
    releasing its checkpoint image immediately rather than letting them
    accumulate across a sweep.
    """

    def __init__(self, agconfig: agConfig, environment: ExecutionEnvironment):
        self._agconfig = agconfig
        self._environment = environment

    @property
    def name(self) -> str:
        return "agency"

    def run_task(self, task: BenchmarkTask) -> BenchmarkResult:
        cfg = self._agconfig.clone()
        prepared = self._environment.prepare(cfg, task)

        skill = agskill(
            name="benchmark_task",
            system_prompt=(
                f"You are working on benchmark task {task.task_id!r} "
                f"({task.benchmark}).\n"
                f"{prepared.prompt_hint}\n"
                "When you are done (or cannot proceed further), report your outcome."
            ),
            input_schema=agdata(instructions=str),
            output_schema=agdata(status=str, summary=str),
        )

        ag = agent(agconfig=cfg)
        try:
            t0 = time.monotonic()
            result = ag.run(skill, agdata(instructions=task.instructions))
            result.wait()
            elapsed_s = time.monotonic() - t0
        finally:
            artifacts = self._environment.collect_artifacts(prepared, ag.sandbox)
            if ag.sandbox is not None:
                ag.sandbox.destroy()
            self._environment.cleanup(prepared)

        result_dict = result.to_dict()
        error = result_dict.get("error")
        completed = error is None

        return BenchmarkResult(
            task_id=task.task_id,
            completed=completed,
            summary=result_dict.get("summary", "") if completed else error,
            patch=artifacts.get("patch"),
            metadata={
                **({"status": result_dict["status"]} if completed else {}),
                **artifacts.get("metadata", {}),
            },
            elapsed_s=elapsed_s,
        )
