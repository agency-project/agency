from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkTask:
    """A single task loaded from a benchmark provider, ready to hand to an agent backend."""

    task_id: str
    benchmark: str
    instructions: str
    workspace: Path | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """The outcome of running one BenchmarkTask through an AgentBackend.

    `completed` reflects whether the agent finished without crashing — it is not
    pass/fail. Pass/fail is determined by BenchmarkProvider.evaluate(), which scores
    the result separately (often via the benchmark's own official harness).
    """

    task_id: str
    completed: bool
    summary: str
    patch: str | None = None
    metadata: dict = field(default_factory=dict)
    elapsed_s: float = 0.0


class BenchmarkProvider(ABC):
    """Loads tasks for one benchmark and scores an agent's result against it."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this benchmark, e.g. "swe-bench"."""

    @abstractmethod
    def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        """Return up to `limit` tasks (all of them if `limit` is None)."""

    @abstractmethod
    def evaluate(self, task: BenchmarkTask, result: BenchmarkResult) -> dict:
        """Score `result` against `task` and return provider-specific metrics."""


class AgentBackend(ABC):
    """Runs a single BenchmarkTask against an agent implementation."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend, e.g. "agency"."""

    @abstractmethod
    def run_task(self, task: BenchmarkTask) -> BenchmarkResult:
        """Execute `task` and return the resulting BenchmarkResult."""


@dataclass
class PreparedEnvironment:
    """What ExecutionEnvironment.prepare() hands back to an AgentBackend for
    one task run: text describing where the agent's files live (spliced into
    the agent's system prompt), plus whatever private state this
    environment's own collect_artifacts()/cleanup() need later."""

    prompt_hint: str
    context: Any = None


class ExecutionEnvironment(ABC):
    """Strategy for how a task's files are provided to an agent's sandbox and
    how artifacts are pulled back out afterward — independent of which agent
    implementation is doing the work.

    A benchmark's *backend* (e.g. Agency) stays the same regardless of how a
    task's environment is shaped; only this strategy changes between, say, a
    host directory bind-mounted into a generic sandbox image (SWE-bench) and
    a container image the task itself defines (Terminal-Bench).

    Callers run these in a fixed order: `prepare()` before the agent runs,
    `collect_artifacts()` after the run finishes but *before* the sandbox is
    destroyed (so implementations that need the live sandbox — e.g. to retag
    a checkpoint image — still can), then `cleanup()` once the sandbox is
    gone.
    """

    @abstractmethod
    def prepare(self, cfg: Any, task: BenchmarkTask) -> PreparedEnvironment:
        """Configure `cfg`'s sandbox for `task` (e.g. mount a workspace / set
        a base image) and return the resulting PreparedEnvironment."""

    @abstractmethod
    def collect_artifacts(self, prepared: PreparedEnvironment, sandbox: Any) -> dict:
        """Called after the agent run, before the sandbox is destroyed.
        `sandbox` is the agent's live sandbox object (or None if one was
        never provisioned). Returns a dict merged into the BenchmarkResult —
        recognized keys are `patch` and `metadata`."""

    def cleanup(self, prepared: PreparedEnvironment) -> None:
        """Release any host-side resources prepare() created (e.g. a temp
        dir). Called after the sandbox has been destroyed. Default: no-op."""
