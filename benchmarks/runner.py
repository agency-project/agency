from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Callable

from .base import AgentBackend, BenchmarkProvider, BenchmarkTask


class Runner:
    """Connects a BenchmarkProvider's tasks to an AgentBackend, collecting a
    BenchmarkResult + evaluate() metrics per task.

    Each task's outcome is flushed to `output_dir/results.jsonl` immediately
    after it completes, so a sweep of hundreds of tasks survives a crash
    partway through with every already-finished task's result intact on
    disk. A single task raising (backend or evaluate()) is caught and
    recorded as a failed task rather than aborting the rest of the sweep.
    """

    def __init__(self, provider: BenchmarkProvider, backend: AgentBackend, output_dir: Path):
        self.provider = provider
        self.backend = backend
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_dir / "results.jsonl"

    def run(
        self,
        tasks: list[BenchmarkTask],
        on_result: "Callable[[dict], None] | None" = None,
    ) -> dict:
        records = []
        with self.results_path.open("w") as f:
            for task in tasks:
                record = self._run_one(task)
                records.append(record)
                f.write(json.dumps(record) + "\n")
                f.flush()
                if on_result is not None:
                    on_result(record)
        return self._build_report(records)

    def _run_one(self, task: BenchmarkTask) -> dict:
        try:
            result = self.backend.run_task(task)
        except Exception as exc:
            return {
                "task_id": task.task_id,
                "completed": False,
                "summary": f"backend raised: {exc}",
                "patch": None,
                "metadata": {},
                "elapsed_s": 0.0,
                "metrics": {"passed": None},
            }

        try:
            metrics = self.provider.evaluate(task, result)
        except Exception as exc:
            metrics = {"passed": None, "error": f"evaluate() raised: {exc}"}

        return {
            "task_id": result.task_id,
            "completed": result.completed,
            "summary": result.summary,
            "patch": result.patch,
            "metadata": result.metadata,
            "elapsed_s": result.elapsed_s,
            "metrics": metrics,
        }

    def _build_report(self, records: list[dict]) -> dict:
        elapsed = [r["elapsed_s"] for r in records]
        completed_count = sum(1 for r in records if r["completed"])
        passed_flags = [
            r["metrics"].get("passed") for r in records if r["metrics"].get("passed") is not None
        ]
        pass_count = sum(1 for p in passed_flags if p)

        return {
            "provider": self.provider.name,
            "backend": self.backend.name,
            "task_count": len(records),
            "completed_count": completed_count,
            "completion_rate": completed_count / len(records) if records else None,
            "pass_count": pass_count,
            "pass_rate": (pass_count / len(passed_flags)) if passed_flags else None,
            "elapsed_s": {
                "mean": statistics.fmean(elapsed) if elapsed else None,
                "p95": _percentile(elapsed, 0.95) if elapsed else None,
            },
            "results": records,
        }

    @staticmethod
    def save_report(report: dict, path: Path) -> None:
        """Write `report` as JSON to `path`, and a human-readable summary
        alongside it at the same path with a `.txt` suffix."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        path.with_suffix(".txt").write_text(_format_summary(report))


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]


def _fmt_rate(rate: "float | None") -> str:
    return f"{rate:.1%}" if rate is not None else "n/a"


def _fmt_seconds(seconds: "float | None") -> str:
    return f"{seconds:.1f}s" if seconds is not None else "n/a"


def _format_summary(report: dict) -> str:
    lines = [
        f"provider        : {report['provider']}",
        f"backend         : {report['backend']}",
        f"tasks           : {report['task_count']}",
        f"completed       : {report['completed_count']} ({_fmt_rate(report['completion_rate'])})",
        f"passed          : {report['pass_count']} ({_fmt_rate(report['pass_rate'])})",
        f"elapsed (mean)  : {_fmt_seconds(report['elapsed_s']['mean'])}",
        f"elapsed (p95)   : {_fmt_seconds(report['elapsed_s']['p95'])}",
    ]
    return "\n".join(lines) + "\n"
