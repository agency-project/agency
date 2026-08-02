from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..base import BenchmarkProvider, BenchmarkResult, BenchmarkTask


class SWEBenchProvider(BenchmarkProvider):
    """Loads SWE-bench tasks from a local JSONL export (a standard HuggingFace
    dataset export — this provider has no `datasets` dependency of its own)
    and runs them through an Agency agent to produce a patch.

    What this provider implements: task loading (clone the repo at
    `base_commit`), Agency execution, patch generation, and exporting
    predictions in the format the *official* SWE-bench evaluation harness
    expects (`write_predictions()`). What it deliberately does NOT implement
    is SWE-bench correctness evaluation — whether an issue was actually
    resolved can only be determined by running that official harness (it
    applies the patch and runs the instance's real test suite). This
    provider's own `evaluate()` never fabricates a pass/fail signal from
    patch size or similarity to the gold patch — resolution status stays
    `None` until the official harness has been run against the exported
    predictions file.
    """

    def __init__(self, tasks_path: Path, cache_dir: Path):
        self.tasks_path = Path(tasks_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "swe-bench"

    def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        records = []
        with self.tasks_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break

        return [self._task_from_record(r) for r in records]

    def evaluate(self, task: BenchmarkTask, result: BenchmarkResult) -> dict:
        patch = result.patch or ""
        changed_lines = sum(
            1
            for line in patch.splitlines()
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        )
        # "passed" (did the issue actually get fixed?) needs the official SWE-bench
        # test-suite harness, which is out of scope here — leave it unknown rather
        # than conflating "produced a patch" with "solved the issue." Patch size/
        # presence is diagnostic only, never treated as a pass/fail proxy.
        return {
            "passed": None,
            "note": (
                "resolution not evaluated locally — run the official SWE-bench "
                "harness against the exported predictions file"
            ),
            "patch_generated": bool(patch.strip()),
            "lines_changed": changed_lines,
        }

    @staticmethod
    def to_prediction(task_id: str, patch: str | None, model_name_or_path: str) -> dict:
        """One record in the official SWE-bench harness's predictions format."""
        return {
            "instance_id": task_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": patch or "",
        }

    @classmethod
    def write_predictions(
        cls,
        records: list[dict],
        path: Path,
        model_name_or_path: str,
    ) -> Path:
        """Write `records` (Runner result records, each with `task_id`/`patch`)
        as a predictions.jsonl in the format the official SWE-bench evaluation
        harness (`python -m swebench.harness.run_evaluation --predictions_path
        ...`) expects. This provider's own `evaluate()` cannot determine pass/
        fail — running that harness against this file is what does."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for record in records:
                prediction = cls.to_prediction(
                    record["task_id"], record.get("patch"), model_name_or_path
                )
                f.write(json.dumps(prediction) + "\n")
        return path

    # ------------------------------------------------------------------
    # Repo checkout — a shared bare mirror per repo (cloned once, fetched by
    # many instances) plus a lightweight per-instance checkout from it, so a
    # dataset with hundreds of instances against the same repo doesn't
    # re-download that repo's history once per instance.
    # ------------------------------------------------------------------

    def _task_from_record(self, record: dict) -> BenchmarkTask:
        instance_id = record["instance_id"]
        workspace = self._checkout(record["repo"], record["base_commit"], instance_id)
        return BenchmarkTask(
            task_id=instance_id,
            benchmark=self.name,
            instructions=record["problem_statement"],
            workspace=workspace,
            metadata={
                "repo": record["repo"],
                "base_commit": record["base_commit"],
                "gold_patch": record.get("patch"),
            },
        )

    def _checkout(self, repo: str, base_commit: str, instance_id: str) -> Path:
        mirror = self._ensure_mirror(repo)

        instance_dir = self.cache_dir / "instances" / instance_id
        if not instance_dir.exists():
            instance_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--no-checkout", str(mirror), str(instance_dir)], check=True
            )
            subprocess.run(["git", "checkout", base_commit], cwd=instance_dir, check=True)
        return instance_dir

    def _ensure_mirror(self, repo: str) -> Path:
        mirror = self.cache_dir / "repos" / f"{repo.replace('/', '__')}.git"
        if not mirror.exists():
            mirror.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--mirror", f"https://github.com/{repo}.git", str(mirror)],
                check=True,
            )
        return mirror
