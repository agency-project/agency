"""Executable published-suite integrations. Optional dependencies are lazy."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

from .common import atomic_json, digest, read_json
from .planning import HARBOR_VERSION, TERMINAL_RELEASE


class PrerequisiteError(RuntimeError):
    pass


def checked_exec(environment, command):
    output, code = environment.exec(command)
    if code:
        raise PrerequisiteError(f"Environment setup failed ({code}): {output}")
    return output


def directory_hash(directory):
    directory = Path(directory)
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            records.append((str(path.relative_to(directory)), path.read_bytes().hex()))
    return digest(records)


class SweBenchAdapter:
    def setup(self, task, environment):
        repository = Path(task["repository_path"])
        if not repository.is_dir():
            raise PrerequisiteError(
                "SWE-bench repository_path missing; use prepare --suite swebench"
            )
        # Archive only the base commit, never evaluator patches, tests or host gold.
        archive = subprocess.run(
            ["git", "-C", str(repository), "archive", "--format=tar", task["base_commit"]],
            check=True,
            capture_output=True,
        ).stdout
        environment.write_file_bytes("/tmp/agency-base.tar", archive)
        checked_exec(
            environment, "mkdir -p /workspace && tar xf /tmp/agency-base.tar -C /workspace"
        )
        checked_exec(
            environment,
            "cd /workspace && git init -q && git add . && "
            "git -c user.name=Fixture -c user.email=fixture@localhost commit -qm base",
        )
        for command in task.get("setup_commands", []):
            checked_exec(environment, command)
        return task["problem_statement"]

    def extract_patch(self, environment):
        checked_exec(environment, "cd /workspace && git add -N .")
        return checked_exec(environment, "cd /workspace && git diff --binary HEAD")

    def export_prediction(self, task, patch, model_id, path):
        prediction = {
            "instance_id": task["id"],
            "model_name_or_path": model_id,
            "model_patch": patch,
        }
        Path(path).write_text(json.dumps(prediction) + "\n")
        return prediction

    def evaluation_command(self, task, predictions, run_id):
        dataset = task.get("dataset", "princeton-nlp/SWE-bench_Verified")
        if dataset not in ("princeton-nlp/SWE-bench_Verified", "princeton-nlp/SWE-bench_Lite"):
            raise ValueError("Only SWE-bench Verified and Lite are supported")
        return [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset,
            "--predictions_path",
            str(predictions),
            "--instance_ids",
            task["id"],
            "--max_workers",
            "1",
            "--run_id",
            run_id,
        ]

    def import_report(self, task, report):
        instance_id = task["id"]
        # Official per-instance report.json maps instance IDs to resolved/test results.
        if instance_id in report and isinstance(report[instance_id], dict):
            verdict = report[instance_id]
            if isinstance(verdict.get("resolved"), bool):
                return {
                    "success": verdict["resolved"],
                    "failure": None if verdict["resolved"] else "task",
                    "official": verdict,
                }
        # Official aggregate reports use resolved_ids, unresolved_ids, error_ids.
        if instance_id in report.get("resolved_ids", []):
            return {"success": True, "failure": None, "official": report}
        if instance_id in report.get("unresolved_ids", []) or instance_id in report.get(
            "empty_patch_ids", []
        ):
            return {"success": False, "failure": "task", "official": report}
        return {"success": None, "failure": "infrastructure", "official": report}

    def evaluate(self, task, prediction, directory, command_runner=subprocess.run):
        run_id = "agency-annotation-" + uuid.uuid4().hex
        directory = Path(directory)
        atomic_json(
            directory / "evaluation.json", {"run_id": run_id, "prediction": str(prediction)}
        )
        completed = command_runner(
            self.evaluation_command(task, prediction, run_id),
            cwd=directory,
            capture_output=True,
            text=True,
        )
        (directory / "evaluation.stdout").write_text(completed.stdout)
        (directory / "evaluation.stderr").write_text(completed.stderr)
        report_path = directory / "logs" / "run_evaluation" / run_id
        reports = list(report_path.rglob("report.json")) if report_path.exists() else []
        if not reports:
            reports = list(directory.glob(f"*{run_id}*.json"))
        for path in reports:
            report = read_json(path)
            result = self.import_report(task, report)
            if result["success"] is not None:
                return {**result, "evaluation_run_id": run_id}
        return {
            "success": None,
            "failure": "infrastructure",
            "evaluation_run_id": run_id,
            "returncode": completed.returncode,
            "error": "No official verdict found; inspect evaluation logs",
        }


class TerminalBenchAdapter:
    def job_config(self, task, trial, model, config, directory):
        if task.get("release") != TERMINAL_RELEASE:
            raise ValueError(f"Terminal-Bench task release must be {TERMINAL_RELEASE}")
        task_path = Path(task["path"]).resolve()
        if not task_path.is_dir():
            raise PrerequisiteError("Terminal task missing; prepare the pinned dataset first")
        if directory_hash(task_path) != task["content_hash"]:
            raise ValueError("Terminal task contents changed after sampling")
        return {
            "job_name": "agency-" + trial["trial_id"],
            "jobs_dir": str(Path(directory).resolve()),
            "n_attempts": 1,
            "n_concurrent_trials": 1,
            "retry": {"max_retries": 0},
            "tasks": [{"path": str(task_path)}],
            "agents": [
                {
                    "import_path": "benchmarks.tool_annotation_effect.harbor_agent:AgencyNativeAgent",
                    "model_name": model["model"],
                    "kwargs": {
                        "arm": trial["arm"],
                        "model_config": model,
                        "experiment_config": config,
                        "run_id": trial["trial_id"],
                    },
                }
            ],
        }

    def launch(self, task, trial, model, config, directory, command_runner=subprocess.run):
        import importlib.metadata

        try:
            version = importlib.metadata.version("harbor")
        except importlib.metadata.PackageNotFoundError:
            raise PrerequisiteError(f"Install optional harbor=={HARBOR_VERSION}") from None
        if version != HARBOR_VERSION:
            raise PrerequisiteError(f"Expected harbor=={HARBOR_VERSION}; found {version}")
        directory = Path(directory)
        job = self.job_config(task, trial, model, config, directory / "harbor")
        atomic_json(directory / "harbor-job.json", job)
        completed = command_runner(
            ["harbor", "run", "--config", str(directory / "harbor-job.json")],
            capture_output=True,
            text=True,
        )
        (directory / "harbor.stdout").write_text(completed.stdout)
        (directory / "harbor.stderr").write_text(completed.stderr)
        reports = sorted((directory / "harbor").rglob("result.json"))
        for path in reports:
            report = read_json(path)
            if report.get("task_name") == task["id"]:
                atomic_json(directory / "harbor-result.json", report)
                return report
        raise PrerequisiteError(
            f"Harbor returned {completed.returncode} without a trial result; inspect logs"
        )

    def import_report(self, report):
        if report.get("exception_info"):
            return {"success": None, "failure": "infrastructure", "official": report}
        rewards = (report.get("verifier_result") or {}).get("rewards")
        if not isinstance(rewards, dict) or "reward" not in rewards:
            return {"success": None, "failure": "infrastructure", "official": report}
        reward = rewards["reward"]
        return {
            "success": reward == 1,
            "reward": reward,
            "failure": None if reward == 1 else "task",
            "official": report,
        }


def prepare(suite, directory, *, lite=False):
    """Explicit future preparation; never called by plan/import/tests."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if suite == "swebench":
        try:
            from datasets import load_dataset
        except ImportError:
            raise PrerequisiteError(
                "Install optional datasets and swebench before prepare"
            ) from None
        dataset = "princeton-nlp/SWE-bench_Lite" if lite else "princeton-nlp/SWE-bench_Verified"
        rows = load_dataset(dataset, split="test")
        tasks = []
        for row in rows:
            repository = directory / "repositories" / row["repo"].replace("/", "__")
            if not repository.exists():
                repository.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--mirror",
                        "https://github.com/" + row["repo"] + ".git",
                        str(repository),
                    ],
                    check=True,
                )
            # Persist gold outside any agent environment, in evaluator-only metadata.
            tasks.append(
                {
                    **row,
                    "id": row["instance_id"],
                    "stratum": row["repo"],
                    "repository_path": str(repository),
                    "dataset": dataset,
                }
            )
        atomic_json(directory / "tasks.json", tasks)
    elif suite == "terminalbench":
        subprocess.run(
            [
                "harbor",
                "datasets",
                "download",
                TERMINAL_RELEASE,
                "--export",
                "--output-dir",
                str(directory),
            ],
            check=True,
        )
        tasks = []
        for instruction in sorted(directory.rglob("instruction.md")):
            path = instruction.parent
            if (path / "task.toml").exists():
                tasks.append(
                    {
                        "id": path.name,
                        "stratum": "terminal",
                        "path": str(path),
                        "release": TERMINAL_RELEASE,
                        "content_hash": directory_hash(path),
                    }
                )
        if not tasks:
            raise PrerequisiteError("Harbor download produced no task directories")
        atomic_json(directory / "tasks.json", tasks)
    else:
        raise ValueError("Local fixtures require no preparation")
    return directory / "tasks.json"
