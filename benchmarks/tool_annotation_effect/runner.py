"""Sequential execution with immutable attempts and explicit partial recovery."""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

from .common import atomic_json, digest, immutable_json, read_json, native, source_hash
from .planning import load_plan, versions


@contextmanager
def experiment_lock(directory):
    path = Path(directory) / ".run.lock"
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another command is mutating this experiment") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run(directory, *, resume_partial=False, executor=None, limit=None):
    from .execution import execute

    executor = executor or execute
    directory = Path(directory).resolve()
    manifest = load_plan(directory)
    if manifest["implementation_hash"] != source_hash():
        raise ValueError(
            "Implementation differs from the saved plan; sync matching code or create a new plan"
        )
    tasks = {(task["suite"], task["id"]): task["data"] for task in manifest["tasks"]}
    models = {model["id"]: model for model in manifest["config"]["models"]}
    processed = 0
    with experiment_lock(directory):
        for trial in manifest["schedule"]:
            if limit is not None and processed >= limit:
                break
            trial_dir = directory / "trials" / trial["trial_id"]
            status_path = trial_dir / "status.json"
            if status_path.exists():
                status = read_json(status_path)
                if status["state"] == "completed":
                    continue
                if not resume_partial:
                    raise RuntimeError(
                        f"Partial attempt in {trial_dir}; use --resume-partial to preserve it and start a fresh environment"
                    )
                previous = trial_dir / status["attempt"]
                atomic_json(
                    previous / "status.json",
                    {**status, "state": "interrupted", "failure": "infrastructure"},
                )
            trial_dir.mkdir(parents=True, exist_ok=True)
            immutable_json(trial_dir / "assignment.json", trial)
            number = len(list(trial_dir.glob("attempt-*"))) + 1
            attempt = trial_dir / f"attempt-{number:03d}"
            attempt.mkdir()
            status = {
                "state": "running",
                "attempt": attempt.name,
                "started_at": time.time(),
                "pid": os.getpid(),
                "versions": versions(),
            }
            atomic_json(status_path, status)
            atomic_json(attempt / "status.json", status)
            immutable_json(attempt / "assignment.json", trial)
            try:
                profiling = manifest["config"].get("profiling", {})
                profile_context = nullcontext()
                if profiling.get("enabled"):
                    from agency import agprof

                    profile_context = agprof.session(
                        attempt / "profile",
                        **{key: value for key, value in profiling.items() if key != "enabled"},
                    )
                with profile_context:
                    result = executor(
                        tasks[(trial["suite"], trial["task_id"])],
                        trial,
                        models[trial["model_id"]],
                        manifest["config"],
                        attempt,
                    )
                result = native("annotations").redact(result)
                atomic_json(attempt / "execution.json", result)
                treatment_events = [
                    event for event in result.get("events", []) if event["kind"] == "treatment"
                ]
                immutable_json(attempt / "treatment-manifest.json", treatment_events)
                from .analysis import normalize

                atomic_json(attempt / "actions.json", normalize(result.get("events", [])))
                status = {
                    **status,
                    "state": "completed",
                    "finished_at": time.time(),
                    "failure": result.get("failure"),
                }
            except Exception as error:
                # Failed assigned attempts are terminal; resume never selectively retries them.
                status = {
                    **status,
                    "state": "completed",
                    "finished_at": time.time(),
                    "failure": "infrastructure",
                    "error_type": type(error).__name__,
                    "error": native("annotations").redact(str(error)),
                }
                atomic_json(
                    attempt / "failure.json",
                    {
                        "failure": "infrastructure",
                        "error_type": type(error).__name__,
                        "error": status["error"],
                    },
                )
            atomic_json(attempt / "status.json", status)
            atomic_json(status_path, status)
            processed += 1
    return processed


def evaluate(directory, evaluator=None):
    from .adapters import SweBenchAdapter, TerminalBenchAdapter
    from .execution import verify_migration
    from .fixtures import evaluate_rag, evaluate_tandem

    directory = Path(directory).resolve()
    manifest = load_plan(directory)
    tasks = {(task["suite"], task["id"]): task["data"] for task in manifest["tasks"]}
    with experiment_lock(directory):
        for trial in manifest["schedule"]:
            trial_dir = directory / "trials" / trial["trial_id"]
            if not (trial_dir / "status.json").exists():
                continue
            status = read_json(trial_dir / "status.json")
            if status["state"] != "completed":
                continue
            attempt = trial_dir / status["attempt"]
            if (attempt / "verifier.json").exists():
                continue
            execution = (
                read_json(attempt / "execution.json")
                if (attempt / "execution.json").exists()
                else {}
            )
            task = tasks[(trial["suite"], trial["task_id"])]
            started = time.monotonic()
            if status.get("failure") == "infrastructure" or (
                status.get("failure") == "budget"
                and trial["suite"] not in ("swebench", "terminalbench")
            ):
                result = {
                    "success": False if status["failure"] == "budget" else None,
                    "failure": status["failure"],
                }
            elif evaluator is not None:
                result = evaluator(task, trial, execution, attempt)
            elif trial["suite"] == "rag":
                result = evaluate_rag(task, execution.get("final_text", ""))
            elif trial["suite"] == "tandem":
                result = evaluate_tandem(task, execution.get("final_text", ""))
            elif trial["suite"] == "migration":
                if not (attempt / "solution.rs").exists():
                    result = {
                        "success": False,
                        "failure": "task",
                        "rust_compilation": False,
                        "behavioral": None,
                    }
                else:
                    result = verify_migration(task, attempt)
            elif trial["suite"] == "swebench":
                result = SweBenchAdapter().evaluate(task, attempt / "prediction.jsonl", attempt)
            else:
                result = TerminalBenchAdapter().import_report(execution.get("harbor_report", {}))
            atomic_json(
                attempt / "verifier.json",
                {
                    **result,
                    "agent_failure": status.get("failure"),
                    "evaluation_seconds": time.monotonic() - started,
                },
            )


def rows(directory):
    directory = Path(directory)
    manifest = load_plan(directory)
    rows = []
    for trial in manifest["schedule"]:
        trial_dir = directory / "trials" / trial["trial_id"]
        attempts = sorted(trial_dir.glob("attempt-*"))
        if not attempts:
            rows.append(
                {
                    **trial,
                    "attempt": None,
                    "success": None,
                    "failure": "missing",
                    "actions": [],
                    "events": [],
                }
            )
        for attempt in attempts:
            status = read_json(attempt / "status.json")
            execution = (
                read_json(attempt / "execution.json")
                if (attempt / "execution.json").exists()
                else {}
            )
            verifier = (
                read_json(attempt / "verifier.json") if (attempt / "verifier.json").exists() else {}
            )
            artifact = None
            for name in ("prediction.patch", "solution.rs", "result.json"):
                if (attempt / name).exists():
                    artifact = digest((attempt / name).read_text())
            rows.append(
                {
                    **trial,
                    "attempt": attempt.name,
                    **execution.get("metrics", {}),
                    "success": verifier.get("success"),
                    "failure": verifier.get("failure")
                    if isinstance(verifier.get("success"), bool)
                    else (
                        verifier.get("failure")
                        or status.get("failure")
                        or ("missing_evaluation" if not verifier else None)
                    ),
                    "agent_failure": verifier.get("agent_failure", status.get("failure")),
                    "state": status["state"],
                    "events": execution.get("events", []),
                    "actions": read_json(attempt / "actions.json")
                    if (attempt / "actions.json").exists()
                    else [],
                    "artifact_hash": artifact,
                    "final_output_hash": digest(execution.get("final_text", "")),
                    **{
                        key: execution.get(key)
                        for key in (
                            "setup_seconds",
                            "agent_seconds",
                            "end_to_end_seconds",
                            "agent_executions",
                        )
                    },
                    "evaluation_seconds": verifier.get("evaluation_seconds"),
                }
            )
    return manifest, rows
