"""Run one downloaded SWE-bench issue in an isolated sandbox with live events."""

import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent / "examples"))
from agency import Agent, agdata, agskill
from agency.observability.agwebui import agwebui
from _common import close_sandboxes, run_example, tutorial_config

ROOT = Path(__file__).resolve().parent


def main():
    task = json.loads((ROOT / "runs/swebench/task.json").read_text())
    cfg, run_dir = tutorial_config("swebench_" + task["instance_id"], harness="codex")
    runtime = next((name for name in ("podman", "docker") if shutil.which(name)), None)
    if runtime is None:
        raise RuntimeError("This task requires Podman or Docker.")
    cfg.sandbox.backend = runtime
    cfg.sandbox.base_image = "localhost/agency-swebench:latest"
    cfg.sandbox.gpu_passthrough = False
    cfg.sandbox.mounts["task_input"] = (str(ROOT / "runs/swebench/input"), "/opt/task-input", "ro")
    cfg.orchestrator.db_path = str(Path(cfg.agent.log_dir) / "global_data.sqlite3")
    cfg.harness_adapter.binary_path = str(
        Path(sys.prefix) / "codex-0.147.0/package/vendor/x86_64-unknown-linux-musl/bin/codex"
    )
    if not Path(cfg.harness_adapter.binary_path).is_file():
        raise RuntimeError("The local Codex 0.147.0 binary is missing from this .venv.")
    learner = Agent("swebench-codex", agconfig=cfg, harness="codex")
    solve = agskill(
        name="repair_swebench_issue",
        prompt=(
            "Fix the supplied repository issue in /workspace. The repository is checked out "
            "at the unresolved SWE-bench base commit. Inspect the code, reproduce the bug, "
            "make a minimal source fix, add a regression test, and run relevant local tests. "
            "Use /opt/task-python/bin/python and /opt/task-python/bin/pytest for repository "
            "tests. A useful existing check is: /opt/task-python/bin/pytest -q "
            "tests/test_requests.py -k 'invalid_url or basic_building or params_are_added'. "
            "Do not fetch upstream fixes or reference solutions. Do not change the .git "
            "directory or install packages into the daemon's Python. Report the changes "
            "and the exact test results; report any limitations honestly."
        ),
        input_schema=agdata(issue=str),
        output_schema=agdata(summary=str),
    )

    def workload():
        print(f"Task: {task['instance_id']} ({task['repo']})")
        print(f"Artifacts: {run_dir}")
        sandbox = learner._ensure_sandbox()
        output, rc = sandbox.exec(
            "cp -a /opt/task-input/. /workspace/ && "
            "git -C /workspace config --global --add safe.directory /workspace && "
            "/opt/task-python/bin/pip install --no-deps -e /workspace",
            timeout=300,
        )
        if rc:
            raise RuntimeError(f"Repository setup failed: {output}")
        head, rc = sandbox.exec("git rev-parse HEAD")
        if rc or head.strip() != task["base_commit"]:
            raise RuntimeError("Repository is not at the task's base commit.")
        input("Open the live profiler, then press Enter to start the SWE-bench task: ")
        try:
            result = learner.run(solve, agdata(issue=task["problem_statement"]))
            result.wait(timeout=1800)
            fields = result.to_dict()
            if "error" in fields:
                raise RuntimeError(fields["error"])
            (run_dir / "agent-summary.txt").write_text(fields["summary"] + "\n")
            print(fields["summary"])
        finally:
            # Include newly added regression files in the exported diff.
            patch, rc = sandbox.exec("git add -N . && git diff --binary", timeout=30)
            if rc == 0:
                (run_dir / "prediction.patch").write_text(patch)
                prediction = {
                    "instance_id": task["instance_id"],
                    "model_name_or_path": "gpt-6-luna/codex",
                    "model_patch": patch,
                }
                (run_dir / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
                print(f"Saved patch: {run_dir / 'prediction.patch'}")
            else:
                print(f"Patch export failed: {patch}")

    try:
        agwebui.run(
            workload,
            run_dir=Path(cfg.agent.log_dir),
            port=int(os.environ.get("AGENCY_WEBUI_PORT", "7862")),
            linger=True,
        )
    finally:
        close_sandboxes([learner])


if __name__ == "__main__":
    run_example(main)
