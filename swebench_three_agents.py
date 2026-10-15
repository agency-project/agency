"""Three Codex agents repair one public SWE-bench issue, with a live profiler."""

import base64
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
TASK_DIR = ROOT / "runs/swebench-three"
RULES = (
    "Work in /workspace at the unresolved issue base commit. Use "
    "/opt/task-python/bin/python and /opt/task-python/bin/pytest for repository tests. "
    "Do not fetch upstream fixes or reference solutions. Do not use paid proxies or "
    "external proxy credentials; reproduce with local servers or mocks. Do not modify "
    ".git or install into the daemon Python. Report exact test commands, results and "
    "environment limitations honestly. "
)


def skill(name, instructions, **inputs):
    return agskill(
        name=name,
        prompt=RULES + instructions,
        input_schema=agdata(**inputs),
        output_schema=agdata(summary=str),
    )


INVESTIGATE = skill(
    "investigate_proxy_issue",
    "Investigate the issue and reproduce it. Explain the likely failure mechanism "
    "and edge cases to the fixer. Read code and run checks, but do not edit source or tests.",
    issue=str,
)
FIX = skill(
    "fix_proxy_issue",
    "Implement a minimal fix for the supplied issue and add regression tests. "
    "Run relevant tests. You own the final source patch.",
    issue=str,
)
BASELINE = skill(
    "plan_proxy_validation",
    "Independently investigate existing proxy/authentication tests, run baseline checks, "
    "and propose regression cases. Do not edit source or tests. Your next task will "
    "validate the fixer's completed patch.",
    issue=str,
)
REFINE = skill(
    "review_parallel_findings",
    "Review your current patch against the investigator's findings and the validator's "
    "baseline report. Address relevant issues, strengthen regression coverage, and rerun "
    "the relevant tests. Leave the final patch in this workspace.",
    issue=str,
    investigation=str,
    baseline=str,
)
VALIDATE = skill(
    "validate_final_patch",
    "The fixer's final patch has now been applied in your workspace. Independently "
    "review it, reproduce the reported scenario, and run relevant tests and broader "
    "local tests where practical. Do not edit the patch. Clearly report any remaining "
    "defects or verification gaps; passing tools alone does not prove issue resolution.",
    issue=str,
    fixer_report=str,
    investigation=str,
)


def resolved(result):
    result.wait(timeout=1800)
    fields = result.to_dict()
    if "error" in fields:
        raise RuntimeError(fields["error"])
    return fields["summary"]


def export_patch(sandbox, run_dir, task):
    encoded_patch, rc = sandbox.exec(
        "set -o pipefail; git add -N . && git diff --binary | base64 -w0", timeout=30
    )
    if rc:
        raise RuntimeError(f"Patch export failed: {encoded_patch}")
    # exec() trims output newlines; encode the diff to preserve its exact bytes.
    patch = base64.b64decode(encoded_patch, validate=True).decode("utf-8")
    (run_dir / "prediction.patch").write_text(patch)
    record = {
        "instance_id": task["instance_id"],
        "model_name_or_path": os.environ.get("AGENCY_LLM_MODEL", "gpt-6-luna")
        + "/codex-three-agents",
        "model_patch": patch,
    }
    (run_dir / "predictions.jsonl").write_text(json.dumps(record) + "\n")
    return patch


def main():
    task = json.loads((TASK_DIR / "task.json").read_text())
    cfg, run_dir = tutorial_config(
        "swebench_three_" + task["instance_id"], harness="codex", max_concurrent_engines=3
    )
    runtime = next((name for name in ("podman", "docker") if shutil.which(name)), None)
    if runtime is None:
        raise RuntimeError("Podman or Docker is required.")
    cfg.sandbox.backend = runtime
    cfg.sandbox.base_image = "localhost/agency-swebench-three:latest"
    cfg.sandbox.gpu_passthrough = False
    cfg.sandbox.mounts["task_input"] = (str(TASK_DIR / "input"), "/opt/task-input", "ro")
    cfg.orchestrator.db_path = str(Path(cfg.agent.log_dir) / "global_data.sqlite3")
    cfg.harness_adapter.binary_path = str(
        Path(sys.prefix) / "codex-0.147.0/package/vendor/x86_64-unknown-linux-musl/bin/codex"
    )
    if not Path(cfg.harness_adapter.binary_path).is_file():
        raise RuntimeError("The pinned Codex binary is missing from .venv.")
    agents = [
        Agent("swe-investigator", agconfig=cfg, harness="codex"),
        Agent("swe-fixer", agconfig=cfg, harness="codex"),
        Agent("swe-validator", agconfig=cfg, harness="codex"),
    ]
    investigator, fixer, validator = agents

    def workload():
        print(f"Task: {task['instance_id']} ({task['repo']})")
        print(f"Artifacts: {run_dir}")
        for agent in agents:
            sandbox = agent._ensure_sandbox()
            output, rc = sandbox.exec(
                "cp -a /opt/task-input/. /workspace/ && "
                "git config --global --add safe.directory /workspace && "
                "/opt/task-python/bin/pip install --no-deps -e /workspace",
                timeout=300,
            )
            if rc:
                raise RuntimeError(f"{agent.agname} repository setup failed: {output}")
            head, rc = sandbox.exec("git rev-parse HEAD")
            if rc or head.strip() != task["base_commit"]:
                raise RuntimeError("Repository is not at the task's base commit.")
        input("Open the live profiler, then press Enter to start all three agents: ")
        try:
            # Launch all three before waiting: each has its own writable copy.
            investigation = investigator.run(INVESTIGATE, agdata(issue=task["problem_statement"]))
            implementation = fixer.run(FIX, agdata(issue=task["problem_statement"]))
            baseline = validator.run(BASELINE, agdata(issue=task["problem_statement"]))
            findings, first_fix, baseline_report = [
                resolved(result) for result in (investigation, implementation, baseline)
            ]
            for name, report in (
                ("investigation", findings),
                ("initial-fix", first_fix),
                ("baseline", baseline_report),
            ):
                (run_dir / f"{name}.txt").write_text(report + "\n")
            fixer_report = resolved(
                fixer.run(
                    REFINE,
                    agdata(
                        issue=task["problem_statement"],
                        investigation=findings,
                        baseline=baseline_report,
                    ),
                )
            )
            (run_dir / "fixer-summary.txt").write_text(fixer_report + "\n")
            patch = export_patch(fixer.sandbox, run_dir, task)
            if not patch.strip():
                raise RuntimeError("The fixer produced an empty patch.")
            encoded = base64.b64encode(patch.encode()).decode()
            output, rc = validator.sandbox.exec(
                f"printf '%s' '{encoded}' | base64 -d > /tmp/fixer.patch && "
                "git apply --check /tmp/fixer.patch && git apply /tmp/fixer.patch",
                timeout=30,
            )
            if rc:
                raise RuntimeError(
                    f"Cannot apply fixer's patch for independent validation: {output}"
                )
            report = resolved(
                validator.run(
                    VALIDATE,
                    agdata(
                        issue=task["problem_statement"],
                        fixer_report=fixer_report,
                        investigation=findings,
                    ),
                )
            )
            (run_dir / "validation-summary.txt").write_text(report + "\n")
            print(f"Fixer:\n{fixer_report}\n\nIndependent validation:\n{report}")
        finally:
            export_patch(fixer.sandbox, run_dir, task)
            print(f"Saved patch: {run_dir / 'prediction.patch'}")

    try:
        agwebui.run(
            workload,
            run_dir=Path(cfg.agent.log_dir),
            port=int(os.environ.get("AGENCY_WEBUI_PORT", "7862")),
            linger=True,
        )
    finally:
        close_sandboxes(agents)


if __name__ == "__main__":
    run_example(main)
