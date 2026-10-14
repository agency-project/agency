"""Run one prepared SWE-bench task through Agency's real Codex harness.

Start agwebui.server separately against RUN_DIR/logs. The task file is a
single official dataset row; reference/test patches are never sent to Codex.
Use the official instance image with Python 3.12 added for Agency's daemon.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import threading
import time
from pathlib import Path

from agency import Agent, agdata, agrawstring, agskill
from agency.configs.agconfig import (
    agconfig,
    agentconfig,
    harnessadapterconfig,
    llmconfig,
    orchestratorconfig,
    resourcesconfig,
    sandboxconfig,
)
from agency.orchestrator import get_orchestrator

from smoke_test_models import load_env_file
from _common import close_sandboxes, run_example


def pace_provider_requests(interval):
    """Share a request-start budget across this example's model requests."""
    import httpx
    import openai

    from agency.llm.openai import _OpenAICompatibleBackend

    lock = threading.Lock()
    next_start = 0.0

    def pace(request):
        nonlocal next_start
        if request.url.path.endswith("/chat/completions"):
            with lock:
                delay = max(0, next_start - time.monotonic())
                if delay:
                    time.sleep(delay)
                next_start = time.monotonic() + interval

    def make_client(backend, timeout):
        return openai.OpenAI(
            api_key=backend.agconfig.llm.api_key,
            base_url=backend.agconfig.llm.base_url,
            timeout=timeout,
            max_retries=8,
            http_client=httpx.Client(timeout=timeout, event_hooks={"request": [pace]}),
        )

    # This opt-in example-local override also paces SDK retries. It does not
    # change Agency's production backend or the separate browser process.
    _OpenAICompatibleBackend.make_client = make_client


def prepare_workspace(actor, task):
    sandbox = actor._ensure_sandbox()
    output, code = sandbox.exec(
        "cp -a /testbed/. /workspace/ && printf '\nagency_runs/\n' >> /workspace/.git/info/exclude",
        workdir="/",
    )
    if code:
        raise RuntimeError(f"Workspace setup failed: {output}")
    output, code = sandbox.exec(
        "git checkout --detach " + shlex.quote(task["base_commit"]) + " && git rev-parse HEAD",
        workdir="/workspace",
    )
    if code or output.strip().splitlines()[-1] != task["base_commit"]:
        raise RuntimeError("Instance image does not match the task's base commit")


def wait_for_report(actor, result, timeout, path):
    try:
        result.wait(timeout=timeout)
    except TimeoutError:
        actor.cancel(result)
        result.wait(timeout=60)
        raise
    payload = result.to_dict()
    path.write_text(json.dumps(payload, indent=2))
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def extract_patch(actor, paths=""):
    patch, code = actor.sandbox.exec(
        "git add -N . && git diff --binary HEAD" + paths, workdir="/workspace"
    )
    if code:
        raise RuntimeError("Patch extraction failed")
    # Sandbox exec strips trailing output whitespace; git apply needs the
    # final patch line terminated even when the edited file has no newline.
    return patch + "\n" if patch else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--codex-binary", required=True)
    parser.add_argument("--source-dir", type=Path, help="Prepared checkout to mount at /testbed")
    parser.add_argument("--test-python", default="/opt/miniconda3/envs/testbed/bin/python")
    parser.add_argument("--harness-python", default="python3")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--request-interval", type=float, default=0)
    args = parser.parse_args()
    load_env_file(args.env_file)
    if args.request_interval > 0:
        pace_provider_requests(args.request_interval)
    os.environ.setdefault("AGENCY_PROFILE", "0")
    task = json.loads(args.task_file.read_text())
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)

    # Mount runtime code separately so credentials and reference patches stay on the host.
    runtime = directory / "runtime"
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(
        source / "agency", runtime / "agency", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(source / "pyproject.toml", runtime / "pyproject.toml")
    mounts = {"_agency_package": (str(runtime), "/opt/agency_pkg", "ro")}
    if args.source_dir:
        mounts["task_source"] = (str(args.source_dir.resolve()), "/testbed", "ro")
    cfg = agconfig(
        llmconfig(
            provider="openai",
            model=args.model,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ["OPENAI_API_KEY"],
            reasoning_effort="none",
            max_completion_tokens=4096,
        ),
        agentconfig(
            harness="codex", log_dir=str(directory / "logs"), output_dir=str(directory / "output")
        ),
        harnessadapterconfig(binary_path=args.codex_binary),
        sandboxconfig(
            base_image=args.image,
            gpu_passthrough=False,
            harness_python_path=args.harness_python,
            mounts=mounts,
        ),
        orchestratorconfig(db_path=str(directory / "logs/global_data.sqlite3")),
        resourcesconfig(idle_cpus=2, idle_memory="4g"),
    )
    pool = get_orchestrator(cfg)
    actor = Agent("swebench-codex", agconfig=cfg, harness="codex")
    skill = agskill(
        name="resolve_swebench_issue",
        prompt=(
            "Fix the reported issue in /workspace. Inspect the code, reproduce the bug, "
            "make a minimal patch, and run relevant tests. "
            f"Use {args.test_python} for repository tests. "
            "Edit with Python or sed if apply_patch is unavailable. "
            "Leave changes in the working tree without committing. "
            "Finish with a brief report of changes and actual test results."
        ),
        input_schema=agdata(issue=agrawstring),
        output_schema=agdata(report=agrawstring),
    )
    status = "failed"
    result = None
    pool.data_logger.record_event(
        "workload_started", {"task": task["instance_id"], "model": args.model}, flush=True
    )
    try:
        prepare_workspace(actor, task)
        # Dataset rows can contain CRLF, which the terminal harness rejects.
        issue = task["problem_statement"].replace("\r\n", "\n").replace("\r", "\n")
        result = actor.run(skill, agdata(issue=issue), max_steps=50)
        report = wait_for_report(actor, result, args.timeout, directory / "agent-result.json")
        patch = extract_patch(actor)
        (directory / "prediction.patch").write_text(patch)
        prediction = {
            "instance_id": task["instance_id"],
            "model_name_or_path": args.model,
            "model_patch": patch,
        }
        (directory / "prediction.jsonl").write_text(json.dumps(prediction) + "\n")
        print(report["report"])
        print(f"patch: {directory / 'prediction.patch'}")
        status = "completed"
    finally:
        try:
            if result is not None and result.is_pending():
                actor.cancel(result)
                result.wait(timeout=60)
        finally:
            close_sandboxes([actor])
            pool.data_logger.record_event("done", {"status": status}, flush=True)
        print(f"status: {status}; artifacts: {directory}")


if __name__ == "__main__":
    run_example(main)
