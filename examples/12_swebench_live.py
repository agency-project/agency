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
import sys
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
from agency.observability.agwebui import _flush_loop
from agency.orchestrator import get_orchestrator

from smoke_test_models import load_env_file


def pace_provider_requests(interval):
    """Share a request-start budget across this example's two model clients."""
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
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--agents", type=int, choices=(1, 2), default=1)
    parser.add_argument("--request-interval", type=float, default=0)
    args = parser.parse_args()
    load_env_file(args.env_file)
    if args.request_interval > 0:
        pace_provider_requests(args.request_interval)
    os.environ.setdefault("AGENCY_PROFILE", "0")
    task = json.loads(args.task_file.read_text())
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    # Agency mounts its package root. Supply only runtime code here, so the
    # host .env and evaluator's gold patches cannot enter that mount.
    runtime = directory / "runtime"
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(
        source / "agency", runtime / "agency", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(source / "pyproject.toml", runtime / "pyproject.toml")
    cfg = agconfig(
        llmconfig(
            provider="openai",
            model=args.model,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ["OPENAI_API_KEY"],
            reasoning_effort="none",
            context_limit=196000,
            max_completion_tokens=4096,
            idle_timeout=180,
            stream_timeout=180,
        ),
        agentconfig(
            harness="codex", log_dir=str(directory / "logs"), output_dir=str(directory / "output")
        ),
        harnessadapterconfig(binary_path=args.codex_binary),
        sandboxconfig(
            backend="docker",
            base_image=args.image,
            gpu_passthrough=False,
            harness_python_path="/opt/agency-harness-venv/bin/python",
            mounts={"_agency_package": (str(runtime), "/opt/agency_pkg", "ro")},
        ),
        orchestratorconfig(
            max_concurrent_engines=args.agents, db_path=str(directory / "logs/global_data.sqlite3")
        ),
        resourcesconfig(idle_cpus=2, idle_memory="4g"),
    )
    pool = get_orchestrator(cfg, default_db_path=directory / "logs/global_data.sqlite3")
    actor = Agent("swebench-codex-luna", agconfig=cfg, harness="codex")
    tester = (
        Agent("swebench-regression-luna", agconfig=cfg, harness="codex")
        if args.agents == 2
        else None
    )
    actors = [actor] if tester is None else [actor, tester]
    stop = threading.Event()
    threading.Thread(target=_flush_loop, args=(stop,), daemon=True).start()
    status = "failed"
    result = None
    pending = []
    pool.data_logger.record_event(
        "workload_started",
        {
            "task": f"SWE-bench Lite · {task['instance_id']}",
            "model": args.model,
            "harness": "codex",
            "dataset": "princeton-nlp/SWE-bench_Lite",
        },
        flush=True,
    )
    try:
        for member in actors:
            prepare_workspace(member, task)
        skill = agskill(
            name="resolve_swebench_issue",
            prompt=(
                "Fix the reported issue in the /workspace repository. "
                "Inspect the code, reproduce the bug, make the patch, and run relevant tests. "
                "Use /opt/miniconda3/envs/testbed/bin/python for the repository tests. "
                "Use grep/find inside /workspace (rg is unavailable), and inspect focused code excerpts. "
                "The shell apply_patch command is unavailable; edit using a Python script or sed. "
                + (
                    "Change source code only; another agent is writing the regression test. "
                    if tester
                    else ""
                )
                + "Leave changes in the working tree. Do not commit. Finish with a brief "
                "report of changes and actual verification results."
            ),
            input_schema=agdata(issue=agrawstring),
            output_schema=agdata(report=agrawstring),
        )
        result = actor.run(skill, agdata(issue=task["problem_statement"]), max_steps=50)
        pending.append((actor, result))
        if tester is not None:
            test_skill = agskill(
                name="write_swebench_regression",
                prompt=(
                    "Independently reproduce the reported issue in /workspace. "
                    "Add a focused regression test to the existing test_requests.py file. "
                    "Do not modify source code. Another agent is implementing the fix. "
                    "Run your test against the unchanged source and report the actual failure. "
                    "Use /opt/miniconda3/envs/testbed/bin/python for tests. Do not commit. "
                    "Use grep/find inside /workspace (rg is unavailable), and focused code excerpts. "
                    "The shell apply_patch command is unavailable; edit using a Python script. "
                    "For a mocked Response, mark its empty content consumed and provide a no-op close. "
                    "Finish with the test selector, failure evidence, and edge cases to review."
                ),
                input_schema=agdata(issue=agrawstring),
                output_schema=agdata(report=agrawstring),
            )
            test_result = tester.run(
                test_skill, agdata(issue=task["problem_statement"]), max_steps=50
            )
            pending.append((tester, test_result))
        wait_for_report(actor, result, args.timeout, directory / "agent-result.json")
        if tester is not None:
            test_report = wait_for_report(
                tester, test_result, args.timeout, directory / "tester-result.json"
            )
            patch = extract_patch(tester, " -- test_requests.py")
            if not patch.strip():
                retry = tester.run(
                    test_skill,
                    agdata(
                        issue=(
                            task["problem_statement"]
                            + "\n\nYour previous turn produced no test patch. Use Python to edit "
                            "test_requests.py directly; do not rely on a shell apply_patch command. "
                            "Complete the edit and execute the regression on unchanged source."
                        )
                    ),
                    max_steps=30,
                )
                pending.append((tester, retry))
                test_report = wait_for_report(
                    tester, retry, args.timeout, directory / "tester-followup-result.json"
                )
                patch = extract_patch(tester, " -- test_requests.py")
            if not patch.strip():
                raise RuntimeError("The regression agent did not produce its test patch")
            (directory / "regression.patch").write_text(patch)
            actor.sandbox.write_file("/tmp/regression.patch", patch)
            output, code = actor.sandbox.exec(
                "git apply /tmp/regression.patch", workdir="/workspace"
            )
            if code:
                raise RuntimeError(f"Regression patch integration failed: {output}")
            tester.data_logger.record_event(
                "handoff",
                {
                    "to": str(actor.agname),
                    "task": "Regression test patch applied to implementer's workspace",
                    "artifact": "regression.patch",
                },
                flush=True,
            )
            review = actor.run(
                skill,
                agdata(
                    issue=(
                        "The regression agent's test has now been applied to your workspace. "
                        "Verify the combined fix and regression test, review its edge cases, "
                        "and correct source code if needed. Leave the test intact. "
                        "Report actual commands and results.\n\n" + json.dumps(test_report)
                    )
                ),
                max_steps=30,
            )
            pending.append((actor, review))
            wait_for_report(actor, review, args.timeout, directory / "integration-result.json")
        status = "completed"
    finally:
        active_error = sys.exc_info()[0] is not None
        for member, handle in pending:
            if handle.is_pending():
                member.cancel(handle)
                try:
                    handle.wait(timeout=60)
                except Exception as error:
                    print(
                        f"Cancellation cleanup failed for {member.agname}: {error}", file=sys.stderr
                    )
        collection_error = None
        try:
            if actor.sandbox is not None:
                patch = extract_patch(actor)
                (directory / "prediction.patch").write_text(patch)
                prediction = {
                    "instance_id": task["instance_id"],
                    "model_name_or_path": args.model,
                    "model_patch": patch,
                }
                (directory / "prediction.jsonl").write_text(json.dumps(prediction) + "\n")
        except Exception as error:
            collection_error = str(error)
            status = "failed"
            (directory / "collection-error.json").write_text(
                json.dumps({"error": collection_error})
            )
        finally:
            try:
                for member in actors:
                    if member.sandbox is not None:
                        member.sandbox.destroy()
            finally:
                pool.data_logger.record_event(
                    "done", {"status": status}, update_latest_snapshot=True, flush=True
                )
                for member in actors:
                    member.data_logger.flush()
                stop.set()
                pool.shutdown()
        print(
            json.dumps(
                {
                    "task": task["instance_id"],
                    "model": args.model,
                    "harness": "codex",
                    "agents": args.agents,
                    "status": status,
                    "run_dir": str(directory),
                }
            )
        )
        if collection_error and not active_error:
            raise RuntimeError(collection_error)


if __name__ == "__main__":
    main()
