"""One isolated EC2 execution; standalone imports Agency only to render the common prompt."""

from pathlib import Path
import hashlib
import json
import os
import secrets
import shlex
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["AGENCY_PROFILE"] = "0"
from recording_proxy import RecordingProxy, summarize
from settings import credential as read_credential
from settings import validate_host

PYTHON = "/opt/agency-harness-venv/bin/python"
MODEL = "claude-haiku-5-5"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def docker(*args, check=True, timeout=180):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"Docker {args[0]} failed: {result.stderr[-1800:]}")
    return result.stdout


def credential():
    return read_credential(ROOT)


def prompt_for(task):
    from agency import agdata, agrawstring, agskill
    from agency.harness.daemon import _render_attempt_prompt
    from agency.harness.protocol import HarnessAttemptRequest, PromptPayload

    skill = agskill(
        name="repair_issue",
        prompt=(
            "Fix the reported issue in /workspace. Inspect the code, reproduce the bug, "
            "make a minimal source patch, and run relevant local tests. "
            "Use /opt/miniconda3/envs/testbed/bin/python for project code and tests, from /workspace. "
            "Do not fetch upstream fixes or reference solutions. "
            "Leave changes in the working tree without committing. "
            "Finish with a brief report of your changes and actual test results."
        ),
        input_schema=agdata(issue=agrawstring),
        output_schema=agdata(report=agrawstring),
        max_output_schema_retries=0,
    )
    value = agdata(issue=task["problem_statement"].replace("\r\n", "\n"))
    prompt = "[Agency run]\n" + _render_attempt_prompt(
        HarnessAttemptRequest(
            prompt=PromptPayload(skill._build_prompt(), skill.build_user_content(value)),
            harness="claude_code",
        )
    )
    return skill, value, prompt


def run(job):
    configuration = json.loads((ROOT / "configuration.json").read_text())
    actual_host, _ = validate_host(ROOT)
    assert actual_host == configuration["execution_host"], "Prepared host identity changed"
    assert (
        hashlib.sha256((ROOT / "task_manifest.json").read_bytes()).hexdigest()
        == configuration["manifest_sha256"]
    )
    task = next(
        t
        for t in json.loads((ROOT / "public_tasks.json").read_text())
        if t["instance_id"] == job["instance_id"]
    )
    directory = (
        ROOT
        / ("diagnostics" if job.get("diagnostic") else "attempts")
        / job["run_id"]
        / str(job["attempt"])
    )
    directory.mkdir(parents=True, exist_ok=False)
    control = directory / "control"
    write(control / "config.json", {"gate_timeout_s": 120})
    arm = job["arm"]
    os.environ["AGENCY_RUNS_ROOT"] = str(directory / "state")
    token = secrets.token_urlsafe(32)
    key = credential()
    gateway = configuration["docker_gateway"]
    os.environ["NO_PROXY"] = gateway + ",127.0.0.1,localhost"
    image = configuration["images"][task["instance_id"]]["image_id"]
    proxy = RecordingProxy(directory / "provider", key, gateway, token)
    write(control / "recorder.json", {"base_url": proxy.url, "proxy_token": token})
    os.environ["ANTHROPIC_BASE_URL"] = proxy.url
    os.environ["ANTHROPIC_API_KEY"] = token
    mounts = {
        "python_runtime": (configuration["runtime"] + "/agency-python", "/opt/agency-python", "ro"),
        "harness_runtime": (
            configuration["runtime"] + "/agency-harness-venv",
            "/opt/agency-harness-venv",
            "ro",
        ),
        "binary": (str(ROOT / "binary"), "/study-binary", "ro"),
        "launcher": (str(ROOT / "launcher"), str(ROOT / "launcher"), "ro"),
        "driver": (str(ROOT / "driver"), "/smoke-driver", "ro"),
        "capture": (str(ROOT / "capture"), "/study-capture", "ro"),
        "support": (str(ROOT / "support"), "/study-support", "ro"),
        "control": (str(control), "/study-control", "rw"),
        "output": (str(directory), "/smoke-output", "rw"),
    }
    container = f"ni-haiku-{job['run_id']}-{job['attempt']}"
    actor = None
    patch = None
    started = time.time()
    result = {
        **job,
        "status": "infrastructure_failed",
        "start_unix_s": started,
        "model": MODEL,
        "timeout": False,
        "exit_status": None,
        "directory": str(directory.relative_to(ROOT)),
    }
    try:
        skill, value, prompt = prompt_for(task)
        assert (
            skill.policy.tool_hooks is None
            and skill.policy.syscall_hooks is None
            and not skill.policy.default_to_deny
        )
        write(
            directory / "policy.json",
            {"tool_hooks": None, "syscall_hooks": None, "default_to_deny": False},
        )
        (directory / "submitted-prompt.txt").write_text(prompt)
        if arm == "standalone":
            args = [
                "run",
                "-d",
                "--init",
                "--cap-add=SYS_PTRACE",
                "--name",
                container,
                "--network=bridge",
                "--cpus=2",
                "--memory=4g",
                "--memory-swap=4g",
                "--cpu-shares=1024",
            ]
            for src, dest, mode in mounts.values():
                args += [
                    "--mount",
                    f"type=bind,src={src},dst={dest}" + (",readonly" if mode == "ro" else ""),
                ]
            docker(*args, image, "tail", "-f", "/dev/null")
        else:
            from agency import Agent
            from agency.configs.agconfig import (
                agconfig,
                agentconfig,
                harnessadapterconfig,
                llmconfig,
                orchestratorconfig,
                resourcesconfig,
                sandboxconfig,
                schemaconfig,
                toolconfig,
            )

            cfg = agconfig(
                llmconfig(
                    provider="anthropic",
                    model=MODEL,
                    api_key=token,
                    base_url=proxy.url,
                    context_limit=1000000,
                    sdk_max_retries=0,
                    max_retries=0,
                    unusable_response_retries=0,
                ),
                agentconfig(
                    harness="claude_code",
                    annotation_arm="baseline",
                    log_dir=str(directory / "logs"),
                    output_dir=str(directory / "output"),
                ),
                harnessadapterconfig(
                    binary_path=str(ROOT / "launcher/claude"), allow_subagents=False
                ),
                sandboxconfig(
                    backend="docker",
                    base_image=image,
                    gpu_passthrough=False,
                    harness_python_path="/study-support/harness-python",
                    mounts=mounts,
                ),
                resourcesconfig(idle_cpus=2, idle_memory="4g"),
                orchestratorconfig(db_path=str(directory / "logs/global.sqlite3")),
                schemaconfig(input_offload_chars=1000000000, offload_context_fraction=1.0),
                toolconfig(output_offload_chars=1000000000),
            )
            write(
                directory / "agency_configuration.json",
                {
                    name: getattr(cfg, name).safe_snapshot()
                    for name in ("llm", "agent", "harness_adapter", "schema", "tool", "resources")
                },
            )
            actor = Agent(job["run_id"], agconfig=cfg, harness="claude_code")
            sandbox = actor._ensure_sandbox()
            sandbox._name = container
            sandbox._backend._name = container
            output, code = sandbox.exec("true", workdir="/")
            if code:
                raise RuntimeError(output)
            container = sandbox._backend._container_name()
        docker(
            "update", "--cpus=2", "--memory=4g", "--memory-swap=4g", "--cpu-shares=1024", container
        )
        initial = json.loads(docker("inspect", container))[0]
        limits = {
            k: initial["HostConfig"].get(k)
            for k in ("NanoCpus", "Memory", "MemorySwap", "CpuShares", "NetworkMode", "CapAdd")
        }
        write(directory / "resources.json", limits)
        assert (
            limits["NanoCpus"] == 2000000000
            and limits["Memory"] == limits["MemorySwap"] == 4294967296
        )
        write(
            directory / "environment.json",
            {"image_id": initial["Image"], "base_commit": task["base_commit"]},
        )
        prepare = " && ".join(
            [
                "git -C /testbed checkout --detach " + shlex.quote(task["base_commit"]),
                "git -C /testbed reset --hard " + shlex.quote(task["base_commit"]),
                "if [ -d /workspace ] && [ ! -L /workspace ]; then rmdir /workspace; fi",
                "if [ ! -e /workspace ]; then ln -s /testbed /workspace; fi",
                "git config --global --add safe.directory /testbed",
                "git config --global --add safe.directory /workspace",
                "git -C /testbed rev-parse HEAD",
            ]
        )
        prepared = docker("exec", container, "/bin/bash", "-c", prepare)
        (directory / "checkout.txt").write_text(prepared)
        assert prepared.strip().splitlines()[-1] == task["base_commit"]
        # Fresh immutable images retain compiled/installed ignored files required by the benchmark.
        # Deleting ignored files would destroy matplotlib's extension modules.
        module = task["repo"].split("/")[-1].replace("-", "_")
        if module == "sphinx":
            module = "sphinx"
        baseline = docker(
            "exec",
            "-w",
            "/testbed",
            container,
            "/opt/miniconda3/envs/testbed/bin/python",
            "-c",
            f"import {module}; print({module}.__file__)",
        )
        (directory / "baseline-import.txt").write_text(baseline)
        if arm == "agency":
            daemon_import = docker(
                "exec",
                "-e",
                "PYTHONPATH=/opt/agency_pkg",
                container,
                "/study-support/harness-python",
                "-c",
                "import agency; print(agency.__file__)",
            )
            (directory / "daemon-import.txt").write_text(daemon_import)
        result["status"] = "running"
        write(directory / "state.json", result)
        invocation = time.monotonic()

        def export_patch():
            return docker(
                "exec",
                "-w",
                "/workspace",
                container,
                "/bin/bash",
                "-c",
                "git add -N . && git diff --binary HEAD",
            )

        def release():
            nonlocal patch
            if (control / "native-stop.json").exists() and not (control / "release").exists():
                patch = export_patch()
                (control / "release").write_text(str(time.monotonic()))

        if arm == "standalone":
            settings = dict(
                base_url=proxy.url,
                proxy_token=token,
                binary="/study-binary/2.1.299",
                prompt=prompt,
                timeout=configuration["timeout_s"],
                hook_python=PYTHON,
                stop_gate=True,
                gate_timeout_s=120,
            )
            with (directory / "driver-output.txt").open("w") as log:
                proc = subprocess.Popen(
                    [
                        "docker",
                        "exec",
                        "-i",
                        container,
                        PYTHON,
                        "/smoke-driver/standalone_driver.py",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                proc.stdin.write(json.dumps(settings))
                proc.stdin.close()
                while proc.poll() is None:
                    release()
                    if time.monotonic() - invocation > configuration["timeout_s"] + 60:
                        proc.kill()
                        proc.wait()
                        raise TimeoutError("Standalone controller timed out")
                    time.sleep(0.1)
            result["exit_status"] = proc.returncode
            raw = json.loads((directory / "result.json").read_text())
            result["status"] = raw["status"]
            result["timeout"] = raw["status"] == "timeout"
            patch = export_patch()
        else:
            future = actor.run(skill, value, max_steps=None)
            while True:
                release()
                try:
                    future.wait(timeout=0.1)
                    break
                except TimeoutError:
                    if time.monotonic() - invocation > configuration["timeout_s"]:
                        actor.cancel(future)
                        raise TimeoutError("Agency task timed out")
            data = future.to_dict()
            result["agent_result"] = data
            result["status"] = "failed" if "error" in data else "completed"
            result["exit_status"] = 1 if "error" in data else 0
        if (control / "native-stop.json").exists():
            marker = json.loads((control / "native-stop.json").read_text())
            result["native_completion_wall_s"] = marker["monotonic_s"] - invocation
    except TimeoutError as exc:
        result.update(status="timeout", timeout=True, error=str(exc), exit_status=124)
        if patch is None:
            try:
                patch = docker(
                    "exec",
                    "-w",
                    "/workspace",
                    container,
                    "/bin/bash",
                    "-c",
                    "git add -N . && git diff --binary HEAD",
                    timeout=10,
                )
            except Exception as capture_error:
                result["patch_capture_error"] = str(capture_error)
    except Exception as exc:
        result.update(
            status="infrastructure_failed",
            error=str(exc).replace(key, "[REDACTED]").replace(token, "[REDACTED]"),
        )
        (directory / "exception.txt").write_text(
            traceback.format_exc().replace(key, "[REDACTED]").replace(token, "[REDACTED]")
        )
    finally:
        if patch is not None:
            (directory / "prediction.patch").write_text(patch)
            result["patch_sha256"] = hashlib.sha256(patch.encode()).hexdigest()
        if actor is not None and actor.sandbox is not None:
            try:
                actor.sandbox.destroy()
            except Exception as exc:
                result["cleanup_error"] = str(exc)
        else:
            docker("rm", "-f", container, check=False, timeout=60)
        proxy.close()
        result["provider"] = summarize(directory / "provider")
        if result["status"] == "failed" and not result["provider"]["model_requests"]:
            result["status"] = "infrastructure_failed"
            result["failure_category"] = "native_harness_startup_before_inference"
        elif result["status"] != "completed":
            result["failure_category"] = (
                "provider"
                if any(
                    c.get("error") or c.get("stream_errors") or c.get("status", 0) >= 400
                    for c in result["provider"]["calls"]
                )
                else "agent_or_timeout"
            )
        result["end_unix_s"] = time.time()
        result["wall_s"] = time.time() - started
        write(directory / "summary.json", result)
    print(
        json.dumps({k: result.get(k) for k in ("run_id", "status", "wall_s", "error")}), flush=True
    )


if __name__ == "__main__":
    run(json.loads(sys.argv[1]))
