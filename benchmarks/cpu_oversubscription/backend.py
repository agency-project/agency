"""Docker confinement around the existing Agency Codex gateway and SWE-bench adapter."""

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .config import safe_json
from .linux import CGROUP, command, join_cgroup, processes, verify_background, verify_cgroup

LABEL = "agency.cpu-experiment"


class DockerEnvironment:
    def __init__(self, name):
        self.name = name

    def exec(self, script):
        result = subprocess.run(
            ["docker", "exec", "-w", "/workspace", self.name, "sh", "-c", script],
            capture_output=True,
            timeout=30,
        )
        return result.stdout.decode(errors="replace"), result.returncode

    def write_file(self, path, content):
        command(
            [
                "docker",
                "exec",
                "-i",
                self.name,
                "sh",
                "-c",
                f"mkdir -p {shlex.quote(str(Path(path).parent))}; cat > {shlex.quote(path)}",
            ],
            input=content.encode(),
        )


def container_cgroup(name):
    pid = int(command(["docker", "inspect", "--format", "{{.State.Pid}}", name]).strip())
    relative = (Path("/proc") / str(pid) / "cgroup").read_text().strip().split("0::", 1)[1]
    path = Path("/sys/fs/cgroup") / relative.lstrip("/")
    if not path.is_relative_to(CGROUP):
        raise RuntimeError("Docker container escaped the experiment cgroup")
    return path


def prepare_container(name, task, cfg, cpu, run_id):
    command(
        [
            "docker",
            "create",
            "--name",
            name,
            "--label",
            f"{LABEL}={run_id}",
            "--init",
            "--cgroup-parent",
            "agencycpu.slice",
            "--cpuset-cpus",
            str(cpu),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "0:0",
            "--add-host",
            "host.docker.internal:host-gateway",
            "--network",
            "bridge",
            "--mount",
            f"type=bind,src={cfg['codex_binary']},dst=/opt/agency-codex/codex,readonly",
            "--entrypoint",
            "/bin/sleep",
            task["image"],
            "infinity",
        ]
    )
    command(["docker", "start", name])
    path = container_cgroup(name)
    verify_cgroup(path, [cpu])
    version = command(["docker", "exec", name, "/opt/agency-codex/codex", "--version"]).strip()
    if version != "codex-cli " + cfg["codex_version"]:
        raise RuntimeError("Linux Codex binary cannot execute inside the prepared image")
    # Official instance images already contain the task's dependency environment.
    # No repository/data/credentials from the host are mounted.
    command(
        [
            "docker",
            "exec",
            name,
            "sh",
            "-c",
            "test ! -e /workspace && ln -s /testbed /workspace && "
            'test "$(git -C /testbed rev-parse HEAD)" = '
            + shlex.quote(task["base_commit"])
            + " && git -C /testbed diff --quiet && git -C /testbed diff --cached --quiet"
            " && test -x /opt/miniconda3/envs/testbed/bin/python",
        ]
    )
    return str(path)


def event(directory, attempt_id, kind, **fields):
    value = {"attempt_id": attempt_id, "kind": kind, "time": time.monotonic(), **fields}
    with (directory / "journal.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return value


def worker(directory, slot):
    """Runs in a private host cgroup; secrets arrive over an anonymous stdin pipe."""
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text())
    cfg = plan["config"]
    layout = plan["layout"]
    join_cgroup(Path(plan["worker_cgroups"][slot]), layout["assignments"][slot])
    secret = json.loads(sys.stdin.readline())
    os.environ[cfg["api_key_env"]] = secret["key"]
    os.environ[cfg["base_url_env"]] = secret["url"]
    del secret
    # Import the existing execution stack only after confinement is established.
    from benchmarks.tool_annotation_effect.codex import CodexGateway, codex_files
    from benchmarks.tool_annotation_effect.adapters import SweBenchAdapter
    from benchmarks.tool_annotation_effect.common import native

    safe_json(directory / f"ready-{slot}.json", {"ready": True})
    while not (directory / "start.json").exists():
        time.sleep(0.01)
    timing = json.loads((directory / "start.json").read_text())
    deadline = timing["deadline"]
    model = {
        "id": cfg["model"],
        "model": cfg["model"],
        "provider": "openai",
        "base_url_env": cfg["base_url_env"],
        "api_key_env": cfg["api_key_env"],
        "settings": cfg["model_settings"],
    }
    for index, task in enumerate(plan["manifest"]["agents"][slot]):
        if time.monotonic() >= deadline:
            break
        attempt_id = f"a{slot}-t{index}"
        target = directory / attempt_id
        name = plan["containers"][attempt_id]
        environment = DockerEnvironment(name)
        remaining = deadline - time.monotonic()
        experiment_config = {
            "context_limit": cfg["context_limit"],
            "codex_version": cfg["codex_version"],
            "codex_gateway_host": cfg["gateway_host"],
            "budgets": {"max_steps": cfg["max_steps"], "timeout_s": remaining},
        }
        gateway = CodexGateway(
            model,
            experiment_config,
            {"arm": "baseline", "trial_id": plan["run_id"] + "-" + attempt_id},
            target,
        )
        # Also scrub the ephemeral credential from subsequent exception messages.
        os.environ["AGENCY_EXPERIMENT_GATEWAY_TOKEN"] = gateway.token
        # The existing gateway retries are additionally bounded by the shared deadline.
        gateway.client.deadline = deadline
        with gateway.serve() as url:
            prompt = (
                "Fix the issue in /workspace. Use /opt/miniconda3/envs/testbed/bin/python "
                "for tests. Leave changes uncommitted. Do not launch background jobs.\n\n"
                + task["problem_statement"]
            )
            files, script = codex_files(gateway, url, prompt, experiment_config)
            for path, content in files.items():
                environment.write_file(path, content)
            # Persist a write-ahead start before launching. A boundary interruption is
            # conservatively counted as an interrupted attempt, never as a completion.
            if time.monotonic() >= deadline:
                break
            event(directory / f"worker-{slot}", attempt_id, "started")
            # Stream logs to the host rather than relying on container teardown/export.
            script = script.replace(
                " > /opt/agency-codex-state/stdout.jsonl 2> /opt/agency-codex-state/stderr.txt",
                " 2>&1",
            )
            process = subprocess.Popen(
                ["docker", "exec", "-w", "/workspace", name, "sh", "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            with (target / "codex.log").open("a") as output:
                for line in iter(process.stdout.readline, b""):
                    output.write(
                        native("annotations").redact(
                            line.decode(errors="replace"), (gateway.token,)
                        )
                    )
                    output.flush()
            code = process.wait()
            if time.monotonic() >= deadline:
                break
            patch = SweBenchAdapter().extract_patch(environment)
            # extract_patch operates on /workspace; preserve the binary diff verbatim.
            (target / "prediction.patch").write_text(
                native("annotations").redact(patch, (gateway.token,))
            )
            SweBenchAdapter().export_prediction(
                {"id": task["instance_id"]},
                native("annotations").redact(patch, (gateway.token,)),
                cfg["model"],
                target / "prediction.jsonl",
            )
            safe_json(
                target / "gateway-result.json", {"error": gateway.error, "calls": gateway.calls}
            )
            # Kill lingering container descendants before assigning the next task.
            command(["docker", "stop", "--time", "0", name], timeout=10)
            if time.monotonic() < deadline:
                event(directory / f"worker-{slot}", attempt_id, "finished", returncode=code)
    safe_json(directory / f"done-{slot}.json", {"done": True, "time": time.monotonic()})


def verify_running(plan):
    verify_cgroup(CGROUP, plan["layout"]["agent_cores"])
    verify_background(plan["layout"])
    for slot, path in enumerate(plan["worker_cgroups"]):
        verify_cgroup(Path(path), [plan["layout"]["assignments"][slot]])
    for attempt_id, raw in plan["container_cgroups"].items():
        path = Path(raw)
        if path.exists():
            cpu = plan["layout"]["assignments"][int(attempt_id[1])]
            try:
                verify_cgroup(path, [cpu])
            except FileNotFoundError:
                if path.exists():
                    raise
    for pid in processes():
        try:
            if (
                os.sched_getscheduler(pid) != os.SCHED_OTHER
                or os.getpriority(os.PRIO_PROCESS, pid) != 0
            ):
                raise RuntimeError("Experiment process changed its default scheduling policy")
        except ProcessLookupError:
            continue
