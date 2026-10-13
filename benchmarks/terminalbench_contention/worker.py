"""One agent: one Terminal-Bench task in one fresh Agency Codex sandbox.

Run as its own process (one per task) by the runner:

    python -m benchmarks.terminalbench_contention.worker <agent-dir>/job.json

Lifecycle, each step leaving a durable marker in the agent directory:

  setup      create the Agent and its container, verify isolation  -> ready.json
  gate       wait for the runner's start.json (release + deadlines)
  execute    run the Codex harness through Agency                  -> agent-finished.json
  hand-off   wait for the runner to collect artifacts              (collected.json)
  cleanup    destroy the container, flush traces                   -> task.json (final)

The model path is Agency's own: Codex -> Agency proxy -> OpenAI-compatible
backend -> LiteLLM. ``api_instrument`` only observes that path and applies the
shared admission budget; nothing is bypassed.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from .api_budget import BudgetPolicy, SharedBudget
from .sampler import read_cgroup, scan_process_io, scan_threads
from .topology import format_cpu_list, parse_cpu_list
from .util import atomic_json, clock_anchor, load_env_file, read_json, read_json_or, redact

CGROUP_ROOT = Path("/sys/fs/cgroup")
ROOT_USERS = ("", "root", "0", "0:0", "root:root")

PROMPT = (
    "You are working autonomously in a Linux container. Complete the task below. "
    "The task's files are in {workdir}, which is the task's working directory; do your "
    "work there. When you are done, submit a short report of what you did."
)

AFFINITY_PROBE = (
    "import os,json;"
    "before=sorted(os.sched_getaffinity(0));"
    "os.sched_setaffinity(0,set(range(os.cpu_count())));"
    "print(json.dumps(dict(initial=before,after_widening=sorted(os.sched_getaffinity(0)))))"
)


class Interrupted(Exception):
    """The runner (or operator) asked this worker to stop."""


class ContainerTrace(threading.Thread):
    """High-rate kernel counters for this agent's container.

    One row per tick: [t_ns, cpu_usage_usec, cpu_user_usec, cpu_system_usec,
    cpu_stall_usec, memory_stall_usec, io_stall_usec, memory_bytes]. All values
    are read from the container's cgroup files (measured, cumulative except
    memory). Rows stop when the container is stopped and its cgroup disappears.
    """

    COLUMNS = (
        "t_ns",
        "cpu_usage_usec",
        "cpu_user_usec",
        "cpu_system_usec",
        "cpu_stall_usec",
        "memory_stall_usec",
        "io_stall_usec",
        "memory_bytes",
    )

    def __init__(self, cgroup: Path, output: Path, interval: float) -> None:
        super().__init__(name="tb-container-trace", daemon=True)
        self.cgroup, self.output, self.interval = Path(cgroup), Path(output), interval
        self._halt = threading.Event()
        self.rows = 0

    def read(self) -> "list | None":
        try:
            cpu = dict(
                line.split()
                for line in (self.cgroup / "cpu.stat").read_text().splitlines()
                if len(line.split()) == 2
            )
            row = [
                time.monotonic_ns(),
                int(cpu["usage_usec"]),
                int(cpu["user_usec"]),
                int(cpu["system_usec"]),
            ]
            for resource in ("cpu", "memory", "io"):
                text = (self.cgroup / f"{resource}.pressure").read_text()
                row.append(int(text.split("total=", 1)[1].split()[0]))
            row.append(int((self.cgroup / "memory.current").read_text()))
            return row
        except (OSError, KeyError, ValueError, IndexError):
            return None

    def run(self) -> None:
        with self.output.open("a", buffering=1) as stream:
            stream.write(json.dumps(dict(columns=self.COLUMNS)) + "\n")
            while not self._halt.is_set():
                row = self.read()
                if row is not None:
                    stream.write(json.dumps(row) + "\n")
                    self.rows += 1
                self._halt.wait(self.interval)

    def halt(self) -> None:
        self._halt.set()
        if self.is_alive():
            self.join(timeout=5)


def cgroup_of(pid) -> "Path | None":
    try:
        text = Path(f"/proc/{pid}/cgroup").read_text().strip()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("0::"):
            return CGROUP_ROOT / line.split("::", 1)[1].lstrip("/")
    return None


def container_scope(cgroup: "Path | None") -> "Path | None":
    """The libpod scope holding the whole container (payload and conmon)."""
    if cgroup is None:
        return None
    for candidate in (cgroup, *cgroup.parents):
        if candidate.name.startswith("libpod-") and candidate.name.endswith(".scope"):
            return candidate
    return cgroup


def counters(path: "Path | None") -> "dict | None":
    if path is None:
        return None
    record = read_cgroup(path)
    if record is None:
        return None
    record["threads"] = scan_threads(path)
    if "io" not in record:
        record["io"] = scan_process_io(path)
    for name in ("cpu.max", "cpuset.cpus.effective", "memory.max", "memory.swap.max", "pids.max"):
        try:
            record[name] = (path / name).read_text().strip()
        except OSError:
            record[name] = None  # unavailable is never reported as zero
    return record


def verify_isolation(job, sandbox, inspection, container_cgroup, directory) -> dict:
    """Check that this container is confined, unthrottled and writes only its own state."""
    expected = set(job["cpus"]) if job.get("cpus") else None
    host = inspection["HostConfig"]
    problems = []
    report = dict(
        container=sandbox._name,
        image_id=inspection.get("Image"),
        cpuset_requested=host.get("CpusetCpus"),
        nano_cpus=host.get("NanoCpus"),
        cpu_quota=host.get("CpuQuota"),
        cpu_shares=host.get("CpuShares"),
        memory_limit=host.get("Memory"),
        memory_swap=host.get("MemorySwap"),
        pids_limit=host.get("PidsLimit"),
        cgroup_parent=host.get("CgroupParent"),
        network_mode=host.get("NetworkMode"),
        privileged=host.get("Privileged"),
        user=inspection["Config"].get("User") or "",
        container_cgroup=str(container_cgroup) if container_cgroup else None,
        host_affinity=sorted(os.sched_getaffinity(0)),
    )
    if host.get("Privileged"):
        problems.append("container is privileged")
    if host.get("NanoCpus") or host.get("CpuQuota"):
        problems.append("container has a CPU quota")
    if report["user"] not in ROOT_USERS:
        problems.append(f"image default user is {report['user']!r}, expected root")
    expected_image = job["task"].get("environment_image_id")
    if expected_image and inspection.get("Image") != expected_image:
        problems.append("container image ID differs from the prepared image")
    for name in ("cpu.max", "cpuset.cpus.effective", "memory.max", "memory.swap.max"):
        try:
            report[name] = (container_cgroup / name).read_text().strip()
        except (OSError, TypeError):
            report[name] = None
    if report["cpu.max"] is not None and not report["cpu.max"].startswith("max"):
        problems.append(f"cgroup cpu.max is {report['cpu.max']!r}: a CPU quota is in force")
    if expected is not None:
        if host.get("CpusetCpus") != format_cpu_list(expected):
            problems.append("container cpuset differs from the experimental pool")
        effective = report["cpuset.cpus.effective"]
        if effective is None or set(parse_cpu_list(effective)) != expected:
            problems.append(f"effective cpuset {effective!r} is not the experimental pool")
        if not set(report["host_affinity"]) <= expected:
            problems.append("host-side agent process can run outside the experimental pool")
        python = job["sandbox"]["harness_python_container"]
        output, code = sandbox.exec(f"{python} -c {json.dumps(AFFINITY_PROBE)}")
        try:
            probe = json.loads(output.strip().splitlines()[-1]) if not code else None
        except (ValueError, IndexError):
            probe = None
        report["affinity_probe"] = probe
        if probe is None or not (
            set(probe["initial"]) == expected and set(probe["after_widening"]) == expected
        ):
            problems.append("a process inside the container can leave the experimental pool")
    # Writable host paths must belong to this agent alone.
    agent_root = str(Path(directory).resolve())
    writable, shared_writable = [], []
    for mount in inspection.get("Mounts", []):
        source = mount.get("Source", "")
        entry = dict(source=source, destination=mount.get("Destination"), rw=bool(mount.get("RW")))
        if not entry["rw"]:
            continue
        writable.append(entry)
        private = source.startswith(agent_root + os.sep) or source.startswith(
            job["sandbox"]["agency_run_tmp_prefix"]
        )
        if not private:
            shared_writable.append(source)
    report["writable_mounts"] = writable
    if shared_writable:
        problems.append(f"writable host mounts outside this agent's directory: {shared_writable}")
    report.update(problems=problems, verified=not problems)
    return report


def build_config(job, directory, api_key, base_url):
    from agency.configs.agconfig import (
        agconfig,
        agentconfig,
        harnessadapterconfig,
        llmconfig,
        orchestratorconfig,
        resourcesconfig,
        sandboxconfig,
    )

    model, sandbox, task = job["model"], job["sandbox"], job["task"]
    flags = [
        "--pull=never",
        f"--pids-limit={sandbox['pids_limit']}",
        f"--label={sandbox['run_label']}",
        f"--label={sandbox['task_label']}",
    ]
    if sandbox.get("memory_swap"):
        flags.append(f"--memory-swap={sandbox['memory_swap']}")
    if sandbox.get("cgroup_parent"):
        flags.append(f"--cgroup-parent={sandbox['cgroup_parent']}")
    if sandbox.get("network") == "none":
        flags.append("--network=none")
    return agconfig(
        llmconfig(
            provider=model["provider"],
            model=model["model"],
            reasoning_effort=model["reasoning_effort"],
            api_key=api_key,
            base_url=base_url,
            context_limit=model["context_limit"],
            max_completion_tokens=model["max_completion_tokens"],
            idle_timeout=model["idle_timeout"],
            stream_timeout=model["stream_timeout"],
        ),
        agentconfig(
            harness="codex",
            log_dir=str(directory / "logs"),
            output_dir=str(directory / "output"),
            experiment_run_id=job["run_id"],
            experiment_agent_id=job["agent_id"],
        ),
        harnessadapterconfig(binary_path=job["codex_binary"]),
        sandboxconfig(
            backend="podman",
            base_image=task["environment"]["image"],
            gpu_passthrough=False,
            cpuset_cpus=format_cpu_list(job["cpus"]) if job.get("cpus") else None,
            cpuset_mems=sandbox.get("cpuset_mems"),
            flags=flags,
            harness_python_path=sandbox["harness_python_container"],
            mounts={
                # A frozen copy of the Agency package, not the repository (which
                # holds the operator's .env), is what the sandbox can read.
                "_agency_package": (job["runtime_source"], "/opt/agency_pkg", "ro"),
                "harness_python": (sandbox["harness_python_host"], "/opt/agency-python", "ro"),
            },
        ),
        # No CPU quota: every agent competes for the same pool. Memory is a
        # safety limit equal to what the task itself declares.
        resourcesconfig(idle_cpus=None, idle_memory=sandbox["memory"]),
        orchestratorconfig(
            max_concurrent_engines=1, db_path=str(directory / "logs/global_data.sqlite3")
        ),
    )


def run(job: dict) -> int:
    # The profiler's automatic re-exec into its own scope would leave the
    # cgroup the runner placed this process in; use its manual session API.
    os.environ["AGENCY_PROFILE"] = "0"
    if job.get("tiktoken_cache"):
        os.environ["TIKTOKEN_CACHE_DIR"] = job["tiktoken_cache"]
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    task = job["task"]
    secrets: list = []
    state = dict(
        schema_version=1,
        run_id=job["run_id"],
        slot=job["slot"],
        agent_id=job["agent_id"],
        task_id=task["task_id"],
        task_digest=task["digest"],
        status="setup",
        completed=False,
        censored=False,
        pid=os.getpid(),
        clock=clock_anchor(),
        setup_start_ns=time.monotonic_ns(),
        host_cgroup=str(cgroup_of(os.getpid())),
    )

    def save():
        atomic_json(directory / "task.json", state)

    def safe(value) -> str:
        return redact(value, secrets)

    save()

    def interrupted(signum, _frame):
        raise Interrupted(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    actor = sandbox = None
    container_cgroup = None
    host_cgroup = cgroup_of(os.getpid())
    profiler_started = False
    measured_start = None
    deadline_ns = None
    telemetry = None
    trace = None
    marks: dict = {}
    exit_code = 0
    try:
        model = job["model"]
        if job.get("mock"):
            api_key, base_url = "mock-key", job["mock"]["base_url"]
        else:
            environment = {**load_env_file(job["env_file"]), **os.environ}
            api_key = environment.get(model["api_key_env"])
            base_url = environment.get(model["base_url_env"])
            if not api_key or not base_url:
                raise RuntimeError(
                    f"{model['api_key_env']} / {model['base_url_env']} are not configured"
                )
            secrets.append(api_key)

        from agency import Agent, agdata, agrawstring, agskill
        from agency.llm.openai import _OpenAICompatibleBackend
        from agency.observability.profiler import agprof
        from agency.orchestrator import peek_orchestrator

        from . import api_instrument

        def task_deadline():
            return deadline_ns / 1e9 if deadline_ns is not None else None

        budget = SharedBudget(
            job["api"]["directory"],
            BudgetPolicy.from_dict(job["api"]["policy"]),
            deadline=task_deadline,
        )
        telemetry = api_instrument.ApiTelemetry(
            budget,
            directory / "api-requests.jsonl",
            agent_id=job["agent_id"],
            route=model["route"],
        )
        api_instrument.install(_OpenAICompatibleBackend, telemetry)

        cfg = build_config(job, directory, api_key, base_url)
        atomic_json(directory / "harness-configuration.json", cfg.safe_snapshot())
        actor = Agent(f"tb-{job['agent_id']}", agconfig=cfg, harness="codex")
        sandbox = actor._ensure_sandbox()
        # Harbor's agent environments expose these directories; tasks may use them.
        output, code = sandbox.exec(
            "mkdir -p /logs/agent /logs/artifacts /logs/verifier && "
            "chmod 777 /logs/agent /logs/artifacts /logs/verifier && pwd"
        )
        if code:
            raise RuntimeError(f"Container setup command failed: {safe(output)[-500:]}")
        inspection = json.loads(
            subprocess.check_output(["podman", "inspect", sandbox._name], text=True, timeout=60)
        )[0]
        container_cgroup = container_scope(cgroup_of(inspection["State"]["Pid"]))
        isolation = verify_isolation(job, sandbox, inspection, container_cgroup, directory)
        atomic_json(directory / "isolation.json", isolation)
        workdir = inspection["Config"].get("WorkingDir") or "/"
        state.update(
            container_name=sandbox._name,
            container_id=inspection["Id"],
            container_pid=inspection["State"]["Pid"],
            container_cgroup=str(container_cgroup),
            container_image_id=inspection.get("Image"),
            workdir=workdir,
            isolation_verified=isolation["verified"],
        )
        if not isolation["verified"] and job["sandbox"].get("require_isolation", True):
            raise RuntimeError("Isolation check failed: " + "; ".join(isolation["problems"]))
        state.update(status="ready", ready_ns=time.monotonic_ns())
        state["setup_seconds"] = (state["ready_ns"] - state["setup_start_ns"]) / 1e9
        save()
        atomic_json(
            directory / "ready.json",
            dict(
                ready_ns=state["ready_ns"],
                container_name=sandbox._name,
                container_id=inspection["Id"],
                container_cgroup=str(container_cgroup),
                host_cgroup=str(host_cgroup),
                writable_mounts=isolation["writable_mounts"],
            ),
        )

        # Gate: the runner releases this agent and sets its deadlines.
        gate = directory / "start.json"
        gate_deadline = time.monotonic() + job["timeouts"]["setup_seconds"]
        while not gate.exists():
            if time.monotonic() > gate_deadline:
                raise RuntimeError("Runner never released this agent")
            time.sleep(0.02)
        release = read_json(gate)
        if release.get("cancel"):
            state.update(status="not_started", censored=True, error="run stopped before release")
            return 0
        measured_start = time.monotonic_ns()
        deadline_ns = measured_start + int(job["timeouts"]["task_seconds"] * 1e9)
        if release.get("run_deadline_ns"):
            deadline_ns = min(deadline_ns, int(release["run_deadline_ns"]))
        state.update(status="running", start_ns=measured_start, deadline_ns=deadline_ns)
        save()
        atomic_json(
            directory / "counters-start.json",
            dict(
                t=time.monotonic_ns(),
                host=counters(host_cgroup),
                container=counters(container_cgroup),
            ),
        )
        trace = ContainerTrace(
            container_cgroup,
            directory / "container-series.jsonl",
            job["profiling"].get("container_trace_seconds", 0.1),
        )
        trace.start()
        # Agency checkpoints and then stops (hibernates) the container when the
        # skill completes, which removes its cgroup. Read the counters at those
        # two boundaries so the task's totals are exact and the checkpoint is
        # visible as its own interval.
        original_commit, original_stop = sandbox.commit, sandbox.stop

        def commit(*args, **kwargs):
            if "commit_start_ns" not in marks:
                marks["pre_commit"] = counters(container_cgroup)
                marks["commit_start_ns"] = time.monotonic_ns()
            try:
                return original_commit(*args, **kwargs)
            finally:
                marks["commit_end_ns"] = time.monotonic_ns()

        def stop(*args, **kwargs):
            if "stop_ns" not in marks:
                marks["pre_stop"] = counters(container_cgroup)
                marks["stop_ns"] = time.monotonic_ns()
            return original_stop(*args, **kwargs)

        try:
            sandbox.commit, sandbox.stop = commit, stop
            state["lifecycle_hooks"] = True
        except AttributeError:
            state["lifecycle_hooks"] = False
        agprof.start(
            directory / "profile",
            sample_hz=job["profiling"]["agprof_sample_hz"],
            sample_gpu=False,
            auto_functions=False,
        )
        profiler_started = True
        skill = agskill(
            name="terminal_bench_task",
            prompt=PROMPT.format(workdir=workdir),
            input_schema=agdata(task=str),
            output_schema=agdata(report=agrawstring),
        )
        instruction = Path(job["instruction_file"]).read_text()
        remaining = (deadline_ns - time.monotonic_ns()) / 1e9
        if remaining <= 0:
            state.update(status="timeout", censored=True, end_ns=time.monotonic_ns())
        else:
            kwargs = {}
            if job["agent"].get("max_steps"):
                kwargs["max_steps"] = job["agent"]["max_steps"]
            pending = actor.run(skill, agdata(task=instruction), **kwargs)
            try:
                pending.wait(timeout=max(0.01, remaining))
            except TimeoutError:
                state.update(status="timeout", censored=True, end_ns=time.monotonic_ns())
            else:
                end = time.monotonic_ns()
                payload = pending.to_dict()
                atomic_json(
                    directory / "agent-result.json",
                    json.loads(safe(json.dumps(payload, default=str))),
                )
                if end > deadline_ns:
                    state.update(status="timeout", censored=True, end_ns=end)
                elif "error" in payload:
                    state.update(
                        status="agent_error", end_ns=end, error=safe(payload["error"])[-2000:]
                    )
                else:
                    state.update(status="completed", completed=True, end_ns=end)
    except Interrupted as exc:
        state.update(
            status="cancelled" if measured_start else "setup_interrupted",
            censored=measured_start is not None,
            error=safe(exc),
            end_ns=time.monotonic_ns(),
        )
    except Exception as exc:
        state.update(
            status="agent_error" if measured_start else "setup_failed",
            error=safe(f"{type(exc).__name__}: {exc}")[-2000:],
            traceback=safe(traceback.format_exc())[-6000:],
            end_ns=time.monotonic_ns(),
        )
        exit_code = 0 if measured_start else 3
    finally:
        # From here on a second signal must not abort trace and state flushing.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if trace is not None:
                trace.halt()
                state["container_trace_rows"] = trace.rows
            if measured_start:
                live = counters(container_cgroup)  # None once the container is stopped
                atomic_json(
                    directory / "counters-end.json",
                    dict(
                        t=time.monotonic_ns(),
                        host=counters(host_cgroup),
                        # Agent work only: read just before Agency's checkpoint.
                        container=marks.get("pre_commit") or live or marks.get("pre_stop"),
                        container_source=(
                            "before checkpoint"
                            if marks.get("pre_commit")
                            else "at task end"
                            if live
                            else "before stop"
                            if marks.get("pre_stop")
                            else "unavailable"
                        ),
                        # Includes the checkpoint: read just before the stop.
                        container_final=marks.get("pre_stop") or live,
                    ),
                )
                for key in ("commit_start_ns", "commit_end_ns", "stop_ns"):
                    if key in marks:
                        state[key] = marks[key]
                state["wall_seconds"] = max(0.0, (state["end_ns"] - measured_start) / 1e9)
            if actor is not None and state["status"] not in ("completed",):
                try:
                    actor.stop()
                except Exception as exc:
                    state["stop_error"] = safe(exc)[-500:]
            if profiler_started:
                try:
                    from agency.observability.profiler import agprof

                    agprof.stop()
                    state["profile_summary"] = agprof.summary_metrics()
                except Exception as exc:
                    state["profile_error"] = safe(exc)[-500:]
            save()
            if sandbox is not None and measured_start:
                # Hand the still-running container to the runner for artifact
                # collection (done on reserved CPUs), then tear it down.
                atomic_json(
                    directory / "agent-finished.json",
                    dict(
                        end_ns=state.get("end_ns"),
                        status=state["status"],
                        container_name=state.get("container_name"),
                    ),
                )
                waited = time.monotonic() + job["timeouts"]["collect_seconds"]
                while not (directory / "collected.json").exists():
                    if time.monotonic() > waited:
                        state["collect_wait"] = "timed out waiting for the runner"
                        break
                    time.sleep(0.05)
                state["collected_ns"] = time.monotonic_ns()
            if actor is not None:
                try:
                    actor.data_logger.flush()
                    if actor.sandbox is not None:
                        actor.sandbox.destroy()
                        actor.sandbox = None
                except Exception as exc:
                    state["cleanup_error"] = safe(exc)[-500:]
            try:
                from agency.orchestrator import peek_orchestrator

                orchestrator = peek_orchestrator()
                if orchestrator is not None:
                    orchestrator.flush()
                    orchestrator.shutdown(timeout_s=15)
            except Exception as exc:
                state["orchestrator_error"] = safe(exc)[-500:]
        finally:
            state["cleanup_end_ns"] = time.monotonic_ns()
            state["finished"] = True
            save()
    return exit_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    args = parser.parse_args(argv)
    job = read_json_or(args.job)
    if not isinstance(job, dict):
        print(f"Unreadable job file {args.job}", file=sys.stderr)
        return 4
    return run(job)


if __name__ == "__main__":
    raise SystemExit(main())
