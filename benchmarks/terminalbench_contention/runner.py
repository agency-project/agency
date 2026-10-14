"""Run one condition (maximum concurrent agents) over the fixed task corpus.

The runner is a plain foreground program: it executes exactly one planned run
and exits. It does not retry, reschedule, or supervise itself; a separate
operator decides what to do with its exit code and artifacts.

Scheduling is deliberately simple: tasks launch in the repetition's seeded
order; at most ``agents`` are in flight; when one finishes (its container is
gone) the next pending task launches. Every task runs at most once.

Measured interval: from the simultaneous release of the first wave of agents
(after all of them finished setup) until the last agent stops executing.
Artifact collection runs here, on the reserved CPUs. Grading runs after the
measured interval and never overlaps it.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import analysis, cache_control
from . import config as config_module
from .api_budget import BudgetPolicy, SharedBudget, verify_ledger
from .placement import ProcessLauncher, SystemdLauncher, WorkloadSlice, restore_stale
from .sampler import ResourceSampler
from .util import (
    PACKAGE,
    REPOSITORY,
    append_jsonl,
    atomic_json,
    capture,
    clock_anchor,
    git_state,
    read_json_or,
    sha256_file,
    sha256_tree,
)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_PRECONDITION = 3  # nothing was measured: environment not ready
EXIT_INVALID = 4  # the run was aborted or its evidence is unusable
EXIT_DEGRADED = 5  # the run finished but some tasks had infrastructure failures
EXIT_INTERRUPTED = 130

WORKER_MODULE = "benchmarks.terminalbench_contention.worker"
BRIDGE = PACKAGE / "harbor_side" / "bridge.py"


class PreconditionError(RuntimeError):
    """The run cannot start; nothing has been measured."""


class RunAborted(RuntimeError):
    """The run was stopped after it started."""


@dataclass
class Slot:
    index: int  # 1-based position in the launch order
    task: dict
    directory: Path
    state: str = "pending"
    handle: object = None
    launched_ns: "int | None" = None
    ready_ns: "int | None" = None
    released_ns: "int | None" = None
    finished_seen_ns: "int | None" = None
    exited_ns: "int | None" = None
    returncode: "int | None" = None
    collect_future: object = None
    collection: "dict | None" = None
    terminate_sent_ns: "int | None" = None
    kill_sent: bool = False
    ready_info: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def agent_id(self) -> str:
        return f"A{self.index:02d}"

    @property
    def in_flight(self) -> bool:
        return self.state not in ("pending", "exited")


def slot_directory(run_dir: Path, index: int, task: dict) -> Path:
    return run_dir / "agents" / f"{index:02d}-{task['directory']}"


def next_attempt(parent: Path) -> Path:
    attempts = sorted(parent.glob("attempt-[0-9][0-9]"))
    return parent / f"attempt-{len(attempts) + 1:02d}"


def freeze_runtime_source(run_dir: Path, repository: Path = REPOSITORY) -> dict:
    """Copy the Agency package the sandboxes will import; never the repository."""
    target = run_dir / "runtime-source"
    shutil.copytree(
        repository / "agency",
        target / "agency",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "perfetto"),
    )
    shutil.copy2(repository / "pyproject.toml", target / "pyproject.toml")
    return dict(path=str(target), sha256=sha256_tree(target))


class HarborCollector:
    """Collect a finished agent's declared artifacts with Harbor's ArtifactHandler."""

    def __init__(self, harbor_python, model, agent_version, timeout):
        self.harbor_python, self.model = str(harbor_python), model
        self.agent_version, self.timeout = agent_version, timeout

    def __call__(self, slot: Slot, container: str, run_dir: Path) -> dict:
        trial = run_dir / "grading" / "source" / slot.directory.name
        # Agency stops (hibernates) a container whose skill completed. Resume
        # that same container, with its writable layer, so Harbor can read it.
        running = capture(
            ["podman", "inspect", "-f", "{{.State.Running}} {{.State.Paused}}", container],
            timeout=60,
        )
        resumed = None
        if running["returncode"] != 0:
            return dict(ok=False, error=f"container {container} no longer exists")
        flags = running["stdout"].split()
        if flags[:1] != ["true"]:
            resumed = capture(["podman", "start", container], timeout=120)
            if resumed["returncode"] != 0:
                return dict(
                    ok=False,
                    error="could not resume the stopped container: " + resumed["stderr"][-500:],
                )
        elif flags[1:2] == ["true"]:
            capture(["podman", "unpause", container], timeout=60)
        argv = [
            self.harbor_python,
            str(BRIDGE),
            "collect",
            "--task-dir",
            slot.task["task_dir"],
            "--container",
            container,
            "--trial-dir",
            str(trial),
            "--model",
            self.model,
            "--agent-version",
            self.agent_version,
        ]
        result = capture(argv, timeout=self.timeout)
        return _bridge_result(result, trial_dir=str(trial), resumed=resumed is not None)


class HarborGrader:
    """Grade a collected record with Harbor's regrade trial and Verifier."""

    def __init__(self, harbor_python, timeout, label):
        self.harbor_python, self.timeout, self.label = str(harbor_python), timeout, label

    def __call__(self, slot_name: str, task: dict, source: Path, run_dir: Path) -> dict:
        argv = [
            self.harbor_python,
            str(BRIDGE),
            "regrade",
            "--source",
            str(source),
            "--task-dir",
            task["task_dir"],
            "--trials-dir",
            str(run_dir / "grading" / "results"),
            "--trial-name",
            slot_name,
            "--timeout",
            str(self.timeout),
            "--label",
            self.label,
        ]
        return _bridge_result(capture(argv, timeout=self.timeout + 120))


def _bridge_result(result: dict, **extra) -> dict:
    import json

    lines = [line for line in result.get("stdout", "").splitlines() if line.startswith("{")]
    try:
        payload = json.loads(lines[-1]) if lines else None
    except ValueError:
        payload = None
    if payload is None:
        payload = dict(
            ok=False,
            error=result.get("error") or (result.get("stderr") or "")[-2000:] or "no bridge output",
        )
    payload.setdefault("returncode", result.get("returncode"))
    payload.update(extra)
    return payload


class Runner:
    def __init__(
        self,
        config: dict,
        manifest: dict,
        entry: dict,
        *,
        prepared: dict,
        layout,
        run_dir: Path,
        launcher,
        collector,
        grader=None,
        workload: "WorkloadSlice | None" = None,
        worker_module: str = WORKER_MODULE,
        mock: "dict | None" = None,
        poll_seconds: float = 0.2,
        stop_file: "Path | None" = None,
        sample: bool = True,
        confine_self: bool = True,
        job_extra: "dict | None" = None,
        freeze_source: bool = True,
    ) -> None:
        self.config, self.manifest, self.entry = config, manifest, entry
        self.prepared, self.layout, self.run_dir = prepared, layout, Path(run_dir)
        self.launcher, self.collector, self.grader = launcher, collector, grader
        self.workload, self.worker_module, self.mock = workload, worker_module, mock
        self.poll_seconds, self.stop_file = poll_seconds, stop_file
        self.sample, self.confine_self = sample, confine_self
        self.job_extra = job_extra or {}
        self.freeze_source = freeze_source
        self.agents = entry["agents"]
        self.run_id = f"{entry['run_label']}-{self.run_dir.name}"
        self.label_prefix = config["sandbox"]["label_prefix"]
        self.run_label = f"{self.label_prefix}.run={self.run_id}"
        tasks = {t["task_id"]: t for t in manifest["tasks"]}
        self.slots = [
            Slot(
                index=i,
                task=self._task_record(tasks[task_id]),
                directory=slot_directory(self.run_dir, i, tasks[task_id]),
            )
            for i, task_id in enumerate(entry["launch_order"], 1)
        ]
        self.status = dict(
            run_id=self.run_id,
            status="created",
            **{k: entry[k] for k in ("run_label", "condition", "agents", "repetition")},
        )
        self.stop_reason: "str | None" = None
        self.measured_start_ns: "int | None" = None
        self.run_deadline_ns: "int | None" = None
        self.budget: "SharedBudget | None" = None
        self.sampler: "ResourceSampler | None" = None
        self.peak_in_flight = 0
        self.peak_executing = 0
        self._signals = 0
        self._low_memory_since = None
        self._last_safety = 0.0

    # ------------------------------------------------------------- utilities

    def _task_record(self, task: dict) -> dict:
        prepared = (self.prepared.get("tasks") or {}).get(task["task_id"], {})
        return {**task, **prepared}

    def event(self, kind: str, **fields) -> None:
        append_jsonl(
            self.run_dir / "events.jsonl", dict(t=time.monotonic_ns(), kind=kind, **fields)
        )

    def set_status(self, status: str, **fields) -> None:
        self.status.update(status=status, updated_ns=time.monotonic_ns(), **fields)
        atomic_json(self.run_dir / "status.json", self.status)

    def in_flight(self) -> list:
        return [s for s in self.slots if s.in_flight]

    def executing(self) -> list:
        return [s for s in self.slots if s.state == "running"]

    def agent_cgroups(self) -> dict:
        cgroups = {}
        for slot in list(self.slots):
            if slot.state in ("ready", "running", "finished") and slot.ready_info:
                cgroups[slot.index] = dict(
                    host=slot.ready_info.get("host_cgroup"),
                    container=slot.ready_info.get("container_cgroup"),
                )
        return cgroups

    # ------------------------------------------------------------------ jobs

    def make_job(self, slot: Slot) -> dict:
        config, task = self.config, slot.task
        memory = config["memory"]
        container_memory = (
            f"{task['environment']['memory_mb']}m"
            if memory["container_memory"] == "task"
            else memory["container_memory"]
        )
        swap = (
            None
            if memory["container_swap"] == "default"
            else ("-1" if memory["container_swap"] == "unlimited" else memory["container_swap"])
        )
        harness_root = Path(self.prepared["harness_python"])
        job = dict(
            run_id=self.run_id,
            slot=slot.index,
            agent_id=slot.agent_id,
            directory=str(slot.directory),
            task=task,
            instruction_file=task["instruction_file"],
            cpus=list(self.layout.pool_cpus) if self.layout is not None else None,
            model=dict(config["model"]),
            agent=dict(config["agent"]),
            api=dict(directory=str(self.run_dir / "api-metrics" / "ledger"), policy=config["api"]),
            env_file=str(REPOSITORY / config["model"]["env_file"]),
            runtime_source=self.prepared["runtime_source"],
            codex_binary=str(REPOSITORY / config["agent"]["codex_binary"]),
            tiktoken_cache=self.prepared.get("tiktoken_cache"),
            timeouts=dict(config["timeouts"]),
            profiling=dict(config["profiling"]),
            mock=self.mock,
            sandbox=dict(
                memory=container_memory,
                memory_swap=swap,
                pids_limit=memory["container_pids_limit"],
                cgroup_parent=self.workload.name if self.workload is not None else None,
                cpuset_mems=(
                    ",".join(map(str, self.layout.numa_nodes)) if self.layout is not None else None
                ),
                network=config["sandbox"]["network"],
                run_label=self.run_label,
                task_label=f"{self.label_prefix}.task={self.run_id}.{slot.index:02d}",
                harness_python_host=str(harness_root),
                harness_python_container="/opt/agency-python/bin/python3.12",
                agency_run_tmp_prefix=f"/tmp/agency-{os.getuid()}/",
                require_isolation=True,
            ),
        )
        job.update(self.job_extra)
        return job

    def launch(self, slot: Slot) -> None:
        slot.directory.mkdir(parents=True, exist_ok=False)
        job_path = slot.directory / "job.json"
        atomic_json(job_path, self.make_job(slot))
        slot.handle = self.launcher.start(
            self.worker_module, job_path, slot.directory / "execution.log", slot.index
        )
        slot.state, slot.launched_ns = "setup", time.monotonic_ns()
        self.event("launch", slot=slot.index, task=slot.task["task_id"], unit=slot.handle.name)

    def release(self, slots: list) -> None:
        now = time.monotonic_ns()
        for slot in slots:
            atomic_json(
                slot.directory / "start.json",
                dict(release_ns=now, run_deadline_ns=self.run_deadline_ns),
            )
            slot.state, slot.released_ns = "running", now
            self.event("release", slot=slot.index, task=slot.task["task_id"])

    # ------------------------------------------------------------- lifecycle

    def prepare_directories(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=False)
        for name in ("agents", "traces", "resource-metrics", "api-metrics", "grading"):
            (self.run_dir / name).mkdir()
        config = self.config
        atomic_json(
            self.run_dir / "config.json",
            dict(config=config, fingerprint=config_module.fingerprint(config)),
        )
        atomic_json(self.run_dir / "plan.json", self.entry)

    def write_metadata(self, extra: dict) -> None:
        config = self.config
        locks = {p.name: sha256_file(p) for p in sorted((PACKAGE / "locks").glob("*.txt"))}
        metadata = dict(
            schema_version=1,
            run_id=self.run_id,
            run_label=self.entry["run_label"],
            condition=self.entry["condition"],
            agents=self.agents,
            repetition=self.entry["repetition"],
            order_seed=self.entry["order_seed"],
            launch_order=self.entry["launch_order"],
            clock=clock_anchor(),
            git=git_state(),
            benchmark_source_sha256=sha256_tree(PACKAGE),
            dataset=config["benchmark"]["dataset"],
            corpus_sha256=self.manifest["corpus_sha256"],
            task_digests={t["task_id"]: t["digest"] for t in self.manifest["tasks"]},
            images={
                t["task_id"]: dict(
                    environment=t["environment"]["image"], verifier=t["verifier"]["image"]
                )
                for t in self.manifest["tasks"]
            },
            harbor_version=config["benchmark"]["harbor_version"],
            dependency_locks_sha256=locks,
            model={k: v for k, v in config["model"].items() if not k.endswith("_env")},
            credential_sources=dict(
                api_key_env=config["model"]["api_key_env"],
                base_url_env=config["model"]["base_url_env"],
                note="values are read from the environment at run time and never stored",
            ),
            agent=config["agent"],
            cpu=dict(config=config["cpu"], layout=self.layout.to_dict() if self.layout else None),
            memory=config["memory"],
            api_budget=config["api"],
            api_policy_fingerprint=BudgetPolicy.from_dict(config["api"]).fingerprint(),
            timeouts=config["timeouts"],
            timeout_policy=(
                "Each task has one wall-clock cap (timeouts.task_seconds) measured from its "
                "release. A task at the cap is stopped, marked censored and still graded. "
                "Tasks never released before the run ended are recorded as not_started."
            ),
            cache_control=cache_control.strategy(config),
            launcher=self.launcher.kind,
            worker_module=self.worker_module,
            mock_model=bool(self.mock),
            python=sys.version,
            **extra,
        )
        atomic_json(self.run_dir / "metadata.json", metadata)

    def _on_signal(self, signum, _frame) -> None:
        self._signals += 1
        self.stop_reason = self.stop_reason or f"runner received signal {signum}"

    def safety_checks(self) -> None:
        # The run ceiling is checked on every pass so no task is launched past it.
        if self.run_deadline_ns is not None and time.monotonic_ns() >= self.run_deadline_ns:
            self.stop_reason = self.stop_reason or "run time limit reached"
        now = time.monotonic()
        if now - self._last_safety < 1.0:
            return
        self._last_safety = now
        if self.stop_file is not None and self.stop_file.exists():
            self.stop_reason = self.stop_reason or f"STOP file present at {self.stop_file}"
        memory = self.config["memory"]
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    available_mb = int(line.split()[1]) // 1024
                    if available_mb < memory["host_min_available_mb"]:
                        self._low_memory_since = self._low_memory_since or now
                        if now - self._low_memory_since >= memory["host_min_available_seconds"]:
                            self.stop_reason = self.stop_reason or (
                                f"host MemAvailable below {memory['host_min_available_mb']} MB "
                                f"for {memory['host_min_available_seconds']} s"
                            )
                    else:
                        self._low_memory_since = None
        except OSError:
            pass
        if shutil.disk_usage(self.run_dir).free < memory["min_free_disk_gb"] * 1024**3:
            self.stop_reason = self.stop_reason or "free disk space below the configured minimum"

    # ------------------------------------------------------------- main loop

    def step(self, pool: ThreadPoolExecutor) -> None:
        now = time.monotonic_ns()
        timeouts = self.config["timeouts"]
        grace_ns = int(timeouts["stop_grace_seconds"] * 1e9)
        for slot in self.slots:
            if slot.state in ("pending", "exited"):
                continue
            if slot.state == "setup":
                if (slot.directory / "ready.json").exists():
                    slot.ready_info = read_json_or(slot.directory / "ready.json") or {}
                    slot.state, slot.ready_ns = "ready", now
                    self.event("ready", slot=slot.index)
                elif (
                    now - slot.launched_ns > timeouts["setup_seconds"] * 1e9 and not slot.kill_sent
                ):
                    slot.notes.append("setup exceeded timeouts.setup_seconds; worker killed")
                    slot.handle.kill()
                    slot.kill_sent = True
            if slot.state == "running":
                if (slot.directory / "agent-finished.json").exists():
                    slot.state, slot.finished_seen_ns = "finished", now
                    finished = read_json_or(slot.directory / "agent-finished.json") or {}
                    self.event("agent_finished", slot=slot.index, status=finished.get("status"))
                    slot.collect_future = pool.submit(
                        self._collect, slot, finished.get("container_name")
                    )
                else:
                    cap_ns = slot.released_ns + int(timeouts["task_seconds"] * 1e9)
                    if self.run_deadline_ns is not None:
                        cap_ns = min(cap_ns, self.run_deadline_ns)
                    if slot.terminate_sent_ns is None and now > cap_ns + grace_ns:
                        slot.notes.append("worker missed its deadline; SIGTERM sent by the runner")
                        slot.handle.terminate()
                        slot.terminate_sent_ns = now
                        self.event("terminate", slot=slot.index, reason="deadline backstop")
                    elif (
                        slot.terminate_sent_ns is not None
                        and not slot.kill_sent
                        and now > slot.terminate_sent_ns + grace_ns
                    ):
                        slot.notes.append("worker ignored SIGTERM; killed by the runner")
                        slot.handle.kill()
                        slot.kill_sent = True
                        self.event("kill", slot=slot.index)
            if slot.state == "finished" and slot.collect_future.done():
                try:
                    slot.collection = slot.collect_future.result()
                except Exception as exc:  # collection failure is recorded, not fatal
                    slot.collection = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
                atomic_json(slot.directory / "collected.json", slot.collection)
                slot.state = "collected"
                self.event("collected", slot=slot.index, ok=bool(slot.collection.get("ok")))
            returncode = slot.handle.poll()
            if returncode is not None:
                if slot.state == "finished" and slot.collect_future is not None:
                    continue  # let the collection result be recorded first
                slot.returncode, slot.exited_ns = returncode, now
                previous, slot.state = slot.state, "exited"
                self.event("exit", slot=slot.index, returncode=returncode, from_state=previous)

    def _collect(self, slot: Slot, container: "str | None") -> dict:
        started = time.monotonic_ns()
        if not container:
            return dict(ok=False, error="worker reported no container name")
        result = self.collector(slot, container, self.run_dir)
        result["collect_seconds"] = (time.monotonic_ns() - started) / 1e9
        return result

    def cancel_outstanding(self, reason: str) -> None:
        """Stop everything in flight; tasks never launched stay not_started."""
        self.event("stop", reason=reason)
        for slot in self.slots:
            if slot.state in ("setup", "ready"):
                atomic_json(slot.directory / "start.json", dict(cancel=True, reason=reason))
            if slot.state in ("setup", "ready", "running") and slot.terminate_sent_ns is None:
                slot.handle.terminate()
                slot.terminate_sent_ns = time.monotonic_ns()

    def run_loop(self) -> None:
        first_wave = self.slots[: self.agents]
        stopping = False
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="tb-collect") as pool:
            for slot in first_wave:
                self.launch(slot)
            self.set_status("setup", first_wave=len(first_wave))
            # Barrier: nothing is measured until the whole first wave is ready.
            while self.measured_start_ns is None:
                self.step(pool)
                self.safety_checks()
                failed = [s for s in first_wave if s.state == "exited"]
                if failed or self.stop_reason:
                    reason = self.stop_reason or (
                        "first-wave setup failed for slot(s) "
                        + ", ".join(str(s.index) for s in failed)
                    )
                    self.cancel_outstanding(reason)
                    self._drain(pool)
                    raise PreconditionError(reason)
                if all(s.state == "ready" for s in first_wave):
                    self.measured_start_ns = time.monotonic_ns()
                    self.run_deadline_ns = self.measured_start_ns + int(
                        self.entry["run_timeout_seconds"] * 1e9
                    )
                    self.release(first_wave)
                    self.peak_in_flight = self.peak_executing = len(first_wave)
                    self.set_status(
                        "running",
                        measured_start_ns=self.measured_start_ns,
                        run_deadline_ns=self.run_deadline_ns,
                    )
                    self.event("measured_start", agents=len(first_wave))
                    break
                time.sleep(self.poll_seconds)
            while True:
                self.step(pool)
                self.safety_checks()
                if self.stop_reason and not stopping:
                    stopping = True
                    self.cancel_outstanding(self.stop_reason)
                if not stopping:
                    # Refill: an agent released as soon as it is ready.
                    ready = [s for s in self.slots if s.state == "ready"]
                    if ready:
                        self.release(ready)
                    pending = [s for s in self.slots if s.state == "pending"]
                    while pending and len(self.in_flight()) < self.agents:
                        self.launch(pending.pop(0))
                self.peak_in_flight = max(self.peak_in_flight, len(self.in_flight()))
                self.peak_executing = max(self.peak_executing, len(self.executing()))
                if not self.in_flight() and (
                    stopping or not any(s.state == "pending" for s in self.slots)
                ):
                    break
                if stopping and self._signals >= 2:
                    for slot in self.in_flight():
                        slot.handle.kill()
                time.sleep(self.poll_seconds)

    def _drain(self, pool: ThreadPoolExecutor) -> None:
        """Wait (bounded) for cancelled workers to exit, then kill stragglers."""
        deadline = time.monotonic() + self.config["timeouts"]["stop_grace_seconds"]
        while self.in_flight() and time.monotonic() < deadline:
            self.step(pool)
            time.sleep(self.poll_seconds)
        for slot in self.in_flight():
            slot.handle.kill()
            slot.kill_sent = True
            slot.notes.append("killed while draining a stopped run")
        deadline = time.monotonic() + 15
        while self.in_flight() and time.monotonic() < deadline:
            self.step(pool)
            time.sleep(self.poll_seconds)

    # ------------------------------------------------------------- finishing

    def remove_leftover_containers(self) -> list:
        """Force-remove only this run's containers; return the names found."""
        listing = capture(
            ["podman", "ps", "-a", "--filter", f"label={self.run_label}", "--format", "{{.Names}}"],
            timeout=60,
        )
        names = listing["stdout"].split()
        for name in names:
            capture(["podman", "rm", "--force", "--time", "0", name], timeout=120)
        return names

    def collect_task_records(self) -> list:
        records = []
        for slot in self.slots:
            state = read_json_or(slot.directory / "task.json") or {}
            status = state.get("status")
            if slot.state == "pending":
                status = "not_started"
            elif not state.get("finished"):
                # The worker process ended without writing its final record.
                status = "worker_died" if slot.released_ns else "setup_failed"
            record = dict(
                slot=slot.index,
                agent_id=slot.agent_id,
                task_id=slot.task["task_id"],
                directory=slot.directory.name,
                status=status,
                completed=bool(state.get("completed")) and status == "completed",
                censored=bool(state.get("censored")) or status in ("not_started", "worker_died"),
                worker_returncode=slot.returncode,
                launched_ns=slot.launched_ns,
                ready_ns=state.get("ready_ns"),
                start_ns=state.get("start_ns"),
                end_ns=state.get("end_ns"),
                deadline_ns=state.get("deadline_ns"),
                exited_ns=slot.exited_ns,
                setup_seconds=state.get("setup_seconds"),
                wall_seconds=state.get("wall_seconds"),
                turnaround_seconds=(
                    (slot.exited_ns - state["end_ns"]) / 1e9
                    if slot.exited_ns and state.get("end_ns")
                    else None
                ),
                isolation_verified=state.get("isolation_verified"),
                error=state.get("error"),
                collection_ok=bool(slot.collection and slot.collection.get("ok")),
                collection=slot.collection,
                notes=slot.notes,
            )
            records.append(record)
        return records

    def grade(self, records: list) -> dict:
        """Grade every collected task. Runs after the measured interval."""
        verdicts = {}
        if self.grader is None:
            return verdicts
        gradable = [r for r in records if r["collection_ok"]]

        def one(record):
            slot = self.slots[record["slot"] - 1]
            source = self.run_dir / "grading" / "source" / record["directory"]
            try:
                return record["directory"], self.grader(
                    record["directory"], slot.task, source, self.run_dir
                )
            except Exception as exc:
                return record["directory"], dict(ok=False, error=f"{type(exc).__name__}: {exc}")

        with ThreadPoolExecutor(max_workers=self.config["grading"]["parallel"]) as pool:
            for name, verdict in pool.map(one, gradable):
                verdicts[name] = verdict
                atomic_json(self.run_dir / "grading" / "verdicts.json", verdicts)
        atomic_json(self.run_dir / "grading" / "verdicts.json", verdicts)
        return verdicts

    def execute(self, *, grade: bool = True) -> int:
        """Run the planned condition. Returns a process exit code."""
        self.prepare_directories()
        self.set_status("preparing")
        previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        affinity_before = os.sched_getaffinity(0)
        placement = None
        exit_code = EXIT_OK
        error = None
        sampler_report = None
        try:
            runtime = (
                freeze_runtime_source(self.run_dir)
                if self.freeze_source
                else dict(path=self.prepared.get("runtime_source"), sha256=None)
            )
            self.prepared = {**self.prepared, "runtime_source": runtime["path"]}
            for slot in self.slots:  # refresh records that depend on self.prepared
                slot.task = self._task_record(slot.task)
            policy = BudgetPolicy.from_dict(self.config["api"])
            ledger = self.run_dir / "api-metrics" / "ledger"
            self.budget = SharedBudget.create(ledger, policy, run_id=self.run_id)
            if self.workload is not None:
                placement = self.workload.enter()
            if self.confine_self and self.layout is not None:
                os.sched_setaffinity(
                    0, set(self.layout.reserved_cpus) & affinity_before or affinity_before
                )
            self.write_metadata(
                dict(
                    runtime_source=runtime,
                    placement=placement,
                    runner_affinity=sorted(os.sched_getaffinity(0)),
                    host_state_at_start=cache_control.host_state(),
                )
            )
            if self.sample:
                self.sampler = ResourceSampler(
                    self.run_dir / "resource-metrics" / "samples.jsonl",
                    interval=self.config["profiling"]["host_sample_seconds"],
                    pool_cpus=self.layout.pool_cpus if self.layout else os.sched_getaffinity(0),
                    reserved_cpus=self.layout.reserved_cpus if self.layout else (),
                    workload_cgroup=self.workload.cgroup if self.workload else None,
                    agents=self.agent_cgroups,
                    api_snapshot=self.budget.snapshot,
                    active=lambda: len(self.executing()),
                    disk_path=self.run_dir,
                )
                atomic_json(
                    self.run_dir / "resource-metrics" / "capabilities.json",
                    self.sampler.capabilities,
                )
                self.sampler.start()
            self.run_loop()
        except PreconditionError as exc:
            exit_code, error = EXIT_PRECONDITION, str(exc)
        except Exception as exc:  # preserve diagnostics for anything unexpected
            import traceback

            # Before the measured interval began nothing was measured.
            exit_code = EXIT_INVALID if self.measured_start_ns else EXIT_PRECONDITION
            error = f"{type(exc).__name__}: {exc}"
            atomic_json(
                self.run_dir / "runner-error.json",
                dict(error=error, traceback=traceback.format_exc()),
            )
            for slot in self.in_flight():
                slot.handle.kill()
        finally:
            measured_end_ns = time.monotonic_ns()
            if self.sampler is not None:
                sampler_report = self.sampler.stop()
                atomic_json(self.run_dir / "resource-metrics" / "sampler.json", sampler_report)
            for slot in self.slots:  # no worker may outlive its run
                if slot.handle is not None and slot.handle.poll() is None:
                    slot.handle.kill()
                    slot.notes.append("worker still alive at run end; killed")
            leftovers = self.remove_leftover_containers()
            stale_units = self.launcher.stale_units(getattr(self.launcher, "unit_prefix", "\0"))
            restore = self.workload.restore() if self.workload is not None else None
            os.sched_setaffinity(0, affinity_before)
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        records = self.collect_task_records()
        ledger_report = verify_ledger(self.run_dir / "api-metrics" / "ledger")
        atomic_json(self.run_dir / "api-metrics" / "verification.json", ledger_report)
        if self.stop_reason and exit_code == EXIT_OK:
            exit_code = EXIT_INTERRUPTED if self._signals else EXIT_INVALID
            error = self.stop_reason
        ends = [r["end_ns"] for r in records if r["end_ns"]]
        if self.measured_start_ns is not None and ends:
            measured_end_ns = max(ends)
        run_record = dict(
            run_id=self.run_id,
            run_label=self.entry["run_label"],
            condition=self.entry["condition"],
            repetition=self.entry["repetition"],
            order_seed=self.entry["order_seed"],
            status="completed"
            if exit_code == EXIT_OK
            else ("not_started" if exit_code == EXIT_PRECONDITION else "aborted"),
            error=error,
            measured_start_ns=self.measured_start_ns,
            measured_end_ns=measured_end_ns if self.measured_start_ns else None,
            makespan_seconds=(
                (measured_end_ns - self.measured_start_ns) / 1e9 if self.measured_start_ns else None
            ),
            agents=self.agents,
            peak_in_flight=self.peak_in_flight,
            peak_executing=self.peak_executing,
            leaked_containers=leftovers,
            stale_units=stale_units,
            placement_restore=restore,
            sampler=sampler_report,
            ledger=ledger_report,
            tasks=records,
        )
        atomic_json(self.run_dir / "run.json", run_record)
        self.set_status(
            "grading" if grade and exit_code == EXIT_OK else run_record["status"], error=error
        )
        verdicts = {}
        if grade and self.measured_start_ns is not None:
            verdicts = self.grade(records)
        summary = analysis.finalize(self.run_dir, run_record, verdicts, self.config)
        infrastructure = summary["counts"]["infrastructure_failures"]
        if exit_code == EXIT_OK and (
            infrastructure
            or summary["counts"].get("agent_errors")
            or leftovers
            or not ledger_report["ok"]
            or summary["counts"].get("grading_failures")
        ):
            exit_code = EXIT_DEGRADED
        summary["exit_code"] = exit_code
        atomic_json(self.run_dir / "summary.json", summary)
        self.set_status(summary["status"], exit_code=exit_code, error=error)
        return exit_code


class RunLock:
    """Only one run at a time may use the experimental pool and the gateway."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise PreconditionError("Another benchmark run holds the run lock") from None
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


def stale_containers(label_prefix: str) -> list:
    listing = capture(
        ["podman", "ps", "-a", "--filter", f"label={label_prefix}.run", "--format", "{{.Names}}"],
        timeout=60,
    )
    return listing["stdout"].split()


def build_runner(
    config, manifest, entry, prepared, layout, *, mock=None, attempt_parent=None, use_systemd=True
) -> Runner:
    """Wire the real components together for one planned run."""
    root = config_module.artifact_root(config)
    parent = attempt_parent or (root / "runs" / entry["run_label"])
    run_dir = next_attempt(parent)
    restore_stale(root / "placement-lease.json")
    workload = launcher = None
    token = f"agencytb-{entry['run_label'].lower()}-{run_dir.name[-2:]}"
    python = REPOSITORY / ".venv" / "bin" / "python"
    runtime_max = (
        config["timeouts"]["task_seconds"]
        + config["timeouts"]["setup_seconds"]
        + config["timeouts"]["collect_seconds"]
        + 3 * config["timeouts"]["stop_grace_seconds"]
    )
    if use_systemd:
        workload = WorkloadSlice(
            layout,
            config["memory"],
            root / "placement-lease.json",
            confine_session=config["cpu"]["confine_user_session"],
        )
        launcher = SystemdLauncher(
            slice_name=workload.name,
            pool_cpus=layout.pool_cpus,
            python=python,
            working_directory=REPOSITORY,
            pythonpath=str(REPOSITORY),
            runtime_max_seconds=runtime_max,
            unit_prefix=token,
        )
    else:
        launcher = ProcessLauncher(
            pool_cpus=layout.pool_cpus,
            python=python,
            working_directory=REPOSITORY,
            pythonpath=str(REPOSITORY),
        )
    collector = HarborCollector(
        prepared["harbor_python"],
        config["model"]["model"],
        config["agent"]["codex_version"],
        config["timeouts"]["collect_seconds"],
    )
    grader = HarborGrader(
        prepared["harbor_python"],
        config["timeouts"]["grade_seconds"],
        f"{config['sandbox']['label_prefix']}.grade={entry['run_label']}",
    )
    return Runner(
        config,
        manifest,
        entry,
        prepared=prepared,
        layout=layout,
        run_dir=run_dir,
        launcher=launcher,
        collector=collector,
        grader=grader,
        workload=workload,
        mock=mock,
        stop_file=root / "STOP",
    )
