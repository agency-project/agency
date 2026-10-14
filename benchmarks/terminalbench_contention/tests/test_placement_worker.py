"""Workload placement (shared slice, CPU affinity) and the worker's isolation checks."""

import json
import os
import sys
import time
from pathlib import Path

import pytest

from benchmarks.terminalbench_contention import placement, topology, worker
from benchmarks.terminalbench_contention.util import REPOSITORY, read_json

MEMORY = dict(workload_memory_max="2G", workload_swap_max="max", workload_tasks_max=4096)

try:
    LAYOUT = topology.detect_layout(
        dict(experimental_physical_cores=6, reserved_physical_cores=2, smt_policy="siblings")
    )
except topology.InsufficientCpuError:
    LAYOUT = None

needs_systemd = pytest.mark.skipif(
    LAYOUT is None or not placement.systemd_available(),
    reason="needs a user systemd manager and an 8-core host",
)

PROBE = (
    "import json, os, sys, pathlib\n"
    "before = sorted(os.sched_getaffinity(0))\n"
    "os.sched_setaffinity(0, set(range(os.cpu_count())))\n"
    "after = sorted(os.sched_getaffinity(0))\n"
    "cgroup = pathlib.Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1]\n"
    "base = pathlib.Path('/sys/fs/cgroup') / cgroup.lstrip('/')\n"
    "quota = base / 'cpu.max'\n"
    "out = dict(before=before, after=after, cgroup=cgroup,\n"
    "           cpu_max=quota.read_text().strip() if quota.exists() else 'max (no cpu controller)',\n"
    "           cpuset=(base / 'cpuset.cpus.effective').read_text().strip())\n"
    "pathlib.Path(sys.argv[1]).with_suffix('.out').write_text(json.dumps(out))\n"
)


class FakeSystemd:
    """Records systemctl calls and answers `show`."""

    def __init__(self, tmp_path, effective):
        self.calls = []
        self.root = tmp_path / "cgroupfs"
        (self.root / "user" / "agencytb.slice").mkdir(parents=True)
        (self.root / "user" / "agencytb.slice" / "cpuset.cpus.effective").write_text(
            effective + "\n"
        )

    def capture(self, argv, timeout=60, **_):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        stdout = "AllowedCPUs=0-15\n" if "show" in argv else ""
        return dict(argv=argv, returncode=0, stdout=stdout, stderr="")

    def checked(self, argv, timeout=60, **_):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        return "/user/agencytb.slice" if "--property=ControlGroup" in argv else ""


def fake_slice(tmp_path, monkeypatch, effective, confine=True):
    fake = FakeSystemd(tmp_path, effective)
    monkeypatch.setattr(placement, "capture", fake.capture)
    monkeypatch.setattr(placement, "checked", fake.checked)
    monkeypatch.setattr(placement.WorkloadSlice, "cgroup_root", fake.root)
    workload = placement.WorkloadSlice(
        LAYOUT, MEMORY, tmp_path / "lease.json", confine_session=confine
    )
    return fake, workload


@pytest.mark.skipif(LAYOUT is None, reason="needs an 8-core host")
def test_slice_sets_one_shared_pool_and_restores_the_session(tmp_path, monkeypatch):
    fake, workload = fake_slice(tmp_path, monkeypatch, LAYOUT.pool_arg)
    lease = tmp_path / "lease.json"
    applied = workload.enter()
    set_calls = [c for c in fake.calls if "set-property" in c]
    slice_call = next(c for c in set_calls if placement.SLICE in c)
    assert f"AllowedCPUs={LAYOUT.pool_arg}" in slice_call and "MemoryMax=2G" in slice_call
    assert "MemorySwapMax=infinity" in slice_call and "--runtime" in slice_call
    # No per-agent CPU quota or weight is ever requested.
    assert not any("CPUQuota" in part or "CPUWeight" in part for c in fake.calls for part in c)
    session = [c for c in set_calls if placement.SLICE not in c]
    assert {c[4] for c in session} == set(placement.SESSION_UNITS)
    assert all(f"AllowedCPUs={LAYOUT.reserved_arg}" in c for c in session)
    assert read_json(lease)["active"] is True
    assert applied["session_confined_to"] == LAYOUT.reserved_arg
    assert applied["effective_cpus"] == LAYOUT.pool_arg
    fake.calls.clear()
    workload.restore()
    restored = [c for c in fake.calls if "set-property" in c]
    assert {c[4] for c in restored} == set(placement.SESSION_UNITS)
    assert all("AllowedCPUs=0-15" in c for c in restored)
    assert read_json(lease)["active"] is False
    assert any(c[:3] == ["systemctl", "--user", "stop"] for c in fake.calls)


@pytest.mark.skipif(LAYOUT is None, reason="needs an 8-core host")
def test_slice_refuses_to_run_when_the_cpuset_is_not_enforced(tmp_path, monkeypatch):
    _, workload = fake_slice(tmp_path, monkeypatch, "0-15", confine=False)
    with pytest.raises(RuntimeError, match="cpuset controller must be delegated"):
        workload.enter()


def test_stale_session_confinement_is_restored_from_the_lease(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        placement,
        "capture",
        lambda argv, **_: calls.append(list(argv)) or dict(returncode=0, stdout="", stderr=""),
    )
    lease = tmp_path / "lease.json"
    assert placement.restore_stale(lease) is None
    lease.write_text(
        json.dumps(
            dict(
                active=True,
                original={
                    "app.slice": {"AllowedCPUs": "0-15"},
                    "session.slice": {"AllowedCPUs": ""},
                },
            )
        )
    )
    assert placement.restore_stale(lease) == {"app.slice": 0, "session.slice": 0}
    assert [
        "systemctl",
        "--user",
        "set-property",
        "--runtime",
        "app.slice",
        "AllowedCPUs=0-15",
    ] in calls
    assert read_json(lease)["active"] is False
    assert placement.restore_stale(lease) is None


def test_process_launcher_confines_workers_to_the_pool(tmp_path):
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 2:
        pytest.skip("needs two CPUs")
    pool = cpus[-1:]
    module = tmp_path / "probe_module.py"
    module.write_text(PROBE.replace("os.sched_setaffinity(0, set(range(os.cpu_count())))\n", ""))
    launcher = placement.ProcessLauncher(
        pool_cpus=pool, working_directory=tmp_path, pythonpath=str(tmp_path)
    )
    handle = launcher.start("probe_module", tmp_path / "job.json", tmp_path / "log.txt", 1)
    assert placement.wait_for(lambda: handle.poll() is not None, 20)
    assert handle.poll() == 0, (tmp_path / "log.txt").read_text()
    assert read_json(tmp_path / "job.out")["before"] == pool
    assert launcher.kind == "process" and launcher.stale_units("x") == []


def test_process_handle_terminate_and_kill(tmp_path):
    module = tmp_path / "sleeper.py"
    module.write_text(
        "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(60)\n"
    )
    launcher = placement.ProcessLauncher(working_directory=tmp_path, pythonpath=str(tmp_path))
    handle = launcher.start("sleeper", tmp_path / "job.json", tmp_path / "log.txt", 1)
    time.sleep(0.5)
    handle.terminate()
    time.sleep(0.3)
    assert handle.poll() is None  # it ignores SIGTERM
    handle.kill()
    assert handle.poll() is not None


@needs_systemd
def test_systemd_slice_enforces_the_cpu_pool_without_a_quota(tmp_path):
    """Real kernel enforcement: a worker cannot widen its affinity beyond the pool."""
    workload = placement.WorkloadSlice(
        LAYOUT, MEMORY, tmp_path / "lease.json", confine_session=False, name="agencytbtest.slice"
    )
    module = tmp_path / "probe_module.py"
    module.write_text(PROBE)
    try:
        applied = workload.enter()
        assert applied["effective_cpus"] == LAYOUT.pool_arg
        launcher = placement.SystemdLauncher(
            slice_name=workload.name,
            pool_cpus=LAYOUT.pool_cpus,
            python=sys.executable,
            working_directory=tmp_path,
            pythonpath=str(tmp_path),
            runtime_max_seconds=60,
            unit_prefix=f"agencytbtest-{os.getpid()}",
        )
        handles = [
            launcher.start("probe_module", tmp_path / f"job{i}.json", tmp_path / f"log{i}.txt", i)
            for i in (1, 2)
        ]
        assert placement.wait_for(lambda: all(h.poll() is not None for h in handles), 60)
        assert [h.poll() for h in handles] == [0, 0], (tmp_path / "log1.txt").read_text()
        results = [read_json(tmp_path / f"job{i}.out") for i in (1, 2)]
    finally:
        workload.restore()
    for result in results:
        assert result["before"] == LAYOUT.pool_cpus
        assert result["after"] == LAYOUT.pool_cpus  # widening is refused by the cpuset
        assert result["cpuset"] == LAYOUT.pool_arg
        assert result["cpu_max"].startswith("max")  # no CPU quota
        assert "agencytbtest.slice" in result["cgroup"]
    # Both workers share one slice (one pool), in separate units.
    assert results[0]["cgroup"] != results[1]["cgroup"]
    assert Path(results[0]["cgroup"]).parent == Path(results[1]["cgroup"]).parent


# ------------------------------------------------------------- worker checks


class FakeSandbox:
    _name = "sandbox-test"

    def __init__(self, probe):
        self.probe = probe

    def exec(self, command):
        return json.dumps(self.probe), 0


def isolation_case(tmp_path, *, host=None, mounts=None, files=None, probe=None, cpus=(2, 3)):
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir(exist_ok=True)
    defaults = {
        "cpu.max": "max 100000",
        "cpuset.cpus.effective": "2-3",
        "memory.max": "4294967296",
        "memory.swap.max": "max",
    }
    for name, value in {**defaults, **(files or {})}.items():
        (cgroup / name).write_text(value + "\n")
    agent = tmp_path / "agent"
    agent.mkdir(exist_ok=True)
    job = dict(
        cpus=list(cpus),
        task=dict(environment_image_id="img-1"),
        sandbox=dict(
            harness_python_container="/opt/agency-python/bin/python3.12",
            agency_run_tmp_prefix="/tmp/agency-test/",
        ),
    )
    inspection = dict(
        Image="img-1",
        Config=dict(User=""),
        HostConfig={
            **dict(CpusetCpus="2-3", NanoCpus=0, CpuQuota=0, Memory=4294967296, Privileged=False),
            **(host or {}),
        },
        Mounts=mounts
        if mounts is not None
        else [
            dict(Source=str(agent / "logs"), Destination="/var/run/agency_logs", RW=True),
            dict(Source="/opt/shared", Destination="/opt/agency-python", RW=False),
            dict(Source="/tmp/agency-test/run/gw", Destination="/gw", RW=True),
        ],
    )
    probe = probe or dict(initial=list(cpus), after_widening=list(cpus))
    return job, FakeSandbox(probe), inspection, cgroup, agent


def check(monkeypatch, case):
    job, sandbox, inspection, cgroup, agent = case
    monkeypatch.setattr(worker.os, "sched_getaffinity", lambda pid: set(job["cpus"]))
    return worker.verify_isolation(job, sandbox, inspection, cgroup, agent)


def test_a_correctly_confined_container_passes(tmp_path, monkeypatch):
    report = check(monkeypatch, isolation_case(tmp_path))
    assert report["verified"] and report["problems"] == []
    assert report["cpu.max"] == "max 100000" and len(report["writable_mounts"]) == 2


@pytest.mark.parametrize(
    "change,phrase",
    [
        (dict(host=dict(NanoCpus=2_000_000_000)), "CPU quota"),
        (dict(host=dict(CpuQuota=50000)), "CPU quota"),
        (dict(files={"cpu.max": "200000 100000"}), "cpu.max"),
        (dict(host=dict(CpusetCpus="0-15")), "cpuset differs"),
        (dict(files={"cpuset.cpus.effective": "0-15"}), "effective cpuset"),
        (
            dict(probe=dict(initial=[2, 3], after_widening=list(range(16)))),
            "leave the experimental",
        ),
        (dict(host=dict(Privileged=True)), "privileged"),
        (
            dict(mounts=[dict(Source="/home/shared/cache", Destination="/cache", RW=True)]),
            "writable host mounts outside",
        ),
    ],
)
def test_isolation_violations_are_detected(tmp_path, monkeypatch, change, phrase):
    report = check(monkeypatch, isolation_case(tmp_path, **change))
    assert not report["verified"]
    assert any(phrase in problem for problem in report["problems"]), report["problems"]


def test_wrong_image_and_non_root_user_are_detected(tmp_path, monkeypatch):
    case = isolation_case(tmp_path)
    case[2]["Image"] = "another-image"
    case[2]["Config"]["User"] = "agent"
    problems = check(monkeypatch, case)["problems"]
    assert any("image ID differs" in p for p in problems)
    assert any("expected root" in p for p in problems)


def test_host_process_outside_the_pool_is_detected(tmp_path, monkeypatch):
    job, sandbox, inspection, cgroup, agent = isolation_case(tmp_path)
    monkeypatch.setattr(worker.os, "sched_getaffinity", lambda pid: {0, 2, 3})
    report = worker.verify_isolation(job, sandbox, inspection, cgroup, agent)
    assert any("host-side agent process" in p for p in report["problems"])


def test_agent_configuration_has_no_cpu_quota_and_reads_only_frozen_sources(tmp_path):
    job = dict(
        run_id="r",
        agent_id="A01",
        codex_binary="/x/codex",
        cpus=[2, 3, 10, 11],
        runtime_source=str(tmp_path / "runtime-source"),
        model=dict(
            provider="openai",
            model="gpt-6-luna",
            reasoning_effort="none",
            context_limit=196000,
            max_completion_tokens=4096,
            idle_timeout=180,
            stream_timeout=180,
        ),
        task=dict(environment=dict(image="docker.io/x/y@sha256:" + "a" * 64)),
        sandbox=dict(
            pids_limit=4096,
            run_label="agency.tb.run=r",
            task_label="agency.tb.task=r.01",
            memory="4096m",
            memory_swap="-1",
            cgroup_parent="agencytb.slice",
            cpuset_mems="0",
            network="default",
            harness_python_host="/env/harness",
            harness_python_container="/opt/p",
        ),
    )
    cfg = worker.build_config(job, tmp_path, "key", "http://gateway.invalid/v1")
    assert cfg.resources.idle_cpus is None  # never a per-agent CPU quota
    assert cfg.resources.idle_memory == "4096m"
    assert cfg.sandbox.cpuset_cpus == "2-3,10-11" and cfg.sandbox.gpu_passthrough is False
    assert cfg.sandbox.backend == "podman" and cfg.sandbox.persistent is False
    assert (
        "--pull=never" in cfg.sandbox.flags
        and "--cgroup-parent=agencytb.slice" in cfg.sandbox.flags
    )
    assert not any(
        flag.startswith(("--cpus", "--cpu-quota", "--cpu-shares", "--privileged"))
        for flag in cfg.sandbox.flags
    )
    # The sandbox imports a frozen copy of Agency, never the repository with its .env.
    package = cfg.sandbox.mounts["_agency_package"]
    assert package == (str(tmp_path / "runtime-source"), "/opt/agency_pkg", "ro")
    assert str(REPOSITORY) != package[0] and all(m[2] == "ro" for m in cfg.sandbox.mounts.values())
    # Existing Luna configuration through Agency's OpenAI-compatible (LiteLLM) backend.
    assert cfg.llm.provider == "openai" and cfg.llm.model == "gpt-6-luna"
    assert cfg.llm.max_completion_tokens == 4096 and cfg.agent.harness == "codex"
    assert cfg.harness_adapter.session_resume_id is None
    snapshot = json.dumps(cfg.safe_snapshot())
    assert '"key"' not in snapshot and "api_key" not in cfg.safe_snapshot()["llm"]
    none = worker.build_config(
        {**job, "sandbox": {**job["sandbox"], "network": "none"}},
        tmp_path,
        "key",
        "http://gateway.invalid/v1",
    )
    assert "--network=none" in none.sandbox.flags


def test_container_scope_resolution():
    base = Path("/sys/fs/cgroup/user.slice/agencytb.slice")
    scope = base / "libpod-abc.scope"
    assert worker.container_scope(scope / "container") == scope
    assert worker.container_scope(scope) == scope
    assert worker.container_scope(base / "other") == base / "other"
    assert worker.container_scope(None) is None
    assert worker.cgroup_of(os.getpid()) is not None and worker.cgroup_of(2**31 - 5) is None


def test_container_trace_reads_cumulative_counters(tmp_path):
    cgroup = tmp_path / "cg"
    cgroup.mkdir()
    (cgroup / "cpu.stat").write_text("usage_usec 500\nuser_usec 300\nsystem_usec 200\n")
    for name, total in (("cpu", 11), ("memory", 22), ("io", 33)):
        (cgroup / f"{name}.pressure").write_text(
            f"some avg10=0.00 avg60=0.00 avg300=0.00 total={total}\nfull avg10=0.00 total=1\n"
        )
    (cgroup / "memory.current").write_text("4096\n")
    trace = worker.ContainerTrace(cgroup, tmp_path / "series.jsonl", 0.02)
    row = trace.read()
    assert row[1:] == [500, 300, 200, 11, 22, 33, 4096]
    trace.start()
    time.sleep(0.15)
    (cgroup / "cpu.stat").unlink()  # the container stopped: rows simply end
    time.sleep(0.05)
    trace.halt()
    lines = (tmp_path / "series.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["columns"][0] == "t_ns" and len(lines) >= 3
    assert trace.read() is None
