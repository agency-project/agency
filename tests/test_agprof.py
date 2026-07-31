"""Tests for environment-controlled profiler lifecycle scopes."""

import json
import time
from contextlib import contextmanager

import pytest

from agency.profiler import agprof


def test_complete_summary_includes_per_process_workload_and_gpu_metrics(monkeypatch):
    second = 1_000_000_000
    mebibyte = 2**20
    records = [
        (7, "turn0", 0, 100_000_000, 25_000_000, 5_000_000),
    ]
    monkeypatch.setattr(
        agprof,
        "_process_info",
        {
            "101-10": {
                "identity": "101-10",
                "pid": 101,
                "comm": "python",
                "cmdline": "python benchmark.py",
                "sandbox": None,
                "cgroup": "/agprof.slice/run.scope",
                "display_name": "python (PID 101)",
                "trace_pid": 101,
                "first_seen_ns": 0,
            }
        },
    )
    samples = [
        (0, "workload:cpu_us", 1_000_000.0),
        (0, "workload:memory_mb", 100.0),
        (0, "workload:io_r", 0.0),
        (0, "proc:101-10:cpu_s", 1.0),
        (0, "proc:101-10:rss_mb", 30.0),
        (0, "proc:101-10:vms_mb", 90.0),
        (0, "proc:101-10:io_r", 0.0),
        (0, "cg:writer:cpu_us", 0.0),
        (0, "gpu0:util_pct", 10.0),
        (second, "workload:cpu_us", 1_500_000.0),
        (second, "workload:memory_mb", 120.0),
        (second, "workload:io_r", 2.0 * mebibyte),
        (second, "proc:101-10:cpu_s", 1.2),
        (second, "proc:101-10:rss_mb", 40.0),
        (second, "proc:101-10:vms_mb", 100.0),
        (second, "proc:101-10:io_r", 1.0 * mebibyte),
        (second, "cg:writer:cpu_us", 250_000.0),
        (second, "gpu0:util_pct", 30.0),
    ]

    summary = agprof._build_run_summary(
        records,
        samples,
        [(0, 0, 2 * second, "writer")],
        started_ns=0,
        ended_ns=2 * second,
        sample_hz=10.0,
        sample_gpu=True,
        gpu_sampling_available=True,
    )

    assert summary["duration_ms"] == 2000.0
    span = summary["span_metrics"][0]
    assert span["label"] == "turn"
    assert span["calls"] == 1
    assert span["started"] == 1
    assert span["wall_ms"] == 100.0
    assert span["cpu_ms"] == 25.0
    assert span["runqueue_ms"] == 5.0
    assert span["blocked_ms"] == 70.0
    assert span["cpu_percent"] == 25.0
    assert span["p50_ms"] == 100.0
    assert span["p95_ms"] == 100.0
    resources = {row["name"]: row for row in summary["resource_metrics"]}
    assert resources["workload_total:cpu_pct"]["mean"] == 50.0
    assert resources["workload_total:memory_mb"]["mean"] == 110.0
    assert resources["workload_total:io_read_mb_s"]["mean"] == 2.0
    assert resources["workload_total:io_read_mb_s"]["total"] == 2.0
    assert resources["process:101-10:cpu_pct"]["mean"] == 20.0
    assert resources["process:101-10:rss_mb"]["mean"] == 35.0
    assert resources["process:101-10:vms_mb"]["max"] == 100.0
    assert resources["process:101-10:io_read_mb_s"]["total"] == 1.0
    assert summary["process_metrics"][0]["pid"] == 101
    assert summary["process_metrics"][0]["name"] == "python"
    assert summary["workload_metrics"]["cpu_average_percent"] == 50.0
    assert resources["sandbox:writer:cpu_pct"]["mean"] == 25.0
    assert resources["sandbox:writer:cpu_pct"]["total"] == 0.25
    assert resources["gpu0:util_pct"]["mean"] == 20.0
    assert summary["gpu_lease_metrics"] == [
        {
            "gpu_id": 0,
            "label": "writer",
            "leases": 1,
            "total_ms": 2000.0,
            "mean_ms": 2000.0,
            "max_ms": 2000.0,
        }
    ]


def test_stop_writes_json_and_markdown_summaries(monkeypatch, tmp_path):
    class FakeProfiler:
        def stop(self):
            pass

    monkeypatch.setattr(agprof, "_session", object())
    monkeypatch.setattr(agprof, "_profiler", FakeProfiler())
    monkeypatch.setattr(agprof, "_out_dir", tmp_path)
    monkeypatch.setattr(agprof, "_sampler", None)
    monkeypatch.setattr(agprof, "_session_started_ns", time.perf_counter_ns() - 1_000_000)
    monkeypatch.setattr(agprof, "_session_sample_hz", 0.0)
    monkeypatch.setattr(agprof, "_session_sample_gpu", False)
    monkeypatch.setattr(
        agprof,
        "_records",
        [(7, "stage:work", 0, 1_000_000, 500_000, 100_000)],
    )
    monkeypatch.setattr(agprof, "_samples", [])
    monkeypatch.setattr(agprof, "_leases", [])
    monkeypatch.setattr(agprof, "_leases_open", {})
    monkeypatch.setattr(
        agprof,
        "_inject_trace_args",
        lambda out_dir, records, interrupted_spans: None,
    )

    agprof.stop()

    machine_summary = json.loads((tmp_path / "summary.json").read_text())
    human_summary = (tmp_path / "summary.md").read_text()
    assert machine_summary["schema_version"] == 4
    assert "workload_metrics" in machine_summary
    assert machine_summary["process_metrics"] == []
    assert "host_metrics" not in machine_summary
    assert machine_summary["span_metrics"][0]["label"] == "stage:work"
    assert "# agprof summary" in human_summary
    assert "| stage:work |" in human_summary
    assert agprof.summary_metrics() == machine_summary


def test_derived_rollups_include_outcomes_percentiles_tokens_energy_and_interruptions():
    second = 1_000_000_000
    records = [
        (1, "run0:writer:a", 0, second, 100_000_000, 0, {"outcome": "success"}),
        (2, "run1:writer:b", 0, 2 * second, 100_000_000, 0, {"outcome": "failure"}),
        (
            1,
            "llm:attempt[0]",
            0,
            400_000_000,
            10_000_000,
            0,
            {
                "outcome": "success",
                "ttft_ms": 100.0,
                "generation_ms": 300.0,
                "input_tokens": 20,
                "output_tokens": 30,
            },
        ),
        (
            2,
            "llm:attempt[0]",
            0,
            600_000_000,
            10_000_000,
            0,
            {"outcome": "failure", "retrying": True},
        ),
        (
            2,
            "llm:attempt[1]",
            0,
            800_000_000,
            10_000_000,
            0,
            {
                "outcome": "success",
                "ttft_ms": 200.0,
                "generation_ms": 600.0,
                "input_tokens": 40,
                "output_tokens": 60,
            },
        ),
        (1, "tool:write", 0, 200_000_000, 1_000_000, 0, {"outcome": "success"}),
        (2, "tool:read", 0, 300_000_000, 1_000_000, 0, {"outcome": "failure"}),
    ]
    samples = [
        (0, "gpu0:power_w", 10.0),
        (second, "gpu0:power_w", 20.0),
    ]
    interrupted = [
        {
            "thread_id": 3,
            "label": "run2:writer:c",
            "started_ns": 500_000_000,
            "duration_ms": 1500.0,
            "outcome": "interrupted",
        },
        {
            "thread_id": 3,
            "label": "tool:ask_human",
            "started_ns": 600_000_000,
            "duration_ms": 1400.0,
            "outcome": "interrupted",
        },
    ]

    summary = agprof._build_run_summary(
        records,
        samples,
        [],
        interrupted_spans=interrupted,
        started_ns=0,
        ended_ns=2 * second,
        sample_hz=10.0,
        sample_gpu=True,
        gpu_sampling_available=True,
    )

    assert summary["run_metrics"]["started"] == 3
    assert summary["run_metrics"]["completed"] == 2
    assert summary["run_metrics"]["failed"] == 1
    assert summary["run_metrics"]["interrupted"] == 1
    assert summary["run_metrics"]["completed_per_second"] == 1.0
    assert summary["run_metrics"]["p50_ms"] == 1500.0
    assert summary["llm_metrics"]["calls"] == 2
    assert summary["llm_metrics"]["retries"] == 1
    assert summary["llm_metrics"]["input_tokens"] == 60
    assert summary["llm_metrics"]["output_tokens"] == 90
    assert summary["llm_metrics"]["ttft"]["p50_ms"] == 150.0
    assert summary["llm_metrics"]["output_tokens_per_second"] == 100.0
    assert summary["tool_metrics"]["started"] == 3
    assert summary["tool_metrics"]["failed"] == 1
    assert summary["tool_metrics"]["interrupted"] == 1
    assert summary["gpu_metrics"][0]["energy_j"] == 15.0
    assert summary["incomplete_spans"][0]["outcome"] == "interrupted"
    markdown = agprof._render_summary_markdown(summary)
    assert "### Tool outcomes" in markdown
    assert "| read | 1/1 | 0 | 1 | 0 |" in markdown
    assert "| ask_human | 0/1 | 0 | 0 | 1 | n/a | n/a |" in markdown
    assert "## Per-process metrics" in markdown
    assert "## Workload aggregate" in markdown
    assert "Host metrics" not in markdown


def test_stop_snapshots_open_spans_as_interrupted(monkeypatch, tmp_path):
    class FakeRecordFunction:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class FakeProfiler:
        def stop(self):
            pass

    monkeypatch.setattr(agprof, "_session", object())
    monkeypatch.setattr(agprof, "_profiler", FakeProfiler())
    monkeypatch.setattr(agprof, "_out_dir", tmp_path)
    monkeypatch.setattr(agprof, "_sampler", None)
    monkeypatch.setattr(agprof, "_session_started_ns", time.perf_counter_ns() - 1_000_000)
    monkeypatch.setattr(agprof, "_session_sample_hz", 0.0)
    monkeypatch.setattr(agprof, "_session_sample_gpu", False)
    monkeypatch.setattr(agprof, "_records", [])
    monkeypatch.setattr(agprof, "_samples", [])
    monkeypatch.setattr(agprof, "_leases", [])
    monkeypatch.setattr(agprof, "_leases_open", {})
    monkeypatch.setattr(agprof, "_open_spans", {})
    monkeypatch.setattr(agprof, "_interrupted_spans", [])
    monkeypatch.setattr(
        agprof,
        "_inject_trace_args",
        lambda out_dir, records, interrupted_spans: None,
    )

    active = agprof._TimedSpan(FakeRecordFunction, "run0:test:agent")
    active.__enter__()
    agprof.stop()
    active.__exit__(None, None, None)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["run_metrics"]["started"] == 1
    assert summary["run_metrics"]["completed"] == 0
    assert summary["run_metrics"]["interrupted"] == 1
    assert summary["incomplete_spans"][0]["label"] == "run0:test:agent"
    assert not agprof._records


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "workload"),
        ("", "workload"),
        ("workload", "workload"),
        ("WORKLOAD", "workload"),
        ("invalid", "workload"),
        ("process", "process"),
        (" PROCESS ", "process"),
    ],
)
def test_profile_scope(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("AGENCY_PROFILE_SCOPE", raising=False)
    else:
        monkeypatch.setenv("AGENCY_PROFILE_SCOPE", value)

    assert agprof.profile_scope() == expected


def test_non_linux_environment_profiling_fails_before_cgroup_or_profiler(monkeypatch):
    monkeypatch.setenv("AGENCY_PROFILE", "1")
    monkeypatch.setattr(agprof.sys, "platform", "darwin")
    monkeypatch.setattr(
        agprof,
        "_ensure_environment_cgroup",
        lambda: pytest.fail("must reject before cgroup setup"),
    )
    monkeypatch.setattr(
        agprof,
        "_maybe_autostart",
        lambda: pytest.fail("must reject before profiler startup"),
    )

    with pytest.raises(RuntimeError, match="profiling is Linux-only"):
        agprof._initialize_environment_profiling()


def test_environment_cgroup_reexec_wraps_original_command(monkeypatch):
    captured = {}
    monkeypatch.delenv("AGENCY_PROFILE_CGROUP", raising=False)
    monkeypatch.setattr(agprof.sys, "orig_argv", ["/venv/bin/python", "bench.py", "--quick"])
    monkeypatch.setattr(agprof.os, "getuid", lambda: 1234)
    monkeypatch.setattr(agprof.os, "getgid", lambda: 5678)
    monkeypatch.setenv("USER", "benchmark")
    monkeypatch.setenv("HOME", "/home/benchmark")
    monkeypatch.setattr(agprof.uuid, "uuid4", lambda: type("U", (), {"hex": "abcdef012345"})())
    monkeypatch.setattr(
        agprof.os,
        "execvp",
        lambda executable, argv: captured.update(executable=executable, argv=argv),
    )

    agprof._ensure_environment_cgroup()

    assert captured["executable"] == "sudo"
    command = captured["argv"]
    slice_arg = next(arg for arg in command if arg.startswith("--slice="))
    slice_name = slice_arg.split("=", 1)[1]
    assert slice_name.startswith("agprof-")
    assert f"AGENCY_PROFILE_CGROUP_PARENT={slice_name}" in command
    assert any(
        arg == f"AGENCY_PROFILE_CGROUP=/sys/fs/cgroup/agprof.slice/{slice_name}" for arg in command
    )
    assert command[-3:] == ["/venv/bin/python", "bench.py", "--quick"]


def test_workload_cgroup_sampler_reads_aggregate_cpu_memory_and_io(tmp_path, monkeypatch):
    (tmp_path / "cpu.stat").write_text("usage_usec 2500000\nuser_usec 2000000\n")
    (tmp_path / "memory.current").write_text(str(64 * 2**20))
    (tmp_path / "io.stat").write_text("8:0 rbytes=1048576 wbytes=2097152 rios=1 wios=2\n")
    sampler = object.__new__(agprof._Sampler)
    sampler._process_cgroup = tmp_path
    monkeypatch.setattr(agprof, "_samples", [])

    sampler._tick_workload_cgroup(99)

    assert agprof._samples == [
        (99, "workload:cpu_us", 2_500_000.0),
        (99, "workload:memory_mb", 64.0),
        (99, "workload:io_r", 1_048_576.0),
        (99, "workload:io_w", 2_097_152.0),
    ]


def _write_fake_proc(proc_root, pid, *, start_ticks, cpu_ticks, comm="python", io=True):
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "S",
        "1",
        "1",
        "1",
        "0",
        "-1",
        "0",
        "0",
        "0",
        "0",
        "0",
        str(cpu_ticks),
        "0",
        "0",
        "0",
        "20",
        "0",
        "1",
        "0",
        str(start_ticks),
        str(100 * 2**20),
        "256",
    ]
    (process_dir / "stat").write_text(f"{pid} ({comm}) " + " ".join(fields))
    (process_dir / "cmdline").write_bytes(f"{comm}\0worker.py\0".encode())
    if io:
        (process_dir / "io").write_text("read_bytes: 1048576\nwrite_bytes: 2097152\n")


def test_sampler_discovers_recursive_cgroup_pids_and_samples_each_process(tmp_path, monkeypatch):
    cgroup = tmp_path / "cgroup"
    child_cgroup = cgroup / "docker.scope"
    child_cgroup.mkdir(parents=True)
    (cgroup / "cgroup.procs").write_text("101\n")
    (child_cgroup / "cgroup.procs").write_text("202\n")
    proc_root = tmp_path / "proc"
    _write_fake_proc(proc_root, 101, start_ticks=10, cpu_ticks=150)
    _write_fake_proc(proc_root, 202, start_ticks=20, cpu_ticks=75, comm="container-python")

    sampler = object.__new__(agprof._Sampler)
    sampler._process_cgroup = cgroup
    sampler._proc_root = proc_root
    sampler._clock_ticks = 100
    sampler._page_mb = 4096 / 2**20
    monkeypatch.setattr(agprof, "_samples", [])
    monkeypatch.setattr(agprof, "_process_info", {})
    monkeypatch.setattr(agprof, "_cg_registry", {"worker": str(child_cgroup)})

    sampler._tick_processes(99)

    assert set(agprof._process_info) == {"101-10", "202-20"}
    assert agprof._process_info["101-10"]["display_name"] == "python (PID 101)"
    assert agprof._process_info["202-20"]["display_name"] == ("container-python [worker] (PID 202)")
    assert (99, "proc:101-10:cpu_s", 1.5) in agprof._samples
    assert (99, "proc:101-10:rss_mb", 1.0) in agprof._samples
    assert (99, "proc:101-10:vms_mb", 100.0) in agprof._samples
    assert (99, "proc:202-20:io_w", 2_097_152.0) in agprof._samples


def test_sampler_skips_pid_that_exits_between_discovery_and_proc_read(tmp_path, monkeypatch):
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "cgroup.procs").write_text("303\n")
    sampler = object.__new__(agprof._Sampler)
    sampler._process_cgroup = cgroup
    sampler._proc_root = tmp_path / "proc"
    sampler._clock_ticks = 100
    sampler._page_mb = 4096 / 2**20
    monkeypatch.setattr(agprof, "_samples", [])
    monkeypatch.setattr(agprof, "_process_info", {})

    sampler._tick_processes(99)

    assert agprof._samples == []
    assert agprof._process_info == {}


def test_sampler_keeps_pid_reuse_as_two_process_identities(tmp_path, monkeypatch):
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "cgroup.procs").write_text("404\n")
    proc_root = tmp_path / "proc"
    sampler = object.__new__(agprof._Sampler)
    sampler._process_cgroup = cgroup
    sampler._proc_root = proc_root
    sampler._clock_ticks = 100
    sampler._page_mb = 4096 / 2**20
    monkeypatch.setattr(agprof, "_samples", [])
    monkeypatch.setattr(agprof, "_process_info", {})

    _write_fake_proc(proc_root, 404, start_ticks=10, cpu_ticks=10)
    sampler._tick_processes(1)
    _write_fake_proc(proc_root, 404, start_ticks=20, cpu_ticks=5)
    sampler._tick_processes(2)

    assert set(agprof._process_info) == {"404-10", "404-20"}
    assert agprof._process_info["404-10"]["trace_pid"] == 404
    assert agprof._process_info["404-20"]["trace_pid"] >= 1_000_000_000


def test_sampler_updates_container_label_when_registry_arrives_late(tmp_path, monkeypatch):
    cgroup = tmp_path / "cgroup"
    container_cgroup = cgroup / "docker.scope"
    container_cgroup.mkdir(parents=True)
    (container_cgroup / "cgroup.procs").write_text("505\n")
    proc_root = tmp_path / "proc"
    _write_fake_proc(proc_root, 505, start_ticks=50, cpu_ticks=10)
    sampler = object.__new__(agprof._Sampler)
    sampler._process_cgroup = cgroup
    sampler._proc_root = proc_root
    sampler._clock_ticks = 100
    sampler._page_mb = 4096 / 2**20
    monkeypatch.setattr(agprof, "_samples", [])
    monkeypatch.setattr(agprof, "_process_info", {})
    monkeypatch.setattr(agprof, "_cg_registry", {})

    sampler._tick_processes(1)
    assert agprof._process_info["505-50"]["sandbox"] is None
    agprof._cg_registry["worker"] = str(container_cgroup)
    sampler._tick_processes(2)

    assert agprof._process_info["505-50"]["sandbox"] == "worker"
    assert agprof._process_info["505-50"]["display_name"] == "python [worker] (PID 505)"


def test_trace_injection_creates_independent_tensorboard_process_groups(monkeypatch):
    second = 1_000_000_000
    monkeypatch.setattr(agprof, "_clock_mark_ns", 0)
    monkeypatch.setattr(agprof, "_leases", [])
    monkeypatch.setattr(
        agprof,
        "_process_info",
        {
            "101-10": {
                "pid": 101,
                "comm": "python",
                "cmdline": "python main.py",
                "cgroup": "/agprof/run.scope",
                "display_name": "python (PID 101)",
                "trace_pid": 101,
                "first_seen_ns": 0,
            },
            "202-20": {
                "pid": 202,
                "comm": "python",
                "cmdline": "python child.py",
                "cgroup": "/agprof/run.scope",
                "display_name": "python (PID 202)",
                "trace_pid": 202,
                "first_seen_ns": 1,
            },
        },
    )
    monkeypatch.setattr(
        agprof,
        "_samples",
        [
            (0, "proc:101-10:cpu_s", 1.0),
            (0, "proc:101-10:rss_mb", 10.0),
            (0, "proc:202-20:cpu_s", 2.0),
            (0, "proc:202-20:rss_mb", 20.0),
            (second, "proc:101-10:cpu_s", 1.5),
            (second, "proc:101-10:rss_mb", 11.0),
            (second, "proc:202-20:cpu_s", 2.25),
            (second, "proc:202-20:rss_mb", 21.0),
        ],
    )
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "pid": 101,
                "tid": 1,
                "ts": 1000.0,
                "dur": 1.0,
                "name": "agprof:clock_sync",
            }
        ]
    }

    counters, _leases = agprof._inject_timelines(trace)

    process_names = {
        event["pid"]: event["args"]["name"]
        for event in trace["traceEvents"]
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    assert process_names[101] == "python (PID 101)"
    assert process_names[202] == "python (PID 202)"
    child_counters = {
        event["name"]
        for event in trace["traceEvents"]
        if event.get("ph") == "C" and event["pid"] == 202
    }
    assert child_counters == {"cpu_percent", "rss_mb"}
    assert counters == 6


def test_container_cgroup_parent_accepts_only_profiler_slice(monkeypatch):
    monkeypatch.setenv("AGENCY_PROFILE_CGROUP_PARENT", "agprof-12ab.slice")
    assert agprof.container_cgroup_parent() == "agprof-12ab.slice"
    monkeypatch.setenv("AGENCY_PROFILE_CGROUP_PARENT", "../../system.slice")
    assert agprof.container_cgroup_parent() is None


def test_workload_scope_owns_session_exactly_around_workload(monkeypatch):
    events = []
    profiler = object()
    monkeypatch.setenv("AGENCY_PROFILE", "1")
    monkeypatch.delenv("AGENCY_PROFILE_SCOPE", raising=False)
    monkeypatch.setenv("AGENCY_PROFILE_DIR", "custom-trace")
    monkeypatch.setattr(agprof, "enabled", lambda: False)
    monkeypatch.setattr(
        agprof,
        "start",
        lambda out_dir: events.append(("start", out_dir)) or profiler,
    )
    monkeypatch.setattr(agprof, "stop", lambda: events.append(("stop", None)))

    with agprof.workload() as active:
        events.append(("workload", None))
        assert active is profiler

    assert events == [
        ("start", "custom-trace"),
        ("workload", None),
        ("stop", None),
    ]


def test_workload_scope_stops_session_when_workload_raises(monkeypatch):
    events = []
    monkeypatch.setenv("AGENCY_PROFILE", "true")
    monkeypatch.setenv("AGENCY_PROFILE_SCOPE", "workload")
    monkeypatch.setattr(agprof, "enabled", lambda: False)
    monkeypatch.setattr(agprof, "start", lambda out_dir: events.append("start"))
    monkeypatch.setattr(agprof, "stop", lambda: events.append("stop"))

    with pytest.raises(RuntimeError, match="boom"):
        with agprof.workload():
            events.append("workload")
            raise RuntimeError("boom")

    assert events == ["start", "workload", "stop"]


@pytest.mark.parametrize(
    ("profile_value", "scope"),
    [
        ("", "workload"),
        ("0", "workload"),
        ("1", "process"),
    ],
)
def test_workload_context_does_not_own_other_lifecycles(monkeypatch, profile_value, scope):
    monkeypatch.setenv("AGENCY_PROFILE", profile_value)
    monkeypatch.setenv("AGENCY_PROFILE_SCOPE", scope)
    monkeypatch.setattr(
        agprof,
        "start",
        lambda out_dir: pytest.fail("workload context must not start profiling"),
    )
    monkeypatch.setattr(
        agprof,
        "stop",
        lambda: pytest.fail("workload context must not stop profiling"),
    )

    with agprof.workload():
        pass


def test_workload_context_preserves_explicit_active_session(monkeypatch):
    existing = object()
    monkeypatch.setenv("AGENCY_PROFILE", "1")
    monkeypatch.setenv("AGENCY_PROFILE_SCOPE", "workload")
    monkeypatch.setattr(agprof, "enabled", lambda: True)
    monkeypatch.setattr(agprof, "_profiler", existing)
    monkeypatch.setattr(
        agprof,
        "start",
        lambda out_dir: pytest.fail("an explicit session is already active"),
    )
    monkeypatch.setattr(
        agprof,
        "stop",
        lambda: pytest.fail("must not stop an explicit session"),
    )

    with agprof.workload() as active:
        assert active is existing


def test_process_scope_is_the_only_environment_autostart(monkeypatch):
    events = []
    monkeypatch.setenv("AGENCY_PROFILE", "1")
    monkeypatch.setenv("AGENCY_PROFILE_SCOPE", "process")
    monkeypatch.setenv("AGENCY_PROFILE_DIR", "process-trace")
    monkeypatch.setattr(agprof, "start", lambda out_dir: events.append(("start", out_dir)))
    monkeypatch.setattr(agprof.atexit, "register", lambda fn: events.append(("register", fn)))

    agprof._maybe_autostart()

    assert events == [("start", "process-trace"), ("register", agprof.stop)]


@pytest.mark.parametrize("scope", [None, "workload", "invalid"])
def test_default_and_invalid_scopes_do_not_autostart(monkeypatch, scope):
    monkeypatch.setenv("AGENCY_PROFILE", "1")
    if scope is None:
        monkeypatch.delenv("AGENCY_PROFILE_SCOPE", raising=False)
    else:
        monkeypatch.setenv("AGENCY_PROFILE_SCOPE", scope)
    monkeypatch.setattr(
        agprof,
        "start",
        lambda out_dir: pytest.fail("non-process scope must not autostart"),
    )
    monkeypatch.setattr(
        agprof.atexit,
        "register",
        lambda fn: pytest.fail("non-process scope must not register process cleanup"),
    )

    agprof._maybe_autostart()


def test_webui_marks_only_supplied_function_as_workload(monkeypatch, tmp_path):
    import agency.agwebui as agwebui_module

    events = []

    @contextmanager
    def workload():
        events.append("profile-start")
        try:
            yield
        finally:
            events.append("profile-stop")

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self):
            events.append("server-stop")

        def wait(self, timeout):
            return 0

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def setsockopt(self, *args):
            pass

        def connect_ex(self, address):
            return 1

    monkeypatch.setattr(agwebui_module.agprof, "workload", workload)
    monkeypatch.setattr(agwebui_module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(agwebui_module.urllib.request, "urlopen", lambda *args, **kwargs: None)
    monkeypatch.setattr("socket.socket", lambda *args, **kwargs: FakeSocket())
    monkeypatch.setattr(agwebui_module.atexit, "register", lambda fn: None)

    def fn():
        events.append("workload")

    agwebui_module.agwebui.run(fn, run_dir=tmp_path, port=17860, linger=False)

    assert events[:3] == ["profile-start", "workload", "profile-stop"]
