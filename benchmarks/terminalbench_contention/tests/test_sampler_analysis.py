"""Resource sampling parsers and the trajectory / cross-resource analysis."""

import json
import os
import time
from pathlib import Path

import pytest

from benchmarks.terminalbench_contention import analysis, sampler
from benchmarks.terminalbench_contention.util import read_jsonl

S = 1_000_000_000  # one second in ns


# ------------------------------------------------------------------- parsers


def test_pressure_parser():
    text = (
        "some avg10=1.50 avg60=0.40 avg300=0.10 total=123456\n"
        "full avg10=0.25 avg60=0.00 avg300=0.00 total=789\n"
    )
    assert sampler.parse_pressure(text) == dict(
        some_usec=123456, some_avg10=1.5, full_usec=789, full_avg10=0.25
    )
    assert sampler.parse_pressure("some avg10=0.00 avg60=0.00 avg300=0.00 total=5\n") == dict(
        some_usec=5, some_avg10=0.0
    )
    assert sampler.parse_pressure("garbage") == {}


def test_keyed_and_io_and_meminfo_parsers():
    assert sampler.parse_flat_keyed("usage_usec 10\nuser_usec 7\nbad line here\nx y\n") == dict(
        usage_usec=10, user_usec=7
    )
    assert sampler.parse_flat_keyed("anon 1\nfile 2\n", {"file"}) == dict(file=2)
    io = sampler.parse_io_stat(
        "259:0 rbytes=100 wbytes=50 rios=3 wios=2 dbytes=0\n8:0 rbytes=1 wbytes=2 rios=1 wios=1\n"
    )
    assert io == dict(rbytes=101, wbytes=52, rios=4, wios=3)
    assert sampler.parse_meminfo(
        "MemTotal:  1000 kB\nMemAvailable: 400 kB\nFoo: 1 kB\n", {"MemTotal", "MemAvailable"}
    ) == dict(MemTotal=1024000, MemAvailable=409600)


def test_proc_stat_is_aggregated_per_cpu_set():
    text = (
        "cpu  100 0 100 800 0 0 0 0 0 0\n"
        "cpu0 10 0 10 80 5 0 0 0 0 0\n"
        "cpu1 20 1 20 60 0 0 0 0 0 0\n"
        "cpu2 30 0 30 40 0 1 2 3 0 0\n"
        "ctxt 4242\nprocesses 7\nprocs_running 3\nprocs_blocked 1\n"
    )
    stat = sampler.parse_proc_stat(text, dict(pool={1, 2}, reserved={0}))
    assert stat["pool"] == dict(
        user=50, nice=1, system=50, idle=100, iowait=0, irq=1, softirq=2, steal=3
    )
    assert stat["reserved"]["user"] == 10 and stat["reserved"]["iowait"] == 5
    assert stat["ctxt"] == 4242 and stat["procs_running"] == 3 and stat["procs_blocked"] == 1


def test_schedstat_and_diskstats_parsers():
    schedstat = (
        "version 17\ntimestamp 1\n"
        "cpu0 0 0 0 0 0 0 100 900 5\n"
        "cpu1 0 0 0 0 0 0 200 1100 7\n"
        "domain0 03 0 0\n"
    )
    assert sampler.parse_schedstat(schedstat, {1}) == dict(run_delay_ns=1100, timeslices=7)
    assert sampler.parse_schedstat(schedstat, {0, 1}) == dict(run_delay_ns=2000, timeslices=12)
    assert sampler.parse_schedstat(schedstat, {9}) is None
    disk = " 259 0 nvme0n1 10 0 2000 5 20 0 4000 9 1 77 99 0 0 0 0\n"
    parsed = sampler.parse_diskstats(disk, "nvme0n1")
    assert parsed["read_bytes"] == 2000 * 512 and parsed["write_bytes"] == 4000 * 512
    assert parsed["io_ms"] == 77 and sampler.parse_diskstats(disk, "sda") is None


def test_own_cgroup_can_be_read_and_threads_scanned():
    line = Path("/proc/self/cgroup").read_text().strip().splitlines()[-1]
    cgroup = sampler.CGROUP_ROOT / line.split("::", 1)[1].lstrip("/")
    record = sampler.read_cgroup(cgroup)
    if record is None:
        pytest.skip("cgroup v2 cpu.stat is not readable here")
    assert record["cpu"]["usage_usec"] > 0 and "current" in record["memory"]
    threads = sampler.scan_threads(cgroup)
    assert threads["threads"] >= 1 and threads["run_ns"] > 0
    io = sampler.scan_process_io(cgroup)
    assert io["processes"] >= 1 and io["source"] == "proc.pid.io"
    assert sampler.read_cgroup(cgroup / "does-not-exist") is None
    assert sampler.cgroup_threads(cgroup / "does-not-exist") == []


def test_sampler_writes_time_series_and_reports_capabilities(tmp_path):
    output = tmp_path / "metrics" / "samples.jsonl"
    cpus = sorted(os.sched_getaffinity(0))
    instance = sampler.ResourceSampler(
        output,
        interval=0.05,
        pool_cpus=cpus[1:] or cpus,
        reserved_cpus=cpus[:1],
        api_snapshot=lambda: dict(utilization=0.5, waiters=2, totals=dict(x=1)),
        active=lambda: 3,
        disk_path=tmp_path,
    )
    instance.start()
    time.sleep(0.4)
    report = instance.stop()
    rows = read_jsonl(output, strict=True)
    assert report["samples"] == len(rows) >= 4 and report["errors"] == 0
    assert all(b["t"] > a["t"] for a, b in zip(rows, rows[1:]))
    first = rows[0]
    assert first["active"] == 3 and first["api"] == dict(utilization=0.5, waiters=2)
    assert set(first["host"]["cpu"]["pool"]) >= {"user", "idle", "iowait"}
    assert "MemAvailable" in first["host"]["memory"]
    capabilities = instance.capabilities
    assert capabilities["units"] and "not_measured" in capabilities
    assert isinstance(capabilities["host_psi"]["cpu"], bool)
    json.dumps(capabilities)


def test_a_broken_api_snapshot_does_not_stop_sampling(tmp_path):
    def broken():
        raise RuntimeError("ledger corrupted")

    instance = sampler.ResourceSampler(
        tmp_path / "s.jsonl", interval=0.05, pool_cpus=[0], reserved_cpus=[], api_snapshot=broken
    )
    record = instance.sample()
    assert record["api"] == dict(error="RuntimeError") and "host" in record


# ------------------------------------------------------------------ timeline


def test_overlapping_intervals_resolve_by_priority_and_gaps_are_inferred():
    intervals = [
        ("api_wait", 1 * S, 3 * S),
        ("model_wait", 3 * S, 4 * S),
        ("model_stream", 4 * S, 6 * S),
        ("tool", 5 * S, 8 * S),
    ]
    segments = analysis.resolve_timeline(intervals, 0, 10 * S)
    assert segments == [
        ("other", 0, 1 * S),
        ("api_wait", 1 * S, 3 * S),
        ("model_wait", 3 * S, 4 * S),
        ("model_stream", 4 * S, 5 * S),
        ("tool", 5 * S, 8 * S),
        ("other", 8 * S, 10 * S),
    ]
    seconds = analysis.state_seconds(segments)
    assert seconds == dict(
        tool=3, model_stream=1, model_wait=1, api_wait=2, checkpoint=0, harness_startup=0, other=3
    )
    assert sum(seconds.values()) == 10  # states partition the execution window
    assert analysis.STATE_SOURCE["other"].startswith("inferred")
    assert all(analysis.STATE_SOURCE[s].startswith("measured") for s in analysis.STATES[:-1])


def test_intervals_are_clipped_to_the_execution_window():
    segments = analysis.resolve_timeline(
        [("tool", -5 * S, 2 * S), ("tool", 9 * S, 99 * S)], 0, 10 * S
    )
    assert segments == [("tool", 0, 2 * S), ("other", 2 * S, 9 * S), ("tool", 9 * S, 10 * S)]
    assert analysis.resolve_timeline([("tool", 0, S)], 5, 5) == []
    assert analysis.overlap_seconds(segments, 1 * S, 3 * S) == dict(tool=1.0, other=1.0)


def test_request_records_become_wait_and_stream_intervals():
    requests = [
        dict(ready_ns=1 * S, admitted_ns=3 * S, first_chunk_ns=4 * S, completed_ns=6 * S),
        dict(ready_ns=7 * S, admitted_ns=7 * S, first_chunk_ns=None, completed_ns=8 * S),
        dict(ready_ns=9 * S, admitted_ns=None),
    ]
    assert analysis.request_intervals(requests) == [
        ("api_wait", 1 * S, 3 * S),
        ("model_wait", 3 * S, 4 * S),
        ("model_stream", 4 * S, 6 * S),
        ("model_wait", 7 * S, 8 * S),
    ]


def test_series_delta_interpolates_cumulative_counters():
    rows = [
        [0, 0, 0, 0, 0, 0, 0, 1],
        [1 * S, 1_000_000, 0, 0, 100, 0, 0, 1],
        [2 * S, 3_000_000, 0, 0, 100, 0, 0, 1],
    ]
    assert analysis.series_delta(rows, 0, 2 * S, 1) == 3_000_000
    assert analysis.series_delta(rows, S // 2, 3 * S // 2, 1) == pytest.approx(1_500_000)
    assert analysis.series_delta(rows, 1 * S, 10 * S, 1) == 2_000_000  # clipped at the end
    assert analysis.series_delta(rows, 5 * S, 6 * S, 1) is None
    assert analysis.series_delta(rows[:1], 0, S, 1) is None


def host_sample(t, pool_busy, pool_idle, cpu_some, api_util, waiters, workload_usage):
    return dict(
        t=t,
        active=2,
        host=dict(
            cpu=dict(
                pool=dict(
                    user=pool_busy,
                    nice=0,
                    system=0,
                    idle=pool_idle,
                    iowait=0,
                    irq=0,
                    softirq=0,
                    steal=0,
                ),
                procs_running=4,
            ),
            pressure=dict(cpu=dict(some_usec=cpu_some, full_usec=0)),
            vmstat=dict(pswpout=0, pgscan_direct=0),
            schedstat=dict(run_delay_ns=t // 10),
        ),
        workload=dict(
            cpu=dict(usage_usec=workload_usage),
            pressure=dict(cpu=dict(some_usec=cpu_some, full_usec=0)),
            memory=dict(current=1000 + t // S, events=dict(oom_kill=0)),
        ),
        api=dict(utilization=api_util, waiters=waiters),
    )


def test_host_intervals_use_consistent_units_and_denominators():
    samples = [
        host_sample(0, 0, 0, 0, 0.1, 0, 0),
        host_sample(1 * S, 600, 600, 500_000, 0.9, 2, 6_000_000),
        host_sample(2 * S, 600, 1800, 500_000, 0.2, 0, 6_000_000),
    ]
    first, second = analysis.host_intervals(samples)
    assert first["seconds"] == 1.0 and first["pool_cpu_utilization"] == 0.5
    assert first["workload_cpu_some_fraction"] == 0.5  # 0.5 s stalled in a 1 s interval
    assert first["workload_cpu_seconds"] == 6.0 and first["api_utilization"] == 0.9
    assert second["pool_cpu_utilization"] == 0.0 and second["workload_cpu_some_fraction"] == 0
    assert first["pool_run_delay_seconds"] == pytest.approx(0.1)
    summary = analysis.resource_summary([first, second], samples)
    assert summary["pool_cpu_utilization"] == dict(mean=0.25, peak=0.5)
    assert summary["stall_seconds"]["workload_cpu_some"] == 0.5
    assert summary["workload_cpu_seconds"] == 6.0 and summary["workload_oom_kills"] == 0
    assert summary["peak_api_waiters"] == 2 and summary["sampled_seconds"] == 2.0


def test_cross_resource_metrics_on_a_known_schedule():
    samples = [
        host_sample(0, 0, 0, 0, 0.1, 0, 0),
        host_sample(1 * S, 600, 600, 500_000, 0.9, 1, 6_000_000),
        host_sample(2 * S, 900, 1500, 500_000, 0.2, 0, 6_000_000),
        host_sample(3 * S, 900, 2700, 500_000, 0.2, 0, 6_000_000),
    ]
    intervals = analysis.host_intervals(samples)
    timelines = {
        # agent 1 runs a tool for the first two seconds, then thinks
        "1": [("tool", 0, 2 * S), ("model_stream", 2 * S, 3 * S)],
        # agent 2 waits for the API, then thinks, then is idle
        "2": [("api_wait", 0, 1 * S), ("model_stream", 1 * S, 2 * S), ("other", 2 * S, 3 * S)],
    }
    cross, rows = analysis.cross_resource(timelines, intervals, 3.0)
    assert [r["agents_tool"] for r in rows] == [1, 1, 0]
    assert [r["agents_inference"] for r in rows] == [0, 1, 1]
    assert [r["agents_api_wait"] for r in rows] == [1, 0, 0]
    overlap = cross["inference_tool_overlap"]
    assert overlap["seconds_both_present"] == 1.0  # second 1-2: tool and inference together
    assert overlap["seconds_tool_only"] == 1.0 and overlap["seconds_inference_only"] == 1.0
    assert overlap["agent_pair_seconds"] == 1.0
    both = cross["simultaneous_cpu_and_api_contention"]
    assert both["seconds"] == 1.0 and both["seconds_cpu_contended"] == 1.0
    assert both["seconds_api_waiting"] == 1.0
    waiting = cross["utilization_while_api_waiting"]
    assert waiting["pool_cpu"] == dict(mean=0.5, seconds=1.0)
    assert waiting["pool_cpu_otherwise"]["seconds"] == 2.0
    assert cross["utilization_while_tools_run"]["pool_cpu"]["seconds"] == 2.0
    assert cross["mean_agents"]["tool"] == pytest.approx(2 / 3)
    assert cross["covered_seconds"] == 3.0


def test_agent_series_reports_measured_stall_per_interval():
    def agent(t, usage, stall, delay):
        return dict(
            t=t,
            agents={
                "3": dict(
                    container=dict(
                        cpu=dict(usage_usec=usage),
                        pressure=dict(cpu=dict(some_usec=stall)),
                        memory=dict(current=5),
                        threads=dict(
                            run_delay_ns=delay,
                            runnable=2,
                            threads=9,
                            voluntary_switches=t // S,
                            involuntary_switches=0,
                        ),
                    )
                )
            },
        )

    series = analysis.agent_series(
        [agent(0, 0, 0, 0), agent(2 * S, 3_000_000, 400_000, 10**8), dict(t=3 * S)]
    )
    [row] = series["3"]
    assert row["seconds"] == 2.0 and row["container_cpu_seconds"] == 3.0
    assert row["container_cpu_stall_seconds"] == 0.4
    assert row["container_run_delay_seconds_lower_bound"] == 0.1
    assert row["container_runnable_threads"] == 2


def test_perfetto_trace_has_one_track_per_agent_and_counter_tracks():
    intervals = analysis.host_intervals(
        [host_sample(0, 0, 0, 0, 0.1, 0, 0), host_sample(S, 1, 1, 0, 0.3, 0, 0)]
    )
    trace = analysis.perfetto_trace(
        {"1": [("tool", 0, S)], "2": [("other", 0, S)]}, intervals, 0, {"1": "A01 x", "2": "A02 y"}
    )
    events = trace["traceEvents"]
    slices = [e for e in events if e["ph"] == "X"]
    assert {e["tid"] for e in slices} == {1, 2} and slices[0]["dur"] == 1e6
    assert any(e["ph"] == "C" and e["name"] == "pool CPU utilization" for e in events)
    assert {e["args"]["name"] for e in events if e["name"] == "thread_name"} == {"A01 x", "A02 y"}
    json.dumps(trace)
