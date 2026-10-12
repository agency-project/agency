"""GPU accounting regressions run without a driver or GPU workload."""

from types import SimpleNamespace as NS

import pytest

from agency.observability.profiler import agprof, agprof_trace
from agency.observability.profiler.gpu import (
    GPUCollector,
    VALID,
    NO_SAMPLES,
    UNSUPPORTED,
    ERROR,
    STALE,
    UNAVAILABLE,
)
from agency.observability.agwebui.investigator import normalize_trace


class NVMLError(Exception):
    def __init__(self, value):
        self.value = value


class NVML:
    def __init__(self):
        self.memory = []
        self.utilization = []
        self.fail = {}
        self.calls = []

    def get(self, name, value):
        self.calls.append(name)
        if name in self.fail:
            raise NVMLError(self.fail[name])
        return value

    def nvmlDeviceGetUUID(self, handle):
        return self.get("uuid", "GPU-test")

    def nvmlDeviceGetUtilizationRates(self, handle):
        return self.get("device_util", NS(gpu=75))

    def nvmlDeviceGetMemoryInfo(self, handle):
        return self.get("device_mem", NS(used=900 * 2**20))

    def nvmlDeviceGetPowerUsage(self, handle):
        return self.get("power", 100000)

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return self.get("process_mem", self.memory)

    def nvmlDeviceGetProcessUtilization(self, handle, last):
        return self.get("process_util", self.utilization)


def proc(root, pid, start=100, sandbox="box"):
    directory = root / str(pid)
    directory.mkdir(exist_ok=True)
    fields = ["0"] * 22
    fields[19] = str(start)
    (directory / "stat").write_text(f"{pid} (worker) " + " ".join(fields))
    (directory / "cgroup").write_text(f"0::/{sandbox}\n")


def mem(pid, mb):
    return NS(pid=pid, usedGpuMemory=mb * 2**20)


def util(pid, value, at=1_500_000):
    return NS(pid=pid, smUtil=value, timeStamp=at)


@pytest.fixture
def collector(tmp_path):
    nvml = NVML()
    samples = []
    c = GPUCollector(
        nvml,
        samples,
        lambda: {"box": "/sys/fs/cgroup/box", "other": "/sys/fs/cgroup/other"},
        proc_root=tmp_path,
    )
    c.started_us = 1_000_000
    c.boot_us = 0
    c.clock_ticks = 1000
    proc(tmp_path, 11)
    proc(tmp_path, 12)
    return c


def values(c):
    return {name: value for _, name, value in c.samples}


def prefix(pid=11, start=100, sandbox="box"):
    return f"gpu0:proc:{pid}-{start}:sandbox:{sandbox}:device:GPU-test"


def test_distinct_gpu_processes_deduplicate_memory_and_do_not_sum_sm(collector):
    c = collector
    c.nvml.memory = [mem(11, 100), mem(12, 200), mem(11, 100)]
    c.nvml.utilization = [util(11, 20), util(12, 30), util(11, 20)]
    c.tick(2_000_000_000, 0, None, 2)
    v = values(c)
    assert v[prefix() + ":mem_mb"] == 100
    assert v[prefix(12) + ":mem_mb"] == 200
    assert v["gpu0:processes:mem_mb"] == 300
    assert v["gpu0:sandbox:box:process_mem_mb"] == 300
    assert v["gpu0:mem_mb"] == 900
    assert v[prefix() + ":util_pct"] == 20
    assert v[prefix(12) + ":util_pct"] == 30
    assert v["gpu0:util_pct"] == 75
    assert "gpu0:processes:util_pct" not in v
    assert not any("sandbox:box:util_pct" in name for name in v)
    assert len(c.samples) == len({(at, name) for at, name, _ in c.samples})
    assert (1_500_000_000, prefix() + ":util_pct", 20) in c.samples


def test_reused_pid_and_migrating_process_refresh_identity(collector):
    c = collector
    c.nvml.memory = [mem(11, 100)]
    c.tick(2_000_000_000, 0, None, 2)
    proc(c.proc_root, 11, start=2500)
    c.nvml.utilization = [util(11, 80, 2_200_000), util(11, 10, 2_900_000)]
    c.tick(3_000_000_000, 0, None, 3)
    v = values(c)
    assert v[prefix(start=2500) + ":mem_mb"] == 100
    assert v[prefix() + ":mem_mb:state"] == STALE
    assert [value for _, name, value in c.samples if name == prefix(start=2500) + ":util_pct"] == [
        10
    ]
    proc(c.proc_root, 11, start=2500, sandbox="other")
    c.nvml.utilization = [util(11, 90, 3_500_000)]
    c.tick(4_000_000_000, 0, None, 4)
    v = values(c)
    assert v[prefix(start=2500, sandbox="other") + ":mem_mb"] == 100
    assert v[prefix(start=2500) + ":mem_mb:state"] == STALE
    assert prefix(start=2500, sandbox="other") + ":util_pct" not in v
    assert v["gpu0:sandbox:box:process_mem_mb"] == 0
    assert v["gpu0:sandbox:other:process_mem_mb"] == 100


def test_zero_no_samples_unsupported_failure_and_exit_are_distinct(collector):
    c = collector
    c.nvml.memory = [mem(11, 0)]
    c.nvml.utilization = [util(11, 0)]
    c.tick(2_000_000_000, 0, None, 2)
    v = values(c)
    assert v[prefix() + ":mem_mb"] == v[prefix() + ":util_pct"] == 0
    assert v[prefix() + ":util_pct:state"] == VALID
    c.nvml.fail["process_util"] = 6
    c.tick(3_000_000_000, 0, None, 3)
    assert values(c)[prefix() + ":util_pct:state"] == NO_SAMPLES
    c.nvml.fail["process_util"] = 3
    c.tick(4_000_000_000, 0, None, 4)
    assert values(c)[prefix() + ":util_pct:state"] == UNSUPPORTED
    c.nvml.fail["process_util"] = 999
    c.tick(5_000_000_000, 0, None, 5)
    assert values(c)[prefix() + ":util_pct:state"] == ERROR
    c.nvml.fail.clear()
    c.nvml.memory = []
    c.nvml.utilization = []
    c.tick(6_000_000_000, 0, None, 6)
    assert values(c)[prefix() + ":mem_mb:state"] == STALE
    assert values(c)["gpu0:processes:mem_mb"] == 0
    assert len([1 for _, name, _ in c.samples if name == prefix() + ":util_pct"]) == 1


@pytest.mark.parametrize(
    "failure", ["device_util", "device_mem", "power", "process_mem", "process_util"]
)
def test_nvml_queries_fail_independently(collector, failure):
    c = collector
    c.nvml.memory = [mem(11, 100)]
    c.nvml.utilization = [util(11, 20)]
    c.nvml.fail[failure] = 3
    c.tick(2_000_000_000, 0, None, 2)
    names = {
        "device_util": "gpu0:util_pct",
        "device_mem": "gpu0:mem_mb",
        "power": "gpu0:power_w",
        "process_mem": "gpu0:processes:mem_mb",
        "process_util": "gpu0:processes:util_pct",
    }
    v = values(c)
    assert v[names[failure] + ":state"] == UNSUPPORTED
    for query, name in names.items():
        assert query in c.nvml.calls
        if query != failure:
            assert v[name + ":state"] == VALID


@pytest.mark.parametrize("missing", [None, -1, 2**64 - 1])
def test_unavailable_memory_does_not_become_zero_or_partial_total(collector, missing):
    c = collector
    c.nvml.memory = [mem(11, 100), NS(pid=12, usedGpuMemory=missing)]
    c.tick(2_000_000_000, 0, None, 2)
    v = values(c)
    assert "gpu0:processes:mem_mb" not in v
    assert v["gpu0:processes:mem_mb:state"] == UNAVAILABLE
    assert v["gpu0:sandbox:box:process_mem_mb:state"] == UNAVAILABLE
    assert v[prefix() + ":mem_mb"] == 100


def test_missing_proc_metadata_keeps_pid_but_never_guesses_owner(collector):
    c = collector
    c.nvml.memory = [mem(99, 20)]
    c.tick(2_000_000_000, 0, None, 2)
    v = values(c)
    assert v["gpu0:proc:99-unknown:sandbox:~:device:GPU-test:mem_mb"] == 20
    assert not any(":sandbox:box:" in name for name in v)


def test_live_export_and_saved_trace_preserve_process_samples_and_states(collector, monkeypatch):
    c = collector
    c.nvml.memory = [mem(11, 100), mem(12, 200)]
    c.nvml.utilization = [util(11, 20, 1_400_000), util(11, 30, 1_600_000)]
    sampler = object.__new__(agprof._Sampler)
    sampler._nvml = c.nvml
    sampler._handles = [None]
    sampler._gpu_collector = c
    sampler._tick_workload_cgroup = lambda t: None
    sampler._tick_processes = lambda t: None
    sampler._tick_cgroups = lambda t: None
    events = []
    monkeypatch.setattr(agprof, "_samples", c.samples)
    monkeypatch.setattr(
        agprof,
        "_profile_data_logger",
        NS(record_event=lambda kind, payload: events.append(payload)),
    )
    monkeypatch.setattr(agprof.time, "perf_counter_ns", lambda: 2_000_000_000)
    monkeypatch.setattr(agprof.time, "time", lambda: 2)
    sampler._tick()
    exported = [
        (row["sampled_at"], row["values"][prefix() + ":util_pct"])
        for row in events
        if prefix() + ":util_pct" in row["values"]
    ]
    assert exported == [(1.4, 20), (1.6, 30)]
    assert events[-1]["values"]["gpu0:processes:mem_mb"] == 300
    trace = agprof_trace.build_trace([], c.samples, [], process_info={})
    saved = normalize_trace(trace, {})
    assert [point[1] for point in saved["counters"][prefix() + ":util_pct"]] == [20, 30]
    assert saved["counters"]["gpu0:processes:mem_mb"][0][1] == 300
    assert prefix(12) + ":util_pct:state" in saved["counters"]


def test_device_identity_separates_the_same_process_on_two_gpus(collector):
    c = collector
    c.nvml.memory = [mem(11, 100)]
    c.tick(2_000_000_000, 0, None, 2)
    c.tick(2_000_000_000, 1, None, 2)
    v = values(c)
    assert v[prefix() + ":mem_mb"] == 100
    assert v[prefix().replace("gpu0:", "gpu1:") + ":mem_mb"] == 100
    assert v["gpu0:processes:mem_mb"] == v["gpu1:processes:mem_mb"] == 100


def test_conflicting_duplicate_rows_are_unavailable_instead_of_guessed(collector):
    c = collector
    c.nvml.memory = [mem(11, 100), mem(11, 200)]
    c.tick(2_000_000_000, 0, None, 2)
    v = values(c)
    assert prefix() + ":mem_mb" not in v
    assert "gpu0:processes:mem_mb" not in v
    assert v["gpu0:processes:mem_mb:state"] == UNAVAILABLE


def test_summary_keeps_collection_states_out_of_numeric_metric_statistics(collector):
    c = collector
    c.nvml.fail["power"] = 3
    c.tick(2_000_000_000, 0, None, 2)
    summary = agprof._build_run_summary(
        [],
        c.samples,
        [],
        started_ns=1_000_000_000,
        ended_ns=2_000_000_000,
        sample_hz=10,
        sample_gpu=True,
        gpu_sampling_available=True,
    )
    assert summary["gpu_collection_states"]["gpu0:power_w"]["state"] == UNSUPPORTED
    assert all(not metric["name"].endswith(":state") for metric in summary["resource_metrics"])


def test_buffered_sample_after_empty_poll_restores_current_state(collector):
    c = collector
    c.nvml.memory = [mem(11, 100)]
    c.tick(2_000_000_000, 0, None, 2)
    assert values(c)[prefix() + ":util_pct:state"] == NO_SAMPLES
    c.nvml.utilization = [util(11, 25, 1_900_000)]
    c.tick(3_000_000_000, 0, None, 3)
    assert (1_900_000_000, prefix() + ":util_pct", 25) in c.samples
    assert (3_000_000_000, prefix() + ":util_pct:state", VALID) in c.samples
    trace = agprof_trace.build_trace([], c.samples, [], process_info={}, started_ns=0)
    saved = normalize_trace(trace, {})
    assert saved["counters"][prefix() + ":util_pct:state"] == [[2, NO_SAMPLES], [3, VALID]]
