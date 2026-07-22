"""Tests for agresources — GPU/CPU/memory pool and host detection."""

import os
import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from agency.agconfig import agConfig
from agency.agresources import (
    agResourcePool,
    amd_render_node_paths_by_pci_bus,
    detect_cpus,
    detect_gpus,
    detect_memory_mb,
    _cvd_filter,
    _AgResourcePoolFields,
)


# ---------------------------------------------------------------------------
# _cvd_filter
# ---------------------------------------------------------------------------


def test_cvd_filter_no_env_passes_all(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _cvd_filter([0, 1, 2]) == [0, 1, 2]


def test_cvd_filter_restricts_to_allowed(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    assert _cvd_filter([0, 1, 2, 3]) == [0, 2]


def test_cvd_filter_empty_string_passes_all(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert _cvd_filter([0, 1]) == [0, 1]


def test_cvd_filter_nodevfiles_passes_all(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "NoDevFiles")
    assert _cvd_filter([0, 1]) == [0, 1]


def test_cvd_filter_single_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert _cvd_filter([0, 1, 2, 3]) == [3]


def test_cvd_filter_id_not_in_pool_excluded(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    assert _cvd_filter([0, 1, 2]) == []


def test_cvd_filter_hip_visible_devices_restricts(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1,3")
    assert _cvd_filter([0, 1, 2, 3]) == [1, 3]


def test_cvd_filter_rocr_visible_devices_restricts(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2")
    assert _cvd_filter([0, 1, 2, 3]) == [2]


def test_cvd_filter_cuda_takes_priority_over_hip(monkeypatch):
    """If both happen to be set, CUDA_VISIBLE_DEVICES wins -- matches the
    order _cvd_filter checks them in."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    assert _cvd_filter([0, 1]) == [0]


def test_cvd_filter_hip_nodevfiles_passes_all(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "NoDevFiles")
    assert _cvd_filter([0, 1]) == [0, 1]


# ---------------------------------------------------------------------------
# detect_gpus
# ---------------------------------------------------------------------------


def test_detect_gpus_returns_list():
    result = detect_gpus()
    assert isinstance(result, list)
    assert all(isinstance(g, int) for g in result)


def test_detect_gpus_nvidia_smi_unavailable_returns_empty(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with patch("agency.agresources.subprocess.run", side_effect=FileNotFoundError):
        assert detect_gpus() == []


def test_detect_gpus_nvidia_smi_nonzero_exit_returns_empty(monkeypatch):
    mock = MagicMock()
    mock.returncode = 1
    mock.stdout = ""
    with patch("agency.agresources.subprocess.run", return_value=mock):
        assert detect_gpus() == []


def test_detect_gpus_parses_nvidia_smi_output(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = "0\n1\n2\n"
    with patch("agency.agresources.subprocess.run", return_value=mock):
        assert detect_gpus() == [0, 1, 2]


def test_detect_gpus_cvd_filters_nvidia_output(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = "0\n1\n2\n"
    with patch("agency.agresources.subprocess.run", return_value=mock):
        assert detect_gpus() == [0, 2]


def _run_nvidia_fails_rocm_succeeds(rocm_stdout):
    """Build a subprocess.run stub: nvidia-smi raises FileNotFoundError (not
    installed), rocm-smi succeeds with the given stdout -- simulating an
    AMD-only host, exactly the fallback path detect_gpus() is meant to take."""

    def _run(cmd, *a, **kw):
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError("no nvidia-smi")
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = rocm_stdout
        return mock

    return _run


def test_detect_gpus_falls_back_to_rocm_smi_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    stdout = "device,Device Name\ncard0,AMD Instinct MI350X\ncard1,AMD Instinct MI350X\n"
    with patch(
        "agency.agresources.subprocess.run",
        side_effect=_run_nvidia_fails_rocm_succeeds(stdout),
    ):
        assert detect_gpus() == [0, 1]


def test_detect_gpus_rocm_smi_cvd_filters_via_hip_visible_devices(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    stdout = "device,Device Name\ncard0,AMD Instinct MI350X\ncard1,AMD Instinct MI350X\n"
    with patch(
        "agency.agresources.subprocess.run",
        side_effect=_run_nvidia_fails_rocm_succeeds(stdout),
    ):
        assert detect_gpus() == [1]


# ---------------------------------------------------------------------------
# amd_render_node_paths_by_pci_bus -- regression coverage for a real finding
# on 8x MI350X hardware: rocm-smi's GPU index does NOT correspond to sorted
# /dev/dri/renderD* order (each GPU there exposes itself plus 7 XCD/compute-
# partition sibling render nodes, 64 nodes total for 8 GPUs, and even the
# primary node's number doesn't sort in GPU-index order -- GPU 3's real node
# was the numerically LOWEST of the 64 present). These tests mock both
# `rocm-smi --showbus` (subprocess.run) and the /sys/class/drm/*/device
# symlink resolution (os.path.realpath) so they run identically with or
# without real ROCm hardware -- see TestAmdRenderNodeLiveHardware in
# tests/agsandbox_backends/test_container.py for the check against real
# hardware.
# ---------------------------------------------------------------------------


def _showbus_stdout(bus_by_gpu_id):
    lines = ["device,PCI Bus"]
    for gpu_id, bus in bus_by_gpu_id.items():
        lines.append(f"card{gpu_id},{bus}")
    return "\n".join(lines) + "\n"


def _realpath_stub(bus_by_render_name):
    """Stub for os.path.realpath: resolves /sys/class/drm/<name>/device to a
    fake sysfs path ending in the given PCI bus, or (if the name maps to
    None) a non-PCI platform-device path -- mirrors the real
    "amdgpu_xcp_N" sysfs layout an XCD/compute-partition sibling render
    node resolves to on real MI300/MI350 hardware."""

    def _realpath(path):
        name = path.split("/")[-2]
        bus = bus_by_render_name.get(name)
        if bus is None:
            return f"/sys/devices/platform/amdgpu_xcp_{name}"
        return f"/sys/devices/pci0000:00/0000:00:01.1/{bus}"

    return _realpath


def test_amd_render_node_paths_by_pci_bus_returns_none_when_rocm_smi_missing():
    with patch("agency.agresources.subprocess.run", side_effect=FileNotFoundError):
        assert amd_render_node_paths_by_pci_bus(["/dev/dri/renderD128"]) is None


def test_amd_render_node_paths_by_pci_bus_returns_none_on_nonzero_exit():
    mock = MagicMock()
    mock.returncode = 1
    mock.stdout = ""
    with patch("agency.agresources.subprocess.run", return_value=mock):
        assert amd_render_node_paths_by_pci_bus(["/dev/dri/renderD128"]) is None


def test_amd_render_node_paths_by_pci_bus_returns_none_when_a_gpu_bus_is_unmatched():
    """Only one of two GPUs' PCI buses resolves to a candidate render node --
    the mapping must be refused entirely (caller falls back to naive order)
    rather than half-applied to just the GPUs that happened to match."""
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = _showbus_stdout({0: "0000:05:00.0", 1: "0000:15:00.0"})
    with patch("agency.agresources.subprocess.run", return_value=mock):
        with patch(
            "agency.agresources.os.path.realpath",
            side_effect=_realpath_stub({"renderD128": "0000:05:00.0"}),
        ):
            assert amd_render_node_paths_by_pci_bus(["/dev/dri/renderD128"]) is None


def test_amd_render_node_paths_by_pci_bus_reorders_to_match_gpu_index():
    """Miniature reproduction of the real 8x MI350X finding: sorted
    /dev/dri order does NOT correspond to rocm-smi's GPU index -- here
    renderD128 is actually GPU 1's node and renderD129 is actually GPU 0's,
    the reverse of naive sorted order."""
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = _showbus_stdout({0: "0000:75:00.0", 1: "0000:05:00.0"})
    with patch("agency.agresources.subprocess.run", return_value=mock):
        with patch(
            "agency.agresources.os.path.realpath",
            side_effect=_realpath_stub(
                {"renderD128": "0000:05:00.0", "renderD129": "0000:75:00.0"}
            ),
        ):
            result = amd_render_node_paths_by_pci_bus(
                ["/dev/dri/renderD128", "/dev/dri/renderD129"]
            )
    assert result == ["/dev/dri/renderD129", "/dev/dri/renderD128"]


def test_amd_render_node_paths_by_pci_bus_ignores_non_pci_xcp_sibling_nodes():
    """XCD/compute-partition sibling render nodes (sysfs parent is a
    "amdgpu_xcp_N" platform device, not a PCI device) must never be picked
    as a GPU's primary node -- only the one with a real resolvable PCI bus
    can match a GPU."""
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = _showbus_stdout({0: "0000:05:00.0", 1: "0000:15:00.0"})
    with patch("agency.agresources.subprocess.run", return_value=mock):
        with patch(
            "agency.agresources.os.path.realpath",
            side_effect=_realpath_stub(
                {
                    "renderD128": "0000:05:00.0",
                    "renderD129": None,  # XCP sibling of GPU 0
                    "renderD136": "0000:15:00.0",
                }
            ),
        ):
            result = amd_render_node_paths_by_pci_bus(
                ["/dev/dri/renderD128", "/dev/dri/renderD129", "/dev/dri/renderD136"]
            )
    assert result == ["/dev/dri/renderD128", "/dev/dri/renderD136"]


# ---------------------------------------------------------------------------
# detect_cpus
# ---------------------------------------------------------------------------


def test_detect_cpus_returns_positive_int():
    result = detect_cpus()
    assert isinstance(result, int)
    assert result >= 1


def test_detect_cpus_os_cpu_count_none_returns_one(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert detect_cpus() == 1


# ---------------------------------------------------------------------------
# detect_memory_mb
# ---------------------------------------------------------------------------


def test_detect_memory_mb_returns_positive_int():
    result = detect_memory_mb()
    assert isinstance(result, int)
    assert result > 0


def test_detect_memory_mb_fallback_when_proc_missing(monkeypatch, tmp_path):
    fake = tmp_path / "meminfo"
    fake.write_text("Garbage: 0\n")
    with patch("builtins.open", side_effect=FileNotFoundError):
        with patch("agency.agresources.subprocess.run", side_effect=FileNotFoundError):
            assert detect_memory_mb() == _AgResourcePoolFields.memory_detect_fallback_mb.default


# ---------------------------------------------------------------------------
# agResourcePool construction
# ---------------------------------------------------------------------------


def test_pool_explicit_gpus():
    pool = agResourcePool(gpus=[0, 1], total_cpus=8, total_memory_mb=16384)
    assert pool.gpus == [0, 1]
    assert pool.total_cpus == 8
    assert pool.total_memory_mb == 16384


def test_pool_empty_gpus():
    pool = agResourcePool(gpus=[], total_cpus=4, total_memory_mb=8192)
    assert pool.gpus == []
    assert pool._free_gpus == set()


def test_pool_default_idle_values():
    """idle_cpus keeps its fixed default; idle_memory defaults to None (no
    cap) rather than an arbitrary fixed constant like "4096m" — sandboxes are
    torn down after use, not reset-and-reused indefinitely, so there's no
    idle container to bound by default. container.py/update_limits() both
    treat None as "omit --memory", Docker's own native unlimited behavior."""
    pool = agResourcePool(gpus=[], total_cpus=4, total_memory_mb=8192)
    assert pool.idle_cpus == _AgResourcePoolFields.idle_cpus.default
    assert pool.idle_memory is None


def test_disconnected_fields_instance_idle_memory_defaults_to_none():
    """agsandbox_backends/container.py's container-creation path reads
    idle_memory through a fresh _AgResourcePoolFields(sandbox_agconfig) bound
    to the SANDBOX's own agconfig, not agResourcePool's — a completely
    different, unrelated agConfig instance that never has idle_memory
    explicitly set on it. That means idle_memory's own class-level default
    (not anything set inside agResourcePool.__init__) is what actually
    reaches real container creation, and it must be None so --memory is
    omitted there too, not a fixed constant regardless of host size."""
    fields = _AgResourcePoolFields(agConfig())
    assert fields.idle_memory is None


def test_pool_explicit_idle_memory_overrides_default():
    pool = agResourcePool(gpus=[], total_memory_mb=8192, idle_memory="1g")
    assert pool.idle_memory == "1g"


def test_pool_initial_acquired_counts_are_zero():
    pool = agResourcePool(gpus=[0, 1], total_cpus=4, total_memory_mb=8192)
    assert pool._gpus_acquired == 0
    assert pool.cpus_acquired == 0.0
    assert pool.memory_acquired_mb == 0


def test_pool_repr():
    pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
    r = repr(pool)
    assert "agResourcePool" in r
    assert "total_cpus=4" in r


# ---------------------------------------------------------------------------
# GPU acquire / release
# ---------------------------------------------------------------------------


def test_single_gpu_acquire_release():
    pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
    gpu_id = pool.acquire_gpu()
    assert gpu_id == 0
    assert pool._gpus_acquired == 1
    pool.release_gpu(gpu_id)
    assert pool._gpus_acquired == 0


def test_acquire_returns_any_free_gpu():
    pool = agResourcePool(gpus=[0, 1], total_cpus=4, total_memory_mb=8192)
    g1 = pool.acquire_gpu()
    g2 = pool.acquire_gpu()
    assert {g1, g2} == {0, 1}
    pool.release_gpu(g1)
    pool.release_gpu(g2)


def test_acquire_blocks_until_release():
    pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
    pool.acquire_gpu()

    acquired_after = threading.Event()

    def waiter():
        pool.acquire_gpu(timeout=5.0)
        acquired_after.set()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()

    time.sleep(0.05)
    assert not acquired_after.is_set()
    pool.release_gpu(0)
    acquired_after.wait(timeout=5.0)
    assert acquired_after.is_set()
    t.join(timeout=5.0)


def test_acquire_timeout_raises():
    pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
    pool.acquire_gpu()
    with pytest.raises(TimeoutError):
        pool.acquire_gpu(timeout=0.1)
    pool.release_gpu(0)


def test_acquire_does_not_poll_via_sleep(monkeypatch):
    """acquire_gpu() blocks on Condition.wait(), not a sleep/retry loop --
    unlike the old per-GPU-semaphore round-robin polling design, nothing in
    the wait path should ever call time.sleep(). Uses a threading.Event
    (not time.sleep) to sequence the main thread, since agency.agresources'
    `time` import IS the stdlib time module -- patching time.sleep there
    patches it everywhere in this process, including a time.sleep() call
    made directly from this test."""
    pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
    pool.acquire_gpu()

    slept = []
    monkeypatch.setattr("agency.agresources.time.sleep", lambda s: slept.append(s))

    started = threading.Event()

    def waiter():
        started.set()
        pool.acquire_gpu(timeout=2.0)

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    started.wait(timeout=5.0)
    pool.release_gpu(0)
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert slept == []


def test_release_wakes_a_waiter_promptly():
    """release_gpu() must wake a blocked waiter directly (via notify()),
    not leave it discovering the free GPU only on its next poll tick --
    the wakeup should land in well under what a 0.25s poll interval would
    have cost."""
    pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
    pool.acquire_gpu()

    woke_at = []

    def waiter():
        pool.acquire_gpu(timeout=5.0)
        woke_at.append(time.monotonic())

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    time.sleep(0.05)  # ensure the waiter is parked in wait() before releasing
    released_at = time.monotonic()
    pool.release_gpu(0)
    t.join(timeout=5.0)
    assert woke_at, "waiter never acquired the released GPU"
    assert woke_at[0] - released_at < 0.05


def test_multiple_waiters_each_get_woken_exactly_once():
    """With N GPUs freed one at a time, exactly N waiters (out of more than
    N contenders) should succeed -- notify() must wake one waiter per
    release, never zero (a waiter stuck forever) or more than one racing
    for the same freed id."""
    pool = agResourcePool(gpus=[0, 1], total_cpus=4, total_memory_mb=8192)
    pool.acquire_gpu()
    pool.acquire_gpu()

    results = []
    lock = threading.Lock()

    def waiter():
        try:
            gpu_id = pool.acquire_gpu(timeout=5.0)
            with lock:
                results.append(gpu_id)
        except TimeoutError:
            pass

    threads = [threading.Thread(target=waiter, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    pool.release_gpu(0)
    pool.release_gpu(1)
    for t in threads:
        t.join(timeout=5.0)

    assert sorted(results) == [0, 1]


def test_release_unknown_gpu_is_safe():
    pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
    pool.release_gpu(99)  # must not raise


def test_release_double_release_warns(capsys):
    pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
    pool.acquire_gpu()
    pool.release_gpu(0)
    pool.release_gpu(0)  # double-release — warns, does not raise
    captured = capsys.readouterr()
    assert "WARNING" in captured.out


def test_release_gpu_is_immediate_no_polling(monkeypatch):
    """release_gpu() no longer takes (or needs) an is_clear predicate to
    poll: callers (backend stop()/destroy()) already run their own kill +
    container-removal/jail-rmtree teardown synchronously BEFORE calling
    this, so that ordering alone is the confirmation the sandbox has
    exited -- see agresources.release_gpu()'s docstring. Release must be
    immediate, with no sleep/poll loop of its own."""
    pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
    pool.acquire_gpu()

    slept = []
    monkeypatch.setattr("agency.agresources.time.sleep", lambda s: slept.append(s))

    pool.release_gpu(0)
    assert slept == []
    assert pool._gpus_acquired == 0


# ---------------------------------------------------------------------------
# CPU / memory notify
# ---------------------------------------------------------------------------


def test_notify_cpu_acquired_adds():
    pool = agResourcePool(gpus=[], total_cpus=8, total_memory_mb=16384)
    pool.notify_cpu_acquired(cpus=2.0, memory_mb=1024)
    assert pool.cpus_acquired == 2.0
    assert pool.memory_acquired_mb == 1024


def test_notify_cpu_released_subtracts():
    pool = agResourcePool(gpus=[], total_cpus=8, total_memory_mb=16384)
    pool.notify_cpu_acquired(cpus=4.0, memory_mb=2048)
    pool.notify_cpu_released(cpus=2.0, memory_mb=1024)
    assert pool.cpus_acquired == 2.0
    assert pool.memory_acquired_mb == 1024


def test_notify_cpu_released_floors_at_zero():
    pool = agResourcePool(gpus=[], total_cpus=8, total_memory_mb=16384)
    pool.notify_cpu_released(cpus=99.0, memory_mb=999999)
    assert pool.cpus_acquired == 0.0
    assert pool.memory_acquired_mb == 0


def test_notify_thread_safe():
    pool = agResourcePool(gpus=[], total_cpus=32, total_memory_mb=65536)
    errors = []

    def worker():
        try:
            for _ in range(50):
                pool.notify_cpu_acquired(1.0, 100)
                pool.notify_cpu_released(1.0, 100)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert not errors
    assert pool.cpus_acquired == 0.0


# ---------------------------------------------------------------------------
# change_config / get_config_copy
# ---------------------------------------------------------------------------


def test_pool_change_config_replaces_agconfig():
    from agency.agconfig import agConfig

    pool = agResourcePool(gpus=[], total_cpus=8, total_memory_mb=8192)
    pool.change_config(agConfig({"agResourcePool": {"idle_cpus": 2.0}}))
    assert pool._agconfig.get("agResourcePool", "idle_cpus") == 2.0


def test_pool_change_config_clones_given_agconfig():
    from agency.agconfig import agConfig

    pool = agResourcePool(gpus=[], total_cpus=8, total_memory_mb=8192)
    new_cfg = agConfig({"agResourcePool": {"idle_cpus": 2.0}})
    pool.change_config(new_cfg)
    new_cfg.agResourcePool.idle_cpus = 9.0
    assert pool._agconfig.get("agResourcePool", "idle_cpus") == 2.0


def test_pool_get_config_copy_returns_clone_not_same_object():
    pool = agResourcePool(gpus=[], total_cpus=8, total_memory_mb=8192)
    copy = pool.get_config_copy()
    assert copy is not pool._agconfig


def test_pool_get_config_copy_reflects_current_values():
    from agency.agconfig import agConfig

    pool = agResourcePool(
        gpus=[],
        total_cpus=8,
        total_memory_mb=8192,
        agconfig=agConfig({"agResourcePool": {"idle_cpus": 2.0}}),
    )
    assert pool.get_config_copy().agResourcePool.idle_cpus == 2.0


def test_mutating_pool_get_config_copy_does_not_affect_pool():
    from agency.agconfig import agConfig

    pool = agResourcePool(
        gpus=[],
        total_cpus=8,
        total_memory_mb=8192,
        agconfig=agConfig({"agResourcePool": {"idle_cpus": 2.0}}),
    )
    copy = pool.get_config_copy()
    copy.agResourcePool.idle_cpus = 9.0
    assert pool._agconfig.get("agResourcePool", "idle_cpus") == 2.0


def test_pool_change_config_none_resets_to_default_agconfig():
    from agency.agconfig import agConfig

    pool = agResourcePool(
        gpus=[],
        total_cpus=8,
        total_memory_mb=8192,
        agconfig=agConfig({"agResourcePool": {"idle_cpus": 2.0}}),
    )
    pool.change_config(None)
    # No agconfig -> field falls back to its DynamicConfigParam default, not the old value.
    assert (
        pool.get_config_copy().agResourcePool.idle_cpus == _AgResourcePoolFields.idle_cpus.default
    )
