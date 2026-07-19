"""Tests for agresources — GPU/CPU/memory pool and host detection."""

import os
import threading
from unittest.mock import patch, MagicMock

import pytest

from agency.agconfig import agConfig
from agency.agresources import (
    agResourcePool,
    detect_cpus,
    detect_gpus,
    detect_memory_mb,
    _cvd_filter,
    _gpu_compute_pids,
    _nvidia_gpu_compute_pids,
    _rocm_gpu_compute_pids,
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
    AMD-only host, exactly the fallback path detect_gpus()/_gpu_compute_pids()
    are meant to take."""

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
    assert pool._gpu_locks == {}


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

    import time

    time.sleep(0.05)
    assert not acquired_after.is_set()
    pool.release_gpu(0)  # release_gpu sleeps 3s internally before freeing the slot
    acquired_after.wait(timeout=5.0)
    assert acquired_after.is_set()
    t.join(timeout=5.0)


def test_acquire_timeout_raises():
    pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
    pool.acquire_gpu()
    with pytest.raises(TimeoutError):
        pool.acquire_gpu(timeout=0.1)
    pool.release_gpu(0)


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


def test_release_gpu_waits_for_straggler_compute_process_to_clear(monkeypatch):
    """release_gpu() must not free the slot while nvidia-smi still shows a
    compute process (e.g. a backgrounded job the caller forgot to wait on)
    on that physical GPU — it should poll until the process list clears."""
    pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
    pool.acquire_gpu()

    calls = {"n": 0}

    def fake_pids(gpu_id):
        calls["n"] += 1
        return {999999} if calls["n"] == 1 else set()

    monkeypatch.setattr("agency.agresources._gpu_compute_pids", fake_pids)
    monkeypatch.setattr("agency.agresources.time.sleep", lambda s: None)

    pool.release_gpu(0)
    assert calls["n"] == 2  # polled once more after the straggler cleared
    assert pool._gpus_acquired == 0


def test_release_gpu_gives_up_after_timeout_and_warns(monkeypatch, capsys):
    """A straggler that never exits must not wedge the GPU as permanently
    unreleasable — release proceeds anyway once the wait deadline passes,
    with a warning identifying the leftover pid(s)."""
    pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
    pool.acquire_gpu()

    monkeypatch.setattr("agency.agresources._gpu_compute_pids", lambda gpu_id: {123456})
    times = iter([0.0, 1000.0])
    monkeypatch.setattr("agency.agresources.time.monotonic", lambda: next(times))

    pool.release_gpu(0)
    captured = capsys.readouterr()
    assert "still shows compute processes" in captured.out
    assert "123456" in captured.out
    assert pool._gpus_acquired == 0


def test_release_gpu_skips_wait_when_nvidia_smi_unavailable(monkeypatch):
    """On hosts without nvidia-smi (or non-NVIDIA hardware), the check can't
    run at all — release must proceed immediately rather than block."""
    pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
    pool.acquire_gpu()

    monkeypatch.setattr("agency.agresources._gpu_compute_pids", lambda gpu_id: None)
    slept = []
    monkeypatch.setattr("agency.agresources.time.sleep", lambda s: slept.append(s))

    pool.release_gpu(0)
    assert slept == []
    assert pool._gpus_acquired == 0


def test_release_gpu_with_own_pids_ignores_unrelated_process():
    """A process on the physical GPU that ISN'T one of own_pids (an
    unrelated tenant sharing the device) must never count as a straggler --
    release proceeds immediately, no waiting, no warning."""
    pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
    pool.acquire_gpu()

    with patch("agency.agresources._gpu_compute_pids", lambda gpu_id: {424242}):
        pool.release_gpu(0, own_pids={111, 222})  # 424242 is unrelated -- ignored
    assert pool._gpus_acquired == 0


def test_release_gpu_with_own_pids_waits_for_own_straggler(monkeypatch):
    """A process that IS in own_pids must still be waited on, even though
    other unrelated PIDs are also present on the device."""
    pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
    pool.acquire_gpu()

    # First poll reports our straggler (999999) alongside an unrelated PID
    # (424242, someone else's tenant); second poll reports only the
    # unrelated one -- ours has cleared, so release must proceed.
    responses = iter([{999999, 424242}, {424242}])
    monkeypatch.setattr("agency.agresources._gpu_compute_pids", lambda gpu_id: next(responses))
    monkeypatch.setattr("agency.agresources.time.sleep", lambda s: None)

    pool.release_gpu(0, own_pids={999999})
    assert pool._gpus_acquired == 0


def test_wait_for_gpu_clear_timeout_warning_lists_only_own_stragglers(monkeypatch, capsys):
    """When own_pids is given, the give-up warning must only name PIDs that
    are actually ours -- not every PID nvidia-smi/rocm-smi happens to report
    on the device (which would misleadingly look like our own leak)."""
    pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
    pool.acquire_gpu()

    monkeypatch.setattr("agency.agresources._gpu_compute_pids", lambda gpu_id: {999999, 424242})
    times = iter([0.0, 1000.0])
    monkeypatch.setattr("agency.agresources.time.monotonic", lambda: next(times))

    pool.release_gpu(0, own_pids={999999})
    captured = capsys.readouterr()
    assert "999999" in captured.out
    assert "424242" not in captured.out


# ---------------------------------------------------------------------------
# _gpu_compute_pids — nvidia/rocm dispatch and rocm-smi parsing
# ---------------------------------------------------------------------------


def test_gpu_compute_pids_uses_nvidia_when_available():
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = "1234\n5678\n"
    with patch("agency.agresources.subprocess.run", return_value=mock) as run:
        assert _gpu_compute_pids(0) == {1234, 5678}
        # Only the nvidia-smi call should have been made -- no rocm-smi fallback.
        assert run.call_count == 1
        assert run.call_args.args[0][0] == "nvidia-smi"


def test_gpu_compute_pids_falls_back_to_rocm_when_nvidia_missing():
    rocm_out = (
        "PID 111 is using 0 DRM device(s)\n"
        "PID 222 is using 1 DRM device(s):\n"
        "1 \n"
        "PID 333 is using 1 DRM device(s):\n"
        "0 \n"
    )

    def _run(cmd, *a, **kw):
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError("no nvidia-smi")
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = rocm_out
        return mock

    with patch("agency.agresources.subprocess.run", side_effect=_run):
        assert _gpu_compute_pids(1) == {222}
        assert _gpu_compute_pids(0) == {333}


def test_nvidia_gpu_compute_pids_returns_none_when_unavailable():
    with patch("agency.agresources.subprocess.run", side_effect=FileNotFoundError):
        assert _nvidia_gpu_compute_pids(0) is None


def test_gpu_compute_pids_returns_none_when_neither_available():
    with patch("agency.agresources.subprocess.run", side_effect=FileNotFoundError):
        assert _gpu_compute_pids(0) is None


def test_rocm_gpu_compute_pids_parses_multi_gpu_process():
    """A process using more than one DRM device (tensor-parallel) must be
    attributed to every gpu_id it lists, not just the first."""
    rocm_out = "PID 999 is using 2 DRM device(s):\n0 1 \n"
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = rocm_out
    with patch("agency.agresources.subprocess.run", return_value=mock):
        assert _rocm_gpu_compute_pids(0) == {999}
        assert _rocm_gpu_compute_pids(1) == {999}
        assert _rocm_gpu_compute_pids(2) == set()


def test_rocm_gpu_compute_pids_none_when_command_fails():
    mock = MagicMock()
    mock.returncode = 1
    with patch("agency.agresources.subprocess.run", return_value=mock):
        assert _rocm_gpu_compute_pids(0) is None


def test_rocm_gpu_compute_pids_none_on_exception():
    with patch("agency.agresources.subprocess.run", side_effect=FileNotFoundError):
        assert _rocm_gpu_compute_pids(0) is None


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
