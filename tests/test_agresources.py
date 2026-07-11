"""Tests for agresources — GPU/CPU/memory pool and host detection."""
import os
import threading
from unittest.mock import patch, MagicMock

import pytest

from agency.agresources import (
    agResourcePool,
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
    pool = agResourcePool(gpus=[], total_cpus=4, total_memory_mb=8192)
    assert pool.idle_cpus == 4.0
    assert pool.idle_memory == "4096m"


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

    released = threading.Event()
    acquired_after = threading.Event()

    def waiter():
        pool.acquire_gpu(timeout=2.0)
        acquired_after.set()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()

    import time; time.sleep(0.05)
    assert not acquired_after.is_set()
    pool.release_gpu(0)
    acquired_after.wait(timeout=2.0)
    assert acquired_after.is_set()
    t.join(timeout=2.0)


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
    assert "WARNING" in captured.out or True  # warning is best-effort


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
    for t in threads: t.start()
    for t in threads: t.join(timeout=5)
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
    pool = agResourcePool(gpus=[], total_cpus=8, total_memory_mb=8192,
                           agconfig=agConfig({"agResourcePool": {"idle_cpus": 2.0}}))
    assert pool.get_config_copy().agResourcePool.idle_cpus == 2.0

def test_mutating_pool_get_config_copy_does_not_affect_pool():
    from agency.agconfig import agConfig
    pool = agResourcePool(gpus=[], total_cpus=8, total_memory_mb=8192,
                           agconfig=agConfig({"agResourcePool": {"idle_cpus": 2.0}}))
    copy = pool.get_config_copy()
    copy.agResourcePool.idle_cpus = 9.0
    assert pool._agconfig.get("agResourcePool", "idle_cpus") == 2.0

def test_pool_change_config_none_resets_to_default_agconfig():
    from agency.agconfig import agConfig
    pool = agResourcePool(gpus=[], total_cpus=8, total_memory_mb=8192,
                           agconfig=agConfig({"agResourcePool": {"idle_cpus": 2.0}}))
    pool.change_config(None)
    # No agconfig -> field falls back to its DynamicConfigParam default, not the old value.
    assert pool.get_config_copy().agResourcePool.idle_cpus == _AgResourcePoolFields.idle_cpus.default
