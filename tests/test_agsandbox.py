"""Unit tests for agSandbox and agResourcePool.

Sandbox tests that create real containers are marked with @pytest.mark.docker
and skipped automatically when Docker/Podman is unavailable.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid

import pytest
from unittest.mock import MagicMock, patch

from agency.agdata import agdata, agerror
from agency.agresources import agResourcePool


def _worker_import_agent():
    """Top-level so ProcessPoolExecutor can pickle it."""
    from agency.agent import agent  # noqa: F401
    import multiprocessing

    return multiprocessing.current_process().name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _docker_available() -> bool:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


docker = pytest.mark.skipif(not _docker_available(), reason="Docker daemon not reachable")


def _nvidia_smi_available() -> bool:
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


nvidia_smi = pytest.mark.skipif(not _nvidia_smi_available(), reason="nvidia-smi not available")


def _make_sandbox(**kwargs):
    """Build an agSandbox for these tests, always forcing the docker backend.

    This test file verifies container lifecycle by shelling out to the
    ``docker`` CLI directly (docker ps/images/rmi/...), so it needs every
    sandbox it builds to actually be a docker container regardless of the
    process-wide auto-detected default (which now prefers podman when both
    are usable -- see agsandbox_backends.base.agsandbox_backend.for_config()).
    """
    from agency.agconfig import agConfig
    from agency.agsandbox import agSandbox
    from agency.agsandbox_backends import agSandboxBackendConfig

    uid = str(uuid.uuid4())
    agconfig = kwargs.pop("agconfig", None)
    cfg = agConfig(agSandboxBackendConfig(backend="docker"), agconfig)
    return agSandbox(uid, agconfig=cfg, **kwargs)


def _agconfig_with_output_dir(output_dir):
    """Build an agconfig mounting output_dir at /agent_output, replacing the
    old output_dir= constructor kwarg."""
    from agency.agconfig import agConfig
    from agency.agsandbox import agSandboxConfig

    cfg = agConfig()
    agSandboxConfig(cfg).add_mount("agent_output", output_dir, "/agent_output")
    return cfg


# ---------------------------------------------------------------------------
# detect_gpus
# ---------------------------------------------------------------------------


class TestDetectGpus:
    def test_returns_list(self):
        from agency.agresources import detect_gpus

        gpus = detect_gpus()
        assert isinstance(gpus, list)
        assert all(isinstance(g, int) for g in gpus)

    def test_nvidia_smi_unavailable_returns_empty(self, monkeypatch):
        from agency.agresources import detect_gpus

        monkeypatch.setattr(
            "subprocess.run", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError())
        )
        assert detect_gpus() == []

    def test_nvidia_smi_nonzero_exit_returns_empty(self, monkeypatch):
        from unittest.mock import MagicMock
        from agency.agresources import detect_gpus

        mock = MagicMock()
        mock.returncode = 1
        mock.stdout = ""
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock)
        assert detect_gpus() == []

    def test_nvidia_smi_parses_indices(self, monkeypatch):
        from unittest.mock import MagicMock
        from agency.agresources import detect_gpus

        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "0\n1\n2\n"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock)
        # Isolate from whatever CUDA_VISIBLE_DEVICES happens to be set to on
        # the host running this test -- this test is about nvidia-smi output
        # parsing, not _cvd_filter (see TestCvdFilter for that).
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert detect_gpus() == [0, 1, 2]

    def test_cvd_filter_applied_to_nvidia_smi_output(self, monkeypatch):
        from unittest.mock import MagicMock
        from agency.agresources import detect_gpus

        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "0\n1\n2\n3\n"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
        assert detect_gpus() == [1, 3]

    def test_cvd_unset_returns_all_from_nvidia_smi(self, monkeypatch):
        from unittest.mock import MagicMock
        from agency.agresources import detect_gpus

        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "0\n1\n2\n"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert detect_gpus() == [0, 1, 2]


# ---------------------------------------------------------------------------
# _cvd_filter
# ---------------------------------------------------------------------------


class TestCvdFilter:
    def test_no_env_var_passes_all(self, monkeypatch):
        from agency.agresources import _cvd_filter

        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert _cvd_filter([0, 1, 2, 3]) == [0, 1, 2, 3]

    def test_filters_to_allowed_subset(self, monkeypatch):
        from agency.agresources import _cvd_filter

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,3,5,7")
        assert _cvd_filter([0, 1, 2, 3, 4, 5, 6, 7]) == [0, 3, 5, 7]

    def test_single_gpu(self, monkeypatch):
        from agency.agresources import _cvd_filter

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
        assert _cvd_filter([0, 1, 2, 3]) == [3]

    def test_empty_string_passes_all(self, monkeypatch):
        from agency.agresources import _cvd_filter

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert _cvd_filter([0, 1, 2]) == [0, 1, 2]

    def test_nodevfiles_passes_all(self, monkeypatch):
        from agency.agresources import _cvd_filter

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "NoDevFiles")
        assert _cvd_filter([0, 1, 2]) == [0, 1, 2]

    def test_none_string_passes_all(self, monkeypatch):
        from agency.agresources import _cvd_filter

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "none")
        assert _cvd_filter([0, 1, 2]) == [0, 1, 2]

    def test_cvd_id_not_in_pool_ignored(self, monkeypatch):
        from agency.agresources import _cvd_filter

        # CVD says GPU 9 is allowed but nvidia-smi only reported [0,1,2]
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,9")
        assert _cvd_filter([0, 1, 2]) == [0]

    def test_preserves_order_from_pool_list(self, monkeypatch):
        from agency.agresources import _cvd_filter

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,3,1")
        # Order follows the pool list, not CVD order
        assert _cvd_filter([0, 1, 2, 3, 4, 5]) == [1, 3, 5]

    def test_pool_auto_detect_respects_cvd(self, monkeypatch):
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "0\n1\n2\n3\n"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,3,5,7")
        pool = agResourcePool()
        # Only GPUs 0 and 3 overlap between [0,1,2,3] and {0,3,5,7}
        assert pool.gpus == [0, 3]


class TestDetectCpus:
    def test_returns_positive_int(self):
        from agency.agresources import detect_cpus

        cpus = detect_cpus()
        assert isinstance(cpus, int)
        assert cpus >= 1

    def test_os_cpu_count_none_returns_one(self, monkeypatch):
        from agency.agresources import detect_cpus

        monkeypatch.setattr("os.cpu_count", lambda: None)
        assert detect_cpus() == 1


class TestDetectMemoryMb:
    def test_returns_positive_int(self):
        from agency.agresources import detect_memory_mb

        mb = detect_memory_mb()
        assert isinstance(mb, int)
        assert mb > 0

    def test_fallback_when_proc_missing(self, monkeypatch, tmp_path):
        from unittest.mock import MagicMock
        from agency.agresources import detect_memory_mb

        # Point /proc/meminfo to a non-existent path and make sysctl fail
        monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(OSError()))
        mock = MagicMock()
        mock.returncode = 1
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock)
        assert detect_memory_mb() == 4096  # safe fallback


class TestPoolAutoDetect:
    def test_pool_auto_detects_gpus(self, monkeypatch):
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "0\n1\n"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock)
        # Isolate from the host's real CUDA_VISIBLE_DEVICES -- this test is
        # about auto-detection wiring, not CVD filtering (see
        # test_pool_auto_detect_respects_cvd for that).
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        pool = agResourcePool()
        assert pool.gpus == [0, 1]

    def test_pool_auto_detects_cpus(self, monkeypatch):
        monkeypatch.setattr("os.cpu_count", lambda: 16)
        pool = agResourcePool(gpus=[])
        assert pool.total_cpus == 16

    def test_pool_explicit_overrides_detection(self):
        pool = agResourcePool(gpus=[0], total_cpus=4, total_memory_mb=8192)
        assert pool.gpus == [0]
        assert pool.total_cpus == 4
        assert pool.total_memory_mb == 8192

    def test_agent_has_default_pool(self):
        from agency.agent import agent

        assert agent.agresource_pool is not None
        assert isinstance(agent.agresource_pool.total_cpus, int)
        assert isinstance(agent.agresource_pool.total_memory_mb, int)


# ---------------------------------------------------------------------------
# agResourcePool
# ---------------------------------------------------------------------------


class TestAgResourcePool:
    def test_single_gpu_acquire_release(self):
        pool = agResourcePool(gpus=[0])
        gid = pool.acquire_gpu()
        assert gid == 0
        pool.release_gpu(gid)

    def test_two_gpus_both_acquired(self):
        pool = agResourcePool(gpus=[0, 1])
        g1 = pool.acquire_gpu()
        g2 = pool.acquire_gpu()
        assert {g1, g2} == {0, 1}
        pool.release_gpu(g1)
        pool.release_gpu(g2)

    def test_acquire_blocks_until_released(self):
        pool = agResourcePool(gpus=[0])
        pool.acquire_gpu()  # hold the only GPU

        acquired: list[int] = []

        def _waiter():
            acquired.append(pool.acquire_gpu())

        t = threading.Thread(target=_waiter)
        t.start()
        time.sleep(0.1)
        assert acquired == []  # still blocked
        pool.release_gpu(0)
        t.join(timeout=2)
        assert acquired == [0]

    def test_acquire_timeout_raises(self):
        pool = agResourcePool(gpus=[0])
        pool.acquire_gpu()  # exhaust pool
        with pytest.raises(TimeoutError):
            pool.acquire_gpu(timeout=0.2)

    def test_release_unowned_gpu_is_safe(self):
        pool = agResourcePool(gpus=[0])
        pool.release_gpu(0)  # never acquired — should not raise

    def test_release_unknown_gpu_is_safe(self):
        pool = agResourcePool(gpus=[0])
        pool.release_gpu(99)  # not in pool — should not raise

    def test_repr(self):
        pool = agResourcePool(gpus=[0, 1], idle_cpus=1.0, idle_memory="1g")
        r = repr(pool)
        assert "agResourcePool" in r
        assert "[0, 1]" in r
        assert "total_cpus" in r
        assert "total_memory_mb" in r


# ---------------------------------------------------------------------------
# GPU presence markers
# ---------------------------------------------------------------------------


class TestGpuMarkers:
    """GPU markers are now allocated in-process via ctypes (no subprocesses)."""

    def test_mark_gpus_false_does_not_call_allocate(self):
        from agency import agresources

        with patch.object(agresources, "_allocate_gpu_markers") as mock_alloc:
            agResourcePool(gpus=[0, 1], mark_gpus=False)
        mock_alloc.assert_not_called()

    def test_mark_gpus_true_empty_gpu_list_does_not_call_allocate(self):
        from agency import agresources

        with patch.object(agresources, "_allocate_gpu_markers") as mock_alloc:
            agResourcePool(gpus=[], mark_gpus=True)
        mock_alloc.assert_not_called()

    def test_mark_gpus_true_calls_allocate_with_gpu_list(self):
        from agency import agresources

        with patch.object(agresources, "_allocate_gpu_markers") as mock_alloc:
            agResourcePool(gpus=[0, 1], mark_gpus=True)
        mock_alloc.assert_called_once_with([0, 1])

    def test_mark_gpus_true_single_gpu_calls_allocate(self):
        from agency import agresources

        with patch.object(agresources, "_allocate_gpu_markers") as mock_alloc:
            agResourcePool(gpus=[2], mark_gpus=True)
        mock_alloc.assert_called_once_with([2])

    def test_no_marker_procs_attribute(self):
        pool = agResourcePool(gpus=[0], mark_gpus=False)
        assert not hasattr(pool, "_marker_procs")

    def test_no_stop_gpu_markers_method(self):
        pool = agResourcePool(gpus=[0], mark_gpus=False)
        assert not hasattr(pool, "_stop_gpu_markers")

    def test_allocate_gpu_markers_skips_on_no_libcuda(self):
        from agency.agresources import _allocate_gpu_markers
        import ctypes

        with patch.object(ctypes, "CDLL", side_effect=OSError("libcuda.so.1 not found")):
            _allocate_gpu_markers([0, 1])  # must not raise

    def test_allocate_gpu_markers_skips_on_cuinit_failure(self):
        from agency.agresources import _allocate_gpu_markers
        import ctypes

        mock_cuda = MagicMock()
        mock_cuda.cuInit.return_value = 1  # CUDA_ERROR_NOT_INITIALIZED
        with patch.object(ctypes, "CDLL", return_value=mock_cuda):
            _allocate_gpu_markers([0])  # must not raise
        mock_cuda.cuCtxCreate_v2.assert_not_called()

    def test_allocate_gpu_markers_remaps_cuda_device_indices_with_cvd(self):
        """When CUDA_VISIBLE_DEVICES=0,3,5,7, physical IDs must be remapped to
        CUDA device indices 0-3 before calling cuCtxCreate_v2.  This is the
        exact bug that caused markers to be missing on GPUs 3 and 5."""
        from agency.agresources import _allocate_gpu_markers
        import ctypes

        mock_cuda = MagicMock()
        mock_cuda.cuInit.return_value = 0  # success
        mock_cuda.cuCtxCreate_v2.return_value = 0
        mock_cuda.cuMemAlloc_v2.return_value = 0
        with patch.object(ctypes, "CDLL", return_value=mock_cuda):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,3,5,7"}):
                _allocate_gpu_markers([0, 3, 5, 7])
        # Extract the device argument (3rd positional arg) from each call
        called_devs = [call.args[2] for call in mock_cuda.cuCtxCreate_v2.call_args_list]
        assert called_devs == [0, 1, 2, 3], (
            f"Expected CUDA device indices [0,1,2,3], got {called_devs}. "
            "Physical GPU IDs were passed directly instead of being remapped."
        )

    def test_allocate_gpu_markers_falls_back_to_rocm_when_no_libcuda(self):
        """On an AMD-only host (no libcuda.so.1 at all), markers must be
        allocated via ROCm/HIP instead of silently doing nothing."""
        from agency.agresources import _allocate_gpu_markers
        import ctypes

        mock_hip = MagicMock()
        mock_hip.hipInit.return_value = 0
        mock_hip.hipSetDevice.return_value = 0
        mock_hip.hipMalloc.return_value = 0

        def _cdll(name, *a, **kw):
            if name == "libcuda.so.1":
                raise OSError("libcuda.so.1 not found")
            assert name == "libamdhip64.so"
            return mock_hip

        with patch.object(ctypes, "CDLL", side_effect=_cdll):
            _allocate_gpu_markers([0, 1])
        assert mock_hip.hipSetDevice.call_count == 2
        assert mock_hip.hipMalloc.call_count == 2

    def test_allocate_gpu_markers_skips_rocm_on_hipinit_failure(self):
        from agency.agresources import _allocate_gpu_markers
        import ctypes

        mock_hip = MagicMock()
        mock_hip.hipInit.return_value = 1  # failure

        def _cdll(name, *a, **kw):
            if name == "libcuda.so.1":
                raise OSError("libcuda.so.1 not found")
            return mock_hip

        with patch.object(ctypes, "CDLL", side_effect=_cdll):
            _allocate_gpu_markers([0])  # must not raise
        mock_hip.hipSetDevice.assert_not_called()

    def test_allocate_gpu_markers_skips_entirely_when_neither_cuda_nor_rocm_present(self):
        from agency.agresources import _allocate_gpu_markers
        import ctypes

        with patch.object(ctypes, "CDLL", side_effect=OSError("not found")):
            _allocate_gpu_markers([0, 1])  # must not raise

    def test_allocate_gpu_markers_remaps_hip_device_indices_with_hip_visible_devices(self):
        """When HIP_VISIBLE_DEVICES=1,3, physical IDs must be remapped to HIP
        device indices 0-1 before calling hipSetDevice -- same remap bug class
        as the CUDA/CUDA_VISIBLE_DEVICES case above, for the ROCm path."""
        from agency.agresources import _allocate_gpu_markers
        import ctypes

        mock_hip = MagicMock()
        mock_hip.hipInit.return_value = 0
        mock_hip.hipSetDevice.return_value = 0
        mock_hip.hipMalloc.return_value = 0

        def _cdll(name, *a, **kw):
            if name == "libcuda.so.1":
                raise OSError("libcuda.so.1 not found")
            return mock_hip

        with patch.object(ctypes, "CDLL", side_effect=_cdll):
            with patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "1,3"}):
                _allocate_gpu_markers([1, 3])
        called_devs = [call.args[0] for call in mock_hip.hipSetDevice.call_args_list]
        assert called_devs == [0, 1], (
            f"Expected HIP device indices [0,1], got {called_devs}. "
            "Physical GPU IDs were passed directly instead of being remapped."
        )

    def test_allocate_gpu_markers_does_not_try_rocm_when_cuda_available(self):
        """CUDA present and working -- ROCm/HIP must never be attempted."""
        from agency.agresources import _allocate_gpu_markers
        import ctypes

        mock_cuda = MagicMock()
        mock_cuda.cuInit.return_value = 0
        mock_cuda.cuCtxCreate_v2.return_value = 0
        mock_cuda.cuMemAlloc_v2.return_value = 0
        with patch.object(ctypes, "CDLL", return_value=mock_cuda) as cdll:
            _allocate_gpu_markers([0])
        cdll.assert_called_once_with("libcuda.so.1")

    def test_non_main_process_name_blocks_allocation(self):
        """The MainProcess guard must block _allocate_gpu_markers in worker processes."""
        from agency import agresources

        mock_proc = MagicMock()
        mock_proc.name = "ForkPoolWorker-1"
        with patch("multiprocessing.current_process", return_value=mock_proc):
            with patch.object(agresources, "_allocate_gpu_markers") as mock_alloc:
                agResourcePool(gpus=[0], mark_gpus=True)
        mock_alloc.assert_not_called()

    def test_subprocess_import_does_not_call_allocate(self):
        """Importing agent in a subprocess must not call _allocate_gpu_markers."""
        script = (
            "import sys; "
            "from unittest.mock import patch; "
            "from agency import agresources; "
            "calls = []; "
            "original = agresources._allocate_gpu_markers; "
            "agresources._allocate_gpu_markers = lambda ids: calls.append(ids) or original(ids); "
            "from agency.agent import agent; "
            "assert calls == [], f'allocate called: {calls}'"
        )
        child = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=15,
        )
        assert child.returncode == 0, child.stderr.decode()


# ---------------------------------------------------------------------------
# agSandbox — change_config / get_config_copy
# ---------------------------------------------------------------------------


class TestAgSandboxChangeConfigAndGetConfigCopy:
    def test_change_config_replaces_agconfig(self):
        from agency.agconfig import agConfig

        sb = _make_sandbox(agconfig=agConfig({"agllm_backend": {"temperature": 0.7}}))
        sb.change_config(agConfig({"agllm_backend": {"temperature": 0.2}}))
        assert sb._agconfig.get("agllm_backend", "temperature") == 0.2

    def test_change_config_clones_given_agconfig(self):
        from agency.agconfig import agConfig

        sb = _make_sandbox(agconfig=agConfig())
        new_cfg = agConfig({"agllm_backend": {"temperature": 0.2}})
        sb.change_config(new_cfg)
        new_cfg.agllm_backend.temperature = 0.9
        assert sb._agconfig.get("agllm_backend", "temperature") == 0.2

    def test_get_config_copy_returns_clone_not_same_object(self):
        from agency.agconfig import agConfig

        cfg = agConfig({"agllm_backend": {"temperature": 0.7}})
        sb = _make_sandbox(agconfig=cfg)
        copy = sb.get_config_copy()
        assert copy is not sb._agconfig

    def test_get_config_copy_reflects_current_values(self):
        from agency.agconfig import agConfig

        sb = _make_sandbox(agconfig=agConfig({"agllm_backend": {"temperature": 0.7}}))
        assert sb.get_config_copy().agllm_backend.temperature == 0.7

    def test_mutating_get_config_copy_does_not_affect_sandbox(self):
        from agency.agconfig import agConfig

        sb = _make_sandbox(agconfig=agConfig({"agllm_backend": {"temperature": 0.7}}))
        copy = sb.get_config_copy()
        copy.agllm_backend.temperature = 0.1
        assert sb._agconfig.get("agllm_backend", "temperature") == 0.7

    @docker
    def test_get_config_copy_none_when_no_agconfig(self):
        # Bypasses _make_sandbox()'s forced backend="docker" agconfig on purpose --
        # this test is specifically about the truly-no-agconfig-at-all pathway.
        from agency.agsandbox import agSandbox

        sb = agSandbox(str(uuid.uuid4()))
        try:
            assert sb.get_config_copy() is None
        finally:
            sb.destroy()

    def test_change_config_none_clears_agconfig(self):
        from agency.agconfig import agConfig

        sb = _make_sandbox(agconfig=agConfig({"agllm_backend": {"temperature": 0.7}}))
        sb.change_config(None)
        assert sb.get_config_copy() is None


# ---------------------------------------------------------------------------
# agSandbox — container lifecycle
# ---------------------------------------------------------------------------


class TestAgSandboxLifecycle:
    def test_lifecycle_tag_is_lowercase(self):
        """_lifecycle_tag() must be fully lowercase — Docker rejects uppercase repository names."""
        from agency.agsandbox_backends.docker import _DockerBackend

        backend = _DockerBackend.__new__(_DockerBackend)
        backend._name = "GenerationAgent_4816622_0000"
        tag = backend._lifecycle_tag()
        assert tag == tag.lower(), f"lifecycle tag must be lowercase, got {tag!r}"
        assert "generationagent" in tag

    def test_lifecycle_tag_format(self):
        from agency.agsandbox_backends.docker import _DockerBackend

        backend = _DockerBackend.__new__(_DockerBackend)
        backend._name = "myagent_0000"
        assert backend._lifecycle_tag() == "agency/lifecycle-myagent_0000"

    @docker
    def test_container_starts_and_destroys(self):
        sb = _make_sandbox()
        name = sb._backend._container_name()
        # Container is started lazily on first use
        sb.exec("true")
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        assert name in result.stdout
        sb.destroy()
        result2 = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        assert name not in result2.stdout

    @docker
    def test_commit_creates_image(self):
        sb = _make_sandbox()
        tag = f"agency/test-commit-{sb._agname}"
        try:
            sb.write_file("/workspace/marker.txt", "committed\n")
            sb.commit(tag)
            # Image should exist
            result = subprocess.run(
                ["docker", "images", "-q", tag],
                capture_output=True,
                text=True,
            )
            assert result.stdout.strip() != ""
        finally:
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
            sb.destroy()

    @docker
    def test_commit_works_on_a_sandbox_object_that_never_started_it_itself(self):
        """commit() must succeed on a sandbox object that never itself ran
        _ensure_started() but whose container is already running (started by
        a worker process or another sandbox instance) -- it always checks
        _container_running() directly, never a per-process memory flag.

        Simulated by creating two BACKEND objects with the same name:
        sb_worker starts the container, main_backend (whose own copy never
        touched it) tries to commit it. main_backend is built directly via
        agsandbox_backend.for_config() rather than a second agSandbox(...)
        call, since agSandbox's own agname is deduplicated on every
        construction (see agsandbox.py's __init__) -- passing the same
        agname through it again would produce a DIFFERENT identity, not
        simulate a second view of the same one (matching what cloudpickle
        actually does across worker processes: it preserves the
        already-computed name rather than reallocating it)."""
        from agency.agconfig import agConfig
        from agency.agsandbox import agSandbox
        from agency.agsandbox_backends import agSandboxBackendConfig, agsandbox_backend

        agname = str(uuid.uuid4())
        cfg = agConfig(agSandboxBackendConfig(backend="docker"))
        sb_worker = agSandbox(agname, agconfig=cfg)  # "worker" — starts the container
        main_backend = agsandbox_backend.for_config(
            cfg,
            agname=sb_worker._backend._agname,
            name=sb_worker._backend._name,
            checkpoint_image=None,
            base_image=sb_worker.base_image,
            mounts={},
        )  # "main process" — same name, never started it
        tag = f"agency/test-commit-started-false-{agname[:8]}"
        try:
            # Worker starts container and writes a file.
            sb_worker.write_file("/workspace/marker.txt", "worker-written\n")
            # commit() must detect the running container via docker inspect and succeed.
            assert main_backend.commit(tag) is True
            result = subprocess.run(
                ["docker", "images", "-q", tag],
                capture_output=True,
                text=True,
            )
            assert result.stdout.strip() != "", (
                "image must exist even though main_backend never started it"
            )
        finally:
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
            sb_worker.destroy()

    @docker
    def test_commit_returns_false_when_container_not_running(self):
        """commit() must return False (not crash) if no container is running."""
        sb = _make_sandbox()
        tag = f"agency/test-commit-no-container-{sb._agname}"
        # No container was ever started — nothing to commit.
        assert sb.commit(tag) is False
        # No image should have been created.
        result = subprocess.run(
            ["docker", "images", "-q", tag],
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == ""

    @docker
    def test_ensure_started_reuses_running_container(self):
        """_ensure_started() must reuse a container already running in Docker rather
        than destroying it and starting fresh — the cross-worker-process file-persistence fix."""
        from agency.agconfig import agConfig
        from agency.agsandbox_backends import agSandboxBackendConfig, agsandbox_backend

        sb = _make_sandbox()
        # Start the container and write a sentinel file.
        sb.write_file("/workspace/persist.txt", "still-here\n")
        assert sb._backend._container_running() is True
        # A second backend object with the SAME name, standing in for a fresh
        # worker-process copy (matching what cloudpickle actually does --
        # preserving the already-computed _name rather than reallocating it).
        # Built directly via agsandbox_backend.for_config() rather than a
        # second agSandbox(...) call: agSandbox's own agname is deduplicated
        # on every construction (see agsandbox.py's __init__), so passing
        # sb's agname through it again would produce a DIFFERENT name, not
        # the same one this test needs to simulate reuse.
        cfg = agConfig(agSandboxBackendConfig(backend="docker"))
        worker_backend = agsandbox_backend.for_config(
            cfg,
            agname=sb._backend._agname,
            name=sb._backend._name,
            checkpoint_image=None,
            base_image=sb.base_image,
            mounts={},
        )
        worker_backend._ensure_started()
        # The file written before the reset must still be present.
        content = sb.read_file("/workspace/persist.txt")
        assert "still-here" in content
        sb.destroy()

    @docker
    def test_files_persist_across_process_pool_tool_calls(self):
        """Files written by the write tool in one worker process must be readable
        by the read tool in a subsequent worker process call (regression test for
        the cross-worker container-destruction bug)."""
        sb = _make_sandbox()
        from agency.tools import make_sandboxed_tools

        tools = {t.name: t for t in make_sandboxed_tools(sb)}
        try:
            # write runs in a process-pool worker
            w = tools["write"](agdata(file_path="/workspace/cross.txt", content="cross-worker\n"))
            assert not isinstance(w, agerror), f"write failed: {w}"
            # read also runs in a process-pool worker; must find the file
            r = tools["read"](agdata(file_path="/workspace/cross.txt"))
            assert not isinstance(r, agerror), f"read failed after cross-worker write: {r}"
            assert "cross-worker" in r.content
        finally:
            sb.destroy()

    @docker
    def test_checkpoint_restore_preserves_files(self):
        """sb2 must start its container FROM the tag (a real `docker run`,
        which only needs the image to still exist at that moment) BEFORE
        sb1.destroy() runs: commit(tag) sets self._checkpoint_image = tag
        (see commit()'s docstring), and destroy() unconditionally removes
        whatever image self._checkpoint_image points at -- exactly this
        tag. Once sb2's container has actually been created from it,
        removing the tag afterward is irrelevant (the container already
        holds everything it needs)."""
        tag = f"agency/test-ckpt-restore-{__import__('uuid').uuid4().hex[:8]}"
        sb1 = _make_sandbox()
        sb2 = None
        try:
            sb1.write_file("/workspace/data.txt", "restored\n")
            sb1.commit(tag)

            sb2 = _make_sandbox(checkpoint_image=tag)
            content = sb2.read_file("/workspace/data.txt")
            assert content == "restored\n"
        finally:
            sb1.destroy()
            if sb2 is not None:
                sb2.destroy()
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)

    @docker
    def test_output_dir_agent_can_write_and_read(self, tmp_path):
        agname = "test-agent"
        output_dir = tmp_path / "agent_output" / agname
        sb = _make_sandbox(agconfig=_agconfig_with_output_dir(output_dir))
        out, rc = sb.exec("echo hello > /agent_output/result.txt")
        assert rc == 0
        assert (output_dir / "result.txt").read_text().strip() == "hello"
        sb.destroy()

    @docker
    def test_output_dir_shared_across_agents(self, tmp_path):
        # Each agent gets its own subdir; they can still see each other's files
        # via the parent mount if needed, but here we test per-agent isolation.
        out_dir1 = tmp_path / "agent_output" / "brave-fox"
        out_dir2 = tmp_path / "agent_output" / "swift-hawk"
        sb1 = _make_sandbox(agconfig=_agconfig_with_output_dir(out_dir1))
        sb2 = _make_sandbox(agconfig=_agconfig_with_output_dir(out_dir2))
        sb1.exec("echo from_agent1 > /agent_output/out.txt")
        out, rc = sb1.exec("cat /agent_output/out.txt")
        assert rc == 0
        assert "from_agent1" in out
        assert (out_dir1 / "out.txt").read_text().strip() == "from_agent1"
        sb1.destroy()
        sb2.destroy()

    @docker
    def test_commit_creates_checkpoint_image_without_removing_container(self):
        """commit() (the old stop(commit=True)'s checkpoint half, now a
        standalone call) commits state to agency/lifecycle-<name> WITHOUT
        touching the container's existence at all -- it keeps running
        (checkpointing in place) exactly as before, no run/rm involved."""
        sb = _make_sandbox()
        name = sb._backend._container_name()
        lifecycle_tag = sb._backend._lifecycle_tag()
        try:
            sb.write_file("/workspace/marker.txt", "lifecycle\n")
            assert sb.commit() is True
            # Container must still be running -- commit() never removes it.
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
            )
            assert name in result.stdout, "container must still be running after commit()"
            # Lifecycle image must exist
            img = subprocess.run(
                ["docker", "images", "-q", lifecycle_tag],
                capture_output=True,
                text=True,
            )
            assert img.stdout.strip() != "", "lifecycle image must exist after commit()"
            # _checkpoint_image must be set
            assert sb._checkpoint_image == lifecycle_tag
        finally:
            subprocess.run(["docker", "rmi", "-f", lifecycle_tag], capture_output=True)
            sb.destroy()

    @docker
    def test_rm_container_removes_container_after_commit(self):
        """rm_container() removes the container as a separate, explicit step
        from commit() -- the old single stop(commit=True) call is now this
        composition of two independent calls."""
        sb = _make_sandbox()
        name = sb._backend._container_name()
        lifecycle_tag = sb._backend._lifecycle_tag()
        try:
            sb.write_file("/workspace/marker.txt", "lifecycle\n")
            sb.commit()
            sb.rm_container()
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
            )
            assert name not in result.stdout, "container must be removed after rm_container()"
        finally:
            subprocess.run(["docker", "rmi", "-f", lifecycle_tag], capture_output=True)
            sb.destroy()

    @docker
    def test_rm_container_removes_container_without_image(self):
        """rm_container() removes the container but does not create a lifecycle image."""
        sb = _make_sandbox()
        name = sb._backend._container_name()
        lifecycle_tag = sb._backend._lifecycle_tag()
        try:
            sb.write_file("/workspace/dirty.txt", "dirty\n")
            previous_lifecycle = sb._checkpoint_image  # None on first call
            sb.rm_container()
            # Container must be gone
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
            )
            assert name not in result.stdout, "container must be removed after rm_container()"
            # _checkpoint_image must not have changed
            assert sb._checkpoint_image == previous_lifecycle
            # No lifecycle image should have been created
            img = subprocess.run(
                ["docker", "images", "-q", lifecycle_tag],
                capture_output=True,
                text=True,
            )
            assert img.stdout.strip() == "", "rm_container() must not create a lifecycle image"
        finally:
            subprocess.run(["docker", "rmi", "-f", lifecycle_tag], capture_output=True)
            sb.destroy()

    @docker
    def test_commit_then_rm_container_restores_workspace_on_next_start(self):
        """After commit()+rm_container() (the old stop(commit=True)'s full
        effect, now two explicit calls), _ensure_started() restores
        /workspace from the lifecycle image via a fresh `docker run`."""
        sb = _make_sandbox()
        lifecycle_tag = sb._backend._lifecycle_tag()
        try:
            sb.write_file("/workspace/persistent.txt", "saved\n")
            sb.commit()
            sb.rm_container()
            assert not sb._backend._container_running()
            # Next exec triggers _ensure_started() which runs docker run from lifecycle image.
            out, rc = sb.exec("cat /workspace/persistent.txt")
            assert rc == 0
            assert "saved" in out
        finally:
            subprocess.run(["docker", "rmi", "-f", lifecycle_tag], capture_output=True)
            sb.destroy()

    @docker
    def test_stop_then_exec_resumes_same_container_with_workspace_intact(self):
        """The core new capability this refactor exists for: stop() alone
        (hibernate -- no commit, no image, no rm at all) followed by a
        later exec() resumes the SAME container via `docker start`, with
        the workspace fully intact and untouched. This is materially
        cheaper than commit()+rm_container()+run, and is now the ordinary
        per-tool-call path (see agtool.py)."""
        sb = _make_sandbox()
        name = sb._backend._container_name()
        try:
            sb.write_file("/workspace/persistent.txt", "saved\n")
            container_id_before = subprocess.run(
                ["docker", "inspect", "--format", "{{.Id}}", name],
                capture_output=True,
                text=True,
            ).stdout.strip()

            sb.stop()
            assert not sb._backend._container_running()
            assert sb._checkpoint_image is None, (
                "stop() must not create or touch any checkpoint image"
            )

            out, rc = sb.exec("cat /workspace/persistent.txt")
            assert rc == 0
            assert "saved" in out

            container_id_after = subprocess.run(
                ["docker", "inspect", "--format", "{{.Id}}", name],
                capture_output=True,
                text=True,
            ).stdout.strip()
            assert container_id_after == container_id_before, (
                "exec() after stop() must resume the SAME container (docker "
                "start), not create a fresh one"
            )
        finally:
            sb.destroy()

    @docker
    def test_rm_container_reverts_to_last_checkpoint(self):
        """commit() then rm_container() discards dirty state; next start
        restores from the last committed lifecycle image -- the old
        stop(commit=True)-then-stop(commit=False) revert sequence,
        expressed with the new split API."""
        sb = _make_sandbox()
        lifecycle_tag = sb._backend._lifecycle_tag()
        try:
            # First successful tool call: write file and commit.
            sb.write_file("/workspace/good.txt", "good\n")
            sb.commit()
            sb.rm_container()
            # Second tool call that fails: write a dirty file without committing.
            sb.exec("true")  # restarts from lifecycle image
            sb.write_file("/workspace/dirty.txt", "dirty\n")
            sb.rm_container()
            # Next start must restore from lifecycle image — dirty.txt must not exist.
            sb.exec("true")
            content = sb.read_file("/workspace/good.txt")
            assert "good" in content
            _, dirty_rc = sb.exec("test -f /workspace/dirty.txt")
            assert dirty_rc != 0, "dirty file must not exist after rm_container()"
        finally:
            subprocess.run(["docker", "rmi", "-f", lifecycle_tag], capture_output=True)
            sb.destroy()

    @docker
    def test_destroy_removes_checkpoint_image(self):
        """destroy() cleans up the lifecycle image created by commit()."""
        sb = _make_sandbox()
        lifecycle_tag = sb._backend._lifecycle_tag()
        sb.write_file("/workspace/x.txt", "x\n")
        sb.commit()
        # Confirm image exists before destroy
        img = subprocess.run(
            ["docker", "images", "-q", lifecycle_tag],
            capture_output=True,
            text=True,
        )
        assert img.stdout.strip() != "", "lifecycle image must exist before destroy()"
        sb.destroy()
        # Image must be gone
        img2 = subprocess.run(
            ["docker", "images", "-q", lifecycle_tag],
            capture_output=True,
            text=True,
        )
        assert img2.stdout.strip() == "", "destroy() must remove the lifecycle image"

    @docker
    def test_rm_container_retries_rm_on_first_failure(self):
        """rm_container() retries docker rm -f up to 3 times; succeeds if a later attempt works."""
        sb = _make_sandbox()
        sb.write_file("/workspace/x.txt", "x\n")
        name = sb._backend._container_name()

        call_count = [0]
        real_run = sb._backend._run

        def flaky_run(cmd, **kwargs):
            if "rm" in cmd and "-f" in cmd and name in cmd:
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("simulated rm -f failure")
            return real_run(cmd, **kwargs)

        sb._backend._run = flaky_run
        sb.rm_container()

        assert call_count[0] == 2, "expected one failure then one success"
        assert not sb._backend._container_running()
        # Container must actually be gone after the successful retry
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        assert name not in result.stdout

    @docker
    def test_rm_container_raises_after_all_retries_fail(self):
        """rm_container() raises rather than silently warning when rm -f fails all 3
        attempts -- the caller must see that the container was not confirmed removed."""
        sb = _make_sandbox()
        sb.write_file("/workspace/x.txt", "x\n")

        real_run = sb._backend._run

        def always_fail_rm(cmd, **kwargs):
            if "rm" in cmd and "-f" in cmd:
                raise RuntimeError("simulated persistent failure")
            return real_run(cmd, **kwargs)

        sb._backend._run = always_fail_rm
        try:
            with pytest.raises(RuntimeError, match="simulated persistent failure"):
                sb.rm_container()
        finally:
            # Force cleanup bypassing our mock
            sb._backend._run = real_run
            sb.destroy()

    @docker
    def test_rm_container_then_destroy_after_rm_failure_releases_slot_once(self):
        """If rm_container()'s rm -f exhausts retries, the runtime slot is still held.

        destroy() must be the one that releases it -- exactly once -- when
        the container is actually removed. Regression test for a double
        release of _container_semaphore when both rm_container() and
        destroy() each independently believed they owed a release.
        """
        from agency.agsandbox_backends.container import _container_semaphore

        sb = _make_sandbox()
        sb.write_file("/workspace/x.txt", "x\n")

        real_run = sb._backend._run

        def always_fail_rm(cmd, **kwargs):
            if "rm" in cmd and "-f" in cmd:
                raise RuntimeError("simulated persistent failure")
            return real_run(cmd, **kwargs)

        # write_file() above already forced _ensure_started(), which
        # acquired the sandbox's runtime slot -- capture the value with that
        # slot already held.
        held = _container_semaphore._semlock._get_value()

        sb._backend._run = always_fail_rm
        try:
            with pytest.raises(RuntimeError, match="simulated persistent failure"):
                sb.rm_container()
        finally:
            sb._backend._run = real_run

        # rm never actually succeeded, so the slot must NOT have been
        # released yet -- the container is still really running.
        assert _container_semaphore._semlock._get_value() == held

        sb.destroy()

        # destroy() removes the container for real this time and releases
        # the slot exactly once -- not twice (which would over-credit the
        # semaphore above its true capacity).
        assert _container_semaphore._semlock._get_value() == held + 1

    @docker
    def test_rm_container_on_already_hibernating_container_releases_slot_once(self):
        """rm_container() called on a container that's ALREADY hibernating
        (stop() already ran and already released the runtime slot) must NOT
        release the slot a second time.

        This is the exact shape of a real skill-failure teardown: every
        prior tool call already hibernated the sandbox via stop() before
        agskill.py calls rm_container() on it. multiprocessing.Semaphore
        does not raise on over-release (confirmed empirically -- unlike
        threading.BoundedSemaphore), so a double-release here would
        silently over-credit the semaphore, eventually letting more
        containers run concurrently than the kernel keyring quota actually
        supports -- the same class of bug the quota exists to prevent.
        """
        from agency.agsandbox_backends.container import _container_semaphore

        sb = _make_sandbox()
        sb.write_file("/workspace/x.txt", "x\n")
        # write_file() above already forced _ensure_started(), which
        # acquired the sandbox's runtime slot -- capture the value with that
        # slot already held.
        held = _container_semaphore._semlock._get_value()

        sb.stop()
        assert _container_semaphore._semlock._get_value() == held + 1, (
            "stop() must release the slot exactly once"
        )

        sb.rm_container()
        assert _container_semaphore._semlock._get_value() == held + 1, (
            "rm_container() on an already-hibernating container must not "
            "release the slot again -- it was never re-acquired"
        )

        sb.destroy()
        assert _container_semaphore._semlock._get_value() == held + 1, (
            "destroy() on an already-removed container must not release the slot again either"
        )

    @docker
    def test_concurrent_docker_calls_gated_by_docker_semaphore(self):
        """All docker calls go through _run() which holds _docker_semaphore; peak concurrency <= 8."""
        from agency.agsandbox_backends.container import _docker_semaphore

        sandboxes = [_make_sandbox() for _ in range(4)]
        lifecycle_tags = [sb._backend._lifecycle_tag() for sb in sandboxes]
        for sb in sandboxes:
            sb.write_file("/workspace/x.txt", "x\n")

        concurrent = [0]
        peak = [0]
        lock = threading.Lock()
        real_acquire = _docker_semaphore.acquire
        real_release = _docker_semaphore.release

        def counting_acquire(*a, **kw):
            real_acquire(*a, **kw)
            with lock:
                concurrent[0] += 1
                peak[0] = max(peak[0], concurrent[0])

        def counting_release(*a, **kw):
            with lock:
                concurrent[0] -= 1
            real_release(*a, **kw)

        _docker_semaphore.acquire = counting_acquire
        _docker_semaphore.release = counting_release
        try:
            # commit() exercises docker commit + the depth-check inspect; both
            # must go through the semaphore.
            threads = [threading.Thread(target=sb.commit) for sb in sandboxes]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            _docker_semaphore.acquire = real_acquire
            _docker_semaphore.release = real_release
            for tag in lifecycle_tags:
                subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
            for sb in sandboxes:
                sb.destroy()

        assert peak[0] <= 16, (
            f"peak concurrent docker calls {peak[0]} exceeded semaphore limit of 16"
        )

    @docker
    def test_ensure_started_removes_created_state_container(self):
        """_ensure_started() force-removes a container stuck in 'Created' state before docker run."""
        sb = _make_sandbox()
        name = sb._backend._container_name()
        try:
            # Manually create a container in 'Created' state (no --detach run, just create).
            subprocess.run(
                ["docker", "create", "--name", name, sb.base_image, "tail", "-f", "/dev/null"],
                capture_output=True,
                check=True,
            )
            status = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Status}}", name],
                capture_output=True,
                text=True,
            )
            assert status.stdout.strip() == "created"
            # _ensure_started() must remove the stuck container and start fresh.
            sb._backend._ensure_started()
            assert sb._backend._container_running() is True
            out, rc = sb.exec("echo ok")
            assert rc == 0 and "ok" in out
        finally:
            sb.destroy()

    @docker
    def test_commit_retries_on_first_failure(self):
        """commit() retries docker commit up to 3 times; succeeds if a later attempt works."""
        sb = _make_sandbox()
        sb.write_file("/workspace/x.txt", "x\n")
        lifecycle_tag = sb._backend._lifecycle_tag()

        call_count = [0]
        real_run = sb._backend._run

        def flaky_run(cmd, **kwargs):
            if "commit" in cmd:
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("simulated commit failure")
            return real_run(cmd, **kwargs)

        sb._backend._run = flaky_run
        try:
            assert sb.commit() is True
            assert call_count[0] == 2, "expected one failure then one success"
            assert sb._checkpoint_image == lifecycle_tag
            assert sb._backend._container_running(), (
                "commit() must never touch the container's existence"
            )
            img = subprocess.run(
                ["docker", "images", "-q", lifecycle_tag],
                capture_output=True,
                text=True,
            )
            assert img.stdout.strip() != "", "lifecycle image must exist after successful retry"
        finally:
            sb._backend._run = real_run
            subprocess.run(["docker", "rmi", "-f", lifecycle_tag], capture_output=True)
            sb.destroy()

    @docker
    def test_commit_raises_after_all_retries_fail_container_still_running(self):
        """commit() raises when all 3 commit attempts fail -- and, unlike the
        old stop(commit=True), never touches the container's existence at
        all: the container must still exist and still be running afterward,
        exactly as before the failed commit attempt. _checkpoint_image stays
        at the prior tag (or None) so a subsequent successful commit() -- or
        a later rm_container() revert -- is unaffected."""
        sb = _make_sandbox()
        sb.write_file("/workspace/x.txt", "x\n")
        previous_lifecycle = sb._checkpoint_image

        real_run = sb._backend._run

        def always_fail_commit(cmd, **kwargs):
            if "commit" in cmd:
                raise RuntimeError("simulated persistent commit failure")
            return real_run(cmd, **kwargs)

        sb._backend._run = always_fail_commit
        try:
            with pytest.raises(RuntimeError, match="simulated persistent commit failure"):
                sb.commit()
        finally:
            sb._backend._run = real_run

        assert sb._checkpoint_image == previous_lifecycle  # not updated on all-retry failure
        assert sb._backend._container_running(), (
            "commit() must never remove or stop the container, even on total failure"
        )
        sb.destroy()

    @docker
    def test_ensure_started_resumes_exited_container(self):
        """An externally `docker stop`-ped container (state 'exited', not
        removed) is now RESUMED in place via `docker start` -- the core new
        hibernate/resume capability this refactor exists for -- rather than
        force-removed and recreated. The file written before the external
        stop must survive, and the exact same container (verified by
        container ID) must be reused, not recreated.
        """
        sb = _make_sandbox()
        name = sb._backend._container_name()
        try:
            sb.write_file("/workspace/exited.txt", "still-here\n")
            container_id_before = subprocess.run(
                ["docker", "inspect", "--format", "{{.Id}}", name],
                capture_output=True,
                text=True,
            ).stdout.strip()
            # Externally stop (not remove) the container — puts it in exited state.
            subprocess.run(["docker", "stop", "-t", "0", name], capture_output=True)
            # _ensure_started() must resume (docker start) this SAME container.
            sb._backend._ensure_started()
            assert sb._backend._container_running() is True
            container_id_after = subprocess.run(
                ["docker", "inspect", "--format", "{{.Id}}", name],
                capture_output=True,
                text=True,
            ).stdout.strip()
            assert container_id_after == container_id_before, (
                "an exited container must be resumed via `docker start`, "
                "not force-removed and recreated"
            )
            # The file survives, since the container was resumed, not recreated.
            content = sb.read_file("/workspace/exited.txt")
            assert content == "still-here\n"
        finally:
            sb.destroy()


# ---------------------------------------------------------------------------
# agSandbox — exec
# ---------------------------------------------------------------------------


class TestAgSandboxExec:
    @docker
    def setup_method(self, _):
        self.sb = _make_sandbox()

    @docker
    def teardown_method(self, _):
        self.sb.destroy()

    def test_exec_simple_command(self):
        out, rc = self.sb.exec("echo hello")
        assert rc == 0
        assert "hello" in out

    def test_exec_nonzero_exit(self):
        _, rc = self.sb.exec("exit 42")
        assert rc == 42

    def test_exec_stderr_captured(self):
        out, _ = self.sb.exec("echo err >&2")
        assert "err" in out

    def test_exec_workdir(self):
        self.sb.exec("mkdir -p /tmp/mydir")
        out, rc = self.sb.exec("pwd", workdir="/tmp/mydir")
        assert rc == 0
        assert "/tmp/mydir" in out

    def test_exec_cuda_env_prefix(self):
        self.sb._gpu_id = 3
        out, rc = self.sb.exec("echo $CUDA_VISIBLE_DEVICES")
        assert rc == 0
        assert "3" in out
        self.sb._gpu_id = None


# ---------------------------------------------------------------------------
# agSandbox — file I/O
# ---------------------------------------------------------------------------


class TestAgSandboxFileIO:
    @docker
    def setup_method(self, _):
        self.sb = _make_sandbox()

    @docker
    def teardown_method(self, _):
        self.sb.destroy()

    def test_write_and_read_file(self):
        self.sb.write_file("/workspace/hello.txt", "hello world\n")
        content = self.sb.read_file("/workspace/hello.txt")
        assert content == "hello world\n"

    def test_write_creates_parent_directories(self):
        self.sb.write_file("/workspace/a/b/c.txt", "deep\n")
        content = self.sb.read_file("/workspace/a/b/c.txt")
        assert "deep" in content

    def test_write_special_characters(self):
        payload = "line1\nline2\ttab\n$VAR 'quotes' \"double\"\n"
        self.sb.write_file("/workspace/special.txt", payload)
        assert self.sb.read_file("/workspace/special.txt") == payload

    def test_read_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            self.sb.read_file("/workspace/does_not_exist.txt")

    @docker
    def test_read_directory_raises_is_a_directory_error(self):
        with pytest.raises(IsADirectoryError):
            self.sb.read_file("/workspace")

    @docker
    def test_read_binary_file_raises_unicode_decode_error(self):
        # PNG magic bytes — not valid UTF-8. Use write_file_bytes to guarantee
        # binary content regardless of shell printf \x-escape support.
        self.sb.write_file_bytes("/workspace/binary.bin", bytes([0x89, 0x50, 0x4E, 0x47]))
        with pytest.raises(UnicodeDecodeError):
            self.sb.read_file("/workspace/binary.bin")


class TestAgSandboxReadFileUnit:
    """Unit tests for read_file error cases — no Docker required.

    read_file()'s base64-decode/error-mapping logic lives on the shared
    agsandbox_backends.container._ContainerBackendBase, so these unit tests
    exercise it directly rather than through the agSandbox facade.
    """

    def _make_sb(self):
        from agency.agsandbox_backends.container import _ContainerBackendBase

        sb = _ContainerBackendBase.__new__(_ContainerBackendBase)
        return sb

    def test_read_file_returns_text_content(self):
        import base64

        sb = self._make_sb()
        b64 = base64.b64encode(b"hello world\n").decode()
        with patch.object(sb, "_container_exec", return_value=(b64, 0)):
            assert sb.read_file("/workspace/hello.txt") == "hello world\n"

    def test_read_file_missing_path_raises_file_not_found(self):
        sb = self._make_sb()
        # base64 fails (rc=1), test -d also fails (rc=1) → not a directory
        sb._container_exec = MagicMock(side_effect=[("", 1), ("", 1)])
        with pytest.raises(FileNotFoundError):
            sb.read_file("/workspace/missing.txt")

    def test_read_file_directory_raises_is_a_directory_error(self):
        sb = self._make_sb()
        # base64 fails (rc=1), test -d succeeds (rc=0) → it's a directory
        sb._container_exec = MagicMock(side_effect=[("", 1), ("", 0)])
        with pytest.raises(IsADirectoryError):
            sb.read_file("/workspace/outputs")

    def test_read_file_binary_raises_unicode_decode_error(self):
        import base64

        sb = self._make_sb()
        raw = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])  # PNG header
        b64 = base64.b64encode(raw).decode()
        with patch.object(sb, "_container_exec", return_value=(b64, 0)):
            with pytest.raises(UnicodeDecodeError):
                sb.read_file("/workspace/image.png")

    def test_read_file_valid_utf8_succeeds(self):
        import base64

        sb = self._make_sb()
        content = "def main():\n    pass\n"
        b64 = base64.b64encode(content.encode("utf-8")).decode()
        with patch.object(sb, "_container_exec", return_value=(b64, 0)):
            assert sb.read_file("/workspace/core.py") == content


# ---------------------------------------------------------------------------
# agSandbox — PID tracking
# ---------------------------------------------------------------------------


class TestAgSandboxPIDTracking:
    @docker
    def setup_method(self, _):
        self.sb = _make_sandbox()

    @docker
    def teardown_method(self, _):
        self.sb.destroy()

    def test_background_pid_tracked(self):
        self.sb.exec("sleep 30 &")
        assert len(self.sb._watched_pids) > 0

    def test_foreground_spawned_child_tracked(self):
        # A foreground command that internally forks a child and exits.
        # The child escapes jobs -p but must still be captured via /proc diffing.
        self.sb.write_file(
            "/workspace/spawner.py",
            (
                "import subprocess, time\n"
                "subprocess.Popen(['sleep', '30'])\n"  # detached child, not waited on
            ),
        )
        self.sb.exec("python3 /workspace/spawner.py")
        assert len(self.sb._watched_pids) > 0

    def test_parent_exits_child_survives_still_tracked(self):
        # Parent spawns a child then exits. Child is reparented to PID 1 and
        # escapes any BFS from the original PID. Baseline diff must find it.
        self.sb.write_file(
            "/workspace/spawner.py",
            (
                "import subprocess, os\n"
                "subprocess.Popen(['sleep', '30'])\n"  # child detaches
                "os._exit(0)\n"  # parent exits immediately
            ),
        )
        self.sb.exec("python3 /workspace/spawner.py")
        live = self.sb.get_live_pids()
        # Parent is gone; the orphaned sleep child must still be tracked
        assert len(live) > 0

    def test_process_tree_descendants_tracked(self):
        # A process is tracked; it later spawns children of its own.
        # get_live_pids() must expand the tree and include those grandchildren.
        self.sb.write_file(
            "/workspace/parent.py",
            (
                "import subprocess, time\n"
                "# Spawn two long-lived children after a brief pause\n"
                "time.sleep(0.2)\n"
                "subprocess.Popen(['sleep', '30'])\n"
                "subprocess.Popen(['sleep', '30'])\n"
                "time.sleep(30)\n"  # parent also stays alive
            ),
        )
        self.sb.exec("python3 /workspace/parent.py &")
        time.sleep(0.5)  # let the parent spawn its children
        live = self.sb.get_live_pids()
        # parent + 2 children = at least 3 live PIDs
        assert len(live) >= 3

    def test_double_forked_daemon_tracked(self):
        # Classic Unix double-fork: grandchild is reparented to PID 1 and
        # completely detached from the shell's job table.
        self.sb.write_file(
            "/workspace/daemon.py",
            (
                "import os, time\n"
                "if os.fork() == 0:\n"  # first fork
                "    if os.fork() == 0:\n"  # second fork — grandchild
                "        time.sleep(30)\n"  # grandchild runs in background
                "    os._exit(0)\n"  # intermediate child exits
                "os.wait()\n"  # parent waits for intermediate child
            ),
        )
        self.sb.exec("python3 /workspace/daemon.py")
        assert len(self.sb._watched_pids) > 0

    def test_get_live_pids_returns_running(self):
        self.sb.exec("sleep 30 &")
        live = self.sb.get_live_pids()
        assert len(live) > 0

    def test_get_live_pids_removes_exited(self):
        self.sb.exec("sleep 0.1 &")
        time.sleep(1.0)
        live = self.sb.get_live_pids()
        assert len(live) == 0

    def test_get_live_pids_empty_when_no_background(self):
        self.sb.exec("echo hi")
        assert self.sb.get_live_pids() == set()

    def test_daemon_release_removes_pid_from_monitoring(self):
        self.sb.exec("sleep 30 &")
        live = self.sb.get_live_pids()
        assert len(live) > 0
        for pid in list(live):
            self.sb.release_daemon(pid)
        assert self.sb.get_live_pids() == set()

    def test_daemon_children_also_excluded(self):
        # Release a parent as daemon; children it spawns later must also be excluded.
        self.sb.write_file(
            "/workspace/daemon_parent.py",
            ("import subprocess, time\nsubprocess.Popen(['sleep', '30'])\ntime.sleep(30)\n"),
        )
        self.sb.exec("python3 /workspace/daemon_parent.py &")
        live = self.sb.get_live_pids()
        assert len(live) > 0
        # Release the parent; its child (sleep 30) should also be excluded
        for pid in list(live):
            self.sb.release_daemon(pid)
        time.sleep(0.3)  # let the child spawn
        assert self.sb.get_live_pids() == set()

    def test_pid_status_summary_no_processes(self):
        summary = self.sb.pid_status_summary()
        assert "no background" in summary

    def test_pid_status_summary_with_running_process(self):
        self.sb.exec("sleep 30 &")
        summary = self.sb.pid_status_summary()
        assert "PID" in summary
        assert "running" in summary


# ---------------------------------------------------------------------------
# agSandbox — resource limits
# ---------------------------------------------------------------------------


class TestAgSandboxResourceLimits:
    @docker
    def setup_method(self, _):
        self.sb = _make_sandbox()

    @docker
    def teardown_method(self, _):
        self.sb.destroy()

    def test_update_limits_cpu(self):
        # Should not raise; Docker applies the limit
        self.sb.update_limits(cpus=2.0)

    def test_update_limits_memory(self):
        self.sb.update_limits(memory="256m")

    def test_update_limits_both(self):
        self.sb.update_limits(cpus=1.0, memory="128m")

    def test_release_resources_clears_gpu(self):
        pool = agResourcePool(gpus=[0])
        gpu_id = pool.acquire_gpu()
        self.sb._gpu_id = gpu_id
        self.sb._gpu_virtual = True
        self.sb._gpu_release_fn = pool.release_gpu
        self.sb.release_resources(pool)
        assert self.sb._gpu_id is None
        assert self.sb._gpu_virtual is False
        assert pool._gpus_acquired == 0

    def test_release_resources_none_pool(self):
        # Should not raise even without a pool
        self.sb.release_resources(None)


# ---------------------------------------------------------------------------
# Sandboxed tool factories
# ---------------------------------------------------------------------------


class TestSandboxedTools:
    @docker
    def setup_method(self, _):
        self.sb = _make_sandbox()
        from agency.tools import make_sandboxed_tools

        self.tools = {t.name: t for t in make_sandboxed_tools(self.sb)}

    @docker
    def teardown_method(self, _):
        self.sb.destroy()

    def test_bash_tool_runs_in_container(self):
        result = self.tools["bash"].fn(agdata(command="hostname"))
        assert result.exit_code == 0
        assert result.output.strip() != ""

    def test_write_then_read_tool(self):
        self.tools["write"].fn(agdata(file_path="/workspace/t.txt", content="abc\n"))
        r = self.tools["read"].fn(agdata(file_path="/workspace/t.txt"))
        assert "abc" in r.content

    def test_glob_tool_finds_files(self):
        self.tools["write"].fn(agdata(file_path="/workspace/a.py", content="x\n"))
        self.tools["write"].fn(agdata(file_path="/workspace/b.py", content="y\n"))
        r = self.tools["glob"].fn(agdata(pattern="*.py", path="/workspace"))
        assert len(r.files) >= 2

    def test_grep_tool_finds_pattern(self):
        self.tools["write"].fn(agdata(file_path="/workspace/src.py", content="SECRET=42\n"))
        r = self.tools["grep"].fn(agdata(pattern="SECRET", path="/workspace"))
        assert any("SECRET" in m["text"] for m in r.matches)

    def test_edit_tool_replaces_content(self):
        self.tools["write"].fn(agdata(file_path="/workspace/edit_me.txt", content="foo bar\n"))
        self.tools["edit"].fn(
            agdata(
                file_path="/workspace/edit_me.txt",
                old_string="foo",
                new_string="baz",
            )
        )
        r = self.tools["read"].fn(agdata(file_path="/workspace/edit_me.txt"))
        assert "baz" in r.content
        assert "foo" not in r.content

    def test_daemon_release_tool_stops_monitoring(self):
        # Start a background process, get its PID, release it as daemon,
        # verify the outer loop would no longer wait for it.
        self.tools["bash"].fn(agdata(command="sleep 30 &"))
        live_before = self.sb.get_live_pids()
        assert len(live_before) > 0
        for pid in list(live_before):
            result = self.tools["daemon_release"].fn(agdata(pid=pid))
            assert not isinstance(result, agerror)
        assert self.sb.get_live_pids() == set()

    def test_daemon_release_tool_invalid_pid(self):
        result = self.tools["daemon_release"].fn(agdata(pid="notanint"))
        assert isinstance(result, agerror)

    def test_daemon_release_tool_missing_pid(self):
        result = self.tools["daemon_release"].fn(agdata())
        assert isinstance(result, agerror)


class TestResourceTools:
    @docker
    def setup_method(self, _):
        self.sb = _make_sandbox()
        self.pool = agResourcePool(gpus=[0, 1], idle_cpus=1.0, idle_memory="1024m")
        from agency.tools import make_sandboxed_tools

        self.tools = {t.name: t for t in make_sandboxed_tools(self.sb, self.pool)}

    @docker
    def teardown_method(self, _):
        self.sb.destroy()

    # ── reserve_gpu — virtual reservation only ─────────────────────────────

    def test_reserve_gpu_sets_virtual_flag_no_physical(self):
        """reserve_gpu sets _gpu_virtual=True but takes no physical GPU from the pool."""
        result = self.tools["reserve_gpu"].fn(agdata())
        assert getattr(result, "warning", None) is None
        assert self.sb._gpu_virtual is True
        assert self.sb._gpu_id is None
        assert self.pool._gpus_acquired == 0

    def test_reserve_gpu_idempotent(self):
        """Calling reserve_gpu twice returns an 'already acquired' message; flag unchanged."""
        self.tools["reserve_gpu"].fn(agdata())
        result = self.tools["reserve_gpu"].fn(agdata())
        assert self.sb._gpu_virtual is True
        assert "already" in result.message
        assert self.pool._gpus_acquired == 0

    def test_reserve_gpu_no_gpus_warns_and_does_not_set_flag(self):
        """reserve_gpu returns a warning and leaves _gpu_virtual False when pool has no GPUs."""
        from agency.tools.resource import make_gpu_reserve

        pool_empty = agResourcePool(gpus=[])
        tool = make_gpu_reserve(self.sb, pool_empty)
        result = tool.fn(agdata())
        assert result.warning is not None
        assert self.sb._gpu_virtual is False

    # ── physical GPU acquisition on bash exec ──────────────────────────────

    def test_exec_acquires_physical_gpu_when_virtual_flag_set(self):
        """exec() claims a physical GPU from the pool when _gpu_virtual is True."""
        self.tools["reserve_gpu"].fn(agdata())
        assert self.pool._gpus_acquired == 0
        self.sb.exec("echo hello")
        # Held until stop() (container-exit clear); there is no mid-skill release tool.
        assert self.pool._gpus_acquired == 1
        assert self.sb._gpu_id is not None

    def test_exec_sets_cuda_visible_devices(self):
        """CUDA_VISIBLE_DEVICES is set to a digit (the physical GPU ID) during exec()."""
        self.tools["reserve_gpu"].fn(agdata())
        out, rc = self.sb.exec("echo $CUDA_VISIBLE_DEVICES")
        assert rc == 0
        assert out.strip().isdigit()

    def test_exec_without_reserve_hides_all_gpus(self):
        """Without reserve_gpu, CUDA_VISIBLE_DEVICES is 'NoDevFiles'."""
        out, rc = self.sb.exec("echo $CUDA_VISIBLE_DEVICES")
        assert rc == 0
        assert "NoDevFiles" in out.strip()

    # ── physical GPU release after foreground exec ─────────────────────────

    def test_foreground_exec_holds_physical_gpu_until_stop(self):
        """Physical GPU stays held after exec(); release waits on container exit (stop)."""
        self.tools["reserve_gpu"].fn(agdata())
        self.sb.exec("echo hello")
        assert self.sb._gpu_id is not None
        assert self.pool._gpus_acquired == 1
        assert self.sb._gpu_virtual is True

    def test_consecutive_foreground_execs_reuse_same_physical_gpu(self):
        """Each foreground exec() reuses the already-held physical GPU."""
        self.tools["reserve_gpu"].fn(agdata())
        self.sb.exec("echo first")
        first = self.sb._gpu_id
        assert first is not None
        for _ in range(3):
            self.sb.exec("echo iteration")
            assert self.sb._gpu_id == first
            assert self.pool._gpus_acquired == 1

    def test_virtual_reservation_and_physical_gpu_persist_across_execs(self):
        """_gpu_virtual and the leased GPU stay set across successive exec() calls."""
        self.tools["reserve_gpu"].fn(agdata())
        self.sb.exec("echo first")
        assert self.sb._gpu_virtual is True
        assert self.sb._gpu_id is not None
        out, rc = self.sb.exec("echo $CUDA_VISIBLE_DEVICES")
        assert rc == 0
        assert out.strip().isdigit()

    # ── physical GPU held while background process runs ────────────────────

    def test_physical_gpu_held_while_background_process_running(self):
        """Physical GPU stays held while a background process is alive."""
        self.tools["reserve_gpu"].fn(agdata())
        # A generous ceiling, not an actual wait -- killed explicitly below.
        # Needs enough margin over real `docker exec` round-trip latency
        # (get_live_pids() does its own exec to walk /proc, GPU-passthrough
        # containers add overhead classifying nvidia_entrypoi helpers) that a
        # loaded host doesn't let it expire mid-test.
        self.sb.exec("sleep 30 &")
        live = self.sb.get_live_pids()
        assert len(live) > 0
        assert self.sb._gpu_id is not None
        assert self.pool._gpus_acquired == 1
        self.sb.exec("kill %1 2>/dev/null || true")

    def test_same_physical_gpu_used_for_subsequent_exec_during_background_process(self):
        """While a background process holds the GPU, subsequent exec() calls use the same GPU."""
        self.tools["reserve_gpu"].fn(agdata())
        self.sb.exec("sleep 30 &")  # ceiling, not a real wait -- see comment above
        self.sb.get_live_pids()
        first_gpu_id = self.sb._gpu_id
        assert first_gpu_id is not None
        self.sb.exec("echo checking")
        assert self.sb._gpu_id == first_gpu_id  # same physical GPU, not re-acquired
        self.sb.exec("kill %1 2>/dev/null || true")

    def test_physical_gpu_held_after_background_process_finishes(self):
        """Physical GPU stays held after background work exits; stop() frees it."""
        self.tools["reserve_gpu"].fn(agdata())
        self.sb.exec("sleep 0.1 &")
        time.sleep(1.0)
        self.sb.get_live_pids()
        assert self.sb._gpu_id is not None
        assert self.pool._gpus_acquired == 1
        assert self.sb._gpu_virtual is True

    # ── waiting for physical GPU when pool is exhausted ────────────────────

    def test_exec_blocks_until_pool_gpu_is_freed(self):
        """exec() waits indefinitely for a physical GPU and unblocks once one is released."""
        from agency.tools.resource import make_gpu_reserve

        pool1 = agResourcePool(gpus=[0])
        pool1.acquire_gpu()  # exhaust the only GPU

        sb2 = _make_sandbox()
        tool = make_gpu_reserve(sb2, pool1)
        tool.fn(agdata())  # virtual reservation

        exec_started = threading.Event()
        exec_done = threading.Event()

        def _run():
            exec_started.set()
            sb2.exec("echo hello")  # blocks inside exec() until GPU freed
            exec_done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        exec_started.wait()
        time.sleep(0.2)
        assert not exec_done.is_set()  # still waiting
        pool1.release_gpu(0)  # free the GPU
        exec_done.wait(timeout=60)  # container startup (docker run) can take >5 s
        assert exec_done.is_set()
        sb2.destroy()

    # ── release_resources ─────────────────────────────────────────────────

    def test_release_resources_clears_both_virtual_flag_and_physical_gpu(self):
        """release_resources() clears _gpu_virtual and returns any held physical GPU."""
        self.tools["reserve_gpu"].fn(agdata())
        self.sb.exec("sleep 30 &")
        self.sb.get_live_pids()
        assert self.sb._gpu_id is not None
        self.sb.release_resources(self.pool)
        assert self.sb._gpu_virtual is False
        assert self.sb._gpu_id is None
        assert self.pool._gpus_acquired == 0
        self.sb.exec("kill %1 2>/dev/null || true")

    def test_release_resources_without_reserve_does_not_raise(self):
        """release_resources() is safe when no GPU was ever reserved."""
        self.sb.release_resources(self.pool)
        assert self.sb._gpu_virtual is False
        assert self.sb._gpu_id is None

    # ── reserve_cpu / cpu_release ─────────────────────────────────────────

    def test_reserve_cpu_applies_limits(self):
        result = self.tools["reserve_cpu"].fn(agdata(cpus=2.0, memory="256m"))
        assert not isinstance(result, agerror)

    def test_reserve_cpu_requires_at_least_one_param(self):
        result = self.tools["reserve_cpu"].fn(agdata())
        assert isinstance(result, agerror)

    def test_cpu_release_resets_to_idle(self):
        self.tools["reserve_cpu"].fn(agdata(cpus=4.0, memory="2g"))
        result = self.tools["cpu_release"].fn(agdata())
        assert not isinstance(result, agerror)
        assert "1.0" in result.message
        assert "1024m" in result.message
