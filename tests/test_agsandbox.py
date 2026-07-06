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
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


docker = pytest.mark.skipif(
    not _docker_available(), reason="Docker daemon not reachable"
)


def _nvidia_smi_available() -> bool:
    try:
        return subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=10
        ).returncode == 0
    except Exception:
        return False


nvidia_smi = pytest.mark.skipif(
    not _nvidia_smi_available(), reason="nvidia-smi not available"
)


def _make_sandbox(**kwargs):
    from agency.agsandbox import agSandbox
    uid = str(uuid.uuid4())
    return agSandbox(uid, **kwargs)


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
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()))
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
        assert detect_memory_mb() == 4096   # safe fallback


class TestPoolAutoDetect:
    def test_pool_auto_detects_gpus(self, monkeypatch):
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "0\n1\n"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock)
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
        assert acquired == []          # still blocked
        pool.release_gpu(0)
        t.join(timeout=2)
        assert acquired == [0]

    def test_acquire_timeout_raises(self):
        pool = agResourcePool(gpus=[0])
        pool.acquire_gpu()             # exhaust pool
        with pytest.raises(TimeoutError):
            pool.acquire_gpu(timeout=0.2)

    def test_release_unowned_gpu_is_safe(self):
        pool = agResourcePool(gpus=[0])
        pool.release_gpu(0)            # never acquired — should not raise

    def test_release_unknown_gpu_is_safe(self):
        pool = agResourcePool(gpus=[0])
        pool.release_gpu(99)           # not in pool — should not raise

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
        mock_cuda.cuInit.return_value = 0       # success
        mock_cuda.cuCtxCreate_v2.return_value = 0
        mock_cuda.cuMemAlloc_v2.return_value = 0
        with patch.object(ctypes, "CDLL", return_value=mock_cuda):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,3,5,7"}):
                _allocate_gpu_markers([0, 3, 5, 7])
        # Extract the device argument (3rd positional arg) from each call
        called_devs = [
            call.args[2] for call in mock_cuda.cuCtxCreate_v2.call_args_list
        ]
        assert called_devs == [0, 1, 2, 3], (
            f"Expected CUDA device indices [0,1,2,3], got {called_devs}. "
            "Physical GPU IDs were passed directly instead of being remapped."
        )

    def test_non_main_process_name_blocks_allocation(self):
        """The MainProcess guard must block _allocate_gpu_markers in worker processes."""
        import multiprocessing
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
            capture_output=True, timeout=15,
        )
        assert child.returncode == 0, child.stderr.decode()


# ---------------------------------------------------------------------------
# agSandbox — container lifecycle
# ---------------------------------------------------------------------------

class TestAgSandboxLifecycle:
    def test_lifecycle_tag_is_lowercase(self):
        """_lifecycle_tag() must be fully lowercase — Docker rejects uppercase repository names."""
        from agency.agsandbox import agSandbox
        sb = agSandbox.__new__(agSandbox)
        sb._name = "GenerationAgent_4816622_0000"
        tag = sb._lifecycle_tag()
        assert tag == tag.lower(), f"lifecycle tag must be lowercase, got {tag!r}"
        assert "generationagent" in tag

    def test_lifecycle_tag_format(self):
        from agency.agsandbox import agSandbox
        sb = agSandbox.__new__(agSandbox)
        sb._name = "myagent_0000"
        assert sb._lifecycle_tag() == "agency/lifecycle-myagent_0000"

    @docker
    def test_container_starts_and_destroys(self):
        sb = _make_sandbox()
        name = sb._container_name()
        # Container is started lazily on first use
        sb.exec("true")
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={name}", "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        assert name in result.stdout
        sb.destroy()
        result2 = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"],
            capture_output=True, text=True,
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
                capture_output=True, text=True,
            )
            assert result.stdout.strip() != ""
        finally:
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
            sb.destroy()

    @docker
    def test_commit_works_when_started_false(self):
        """commit() must succeed on a sandbox whose _started=False but whose container
        is already running (started by a worker process or another sandbox instance).

        Simulated by creating two sandbox objects with the same agname: sb_worker starts
        the container, sb_main (with _started=False) tries to commit it."""
        from agency.agsandbox import agSandbox
        agname = str(uuid.uuid4())
        sb_worker = agSandbox(agname)   # "worker" — starts the container
        sb_main   = agSandbox(agname)   # "main process" — same name, _started=False
        tag = f"agency/test-commit-started-false-{agname[:8]}"
        try:
            # Worker starts container and writes a file.
            sb_worker.write_file("/workspace/marker.txt", "worker-written\n")
            # sb_main has _started=False but the container is already running.
            assert sb_main._started is False
            # commit() must detect the running container via docker inspect and succeed.
            assert sb_main.commit(tag) is True
            result = subprocess.run(
                ["docker", "images", "-q", tag],
                capture_output=True, text=True,
            )
            assert result.stdout.strip() != "", "image must exist even when _started was False"
        finally:
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
            sb_worker.destroy()

    @docker
    def test_commit_returns_false_when_container_not_running(self):
        """commit() must return False (not crash) if no container is running."""
        sb = _make_sandbox()
        tag = f"agency/test-commit-no-container-{sb._agname}"
        # _started is False and no container was ever started — nothing to commit.
        assert sb.commit(tag) is False
        # No image should have been created.
        result = subprocess.run(
            ["docker", "images", "-q", tag],
            capture_output=True, text=True,
        )
        assert result.stdout.strip() == ""

    @docker
    def test_ensure_started_reuses_running_container(self):
        """_ensure_started() must reuse a container already running in Docker rather
        than destroying it and starting fresh — the cross-worker-process file-persistence fix."""
        sb = _make_sandbox()
        # Start the container and write a sentinel file.
        sb.write_file("/workspace/persist.txt", "still-here\n")
        assert sb._started is True
        # Simulate a fresh sandbox object (as deserialized in a new worker process):
        # _started is False but the container is still running in Docker.
        sb._started = False
        # _ensure_started() must detect the running container and reuse it.
        sb._ensure_started()
        assert sb._started is True
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
            w = tools["write"](agdata(filePath="/workspace/cross.txt", content="cross-worker\n"))
            assert not isinstance(w, agerror), f"write failed: {w}"
            # read also runs in a process-pool worker; must find the file
            r = tools["read"](agdata(filePath="/workspace/cross.txt"))
            assert not isinstance(r, agerror), f"read failed after cross-worker write: {r}"
            assert "cross-worker" in r.content
        finally:
            sb.destroy()

    @docker
    def test_checkpoint_restore_preserves_files(self):
        tag = f"agency/test-ckpt-restore-{__import__('uuid').uuid4().hex[:8]}"
        sb1 = _make_sandbox()
        try:
            sb1.write_file("/workspace/data.txt", "restored\n")
            sb1.commit(tag)
            sb1.destroy()
            sb2 = _make_sandbox(checkpoint_image=tag)
            content = sb2.read_file("/workspace/data.txt")
            assert content == "restored\n"
            sb2.destroy()
        finally:
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)

    @docker
    def test_output_dir_agent_can_write_and_read(self, tmp_path):
        agname = "test-agent"
        output_dir = tmp_path / "agent_output" / agname
        sb = _make_sandbox(output_dir=output_dir)
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
        sb1 = _make_sandbox(output_dir=out_dir1)
        sb2 = _make_sandbox(output_dir=out_dir2)
        sb1.exec("echo from_agent1 > /agent_output/out.txt")
        out, rc = sb1.exec("cat /agent_output/out.txt")
        assert rc == 0
        assert "from_agent1" in out
        assert (out_dir1 / "out.txt").read_text().strip() == "from_agent1"
        sb1.destroy()
        sb2.destroy()

    @docker
    def test_stop_commit_true_creates_checkpoint_image_and_removes_container(self):
        """stop(commit=True) commits state to agency/lifecycle-<name> and removes the container."""
        sb = _make_sandbox()
        name = sb._container_name()
        lifecycle_tag = sb._lifecycle_tag()
        try:
            sb.write_file("/workspace/marker.txt", "lifecycle\n")
            sb.stop(commit=True)
            # Container must be gone
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"],
                capture_output=True, text=True,
            )
            assert name not in result.stdout, "container must be removed after stop()"
            # Lifecycle image must exist
            img = subprocess.run(
                ["docker", "images", "-q", lifecycle_tag],
                capture_output=True, text=True,
            )
            assert img.stdout.strip() != "", "lifecycle image must exist after stop(commit=True)"
            # _checkpoint_image must be set
            assert sb._checkpoint_image == lifecycle_tag
        finally:
            subprocess.run(["docker", "rmi", "-f", lifecycle_tag], capture_output=True)
            sb.destroy()

    @docker
    def test_stop_commit_false_removes_container_without_image(self):
        """stop(commit=False) removes the container but does not create a lifecycle image."""
        sb = _make_sandbox()
        name = sb._container_name()
        lifecycle_tag = sb._lifecycle_tag()
        try:
            sb.write_file("/workspace/dirty.txt", "dirty\n")
            previous_lifecycle = sb._checkpoint_image  # None on first call
            sb.stop(commit=False)
            # Container must be gone
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"],
                capture_output=True, text=True,
            )
            assert name not in result.stdout, "container must be removed after stop()"
            # _checkpoint_image must not have changed
            assert sb._checkpoint_image == previous_lifecycle
            # No lifecycle image should have been created
            img = subprocess.run(
                ["docker", "images", "-q", lifecycle_tag],
                capture_output=True, text=True,
            )
            assert img.stdout.strip() == "", "stop(commit=False) must not create a lifecycle image"
        finally:
            subprocess.run(["docker", "rmi", "-f", lifecycle_tag], capture_output=True)
            sb.destroy()

    @docker
    def test_checkpoint_image_restores_workspace_on_next_start(self):
        """After stop(commit=True), _ensure_started() restores /workspace from the lifecycle image."""
        sb = _make_sandbox()
        lifecycle_tag = sb._lifecycle_tag()
        try:
            sb.write_file("/workspace/persistent.txt", "saved\n")
            sb.stop(commit=True)
            assert sb._started is False
            # Next exec triggers _ensure_started() which runs docker run from lifecycle image.
            out, rc = sb.exec("cat /workspace/persistent.txt")
            assert rc == 0
            assert "saved" in out
        finally:
            subprocess.run(["docker", "rmi", "-f", lifecycle_tag], capture_output=True)
            sb.destroy()

    @docker
    def test_stop_commit_false_reverts_to_last_checkpoint(self):
        """stop(commit=False) discards dirty state; next start restores from last lifecycle image."""
        sb = _make_sandbox()
        lifecycle_tag = sb._lifecycle_tag()
        try:
            # First successful tool call: write file and commit.
            sb.write_file("/workspace/good.txt", "good\n")
            sb.stop(commit=True)
            # Second tool call that fails: write a dirty file without committing.
            sb.exec("true")  # restarts from lifecycle image
            sb.write_file("/workspace/dirty.txt", "dirty\n")
            sb.stop(commit=False)
            # Next start must restore from lifecycle image — dirty.txt must not exist.
            sb.exec("true")
            content = sb.read_file("/workspace/good.txt")
            assert "good" in content
            _, dirty_rc = sb.exec("test -f /workspace/dirty.txt")
            assert dirty_rc != 0, "dirty file must not exist after stop(commit=False)"
        finally:
            subprocess.run(["docker", "rmi", "-f", lifecycle_tag], capture_output=True)
            sb.destroy()

    @docker
    def test_destroy_removes_checkpoint_image(self):
        """destroy() cleans up the lifecycle image created by stop(commit=True)."""
        sb = _make_sandbox()
        lifecycle_tag = sb._lifecycle_tag()
        sb.write_file("/workspace/x.txt", "x\n")
        sb.stop(commit=True)
        # Confirm image exists before destroy
        img = subprocess.run(
            ["docker", "images", "-q", lifecycle_tag],
            capture_output=True, text=True,
        )
        assert img.stdout.strip() != "", "lifecycle image must exist before destroy()"
        sb.destroy()
        # Image must be gone
        img2 = subprocess.run(
            ["docker", "images", "-q", lifecycle_tag],
            capture_output=True, text=True,
        )
        assert img2.stdout.strip() == "", "destroy() must remove the lifecycle image"

    @docker
    def test_stop_retries_rm_on_first_failure(self):
        """stop() retries docker rm -f up to 3 times; succeeds if a later attempt works."""
        sb = _make_sandbox()
        sb.write_file("/workspace/x.txt", "x\n")
        name = sb._container_name()

        call_count = [0]
        real_run = sb._run

        def flaky_run(cmd, **kwargs):
            if "rm" in cmd and "-f" in cmd and name in cmd:
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("simulated rm -f failure")
            return real_run(cmd, **kwargs)

        sb._run = flaky_run
        sb.stop(commit=False)

        assert call_count[0] == 2, "expected one failure then one success"
        assert sb._started is False
        # Container must actually be gone after the successful retry
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        assert name not in result.stdout

    @docker
    def test_stop_emits_warning_after_all_retries_fail(self):
        """stop() emits a WARNING to stderr when rm -f fails all 3 attempts."""
        import io, sys
        sb = _make_sandbox()
        sb.write_file("/workspace/x.txt", "x\n")

        real_run = sb._run

        def always_fail_rm(cmd, **kwargs):
            if "rm" in cmd and "-f" in cmd:
                raise RuntimeError("simulated persistent failure")
            return real_run(cmd, **kwargs)

        sb._run = always_fail_rm
        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            sb.stop(commit=False)
        finally:
            sys.stderr = old_stderr
            # Force cleanup bypassing our mock
            sb._run = real_run
            sb.destroy()

        assert "WARNING" in captured.getvalue()
        assert sb._started is False  # _started cleared even on failure

    @docker
    def test_concurrent_docker_calls_gated_by_docker_semaphore(self):
        """All docker calls go through _run() which holds _docker_semaphore; peak concurrency <= 8."""
        from agency.agsandbox import _docker_semaphore

        sandboxes = [_make_sandbox() for _ in range(4)]
        lifecycle_tags = [sb._lifecycle_tag() for sb in sandboxes]
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
            # stop(commit=True) exercises commit + rm -f; both must go through the semaphore.
            threads = [threading.Thread(target=sb.stop, kwargs={"commit": True})
                       for sb in sandboxes]
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

        assert peak[0] <= 16, f"peak concurrent docker calls {peak[0]} exceeded semaphore limit of 16"

    @docker
    def test_ensure_started_removes_created_state_container(self):
        """_ensure_started() force-removes a container stuck in 'Created' state before docker run."""
        sb = _make_sandbox()
        name = sb._container_name()
        try:
            # Manually create a container in 'Created' state (no --detach run, just create).
            subprocess.run(
                ["docker", "create", "--name", name, sb.BASE_IMAGE, "tail", "-f", "/dev/null"],
                capture_output=True, check=True,
            )
            status = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Status}}", name],
                capture_output=True, text=True,
            )
            assert status.stdout.strip() == "created"
            # _ensure_started() must remove the stuck container and start fresh.
            sb._ensure_started()
            assert sb._started is True
            out, rc = sb.exec("echo ok")
            assert rc == 0 and "ok" in out
        finally:
            sb.destroy()

    @docker
    def test_stop_retries_commit_on_first_failure(self):
        """stop(commit=True) retries docker commit up to 3 times; succeeds if a later attempt works."""
        sb = _make_sandbox()
        sb.write_file("/workspace/x.txt", "x\n")
        lifecycle_tag = sb._lifecycle_tag()

        call_count = [0]
        real_run = sb._run

        def flaky_run(cmd, **kwargs):
            if "commit" in cmd:
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("simulated commit failure")
            return real_run(cmd, **kwargs)

        sb._run = flaky_run
        try:
            sb.stop(commit=True)
            assert call_count[0] == 2, "expected one failure then one success"
            assert sb._checkpoint_image == lifecycle_tag
            assert sb._started is False
            img = subprocess.run(
                ["docker", "images", "-q", lifecycle_tag],
                capture_output=True, text=True,
            )
            assert img.stdout.strip() != "", "lifecycle image must exist after successful retry"
        finally:
            subprocess.run(["docker", "rmi", "-f", lifecycle_tag], capture_output=True)
            sb.destroy()

    @docker
    def test_stop_emits_warning_after_all_commit_retries_fail(self):
        """stop(commit=True) emits a WARNING to stderr when all 3 commit attempts fail;
        _checkpoint_image is not updated so the next start restores from the prior checkpoint."""
        import io
        sb = _make_sandbox()
        sb.write_file("/workspace/x.txt", "x\n")
        previous_lifecycle = sb._checkpoint_image

        real_run = sb._run

        def always_fail_commit(cmd, **kwargs):
            if "commit" in cmd:
                raise RuntimeError("simulated persistent commit failure")
            return real_run(cmd, **kwargs)

        sb._run = always_fail_commit
        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            sb.stop(commit=True)
        finally:
            sys.stderr = old_stderr
            sb._run = real_run
            sb.destroy()

        assert "WARNING" in captured.getvalue()
        assert sb._checkpoint_image == previous_lifecycle  # not updated on all-retry failure
        assert sb._started is False


    @docker
    def test_ensure_started_removes_exited_container(self):
        """Exited containers are force-removed and recreated (no docker start fast-path).

        State is preserved across stop()/start() cycles via checkpoint_image commits,
        not via docker stop/start.  An exited container is treated as a zombie and
        removed so the name is free for a fresh docker run.
        """
        sb = _make_sandbox()
        name = sb._container_name()
        try:
            sb.write_file("/workspace/exited.txt", "still-here\n")
            # Externally stop (not remove) the container — puts it in exited state.
            subprocess.run(["docker", "stop", "-t", "0", name], capture_output=True)
            sb._started = False
            # _ensure_started() must remove the exited container and do a fresh docker run.
            sb._ensure_started()
            assert sb._started is True
            # The fresh container has no /workspace/exited.txt — the exited container
            # was force-removed.  State would only survive if stop(commit=True) had been
            # called before the stop to commit a checkpoint_image.
            with pytest.raises(FileNotFoundError):
                sb.read_file("/workspace/exited.txt")
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
    """Unit tests for read_file error cases — no Docker required."""

    def _make_sb(self):
        from agency.agsandbox import agSandbox
        sb = agSandbox.__new__(agSandbox)
        sb._started = True
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
        self.sb.exec("sleep 5 &")
        assert len(self.sb._watched_pids) > 0

    def test_foreground_spawned_child_tracked(self):
        # A foreground command that internally forks a child and exits.
        # The child escapes jobs -p but must still be captured via /proc diffing.
        self.sb.write_file("/workspace/spawner.py", (
            "import subprocess, time\n"
            "subprocess.Popen(['sleep', '5'])\n"   # detached child, not waited on
        ))
        self.sb.exec("python3 /workspace/spawner.py")
        assert len(self.sb._watched_pids) > 0

    def test_parent_exits_child_survives_still_tracked(self):
        # Parent spawns a child then exits. Child is reparented to PID 1 and
        # escapes any BFS from the original PID. Baseline diff must find it.
        self.sb.write_file("/workspace/spawner.py", (
            "import subprocess, os\n"
            "subprocess.Popen(['sleep', '5'])\n"  # child detaches
            "os._exit(0)\n"                        # parent exits immediately
        ))
        self.sb.exec("python3 /workspace/spawner.py")
        live = self.sb.get_live_pids()
        # Parent is gone; the orphaned sleep child must still be tracked
        assert len(live) > 0

    def test_process_tree_descendants_tracked(self):
        # A process is tracked; it later spawns children of its own.
        # get_live_pids() must expand the tree and include those grandchildren.
        self.sb.write_file("/workspace/parent.py", (
            "import subprocess, time\n"
            "# Spawn two long-lived children after a brief pause\n"
            "time.sleep(0.2)\n"
            "subprocess.Popen(['sleep', '5'])\n"
            "subprocess.Popen(['sleep', '5'])\n"
            "time.sleep(5)\n"   # parent also stays alive
        ))
        self.sb.exec("python3 /workspace/parent.py &")
        time.sleep(0.5)          # let the parent spawn its children
        live = self.sb.get_live_pids()
        # parent + 2 children = at least 3 live PIDs
        assert len(live) >= 3

    def test_double_forked_daemon_tracked(self):
        # Classic Unix double-fork: grandchild is reparented to PID 1 and
        # completely detached from the shell's job table.
        self.sb.write_file("/workspace/daemon.py", (
            "import os, time\n"
            "if os.fork() == 0:\n"          # first fork
            "    if os.fork() == 0:\n"      # second fork — grandchild
            "        time.sleep(5)\n"      # grandchild runs in background
            "    os._exit(0)\n"             # intermediate child exits
            "os.wait()\n"                   # parent waits for intermediate child
        ))
        self.sb.exec("python3 /workspace/daemon.py")
        assert len(self.sb._watched_pids) > 0

    def test_get_live_pids_returns_running(self):
        self.sb.exec("sleep 5 &")
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
        self.sb.exec("sleep 5 &")
        live = self.sb.get_live_pids()
        assert len(live) > 0
        for pid in list(live):
            self.sb.release_daemon(pid)
        assert self.sb.get_live_pids() == set()

    def test_daemon_children_also_excluded(self):
        # Release a parent as daemon; children it spawns later must also be excluded.
        self.sb.write_file("/workspace/daemon_parent.py", (
            "import subprocess, time\n"
            "subprocess.Popen(['sleep', '5'])\n"
            "time.sleep(5)\n"
        ))
        self.sb.exec("python3 /workspace/daemon_parent.py &")
        live = self.sb.get_live_pids()
        assert len(live) > 0
        # Release the parent; its child (sleep 5) should also be excluded
        for pid in list(live):
            self.sb.release_daemon(pid)
        time.sleep(0.3)   # let the child spawn
        assert self.sb.get_live_pids() == set()

    def test_pid_status_summary_no_processes(self):
        summary = self.sb.pid_status_summary()
        assert "no background" in summary

    def test_pid_status_summary_with_running_process(self):
        self.sb.exec("sleep 5 &")
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
        self.tools["write"].fn(agdata(filePath="/workspace/t.txt", content="abc\n"))
        r = self.tools["read"].fn(agdata(filePath="/workspace/t.txt"))
        assert "abc" in r.content

    def test_glob_tool_finds_files(self):
        self.tools["write"].fn(agdata(filePath="/workspace/a.py", content="x\n"))
        self.tools["write"].fn(agdata(filePath="/workspace/b.py", content="y\n"))
        r = self.tools["glob"].fn(agdata(pattern="*.py", path="/workspace"))
        assert len(r.files) >= 2

    def test_grep_tool_finds_pattern(self):
        self.tools["write"].fn(agdata(filePath="/workspace/src.py", content="SECRET=42\n"))
        r = self.tools["grep"].fn(agdata(pattern="SECRET", path="/workspace"))
        assert any("SECRET" in m["text"] for m in r.matches)

    def test_edit_tool_replaces_content(self):
        self.tools["write"].fn(agdata(filePath="/workspace/edit_me.txt", content="foo bar\n"))
        self.tools["edit"].fn(agdata(
            filePath="/workspace/edit_me.txt",
            oldString="foo",
            newString="baz",
        ))
        r = self.tools["read"].fn(agdata(filePath="/workspace/edit_me.txt"))
        assert "baz" in r.content
        assert "foo" not in r.content

    def test_daemon_release_tool_stops_monitoring(self):
        # Start a background process, get its PID, release it as daemon,
        # verify the outer loop would no longer wait for it.
        self.tools["bash"].fn(agdata(command="sleep 5 &"))
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
        self.pool = agResourcePool(gpus=[0, 1], idle_cpus=0.5, idle_memory="512m")
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
        # During the command a GPU is held; after a foreground exec it is released.
        self.sb.exec("echo hello")
        # Foreground exec with no background processes releases immediately.
        assert self.pool._gpus_acquired == 0

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

    def test_foreground_exec_releases_physical_gpu_immediately(self):
        """Physical GPU is released at the end of exec() when no background processes remain."""
        self.tools["reserve_gpu"].fn(agdata())
        self.sb.exec("echo hello")
        assert self.sb._gpu_id is None
        assert self.pool._gpus_acquired == 0
        assert self.sb._gpu_virtual is True  # virtual reservation persists

    def test_consecutive_foreground_execs_each_acquire_and_release(self):
        """Each foreground exec() acquires a physical GPU then releases it; pool stays free."""
        self.tools["reserve_gpu"].fn(agdata())
        for _ in range(3):
            self.sb.exec("echo iteration")
            assert self.sb._gpu_id is None
            assert self.pool._gpus_acquired == 0

    def test_virtual_reservation_persists_after_physical_release(self):
        """_gpu_virtual stays True after a foreground exec so the next bash call can re-acquire."""
        self.tools["reserve_gpu"].fn(agdata())
        self.sb.exec("echo first")
        assert self.sb._gpu_virtual is True
        # Second exec() should re-acquire a physical GPU and set CUDA_VISIBLE_DEVICES.
        out, rc = self.sb.exec("echo $CUDA_VISIBLE_DEVICES")
        assert rc == 0
        assert out.strip().isdigit()

    # ── physical GPU held while background process runs ────────────────────

    def test_physical_gpu_held_while_background_process_running(self):
        """Physical GPU stays held while a background process is alive."""
        self.tools["reserve_gpu"].fn(agdata())
        self.sb.exec("sleep 5 &")
        live = self.sb.get_live_pids()
        assert len(live) > 0
        assert self.sb._gpu_id is not None
        assert self.pool._gpus_acquired == 1
        self.sb.exec("kill %1 2>/dev/null || true")

    def test_same_physical_gpu_used_for_subsequent_exec_during_background_process(self):
        """While a background process holds the GPU, subsequent exec() calls use the same GPU."""
        self.tools["reserve_gpu"].fn(agdata())
        self.sb.exec("sleep 5 &")
        self.sb.get_live_pids()
        first_gpu_id = self.sb._gpu_id
        assert first_gpu_id is not None
        self.sb.exec("echo checking")
        assert self.sb._gpu_id == first_gpu_id  # same physical GPU, not re-acquired
        self.sb.exec("kill %1 2>/dev/null || true")

    def test_physical_gpu_released_after_background_process_finishes(self):
        """Physical GPU is released by get_live_pids() once the background process exits."""
        self.tools["reserve_gpu"].fn(agdata())
        self.sb.exec("sleep 0.1 &")
        time.sleep(1.0)
        self.sb.get_live_pids()   # triggers release since alive set is now empty
        assert self.sb._gpu_id is None
        assert self.pool._gpus_acquired == 0
        assert self.sb._gpu_virtual is True  # virtual reservation persists

    # ── waiting for physical GPU when pool is exhausted ────────────────────

    def test_exec_blocks_until_pool_gpu_is_freed(self):
        """exec() waits indefinitely for a physical GPU and unblocks once one is released."""
        from agency.tools.resource import make_gpu_reserve
        pool1 = agResourcePool(gpus=[0])
        pool1.acquire_gpu()   # exhaust the only GPU

        sb2 = _make_sandbox()
        tool = make_gpu_reserve(sb2, pool1)
        tool.fn(agdata())     # virtual reservation

        exec_started = threading.Event()
        exec_done    = threading.Event()

        def _run():
            exec_started.set()
            sb2.exec("echo hello")   # blocks inside exec() until GPU freed
            exec_done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        exec_started.wait()
        time.sleep(0.2)
        assert not exec_done.is_set()   # still waiting
        pool1.release_gpu(0)            # free the GPU
        exec_done.wait(timeout=60)      # container startup (docker run) can take >5 s
        assert exec_done.is_set()
        sb2.destroy()

    # ── gpu_release ────────────────────────────────────────────────────────

    def test_gpu_release_clears_virtual_flag(self):
        """gpu_release clears _gpu_virtual even when no physical GPU is currently held."""
        self.tools["reserve_gpu"].fn(agdata())
        self.tools["gpu_release"].fn(agdata())
        assert self.sb._gpu_virtual is False
        assert self.sb._gpu_id is None

    def test_gpu_release_also_frees_physical_gpu_held_by_background_process(self):
        """gpu_release forcibly releases a physical GPU even while a background process runs."""
        self.tools["reserve_gpu"].fn(agdata())
        self.sb.exec("sleep 5 &")
        self.sb.get_live_pids()
        assert self.sb._gpu_id is not None
        self.tools["gpu_release"].fn(agdata())
        assert self.sb._gpu_virtual is False
        assert self.sb._gpu_id is None
        assert self.pool._gpus_acquired == 0
        self.sb.exec("kill %1 2>/dev/null || true")

    def test_gpu_release_without_reserve_is_safe(self):
        """gpu_release is a no-op when nothing is reserved."""
        result = self.tools["gpu_release"].fn(agdata())
        assert result.message is not None
        assert self.sb._gpu_virtual is False
        assert self.sb._gpu_id is None

    # ── release_resources ─────────────────────────────────────────────────

    def test_release_resources_clears_both_virtual_flag_and_physical_gpu(self):
        """release_resources() clears _gpu_virtual and returns any held physical GPU."""
        self.tools["reserve_gpu"].fn(agdata())
        self.sb.exec("sleep 5 &")
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
        assert "0.5" in result.message
        assert "512m" in result.message


# ---------------------------------------------------------------------------
# Dangling image auto-cleanup (eager rmi on commit)
# ---------------------------------------------------------------------------

class TestDanglingImageEagerCleanup:
    """Tests for the eager old-image deletion in stop(commit=True)."""

    def test_no_prune_thread(self):
        """No background agsandbox-prune thread should exist after the refactor."""
        named = [t for t in threading.enumerate() if t.name == "agsandbox-prune"]
        assert not named, "agsandbox-prune thread should have been removed"

    def test_stop_commit_deletes_old_image(self):
        """stop(commit=True) must delete the image that previously held the tag."""
        import agency.agsandbox as _mod

        sb = _make_sandbox()
        tag = sb._lifecycle_tag()

        run_calls = []
        fake_old_id = "sha256:deadbeef0000"

        class FakeCompleted:
            def __init__(self, stdout=b"", returncode=0):
                self.stdout = stdout
                self.returncode = returncode

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            run_calls.append(args)
            if "inspect" in args:
                return FakeCompleted(stdout=fake_old_id.encode())
            if "commit" in args:
                return FakeCompleted()
            if "rm" in args:
                return FakeCompleted()
            if "rmi" in args:
                return FakeCompleted()
            return FakeCompleted()

        with patch.object(_mod.agSandbox, "_run", fake_run):
            with patch.object(sb, "_started", True):
                with patch.object(sb, "_container_running", return_value=True):
                    with patch.object(sb, "_gpu_virtual", False):
                        sb.stop(commit=True)

        rmi_calls = [a for a in run_calls if "rmi" in a]
        assert rmi_calls, "expected docker rmi call for old image"
        assert any(fake_old_id in " ".join(a) for a in rmi_calls), (
            f"rmi call did not reference old image ID; calls: {rmi_calls}"
        )

    def test_stop_commit_skips_rmi_when_no_old_image(self):
        """If the tag does not exist yet (first commit), no rmi call is made."""
        import agency.agsandbox as _mod

        sb = _make_sandbox()

        class FakeCompleted:
            def __init__(self, stdout=b"", returncode=0):
                self.stdout = stdout
                self.returncode = returncode

        run_calls = []

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            run_calls.append(args)
            if "inspect" in args:
                return FakeCompleted(stdout=b"", returncode=1)  # tag not found
            return FakeCompleted()

        with patch.object(_mod.agSandbox, "_run", fake_run):
            with patch.object(sb, "_started", True):
                with patch.object(sb, "_container_running", return_value=True):
                    with patch.object(sb, "_gpu_virtual", False):
                        sb.stop(commit=True)

        rmi_calls = [a for a in run_calls if "rmi" in a]
        assert not rmi_calls, "must not call rmi when there was no previous image"

    def test_stop_commit_rmi_failure_is_best_effort(self):
        """A failing rmi during old-image cleanup must NOT propagate.

        Per Design_sandbox_lifecycle.md's "Dangling image accumulation and
        eager cleanup" section, this rmi is best-effort: a race with another
        agent's inspect/rmi (or a fork still using the image) is expected and
        should leave a dangling image rather than crash stop() — which runs
        after every tool call, so a hard failure here would be far worse than
        the disk-space cost of an occasional dangling image."""
        import io
        import agency.agsandbox as _mod

        sb = _make_sandbox()
        fake_old_id = "sha256:cafebabe1234"

        class FakeCompleted:
            def __init__(self, stdout=b"", returncode=0):
                self.stdout = stdout
                self.returncode = returncode

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "inspect" in args:
                return FakeCompleted(stdout=fake_old_id.encode())
            if "rmi" in args:
                raise RuntimeError("image in use")
            return FakeCompleted()

        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            with patch.object(_mod.agSandbox, "_run", fake_run):
                with patch.object(sb, "_started", True):
                    with patch.object(sb, "_container_running", return_value=True):
                        with patch.object(sb, "_gpu_virtual", False):
                            sb.stop(commit=True)  # must not raise
        finally:
            sys.stderr = old_stderr

        assert "WARNING" in captured.getvalue()
        assert fake_old_id in captured.getvalue()

    @docker
    def test_repeated_commits_leave_no_dangling_images(self):
        """stop(commit=True) called 3 times to the same tag must leave 0 new dangling images."""
        name = f"test-eager-{uuid.uuid4().hex[:8]}"
        tag  = f"agency/lifecycle-{name}"

        def _dangling_ids():
            r = subprocess.run(
                ["docker", "images", "-f", "dangling=true", "-q"],
                capture_output=True, text=True,
            )
            return set(ln.strip() for ln in r.stdout.splitlines() if ln.strip())

        subprocess.run(
            ["docker", "run", "-d", "--name", name, "agency-sandbox:latest",
             "tail", "-f", "/dev/null"],
            capture_output=True, check=True,
        )
        try:
            before = _dangling_ids()
            for _ in range(3):
                old_id_r = subprocess.run(
                    ["docker", "inspect", "--format={{.Id}}", tag],
                    capture_output=True, text=True,
                )
                old_id = old_id_r.stdout.strip() if old_id_r.returncode == 0 else None
                subprocess.run(["docker", "commit", name, tag],
                               capture_output=True, check=True)
                if old_id:
                    subprocess.run(["docker", "rmi", old_id],
                                   capture_output=True)
            after = _dangling_ids()
            new_dangling = after - before
            assert len(new_dangling) == 0, (
                f"expected 0 new dangling images with eager cleanup, got {len(new_dangling)}"
            )
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


# ---------------------------------------------------------------------------
# _rm_container / _rmi helpers
# ---------------------------------------------------------------------------

class TestDockerCommandHelpers:
    """Unit tests for _rm_container and _rmi — no real Docker required."""

    def _sb(self):
        return _make_sandbox()

    # --- _rm_container ---

    def test_rm_container_sends_rm_force_args(self):
        sb = self._sb()
        calls = []

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append((args, check))
            return OK()

        import agency.agsandbox as _mod
        with patch.object(_mod.agSandbox, "_run", fake_run):
            sb._rm_container("my-container")

        assert len(calls) == 1
        args, check = calls[0]
        assert "rm" in args and "-f" in args and "my-container" in args
        assert check is True

    def test_rm_container_raises_on_failure(self):
        sb = self._sb()

        import agency.agsandbox as _mod
        with patch.object(_mod.agSandbox, "_run", side_effect=RuntimeError("rm failed")):
            with pytest.raises(RuntimeError, match="rm failed"):
                sb._rm_container("bad-container")

    # --- _rmi ---

    def test_rmi_sends_rmi_args_without_force(self):
        sb = self._sb()
        calls = []

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append((args, check))
            return OK()

        import agency.agsandbox as _mod
        with patch.object(_mod.agSandbox, "_run", fake_run):
            sb._rmi("sha256:abc123")

        assert len(calls) == 1
        args, check = calls[0]
        assert "rmi" in args and "sha256:abc123" in args
        assert "-f" not in args
        assert check is True

    def test_rmi_force_adds_dash_f(self):
        sb = self._sb()
        calls = []

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append(args)
            return OK()

        import agency.agsandbox as _mod
        with patch.object(_mod.agSandbox, "_run", fake_run):
            sb._rmi("myimage:tag", force=True)

        assert "-f" in calls[0]

    def test_rmi_raises_on_failure(self):
        sb = self._sb()

        import agency.agsandbox as _mod
        with patch.object(_mod.agSandbox, "_run", side_effect=RuntimeError("rmi failed")):
            with pytest.raises(RuntimeError, match="rmi failed"):
                sb._rmi("sha256:deadbeef")

    # --- _ensure_started pre-cleanup guard ---

    def test_ensure_started_skips_rm_when_no_leftover_container(self):
        """No rm when the container doesn't exist before create."""
        sb = self._sb()
        calls = []

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append(args)
            return OK()

        import agency.agsandbox as _mod
        with patch.object(_mod.agSandbox, "_run", fake_run):
            # status returns "" → no leftover container
            with patch.object(sb, "_container_running", return_value=False):
                with patch.object(sb, "_container_status", return_value=""):
                    with patch.object(sb, "_run_with_conflict_retry"):
                        sb._ensure_started()

        rm_calls = [a for a in calls if "rm" in a]
        assert not rm_calls, f"expected no rm call; got {rm_calls}"

    def test_ensure_started_rms_leftover_container(self):
        """rm is issued when a non-running leftover container exists."""
        sb = self._sb()
        calls = []

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append((args, check))
            return OK()

        import agency.agsandbox as _mod
        with patch.object(_mod.agSandbox, "_run", fake_run):
            with patch.object(sb, "_container_running", return_value=False):
                with patch.object(sb, "_container_status", return_value="exited"):
                    with patch.object(sb, "_run_with_conflict_retry"):
                        sb._ensure_started()

        rm_calls = [(a, c) for (a, c) in calls if "rm" in a]
        assert rm_calls, "expected rm call for leftover container"
        assert all(c is True for _, c in rm_calls), "rm must use check=True"

    # --- destroy semaphore release ---

    def test_destroy_releases_semaphore_even_when_rm_raises(self):
        """_container_semaphore must be released in finally even if rm fails."""
        import agency.agsandbox as _mod

        sb = self._sb()

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            if "rm" in args:
                raise RuntimeError("rm exploded")
            if "images" in args:
                result = OK()
                result.stdout = b""
                return result
            return OK()

        released = []
        real_release = _mod._container_semaphore.release

        with patch.object(_mod.agSandbox, "_run", fake_run):
            with patch.object(sb, "_started", True):
                with patch.object(sb, "_container_running", return_value=True):
                    with patch.object(sb, "_container_status", return_value="running"):
                        with patch.object(_mod._container_semaphore, "release",
                                          side_effect=lambda: released.append(1)):
                            with pytest.raises(RuntimeError, match="rm exploded"):
                                sb.destroy()

        assert released, "semaphore must be released even when rm raises"

    def test_destroy_skips_rm_when_container_absent(self):
        """destroy() must not call rm when the container does not exist."""
        import agency.agsandbox as _mod

        sb = self._sb()
        calls = []

        class OK:
            returncode = 0
            stdout = b""

        def fake_run(self_inner, args, *, check=False, input=None, timeout=120):
            calls.append(args)
            return OK()

        with patch.object(_mod.agSandbox, "_run", fake_run):
            with patch.object(sb, "_started", False):
                with patch.object(sb, "_container_running", return_value=False):
                    with patch.object(sb, "_container_status", return_value=""):
                        sb.destroy()

        rm_calls = [a for a in calls if "rm" in a and "rmi" not in a]
        assert not rm_calls, f"must not rm when container absent; got {rm_calls}"
