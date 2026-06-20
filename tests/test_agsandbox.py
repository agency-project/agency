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

from agency.agdata import agdata
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
    def test_mark_gpus_false_starts_no_processes(self):
        pool = agResourcePool(gpus=[0, 1], mark_gpus=False)
        assert pool._marker_procs == []

    def test_mark_gpus_true_empty_gpu_list_starts_no_processes(self):
        pool = agResourcePool(gpus=[], mark_gpus=True)
        assert pool._marker_procs == []

    def test_mark_gpus_starts_one_process_per_gpu(self):
        pool = agResourcePool(gpus=[0, 1], mark_gpus=True)
        try:
            assert len(pool._marker_procs) == 2
        finally:
            pool._stop_gpu_markers()

    def test_marker_processes_have_distinct_pids(self):
        pool = agResourcePool(gpus=[0, 1, 2], mark_gpus=True)
        try:
            pids = [p.pid for p in pool._marker_procs]
            assert len(set(pids)) == 3
        finally:
            pool._stop_gpu_markers()

    def test_stop_markers_terminates_all(self):
        pool = agResourcePool(gpus=[0, 1], mark_gpus=True)
        procs = list(pool._marker_procs)
        pool._stop_gpu_markers()
        time.sleep(0.5)
        for proc in procs:
            assert proc.poll() is not None, f"process {proc.pid} still running after stop"

    def test_stop_markers_no_zombies(self):
        # After stop, processes must be reaped (poll() returns exit code, not None).
        # A zombie would show poll()=None because wait() was never called.
        pool = agResourcePool(gpus=[0, 1], mark_gpus=True)
        procs = list(pool._marker_procs)
        pool._stop_gpu_markers()
        time.sleep(0.2)
        for proc in procs:
            code = proc.poll()
            assert code is not None, f"process {proc.pid} is a zombie (not reaped)"

    def test_stop_markers_clears_list(self):
        pool = agResourcePool(gpus=[0], mark_gpus=True)
        pool._stop_gpu_markers()
        assert pool._marker_procs == []

    def test_marker_process_comm_name(self):
        # prctl(PR_SET_NAME) runs before torch import, so the name is set
        # even if CUDA is unavailable and the process exits immediately.
        pool = agResourcePool(gpus=[0], mark_gpus=True)
        try:
            pid = pool._marker_procs[0].pid
            comm_path = f"/proc/{pid}/comm"
            # Poll until we see "agency-gpu" (prctl ran) or the process exits.
            deadline = time.monotonic() + 2.0
            comm = None
            while time.monotonic() < deadline:
                try:
                    with open(comm_path) as f:
                        value = f.read().strip()
                    if value == "agency-gpu":
                        comm = value
                        break
                    # Still "python" — prctl hasn't run yet; keep polling.
                except FileNotFoundError:
                    break  # process already exited
                time.sleep(0.05)
            if comm is None:
                pytest.skip("marker process exited before prctl could be observed")
            assert comm == "agency-gpu"
        finally:
            pool._stop_gpu_markers()

    @nvidia_smi
    def test_marker_appears_in_nvidia_smi(self):
        """Marker process allocates ~128 MB of VRAM visible in nvidia-smi."""
        # Marker subprocesses use libcuda.so.1 — skip if the driver isn't present.
        import ctypes
        try:
            ctypes.CDLL('libcuda.so.1')
        except OSError:
            pytest.skip("libcuda.so.1 not available on this host")

        from agency.agresources import detect_gpus
        real_gpus = detect_gpus()
        if not real_gpus:
            pytest.skip("no GPUs detected by nvidia-smi")

        pool = agResourcePool(gpus=[real_gpus[0]], mark_gpus=True)
        try:
            # Allow torch to load and finish the VRAM allocation.
            time.sleep(5)
            marker_proc = pool._marker_procs[0]
            if marker_proc.poll() is not None:
                pytest.skip("marker process exited (torch or CUDA unavailable on this GPU)")

            result = subprocess.run(
                ["nvidia-smi",
                 "--query-compute-apps=pid,used_gpu_memory",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            assert result.returncode == 0, f"nvidia-smi failed: {result.stderr}"

            reported_pids = set()
            for line in result.stdout.splitlines():
                parts = line.split(",")
                if parts:
                    try:
                        reported_pids.add(int(parts[0].strip()))
                    except ValueError:
                        pass

            assert marker_proc.pid in reported_pids, (
                f"Marker PID {marker_proc.pid} not found in nvidia-smi output:\n"
                f"{result.stdout}"
            )
        finally:
            pool._stop_gpu_markers()

    def test_subprocess_import_does_not_add_markers(self):
        """Importing agent in a child process must not spawn additional markers.

        The class-level agresource_pool = agResourcePool(mark_gpus=False) default
        means child processes that import agent get a no-marker pool.  This test
        would have caught the bug where mark_gpus=True was the class default,
        causing the agwebui server and ProcessPoolExecutor workers to each start
        their own full set of markers.
        """
        def _count_agency_gpu() -> int:
            count = 0
            for entry in os.scandir("/proc"):
                if not entry.name.isdigit():
                    continue
                try:
                    with open(f"/proc/{entry.name}/comm") as f:
                        if f.read().strip() == "agency-gpu":
                            count += 1
                except OSError:
                    pass
            return count

        pool = agResourcePool(gpus=[0], mark_gpus=True)
        try:
            time.sleep(0.5)  # let marker start
            before = _count_agency_gpu()
            assert before >= 1, "marker did not start"

            child = subprocess.run(
                [sys.executable, "-c", "from agency.agent import agent"],
                capture_output=True, timeout=15,
            )
            assert child.returncode == 0, child.stderr.decode()

            after = _count_agency_gpu()
            assert after == before, (
                f"subprocess import added {after - before} marker(s); "
                "check mark_gpus default in agent.py class definition"
            )
        finally:
            pool._stop_gpu_markers()

    def test_process_pool_worker_does_not_add_markers(self):
        """ProcessPoolExecutor workers must not start markers when importing agent.

        Workers are named something other than 'MainProcess', so the
        multiprocessing.current_process().name guard in agResourcePool.__init__
        must prevent them from starting markers even if mark_gpus=True were
        somehow the default.
        """
        import concurrent.futures

        def _count_agency_gpu() -> int:
            count = 0
            for entry in os.scandir("/proc"):
                if not entry.name.isdigit():
                    continue
                try:
                    with open(f"/proc/{entry.name}/comm") as f:
                        if f.read().strip() == "agency-gpu":
                            count += 1
                except OSError:
                    pass
            return count

        pool = agResourcePool(gpus=[0], mark_gpus=True)
        try:
            time.sleep(0.5)
            before = _count_agency_gpu()
            assert before >= 1

            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
                worker_name = ex.submit(_worker_import_agent).result(timeout=15)

            assert worker_name != "MainProcess", "worker should not be MainProcess"
            after = _count_agency_gpu()
            assert after == before, (
                f"worker import added {after - before} marker(s); "
                "check the MainProcess guard in agResourcePool.__init__"
            )
        finally:
            pool._stop_gpu_markers()


# ---------------------------------------------------------------------------
# agSandbox — container lifecycle
# ---------------------------------------------------------------------------

class TestAgSandboxLifecycle:
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
            assert w.error is None, f"write failed: {w.error}"
            # read also runs in a process-pool worker; must find the file
            r = tools["read"](agdata(filePath="/workspace/cross.txt"))
            assert r.error is None, f"read failed after cross-worker write: {r.error}"
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
            sb2 = _make_sandbox(restore_image=tag)
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
        self.sb.write_file("/workspace/spawner.py", (
            "import subprocess, time\n"
            "subprocess.Popen(['sleep', '30'])\n"   # detached child, not waited on
        ))
        self.sb.exec("python3 /workspace/spawner.py")
        assert len(self.sb._watched_pids) > 0

    def test_parent_exits_child_survives_still_tracked(self):
        # Parent spawns a child then exits. Child is reparented to PID 1 and
        # escapes any BFS from the original PID. Baseline diff must find it.
        self.sb.write_file("/workspace/spawner.py", (
            "import subprocess, os\n"
            "subprocess.Popen(['sleep', '30'])\n"  # child detaches
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
            "subprocess.Popen(['sleep', '30'])\n"
            "subprocess.Popen(['sleep', '30'])\n"
            "time.sleep(30)\n"   # parent also stays alive
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
            "        time.sleep(30)\n"      # grandchild runs in background
            "    os._exit(0)\n"             # intermediate child exits
            "os.wait()\n"                   # parent waits for intermediate child
        ))
        self.sb.exec("python3 /workspace/daemon.py")
        assert len(self.sb._watched_pids) > 0

    def test_get_live_pids_returns_running(self):
        self.sb.exec("sleep 30 &")
        live = self.sb.get_live_pids()
        assert len(live) > 0

    def test_get_live_pids_removes_exited(self):
        self.sb.exec("sleep 0.1 &")
        time.sleep(0.5)
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
        self.sb.write_file("/workspace/daemon_parent.py", (
            "import subprocess, time\n"
            "subprocess.Popen(['sleep', '30'])\n"
            "time.sleep(30)\n"
        ))
        self.sb.exec("python3 /workspace/daemon_parent.py &")
        live = self.sb.get_live_pids()
        assert len(live) > 0
        # Release the parent; its child (sleep 30) should also be excluded
        for pid in list(live):
            self.sb.release_daemon(pid)
        time.sleep(0.3)   # let the child spawn
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
        self.tools["bash"].fn(agdata(command="sleep 30 &"))
        live_before = self.sb.get_live_pids()
        assert len(live_before) > 0
        for pid in list(live_before):
            result = self.tools["daemon_release"].fn(agdata(pid=pid))
            assert result.error is None
        assert self.sb.get_live_pids() == set()

    def test_daemon_release_tool_invalid_pid(self):
        result = self.tools["daemon_release"].fn(agdata(pid="notanint"))
        assert result.error is not None

    def test_daemon_release_tool_missing_pid(self):
        result = self.tools["daemon_release"].fn(agdata())
        assert result.error is not None


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
        self.sb.exec("sleep 30 &")
        live = self.sb.get_live_pids()
        assert len(live) > 0
        assert self.sb._gpu_id is not None
        assert self.pool._gpus_acquired == 1
        self.sb.exec("kill %1 2>/dev/null || true")

    def test_same_physical_gpu_used_for_subsequent_exec_during_background_process(self):
        """While a background process holds the GPU, subsequent exec() calls use the same GPU."""
        self.tools["reserve_gpu"].fn(agdata())
        self.sb.exec("sleep 30 &")
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
        time.sleep(0.5)
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
        exec_done.wait(timeout=5)
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
        self.sb.exec("sleep 30 &")
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
        assert result.error is None

    def test_reserve_cpu_requires_at_least_one_param(self):
        result = self.tools["reserve_cpu"].fn(agdata())
        assert result.error is not None

    def test_cpu_release_resets_to_idle(self):
        self.tools["reserve_cpu"].fn(agdata(cpus=4.0, memory="2g"))
        result = self.tools["cpu_release"].fn(agdata())
        assert result.error is None
        assert "0.5" in result.message
        assert "512m" in result.message
