"""Unit tests for agSandbox and agResourcePool.

Sandbox tests that create real containers are marked with @pytest.mark.docker
and skipped automatically when Docker/Podman is unavailable.
"""
from __future__ import annotations

import subprocess
import threading
import time
import uuid

import pytest

from agency.agdata import agdata
from agency.agresources import agResourcePool


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
        self.sb.release_resources(pool)
        assert self.sb._gpu_id is None

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

    def test_gpu_acquire_sets_gpu_id(self):
        result = self.tools["gpu_acquire"].fn(agdata())
        assert result.gpu_id in (0, 1)
        assert self.sb._gpu_id == result.gpu_id

    def test_gpu_acquire_idempotent(self):
        self.tools["gpu_acquire"].fn(agdata())
        first_id = self.sb._gpu_id
        result = self.tools["gpu_acquire"].fn(agdata())
        assert result.gpu_id == first_id   # same GPU returned

    def test_gpu_release_clears_gpu_id(self):
        self.tools["gpu_acquire"].fn(agdata())
        self.tools["gpu_release"].fn(agdata())
        assert self.sb._gpu_id is None

    def test_gpu_release_without_acquire(self):
        result = self.tools["gpu_release"].fn(agdata())
        assert "no GPU" in result.message

    def test_gpu_acquire_timeout_with_exhausted_pool(self):
        pool = agResourcePool(gpus=[0])
        pool.acquire_gpu()             # exhaust the single GPU
        from agency.tools.resource import make_gpu_acquire
        tool = make_gpu_acquire(self.sb, pool)
        result = tool.fn(agdata(timeout=0.2))
        assert result.error is not None

    def test_cpu_acquire_applies_limits(self):
        result = self.tools["cpu_acquire"].fn(agdata(cpus=2.0, memory="256m"))
        assert result.error is None

    def test_cpu_acquire_requires_at_least_one_param(self):
        result = self.tools["cpu_acquire"].fn(agdata())
        assert result.error is not None

    def test_cpu_release_resets_to_idle(self):
        self.tools["cpu_acquire"].fn(agdata(cpus=4.0, memory="2g"))
        result = self.tools["cpu_release"].fn(agdata())
        assert result.error is None
        assert "0.5" in result.message     # idle_cpus
        assert "512m" in result.message    # idle_memory
