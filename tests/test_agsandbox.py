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
    def test_fork_creates_independent_container(self):
        parent = _make_sandbox()
        child = _make_sandbox(parent_uuid=parent._uuid)
        assert parent._container_name() != child._container_name()
        assert child._snapshot_name is not None
        child.destroy()
        parent.destroy()

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
    def test_destroy_removes_snapshot_image(self):
        parent = _make_sandbox()
        child = _make_sandbox(parent_uuid=parent._uuid)
        snap = child._snapshot_name
        assert snap is not None
        child.destroy()
        result = subprocess.run(
            ["docker", "images", "-q", snap],
            capture_output=True, text=True,
        )
        assert result.stdout.strip() == ""
        parent.destroy()


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
# agSandbox — fork filesystem isolation
# ---------------------------------------------------------------------------

class TestAgSandboxForkIsolation:
    @docker
    def test_fork_inherits_parent_files(self):
        parent = _make_sandbox()
        parent.write_file("/workspace/shared.txt", "from parent\n")
        child = _make_sandbox(parent_uuid=parent._uuid)
        content = child.read_file("/workspace/shared.txt")
        assert "from parent" in content
        child.destroy()
        parent.destroy()

    @docker
    def test_fork_writes_do_not_affect_parent(self):
        parent = _make_sandbox()
        parent.write_file("/workspace/base.txt", "original\n")
        child = _make_sandbox(parent_uuid=parent._uuid)
        child.write_file("/workspace/base.txt", "modified\n")
        # Parent should still have the original content
        assert "original" in parent.read_file("/workspace/base.txt")
        child.destroy()
        parent.destroy()


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


# ---------------------------------------------------------------------------
# Real wall-clock outer monitoring loop integration test
# ---------------------------------------------------------------------------

@docker
@pytest.mark.timeout(60)
def test_outer_loop_real_process_wall_clock():
    """Full integration: real container, real background process, real polling.

    A background job sleeps for JOB_DURATION seconds inside a real container.
    The outer monitoring loop polls every POLL_INTERVAL seconds and must fire
    process_completed within one extra poll cycle after the job exits.

    Expected timeline:
      0s    agent starts, job launched via sandbox.exec("sleep N &")
      ~Ns   job exits; next get_live_pids() poll finds it gone
      ~N+Ps process_completed re-entry fires
      ~N+2Ps skill resolves (after process_completed re-entry returns)

    Total wall time: JOB_DURATION + 2*POLL_INTERVAL + container overhead.
    Asserted range: [JOB_DURATION, JOB_DURATION + POLL_INTERVAL*3 + 5]
    """
    from agency.agent import agent
    from agency.agskill import agskill

    JOB_DURATION  = 8   # seconds the background process runs
    POLL_INTERVAL = 2   # poll every 2s — fast enough to detect promptly

    ag = agent(llm_config={"api_key": "k", "model": "gpt-4o"}, agskills=[])

    agent.poll_interval_s = POLL_INTERVAL
    agent.ping_interval_s = 60   # high ceiling — job should finish well before

    events: list[dict] = []

    def fake_run(llm_cfg, inp, hist, tools, ms, **_):
        event = inp._data.get("_event")
        events.append({"event": event, "t": time.monotonic()})
        if event is None:
            # Launch a real background process in the container
            ag.sandbox.exec(f"sleep {JOB_DURATION} &")
        return agdata(result="ok"), agdata(messages=[]), []

    skill = agskill(name="s", system_prompt="")
    skill.run = fake_run
    ag.agskills = [skill]

    t0 = time.monotonic()
    ag.run("s", agdata()).result
    elapsed = time.monotonic() - t0

    event_names = [e["event"] for e in events]

    # Sequence: initial call → process_completed re-entry
    assert event_names[0] is None,                  "first call must be the initial one"
    assert "process_completed" in event_names,       "process_completed must fire"
    assert "process_update" not in event_names,      "job should finish before ping_interval_s"

    # Timing: completed must fire after the job actually ran
    t_completed = next(e["t"] for e in events if e["event"] == "process_completed") - t0
    assert t_completed >= JOB_DURATION, (
        f"process_completed fired at {t_completed:.1f}s — before job finished at {JOB_DURATION}s"
    )
    assert t_completed <= JOB_DURATION + POLL_INTERVAL * 3 + 5, (
        f"process_completed took {t_completed:.1f}s — too slow "
        f"(expected ≤{JOB_DURATION + POLL_INTERVAL * 3 + 5}s)"
    )

    print(f"\n  job={JOB_DURATION}s  poll={POLL_INTERVAL}s  "
          f"detected at {t_completed:.1f}s  total={elapsed:.1f}s")
