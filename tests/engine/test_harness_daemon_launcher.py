from __future__ import annotations

import json
import shlex
from types import SimpleNamespace

import pytest

from agency.configs.agconfig import (
    agconfig,
    agentconfig,
    dataloggerconfig,
    harnessadapterconfig,
    llmconfig,
)
from agency.engine import harness_daemon_launcher as launcher


class _FakeSandbox:
    def __init__(self, sandbox_agconfig) -> None:
        self.detached = []
        self.agconfig = sandbox_agconfig
        self._backend = SimpleNamespace()

    def exec_detached(self, command, workdir):
        self.detached.append((command, workdir))

    def exec(self, command, **_kwargs):
        if "command -v" in command:
            binary = shlex.split(command)[-1]
            return binary if "/" in binary else f"/usr/bin/{binary}", 0
        return "daemon log", 0


def _fake_sandbox(tmp_path, *namespaces) -> _FakeSandbox:
    """A fake sandbox whose agconfig.data_logger.db_path is pinned under
    pytest's own tmp_path -- keeps _preclaim_host_daemon_log's real
    filesystem writes hermetic to the test instead of touching the real
    process-wide agency_runs_dir() default."""
    cfg = agconfig(dataloggerconfig(db_path=str(tmp_path / "agent_data.sqlite3")), *namespaces)
    return _FakeSandbox(cfg)


def test_ensure_harness_daemon_launches_module_with_gateway_socket_paths(monkeypatch, tmp_path):
    sandbox = _fake_sandbox(tmp_path, harnessadapterconfig(binary_path="/bin/claude"))
    readiness = iter([False, True])
    monkeypatch.setattr(launcher, "_is_ready", lambda _handle, timeout_s=0.5: next(readiness))

    handle = launcher.ensure_harness_daemon(
        sandbox,
        "/tmp/agency/gw/run/host-agent.sock",
        "agent-1",
        "claude_code",
        agconfig=sandbox.agconfig,
        timeout_s=1.0,
    )

    assert handle.sandbox_uds_path == "/tmp/agency/gw/run/sandbox-agent.sock"
    assert handle.container_host_uds_path == "/var/run/agency_llm_gateway/host-agent.sock"
    assert handle.container_sandbox_uds_path == ("/var/run/agency_llm_gateway/sandbox-agent.sock")
    command, workdir = sandbox.detached[0]
    assert "python3 -m agency.harness.daemon" in command
    assert "--host-uds /var/run/agency_llm_gateway/host-agent.sock" in command
    assert "--sandbox-uds /var/run/agency_llm_gateway/sandbox-agent.sock" in command
    assert "--harness claude_code" in command
    assert '"binary_path":"/bin/claude"' in command
    assert f"> {launcher.AGENCY_LOGS_CONTAINER_MOUNT}/daemon-agent-1.log" in command
    assert workdir == "/workspace"


def test_ensure_harness_daemon_default_timeout_is_60s():
    import inspect

    assert inspect.signature(launcher.ensure_harness_daemon).parameters["timeout_s"].default == 60.0


def test_package_bootstrap_runs_before_the_daemon_module_is_ever_imported(monkeypatch, tmp_path):
    """Importing agency.harness.daemon already needs most of these packages
    transitively -- the bootstrap that installs them must run as its own
    dependency-free command ahead of that import, not inside it."""
    sandbox = _fake_sandbox(tmp_path, harnessadapterconfig(binary_path="/bin/claude"))
    monkeypatch.setattr(launcher, "_is_ready", lambda *a, **kw: True)

    launcher.ensure_harness_daemon(
        sandbox, "/tmp/host.sock", "agent-1", "claude_code", agconfig=sandbox.agconfig
    )

    command = sandbox.detached[0][0]
    bootstrap_index = command.index("import importlib, socket, subprocess, sys")
    daemon_index = command.index("-m agency.harness.daemon")
    assert bootstrap_index < daemon_index
    for pkg in launcher._REQUIRED_HARNESS_PACKAGES:
        assert repr(pkg).strip("'\"") in command or pkg in command
    assert " && exec " in command


def test_package_bootstrap_refuses_install_when_harness_python_is_pinned(monkeypatch, tmp_path):
    from agency.configs.agconfig import sandboxconfig

    sandbox = _fake_sandbox(
        tmp_path,
        harnessadapterconfig(binary_path="/bin/claude"),
        sandboxconfig(harness_python_path="/opt/e1a1-venv/bin/python"),
    )
    monkeypatch.setattr(launcher, "_is_ready", lambda *a, **kw: True)

    launcher.ensure_harness_daemon(
        sandbox, "/tmp/host.sock", "agent-1", "claude_code", agconfig=sandbox.agconfig
    )

    command = sandbox.detached[0][0]
    assert "install_missing = False" in command


def test_pinned_harness_python_path_is_passed_through_for_the_daemon_to_check_itself(
    monkeypatch, tmp_path
):
    from agency.configs.agconfig import sandboxconfig

    sandbox = _fake_sandbox(
        tmp_path,
        harnessadapterconfig(binary_path="/bin/claude"),
        sandboxconfig(harness_python_path="/opt/e1a1-venv/bin/python"),
    )
    monkeypatch.setattr(launcher, "_is_ready", lambda _handle, timeout_s=0.5: True)
    launcher.ensure_harness_daemon(
        sandbox,
        "/tmp/agency/gw/run/host-agent.sock",
        "agent-1",
        "claude_code",
        agconfig=sandbox.agconfig,
        timeout_s=1.0,
    )
    command = sandbox.detached[0][0]
    # The daemon reads this back out of its own --config-json to decide
    # whether it may pip-install a missing package into itself (only when
    # unpinned) once it handles its first attempt.
    assert '"harness_python_path":"/opt/e1a1-venv/bin/python"' in command
    assert "exec /opt/e1a1-venv/bin/python -m agency.harness.daemon" in command


def test_ensure_harness_daemon_waits_for_readiness_before_returning(monkeypatch, tmp_path):
    sandbox = _fake_sandbox(tmp_path)
    probes = []

    def ready(_handle, timeout_s=0.5):
        probes.append(timeout_s)
        return len(probes) == 3

    monkeypatch.setattr(launcher, "_is_ready", ready)

    launcher.ensure_harness_daemon(
        sandbox,
        "/tmp/host.sock",
        "agent-1",
        "claude_code",
        timeout_s=1.0,
    )

    assert len(probes) == 3


def test_duplicate_ensure_reuses_ready_daemon_without_relaunching(monkeypatch, tmp_path):
    sandbox = _fake_sandbox(tmp_path)
    readiness = iter([True, True])
    monkeypatch.setattr(launcher, "_is_ready", lambda _handle, timeout_s=0.5: next(readiness))

    first = launcher.ensure_harness_daemon(
        sandbox,
        "/tmp/host.sock",
        "agent-1",
        "claude_code",
        timeout_s=1.0,
    )
    second = launcher.ensure_harness_daemon(
        sandbox,
        "/tmp/host.sock",
        "agent-1",
        "claude_code",
        timeout_s=1.0,
    )

    assert second is first
    assert len(sandbox.detached) == 1


def test_daemon_log_is_preclaimed_host_side_before_container_can_write_it(monkeypatch, tmp_path):
    """The container's own process (commonly root) must open, not create,
    daemon.log -- opening an existing file never changes its ownership, so
    pre-creating it here (world-writable, this test process's own uid) is
    what keeps it out of root's hands regardless of which side writes to it
    first. See _preclaim_host_daemon_log's docstring for the full story."""
    sandbox = _fake_sandbox(tmp_path)
    monkeypatch.setattr(launcher, "_is_ready", lambda _handle, timeout_s=0.5: True)

    launcher.ensure_harness_daemon(
        sandbox,
        "/tmp/host.sock",
        "agent-1",
        "claude_code",
        timeout_s=1.0,
    )

    log_path = launcher._host_daemon_log_path(sandbox, "agent-1")
    assert log_path == tmp_path / "daemon-agent-1.log"
    assert log_path.exists()
    assert oct(log_path.stat().st_mode)[-3:] == "666"


def test_daemon_config_excludes_unrelated_and_secret_host_configuration():
    config = agconfig(
        harnessadapterconfig(binary_path="/bin/claude"),
        llmconfig(api_key="secret"),
        agentconfig(harness="claude_code"),
    )

    assert launcher._daemon_config(config) == {
        "harness_adapter": {"binary_path": "/bin/claude"},
        "sandbox": {"checkpoint_fast_resume": False, "harness_python_path": None},
        "ptrace": {
            "syscalls": list(agconfig().ptrace.syscalls),
            "file_access": False,
            "profiler": None,
            "disable_harness_native_sandbox": True,
        },
    }


@pytest.mark.parametrize("harness", ["claude_code", "codex", "grok", "opencode"])
def test_all_external_harnesses_receive_a_resolved_absolute_binary_path(
    monkeypatch, tmp_path, harness
):
    # The install directory is bind-mounted at its original host path, which
    # the daemon's own (deliberately narrow) PATH can't rediscover on its
    # own -- a bare default binary name (no explicit binary_path configured)
    # must come out resolved to an absolute path.
    sandbox = _fake_sandbox(tmp_path, agentconfig(harness=harness))
    monkeypatch.setattr(launcher, "_is_ready", lambda *a, **kw: True)
    launcher.ensure_harness_daemon(
        sandbox, "/tmp/host.sock", "agent-1", harness, agconfig=sandbox.agconfig
    )
    command = shlex.split(sandbox.detached[0][0])
    config = json.loads(command[command.index("--config-json") + 1])
    resolved = config["harness_adapter"]["binary_path"]
    assert resolved is not None
    assert resolved.startswith("/"), f"expected an absolute path, got {resolved!r}"


def test_explicit_binary_path_is_resolved_the_same_way(monkeypatch, tmp_path):
    configured = "/host/install/claude_code/bin/cli"
    sandbox = _fake_sandbox(
        tmp_path,
        agentconfig(harness="claude_code"),
        harnessadapterconfig(binary_path=configured),
    )
    monkeypatch.setattr(launcher, "_is_ready", lambda *a, **kw: True)
    launcher.ensure_harness_daemon(
        sandbox, "/tmp/host.sock", "agent-1", "claude_code", agconfig=sandbox.agconfig
    )
    command = shlex.split(sandbox.detached[0][0])
    config = json.loads(command[command.index("--config-json") + 1])
    assert config["harness_adapter"]["binary_path"] == configured
