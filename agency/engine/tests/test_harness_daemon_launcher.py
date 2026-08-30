from __future__ import annotations

from agency.agconfig import agConfig
from agency.engine import harness_daemon_launcher as launcher


class _FakeSandbox:
    def __init__(self) -> None:
        self.detached = []

    def exec_detached(self, command, workdir):
        self.detached.append((command, workdir))

    def exec(self, *_args, **_kwargs):
        return "daemon log", 0


def test_ensure_harness_daemon_launches_module_with_gateway_socket_paths(monkeypatch):
    sandbox = _FakeSandbox()
    readiness = iter([False, True])
    monkeypatch.setattr(launcher, "_is_ready", lambda _handle, timeout_s=0.5: next(readiness))

    handle = launcher.ensure_harness_daemon(
        sandbox,
        "/tmp/agency/gw/run/host-agent.sock",
        "agent-1",
        "claude_code",
        agconfig=agConfig({"agharness": {"binary_path": "/bin/claude"}}),
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
    assert workdir == "/workspace"


def test_ensure_harness_daemon_waits_for_readiness_before_returning(monkeypatch):
    sandbox = _FakeSandbox()
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


def test_duplicate_ensure_reuses_ready_daemon_without_relaunching(monkeypatch):
    sandbox = _FakeSandbox()
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


def test_daemon_config_excludes_unrelated_and_secret_host_configuration():
    config = agConfig(
        {
            "agharness": {"binary_path": "/bin/claude"},
            "agllm_backend": {"api_key": "secret"},
            "agent": {"harness": "claude_code"},
        }
    )

    assert launcher._daemon_config(config) == {"agharness": {"binary_path": "/bin/claude"}}
