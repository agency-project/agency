"""Opt-in Linux Docker checks with the real installed external CLIs.

AGENCY_TEST_EXTERNAL_HARNESSES=1 AGENCY_TEST_HARNESS_IMAGE=<image> pytest -s ...
The image must provide system runtimes (notably Node for npm launchers).
The host PATH must include the installations being tested.
"""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from agency.configs.agconfig import (
    agconfig,
    agentconfig,
    harnessadapterconfig,
    llmconfig,
    sandboxconfig,
)
from agency.engine.harness_daemon_launcher import ensure_harness_daemon
from agency.harness.executable import (
    HARNESS_PATH,
    harness_installation_mounts,
    resolve_harness_binary,
)
from agency.sandbox.agsandbox import agSandbox
from agency.utils.agutil import agharness_llm_gateway_dir


pytestmark = pytest.mark.skipif(
    os.environ.get("AGENCY_TEST_EXTERNAL_HARNESSES") != "1",
    reason="opt-in real Linux harness/container test",
)


def _prepare_local_probe(harness: str, binary_path: str) -> str:
    """A script run inside the sandbox that exercises
    prepare_harness_executable_local() and the real process launcher the
    adapters use, the same way the daemon does on its first attempt."""
    return f"""\
from agency.configs.agconfig import agconfig, agentconfig, harnessadapterconfig
from agency.harness.executable import prepare_harness_executable_local
from agency.harness.ptrace.supervisor import agProxyPtrace

config = agconfig(agentconfig(harness={harness!r}), harnessadapterconfig(binary_path={binary_path!r}))
prepared = prepare_harness_executable_local({harness!r}, config)

class StartupPolicy:
    def check(self, agent, event):
        return True

handle = agProxyPtrace(agconfig(), allow_initial_exec=True).launch(
    [prepared, "--version"], {{"PATH": {HARNESS_PATH!r}}},
    cwd="/workspace", policy=StartupPolicy(), ag=None,
)
stdout, stderr, rc = handle.wait(timeout=30)
print(stdout)
print(stderr)
raise SystemExit(rc)
"""


@pytest.mark.parametrize(
    ("harness", "binary"),
    [("claude_code", "claude"), ("codex", "codex"), ("grok", "grok"), ("opencode", "opencode")],
)
@pytest.mark.parametrize("explicit_path", [False, True])
def test_real_installed_cli_launches_from_read_only_installation(harness, binary, explicit_path):
    host_binary = shutil.which(binary)
    assert host_binary is not None, f"Install {binary} and add it to the host PATH"
    config = agconfig(
        agentconfig(harness=harness),
        harnessadapterconfig(binary_path=host_binary if explicit_path else None),
        sandboxconfig(
            backend="docker",
            base_image=os.environ.get(
                "AGENCY_TEST_HARNESS_IMAGE", "docker.io/library/python:3.12-slim"
            ),
        ),
    )
    resolved = resolve_harness_binary(harness, config)
    assert resolved == str(Path(host_binary).resolve())
    host_version = subprocess.check_output([host_binary, "--version"], text=True).strip()

    sandbox = agSandbox(f"external-{harness}", agconfig=config)
    try:
        for source, destination, mode in harness_installation_mounts(config).values():
            assert source == destination and mode == "ro"
            out, rc = sandbox.exec(
                f"touch {shlex.quote(destination + '/.agency-readonly-probe')}", workdir="/"
            )
            assert rc != 0 and "Read-only file system" in out, out

        sandbox.write_file(
            "/workspace/harness-startup-probe.py", _prepare_local_probe(harness, resolved)
        )
        out, rc = sandbox.exec(
            "PYTHONPATH=/opt/agency_pkg python3 /workspace/harness-startup-probe.py", timeout=45
        )
        assert rc == 0 and host_version in out, out
        print(f"{harness}: {host_version}; executable={resolved}", flush=True)

        # Use the real launcher and real daemon too, rather than checking a
        # hand-constructed docker run with mounts different from Agency's.
        host_socket = str(agharness_llm_gateway_dir() / f"host-{sandbox._agname}.sock")
        handle = ensure_harness_daemon(
            sandbox, host_socket, sandbox._agname, harness, agconfig=config, timeout_s=45
        )
        with handle.client(timeout_s=5) as client:
            assert client.is_ready()
    finally:
        sandbox.destroy()


def test_real_codex_reports_missing_image_node_runtime():
    config = agconfig(
        agentconfig(harness="codex"),
        sandboxconfig(backend="docker", base_image="docker.io/library/python:3.12-slim"),
    )
    sandbox = agSandbox("external-missing-node", agconfig=config)
    try:
        _, rc = sandbox.exec(f"PATH={HARNESS_PATH} command -v node")
        if rc == 0:
            pytest.skip("base image already provides Node")
        resolved = resolve_harness_binary("codex", config)
        assert resolved is not None
        sandbox.write_file(
            "/workspace/harness-startup-probe.py", _prepare_local_probe("codex", resolved)
        )
        out, rc = sandbox.exec(
            "PYTHONPATH=/opt/agency_pkg python3 /workspace/harness-startup-probe.py", timeout=45
        )
        assert rc != 0 and "node" in out.lower(), out
    finally:
        sandbox.destroy()


@pytest.mark.parametrize("harness", ["claude_code", "codex", "grok", "opencode"])
def test_real_agent_harness_override_prepares_installation(harness):
    from agency import Agent

    config = agconfig(
        llmconfig(provider="openai", model="unused-startup-test", api_key="unused"),
        sandboxconfig(
            backend="docker",
            base_image=os.environ.get(
                "AGENCY_TEST_HARNESS_IMAGE", "docker.io/library/python:3.12-slim"
            ),
        ),
    )
    # This intentionally leaves config.agent.harness at its native default.
    # No invocation/model request is made; only the public Agent's sandbox
    # construction and host-side binary resolution are exercised.
    owner = Agent(harness=harness, agconfig=config)
    try:
        sandbox = owner._ensure_sandbox()
        assert sandbox.agconfig.agent.harness == harness
        assert resolve_harness_binary(harness, config)
    finally:
        if owner.sandbox is not None:
            owner.sandbox.destroy()
