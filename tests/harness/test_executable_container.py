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
    prepare_harness_executable,
)
from agency.sandbox.agsandbox import agSandbox
from agency.utils.agutil import agharness_llm_gateway_dir


pytestmark = pytest.mark.skipif(
    os.environ.get("AGENCY_TEST_EXTERNAL_HARNESSES") != "1",
    reason="opt-in real Linux harness/container test",
)


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
    sandbox = agSandbox(f"external-{harness}", agconfig=config)
    try:
        prepared = prepare_harness_executable(sandbox, harness, config)
        assert prepared == str(Path(host_binary).resolve())
        out, rc = sandbox.exec(f"PATH={HARNESS_PATH} {shlex.quote(prepared)} --version", timeout=30)
        assert rc == 0, out
        host_version = subprocess.check_output([host_binary, "--version"], text=True).strip()
        assert host_version in out
        print(f"{harness}: {host_version}; executable={prepared}", flush=True)

        for source, destination, mode in harness_installation_mounts(config).values():
            assert source == destination and mode == "ro"
            out, rc = sandbox.exec(
                f"touch {shlex.quote(destination + '/.agency-readonly-probe')}", workdir="/"
            )
            assert rc != 0 and "Read-only file system" in out, out

        # Use the real launcher and real daemon too, rather than checking a
        # hand-constructed docker run with mounts different from Agency's.
        host_socket = str(agharness_llm_gateway_dir() / f"host-{sandbox._agname}.sock")
        handle = ensure_harness_daemon(
            sandbox, host_socket, sandbox._agname, harness, agconfig=config, timeout_s=45
        )
        with handle.client(timeout_s=5) as client:
            assert client.is_ready()

        # Exercise the actual process launcher used by the adapters as well:
        # in particular, Codex's Node launcher spawns its native child here.
        probe = f"""\
from agency.configs.agconfig import agconfig
from agency.harness.ptrace.supervisor import agProxyPtrace

class StartupPolicy:
    def check(self, agent, event):
        return True

handle = agProxyPtrace(agconfig(), allow_initial_exec=True).launch(
    [{prepared!r}, "--version"], {{"PATH": {HARNESS_PATH!r}}},
    cwd="/workspace", policy=StartupPolicy(), ag=None,
)
stdout, stderr, rc = handle.wait(timeout=30)
print(stdout)
print(stderr)
raise SystemExit(rc)
"""
        sandbox.write_file("/workspace/harness-startup-probe.py", probe)
        out, rc = sandbox.exec(
            "PYTHONPATH=/opt/agency_pkg python3 /workspace/harness-startup-probe.py", timeout=45
        )
        assert rc == 0 and host_version in out, out
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
        with pytest.raises(RuntimeError, match="node"):
            prepare_harness_executable(sandbox, "codex", config)
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
    # construction and actual executable startup are exercised.
    owner = Agent(harness=harness, agconfig=config)
    try:
        sandbox = owner._ensure_sandbox()
        assert sandbox.agconfig.agent.harness == harness
        assert prepare_harness_executable(sandbox, harness, config)
    finally:
        if owner.sandbox is not None:
            owner.sandbox.destroy()
