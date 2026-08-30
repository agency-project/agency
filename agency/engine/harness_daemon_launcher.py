"""Launch and discover the sandbox-side Harness Manager daemon."""

from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..agutil import AGENCY_PACKAGE_CONTAINER_MOUNT, ensure_python_packages_in_container
from .clients import SandboxInteractionClient

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..sandbox.agsandbox import agSandbox

_CONTAINER_GATEWAY_DIR = "/var/run/agency_llm_gateway"
_DAEMON_LOG_PATH = "/tmp/agency-harness-daemon.log"


@dataclass(frozen=True)
class DaemonHandle:
    host_uds_path: str
    sandbox_uds_path: str
    container_host_uds_path: str
    container_sandbox_uds_path: str
    engine_name: str

    def client(self, timeout_s: float = 300.0) -> SandboxInteractionClient:
        return SandboxInteractionClient(self.sandbox_uds_path, timeout_s=timeout_s)


def _sandbox_socket_path(host_uds_path: str) -> Path:
    host_socket = Path(host_uds_path)
    if host_socket.name == "host.sock":
        name = "sandbox.sock"
    elif host_socket.name.startswith("host-"):
        name = f"sandbox-{host_socket.name.removeprefix('host-')}"
    else:
        name = f"sandbox-{host_socket.name}"
    return host_socket.with_name(name)


def _container_socket_path(host_path: Path) -> str:
    return f"{_CONTAINER_GATEWAY_DIR}/{host_path.name}"


def _daemon_config(agconfig: "agConfig | None") -> dict:
    if agconfig is None:
        return {}
    # The daemon needs harness/ptrace knobs, not host credentials or live
    # Python objects. Keeping this allow-list narrow also keeps secrets out
    # of the detached process command line.
    return {
        owner: dict(agconfig.data[owner])
        for owner in ("agharness", "agproxy_ptrace")
        if owner in agconfig.data
    }


def _is_ready(handle: DaemonHandle, timeout_s: float = 0.5) -> bool:
    try:
        with handle.client(timeout_s=timeout_s) as client:
            return client.is_ready()
    except Exception:
        return False


def ensure_harness_daemon(
    sandbox: "agSandbox",
    host_uds_path: str,
    engine_name: str,
    harness: str,
    *,
    agconfig: "agConfig | None" = None,
    timeout_s: float = 30.0,
) -> DaemonHandle:
    """Ensure one ready Harness Manager exists for this engine and sandbox."""
    handles = getattr(sandbox, "_agency_harness_daemon_handles", None)
    if handles is None:
        handles = {}
        sandbox._agency_harness_daemon_handles = handles

    existing = handles.get(engine_name)
    if existing is not None and _is_ready(existing):
        return existing

    host_path = Path(host_uds_path)
    sandbox_path = _sandbox_socket_path(host_uds_path)
    handle = DaemonHandle(
        host_uds_path=str(host_path),
        sandbox_uds_path=str(sandbox_path),
        container_host_uds_path=_container_socket_path(host_path),
        container_sandbox_uds_path=_container_socket_path(sandbox_path),
        engine_name=engine_name,
    )
    config_json = json.dumps(_daemon_config(agconfig), separators=(",", ":"))

    ensure_python_packages_in_container(
        sandbox,
        ["fastapi", "uvicorn", "openai", "httpx", "mcp", "pyseccomp"],
        timeout_s=180,
    )

    # Actual launch of the daemon
    command = (
        "PATH=/opt/agency_harness_bin:/usr/local/bin:/usr/bin:/bin "
        f"PYTHONPATH={shlex.quote(AGENCY_PACKAGE_CONTAINER_MOUNT)} "
        "exec python3 -m agency.harness.daemon "
        f"--sandbox-uds {shlex.quote(handle.container_sandbox_uds_path)} "
        f"--host-uds {shlex.quote(handle.container_host_uds_path)} "
        f"--engine-name {shlex.quote(engine_name)} "
        f"--harness {shlex.quote(harness)} "
        f"--config-json {shlex.quote(config_json)} "
        f"> {shlex.quote(_DAEMON_LOG_PATH)} 2>&1"
    )

    sandbox.exec_detached(command, workdir="/workspace")

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _is_ready(handle):
            handles[engine_name] = handle
            return handle
        time.sleep(0.05)

    try:
        log_tail, _ = sandbox.exec(
            f"tail -c 4000 {shlex.quote(_DAEMON_LOG_PATH)} 2>/dev/null",
            timeout=10,
        )
    except Exception as exc:
        log_tail = f"could not read daemon log: {exc}"
    raise RuntimeError(
        f"Harness Manager did not become ready at {handle.sandbox_uds_path!r} "
        f"within {timeout_s}s. Log ({_DAEMON_LOG_PATH}):\n{log_tail}"
    )


__all__ = ["DaemonHandle", "ensure_harness_daemon"]
