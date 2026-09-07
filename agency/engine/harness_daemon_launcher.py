"""Launch and discover the sandbox-side Harness Manager daemon."""

from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils.agutil import (
    AGENCY_LLM_GATEWAY_CONTAINER_MOUNT,
    AGENCY_LOGS_CONTAINER_MOUNT,
    AGENCY_PACKAGE_CONTAINER_MOUNT,
    ensure_python_packages_in_container,
)
from .clients import SandboxInteractionClient

if TYPE_CHECKING:
    from ..configs.agconfig import agconfig as agconfig_cls
    from ..sandbox.agsandbox import agSandbox

_CONTAINER_GATEWAY_DIR = AGENCY_LLM_GATEWAY_CONTAINER_MOUNT
_DAEMON_LOG_PATH = f"{AGENCY_LOGS_CONTAINER_MOUNT}/daemon.log"


@dataclass(frozen=True)
class DaemonHandle:
    host_uds_path: str
    sandbox_uds_path: str
    container_host_uds_path: str
    container_sandbox_uds_path: str
    engine_name: str

    def client(self, timeout_s: "float | None" = 300.0) -> SandboxInteractionClient:
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


def _daemon_config(agconfig: "agconfig_cls | None") -> dict:
    if agconfig is None:
        return {}
    # The daemon needs harness/ptrace knobs, not host credentials or live
    # Python objects. Keeping this allow-list narrow also keeps secrets out
    # of the detached process command line.
    return {
        "harness_adapter": {"binary_path": agconfig.harness_adapter.binary_path},
        "ptrace": {
            "syscalls": list(agconfig.ptrace.syscalls),
            "profiler": agconfig.ptrace.profiler,
            "disable_harness_native_sandbox": agconfig.ptrace.disable_harness_native_sandbox,
        },
    }


def _is_ready(handle: DaemonHandle, timeout_s: float = 0.5) -> bool:
    try:
        with handle.client(timeout_s=timeout_s) as client:
            return client.is_ready()
    except Exception:
        return False


def _host_daemon_log_path(sandbox: "agSandbox") -> Path:
    """This run's host-side counterpart of `_DAEMON_LOG_PATH`, mirroring
    agsandbox.py's own derivation of the directory it bind-mounts as
    `_agency_logs` (agconfig.data_logger.db_path's parent, falling back to
    `_DEFAULT_LOG_DIR`) -- must stay identical to that derivation or this
    touches a different file than the one actually mounted into the
    container."""
    from ..utils.agutil import _DEFAULT_LOG_DIR

    db_path = sandbox.agconfig.data_logger.db_path
    log_dir = Path(db_path).parent if db_path else _DEFAULT_LOG_DIR
    return log_dir / "daemon.log"


def _preclaim_host_daemon_log(sandbox: "agSandbox") -> None:
    """Create this run's daemon.log on the HOST side, world-writable,
    before the container ever touches it.

    Without this, whichever side opens the (bind-mounted, shared) path
    first with a normal, non-existent file wins its ownership. The
    container's shell redirection runs as the container's own user
    (commonly root, unlike this host process) -- if it's first, the
    resulting file is root-owned on the host, inside a directory
    (agency_runs/) meant to be freely removable by whoever ran the agent.
    Opening an EXISTING file for writing never changes its ownership
    (unlike creating one), so pre-creating it here as this process's own
    uid, permissive enough for the container's write, keeps it host-owned
    regardless of which side writes to it after. Safe every time this is
    called: each run gets a brand-new, never-before-existing directory, so
    there is never a stale file from a previous run at this exact path.
    """
    path = _host_daemon_log_path(sandbox)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    path.chmod(0o666)


def ensure_harness_daemon(
    sandbox: "agSandbox",
    host_uds_path: str,
    engine_name: str,
    harness: str,
    *,
    agconfig: "agconfig_cls | None" = None,
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
        ["fastapi", "uvicorn", "openai", "httpx", "mcp", "pyseccomp", "cloudpickle"],
        timeout_s=180,
    )

    _preclaim_host_daemon_log(sandbox)

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
