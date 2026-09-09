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
from ..harness.executable import HARNESS_PATH, prepare_harness_executable
from .clients import HarnessInteractionClient

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

    def client(self, timeout_s: "float | None" = 300.0) -> HarnessInteractionClient:
        return HarnessInteractionClient(self.sandbox_uds_path, timeout_s=timeout_s)


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
    """Host-side counterpart of `_DAEMON_LOG_PATH` -- must match
    agsandbox.py's own derivation of the `_agency_logs` mount source."""
    from ..utils.agutil import _DEFAULT_LOG_DIR

    db_path = sandbox.agconfig.data_logger.db_path
    log_dir = Path(db_path).parent if db_path else _DEFAULT_LOG_DIR
    return log_dir / "daemon.log"


def _preclaim_host_daemon_log(sandbox: "agSandbox") -> None:
    """Touch daemon.log host-side, world-writable, before the container's
    root process can create it first and leave it root-owned."""
    path = _host_daemon_log_path(sandbox)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    path.chmod(0o666)


def _register_daemon(sandbox, handle):
    # The daemon is infrastructure, not a user background job. Explicitly
    # register its PID so a liveness refresh can still discover user children.
    if not hasattr(sandbox, "_register_harness_pid"):
        return  # Minimal direct-call sandbox adapters may not track processes.
    try:
        with handle.client(timeout_s=1) as client:
            identity = client.daemon_identity()
        if identity is not None:
            sandbox._register_harness_pid(*identity)
        else:
            print(
                "[engine] WARNING: harness daemon did not report its PID; hibernation may be deferred"
            )
    except Exception as exc:
        print(
            f"[engine] WARNING: harness daemon PID registration unavailable: {type(exc).__name__}"
        )


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
        _register_daemon(sandbox, existing)
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
    daemon_config = _daemon_config(agconfig)
    config = agconfig if agconfig is not None else sandbox.agconfig
    binary_path = prepare_harness_executable(sandbox, harness, config)
    if binary_path is not None:
        daemon_config.setdefault("harness_adapter", {})["binary_path"] = binary_path
    config_json = json.dumps(daemon_config, separators=(",", ":"))

    ensure_python_packages_in_container(
        sandbox,
        ["fastapi", "uvicorn", "openai", "httpx", "mcp", "pyseccomp", "cloudpickle", "pyte"],
        timeout_s=180,
    )

    _preclaim_host_daemon_log(sandbox)

    # Actual launch of the daemon
    command = (
        f"PATH={HARNESS_PATH} "
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
            _register_daemon(sandbox, handle)
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
