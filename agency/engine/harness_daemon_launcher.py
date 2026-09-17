"""Launch and discover the sandbox-side Harness Manager daemon.

The host resolves where the harness binary actually lives with a plain
host-side `shutil.which()` (see `resolve_harness_binary()` in
`harness/executable.py` -- the install directory is mounted at its original
host path, so this is where it's found), launches the daemon process
detached, then polls its `/health` endpoint until it's up or the timeout
below elapses.

Validating the binary actually starts (falling back to its own PATH lookup
for a binary the host didn't have, e.g. one baked into the sandbox image) is
the daemon's own job, done against itself once it's running (see
`HarnessManager._run_adapter_request()` in `harness/daemon.py` and
`prepare_harness_executable_local()` in `harness/executable.py`) -- a
failure there is reported back over the daemon's existing long-lived
`/harness_attempt` connection to the host, the same way any other attempt
failure is: as a `HarnessAttemptResult(ok=False, error_message=...)`.

Ensuring the daemon's required Python packages are importable can *not*
happen that same way, though: importing `agency.harness.daemon` at all
already pulls in most of them transitively (fastapi/uvicorn directly,
openai/httpx/mcp/cloudpickle via `agency`'s own package `__init__`), so a
missing one means the daemon process cannot even start far enough to run
that check -- see `_package_bootstrap_command()` below, which runs a
dependency-free (no `agency` import) check-and-pip-install script ahead of
`exec`'ing into the daemon module, inside the same detached launch.
"""

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
)
from ..harness.daemon import _REQUIRED_HARNESS_PACKAGES
from ..harness.executable import HARNESS_PATH, resolve_harness_binary
from .clients import HarnessInteractionClient

if TYPE_CHECKING:
    from ..configs.agconfig import agconfig as agconfig_cls
    from .host_servers.host_interaction_server import HostInteractionServer
    from ..sandbox.agsandbox import agSandbox

_CONTAINER_GATEWAY_DIR = AGENCY_LLM_GATEWAY_CONTAINER_MOUNT


def _daemon_log_path(engine_name: str) -> str:
    # Per-engine, not a shared "daemon.log": every agent in a run mounts the
    # same host logs/ directory at this same container path, so a single
    # shared filename would have concurrently-launching agents truncate each
    # other's daemon output.
    return f"{AGENCY_LOGS_CONTAINER_MOUNT}/daemon-{engine_name}.log"


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
            "file_access": agconfig.ptrace.file_access,
            "profiler": agconfig.ptrace.profiler,
            "disable_harness_native_sandbox": agconfig.ptrace.disable_harness_native_sandbox,
        },
        "sandbox": {
            "checkpoint_fast_resume": agconfig.sandbox.checkpoint_fast_resume,
            # Whether the daemon may pip-install a missing required package
            # into itself: only the unpinned default ("python3") interpreter
            # is allowed to self-heal that way -- see HarnessManager's own
            # bootstrap step in harness/daemon.py.
            "harness_python_path": agconfig.sandbox.harness_python_path,
        },
    }


def _package_bootstrap_command(
    harness_python: str,
    packages: "tuple[str, ...]",
    install_missing: bool,
    *,
    engine_name: str,
    container_host_uds_path: str,
) -> str:
    """A dependency-free (no `agency` import) check-and-pip-install, run
    ahead of `exec`'ing into `agency.harness.daemon` itself -- see this
    module's own docstring for why the daemon can't do this check on
    itself. Mirrors `agutil.ensure_python_packages_locally()`'s logic
    exactly; that version is for call sites already deep inside a running
    agency process (e.g. the native harness adapter), where importing
    `agency` is already a given.

    Pings the host's existing `/record_event` over `--host-uds` at each
    step (pure stdlib HTTP-over-AF_UNIX -- no `httpx`, since nothing from
    `agency` is importable yet either): both to print a live "still
    booting, here's what" line in place of the old silent wait (previously
    visible only via the daemon log's tail, and only after a timeout), and
    so `ensure_harness_daemon()`'s own wait loop can reset its deadline on
    each fresh ping -- the daemon can now genuinely take however long a
    cold pip install takes, capped by silence, not by a fixed total.
    """
    script = (
        "import http.client, importlib, json, socket, subprocess, sys\n"
        f"ENGINE = {engine_name!r}\n"
        f"HOST_UDS = {container_host_uds_path!r}\n"
        "\n"
        "class _UDSConnection(http.client.HTTPConnection):\n"
        "    def connect(self):\n"
        "        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "        self.sock.settimeout(self.timeout)\n"
        "        self.sock.connect(HOST_UDS)\n"
        "\n"
        "def ping(step, ok=True):\n"
        "    mark = '▶' if ok is None else ('✓' if ok else '✗')\n"
        "    body = json.dumps({\n"
        "        'type': 'harness_bootstrap_progress',\n"
        "        'payload': {'engine': ENGINE, 'step': step, 'ok': ok},\n"
        "        'term_message': f'[{ENGINE}] BOOT {mark}   {step}',\n"
        "        'print_to_terminal': True,\n"
        "    }).encode()\n"
        "    try:\n"
        "        conn = _UDSConnection('localhost', timeout=3)\n"
        "        conn.request('POST', '/record_event', body=body, "
        "headers={'Content-Type': 'application/json'})\n"
        "        conn.getresponse().read()\n"
        "        conn.close()\n"
        "    except Exception:\n"
        "        pass  # best-effort -- a ping failing must never abort bootstrap\n"
        "\n"
        f"packages = {packages!r}\n"
        f"install_missing = {install_missing!r}\n"
        "ping('checking required packages', ok=None)\n"
        "missing = []\n"
        "for pkg in packages:\n"
        "    try:\n"
        "        importlib.import_module(pkg)\n"
        "    except ImportError:\n"
        "        missing.append(pkg)\n"
        "if not missing:\n"
        "    ping('all required packages already present')\n"
        "    ping('starting harness daemon', ok=None)\n"
        "    raise SystemExit(0)\n"
        "if not install_missing:\n"
        "    msg = f'Pinned harness Python lacks {missing}; refusing package "
        "installation during an experiment'\n"
        "    ping(msg, ok=False)\n"
        "    sys.exit(msg)\n"
        f"ping(f'installing missing packages: {{missing}}', ok=None)\n"
        "try:\n"
        "    with socket.create_connection(('pypi.org', 443), timeout=30):\n"
        "        pass\n"
        "except OSError:\n"
        "    msg = f'cannot install {missing}: no outbound network (probed pypi.org:443)'\n"
        "    ping(msg, ok=False)\n"
        "    sys.exit(msg)\n"
        "proc = subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', "
        "*missing], timeout=180)\n"
        "if proc.returncode != 0:\n"
        "    ping(f'failed to install {missing}', ok=False)\n"
        "    sys.exit(proc.returncode)\n"
        "ping(f'installed: {missing}')\n"
        "ping('starting harness daemon', ok=None)\n"
    )
    return f"{shlex.quote(harness_python)} -c {shlex.quote(script)}"


def _is_ready(handle: DaemonHandle, timeout_s: float = 0.5) -> bool:
    try:
        with handle.client(timeout_s=timeout_s) as client:
            return client.is_ready()
    except Exception:
        return False


def _host_daemon_log_path(sandbox: "agSandbox", engine_name: str) -> Path:
    """Host-side counterpart of `_daemon_log_path()` -- must match
    agsandbox.py's own derivation of the `_agency_logs` mount source."""
    from ..utils.agutil import _DEFAULT_LOG_DIR

    db_path = sandbox.agconfig.data_logger.db_path
    log_dir = Path(db_path).parent if db_path else _DEFAULT_LOG_DIR
    return log_dir / f"daemon-{engine_name}.log"


def _preclaim_host_daemon_log(sandbox: "agSandbox", engine_name: str) -> None:
    """Touch this engine's daemon log host-side, world-writable, before the
    container's root process can create it first and leave it root-owned."""
    path = _host_daemon_log_path(sandbox, engine_name)
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
    timeout_s: float = 120.0,
    progress_source: "HostInteractionServer | None" = None,
) -> DaemonHandle:
    """Ensure one ready Harness Manager exists for this engine and sandbox.

    `progress_source`, when given, is this engine's own host-side
    `HostInteractionServer` -- the same process the bootstrap's `/record_event`
    pings land on, so no network round trip is needed to observe them. Each
    fresh ping resets the wait deadline to `timeout_s` from *now*, so a
    genuinely slow-but-progressing cold pip install isn't penalized by a
    fixed total; only actual silence (a hung or crashed bootstrap) times out.
    """
    config = agconfig if agconfig is not None else sandbox.agconfig
    handles = getattr(sandbox, "_agency_harness_daemon_handles", None)
    if handles is None:
        handles = {}
        sandbox._agency_harness_daemon_handles = handles
        # Checkpointing runs on the backend while daemon discovery runs on
        # the facade.  Share the same handle map so the backend can quiesce
        # and resume every live daemon around a container-level CRIU dump.
        sandbox._backend._agency_harness_daemon_handles = handles

    existing = handles.get(engine_name)
    if existing is not None and config.sandbox.checkpoint_fast_resume:
        # Restoring the sandbox is what makes the retained daemon's UDS
        # listener reachable again. A failed CRIU restore falls back inside
        # the backend, after which this path launches a fresh daemon below.
        sandbox._backend._ensure_started()
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
    daemon_config = _daemon_config(config if config.sandbox.checkpoint_fast_resume else agconfig)
    binary_path = resolve_harness_binary(harness, config)
    if binary_path is not None:
        daemon_config.setdefault("harness_adapter", {})["binary_path"] = binary_path
    config_json = json.dumps(daemon_config, separators=(",", ":"))
    harness_python = config.sandbox.harness_python_path or "python3"

    log_path = _daemon_log_path(engine_name)
    if config.sandbox.checkpoint_zfs_parent:
        # Keep the open log descriptor inside the snapshotted filesystem so
        # CRIU never has to restore a host-owned bind-file handle. Already
        # unique per container -- no shared-mount collision to avoid here.
        log_path = "/var/log/agency-daemon.log"
    else:
        _preclaim_host_daemon_log(sandbox, engine_name)

    bootstrap = _package_bootstrap_command(
        harness_python,
        _REQUIRED_HARNESS_PACKAGES,
        config.sandbox.harness_python_path is None,
        engine_name=engine_name,
        container_host_uds_path=handle.container_host_uds_path,
    )
    daemon_cmd = (
        f"exec {shlex.quote(harness_python)} -m agency.harness.daemon "
        f"--sandbox-uds {shlex.quote(handle.container_sandbox_uds_path)} "
        f"--host-uds {shlex.quote(handle.container_host_uds_path)} "
        f"--engine-name {shlex.quote(engine_name)} "
        f"--harness {shlex.quote(harness)} "
        f"--config-json {shlex.quote(config_json)}"
    )
    # `exec > log 2>&1` (redirecting the launch shell's own fds) rather than
    # a trailing `> log 2>&1` on one command -- the bootstrap step must land
    # in the same log a ready-timeout tails, and it runs as its own command
    # ahead of the final `exec` into the daemon module. `export` (not a bare
    # assignment) is what makes PYTHONPATH -- a variable that, unlike PATH,
    # doesn't already exist in the environment -- actually inherited by the
    # bootstrap and daemon child processes: a plain `VAR=val` before a
    # redirect-only `exec` persists it only as a shell variable, per POSIX,
    # not into the process environment new child processes read from.
    command = (
        f"export PATH={HARNESS_PATH} "
        f"PYTHONPATH={shlex.quote(AGENCY_PACKAGE_CONTAINER_MOUNT)}; "
        f"exec > {shlex.quote(log_path)} 2>&1; "
        f"{bootstrap} && {daemon_cmd}"
    )

    sandbox.exec_detached(command, workdir="/workspace")

    deadline = time.monotonic() + timeout_s
    last_seen_ping_ts = None
    while time.monotonic() < deadline:
        if _is_ready(handle):
            _register_daemon(sandbox, handle)
            handles[engine_name] = handle
            return handle
        if progress_source is not None:
            ping_ts = progress_source.last_bootstrap_ping_ts
            if ping_ts is not None and ping_ts != last_seen_ping_ts:
                last_seen_ping_ts = ping_ts
                deadline = time.monotonic() + timeout_s
        time.sleep(0.05)

    try:
        log_tail, _ = sandbox.exec(
            f"tail -c 4000 {shlex.quote(log_path)} 2>/dev/null",
            timeout=10,
        )
    except Exception as exc:
        log_tail = f"could not read daemon log: {exc}"
    raise RuntimeError(
        f"Harness Manager did not become ready at {handle.sandbox_uds_path!r} "
        f"within {timeout_s}s. Log ({log_path}):\n{log_tail}"
    )


__all__ = ["DaemonHandle", "ensure_harness_daemon"]
