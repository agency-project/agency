# should be moved in interaction servers


"""Host-side launcher for `agmanager_harness.py`'s standalone,
container-side process.

Mirrors `agharness_backends/native.py`'s launch+bridge foundation
(`exec_detached()` inside an already-running container, the bind-mounted
`agency` package, a captured PID released from monitoring so
`agSandbox.wait_for_processes()` doesn't mistake deliberate persistence for
unfinished work) as a PATTERN, not an import -- this process needs
`fastapi`/`uvicorn`/`openai` on its own `PYTHONPATH` the way native's
deliberately stdlib-only entrypoint does not, so `agproxy_llm_in_container.py`'s
launcher (which has the same requirement) is the closer precedent.

One `agmanager_harness.py` process per agent, launched once per sandbox
container's lifetime and reused across every subsequent call on that same
sandbox -- `ensure_launched()` is the idempotent wrapper a caller should
actually use; `launch_in_container()` is the one-shot primitive underneath
it, not idempotent on its own (same split as native.py's
`launch_in_container_entrypoint`/`_ensure_entrypoint`)."""

'''
from __future__ import annotations

import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...sandbox.agsandbox import agSandbox
    from ..agmanager_host.agmanager_host import agHostAgentManager

_ENTRYPOINT_RELATIVE_PATH = "agency/manager/agmanager_harness/agmanager_harness.py"

# Fixed, well-known port inside the container -- safe because each
# container has its own network namespace (no cross-container collision
# risk a fixed HOST port would have). Same reasoning as
# agproxy_llm_in_container.py's own fixed port 8765; this uses a different
# number so the two could, in principle, run in the same container during
# a migration window without colliding.
_CONTAINER_PORT = 8766

_PID_PATH = "/tmp/.agmanager_harness.pid"
_LOG_PATH = "/tmp/.agmanager_harness.log"


def _is_reachable(sandbox: "agSandbox", port: int, timeout_s: float = 2) -> bool:
    """Probe run INSIDE the container via `sandbox.exec()`, not a direct
    host-side HTTP call -- the in-container port lives in the container's
    own network namespace, unreachable from the host the same way
    `agproxy_llm_in_container.py`'s identical-purpose helper explains. Any
    real HTTP response (even an error) proves the process is up; only a
    connection failure means it isn't."""
    script = (
        "import urllib.request as u, urllib.error as e, sys\n"
        "try:\n"
        f"    u.urlopen('http://127.0.0.1:{port}/agprof/status', timeout={timeout_s})\n"
        "except e.HTTPError:\n"
        "    sys.exit(0)\n"
        "except Exception:\n"
        "    sys.exit(1)\n"
        "else:\n"
        "    sys.exit(0)\n"
    )
    cmd = f"python3 -c {shlex.quote(script)}"
    _, rc = sandbox.exec(cmd, timeout=int(timeout_s) + 10)
    return rc == 0


def launch_in_container(
    sandbox: "agSandbox",
    host_manager: "agHostAgentManager",
    *,
    port: int = _CONTAINER_PORT,
    timeout_s: float = 30,
) -> str:
    """Start `agmanager_harness.py` as a persistent, detached process
    inside `sandbox`'s already-running container, bridged to
    `host_manager`'s two UDS listeners (main + profiler), and return the
    container-local base URL a harness launched in the SAME container
    should point its `ANTHROPIC_BASE_URL`/`model_providers.base_url`/
    `--mcp-config` at.

    Not idempotent across repeated calls on the same sandbox -- see
    `ensure_launched` for the idempotent wrapper. `host_manager` must
    already be running (its UDS listeners started) before this is called;
    this function only bridges to them, it does not start them."""
    from ...agutil import AGENCY_PACKAGE_CONTAINER_MOUNT, ensure_python_packages_in_container

    ensure_python_packages_in_container(
        sandbox, ["fastapi", "uvicorn", "openai", "httpx"], timeout_s=180
    )

    host_uds = host_manager.ensure_uds_started()
    container_host_uds = f"/var/run/agency_llm_gateway/{Path(host_uds).name}"
    profiler_uds = host_manager.ensure_profiler_uds_started()
    container_profiler_uds = f"/var/run/agency_llm_gateway/{Path(profiler_uds).name}"

    entrypoint_path = f"{AGENCY_PACKAGE_CONTAINER_MOUNT}/{_ENTRYPOINT_RELATIVE_PATH}"
    # PYTHONPATH is required here (unlike native.py's entrypoint) -- this
    # process genuinely does `import agency` (for agproxy_llm_adapters.py /
    # _native_hooks.py, see agmanager_harness.py's docstring), and
    # running a script by file path doesn't add its package's parent
    # directory to sys.path the way `-m` would.
    cmd = (
        f"echo $$ > {shlex.quote(_PID_PATH)}; "
        f"PYTHONPATH={shlex.quote(str(AGENCY_PACKAGE_CONTAINER_MOUNT))} "
        f"exec python3 {shlex.quote(entrypoint_path)} "
        f"{shlex.quote(container_host_uds)} {shlex.quote(container_profiler_uds)} {port} "
        f"> {shlex.quote(_LOG_PATH)} 2>&1"
    )
    sandbox.exec_detached(cmd, workdir="/workspace")

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _is_reachable(sandbox, port):
            break
        time.sleep(0.2)
    else:
        log_tail, _ = sandbox.exec(f"tail -c 4000 {shlex.quote(_LOG_PATH)} 2>/dev/null", timeout=10)
        raise RuntimeError(
            f"agmanager_harness never became reachable on port {port} inside the "
            f"container. Log ({_LOG_PATH}):\n{log_tail}"
        )

    try:
        pid = int(sandbox.read_file(_PID_PATH).strip())
        sandbox.release_daemon(pid)
    except Exception as e:
        print(f"[agmanager_harness] WARNING: could not release entrypoint PID from monitoring: {e}")

    return f"http://127.0.0.1:{port}"


def ensure_launched(
    sandbox: "agSandbox", host_manager: "agHostAgentManager", *, port: int = _CONTAINER_PORT
) -> str:
    """Idempotent across repeated calls on the same sandbox -- reuses an
    already-launched, still-reachable process (same "launch once per
    sandbox lifetime" pattern as `native.py`'s `_ensure_entrypoint`), and
    only launches fresh if there isn't one yet or it stopped responding
    (e.g. the container was hibernated/resumed or recreated from a
    checkpoint)."""
    existing = getattr(sandbox, "_agmanager_harness_url", None)
    if existing is not None and _is_reachable(sandbox, port):
        return existing
    base_url = launch_in_container(sandbox, host_manager, port=port)
    sandbox._agmanager_harness_url = base_url
    return base_url


def ensure_launched_locally(host_manager: "agHostAgentManager", timeout_s: float = 10) -> str:
    """The bare-host/chroot counterpart to `ensure_launched()` -- for a
    launch with no container to run `agmanager_harness.py` as a separate
    process inside. Runs the exact same `build_app()` ASGI app instead as
    an in-process uvicorn thread, bridged to `host_manager` over the SAME
    UDS paths `ensure_launched()` uses for the container case.

    This isn't a UDS-vs-TCP distinction the way it looks at first --
    `agmanager_host`'s UDS listeners aren't inherently about crossing a
    container boundary, they're just this design's one bridging mechanism;
    a Unix domain socket works identically whether or not a container is
    involved, since bare-host mode never needs the "same socket file
    visible in two filesystem namespaces" property a bind-mount buys for
    the container case. So there's no reason to invent a second, TCP-based
    `_HostBridge` variant here: the one that already exists works verbatim,
    just reached without ever leaving this host.

    Idempotent -- cached on `host_manager` itself (`host_manager.
    _local_harness_url`), not on a sandbox object, since a bare-host launch
    may have none (`ag.sandbox is None` is valid for the fully bare-host
    case, unlike chroot which still has a real sandbox object)."""
    existing = getattr(host_manager, "_local_harness_url", None)
    if existing is not None:
        return existing

    import threading

    import uvicorn

    from .agmanager_harness import build_app
    from ..clients.host_services_client import _HostBridge

    host_uds = host_manager.ensure_uds_started()
    profiler_uds = host_manager.ensure_profiler_uds_started()
    bridge = _HostBridge(host_uds, profiler_uds)
    app = build_app(bridge)

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="agmanager_harness-local")
    thread.start()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not server.started:
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError("agmanager_harness (local/bare-host mode) did not start within timeout")

    port = server.servers[0].sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    # Keep the server object reachable so it isn't garbage-collected out
    # from under its own daemon thread; no explicit stop() -- it shares
    # host_manager's own lifetime, same as the UDS listeners it bridges to.
    host_manager._local_harness_server = server
    host_manager._local_harness_url = base_url
    return base_url


__all__ = ["launch_in_container", "ensure_launched", "ensure_launched_locally"]
'''
