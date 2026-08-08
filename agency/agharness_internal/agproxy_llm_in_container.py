"""Host-side launcher for `agproxy_llm.py` running as a persistent,
in-container process (Phase 2b-ii -- see docs/Design_harness_integration.md
and agharness_internal/_agproxy_llm_in_container_entrypoint.py).

Reuses the exact launch+bridge primitives already built for
`agharness_backends/native.py` (`exec_detached`, the `agency` package
bind-mount, `agutil.ensure_python_packages_in_container`) rather than a
second copy of that plumbing -- this module is genuinely thin: pick a
fixed port, ensure dependencies, launch, wait for it to answer.

Readiness is checked by running a probe INSIDE the container via
`sandbox.exec()`, not a direct HTTP call from the host process -- the
in-container port lives in the container's own network namespace, which
the host cannot reach via plain `127.0.0.1` (no port is published; the
container is already running, created before this launch, so a `-p` flag
can't be retrofitted onto it). Only the harness binary launched in the
SAME container ever needs to reach this port directly, and it already
shares that namespace.
"""

from __future__ import annotations

import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agsandbox import agSandbox

# Fixed, well-known port inside the container -- safe precisely because
# each container has its own network namespace (no cross-container
# collision risk the way a fixed HOST port would have). This process is
# launched detached (fire-and-forget), so there's no stdout to read a
# dynamically-assigned port back from.
_AGPROXY_LLM_IN_CONTAINER_PORT = 8765

_ENTRYPOINT_RELATIVE_PATH = (
    "agency/agharness_internal/_agproxy_llm_in_container_entrypoint.py"
)


def _base_url() -> str:
    return f"http://127.0.0.1:{_AGPROXY_LLM_IN_CONTAINER_PORT}"


def _is_reachable(sandbox: "agSandbox", timeout_s: float = 2) -> bool:
    """Stdlib-only probe run INSIDE the container (see module docstring
    for why this can't be a direct host-side HTTP call). Any real HTTP
    response -- even a 4xx for an unrecognized token -- proves the server
    is up; only a connection failure means it isn't. `urllib.request`
    raises `HTTPError` (a real response) separately from `URLError`
    (connection refused/timeout/etc.), so the two are distinguished
    explicitly rather than treating any exception as "not up"."""
    script = (
        "import urllib.request as u, urllib.error as e, json, sys\n"
        "try:\n"
        f"    u.urlopen(u.Request({_base_url()!r} + '/v1/messages/count_tokens', "
        "data=json.dumps({'messages': []}).encode(), method='POST'), "
        f"timeout={timeout_s})\n"
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


def ensure_agproxy_llm_in_container(
    sandbox: "agSandbox", agconfig: "agConfig | None", timeout_s: float = 60
) -> str:
    """Idempotent: launch agproxy_llm.py's FastAPI app as a persistent,
    detached process inside `sandbox`'s already-running container (if not
    already running), and return the local base_url a harness binary
    launched in the SAME container should use as its own
    `ANTHROPIC_BASE_URL`/`AGPOLICY_BASE_URL`/equivalent.

    The harness's own bearer token must already be registered on
    `agllm_terminus.get_shared_terminus(agconfig)` (not on anything
    returned here) before the harness process starts -- this in-container
    instance has no registry of its own; every request it handles resolves
    the agent purely via the terminus, over the bridge established here.
    """
    base_url = _base_url()
    if _is_reachable(sandbox):
        return base_url

    from ..agutil import (
        AGENCY_PACKAGE_CONTAINER_MOUNT,
        ensure_python_packages_in_container,
    )
    from .agllm_terminus import get_shared_terminus

    ensure_python_packages_in_container(sandbox, ["fastapi", "uvicorn", "openai"], timeout_s=180)

    terminus = get_shared_terminus(agconfig)
    terminus_host_uds = terminus.ensure_uds_started()
    terminus_container_uds = f"/var/run/agency_llm_gateway/{Path(terminus_host_uds).name}"

    entrypoint_path = f"{AGENCY_PACKAGE_CONTAINER_MOUNT}/{_ENTRYPOINT_RELATIVE_PATH}"
    # PYTHONPATH is required here (unlike native.py's entrypoint, which is
    # deliberately stdlib-only) -- this entrypoint genuinely does `import
    # agency`, and running a script by file path doesn't add its package's
    # parent directory to sys.path the way `-m` would. Confirmed as a real
    # bug during development: without this, the process fails immediately
    # with `ModuleNotFoundError: No module named 'agency'` and
    # exec_detached's fire-and-forget nature means that failure is
    # otherwise silent -- the caller only sees the readiness poll below
    # time out with no indication why.
    # stdout/stderr redirected to a log file inside the container --
    # exec_detached is fire-and-forget, so a crash on startup (the
    # PYTHONPATH bug above was caught exactly this way during development)
    # would otherwise be silent: the caller would only see the readiness
    # poll below time out, with no indication why.
    log_path = "/tmp/.agproxy_llm_in_container.log"
    cmd = (
        f"PYTHONPATH={shlex.quote(str(AGENCY_PACKAGE_CONTAINER_MOUNT))} "
        f"python3 {shlex.quote(entrypoint_path)} "
        f"{shlex.quote(terminus_container_uds)} {_AGPROXY_LLM_IN_CONTAINER_PORT} "
        f"> {shlex.quote(log_path)} 2>&1"
    )
    sandbox.exec_detached(cmd, workdir="/workspace")

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _is_reachable(sandbox):
            return base_url
        time.sleep(0.2)
    log_tail, _ = sandbox.exec(f"tail -c 4000 {shlex.quote(log_path)} 2>/dev/null", timeout=10)
    raise RuntimeError(
        f"in-container agproxy_llm never became reachable at {base_url}. "
        f"Log ({log_path}):\n{log_tail}"
    )


__all__ = ["ensure_agproxy_llm_in_container"]
