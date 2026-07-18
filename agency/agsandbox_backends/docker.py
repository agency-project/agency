"""Docker-specific container backend.

Docker is the only container runtime subject to the Linux kernel session-
keyring quota: each running `docker run` holds one session keyring against
the user that started it, and once `/proc/sys/kernel/keys/maxkeys` is
reached, the next `docker run` fails with "unable to create session key:
disk quota exceeded". Rootless Podman uses user namespaces with independent
per-namespace keyrings and is not subject to this quota at all -- see
`.podman`, which never touches `_container_semaphore` and inherits
`_ContainerBackendBase`'s quota hooks (`_is_quota_exhaustion_error()`/
`_wait_for_quota_slot()`/`_quota_diagnostics()`) as no-ops.

`multiprocessing.Semaphore` (rather than `threading.Semaphore`) is used here
so the limit is enforced across all worker processes (which run
`_ensure_started`) and the main process (which calls `stop`/`destroy`), not
just threads within one process.
"""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

from .base import AgSandboxBackendFields
from .container import _ContainerBackendBase


# Hard cap on the number of simultaneously running Docker containers, derived
# from the Linux kernel session-keyring quota (see this module's docstring for
# why only Docker needs this).  multiprocessing.Semaphore is backed by a POSIX
# IPC semaphore so the limit is enforced across all worker processes (which
# run _ensure_started) and the main process (which calls stop/destroy).
def _docker_container_limit() -> int:
    """Return the concurrent-Docker-container cap derived from the kernel keyring quota."""
    _fields = AgSandboxBackendFields()
    try:
        maxkeys = int(Path("/proc/sys/kernel/keys/maxkeys").read_text().strip())
        return max(_fields.container_limit_floor, maxkeys - _fields.container_limit_buffer)
    except OSError:
        return _fields.container_limit_fallback - _fields.container_limit_buffer


def keyring_quota() -> dict[str, int]:
    """Return the current Linux session-keyring quota for diagnostics.

    Returns a dict with ``used``, ``max``, and ``free`` key counts.
    ``used`` is -1 when /proc/keys is not readable (non-root on some kernels).
    """
    try:
        maxkeys = int(Path("/proc/sys/kernel/keys/maxkeys").read_text().strip())
    except OSError:
        maxkeys = -1
    try:
        used = sum(1 for ln in Path("/proc/keys").read_text().splitlines() if ln.strip())
    except OSError:
        used = -1
    free = (maxkeys - used) if (maxkeys >= 0 and used >= 0) else -1
    return {"used": used, "max": maxkeys, "free": free}


def _semaphore_held_count() -> str:
    """Return 'held/limit' for the Docker container-concurrency semaphore, or
    '?/limit' if unreadable.

    Uses sem_getvalue() via the internal _semlock on POSIX (Linux).  The count
    reflects this process's view only — other unrelated processes are not
    tracked by our semaphore but do consume system keyring slots, so comparing
    this number with keyring_quota()['used'] reveals how many slots belong to
    external processes.
    """
    limit = _docker_container_limit()
    try:
        available = _container_semaphore._semlock._get_value()
        held = limit - available
    except Exception:
        held = "?"
    return f"{held}/{limit}"


_container_semaphore: "multiprocessing.Semaphore" = multiprocessing.Semaphore(
    _docker_container_limit()
)


class _DockerBackend(_ContainerBackendBase):
    """Manages a single container for one agent via Docker."""

    _runtime = "docker"

    def _acquire_runtime_slot(self) -> None:
        _container_semaphore.acquire()

    def _release_runtime_slot(self) -> None:
        _container_semaphore.release()

    def _is_quota_exhaustion_error(self, stderr: str) -> bool:
        """Recognize the Linux session-keyring quota exhaustion message."""
        return "session key" in stderr or ("disk quota exceeded" in stderr and "keyring" in stderr)

    def _wait_for_quota_slot(self) -> None:
        """Poll the actual keyring free count from /proc until a slot opens
        up (or give up after keyring_wait_timeout_s).

        The runtime-slot semaphore (_container_semaphore) prevents our own
        containers from exceeding the limit, but external processes can
        consume slots outside our accounting -- polling /proc catches that
        case too.
        """
        deadline = time.monotonic() + self.keyring_wait_timeout_s
        while time.monotonic() < deadline:
            if keyring_quota().get("free", 0) > 0:
                return
            time.sleep(self.keyring_poll_interval_s)

    def _quota_diagnostics(self) -> str:
        quota = keyring_quota()
        return (
            f"[keyring: {quota['used']}/{quota['max']} used, "
            f"framework semaphore: {_semaphore_held_count()} held]"
        )
