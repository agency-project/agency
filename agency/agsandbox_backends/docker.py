"""Docker-specific container backend.

Docker is the only container runtime subject to the Linux kernel session-
keyring quota: each running `docker run` holds one session keyring against
the user that started it, and once `/proc/sys/kernel/keys/maxkeys` is
reached, the next `docker run` fails with "unable to create session key:
disk quota exceeded". Rootless Podman uses user namespaces with independent
per-namespace keyrings and is not subject to this quota at all -- see
`.podman`, which never touches `_container_semaphore`.

`multiprocessing.Semaphore` (rather than `threading.Semaphore`) is used here
so the limit is enforced across all worker processes (which run
`_ensure_started`) and the main process (which calls `stop`/`destroy`), not
just threads within one process.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

from .base import AgSandboxBackendFields
from .container import _ContainerBackendBase, _docker_container_limit


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
