"""Docker-specific container backend.

Nearly everything about running a container is identical between Docker and
Podman and lives in `._ContainerBackendBase` (`.container`), including the
session-keyring-quota machinery -- see that module's docstring for why both
runtimes are subject to the same kernel quota. `_DockerBackend` itself is
just the `_runtime` tag; see `.podman._PodmanBackend` for the one thing that
actually differs between the two (`_resolve_image`).
"""

from __future__ import annotations

from .container import _ContainerBackendBase


class _DockerBackend(_ContainerBackendBase):
    """Manages a single container for one agent via Docker."""

    _runtime = "docker"
