"""Podman-specific container backend.

Podman needs no session-keyring-derived concurrency slot (see `.docker`'s
module docstring) -- rootless Podman's independent per-namespace keyrings
mean `_acquire_runtime_slot`/`_release_runtime_slot` stay the shared base
class's no-ops. The one thing Podman does need that Docker doesn't is a
fully-qualified image name for bare references.
"""

from __future__ import annotations

from .container import _ContainerBackendBase


class _PodmanBackend(_ContainerBackendBase):
    """Manages a single container for one agent via Podman."""

    _runtime = "podman"

    def _resolve_image(self, name: str) -> str:
        """Prefix bare image names with ``localhost/``.

        Podman requires fully-qualified names when no unqualified-search
        registries are configured in /etc/containers/registries.conf.
        Docker accepts bare names fine, so this override is Podman-only.
        """
        if "/" not in name:
            return f"localhost/{name}"
        return name
