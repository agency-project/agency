"""Podman-specific container backend.

Podman shares the session-keyring-derived concurrency slot and quota
handling (`_acquire_runtime_slot`/`_release_runtime_slot`/
`_is_quota_exhaustion_error`/`_wait_for_quota_slot`/`_quota_diagnostics`)
with Docker -- see `.container`'s module docstring for why rootless Podman
(via runc) is subject to the exact same kernel quota as Docker, despite its
per-container user namespaces. Nothing here overrides any of them. The one
thing Podman does need that Docker doesn't is a fully-qualified image name
for bare references.
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
