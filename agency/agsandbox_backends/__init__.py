"""Sandbox backend abstraction for agsandbox.

Split by concrete backend: `.base` (the abstract base class + config +
selection logic), `.container` (shared docker/podman plumbing), `.docker`,
`.podman`, `.chroot`. This package's own namespace re-exports the same public
surface the single-file `agsandbox_backend.py` module used to, so
`from agency.agsandbox_backends import X` (or `from agency import
agsandbox_backends as m; m.X`) works exactly like the old
`from agency.agsandbox_backend import X` did.
"""

from .base import (
    AgSandboxBackendFields,
    agSandboxBackendConfig,
    agsandbox_backend,
    backend_for_image_kind,
)
from .container import (
    _ContainerAlreadyRunning,
    _ContainerBackendBase,
    _RUN_ID,
    get_container_runtime,
    keyring_quota,
    reap_orphaned_containers,
    seed_cache_from_image,
)
from .docker import _DockerBackend
from .podman import _PodmanBackend
from .chroot import _ChrootBackend, chroot_available

__all__ = [
    "AgSandboxBackendFields",
    "agSandboxBackendConfig",
    "agsandbox_backend",
    "backend_for_image_kind",
    "_ContainerAlreadyRunning",
    "_ContainerBackendBase",
    "_RUN_ID",
    "get_container_runtime",
    "keyring_quota",
    "reap_orphaned_containers",
    "seed_cache_from_image",
    "_DockerBackend",
    "_PodmanBackend",
    "_ChrootBackend",
    "chroot_available",
]
