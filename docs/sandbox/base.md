# Sandbox backend selection (`sandbox/base.py`)

> This doc covers backend *selection* and the abstract base class. For a specific backend's own mechanics see [container.md](container.md) (shared docker/podman plumbing), [docker.md](docker.md), [podman.md](podman.md), or [chroot.md](chroot.md). For the `agSandbox` facade that sits in front of all of them, see [../agsandbox.md](../agsandbox.md).

`agSandbox` (in `sandbox/agsandbox.py`) is a thin facade: it resolves the image/mounts vocabulary that's meaningful regardless of backend, then builds and delegates every operation to an `agsandbox_backend` (in `sandbox/base.py`) chosen by `agSandboxBackendConfig.backend`. Three concrete backends exist today, each a real subclass in its own module:

- **`_DockerBackend`** ([docker.md](docker.md)) — a Docker container.
- **`_PodmanBackend`** ([podman.md](podman.md)) — a Podman container.
- **`_ChrootBackend`** ([chroot.md](chroot.md)) — a per-agent directory chrooted into via an unprivileged user+mount namespace, needing no root/sudo/setcap.

`_DockerBackend` and `_PodmanBackend` both subclass `_ContainerBackendBase` ([container.md](container.md)), which holds everything that doesn't differ between the two runtimes — nearly everything. Each leaf class only overrides the handful of things that do.

## Backend selection

`agSandboxBackendConfig.backend` picks the backend: `"auto"` (default), `"podman"`, `"docker"`, or `"chroot"`.

```python
from agency.agconfig import agConfig
from agency.sandbox import agSandboxBackendConfig

cfg = agConfig(agSandboxBackendConfig(backend="chroot"))
sb = agSandbox("myagent", agconfig=cfg)
```

`"auto"` (`sandbox.base._auto_detect_runtime()`) probes in this order: **podman → docker → chroot**, first usable one wins.

- Podman/docker usability: the binary is on `PATH` and `<runtime> info` succeeds (`sandbox.container._runtime_works()`), cached for the process lifetime in `_RUNTIME`.
- Chroot usability (`sandbox.chroot.chroot_available()`, also cached): see [chroot.md](chroot.md) for the two-layered probe (sysctl + live `unshare`, with a `rootlesskit`-wrapped fallback).

An explicit `backend="docker"|"podman"|"chroot"` raises immediately with a clear error if that specific backend isn't usable, rather than silently falling through to another one.

`for_config()` dispatches to the concrete backend's module via a **lazy, function-local import** (`from .docker import _DockerBackend`, etc.) rather than a module-level one — `.docker`/`.podman`/`.chroot` all import `agsandbox_backend` *from* `base.py` to subclass it, so a module-level import the other way would be circular.

## `IMAGE_KIND` and cross-backend checkpoint routing

Every concrete backend declares an `IMAGE_KIND` class attribute identifying the checkpoint/snapshot format its `tag_image`/`export_image`/`import_image`/`delete_image` static methods produce and consume — a chroot snapshot directory and a docker/podman image tag are unrelated formats, so a checkpoint produced by one backend kind is meaningless to another. `_DockerBackend` and `_PodmanBackend` both inherit `IMAGE_KIND = "container"` from `_ContainerBackendBase` (their checkpoint format — a docker/podman image tag — is identical either way, both just auto-detecting the live runtime via `get_container_runtime()`); `_ChrootBackend` uses `IMAGE_KIND = "chroot"`.

`agent.py`'s `save()` records this alongside a checkpoint (`state["sandbox_image_kind"]`) and `load()` uses `backend_for_image_kind(kind)` to route to the matching backend class rather than assuming the container backend unconditionally — see `agSandbox.backend_for_image_kind()` in [../agsandbox.md](../agsandbox.md).

## Static image-level helpers

`agsandbox_backend.tag_image()`/`delete_image()`/`export_image()`/`import_image()` are exposed as static methods on the *base* class (forwarding to `_ContainerBackendBase`'s own versions, via the same lazy-import pattern as `for_config()`) so `agSandbox`'s facade-level static forwarders (used by `agent.py`'s `save()`/`load()`, which have no live backend instance to call through) have something to call without needing to know which concrete backend produced a checkpoint ahead of time — callers that *do* know should prefer `backend_for_image_kind(kind)` instead, exactly as `agent.py` does.

## Shared method implementations

Physical readiness has one backend-neutral facade operation:
`agSandbox.ensure_started()`. `SandboxProvisioner.acquire()` calls it after
acquiring `sandbox._lock` and before host services start. The facade delegates
to the selected backend: Docker and Podman share the container
start/resume/create mechanics in `_ContainerBackendBase`, while chroot
materializes or reuses its workspace. Backend operations may repeat the same
check idempotently for defensive direct use.

`exec()` (GPU env injection + background-PID tracking via the `__BGPIDS__` marker — see [../agsandbox.md](../agsandbox.md)'s "exec wrapper" section), `read_file()`/`read_file_bytes()`/`write_file()`/`write_file_bytes()`, `get_live_pids()`/`pid_status_summary()`/`release_daemon()`, and `release_resources()` are all implemented once, purely in terms of each concrete backend's own `_container_exec()` primitive — so every backend gets them for free rather than reimplementing the same base64/proc-diffing logic three times.
