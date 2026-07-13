# Podman backend (`agsandbox_backends/podman.py`)

> `_PodmanBackend` is a thin subclass of `_ContainerBackendBase` ([container.md](container.md)) — see there for the mechanics shared with Docker ([docker.md](docker.md)). This doc covers only what's genuinely Podman-specific.

Podman needs no session-keyring-derived concurrency slot (see [docker.md](docker.md)) — rootless Podman's independent per-namespace keyrings mean `_acquire_runtime_slot`/`_release_runtime_slot` stay `_ContainerBackendBase`'s shared no-ops; `_PodmanBackend` doesn't override either.

## Image name resolution

Podman requires fully-qualified image names when no unqualified-search registries are configured in `/etc/containers/registries.conf` — an unqualified `docker run agency-sandbox:latest`-equivalent would try (and fail) to resolve `agency-sandbox` against a remote registry. Docker accepts bare names fine. `_PodmanBackend` overrides `_resolve_image()` accordingly:

```python
def _resolve_image(self, name: str) -> str:
    if "/" not in name:
        return f"localhost/{name}"
    return name
```

This is why `images/build.sh` tags Podman's build `localhost/agency-sandbox:latest` rather than bare `agency-sandbox:latest` — see [container.md](container.md)'s "Building the sandbox image" section.

## Auto-selection priority

`agsandbox_backends.base._auto_detect_runtime()`'s `"auto"` backend selection prefers **podman over docker** when both are usable (see [base.md](base.md)) — this is why `images/build.sh` builds for both runtimes, podman first, so a host with both installed doesn't end up auto-selecting a runtime with no local image built for it.
