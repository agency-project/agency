# Podman backend (`agsandbox_backends/podman.py`)

> `_PodmanBackend` is a thin subclass of `_ContainerBackendBase` ([container.md](container.md)) — see there for the mechanics shared with Docker ([docker.md](docker.md)). This doc covers only what's genuinely Podman-specific.

Podman shares the exact same session-keyring-derived concurrency slot as Docker (see [docker.md](docker.md) and [container.md](container.md)) — rootless Podman's per-container user namespaces do **not** exempt it from the kernel session-keyring quota, since `runc` charges the session keyring against the real host UID regardless of namespace. `_acquire_runtime_slot`/`_release_runtime_slot`/`_is_quota_exhaustion_error`/`_wait_for_quota_slot`/`_quota_diagnostics` are all concrete on `_ContainerBackendBase` now; `_PodmanBackend` doesn't override any of them, the same as `_DockerBackend`.

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

## Fast squash: not yet implemented

`_PodmanBackend` doesn't override `_locate_layer_diff_dir()` or `_host_to_container_id()` — see [container.md](container.md)'s "Fast incremental squashing" section and [docker.md](docker.md)'s "Fast squash internals" for what these do on the Docker side (reaching into Docker's own undocumented overlay2 graphdriver on-disk layout, which has no verified Podman equivalent; Podman uses a different storage backend, `containers/storage`). Inheriting `_ContainerBackendBase`'s safe generic defaults (`None` / identity) is completely safe here — it just means Podman stays on the slower export/import squash path (`_squash_commit()`) at every `checkpoint_squash_interval` boundary until someone verifies Podman's own storage layout and adds the equivalent overrides. Covered by `tests/agsandbox_backends/test_podman.py`'s `TestFastSquashHooksUseBaseDefaults`.
