# Docker backend (`agsandbox_backends/docker.py`)

> `_DockerBackend` is a thin subclass of `_ContainerBackendBase` ([container.md](container.md)) — see there for the mechanics shared with Podman ([podman.md](podman.md)). This doc covers only what's genuinely Docker-specific.

Docker is the only container runtime subject to the Linux kernel session-keyring quota: each running `docker run` holds one session keyring against the user that started it, and once `/proc/sys/kernel/keys/maxkeys` is reached, the next `docker run` fails with `"unable to create session key: disk quota exceeded"`. Rootless Podman uses user namespaces with independent per-namespace keyrings and is not subject to this quota at all — see [podman.md](podman.md).

## The keyring-derived concurrency semaphore

`_container_semaphore` (a `multiprocessing.Semaphore`, not `threading.Semaphore` — so the limit is enforced across worker processes running `_ensure_started()` *and* the main process calling `stop`/`destroy`) caps the number of simultaneously running Docker containers, derived from the kernel keyring quota:

```python
def _docker_container_limit() -> int:
    maxkeys = int(Path("/proc/sys/kernel/keys/maxkeys").read_text().strip())
    return max(container_limit_floor, maxkeys - container_limit_buffer)
```

falling back to `container_limit_fallback - container_limit_buffer` if `/proc/sys/kernel/keys/maxkeys` isn't readable.

`_DockerBackend` is the only class that ever touches this semaphore, via the two hook methods `_ContainerBackendBase` defines and defaults to no-ops:

```python
def _acquire_runtime_slot(self) -> None:
    _container_semaphore.acquire()

def _release_runtime_slot(self) -> None:
    _container_semaphore.release()
```

`_ContainerBackendBase`'s shared `_ensure_started()`/`stop()`/`destroy()` call these unconditionally — for `_PodmanBackend` they're just no-ops, so the exact same shared code path costs nothing extra on that runtime.

## Diagnostics

`keyring_quota()` returns the current Linux session-keyring quota (`used`/`max`/`free`) for diagnostics, and `_semaphore_held_count()` returns `"held/limit"` for the semaphore above (via `sem_getvalue()`). Both live in `agsandbox_backends/container.py` (not this module) since `_ContainerBackendBase._run_with_conflict_retry()` — shared by both runtimes — reads them when building its final "retries exhausted" error message; Podman simply never triggers the keyring-specific retry branch that path also guards, since it never produces a "session key"/"keyring" stderr message to match against.

## Dangling image cleanup

`stop(commit=True)` captures the current image ID for the lifecycle tag *before* committing over it, then deletes that old image ID afterward (skipping deletion if another container is still running from it, e.g. a fork) — this is the shared `_ContainerBackendBase.stop()` logic (see [container.md](container.md)), not Docker-specific, but it's what keeps repeated commits to the same tag from silently piling up dangling (untagged) images on disk.
