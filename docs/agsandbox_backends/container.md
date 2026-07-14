# Shared container backend (`agsandbox_backends/container.py`)

> Covers `_ContainerBackendBase`, the shared implementation `_DockerBackend` ([docker.md](docker.md)) and `_PodmanBackend` ([podman.md](podman.md)) both subclass. See [base.md](base.md) for backend selection and [chroot.md](chroot.md) for the non-container alternative.

`_ContainerBackendBase` holds everything that doesn't differ between Docker and Podman — nearly everything. The only two things a leaf class overrides:

- **`_resolve_image(name)`** — identity for Docker (bare image names resolve fine); `_PodmanBackend` prefixes `localhost/` (Podman requires fully-qualified names when no unqualified-search registries are configured).
- **`_acquire_runtime_slot()` / `_release_runtime_slot()`** — no-ops by default; `_DockerBackend` overrides both to acquire/release the session-keyring-derived concurrency semaphore described in [docker.md](docker.md). Podman is exempt (independent per-namespace keyrings), so it never touches this at all.

Containers are created lazily: one starts only when a task actually calls a tool with `run_in_subprocess=True`. Tasks that complete using only host-side tools (web fetch, `ask_human`, paper search, …) never create a container at all. After each successful sandbox tool call, the container state is committed to a lifecycle image and the container is removed — the session keyring (Docker only) and GPU are freed so other agents can use them while the LLM thinks. On the next tool call the container is recreated from the lifecycle image, restoring `/workspace` and all other state.

## Building the sandbox image

```bash
GPU_TYPE=cpu ./images/build.sh   # or nvidia/rocm; auto-detected if omitted
```

`build.sh` builds `agency-sandbox:latest` for **every container runtime installed on the host, podman first** — matching `agsandbox_backends.base._auto_detect_runtime()`'s auto-selection order (see [base.md](base.md)). An image built only for docker would leave podman's separate image store empty, so podman would try (and fail) to pull the image from a registry instead of finding it locally; building for both keeps whichever one auto-selection picks actually usable. Podman's build is tagged `localhost/agency-sandbox:latest` (the `localhost/` prefix it requires to resolve an unqualified name to its own local store instead of a registry, matching `_PodmanBackend._resolve_image()` — see [podman.md](podman.md)); docker's is tagged bare. Each build gets its own smoke test (`import torch`) run against the runtime that built it.

If `HF_TOKEN` is set, it's passed as a `--secret` for gated model downloads during the build. This is written to a private temp file and passed as `--secret id=hf_token,src=<file>` for both runtimes — docker buildx's `--secret id=...,env=VAR` shorthand is **not** portable to podman's buildah, which rejects it ("incorrect secret flag format: should be `--secret id=foo,src=bar`"); the file-based form is the one both accept.

The `images/Dockerfile` installs `ripgrep` on top of `python:3.12-slim`. `Dockerfile.nvidia`/`Dockerfile.rocm` are used instead when `GPU_TYPE` is `nvidia`/`rocm`.

## Container lifecycle

The container exists only during active tool execution. Between tool calls the container is removed, releasing the Linux session keyring (Docker only) and any held GPU so concurrent agents can use those resources.

| Event | What happens |
|---|---|
| `agSandbox.__init__` | No container created — cheap object; `_lifecycle_image=None` |
| Any `run_in_subprocess=True` tool call | `_ensure_started()` runs lazily: if container is already running, reuse it; otherwise `docker rm -f`/`podman rm -f` any leftover zombie, then `run` from `_lifecycle_image` (or `base_image` on first use) |
| After **successful** sandbox tool call | `sandbox.stop(commit=True)`: `commit → agency/lifecycle-<name>`; `rm -f` (retried up to 3×); `_lifecycle_image` updated |
| After **failed** sandbox tool call | `sandbox.stop(commit=False)`: `rm -f` without commit; dirty state discarded; next start restores from previous `_lifecycle_image` |
| Any non-running container detected at startup | Force-removed with `rm -f` before `run` — covers "Exited", "Created" (partial run), and "Dead" states |
| `sandbox.destroy()` | `rm -f` (no-op if already removed); `rmi agency/lifecycle-<name>`; any `pretool-*` images cleaned up |
| `atexit` | All live containers removed (guard against hard-killed processes) |

**Failure revert**: when a tool errors, `stop(commit=False)` discards the container with its partial state. The next tool call recreates from the last successful `_lifecycle_image`, so the agent's workspace is automatically rolled back to the last known-good state. The agent receives `workspace_reverted` in the error response to know this happened.

**Container naming**: each container is named `sandbox-{RUN_ID}-{agname}`, where `_RUN_ID` (`agsandbox_backends.container._RUN_ID`) is a per-process UUID prefix. This prevents cross-run name collisions when an agent crashes without cleanup and is restarted with the same `agname`. The lifecycle image name is produced by `_lifecycle_tag()`, which lowercases the image name — both runtimes require all repository names to be lowercase.

**`stop()` reliability**: `rm -f` is retried up to 3 times. Each attempt goes through `_run()`, which holds `_get_docker_semaphore()`'s semaphore (caps all concurrent daemon calls, both runtimes, at 16 — see `docker_semaphore_limit`). If all retries fail, a `WARNING` is emitted to stderr and the framework continues — `_started` is cleared regardless so the next tool call can attempt a fresh container.

## Orphaned container reaping

`atexit` handlers (the row above) never run if the execution process is killed with `SIGKILL` — that signal can't be caught by any process, so the interpreter never regains control to remove its live containers. A framework-level `SIGTERM` handler (`agutil.sigterm_as_exit()`, used by `agwebui.run()`/`graphui.run()` — see [agwebui.md](../agwebui.md#shutdown-and-signal-handling)) closes that gap for plain `kill`, but SIGKILL still leaves containers behind. `_RUN_ID` being a per-process random UUID (see "Container naming" above) means the *next* run can't just guess an old container's name to reclaim it — it's deliberately randomized to prevent name collisions across runs, not to enable this.

Instead, every container is labelled at `run` time with the PID of the process that owns it:

```
--label agency.owner_pid=<pid>
```

**`self._owner_pid` is captured once, in `_ContainerBackendBase.__init__`**, as `os.getpid()` at construction time — not read live inside `_ensure_started()`. This matters because a tool call with the default `run_in_subprocess=True` dispatches through a `ProcessPoolExecutor` (`agtool.py`), so `_ensure_started()` (and the `docker run` call it makes) can execute inside a **worker** process with its own transient PID, distinct from the main process that actually owns the sandbox's lifecycle. Labelling with a live `os.getpid()` call there would tag containers with a worker PID that exits as soon as the tool call returns, making every container look orphaned to the next reap almost immediately. Capturing the PID once at construction — always in the main process — avoids that.

**At startup**, `_ContainerBackendBase.__init__` calls `reap_orphaned_containers()` (also exported from `agsandbox_backends`) before anything else. It:

1. Lists all containers (any runtime) carrying the `agency.owner_pid` label, via `docker ps -a --filter label=agency.owner_pid --format ...`.
2. For each, checks whether the labelled PID is (a) this process's own PID (skip — it's a container this same run is about to reuse) or (b) still alive on the host via `_pid_alive()` (`os.kill(pid, 0)`; skip if alive).
3. Anything left — labelled with a PID that's neither this process nor alive — is a dead run's container. It's force-removed (`rm -f`) along with its lifecycle image tag (`rmi -f agency/lifecycle-<name>`), and a message is printed noting the reap and the dead owner PID.

Runs only once per process (`_reap_done` flag guarded by a lock) — every subsequent `agSandbox`/backend construction in the same process is a no-op. Any exception during the reap (no container runtime available, daemon unreachable, etc.) is caught and logged as a `WARNING`; it never blocks startup.

This is best-effort cleanup, not a substitute for `sigterm_as_exit()` — it only reclaims resources at the *next* run's startup, and only if something runs `_ContainerBackendBase.__init__` again on the same host afterward.

## GPU device access

`--gpus all` (NVIDIA) or `--device /dev/kfd --device /dev/dri` (AMD/ROCm) is passed to `run` when a GPU is detected on the host (`agsandbox_backends.container._gpu_flags()`, cached process-wide via `agresources.detect_gpus()`). On CPU-only hosts no flag is passed.

Even with those flags, GPUs are **not accessible by default** — every `exec()` call unconditionally exports `CUDA_VISIBLE_DEVICES=""` when no virtual reservation is active, making all GPUs invisible to CUDA. Calling `reserve_gpu` sets only a virtual flag; no physical GPU is taken. When `exec()` runs a bash command and the virtual flag is set, a physical GPU is claimed from the pool at that moment (blocking until one is free) and `CUDA_VISIBLE_DEVICES=<id>` (and `HIP_VISIBLE_DEVICES=<id>` for ROCm) is injected. After a foreground exec with no background processes, the physical GPU is returned to the pool immediately — freeing it for other agents while the LLM thinks. When background processes are alive, the GPU is held until `get_live_pids()` finds them all finished.

## Forking

```
parent.sandbox._checkpoint_image ──tag_image──▶ agency/lifecycle-<fork-agname>
                                            │
                                     (consumed by fork's first _task())
```

Forking copies the parent's checkpoint image tag to a new tag for the fork via `type(parent.sandbox._backend).tag_image(...)` — dispatched to whichever backend class actually produced the checkpoint (`docker tag`/`podman tag` here, a directory copy for chroot — see [chroot.md](chroot.md)), not a bare docker-only call, since a chroot snapshot directory and a docker/podman image tag are unrelated formats. No container is created at fork time — that happens lazily when the fork's first `_task()` runs, restoring from the copied tag.

Because forks wait for `src.ctx.resolve_prev_dependencies()` before construction, the parent's task is always complete before the fork is built, so the checkpoint image is already the committed post-task state.

## Static image-level helpers

`tag_image`/`delete_image`/`export_image`/`import_image` are `@staticmethod`s on `_ContainerBackendBase` that all call `get_container_runtime()` internally to pick the live runtime — they're runtime-agnostic by construction, so both `_DockerBackend` and `_PodmanBackend` share the exact same implementation rather than each needing their own.
