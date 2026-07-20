# Shared container backend (`agsandbox_backends/container.py`)

> Covers `_ContainerBackendBase`, the shared implementation `_DockerBackend` ([docker.md](docker.md)) and `_PodmanBackend` ([podman.md](podman.md)) both subclass. See [base.md](base.md) for backend selection and [chroot.md](chroot.md) for the non-container alternative.

`_ContainerBackendBase` holds everything that doesn't differ between Docker and Podman — nearly everything, including the session-keyring-quota handling. The only thing a leaf class overrides:

- **`_resolve_image(name)`** — identity for Docker (bare image names resolve fine); `_PodmanBackend` prefixes `localhost/` (Podman requires fully-qualified names when no unqualified-search registries are configured).

Session-keyring-derived concurrency (`_acquire_runtime_slot()`/`_release_runtime_slot()`/`_is_quota_exhaustion_error()`/`_wait_for_quota_slot()`/`_quota_diagnostics()`) is **not** one of the things that differs: both Docker and rootless Podman (via `runc`) charge each running container's session keyring against the real host UID's kernel quota (`/proc/sys/kernel/keys/maxkeys`) identically — Podman's per-container user namespaces don't exempt it, since `runc` joins/creates the session keyring before the container process finishes transitioning into its remapped identity. (Confirmed empirically — `/proc/keys` gains a `_ses.*` entry owned by the real host UID across a plain `podman run`/`rm` cycle, same as docker — and upstream: containers/podman#13363, kubernetes-sigs/kind#3806.) Both `_DockerBackend` and `_PodmanBackend` therefore use the same concrete implementations on `_ContainerBackendBase`, described in [docker.md](docker.md)/[podman.md](podman.md).

Containers are created lazily: one starts only when a task actually calls `exec()` (any sandboxed tool touching the sandbox, not just ones flagged `run_in_subprocess=True` — that flag no longer gates whether the container gets stopped/checkpointed; see "Container lifecycle" below). Tasks that complete using only host-side tools (web fetch, `ask_human`, paper search, …) never create a container at all. After each sandbox tool call — success or failure — the container is stopped: committed to a lifecycle image and removed on success, discarded without committing on failure. The session keyring and GPU are freed at that point so other agents can use them while the LLM thinks. On the next tool call the container is recreated from the lifecycle image, restoring `/workspace` and all other state.

## Building the sandbox image

```bash
GPU_TYPE=cpu ./images/build.sh   # or nvidia/rocm; auto-detected if omitted
```

`build.sh` builds `agency-sandbox:latest` for **every container runtime installed on the host, podman first** — matching `agsandbox_backends.base._auto_detect_runtime()`'s auto-selection order (see [base.md](base.md)). An image built only for docker would leave podman's separate image store empty, so podman would try (and fail) to pull the image from a registry instead of finding it locally; building for both keeps whichever one auto-selection picks actually usable. Podman's build is tagged `localhost/agency-sandbox:latest` (the `localhost/` prefix it requires to resolve an unqualified name to its own local store instead of a registry, matching `_PodmanBackend._resolve_image()` — see [podman.md](podman.md)); docker's is tagged bare. Each build gets its own smoke test (`import torch`) run against the runtime that built it.

If `HF_TOKEN` is set, it's passed as a `--secret` for gated model downloads during the build. This is written to a private temp file and passed as `--secret id=hf_token,src=<file>` for both runtimes — docker buildx's `--secret id=...,env=VAR` shorthand is **not** portable to podman's buildah, which rejects it ("incorrect secret flag format: should be `--secret id=foo,src=bar`"); the file-based form is the one both accept.

The `images/Dockerfile` installs `ripgrep` on top of `python:3.12-slim`. `Dockerfile.nvidia`/`Dockerfile.rocm` are used instead when `GPU_TYPE` is `nvidia`/`rocm`.

## Container lifecycle

The container exists only during active tool execution. Between tool calls the container is removed, releasing the Linux session keyring (both runtimes) and any held GPU so concurrent agents can use those resources.

| Event | What happens |
|---|---|
| `agSandbox.__init__` | No container created — cheap object; `_lifecycle_image=None` |
| Any tool call that touches the sandbox | `_ensure_started()` runs lazily: ground truth is always `_container_running()` (a real `docker/podman inspect`), never a per-process flag — see "Cross-process safety" below. If already running, reuse it; otherwise `docker rm -f`/`podman rm -f` any leftover zombie, then `run` from `_lifecycle_image` (or `base_image` on first use) |
| After a tool call, unless background work is still pending | `agtool.py`'s `dispatch_tools()` calls `sandbox.stop(commit=...)` after every tool call — success or failure — regardless of `run_in_subprocess`, **unless** `sandbox._has_pending_background_work()` is true, in which case `stop()` is skipped entirely for that call rather than killing a job the agent just backgrounded. A later tool call that finds nothing pending is what actually checkpoints/tears down. |
| `sandbox.stop(commit=True)` (successful tool call) | `commit → agency/lifecycle-<name>` (retried up to 3×); `rm -f` (retried up to 3×); `_lifecycle_image` updated; GPU released once the container is confirmed gone |
| `sandbox.stop(commit=False)` (failed tool call) | `rm -f` without commit (retried up to 3×); dirty state discarded; next start restores from previous `_lifecycle_image` |
| Any non-running container detected at startup | Force-removed with `rm -f` before `run` — covers "Exited", "Created" (partial run), and "Dead" states |
| `sandbox.destroy()` | `rm -f` (no-op if already removed); GPU released once confirmed gone; `rmi agency/lifecycle-<name>`; any `pretool-*` images cleaned up |
| `atexit` | All live containers removed (guard against hard-killed processes) |

**Failure revert**: when a tool errors, `stop(commit=False)` discards the container with its partial state. The next tool call recreates from the last successful `_lifecycle_image`, so the agent's workspace is automatically rolled back to the last known-good state. The agent receives `workspace_reverted` in the error response — but only when `stop()` actually ran; if it was deferred because background work was still pending, no such note is added, since nothing was actually reverted yet.

**Container naming**: each container is named `sandbox-{RUN_ID}-{agname}`, where `_RUN_ID` (`agsandbox_backends.container._RUN_ID`) is a per-process UUID prefix. This prevents cross-run name collisions when an agent crashes without cleanup and is restarted with the same `agname`. The lifecycle image name is produced by `_lifecycle_tag()`, which lowercases the image name — both runtimes require all repository names to be lowercase.

**`stop()`/`destroy()` reliability**: `rm -f` is retried up to 3 times (each attempt through `_run()`, which holds `_get_docker_semaphore()`'s semaphore — caps all concurrent daemon calls, both runtimes, at 16). Unlike an earlier version of this backend, a persistent failure is **not** silently swallowed:

- `stop()` still attempts every remaining step regardless (the commit-retry block, if any; the runtime-slot/GPU release checks, both gated on the container being confirmed actually gone) but then **raises** the underlying exception at the end — a commit failure and an `rm` failure can both have occurred; the `rm` failure is raised in preference, since an unconfirmed removal is more severe (real resources may still be held) than a missed checkpoint.
- `destroy()` follows the same cleanup-then-raise shape: it still runs the GPU-release check and the checkpoint/pretool image cleanup even if `rm -f` failed, then raises at the end. This matters because `destroy()` is called from `atexit`/`agSandbox.__del__` — a raise there is caught and logged by the atexit wrapper, not fatal, but skipping the rest of cleanup on the first failure (the old behavior) meant a failed `rm` could also leave dangling checkpoint/pretool images behind forever.
- The non-critical cleanup steps — the old-image inspect/delete after a re-commit, and the checkpoint/pretool image deletion in `destroy()` — remain best-effort: they print a `WARNING` and continue rather than raising, since a stray dangling image costs disk space, not correctness.

## Cross-process safety

There is no `self._started` cache anywhere on this backend (or any backend, as of the same redesign) — ground truth for every decision (`_ensure_started()`'s reuse-vs-recreate branch, `stop()`'s and `destroy()`'s early-return/no-op checks) is always `_container_running()`, a real `docker/podman inspect` call, never an in-memory flag. Tool calls with `run_in_subprocess=True` (the default) each get a *fresh* cloudpickled copy of the backend dispatched to a `ProcessPoolExecutor` worker, so a per-process flag would only ever reflect what the object looked like at the *original* process's last pickle — never what a different worker already did to the real container. Every call pays the cost of a real inspect rather than a cached read; the alternative (trusting a flag that could be stale in exactly the case that matters most) was the source of a real bug: a container started by a worker process being invisible to the orchestrating process's own stale flag.

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

GPU passthrough flags are passed to `run` when a GPU is detected on the host (`agsandbox_backends.container._gpu_flags(runtime)`, cached per runtime via `agresources.detect_gpus()`). On CPU-only hosts no flag is passed. NVIDIA differs by runtime — Docker uses `--gpus all` (nvidia-container-toolkit's Docker-specific CLI hook), while Podman needs `--device nvidia.com/gpu=all` (CDI) instead: Podman silently accepts `--gpus all` without erroring, but never mounts the driver/devices, so a container started that way has no GPU access at all despite `podman run` succeeding. AMD/ROCm uses `--device /dev/kfd --device /dev/dri` for both runtimes identically.

Even with those flags, GPUs are **not accessible by default** — every `exec()` call unconditionally exports `CUDA_VISIBLE_DEVICES="NoDevFiles"` when no virtual reservation is active, making all GPUs invisible to CUDA (readonly-exported, so a command can't hijack a different GPU by reassigning the variable inline). Calling `reserve_gpu` sets only a virtual flag; no physical GPU is taken yet. When `exec()` runs a bash command and the virtual flag is set, a physical GPU is claimed from the pool at that moment (blocking until one is free) and `CUDA_VISIBLE_DEVICES=<id>` (and `HIP_VISIBLE_DEVICES=<id>` for ROCm) is injected.

**GPU release is tied to the sandbox's lifetime, not to individual exec() calls.** There is no `_gpu_is_clear()`/polling mechanism at all (removed entirely, on every backend) — `agResourcePool.release_gpu()` releases the semaphore immediately, unconditionally, with no wait. Safety comes purely from call ordering: `stop()`/`destroy()` always remove the container (a real, structurally-guaranteed kill of everything inside it) *before* releasing the GPU, so by the time release happens the container — and anything that might have been using the GPU — is already confirmed gone. There is also no agent-facing `gpu_release` tool anymore: once `reserve_gpu` triggers a real acquisition, the GPU is held until `stop()`/`destroy()` tears the sandbox down, not released back mid-skill on request.

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
