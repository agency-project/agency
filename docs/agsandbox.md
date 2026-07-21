# Sandboxing

> **Lifecycle warning:** `agSandbox` wraps a live sandbox backend (a Docker/Podman container, or a chroot jail — see "Backend selection" below). Cleanup relies on `agSandbox.__del__` and an `atexit` handler. Neither runs on SIGKILL, and `__del__` may silently fail during interpreter shutdown (`sys.meta_path` is None by then). In long-running processes or when spawning many sandboxes, call `sandbox.destroy()` explicitly. There's no ownership flag to manage — sharing a sandbox between agents is just `ag.sandbox = sb` (or `agent(sandbox=sb)`) on each; `agskill` provisions/stops any sandbox it finds on `ag.sandbox` the same way regardless of where it came from, and the object itself is cleaned up once nothing references it anymore (see "Concurrent access" below).

All filesystem operations — bash commands, file reads, file writes, glob searches, grep searches — execute inside a sandbox, never directly on the host. `agSandbox` (in `agsandbox.py`) is a thin facade: it resolves the image/mounts vocabulary that's meaningful regardless of backend, then builds and delegates every operation to an `agsandbox_backend` chosen by `agSandboxBackendConfig.backend`. Three backends exist today, each a real subclass in its own module under `agsandbox_backends/` — see **[agsandbox_backends/base.md](agsandbox_backends/base.md)** for backend selection, and [container.md](agsandbox_backends/container.md)/[docker.md](agsandbox_backends/docker.md)/[podman.md](agsandbox_backends/podman.md)/[chroot.md](agsandbox_backends/chroot.md) for how each one actually works. Everything below describes the facade — construction, the `agSandbox` API, concurrent access, GPU accounting at the facade level, the exec wrapper, file I/O, and config — regardless of which backend is behind it.

## GPU device access

`--gpus all` (docker) / CDI `=all` (podman) / per-file device binds scoped to the currently-leased GPU (chroot, re-derived fresh on every `exec()`) make the host's GPU devices reachable, **once the sandbox has actually called `reserve_gpu()`** — a sandbox that never reserves a GPU gets nothing GPU-related passed/mounted at all, same as a CPU-only host. See [agsandbox_backends/container.md](agsandbox_backends/container.md)/[chroot.md](agsandbox_backends/chroot.md) for the backend-specific mechanics, including why the container backends attach *every* GPU rather than just the one currently leased (it's what lets `stop()` release the physical GPU between tool calls without ever needing to recreate the container).

Even so, GPUs are **not accessible by default** — every `exec()` call unconditionally exports `CUDA_VISIBLE_DEVICES="NoDevFiles"` when no GPU is currently leased, making all GPUs invisible to CUDA (`readonly`-exported, so a command can't hijack a different GPU by reassigning the variable inline). Calling `reserve_gpu` sets only a virtual flag; no physical GPU is taken yet. When `exec()` runs a bash command and the virtual flag is set but no GPU is currently held, a physical GPU is claimed from the pool at that moment (blocking until one is free) and `CUDA_VISIBLE_DEVICES=<id>` is injected — held across every subsequent `exec()` call, not reacquired each time.

**GPU release now happens on every hibernate, not just at teardown.** There is no `gpu_release` tool and no `is_clear`/polling mechanism — `pool.release_gpu()` releases the semaphore immediately, unconditionally, with no wait. Safety comes purely from ordering: `stop()`/`rm_container()`/`destroy()` always stop or tear down whatever the sandbox was running *before* releasing the GPU, so by the time release happens, nothing this backend could see is still using it. A GPU, once actually acquired, is released every time the sandbox hibernates between tool calls (`sandbox.stop()`) and re-acquired (possibly a *different* physical GPU) on the next `exec()` — there's still no way for the agent to free it back to the pool explicitly mid-tool-call, but it's no longer held for the sandbox's whole lifetime either.

## Concurrent access

Each `agSandbox` allocates `self._lock = threading.RLock()` in `__init__`. It exists because a single `agSandbox` instance can be handed to more than one agent (there's no ownership flag preventing this — see the lifecycle warning above), and two skill runs interleaving `exec()` / `stop()` / `_ensure_started()` calls against the same container would corrupt its state.

The lock is **not self-enforcing** — `agSandbox`'s own methods don't acquire it. Instead, `agskill.py`'s `_task()` acquires `ag.sandbox._lock` right after provisioning and holds it for the *entire* skill run, releasing it only after the final teardown (`commit()` on success, `rm_container()` on failure). This makes "one skill run owns this sandbox at a time" an invariant enforced by the caller (agskill), not by `agSandbox` itself. Code that drives a shared `agSandbox` outside of an agskill run (harness scripts, custom orchestration) must coordinate its own access if it needs the same guarantee — see `Design_architecture.md`'s "Per-sandbox mutex" section for the full rationale.

Because `threading.RLock` isn't picklable, `agSandbox` defines `__getstate__`/`__setstate__` to drop `_lock` before pickling and allocate a fresh one on unpickling. This matters because custom tools with `run_in_subprocess=True` (the default) get `cloudpickle`d to a worker process — without this, capturing a sandbox in such a tool's closure would raise `TypeError: cannot pickle '_thread.RLock' object`. All built-in tools (bash, read, write, grep, glob, …) use `run_in_subprocess=False` and never hit this path.

## Shared output directory

When `agent.output_dir` is set, each agent gets its own subdirectory mounted read-write:

```
-v <output_dir>/<agname>:/agent_output/<agname>:rw
```

All agents can write to `/agent_output/<own-agname>/` inside the container; files appear on the host at `agent.output_dir/<agname>/` immediately.

## Forking

```
parent.sandbox._checkpoint_image ──tag_image──▶ agency/lifecycle-<fork-agname>
                                            │
                                     (consumed by fork's first _task())
```

Forking copies the parent's checkpoint image tag to a new tag for the fork via `type(parent.sandbox._backend).tag_image(...)` — dispatched to whichever backend class actually produced the checkpoint (`docker tag`/`podman tag` for the container backends, a directory copy for chroot — see [agsandbox_backends/chroot.md](agsandbox_backends/chroot.md)), not a bare docker-only call, since a chroot snapshot directory and a docker/podman image tag are unrelated formats. No container/jail is created at fork time — that happens lazily when the fork's first `_task()` runs, restoring from the copied tag.

Because forks wait for `src.ctx.resolve_prev_dependencies()` before construction, the parent's task is always complete before the fork is built, so the checkpoint image is already the committed post-task state.

## exec wrapper

> This section (and "Process tracking state" below) describes the **container backends'** before/after PID-diffing specifically. Chroot tracks background work differently — purely via process groups, with no before/after diff and no `_baseline_pids`/`_watched_pids` population at all — see [agsandbox_backends/chroot.md](agsandbox_backends/chroot.md#background-process-tracking).

Every bash command is wrapped before being sent to the container shell:

```sh
exec 2>&1           # merge stderr into stdout (keeps __BGPIDS__ marker intact)

export CUDA_VISIBLE_DEVICES=<id or "">   # set to physical GPU ID if one was acquired at the start of exec(), or "" (NoDevFiles) if no virtual reservation is active

# snapshot /proc before the command
__AGENCY_BEFORE=$(for __d in /proc/[0-9]*; do
  [ -f "$__d/status" ] && echo "${__d##*/}"; done | tr '\n' ' ')
__AGENCY_SHELL=$$

<user command>
__AGENCY_RC=$?

# diff /proc after: any new PID was spawned by the command
__AGENCY_BGPIDS=''
for __d in /proc/[0-9]*; do
  [ -f "$__d/status" ] || continue
  __p=${__d##*/}
  case " $__AGENCY_BEFORE $__AGENCY_SHELL " in
    *" $__p "*) ;;
    *) __AGENCY_BGPIDS="$__AGENCY_BGPIDS $__p" ;;
  esac
done

printf '\n__BGPIDS__:%s' "$__AGENCY_BGPIDS"
exit $__AGENCY_RC
```

The `__BGPIDS__` annotation is stripped before output is returned to the LLM. PIDs extracted from it are written into `sandbox._watched_pids`. The `/proc` diff catches all spawned processes regardless of how they were launched — `&`, `subprocess.Popen`, double-fork daemons — because it compares the full process table rather than relying on shell job control.

## Process tracking state

| Field | Contents |
|---|---|
| `_baseline_pids` | PIDs present when the container was created — never monitored |
| `_watched_pids` | PIDs spawned by user commands currently under monitoring |
| `_daemon_pids` | PIDs explicitly released via `daemon_release` — excluded from monitoring |

`get_live_pids()` reads the full `/proc` table on each call, expands `_watched_pids` to include newly discovered descendants, propagates daemon status down the process tree, and returns the set of non-baseline, non-daemon, non-zombie PIDs.

## File I/O

- **`write_file`** pipes content over stdin (`exec -i`) to avoid shell-quoting issues with arbitrary content.
- **`read_file`** reads raw bytes via `base64 <path>` in the container, then decodes strictly as UTF-8. This gives three distinct, actionable exceptions:

  | Exception | Cause | Detection |
  |---|---|---|
  | `IsADirectoryError` | Path is a directory | `base64` fails; `test -d` confirms |
  | `UnicodeDecodeError` | File contains non-UTF-8 bytes (binary) | Strict `bytes.decode("utf-8")` fails |
  | `FileNotFoundError` | Path does not exist | `base64` fails; `test -d` returns non-zero |

  The base64 approach avoids `errors="replace"` (which would silently corrupt binary detection). Callers that only need to catch "file not readable for any reason" can catch `Exception`; callers that need to distinguish directory from missing from binary should catch each type individually.

- **`read_file_bytes`** reads raw bytes via the same base64 round-trip as `read_file`, but skips the UTF-8 decode. Returns `bytes`. Raises `IsADirectoryError` or `FileNotFoundError` the same way; never raises `UnicodeDecodeError`. Use for binary files (images, audio, compiled artifacts) where text decoding is incorrect.

- **`write_file_bytes`** base64-encodes the `bytes` on the host and decodes inside the container (`printf '%s' <b64> | base64 -d > <path>`), avoiding shell-quoting issues with arbitrary byte sequences.

- Pagination, fuzzy-replace logic, and directory listing all run in Python on the host; only raw bytes travel through the container boundary.

## `agSandbox` API

```python
sb = agSandbox(agname)
sb = agSandbox(agname, agconfig=agConfig(agSandboxConfig().add_mount("out", Path("runs/agent_output"), "/agent_output")))
sb = agSandbox(agname, checkpoint_image="agency/lifecycle-myagent")

# Construction is cheap — the backend does no real work until _ensure_started() runs
# (a docker/podman run, or just an mkdir for chroot).
sb._ensure_started()    # called automatically on first exec(); idempotent

sb._lock  # threading.RLock; held by agskill for the whole skill run — see "Concurrent access" above

sb.exec(cmd, workdir="/workspace", timeout=120) -> (str, int)
sb.read_file(path) -> str           # UTF-8 text; raises UnicodeDecodeError for binary
sb.read_file_bytes(path) -> bytes   # raw bytes; no decode attempt
sb.write_file(path, content)        # UTF-8 text via stdin pipe
sb.write_file_bytes(path, data)     # raw bytes via base64 round-trip
sb.commit(tag=None) -> bool  # False if backend was never started; True after a successful snapshot
                              # to agency/lifecycle-<name> (default) or tag -- container/workspace
                              # stays running/intact either way, nothing is removed
sb.stop()               # hibernate: pause without removing (docker/podman stop); releases the
                         # runtime slot AND the GPU -- see container.md's "GPU device access"
sb.rm_container()       # force-remove/discard outright; releases the runtime slot and the GPU
sb.update_limits(cpus=4.0, memory="8g")   # no-op on the chroot backend -- no cgroup of its own
sb.get_live_pids() -> set[int]
sb.pid_status_summary() -> str
sb.release_daemon(pid)
sb.release_resources(pool)
sb.destroy()            # tear down the backend + delete its checkpoint image/snapshot
sb.image_kind -> str    # "container" or "chroot" -- which backend produced sb._checkpoint_image
```

Every method above is a one-line delegate from the `agSandbox` facade to `sb._backend` (an `agsandbox_backend` subclass — see [agsandbox_backends/base.md](agsandbox_backends/base.md)). `sb._backend` is the thing that actually knows how to talk to Docker/Podman or run `unshare`+`chroot`; the facade only owns what's backend-agnostic (config resolution, the `_lock`, GPU/CPU pool bookkeeping). See [agsandbox_backends/chroot.md](agsandbox_backends/chroot.md) for the chroot backend's own mechanics (mount namespace, directory layout, checkpointing, and what it deliberately doesn't isolate).

## Custom base image and mounts

Preferred: pass an `agConfig` — image/mounts are resolved once per sandbox at
construction (not a shared mutable global, so no race between differently-
configured sandboxes created concurrently):

```python
from agency.agconfig import agConfig
from agency.agsandbox import agSandboxConfig

cfg = agConfig()
agSandboxConfig(cfg).set_base_image("my-registry/custom-image:latest")
agSandboxConfig(cfg).add_mount("hf_cache", "/host/path/to/cache", "/root/.cache/huggingface")
agent.default_agconfig = cfg   # picked up by every agent()/agSandbox() created after this
```

`base_image` is declared as a real field on `agSandbox` (`agSandbox.base_image`),
so the same override can also be spelled as nested attribute access on the
`agConfig` before any sandbox is constructed — equivalent to the
`set_base_image` call above:

```python
cfg.agSandbox.base_image = "my-registry/custom-image:latest"
```

`base_image` and `mounts` (the `host` side of each mount) are resolved by the facade and handed to whichever backend is selected — `_DockerBackend`/`_PodmanBackend` use `base_image` for `docker run`/`podman run`; `_ChrootBackend` ignores it entirely (there's no image concept — see [agsandbox_backends/chroot.md](agsandbox_backends/chroot.md)) but still bind-mounts configured `mounts` into the jail at their `container` path.

## `change_config` / `get_config_copy`

`agSandbox.change_config(new_cfg)` clones `new_cfg` and replaces `sb._agconfig` with it; `sb.get_config_copy()` returns a clone of the sandbox's current agconfig (or `None` if it has none). These exist mainly for consistency with the rest of the object graph — `agent.change_config()` calls `ag.sandbox.change_config()` alongside `ag.llm`/`ag.log` so every object shares one source of truth.

They do **not** let you change a running sandbox's image or mounts: `base_image` and `mounts` are `StaticConfigParam` fields, resolved once at construction and locked from then on (see "Changing a Static field" in [Design_configuration.md](Design_configuration.md)) — `change_config` replaces the agconfig object, but a already-locked static value doesn't re-resolve from it. To change a sandbox's image/mounts, use the clone-and-recreate pattern (`cfg2 = ag.agconfig.clone()`, mount on `cfg2`, `ag.sandbox.destroy()`, `ag.sandbox = None`, `ag.agconfig = cfg2`) documented there instead.

Don't assign `agSandbox.base_image = ...` directly on the class — that
replaces the field descriptor itself rather than setting a value, breaking
the field for every sandbox in the process.
