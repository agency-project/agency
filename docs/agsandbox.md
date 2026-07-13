# Sandboxing

> **Lifecycle warning:** `agSandbox` wraps a live sandbox backend (a Docker/Podman container, or a chroot jail — see "Backend selection" below). Cleanup relies on `agSandbox.__del__` and an `atexit` handler. Neither runs on SIGKILL, and `__del__` may silently fail during interpreter shutdown (`sys.meta_path` is None by then). In long-running processes or when spawning many sandboxes, call `sandbox.destroy()` explicitly. There's no ownership flag to manage — sharing a sandbox between agents is just `ag.sandbox = sb` (or `agent(sandbox=sb)`) on each; `agskill` provisions/stops any sandbox it finds on `ag.sandbox` the same way regardless of where it came from, and the object itself is cleaned up once nothing references it anymore (see "Concurrent access" below).

All filesystem operations — bash commands, file reads, file writes, glob searches, grep searches — execute inside a sandbox, never directly on the host. `agSandbox` (in `agsandbox.py`) is a thin facade: it resolves the image/mounts vocabulary that's meaningful regardless of backend, then builds and delegates every operation to an `agsandbox_backend` (in `agsandbox_backend.py`) chosen by `agSandboxBackendConfig.backend`. Two backends exist today:

- **Container backend** (`_ContainerBackend`) — a Docker or Podman container. Containers are created lazily: one starts only when a task actually calls a tool with `run_in_subprocess=True`. Tasks that complete using only host-side tools (web fetch, `ask_human`, paper search, …) never create a container at all. After each successful sandbox tool call, the container state is committed to a lifecycle image and the container is removed — the session keyring and GPU are freed so other agents can use them while the LLM thinks. On the next tool call the container is recreated from the lifecycle image, restoring `/workspace` and all other state. Everything below through "Custom base image and mounts" describes this backend specifically (it's still the default and the only one with full GPU/resource-limit/checkpoint-image support) — see "Chroot backend" further down for what differs there.
- **Chroot backend** (`_ChrootBackend`) — a per-agent directory chrooted into via an unprivileged `unshare --user --map-root-user --mount`, needing no root/sudo/setcap. Lighter weight, but a narrower isolation guarantee (filesystem containment only) — see "Chroot backend" below.

## Backend selection

`agSandboxBackendConfig.backend` picks the backend: `"auto"` (default), `"podman"`, `"docker"`, or `"chroot"`.

```python
from agency.agconfig import agConfig
from agency.agsandbox_backend import agSandboxBackendConfig

cfg = agConfig(agSandboxBackendConfig(backend="chroot"))
sb = agSandbox("myagent", agconfig=cfg)
```

`"auto"` (`agsandbox_backend._auto_detect_runtime()`) probes in this order: **podman → docker → chroot**, first usable one wins.

- Podman/docker usability: the binary is on `PATH` and `<runtime> info` succeeds (`agsandbox_backend._runtime_works()`), cached for the process lifetime in `_RUNTIME`.
- Chroot usability (`agsandbox_backend.chroot_available()`, also cached): `/proc/sys/kernel/unprivileged_userns_clone` isn't explicitly disabled (the file doesn't exist on distros that ship it enabled upstream) **and** a live `unshare --user --map-root-user --mount true` actually succeeds — the sysctl alone can false-positive on hosts where AppArmor/seccomp additionally restrict unprivileged user namespaces.

An explicit `backend="docker"|"podman"|"chroot"` raises immediately with a clear error if that specific backend isn't usable, rather than silently falling through to another one.

## Building the sandbox image

```bash
GPU_TYPE=cpu ./images/build.sh   # or nvidia/rocm; auto-detected if omitted
```

`build.sh` builds `agency-sandbox:latest` for **every container runtime installed on the host, podman first** — matching `agsandbox_backend`'s auto-selection order (see "Backend selection" above). An image built only for docker would leave podman's separate image store empty, so podman would try (and fail) to pull the image from a registry instead of finding it locally; building for both keeps whichever one auto-selection picks actually usable. Podman's build is tagged `localhost/agency-sandbox:latest` (the `localhost/` prefix it requires to resolve an unqualified name to its own local store instead of a registry); docker's is tagged bare. Each build gets its own smoke test (`import torch`) run against the runtime that built it.

If `HF_TOKEN` is set, it's passed as a `--secret` for gated model downloads during the build. This is written to a private temp file and passed as `--secret id=hf_token,src=<file>` for both runtimes — docker buildx's `--secret id=...,env=VAR` shorthand is **not** portable to podman's buildah, which rejects it ("incorrect secret flag format: should be `--secret id=foo,src=bar`"); the file-based form is the one both accept.

The `images/Dockerfile` installs `ripgrep` on top of `python:3.12-slim`. `Dockerfile.nvidia`/`Dockerfile.rocm` are used instead when `GPU_TYPE` is `nvidia`/`rocm`.

## Container lifecycle

*This section, and everything through "Custom base image and mounts" below, describes the container backend (`_ContainerBackend`) specifically. See "Chroot backend" further down for how the chroot backend's lifecycle differs.*

The container exists only during active tool execution. Between tool calls the container is removed, releasing the Linux session keyring and any held GPU so concurrent agents can use those resources.

| Event | What happens |
|---|---|
| `agSandbox.__init__` | No container created — cheap object; `_lifecycle_image=None` |
| Any `run_in_subprocess=True` tool call | `_ensure_started()` runs lazily: if container is already running, reuse it; otherwise `docker rm -f` any leftover zombie, then `docker run` from `_lifecycle_image` (or `base_image` on first use) |
| After **successful** sandbox tool call | `sandbox.stop(commit=True)`: `docker commit → agency/lifecycle-<name>`; `docker rm -f` (retried up to 3×); `_lifecycle_image` updated |
| After **failed** sandbox tool call | `sandbox.stop(commit=False)`: `docker rm -f` without commit; dirty state discarded; next start restores from previous `_lifecycle_image` |
| Any non-running container detected at startup | Force-removed with `docker rm -f` before `docker run` — covers "Exited", "Created" (partial docker run), and "Dead" states |
| `sandbox.destroy()` | `docker rm -f` (no-op if already removed); `docker rmi agency/lifecycle-<name>`; any `pretool-*` images cleaned up |
| `atexit` | All live containers removed (guard against hard-killed processes) |

**Failure revert**: when a tool errors, `stop(commit=False)` discards the container with its partial state. The next tool call recreates from the last successful `_lifecycle_image`, so the agent's workspace is automatically rolled back to the last known-good state. The agent receives `workspace_reverted` in the error response to know this happened.

**Container naming**: each container is named `sandbox-{RUN_ID}-{agname}`, where `_RUN_ID` is a per-process UUID prefix. This prevents cross-run name collisions when an agent crashes without cleanup and is restarted with the same `agname`. The lifecycle image name is produced by `_lifecycle_tag()`, which lowercases the Docker image name — Docker requires all repository names to be lowercase.

**`stop()` reliability**: `docker rm -f` is retried up to 3 times. Each attempt goes through `_run()`, which holds `_docker_semaphore` (caps all concurrent daemon calls at 16). If all retries fail, a `WARNING` is emitted to stderr and the framework continues — `_started` is cleared regardless so the next tool call can attempt a fresh container.

## Concurrent access

Each `agSandbox` allocates `self._lock = threading.RLock()` in `__init__`. It exists because a single `agSandbox` instance can be handed to more than one agent (there's no ownership flag preventing this — see the lifecycle warning above), and two skill runs interleaving `exec()` / `stop()` / `_ensure_started()` calls against the same container would corrupt its state.

The lock is **not self-enforcing** — `agSandbox`'s own methods don't acquire it. Instead, `agskill.py`'s `_task()` acquires `ag.sandbox._lock` right after provisioning and holds it for the *entire* skill run, releasing it only after the final teardown `stop()`. This makes "one skill run owns this sandbox at a time" an invariant enforced by the caller (agskill), not by `agSandbox` itself. Code that drives a shared `agSandbox` outside of an agskill run (harness scripts, custom orchestration) must coordinate its own access if it needs the same guarantee — see `Design_architecture.md`'s "Per-sandbox mutex" section for the full rationale.

Because `threading.RLock` isn't picklable, `agSandbox` defines `__getstate__`/`__setstate__` to drop `_lock` before pickling and allocate a fresh one on unpickling. This matters because custom tools with `run_in_subprocess=True` (the default) get `cloudpickle`d to a worker process — without this, capturing a sandbox in such a tool's closure would raise `TypeError: cannot pickle '_thread.RLock' object`. All built-in tools (bash, read, write, grep, glob, …) use `run_in_subprocess=False` and never hit this path.

## GPU device access

`--gpus all` is passed to `run` when `nvidia-smi` detects GPUs on the host, mounting the NVIDIA device files into the container. On CPU-only hosts the flag is omitted.

Even with `--gpus all`, GPUs are **not accessible by default** — every `exec()` call unconditionally exports `CUDA_VISIBLE_DEVICES=""` when no virtual reservation is active, making all GPUs invisible to CUDA. Calling `reserve_gpu` sets only a virtual flag; no physical GPU is taken. When `exec()` runs a bash command and the virtual flag is set, a physical GPU is claimed from the pool at that moment (blocking until one is free) and `CUDA_VISIBLE_DEVICES=<id>` is injected. After a foreground exec with no background processes, the physical GPU is returned to the pool immediately — freeing it for other agents while the LLM thinks. When background processes are alive, the GPU is held until `get_live_pids()` finds them all finished.

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

Forking copies the parent's checkpoint image tag to a new tag for the fork via `type(parent.sandbox._backend).tag_image(...)` — dispatched to whichever backend class actually produced the checkpoint (`docker tag`/`podman tag` for the container backend, a directory copy for chroot — see "Chroot backend" below), not a bare docker-only call, since a chroot snapshot directory and a docker image tag are unrelated formats. No container/jail is created at fork time — that happens lazily when the fork's first `_task()` runs, restoring from the copied tag.

Because forks wait for `src.ctx.resolve_prev_dependencies()` before construction, the parent's task is always complete before the fork is built, so the checkpoint image is already the committed post-task state.

## exec wrapper

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
sb.commit(tag) -> bool  # False if backend was never started; True after a successful snapshot
sb.stop(commit=False)   # tear down; if commit=True, snapshot to agency/lifecycle-<name> first
sb.update_limits(cpus=4.0, memory="8g")   # no-op on the chroot backend -- no cgroup of its own
sb.get_live_pids() -> set[int]
sb.pid_status_summary() -> str
sb.release_daemon(pid)
sb.release_resources(pool)
sb.destroy()            # tear down the backend + delete its checkpoint image/snapshot
sb.image_kind -> str    # "container" or "chroot" -- which backend produced sb._checkpoint_image
```

Every method above is a one-line delegate from the `agSandbox` facade to `sb._backend` (an `agsandbox_backend` subclass — see "Backend selection"). `sb._backend` is the thing that actually knows how to talk to Docker/Podman or run `unshare`+`chroot`; the facade only owns what's backend-agnostic (config resolution, the `_lock`, GPU/CPU pool bookkeeping).

## Chroot backend

`_ChrootBackend` (in `agsandbox_backend.py`) trades Docker/Podman's full isolation for a much lighter mechanism, for the case where you just want each agent to see only its own files and its own installed packages, with no access to the host filesystem, and don't need network/process/resource isolation:

- **Mechanism**: every `exec()` call runs inside a fresh `unshare --user --map-root-user --mount` (an unprivileged user + mount namespace — no root, sudo, or `setcap` needed) followed by `chroot` into a per-agent directory. The mount namespace (and everything bind-mounted into it) is torn down automatically when that one process exits — there's no long-lived daemon to exec into the way `docker exec` has a container to attach to.
- **Directory layout**: `<tmp>/agency-chroot-sandboxes/jails/<sandbox-name>/` is the jail root. `workspace/` inside it is a plain host directory (no bind mount needed, since chroot just repoints `/` — `/workspace` inside the jail *is* that directory) and is the only thing that persists across execs. `bin`, `sbin`, `lib`, `lib32`, `lib64`, `usr`, `etc` are bind-mounted read-only from the host on every exec (so the jail gets the host's own interpreters/system libraries without needing a separate image), and `dev` is bind-mounted read-write (unscoped — see below) since most programs assume `/dev/null`, `/dev/urandom`, etc. exist and are writable. A fresh `procfs` is also mounted so PID tracking (below) keeps working.
- **Checkpointing**: `commit`/`restore`/`stop(commit=...)`/`fork`/`tag_image`/`export_image`/`import_image` all operate on the `workspace/` directory instead of a container filesystem. A commit is `cp -a --reflink=auto <workspace> <tmp>/agency-chroot-sandboxes/snapshots/<sanitized-tag>` — a true point-in-time copy (using a filesystem reflink where available, a full copy otherwise), not a hardlink clone that a later in-place write to the live workspace would silently corrupt. `export_image`/`import_image` tar/untar that snapshot directory.
- **Cross-process safety**: tool calls with `run_in_subprocess=True` (the default) each get a *fresh* cloudpickled copy of the backend dispatched to a `ProcessPoolExecutor` worker, so a worker's own `self._started` is unreliable — it reflects whatever the object looked like at the *original* process's last pickle, not what a different worker already did. `_ensure_started()`/`commit()`/`stop()` therefore check the workspace directory's existence on disk as their ground truth instead of trusting `self._started`, exactly analogous to how the container backend falls back to `_container_running()` (querying the docker daemon) instead of trusting its own `_started` in the same situation — there's no daemon here, so the workspace directory itself is the cross-process source of truth. Getting this wrong previously caused a real bug: a file written by one worker was wiped by the very next worker's `_ensure_started()` re-materializing from the last checkpoint, because that worker's own (stale) view said nothing had started yet.
- **What it does *not* isolate**, by design (matches the scope this backend was built for — filesystem containment + independent per-agent installs, not containment against adversarial code):
  - **Network** — the jailed process shares the host's network stack; there is no network namespace.
  - **Processes** — there is no PID namespace. The fresh `procfs` mounted into the jail reflects the *host's* real process table, so a command running inside the jail can *see* every host process (though signalling/killing them still goes through the kernel's normal permission checks against the real, unprivileged host uid the mapped "root" resolves to — it can't act on processes it doesn't own).
  - **CPU/memory** — no cgroup of its own; `update_limits()` is a no-op.
  - **GPU device scoping** — `CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` are still exported the same way as the container backend, but nothing stops a process from seeing every `/dev` entry the host user can.
- **Cross-process checkpoints**: `agent.save()` records which backend produced a checkpoint (`state["sandbox_image_kind"]`, either `"container"` or `"chroot"`) and `agent.load()` routes `import_image`/`tag_image`/`delete_image` to the matching backend class via `agSandbox.backend_for_image_kind(kind)`, forcing the reconstructed sandbox's `backend` config to `"chroot"` when needed — auto-detection (which prefers podman/docker when available) would otherwise pick a backend that can't make sense of a chroot snapshot's tag.

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

`base_image` and `mounts` (the `host` side of each mount) are resolved by the facade and handed to whichever backend is selected — `_ContainerBackend` uses `base_image` for `docker run`/`podman run`; `_ChrootBackend` ignores it entirely (there's no image concept — see "Chroot backend" above) but still bind-mounts configured `mounts` into the jail at their `container` path.

## `change_config` / `get_config_copy`

`agSandbox.change_config(new_cfg)` clones `new_cfg` and replaces `sb._agconfig` with it; `sb.get_config_copy()` returns a clone of the sandbox's current agconfig (or `None` if it has none). These exist mainly for consistency with the rest of the object graph — `agent.change_config()` calls `ag.sandbox.change_config()` alongside `ag.llm`/`ag.log` so every object shares one source of truth.

They do **not** let you change a running sandbox's image or mounts: `base_image` and `mounts` are `StaticConfigParam` fields, resolved once at construction and locked from then on (see "Changing a Static field" in [Design_configuration.md](Design_configuration.md)) — `change_config` replaces the agconfig object, but a already-locked static value doesn't re-resolve from it. To change a sandbox's image/mounts, use the clone-and-recreate pattern (`cfg2 = ag.agconfig.clone()`, mount on `cfg2`, `ag.sandbox.destroy()`, `ag.sandbox = None`, `ag.agconfig = cfg2`) documented there instead.

Don't assign `agSandbox.base_image = ...` directly on the class — that
replaces the field descriptor itself rather than setting a value, breaking
the field for every sandbox in the process.
