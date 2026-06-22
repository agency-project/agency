# Container Sandboxing

All filesystem operations — bash commands, file reads, file writes, glob searches, grep searches — execute inside a Docker or Podman container, never on the host. Containers are created lazily: a container starts only when a task actually calls a tool with `need_sandbox=True`. Tasks that complete using only host-side tools (web fetch, `ask_human`, paper search, …) never create a container at all. When a container is started it is committed to a checkpoint image at the end of the task and destroyed, with state persisted between tasks via those images.

## Runtime detection

`get_container_runtime()` in `agsandbox.py` prefers Docker when both are installed and reachable, falling back to Podman. The result is cached for the process lifetime.

## Building the sandbox image

```bash
# Docker
docker build -t agency-sandbox:latest images/

# Podman (requires localhost/ prefix)
podman build -t localhost/agency-sandbox:latest images/
```

The `images/Dockerfile` installs `ripgrep` on top of `python:3.12-slim`. Both runtimes use the same Dockerfile.

## Container lifecycle

| Event | What happens |
|---|---|
| `agent.__init__` | No container created — `agSandbox` object is created cheaply; `_checkpoint=None` |
| First `need_sandbox=True` tool call | Any stale container removed; `docker run` from `_checkpoint` image (or `BASE_IMAGE` on first task); `_ensure_started()` is called at most once per task |
| `agent.run()` task end (container started) | `docker commit container → agency/ckpt-<pid>-<agname>` (returns `True`); container destroyed; `_checkpoint` updated |
| `agent.run()` task end (no container started) | `commit()` returns `False`; `_checkpoint` unchanged; no Docker calls |
| `agent.__del__` | Checkpoint image removed via `docker rmi` |
| `atexit` | All live containers removed (guards against hard-killed processes) |

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
parent._checkpoint ──docker tag──▶ agency/ckpt-<pid>-<fork-agname>
                                            │
                                     (consumed by fork's first _task())
```

Forking copies the parent's checkpoint image tag to a new tag for the fork via `docker tag`. No container is created at fork time — the fork's container is created lazily when the fork's first `_task()` runs, restoring from the copied tag.

Because forks wait for `src._history._resolve()` before construction, the parent's task is always complete before the fork is built, so the checkpoint image is already the committed post-task state.

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
sb = agSandbox(agname, output_dir=Path("runs/agent_output"))
sb = agSandbox(agname, restore_image="agency/ckpt-p1234-myagent")

# Construction is cheap — no Docker calls until _ensure_started() runs.
sb._ensure_started()    # called automatically on first exec(); idempotent

sb.exec(cmd, workdir="/workspace", timeout=120) -> (str, int)
sb.read_file(path) -> str           # UTF-8 text; raises UnicodeDecodeError for binary
sb.read_file_bytes(path) -> bytes   # raw bytes; no decode attempt
sb.write_file(path, content)        # UTF-8 text via stdin pipe
sb.write_file_bytes(path, data)     # raw bytes via base64 round-trip
sb.commit(tag) -> bool  # False if container never started; True after docker commit
sb.update_limits(cpus=4.0, memory="8g")
sb.get_live_pids() -> set[int]
sb.pid_status_summary() -> str
sb.release_daemon(pid)
sb.release_resources(pool)
sb.destroy()
```

## Custom base image

Override before creating any agents:

```python
from agency.agsandbox import agSandbox
agSandbox.BASE_IMAGE = "my-registry/custom-image:latest"
```
