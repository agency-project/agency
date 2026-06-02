# Container Sandboxing

Each `agent` instance owns exactly one Docker (or Podman) container for the duration of its lifetime. All filesystem operations — bash commands, file reads, file writes, glob searches, grep searches — execute inside that container, never on the host.

## Runtime detection

`get_container_runtime()` in `agsandbox.py` prefers Docker when both are installed and reachable, falling back to Podman. The result is cached for the process lifetime.

## Container lifecycle

| Event | What happens |
|---|---|
| `agent.__init__` (fresh) | `docker run -d [--gpus all] [-v ...] --name sandbox-<uuid> python:3.12-slim tail -f /dev/null` |
| `agent.__init__` (fork) | `docker commit sandbox-<parent>` → `docker run -d [--gpus all] [-v ...] --name sandbox-<uuid> <snapshot>` |
| `agent.__del__` | `docker rm -f sandbox-<uuid>` + `docker rmi <snapshot>` (fork only) |

The base image is `python:3.12-slim`. `ripgrep` is installed at container startup so that `glob` and `grep` tools work inside the container.

## GPU device access

`--gpus all` is passed to `docker run` when `nvidia-smi` detects GPUs on the host, mounting the NVIDIA device files into the container. On CPU-only hosts the flag is omitted.

Even with `--gpus all`, GPUs are **not accessible by default** — every `exec()` call unconditionally exports `CUDA_VISIBLE_DEVICES=""`, making all GPUs invisible to CUDA. A GPU becomes visible only after the agent calls `gpu_acquire`, which sets `CUDA_VISIBLE_DEVICES=<id>` for subsequent exec calls. This prevents runaway GPU use even if the agent runs code that would otherwise access GPUs without going through the resource pool.

## Shared output directory

When `agent.output_dir` is set, two volume mounts are added to `docker run`:

```
-v <output_dir>:/agent_output:ro               # full shared dir, read-only
-v <output_dir>/<uuid>:/agent_output/<uuid>:rw  # own subdir, read-write
```

The more-specific rw mount shadows the parent ro mount for the agent's own directory. The result:
- `/agent_output/<uuid>/` — writable; the agent exports its work here
- `/agent_output/<other-uuid>/` — read-only; the agent can read other agents' exports
- The host directory mirrors the container paths exactly

## Forking

```
parent container  ──docker commit──▶  snapshot-<child-uuid>
                                             │
                                       docker run -d
                                             │
                                             ▼
                                      child container
```

The child starts from the parent's exact filesystem state. Subsequent writes in either direction are fully isolated — the child's container is independent.

## exec wrapper

Every bash command is wrapped before being sent to the container shell:

```sh
exec 2>&1           # merge stderr into stdout (keeps the __BGPIDS__ marker intact)

# always set — "" when no GPU held, "<id>" when gpu_acquire was called
export CUDA_VISIBLE_DEVICES=<id or "">

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

The `__BGPIDS__` annotation is stripped before the output is returned to the LLM. PIDs extracted from it are written into `sandbox._watched_pids`. The `/proc` diff catches all spawned processes regardless of how they were launched — `&`, `subprocess.Popen`, double-fork daemons — because it compares the full process table rather than relying on shell job control.

## Process tracking state

| Field | Contents |
|---|---|
| `_baseline_pids` | PIDs present when the container was created — never monitored |
| `_watched_pids` | PIDs spawned by user commands currently under monitoring |
| `_daemon_pids` | PIDs explicitly released via `daemon_release` — excluded from monitoring |

`get_live_pids()` reads the full `/proc` table on each call, expands `_watched_pids` to include newly discovered descendants, propagates daemon status down the process tree, and returns the set of non-baseline, non-daemon, non-zombie PIDs.

## File I/O

- **`write_file`** pipes content over stdin (`docker exec -i`) to avoid shell-quoting issues with arbitrary content.
- **`read_file`** runs `cat <path>` inside the container; raises `FileNotFoundError` on a non-zero exit.
- Pagination, fuzzy-replace logic, and directory listing all run in Python on the host; only raw bytes travel through the container boundary.

## `agSandbox` API

```python
sb = agSandbox(uuid)                                        # fresh container
sb = agSandbox(uuid, parent_uuid=parent_uuid)               # forked from parent
sb = agSandbox(uuid, output_dir=Path("runs/agent_output"))  # with shared output dir

sb.exec(cmd, workdir="/workspace", timeout=120) -> (str, int)
sb.read_file(path) -> str
sb.write_file(path, content)
sb.update_limits(cpus=4.0, memory="8g")
sb.get_live_pids() -> set[int]
sb.pid_status_summary() -> str
sb.release_daemon(pid)
sb.release_resources(pool)
sb.destroy()
```

## BASE_IMAGE

Override the base image before creating any agents:

```python
from src.agsandbox import agSandbox
agSandbox.BASE_IMAGE = "my-registry/custom-image:latest"
```
