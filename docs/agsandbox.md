# Container Sandboxing

Each `agent` instance owns exactly one Docker or Podman container for the duration of its lifetime. All filesystem operations — bash commands, file reads, file writes, glob searches, grep searches — execute inside that container, never on the host.

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
| `agent.__init__` (fresh) | Any stale container with the same name is removed, then: `run -d [--gpus all] [-v ...] --name sandbox-<agname> agency-sandbox:latest tail -f /dev/null` |
| `agent.__init__` (fork) | `commit sandbox-<parent>` → `run -d [--gpus all] [-v ...] --name sandbox-<agname> <snapshot>` |
| process exit | `atexit` handler calls `destroy()` on all live containers |
| explicit | `ag.sandbox.destroy()` |

Stale containers from a previously hard-killed process are removed automatically at the start of `__init__`, so container name conflicts never block a fresh run.

## GPU device access

`--gpus all` is passed to `run` when `nvidia-smi` detects GPUs on the host, mounting the NVIDIA device files into the container. On CPU-only hosts the flag is omitted.

Even with `--gpus all`, GPUs are **not accessible by default** — every `exec()` call unconditionally exports `CUDA_VISIBLE_DEVICES=""`, making all GPUs invisible to CUDA. A GPU becomes visible only after the agent calls `gpu_acquire`, which sets `CUDA_VISIBLE_DEVICES=<id>` for subsequent exec calls.

## Shared output directory

When `agent.output_dir` is set, each agent gets its own subdirectory mounted read-write:

```
-v <output_dir>/<agname>:/agent_output/<agname>:rw
```

All agents can write to `/agent_output/<own-agname>/` inside the container; files appear on the host at `agent.output_dir/<agname>/` immediately.

## Forking

```
parent container  ──commit──▶  snapshot-<child-agname>
                                       │
                                    run -d
                                       │
                                       ▼
                                child container
```

The child starts from the parent's exact filesystem state. Subsequent writes in either direction are fully isolated. The snapshot image is deleted when the child's container is destroyed.

## exec wrapper

Every bash command is wrapped before being sent to the container shell:

```sh
exec 2>&1           # merge stderr into stdout (keeps __BGPIDS__ marker intact)

export CUDA_VISIBLE_DEVICES=<id or "">   # always set

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
- **`read_file`** runs `cat <path>` inside the container; raises `FileNotFoundError` on a non-zero exit.
- Pagination, fuzzy-replace logic, and directory listing all run in Python on the host; only raw bytes travel through the container boundary.

## `agSandbox` API

```python
sb = agSandbox(agname)
sb = agSandbox(agname, parent_agname="parent")
sb = agSandbox(agname, output_dir=Path("runs/agent_output"))

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

## Custom base image

Override before creating any agents:

```python
from agency.agsandbox import agSandbox
agSandbox.BASE_IMAGE = "my-registry/custom-image:latest"
```
