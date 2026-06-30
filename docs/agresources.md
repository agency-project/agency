# Resource Control

`agResourcePool` manages shared GPU tokens and container CPU/memory limits across all agents. It is auto-detected and pre-set on `agent.agresource_pool` — no configuration is required to use it.

## Auto-detection

```python
from agency.agresources import agResourcePool
pool = agResourcePool()   # detects everything automatically
```

| Resource | Detection method | Fallback |
|---|---|---|
| GPUs | `nvidia-smi --query-gpu=index --format=csv,noheader` | `[]` (no GPUs) |
| CPU count | `os.cpu_count()` | `1` |
| Total memory | `/proc/meminfo` (Linux) or `sysctl hw.memsize` (macOS) | `4096 MB` |

Override any value explicitly:

```python
pool = agResourcePool(gpus=[0, 1], total_cpus=32, total_memory_mb=65536)
agent.agresource_pool = pool
```

## GPU access control

Containers are started with `--gpus all` when GPUs are present on the host, so the NVIDIA device files exist inside the container. However, every `exec()` call unconditionally exports `CUDA_VISIBLE_DEVICES=""` when no virtual reservation is active, making all GPUs invisible to CUDA regardless of what the command does.

`reserve_gpu` sets only a virtual flag (`sandbox._gpu_virtual = True`) — no physical GPU is taken at that point. When `exec()` runs a bash command and the virtual flag is set, a physical GPU is claimed from the pool at that moment (blocking until one is free) and `CUDA_VISIBLE_DEVICES=<id>` is injected for the duration of that exec call. Between bash calls the physical GPU is returned to the pool so other agents can use it. If all GPUs are busy when a bash command runs, `exec()` blocks until one is free.

## GPU semaphores

Each GPU ID gets a `threading.Semaphore(1)`. `acquire_gpu()` is called internally by `exec()` — not by the `reserve_gpu` tool directly. `reserve_gpu` only sets `sandbox._gpu_virtual = True`. The physical semaphore is acquired at exec time: `acquire_gpu()` spins across all semaphores until one is free, then returns the GPU ID and sets `sandbox._gpu_id`. `release_gpu()` releases the semaphore. The physical GPU is released when: (a) a foreground exec completes with no background processes — released immediately inside `exec()`; or (b) `get_live_pids()` finds the alive set empty — released at that point for background processes.

## CPU and memory limits

`cpu_acquire` calls `docker update --cpus=N --memory=Mg` on the live container, adjusting Linux cgroup limits without restarting. `0.5` CPUs means the container is throttled to at most half a core's worth of CPU time — it can see all cores but is rate-limited at the cgroup level. `cpu_release` resets to the idle defaults.

CPU and memory limits are set by the sandbox on each tool call via `update_limits()` and are not automatically restored by agresources. `release_resources()` is a manual call to reduce an `agSandbox`'s reported resource usage in the pool (e.g. when the sandbox is destroyed externally).

## Agent-callable resource tools

| Tool | Parameters | Effect |
|---|---|---|
| `reserve_gpu` | — | Reserves GPU access; a physical GPU is assigned lazily when bash runs. No parameters. |
| `gpu_release` | — | Returns the GPU to the pool; `CUDA_VISIBLE_DEVICES` reset to `""` |
| `reserve_cpu` | `cpus` (float), `memory` (string, e.g. `"8g"`) | Boosts container resource limits |
| `cpu_release` | — | Resets limits back to idle defaults (shown in tool description) |
| `daemon_release` | `pid` (int) | Removes a PID from monitoring — skill completes without waiting for it |

The agent calls these tools itself during a skill, just like any other tool. `daemon_release` is always in the tool list; GPU/CPU tools are added only when `agent.agresource_pool` is set (the default).

## Release guarantee

GPU and CPU/memory teardown is always called in the `finally` block of the `_task()` closure inside `agskill.run()`:

```python
try:
    outer_result, updated_ctx, outer_delta = self.execute_react(
        ag, prev_ctx, skill_input, max_steps,
    )
    ...
finally:
    if ag.sandbox is not None and ag.sandbox._gpu_id is not None:
        resource_pool.release_gpu(ag.sandbox._gpu_id)
    if not ag.is_external_sandbox and ag.sandbox is not None:
        ag.sandbox.stop(commit=True)
```

GPU semaphores and CPU/memory limits are returned even if the skill raises an exception or `max_steps` is exceeded.

## `agResourcePool` API

```python
pool = agResourcePool(
    gpus=None,             # list[int] or None for auto-detect
    total_cpus=None,       # int or None for auto-detect
    total_memory_mb=None,  # int or None for auto-detect
    idle_cpus=0.5,         # CPU limit when idle (0.5 = half a core)
    idle_memory="512m",    # memory limit when idle
    mark_gpus=False,       # launch a marker process per GPU (see below)
)

gpu_id = pool.acquire_gpu()  # blocks until a GPU is free; called internally by exec()
pool.release_gpu(gpu_id)
```

## GPU presence markers

When `mark_gpus=True` is passed to `agResourcePool`, one background process is launched per GPU at pool construction time. Each process:

- Sets its name to `agency-gpu` via `prctl(PR_SET_NAME)` — visible in `ps` and `nvidia-smi`
- Allocates 128 MB of VRAM (`torch.zeros(32M, device='cuda:0')`) and sleeps
- Exits silently if `torch` or CUDA is unavailable

This makes it easy to identify which GPUs are managed by the framework in `nvidia-smi`. The default class-level pool (`agent.agresource_pool`) uses `mark_gpus=True`. Markers are terminated via `atexit` when the process exits.

```python
# Opt out (e.g. in tests or library use):
agent.agresource_pool = agResourcePool(mark_gpus=False)
```
