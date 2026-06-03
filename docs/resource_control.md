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

Containers are started with `--gpus all` when GPUs are present on the host, so the NVIDIA device files exist inside the container. However, every `exec()` call unconditionally exports `CUDA_VISIBLE_DEVICES=""` when no GPU is held, making all GPUs invisible to CUDA regardless of what the command does. This prevents any code — including code that never reads `CUDA_VISIBLE_DEVICES` — from accessing GPUs without going through `gpu_acquire`.

When `gpu_acquire` succeeds, `CUDA_VISIBLE_DEVICES=<id>` is injected into every subsequent exec call for that sandbox until `gpu_release` clears it.

## GPU semaphores

Each GPU ID gets a `threading.Semaphore(1)`. `acquire_gpu()` spins across all semaphores until one is free, then returns the GPU ID and sets `sandbox._gpu_id`. `release_gpu()` releases the semaphore. Acquisition with a timeout raises `TimeoutError` if no GPU becomes free in time.

## CPU and memory limits

`cpu_acquire` calls `docker update --cpus=N --memory=Mg` on the live container, adjusting Linux cgroup limits without restarting. `0.5` CPUs means the container is throttled to at most half a core's worth of CPU time — it can see all cores but is rate-limited at the cgroup level. `cpu_release` resets to the idle defaults.

Idle defaults (`idle_cpus=0.5`, `idle_memory="512m"`) are restored by `release_resources()` when a skill completes.

## Agent-callable resource tools

| Tool | Parameters | Effect |
|---|---|---|
| `gpu_acquire` | `timeout` (optional, seconds) | Blocks until a GPU is free; `CUDA_VISIBLE_DEVICES=<id>` set for all subsequent bash calls |
| `gpu_release` | — | Returns the GPU to the pool; `CUDA_VISIBLE_DEVICES` reset to `""` |
| `cpu_acquire` | `cpus` (float), `memory` (string, e.g. `"8g"`) | Boosts container resource limits |
| `cpu_release` | — | Resets limits back to idle defaults (shown in tool description) |
| `daemon_release` | `pid` (int) | Removes a PID from monitoring — skill completes without waiting for it |

The agent calls these tools itself during a skill, just like any other tool. `daemon_release` is always in the tool list; GPU/CPU tools are added only when `agent.agresource_pool` is set (the default).

## Release guarantee

`sandbox.release_resources(pool)` is always called in the `finally` block of `agent._task()`:

```python
try:
    # outer monitoring loop ...
finally:
    self.sandbox.release_resources(pool)
```

GPU semaphores and CPU/memory limits are returned even if the skill raises an exception or hits `max_outer_iters`.

## `agResourcePool` API

```python
pool = agResourcePool(
    gpus=None,             # list[int] or None for auto-detect
    total_cpus=None,       # int or None for auto-detect
    total_memory_mb=None,  # int or None for auto-detect
    idle_cpus=0.5,         # CPU limit when idle (0.5 = half a core)
    idle_memory="512m",    # memory limit when idle
)

gpu_id = pool.acquire_gpu(timeout=60.0)  # blocks; raises TimeoutError on timeout
pool.release_gpu(gpu_id)
```
