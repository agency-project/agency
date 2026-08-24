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

Containers are started with every GPU on the host attached (`--gpus all` on Docker, `--device nvidia.com/gpu=all` CDI on Podman) **only once the sandbox has actually reserved one** (`reserve_gpu()` called before the container's first `run`) — a sandbox that never reserves a GPU gets zero GPU devices attached at all, not just hidden ones (see [sandbox/container.md](sandbox/container.md)'s "GPU device access" for why every GPU is attached rather than just the one currently leased). Every `exec()` call unconditionally exports `CUDA_VISIBLE_DEVICES=NoDevFiles` when no GPU is currently leased (`readonly`-exported, so a command can't hijack a different GPU by reassigning the variable inline), making all GPUs invisible to CUDA regardless of what the command does.

`reserve_gpu` sets only a virtual flag (`sandbox._gpu_virtual = True`) — no physical GPU is taken at that point. When `exec()` runs a bash command and the virtual flag is set but no GPU is currently held (`sandbox._gpu_id is None`), a physical GPU is claimed from the pool at that moment (blocking until one is free) and `CUDA_VISIBLE_DEVICES=<id>` is injected for that exec call and every one after it — **not released between individual exec() calls or between bash commands**; it's held across the whole sandbox's active period. If all GPUs are busy when a bash command runs, `exec()` blocks until one is free.

## GPU semaphores

Each GPU ID gets a `threading.Semaphore(1)`. `acquire_gpu()` is called internally by `exec()` — not by the `reserve_gpu` tool directly. `reserve_gpu` only sets `sandbox._gpu_virtual = True`. The physical semaphore is acquired at exec time: `acquire_gpu()` spins across all semaphores until one is free, then returns the GPU ID and sets `sandbox._gpu_id`. `release_gpu()` releases the semaphore. The physical GPU is released when: (a) `sandbox.stop()` (hibernate, between tool calls); (b) `sandbox.rm_container()` (skill failure); or (c) `sandbox.destroy()` — never mid-tool-call, and never just because a foreground exec finished or the live-PID set went empty.

## CPU and memory limits

`cpu_acquire` calls `docker update --cpus=N --memory=Mg` on the live container, adjusting Linux cgroup limits without restarting. `cpu_release` resets to the idle defaults (`pool.idle_cpus`/`pool.idle_memory`).

A fresh container is created with the same `--cpus=<idle_cpus> --memory=<idle_memory>` limits (`_ensure_started()` in `agsandbox.py`) — the resting-state footprint every sandbox gets before any `reserve_cpu` call, and the one it's reset back to afterward. `idle_cpus`/`idle_memory` are `DynamicConfigParam`s under the `agResourcePool` owner, defaulting to 4 CPUs / 4096m:

```python
from agency.agconfig import agConfig
from agency.agresources import agResourcePool, agResourcePoolConfig

# Constructor kwargs (convenience, equivalent to the agConfig form below):
pool = agResourcePool(idle_cpus=1.0, idle_memory="1024m")

# agConfig form -- composes with the rest of an agent's config. idle_cpus/
# idle_memory are Dynamic (not locked once read), but agResourcePool clones
# cfg at construction, so changing cfg afterward does not reach pool --
# call pool.change_config(new_cfg) instead:
cfg = agConfig(agResourcePoolConfig(idle_cpus=1.0, idle_memory="1024m"))
pool = agResourcePool(agconfig=cfg)
pool.change_config(agConfig(agResourcePoolConfig(idle_cpus=2.0)))   # live update, reaches pool immediately
```

`pool.get_config_copy()` returns a clone of the pool's current agconfig (handy as a base for building `new_cfg`). `pool.change_config` and `pool.get_config_copy` are the same pair of methods every other framework object with an `agconfig` exposes — see [Design_configuration.md](Design_configuration.md#changing-a-dynamic-field-live).

`idle_cpus`/`idle_memory` are read from the *sandbox's own* `agconfig` at container-creation time (a throwaway `_AgResourcePoolFields(self._agconfig)` instance in `_ensure_started()`) — that's the sandbox's own cloned `agconfig` (`sandbox._agconfig`), not necessarily `agent.agresource_pool`'s. Set them on the `agConfig` you pass to `agent(agconfig=...)` *before* the agent (and its sandbox) is constructed, or call `ag.sandbox.change_config(new_cfg)` once a sandbox already exists — mutating the original `cfg`/`agent.agresource_pool`'s agconfig afterward will not reach an already-built sandbox.

CPU and memory limits are set by the sandbox on each tool call via `update_limits()` and are not automatically restored by agresources. `release_resources()` is a manual call to reduce an `agSandbox`'s reported resource usage in the pool (e.g. when the sandbox is destroyed externally).

## Agent-callable resource tools

| Tool | Parameters | Effect |
|---|---|---|
| `reserve_gpu` | — | Reserves GPU access; a physical GPU is assigned lazily on the first bash call and held for the rest of the sandbox's lifetime. No parameters. |
| `reserve_cpu` | `cpus` (float), `memory` (string, e.g. `"8g"`) | Boosts container resource limits |
| `cpu_release` | — | Resets limits back to idle defaults (shown in tool description) |
| `daemon_release` | `pid` (int) | Removes a PID from monitoring — skill completes without waiting for it |

The agent calls these tools itself during a skill, just like any other tool. `daemon_release` is always in the tool list; GPU/CPU tools are added only when `agent.agresource_pool` is set (the default).

**There is no `gpu_release` tool.** A GPU, once actually acquired, is held for the duration of a single hibernate cycle — released by `sandbox.stop()` between tool calls, same as `rm_container()`/`destroy()` — never by an explicit mid-skill call. This ties the real GPU semaphore's release to the same cadence as the runtime slot rather than to the agent remembering to release it, at the cost of the agent never being able to explicitly signal "done with the GPU early" mid-tool-call.

## Release guarantee

GPU release is not a separate step — it happens *inside* `sandbox.stop()`/`sandbox.rm_container()`/`sandbox.destroy()` themselves. `AgentEngine.execute()` owns the final skill boundary while holding `sandbox._lock`:

```python
sandbox_lock.acquire()
try:
    execution = run_harness(...)
    if execution.ok:
        sandbox.commit()
        if not sandbox._has_pending_background_work():
            sandbox.stop()
    else:
        sandbox.rm_container()
finally:
    sandbox_lock.release()
```

`commit()` never touches the GPU or the runtime slot — the container isn't stopped or removed, so there's nothing to release. The GPU is therefore actually freed every time the sandbox hibernates between tool calls, not just when a skill fails.

`agResourcePool.release_gpu()` itself does no waiting or polling. Safety comes from the calling order above: teardown completes before resource release, and the engine releases `sandbox_lock` last even when execution or teardown raises.

## `agResourcePool` API

```python
pool = agResourcePool(
    gpus=None,                # list[int] or None for auto-detect
    total_cpus=None,          # int or None for auto-detect
    total_memory_mb=None,     # int or None for auto-detect
    idle_cpus=None,           # None -> DynamicConfigParam default, 4.0 -- also the starting limit
    idle_memory=None,         # None -> DynamicConfigParam default, "4096m" -- also the starting limit
    mark_gpus=False,          # launch a marker process per GPU (see below)
    agconfig=None,            # None -> a fresh, private agConfig
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
