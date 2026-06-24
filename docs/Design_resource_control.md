# Design: Resource Control

Every concurrency primitive in the framework — semaphores, locks, events, and queues — is listed here with the resource it guards, how it is acquired, and any timeout or fairness behavior.

---

## Semaphores

### `_llm_call_semaphore` — LLM API concurrency
| | |
|---|---|
| **File** | `agency/agskill.py:288` |
| **Type** | `threading.Semaphore(128)` |
| **Resource** | Number of simultaneous LLM API calls across all agents |
| **Acquisition** | `_llm_call_semaphore_slot()` context manager (line 291); wraps every `_llm_call()` invocation |
| **Release** | On context-manager exit (always, including on exception) |
| **Timeout** | None — callers block indefinitely until a slot is free |

### `_docker_semaphore` — Docker/Podman daemon call throughput
| | |
|---|---|
| **File** | `agency/agsandbox.py:44` |
| **Type** | `threading.Semaphore(16)` |
| **Resource** | All Docker/Podman subprocess calls — held for the duration of every `_run()` invocation |
| **Acquisition** | `with _docker_semaphore:` inside `_run()` |
| **Release** | On context-manager exit (always, including on exception) |
| **Timeout** | None — callers block until a slot is free; the subprocess itself is bounded by per-operation `timeout=` arguments in each `_run()` call |
| **Replaces** | The former `_startup_semaphore` / `_commit_semaphore` / `_shutdown_semaphore` trio. A single semaphore covering all daemon calls is simpler and avoids the deadlock class that arose at the boundary between commit and shutdown. |

The Docker/Podman daemon serializes most operations internally (GPU init via the NVIDIA container runtime, overlay diff, container teardown), so more than ~16 concurrent calls increase contention without reducing wall-clock time.

### `_container_semaphore` — simultaneously running container cap
| | |
|---|---|
| **File** | `agency/agsandbox.py:84` |
| **Type** | `multiprocessing.Semaphore(_docker_container_limit())` |
| **Resource** | Number of simultaneously running Docker containers, derived from the Linux kernel session-keyring quota |
| **Acquisition** | `_container_semaphore.acquire()` inside `_ensure_started()`, immediately before `docker run` |
| **Release** | `_container_semaphore.release()` inside `stop()` — always, including when `docker rm -f` fails after retries |
| **Timeout** | None — blocks until a slot is free |

**Why `multiprocessing.Semaphore`:** backed by a POSIX IPC semaphore (not an in-process counter), so the limit is shared across all worker processes spawned by `ProcessPoolExecutor`. Worker processes inherit the parent's `_RUN_ID` at import time and therefore share container names with the parent; a cross-process counter prevents them from collectively exceeding the kernel keyring limit.

**`_docker_container_limit()`** reads `/proc/sys/kernel/keys/maxkeys` at module import and returns `max(4, maxkeys − 5)`, leaving a 5-slot margin for external tools (ssh, sudo, gpg). Podman is exempt: rootless Podman uses independent user-namespace keyrings and is not subject to this quota.

**Why each running container consumes a keyring slot.** Every `docker run` invocation creates a Linux session keyring under the calling user's UID. The kernel enforces a per-user cap (`/proc/sys/kernel/keys/maxkeys`, typically 200). When the cap is reached, the next `docker run` fails with `"unable to create session key: disk quota exceeded"`. Stopping or removing a container releases its keyring immediately.

**Zombie container problem.** `_container_semaphore` counts only THIS process's containers. If a previous process crashed without cleanup, its containers remain running and consume keyring slots outside the semaphore's accounting. If N zombie containers from a prior run are alive, the effective free slots are `maxkeys − (semaphore_limit + N)`, which can reach 0 even when `_container_semaphore` has available slots.

Mitigation: before starting a new run, remove containers from prior runs:
```
docker ps -aq --filter "name=sandbox-" | xargs -r docker rm -f
```
Automatic startup cleanup was considered and rejected: it would destroy containers belonging to a *concurrently running* agency process that shares the same user account (container names for different runs use different `_RUN_ID` prefixes, but a blanket `rm -f` on `sandbox-*` cannot distinguish them without knowing the other run's `_RUN_ID`).

**Diagnostics.**
- `keyring_quota()` — reads `/proc/sys/kernel/keys/maxkeys` and `/proc/keys` and returns `{"used": int, "max": int, "free": int}`. Reflects ALL running containers across all processes, including zombies.
- `_semaphore_held_count()` — returns `"held/limit"` from the semaphore's internal POSIX counter. Reflects only THIS process's containers. Comparing `keyring_quota()["used"]` with the held count reveals how many keyring slots are held by zombie containers from other processes.

### `_gpu_locks[gpu_id]` — per-GPU ownership
| | |
|---|---|
| **File** | `agency/agresources.py:159` |
| **Type** | `dict[int, threading.Semaphore(1)]` — one binary semaphore per GPU index |
| **Resource** | Exclusive ownership of one physical GPU |
| **Acquisition** | Non-blocking poll inside `acquire_gpu()` (line 221): `sem.acquire(blocking=False)`; the method loops over all GPU semaphores sleeping 0.25 s between passes until one succeeds or a deadline is exceeded |
| **Release** | `pool.release_gpu(gpu_id)` calls `sem.release()` (line 237); guarded by a `try/except ValueError` to prevent crashes on accidental double-release |
| **Timeout** | Optional `timeout` parameter on `acquire_gpu()`; raises `TimeoutError` if deadline passed |

---

## Timeouts (Docker subprocess operations)

All Docker/Podman subprocess calls go through `_run()` in `agency/agsandbox.py`, which passes a `timeout=` to `subprocess.run()`. Each operation uses a constant tuned to its expected duration:

| Constant | Value | Operations |
|---|---|---|
| `_TIMEOUT_INSPECT` | 10 s | `docker inspect`, `nvidia-smi`, `podman info`, runtime detection |
| `_TIMEOUT_EXEC_QUICK` | 5 s | Quick `docker exec` calls (PID snapshot, baseline diff) |
| `_TIMEOUT_DOCKER_RUN` | 120 s | `docker run` — GPU initialization via the NVIDIA container runtime serializes across concurrent containers and can take 60+ s under load |
| `_TIMEOUT_DOCKER_RM` | 30 s | `docker rm -f` — overlay teardown is usually fast but can stall under daemon load |
| `_TIMEOUT_FILE_IO` | 30 s | `docker cp` file transfers in and out of the container |
| `_TIMEOUT_IMAGE` | 15 s | `docker images`, `docker rmi` |
| `_TIMEOUT_COMMIT` | 120 s | `docker commit` — writing a new image layer can be slow for large `/workspace` trees |
| `_TIMEOUT_KEYRING_WAIT` | 120 s | Maximum time to wait for a keyring slot in any retry path before abandoning a `docker run` attempt |

`_TIMEOUT_DOCKER_RUN` and `_TIMEOUT_DOCKER_RM` are deliberately separate: `docker run` can be slow due to GPU init while `docker rm -f` should be fast, and the former timeout used to subsume both (causing premature rm failures when the run path was slow).

---

## Locks

### `_res_lock` — resource counter integrity
| | |
|---|---|
| **File** | `agency/agresources.py:162` |
| **Type** | `threading.Lock` |
| **Resource** | Counters `_gpus_acquired`, `cpus_acquired`, `memory_acquired_mb` |
| **Acquisition** | `with self._res_lock:` around all counter increments and decrements in `acquire_gpu`, `release_gpu`, `acquire_resources`, `release_resources` |

### `_llm_config_lock` — round-robin server selection
| | |
|---|---|
| **File** | `agency/agent.py:29` |
| **Type** | `threading.Lock` |
| **Resource** | Global counter `_llm_config_counter` used to distribute calls across a list of LLM server configs |
| **Acquisition** | `with _llm_config_lock:` inside `_pick_llm_config()` (line 37) |

### `_global_token_lock` — global token counters
| | |
|---|---|
| **File** | `agency/agent.py:346` |
| **Type** | `threading.Lock` (class variable) |
| **Resource** | `_global_input_tokens` and `_global_output_tokens` |
| **Acquisition** | Writer: `_add_global_tokens()` (line 350); reader: `global_token_usage()` (line 364) |

### `_agname_lock` — agent name allocation
| | |
|---|---|
| **File** | `agency/agent.py:83` |
| **Type** | `threading.Lock` |
| **Resource** | `_noun_counters` and `_allocated_agnames` — ensures no two agents get the same name |
| **Acquisition** | `_register_agname()` (line 102), `_allocate_agname()` (line 114) |

### `_pool_lock` — process pool singleton
| | |
|---|---|
| **File** | `agency/agtool.py:25` |
| **Type** | `threading.Lock` |
| **Resource** | `_pool` global (`ProcessPoolExecutor(max_workers=256, mp_context="spawn")`) |
| **Acquisition** | Lazy init in `_get_pool()` (line 31); reset to `None` in tool execution error handler when `BrokenProcessPool` is caught (line 171) |

### `agterm._lock` — terminal color assignment
| | |
|---|---|
| **File** | `agency/agterm.py:72` |
| **Type** | `threading.Lock` (class variable) |
| **Resource** | `_color_counter` and `_agname_colors` — maps agent names to ANSI color codes |
| **Acquisition** | `agterm.__init__()` (color assignment) and `agterm.log()` (UI dispatch) |

### `aglog._lock` — structured log file I/O
| | |
|---|---|
| **File** | `agency/aglog.py:42` |
| **Type** | `threading.Lock` (per instance) |
| **Resource** | `_entries` list, `_events` list, and JSON file writes |
| **Acquisition** | `_record()`, `_tool_call()`, `_lifecycle()`, `entries` property, `token_usage` property, `events` property |

### `agwebui_emitter._lock` — event file and registries
| | |
|---|---|
| **File** | `agency/agwebui/emitter.py:56` |
| **Type** | `threading.Lock` (per instance) |
| **Resource** | JSONL event file writes, `_token_state`, `_agent_registry`, `_team_registry` |
| **Acquisition** | `emit()`, `agent_registered()`, `team_registered()`, `token_update()`, `done()` |

### `server._lock` — web server event index (async)
| | |
|---|---|
| **File** | `agency/agwebui/server.py:51` |
| **Type** | `asyncio.Lock` — created in the FastAPI lifespan handler (line 68) |
| **Resource** | `_sparse_index`, `_file_offset`, `_file_size`, `_first_ts/_last_ts`, `_clients`, `_agent_registry`, `_team_registry` |
| **Acquisition** | `async with _lock:` in `/api/timeline`, `/api/events`, `websocket_endpoint`, and the `_tail_events` background loop |

---

## Queues (communication buffers, not rate limiters)

These are unbounded — they do not throttle resource usage but provide thread-safe message passing.

| Variable | File | Type | Purpose |
|---|---|---|---|
| `q` (SimpleQueue) | `agency/tools/human.py:43` | `queue.SimpleQueue[str]` | Shuttles console `input()` reply from reader thread to tool function |
| `self._inbox` | `agency/agent.py:426` | `queue.Queue[str]` | Per-agent inbox for `agent.send()` mid-skill messages |

---

## Summary Table

| Primitive | File | Type | Slots | Resource |
|---|---|---|---|---|
| `_llm_call_semaphore` | agskill.py:288 | `threading.Semaphore` | 128 | LLM API call concurrency |
| `_docker_semaphore` | agsandbox.py:44 | `threading.Semaphore` | 16 | Docker/Podman daemon call throughput |
| `_container_semaphore` | agsandbox.py:84 | `multiprocessing.Semaphore` | `maxkeys − 5` | Simultaneously running containers (keyring quota) |
| `_gpu_locks[id]` | agresources.py:159 | `threading.Semaphore(1)` per GPU | 1 per GPU | GPU exclusive ownership |
| `_res_lock` | agresources.py:162 | `Lock` | — | Resource counters |
| `_llm_config_lock` | agent.py:29 | `Lock` | — | LLM server round-robin counter |
| `_global_token_lock` | agent.py:346 | `Lock` | — | Global token counters |
| `_agname_lock` | agent.py:83 | `Lock` | — | Agent name uniqueness |
| `_pool_lock` | agtool.py:25 | `Lock` | — | ProcessPoolExecutor singleton |
| `agterm._lock` | agterm.py:72 | `Lock` | — | Terminal color assignment |
| `aglog._lock` | aglog.py:42 | `Lock` per instance | — | Log file I/O |
| `emitter._lock` | agwebui/emitter.py:56 | `Lock` per instance | — | Event file + registries |
| `server._lock` | agwebui/server.py:51 | `asyncio.Lock` | — | Web server event index |
| `self._inbox` | agent.py:426 | `Queue` (unbounded) | ∞ | Mid-skill agent inbox |
