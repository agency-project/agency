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

### `_startup_semaphore` — container startup concurrency
| | |
|---|---|
| **File** | `agency/agsandbox.py:37` |
| **Type** | `threading.Semaphore(8)` |
| **Resource** | Number of Docker/Podman containers starting simultaneously |
| **Acquisition** | `with _startup_semaphore:` inside `agSandbox._ensure_started()` (line 199) |
| **Release** | On context-manager exit |
| **Timeout** | None — the NVIDIA runtime serializes GPU init internally, so >8 concurrent launches add contention without reducing wall-clock time |

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

## Synchronization Events

### `ready` / `done` — TUI startup and shutdown
| | |
|---|---|
| **File** | `agency/agui.py:547–548` |
| **Type** | `threading.Event` |
| **Resource** | Ordering between the main Textual UI thread and the worker thread that runs user code |
| **Protocol** | `ready.set()` in `_AgencyApp.on_mount()`; worker calls `ready.wait(timeout=10)` before starting. `done.set()` in `on_unmount()` to signal shutdown. |

---

## Queues (communication buffers, not rate limiters)

These are unbounded — they do not throttle resource usage but provide thread-safe message passing.

| Variable | File | Type | Purpose |
|---|---|---|---|
| `q` (SimpleQueue) | `agency/tools/human.py:43` | `queue.SimpleQueue[str]` | Shuttles console `input()` reply from reader thread to tool function |
| `reply_q` | `agency/agui.py:613` | `queue.Queue[str]` | Shuttles TUI human-reply from UI event handler to blocking tool call |
| `self._inbox` | `agency/agent.py:426` | `queue.Queue[str]` | Per-agent inbox for `agent.send()` mid-skill messages |

---

## Summary Table

| Primitive | File | Type | Slots | Resource |
|---|---|---|---|---|
| `_llm_call_semaphore` | agskill.py:288 | `Semaphore` | 128 | LLM API call concurrency |
| `_startup_semaphore` | agsandbox.py:37 | `Semaphore` | 8 | Container startup concurrency |
| `_gpu_locks[id]` | agresources.py:159 | `Semaphore(1)` per GPU | 1 per GPU | GPU exclusive ownership |
| `_res_lock` | agresources.py:162 | `Lock` | — | Resource counters |
| `_llm_config_lock` | agent.py:29 | `Lock` | — | LLM server round-robin counter |
| `_global_token_lock` | agent.py:346 | `Lock` | — | Global token counters |
| `_agname_lock` | agent.py:83 | `Lock` | — | Agent name uniqueness |
| `_pool_lock` | agtool.py:25 | `Lock` | — | ProcessPoolExecutor singleton |
| `agterm._lock` | agterm.py:72 | `Lock` | — | Terminal color assignment |
| `aglog._lock` | aglog.py:42 | `Lock` per instance | — | Log file I/O |
| `emitter._lock` | agwebui/emitter.py:56 | `Lock` per instance | — | Event file + registries |
| `server._lock` | agwebui/server.py:51 | `asyncio.Lock` | — | Web server event index |
| `ready` / `done` | agui.py:547 | `threading.Event` | — | TUI startup/shutdown sync |
| `self._inbox` | agent.py:426 | `Queue` (unbounded) | ∞ | Mid-skill agent inbox |
