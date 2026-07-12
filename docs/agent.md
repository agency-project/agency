# Agent

The `agent` class is the top-level orchestrator. It manages a sandbox container lifecycle and a shared conversation history. Skills are passed directly to `run()` and always run asynchronously; the caller blocks only when it reads a result field.

## Construction

`agent` takes its LLM config exclusively via `agconfig=` — there is no `llm_config=` parameter. Build an `agConfig` with `agConfig(agLLMBackendConfig(...))` before constructing — see [`agconfig.md`](agconfig.md) for why this is always the form, even for a single owner's fields:

```python
from agency import agent, agskill
from agency.agconfig import agConfig
from agency.agllm_backend import agLLMBackendConfig

cfg = agConfig(agLLMBackendConfig(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="meta-llama/Llama-3.1-8B-Instruct",
))

ag = agent(agconfig=cfg)
```

The `agconfig` is passed to every skill run. Any OpenAI-compatible endpoint works via `cfg.agllm_backend.base_url`. See [`agllm.md`](agllm.md) for the full field reference.

No sandbox is created at construction time. Within a task, a sandbox is started lazily — only when a tool with `run_in_subprocess=True` is first called. Tasks that use only host-side tools never create a sandbox at all. When a sandbox is started, it is committed to a checkpoint image (`agency/ckpt-<pid>-<agname>`) when the task completes and then destroyed.

An optional `context_limit` field pins the model's context window size for auto-compaction. If omitted, the agent queries the endpoint at startup (vLLM exposes `max_model_len`). Compaction is silently disabled when the limit cannot be determined.

```python
# explicit override — useful for non-vLLM backends
cfg.agllm_backend.context_limit = 131072
```

The agent clones whatever `agconfig` it's given at construction time, and `ag.llm` (and its backend) clones it again — so `ag.agconfig`, `ag.llm._agconfig`, and `ag.llm.backend._agconfig` are independent copies, not the same object. Mutating `cfg` (or even `ag.agconfig`) after construction does **not** reach `ag.llm`. This is what stops two agents built from the same `cfg` from silently affecting each other — see [Design_configuration.md](Design_configuration.md) for the full rationale.

To change the LLM config live, call `ag.change_config(new_cfg)`:

```python
new_cfg = agConfig(agLLMBackendConfig(model="claude-sonnet-5", api_key="..."))
ag.change_config(new_cfg)
```

This clones `new_cfg` and pushes that clone down through `ag.llm` (and its backend), `ag.log`, and `ag.sandbox` — the very next LLM call sees it. `ag.get_config_copy()` returns a clone of the agent's current `agconfig` (or `None` if it has none) — useful for building a modified `new_cfg` from the agent's live settings:

```python
cfg = ag.get_config_copy()
cfg.agllm_backend.temperature = 0.2
ag.change_config(cfg)
```

Inside an `agteam`, agents created with no explicit `agconfig=` automatically inherit the team's `agconfig` — see [`agteam.md`](agteam.md).

## Running a skill

```python
skill = agskill(name="answer", system_prompt="Answer the question.")
result = ag.run(skill, agdata(question="What is 2+2?"))
# non-blocking — result is a pending agdata
print(result.answer)   # blocks here until the skill finishes
```

`run()` is always non-blocking. It calls `skill.run(self, ...)`, which schedules a thread and returns a pending `agdata` immediately. The actual ReAct loop is driven by `agskill.execute_react()`, which runs synchronously inside that thread. The result resolves only when the full skill — including all background processes the agent may have launched — has finished and all resources have been released.

## Serialized history

Calls on the **same** agent are automatically serialized: each `run()` chains on the previous ctx future, so history is always consistent even under concurrent callers.

```python
r1 = ag.run(search_skill, agdata(query="..."))
r2 = ag.run(summarize_skill, agdata(text=r1.text))   # waits for r1 internally
```

## Forking

```python
child = agent(ag)
```

Forking blocks until the parent's in-flight task completes, then deep-copies the resolved ctx and copies the parent's checkpoint image via `docker tag`. The child's sandbox is not started at fork time — it is created lazily when the child's first `run()` executes, restoring from the copied checkpoint. All subsequent writes in either direction are isolated.

## Class-level configuration

Set once before creating agents:

| Variable | Default | Meaning |
|---|---|---|
| `agent.log_dir` | `None` | Directory for per-agent JSONL logs |
| `agent.output_dir` | `None` | Shared output directory mounted into every sandbox |
| `agent.agresource_pool` | auto-detected | Shared GPU/CPU/memory pool |
| `agent.ping_interval_s` | `300` | Max seconds `wait_for_processes` waits before injecting a status ping |
| `agent.poll_interval_s` | `5` | `get_live_pids()` poll granularity inside each ping window |
| `agent.max_outer_iters` | `144` | **Unused** — kept for backwards compatibility; process monitoring is now bounded by `AGSKILL_REACT_MAX_STEPS` inside `agskill.execute_react()` |

## Shared output directory

When `agent.output_dir` is set, every sandbox gets a per-agent subdirectory mounted at `/agent_output/<agname>`:

```python
agent.output_dir = Path("runs/agent_output")
ag = agent(...)

# inside sandbox: write to /agent_output/agent_smith/report.md
# on host:          runs/agent_output/agent_smith/report.md

# access the paths
ag.container_output_path   # → "/agent_output/agent_smith"
ag.output_path             # → Path("runs/agent_output/agent_smith")
```

See [agsandbox.md](agsandbox.md) for mount implementation details.

## Agent naming

Each agent is assigned a unique pronounceable name (adjective + noun, e.g. `swift_hawk`) if none is provided. Pass `agname` to use a specific base name:

```python
ag = agent(agconfig=cfg, agname="worker")
# ag.agname == "worker_0000"
```

The name is always postfixed with `_XXXX` (a 4-character base-36 counter, digits `0-9` then `a-z`) to guarantee global uniqueness for the process lifetime. The first agent with a given base name gets `_0000`, the tenth `_000a`, the 36th `_0010`, and so on. The 4-character suffix supports 36⁴ = 1 679 616 unique values per base name.

## Lifecycle and cleanup

sandboxs are created lazily — only when a task calls a tool with `run_in_subprocess=True` for the first time. Tasks that use only host-side tools (web fetch, `ask_human`, paper search, …) complete without ever starting a sandbox. When a sandbox is started, stale sandboxs from a previous run (e.g. after a hard kill) are removed first. The sandbox is stopped with `ag.sandbox.stop(commit=True)` — a single call that snapshots and stops the container — at task end. An `atexit` handler removes any sandboxs still running at process exit.

## UI callbacks

Three internal attributes are available for custom monitoring:

| Attribute | Type | Updated |
|---|---|---|
| `ag._state` | `agent_state` | After every state transition (`inactive`, `skill`, `llm`, `tool`, `proc_wait`, `human`, `paused`, `blocked_on_dependency`, `finished`, `error`) — see [Pause and resume](#pause-and-resume) |
| `ag._snapshot_messages` | `list[dict]` | After every LLM response and tool result within a skill |
| `ag.inbox` | `queue.Queue[str]` | Drain to inject a user message before the next LLM call |

`ag._state` (an `agent_state` instance, defined in `agent.py`) is more than a display value — it's also the single source of truth pause/resume coordination reads from: `.state`/`.skill`/`.tool` are what the webui shows, while `.run_allowed`/`.paused_ack` (both `threading.Event`) and `.blocked_on` (an `agent | None`) are real synchronization primitives. The only way to change the display fields is `agent_state.update_state(new_state, skill=None, tool=None, *, unless_in=())`, which applies the whole read-check-write atomically under one lock — never mutate `.state`/`.skill`/`.tool` directly.

## Pause and resume

```python
ag.pause()                  # request a stop at the next safe checkpoint
ag.is_paused()              # True once it actually stopped (not merely requested)
ag.resume()                 # clear the request

from agency import wait_all_paused, wait_all_resumed
wait_all_paused([ag])       # block until settled (see below), or until timeout=
wait_all_resumed([ag])
```

`pause()` is non-blocking and takes effect only at the next ReAct-loop checkpoint (`agent._check_pause()`, called at the top of every iteration in `agskill.execute_react()`, and once before the loop starts) — never mid-LLM-call or mid-tool-call, so an in-flight call always finishes first. `is_paused()` reads `ag._state.paused_ack` directly (the real synchronization primitive), not the cosmetic state string.

`is_settled()` is what makes `wait_all_paused()` deadlock-safe across a dependency chain: an agent whose worker thread is blocked resolving another agent's still-pending result (e.g. via a nested `agdata` field, or `agent.fork()`) is tagged `blocked_on_dependency` with `ag._state.blocked_on` pointing at the upstream agent, and `is_settled()` recurses through that chain — an agent blocked on an already-*paused* upstream counts as settled, so waiting for a whole dependency graph to stop never hangs on an agent that can never reach its own checkpoint. This is unrelated to the future-chain registration-order deadlocks covered in `docs/Design_deadlock.md` — see `agency/agpause.py` for the pause-specific cross-agent coordination (thread-local worker tracking, `wait_all_paused`/`wait_all_resumed`).
