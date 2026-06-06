# Agent

The `agent` class is the top-level orchestrator. It owns a sandbox container, a shared conversation history, and a pool of tools. Skills are passed directly to `run()` and always run asynchronously; the caller blocks only when it reads a result field.

## Construction

```python
from agency import agent, agskill

ag = agent(
    llm_config={
        "base_url": "http://localhost:8000/v1",
        "api_key":  "EMPTY",
        "model":    "meta-llama/Llama-3.1-8B-Instruct",
    },
)
```

`llm_config` is passed to every skill run. Any OpenAI-compatible endpoint works via `base_url`.

An optional `"context_limit"` key in `llm_config` pins the model's context window size for auto-compaction. If omitted, the agent queries the endpoint at startup (vLLM exposes `max_model_len`). Compaction is silently disabled when the limit cannot be determined.

```python
# explicit override — useful for non-vLLM backends
llm_config = {..., "context_limit": 131072}
```

## Running a skill

```python
skill = agskill(name="answer", system_prompt="Answer the question.")
result = ag.run(skill, agdata(question="What is 2+2?"))
# non-blocking — result is a pending agdata
print(result.answer)   # blocks here until the skill finishes
```

`run()` is always non-blocking. It returns a pending `agdata` immediately and resolves only when the full skill — including all background processes the agent may have launched — has finished and all resources have been released.

## Serialized history

Calls on the **same** agent are automatically serialized: each `run()` chains on the previous history future, so history is always consistent even under concurrent callers.

```python
r1 = ag.run(search_skill, agdata(query="..."))
r2 = ag.run(summarize_skill, agdata(text=r1.text))   # waits for r1 internally
```

## Forking

```python
child = agent(ag)
```

Forking blocks until the parent's in-flight task completes, then deep-copies the resolved history and snapshots the parent's container via `docker/podman commit`. The child starts from the parent's exact filesystem state. All subsequent writes in either direction are isolated. Forked agents run concurrently.

## Class-level configuration

Set once before creating agents:

| Variable | Default | Meaning |
|---|---|---|
| `agent.log_dir` | `None` | Directory for per-agent JSONL logs |
| `agent.output_dir` | `None` | Shared output directory mounted into every container |
| `agent.agresource_pool` | auto-detected | Shared GPU/CPU/memory pool |
| `agent.ping_interval_s` | `300` | Max seconds between process-status re-entries |
| `agent.poll_interval_s` | `5` | Liveness poll granularity within each ping window |
| `agent.max_outer_iters` | `144` | Safety cap on outer loop iterations (~12 hours at 5-min intervals) |

## Shared output directory

When `agent.output_dir` is set, every container gets a per-agent subdirectory mounted at `/agent_output/<agname>`:

```python
agent.output_dir = Path("runs/agent_output")
ag = agent(...)

# inside container: write to /agent_output/agent_smith/report.md
# on host:          runs/agent_output/agent_smith/report.md

# access the paths
ag.container_output_path   # → "/agent_output/agent_smith"
ag.output_path             # → Path("runs/agent_output/agent_smith")
```

See [container.md](container.md) for mount implementation details.

## Agent naming

Each agent is assigned a unique pronounceable name (adjective + noun, e.g. `swift_hawk`) if none is provided. Pass `agname` to use a specific base name:

```python
ag = agent(llm_config, agname="worker")
# ag.agname == "worker_000"
```

The name is always postfixed with `_XXX` (a 3-character base-36 counter, digits `0-9` then `a-z`) to guarantee global uniqueness for the process lifetime. The first agent with a given base name gets `_000`, the tenth `_00a`, the 36th `_010`, and so on. The 3-character suffix supports 36³ = 46 656 unique values per base name.

## Lifecycle and cleanup

Containers are destroyed automatically on process exit via `atexit`. Stale containers from a previous run (e.g. after a hard kill) are removed at startup before the new container is created. Call `ag.sandbox.destroy()` explicitly for deterministic cleanup.

## UI callbacks

Three internal callbacks are used by `agUI` but are also available for custom monitoring:

| Attribute | Type | Updated |
|---|---|---|
| `ag._ui_state` | `dict` | After every state transition (`inactive`, `skill`, `llm`, `tool`, `proc_wait`, `human`) |
| `ag._snapshot_messages` | `list[dict]` | After every LLM response and tool result within a skill |
| `ag._inbox` | `queue.Queue[str]` | Drain to inject a user message before the next LLM call |
