# Agent

The `agent` class is the top-level orchestrator. It owns a sandbox container, a shared conversation history, a set of skills, and a pool of tools. Skills are submitted by name and always run asynchronously; the caller blocks only when it actually reads a result field.

## Construction

```python
from src.agent import agent
from src.agskill import agskill

ag = agent(
    llm_config={"api_key": "...", "model": "claude-opus-4-8", "base_url": "..."},
    agskills=[skill_a, skill_b],
)
```

`llm_config` is passed unchanged to every skill run. Any OpenAI-compatible endpoint works via `base_url`.

## Running a skill

```python
result = ag.run("skill_name", agdata(question="What is 2+2?"))
# non-blocking — result is a pending agdata
print(result.answer)   # blocks here until the skill finishes
```

`run()` is always non-blocking. It returns a pending `agdata` immediately and resolves only when the full skill — including all background processes the agent may have launched — has finished and all resources have been released.

## Serialized history

Calls on the **same** agent are automatically serialized: each `run()` chains on the previous one's history future, so history is always consistent even under concurrent callers.

```python
r1 = ag.run("search", agdata(query="..."))
r2 = ag.run("summarize", agdata(text=r1.text))   # waits for r1 internally
```

## Forking

```python
child = agent(ag)   # or ag.fork()
```

Forking blocks until the parent's in-flight task completes, then deep-copies the resolved history and snapshots the parent's container via `docker commit`. The child starts from the parent's exact filesystem state. All subsequent writes in either direction are isolated. Forked agents run concurrently.

## Class-level configuration

Set once before creating agents:

| Variable | Default | Meaning |
|---|---|---|
| `agent.log_dir` | `None` | Directory for per-agent JSONL logs |
| `agent.output_dir` | `None` | Shared output directory mounted into every container (see below) |
| `agent.agresource_pool` | auto-detected | Shared GPU/CPU/memory pool |
| `agent.ping_interval_s` | `300` | Max seconds between process-status re-entries |
| `agent.poll_interval_s` | `5` | Liveness poll granularity within each ping window |
| `agent.max_outer_iters` | `144` | Safety cap on outer loop iterations (~12 hours at 5-min intervals) |

## Shared output directory

When `agent.output_dir` is set, every container gets a shared volume mounted at `/agent_output`:

```python
agent.output_dir = Path("runs/agent_output")
```

All agents share the same `/agent_output` directory with full read-write access. Files written there appear immediately on the host at `agent.output_dir/`.

```python
agent.output_dir = Path("runs/agent_output")
# inside container: write to /agent_output/report.md
# on host:          runs/agent_output/report.md
```

See [container.md](container.md) for mount implementation details.

## Lifecycle

`__del__` calls `sandbox.destroy()` as a best-effort cleanup. For deterministic cleanup, call `ag.sandbox.destroy()` explicitly.
