# agsync

`agsync` blocks until every in-flight agent task in a set of agents and/or teams has finished. It is the barrier primitive for parallel execution.

## Import

```python
from agency import agsync
```

## Signature

```python
agsync(targets: agent | agteam | list[agent | agteam]) -> None
```

## What it waits on

For each item in `targets`:

| Input type | What is resolved |
|---|---|
| `agent` | `agent.ctx` — the tail of the agent's context chain |
| `agteam` | the team's background thread (if `run()` was called) + `ctx` of every currently-tracked agent |

For an `agteam`, "currently tracked agents" means:

* Agents created in `setup()` — always tracked (the ContextVar is set during `setup()` too).
* Fork agents (`agent(parent)`) created anywhere inside `run()` — **automatically tracked** because `agteam` sets a context variable for the duration of `run()` and `agent.__init__` registers itself when that variable is set.

Agents that have already completed and have no other live reference are GC'd from the `WeakSet` automatically — `agsync` skips them since they are already done.

## Usage

### Single agent

```python
agsync(my_agent)
```

### Single team

```python
team = ResearchTeam(topic="KV cache")
team.run()
agsync(team)    # blocks until the workflow and all its agents finish
```

### Fan-out barrier

```python
topics  = ["KV cache", "flash attention", "speculative decoding"]
teams   = [ResearchTeam(topic=t) for t in topics]
pending = [t.run() for t in teams]   # all three start immediately

# Do other work while teams run…

agsync(teams)          # barrier — wait for all three teams
for r in pending:
    print(r.report_path)   # no blocking; all resolved
```

### Mixed agents and teams

```python
agsync([agent_a, team_b, team_c])
```

### Fire and forget (no return value needed)

```python
team = BuildTeam(output_path="/workspace/out")
team.run()       # starts background work, return value ignored
agsync(team)     # wait for completion
```

## Error behaviour

If a team's `run()` raised an exception, `agsync` re-raises it at the barrier. If multiple teams failed, the first exception encountered is raised; the rest are silently swallowed (their errors are still visible via the pending `agdata` objects returned by `run()`).

## Serializing sequential calls on a shared agent

When an `agteam` holds a persistent agent in `setup()` and calls it inside `run()`, concurrent `run()` invocations on the same team instance will race to register on that agent's history chain. Because `agteam.run()` is non-blocking, both invocations can submit their work and call the shared agent before either one finishes, producing a non-deterministic registration order that can deadlock (see [deadlock.md](deadlock.md)).

The fix is an `agsync` barrier between the two calls, placed so the first invocation fully completes — including resolving the shared agent's history — before the second one registers:

```python
class WriterTeam(agteam):
    def setup(self):
        self.feedback_team = FeedbackTeam(llm_config=self.llm_config)
        ...

    def run(self, scene_goal, design_doc, previous_scenes=""):
        # First call — must complete before the loop submits the second call,
        # because both calls share feedback_team.main_feedback (a persistent agent).
        feedback_doc = self.feedback_team.run(
            section_goal=scene_goal, design_doc=design_doc,
            previous_scenes=previous_scenes,
        )
        agsync(self.feedback_team)   # barrier: wait for all agent histories to resolve

        while True:
            plan_doc      = planner.run(self.plan_skill, agdata(feedback=feedback_doc, ...))
            current_draft = writer.run(self.write_skill, agdata(plan=plan_doc, ...))

            # Safe to submit now: agsync guarantees the first call has fully registered
            # and resolved on the shared agent before this second call registers.
            review = self.feedback_team.run(
                section_goal=scene_goal, design_doc=design_doc,
                previous_scenes=previous_scenes, draft=current_draft,
            )
            if review.is_good_enough:
                break
            feedback_doc = review
```

`agsync` is the right primitive here — not field access on the pending result — because it also waits for all tracked agent contexts to resolve, not just the team's background thread. The shared agent's context is what the second invocation chains on.

## Relationship to field access on pending agdata

Both `agsync(team)` and reading a field on the pending `agdata` returned by `run()` will block until the team finishes:

| | field access on pending `agdata` | `agsync(team)` |
|---|---|---|
| Blocks until | `run()` thread completes | `run()` thread + all tracked agent contexts |
| Re-raises exception | yes, on first access | yes |
| Use when | you want a specific result field | you want a hard barrier without caring about the return value |

For most use cases they are equivalent. `agsync` is more thorough because it also resolves contexts of fork agents that may have been created inside `run()` but whose results were never returned.

## How dynamic agent tracking works

`agteam` uses a `contextvars.ContextVar` (`_active_team`) that is set to the current team instance for the duration of both `setup()` and `run()`. `agent.__init__` checks this variable and adds `self` to `_active_team._agents` (a `WeakSet`) if it is set. Because `contextvars` is thread-aware, this works correctly when teams run in their own daemon threads — each team's thread has its own context.

```
team.run() called
  → background thread starts, _active_team = team
  → user code runs
      → agent(parent)  ← __init__ sees _active_team, adds self to team._agents
      → agent(parent)  ← same
  → _active_team reset to None
agsync(team) joins the thread, then resolves all tracked agent contexts
```
