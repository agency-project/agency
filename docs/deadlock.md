# Deadlock patterns

This document explains how deadlocks can arise when future-producing objects — `agent` instances, `agteam` instances, or pending `agdata` values — are shared across threads running in parallel, and how to avoid them.

## The general rule

A deadlock can occur whenever the **same future-producing object**, usually `agent` instance, `agteam` instance, or pending `agdata` value, is accessed from **two or more threads running in parallel**, and those threads have a data dependency between them.

The threads in question are typically:

- Two `agteam.run()` bodies executing concurrently in the pool.
- The main thread and an `agteam.run()` body executing concurrently.
- Two separate agteam instances whose run() bodies both reference the same external agent or agdata.

The common thread (no pun intended): a shared object is called or read from multiple concurrent threads in a non-deterministic order, creating a registration on a future chain that is inverted relative to the actual data dependency. When the data dependency forms a cycle with the inverted chain, nothing can make progress.

## Background: how agent history serialization works

Every `agent` instance maintains a `_history` future chain. Each call to `agent.run()` reads the current tail of the chain as `prev_history` and writes a new pending future as the new tail:

```
agent.run() call 1:  prev_history = empty   →  _history = hf1
agent.run() call 2:  prev_history = hf1     →  _history = hf2
agent.run() call 3:  prev_history = hf2     →  _history = hf3
```

Each task waits for `prev_history` to resolve before starting its LLM call. This serializes calls on the same agent in **registration order** — the order in which `agent.run()` was actually called.

This mechanism is safe as long as `agent.run()` is only called from a single thread at a time. When multiple threads call `agent.run()` on the same instance concurrently, the registration order is non-deterministic. If the resulting order inverts a data dependency, a deadlock forms.

## Why data dependencies do not prevent the race

`agent.run()` is non-blocking. It accepts pending `agdata` objects as inputs and resolves them lazily inside the task — not at the call site. So a thread can call `agent.run(skill, agdata(x=pending_result))` and return in microseconds, before `pending_result` has resolved. The registration on the history chain happens at call time, not at execution time.

The thread pool has no awareness of agdata dependency chains. It dispatches tasks as threads become free, regardless of whether a task's inputs are ready.

---

## Example 1: same agteam instance called twice before the first completes

This is the most common pattern. A `WriterTeam` holds a `FeedbackTeam` in `setup()`. The `FeedbackTeam` has a persistent `self.main_feedback` agent. `WriterTeam.run()` calls `self.feedback_team.run()` twice: once before the write loop for initial feedback, and once inside the loop to review the draft.

```python
class WriterTeam(agteam):
    def setup(self):
        self.feedback_team = FeedbackTeam(llm_config=self.llm_config)
        ...

    def run(self, scene_goal, design_doc, previous_scenes=""):
        feedback_doc  = self.feedback_team.run(...)          # FT1 — submitted to pool
        plan_doc      = planner.run(agdata(feedback=feedback_doc, ...))
        current_draft = writer.run(agdata(plan=plan_doc, ...))
        review        = self.feedback_team.run(draft=current_draft, ...)  # FT2 — submitted to pool
```

All four submissions happen before any result arrives. FT1 and FT2 are both in the pool queue. If FT2's body executes first:

```
FT2 body:  self.main_feedback.run(...)  →  prev=empty,  _history=hf2
FT1 body:  self.main_feedback.run(...)  →  prev=hf2,    _history=hf1
```

The deadlock cycle:

```
FT1's main_feedback task  →  waits on hf2 (FT2's history)
hf2 resolves when         →  FT2's main_feedback task completes
FT2's inputs depend on    →  current_draft → writer → planner → feedback_doc
feedback_doc is           →  FT1's result future
FT1's result future set when  →  FT1's main_feedback task completes
                                    └─ which is waiting on hf2  ← CYCLE
```

**Fix:** `agsync` barrier between the two calls.

```python
feedback_doc = self.feedback_team.run(...)
agsync(self.feedback_team)   # wait for FT1 including all agent histories

# FT2 is only submitted after FT1 has fully registered and resolved.
review = self.feedback_team.run(draft=current_draft, ...)
```

---

## Example 2: agent shared between two separate agteam instances

An agent is created outside any team and passed to two teams that both run concurrently. Each team's `run()` body calls `shared.run()` from its own pool thread.

```python
shared = agent(llm_config=LLM_CONFIG, agname="Shared")

class TeamA(agteam):
    def run(self):
        result = self.shared.run(skill_a, agdata(x="input_a"))
        return result

class TeamB(agteam):
    def run(self):
        # TeamB's result depends on TeamA's output
        a_result = self.a_result          # pending agdata produced by TeamA
        return self.shared.run(skill_b, agdata(x=a_result))

a = TeamA(shared=shared)
b = TeamB(shared=shared, a_result=a.run())   # b depends on a's result

b.run()   # both pool threads race to call shared.run()
```

If TeamB's body executes first:

```
TeamB body:  shared.run(skill_b, ...)  →  prev=empty,  _history=hfB
TeamA body:  shared.run(skill_a, ...)  →  prev=hfB,    _history=hfA
```

TeamA's task waits on `hfB`. `hfB` resolves only when TeamB's task finishes. TeamB's task is waiting on `a_result`, which is TeamA's result future. TeamA's result is set when TeamA's task completes. TeamA's task is waiting on `hfB`. Cycle.

**Fix:** ensure TeamA completes before TeamB is submitted.

```python
a = TeamA(shared=shared)
r = a.run()
agsync(a)           # TeamA fully done — shared._history resolved

b = TeamB(shared=shared, a_result=r)
b.run()             # TeamB registers on shared after TeamA
```

---

## Example 3: main thread and agteam sharing an agent

The main thread calls `agent.run()` and then submits an agteam that also calls the same agent — but the agteam's pool thread can execute before the main thread's first call has registered.

```python
shared = agent(llm_config=LLM_CONFIG, agname="Shared")

class SummaryTeam(agteam):
    def run(self):
        # Expects to run after the outline is ready
        return self.shared.run(summary_skill, agdata(outline=self.outline))

outline = shared.run(outline_skill, agdata(topic="KV cache"))   # main thread
team = SummaryTeam(shared=shared, outline=outline)
team.run()   # pool thread may call shared.run() before the main thread's call above registers
```

The main thread calls `shared.run(outline_skill, ...)` which sets `_history = hf_outline`. Then `team.run()` is submitted. If the pool thread picks up team's body immediately and reaches `self.shared.run(summary_skill, ...)` before the main thread's task has resolved `hf_outline`:

```
pool thread:  shared.run(summary_skill, ...)  →  prev=hf_outline, _history=hf_summary
```

Here the registration order is actually correct (main thread registered first), so no deadlock forms. But if the main thread had not yet called `shared.run()` at all before submitting the team — for example, because the outline result was itself a pending agdata passed in — the pool thread could register on `shared` before the main thread does, inverting the chain.

**Safer pattern:** call `agsync(shared)` after the main thread's agent call before submitting any team that uses the same agent.

```python
outline = shared.run(outline_skill, agdata(topic="KV cache"))
agsync(shared)   # wait for shared._history to resolve

team = SummaryTeam(shared=shared, outline=outline)
team.run()
```

---

## Summary

| Shared object | Parallel contexts | Risk |
|---|---|---|
| `agent` in `agteam.setup()` | same instance's `run()` called twice concurrently | inverted history chain → deadlock if data dependency exists |
| `agent` passed to two agteam instances | both `run()` bodies access it concurrently | same |
| `agent` used on main thread and inside agteam | pool thread may register before main thread | same if order is inverted |
| pending `agdata` passed to two concurrent agteams | both bodies receive the same unresolved future | safe to read; deadlock only if a circular data dependency is separately introduced |

## Avoiding the pattern

**Preferred:** create agents locally inside `run()` rather than in `setup()`. Each call gets its own independent history chain with no shared state to race on.

**When persistent agent history across calls is intentional:** use `agsync` as an explicit ordering barrier. Place it after the first call and before any submission that could reach the shared agent concurrently. `agsync` waits for the team's background thread **and** all tracked agent histories to resolve — field access on a pending `agdata` alone is not sufficient because it does not resolve agent history futures.
