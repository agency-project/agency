# OpenAI Agents Examples — Ported to Agency

Ports of the [openai/openai-agents-python](https://github.com/openai/openai-agents-python/tree/main/examples) examples, rewritten using this framework. The goal is to verify which patterns are directly supported, which require workarounds, and which are not yet possible.

## Concept mapping

| OpenAI agents | Agency |
|---|---|
| `Agent(name, instructions, tools)` | `agent(llm_config, agskills=[agskill(name, system_prompt, tools)])` |
| `Runner.run(agent, input)` | `ag.run(skill_name, agdata(...))` — sync, blocks on field access |
| `@function_tool` | `agtool(name, description, fn, params)` |
| `output_type=SomeModel` | `output_schema=agdata(...)` + `output_validator` |
| `handoffs=[...]` | Explicit routing skill + skill dispatch by caller |
| `@input_guardrail` | Classifier skill run sequentially before main skill |
| `@output_guardrail` | `output_validator=` on `agskill` |
| Parallelization via `asyncio.gather` | Fork fan-out: `[agent(parent).run(...) for ...]` |
| `asyncio.run(main())` | Plain synchronous code — no asyncio needed |

## Running

Set environment variables for your LLM endpoint:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=EMPTY
export VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

Then run any example:

```bash
python examples/openai-agent-examples/basic/hello_world.py
python examples/openai-agent-examples/agent_patterns/parallelization.py
```

---

## Framework gaps

### 1. Native handoffs

**OpenAI agents:** An agent can hand off control to another agent *mid-run* via `handoffs=[agent_b]`. The triage agent calls a handoff tool; the framework transparently switches execution to the target agent, carrying full conversation history. This happens inside a single `Runner.run()` call — the caller is unaware.

**Agency:** No mid-run handoff mechanism. The caller must explicitly drive dispatch: run the triage skill, inspect the output, then call the appropriate skill. This is functionally equivalent but the routing logic lives in user code rather than inside the agent.

**Impact:** Multi-level triage trees (triage → sub-triage → specialist) require more orchestration code.

---

### 2. No streaming

**OpenAI agents:** `Runner.run_streamed()` yields a live stream of events — text deltas, tool call starts, tool results, handoff notifications — as they occur token by token.

**Agency:** `ag.run()` is non-blocking (returns a pending `agdata` immediately) but the result only resolves when the entire skill is complete. There is no incremental token stream. Terminal output from the sandbox (e.g. `tail -f log`) can be read via `sandbox.exec()` inside a monitoring loop, but there is no LLM token-level streaming.

**Impact:** No live typing effect in UIs. Long LLM generations are invisible to the caller until they finish.

---

### 3. No built-in tracing

**OpenAI agents:** `with trace("name"):` wraps a block in a named span, linking all agent runs, tool calls, and handoffs within it into a single observable trace. Integrates with OpenTelemetry-compatible backends.

**Agency:** `aglog` records every skill call, tool invocation, and lifecycle event to a per-agent JSONL file. This is post-hoc structured logging, not live distributed tracing. There is no span hierarchy, no trace IDs, and no OpenTelemetry export.

**Impact:** Cross-agent correlation (e.g. linking a fork's tool calls back to the parent run) requires manual log correlation by UUID or agname.

---

### 4. ~~Synchronous only (no async)~~ — **Closed**

**OpenAI agents:** Fully async — `await Runner.run(...)`. Works in async web frameworks (FastAPI, aiohttp) and Jupyter notebooks.

**Agency:** `ag.run()` uses a thread pool for real concurrency. `ag.asyncio_run()` is an async wrapper that makes it awaitable from any asyncio context without blocking the event loop:

```python
# Single await
result = await ag.asyncio_run("summarise", agdata(text=text))

# Parallel execution — equivalent to asyncio.gather in the original
r1, r2, r3 = await asyncio.gather(
    agent(parent).asyncio_run("translate", agdata(text=msg)),
    agent(parent).asyncio_run("translate", agdata(text=msg)),
    agent(parent).asyncio_run("translate", agdata(text=msg)),
)
```

The underlying execution still uses threads (not coroutines), but from the caller's perspective `asyncio_run()` is fully awaitable and composes correctly with `asyncio.gather`, FastAPI endpoints, and async Jupyter cells.

---

### 5. Parallel input guardrails

**OpenAI agents:** `@input_guardrail` runs *in parallel* with the main agent. If the guardrail trips, the main agent is aborted mid-flight — no tokens are wasted finishing a response that will be refused.

**Agency:** The classifier skill runs *sequentially* before the main skill. The main skill only starts after the guardrail passes. Functionally equivalent, but slightly slower because the check and the main run are serialised rather than overlapped.

**Impact:** Latency: the total time is `guardrail_time + main_time` instead of `max(guardrail_time, main_time)`.

---

### 6. No RunContextWrapper

**OpenAI agents:** A `RunContextWrapper` object is threaded through the entire run — accessible in tools, guardrails, and hooks. It carries arbitrary user-defined context (e.g. database connections, user IDs) without needing global state.

**Agency:** No equivalent. Tools capture their dependencies via closures at construction time (e.g. `make_bash(sandbox)` captures the sandbox). Shared state between tools in the same run must be managed externally.

**Impact:** Patterns that need per-request context (e.g. injecting the current user's ID into every tool call) require manual closure plumbing.

---

### 7. No session / memory persistence

**OpenAI agents:** The `memory/` examples show Redis, SQLite, MongoDB, and file-backed session stores that persist conversation history across process restarts and multiple API requests.

**Agency:** Conversation history lives in `agent._history` (in-memory) and is logged to a JSONL file. There is no built-in mechanism to restore history from storage in a new process. Persistence requires reading the JSONL log and reconstructing the messages list manually.

**Impact:** Stateful chatbots that survive server restarts require external state management.

---

### 8. No human-in-the-loop tool approval

**OpenAI agents:** The `human_in_the_loop` examples show a pattern where tool calls can be paused, serialised to disk, and resumed after a human approves or rejects the action. State is fully serialisable.

**Agency:** No built-in pause/resume mechanism. The outer monitoring loop re-enters the agent with `process_update` / `process_completed` events, but there is no way to interrupt an in-progress ReAct loop mid-tool-call for human approval.

**Impact:** Agentic workflows that require human sign-off before executing destructive tools (e.g. `rm -rf`, database writes) must implement approval logic at the skill level by having the agent ask for confirmation as part of its reasoning.
