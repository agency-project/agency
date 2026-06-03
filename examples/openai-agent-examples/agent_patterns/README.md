# Agent Pattern Examples

Ports of the `agent_patterns/` examples from openai-agents-python.

---

## agents_as_tools.py

**Original pattern:** Sub-agents are exposed as callable tools via `agent.as_tool(...)`. The orchestrator LLM decides which translator to call and the framework executes the sub-agent synchronously as a tool invocation.

**Port:** Each translator is wrapped in a plain `agtool` whose `fn` creates a fresh sub-agent, calls `run()`, and blocks on the result. The tool returns the translation as `agdata`. From the orchestrator LLM's perspective it is just a function call — identical behaviour to the original.

```
agent.as_tool(tool_name=..., tool_description=...)
→ agtool(fn=lambda arg: agent(...).run("translate", agdata(...)).translation)
```

**Run:**
```bash
python examples/openai-agent-examples/agent_patterns/agents_as_tools.py
```

---

## routing.py

**Original pattern:** A triage agent detects the language and performs a `handoff` — transferring control and conversation history to a language-specific agent mid-run.

**Port:** Handoffs are not native to this framework. Instead, the triage is a skill that returns the detected language; the caller then dispatches to the matching skill on the same agent. History accumulates naturally because all skills share the agent's conversation context.

```
triage_agent.handoffs=[french_agent, spanish_agent, english_agent]
→ triage_skill returns {language: "french"} → ag.run("french", ...)
```

**Gap:** In the original, the handoff happens *inside* a single `Runner.run()` call — the triage agent can hand off mid-turn without returning to the caller. In our port, the caller drives the dispatch explicitly between two `ag.run()` calls.

**Run:**
```bash
python examples/openai-agent-examples/agent_patterns/routing.py
```

---

## parallelization.py

**Original pattern:** `asyncio.gather()` runs the same agent three times concurrently.

**Port:** Two equivalent patterns are shown:

**Async** — `asyncio_run()` + `asyncio.gather()`, directly equivalent to the original:
```python
r1, r2, r3 = await asyncio.gather(
    agent(parent).asyncio_run("translate", agdata(text=msg)),
    agent(parent).asyncio_run("translate", agdata(text=msg)),
    agent(parent).asyncio_run("translate", agdata(text=msg)),
)
```

**Sync** — fork fan-out without asyncio; `run()` returns pending `agdata` immediately and all three resolve concurrently in the thread pool:
```python
pending = [agent(parent).run("translate", agdata(text=msg)) for _ in range(3)]
translations = [r.translation for r in pending]  # each blocks until its fork finishes
```

**Run:**
```bash
python examples/openai-agent-examples/agent_patterns/parallelization.py
```

---

## deterministic.py

**Original pattern:** A fixed pipeline — outline → checker (gate) → story — each step a separate `Runner.run()` call with explicit conditional logic between steps.

**Port:** Direct translation — three sequential `ag.run()` calls on the same agent with Python `if` gates between them. History accumulates across all three steps so the story agent has full context.

```
outline_result = await Runner.run(outline_agent, ...)
checker_result = await Runner.run(checker_agent, outline_result.final_output)
if not checker_result.final_output.good_quality: exit()
story_result = await Runner.run(story_agent, ...)

→ r1 = ag.run("outline", agdata(prompt=...))
  r2 = ag.run("check", agdata(outline=r1.outline))
  if not r2.good_quality: raise SystemExit
  r3 = ag.run("story", agdata(outline=r1.outline))
```

**Run:**
```bash
python examples/openai-agent-examples/agent_patterns/deterministic.py
```

---

## llm_as_a_judge.py

**Original pattern:** Generator and evaluator run in a loop. The evaluator's feedback is appended to the input list for the next generator iteration.

**Port:** Same loop in plain Python. The evaluator feedback is fed back as part of the next `generate` skill's input. History accumulates on the agent so the generator always has full context.

**Run:**
```bash
python examples/openai-agent-examples/agent_patterns/llm_as_a_judge.py
```

---

## input_guardrails.py

**Original pattern:** `@input_guardrail` decorator runs a check in parallel to the main agent. If it trips, `InputGuardrailTripwireTriggered` is raised before the main agent produces output.

**Port:** No native guardrail hooks. The classifier runs as a regular skill before the main skill. The `if check.is_math_homework` branch catches the trip and returns a refusal message.

```
@input_guardrail async def math_guardrail(...) → GuardrailFunctionOutput
→ check = ag.run("check_input", agdata(message=...))
  if check.is_math_homework: print("Sorry...")
  else: ag.run("support", agdata(message=...))
```

**Gap:** In the original the guardrail runs *in parallel* with the main agent and can abort it mid-flight. In our port the check is sequential — the main skill only starts after the guardrail passes. This is slightly slower but functionally equivalent for most use cases.

**Run:**
```bash
python examples/openai-agent-examples/agent_patterns/input_guardrails.py
```

---

## output_guardrails.py

**Original pattern:** `@output_guardrail` runs after the agent produces output. If it trips, `OutputGuardrailTripwireTriggered` is raised.

**Port:** `output_validator=` on `agskill` is the native equivalent — it runs after the LLM produces a final answer and before the skill resolves. If validation fails and retries are exhausted, `AgError` is raised by field access on the result.

```
@output_guardrail async def sensitive_data_check(...) → GuardrailFunctionOutput
→ agskill(output_validator=lambda r: ["error msg"] if "650" in r.response else [])
```

**Run:**
```bash
python examples/openai-agent-examples/agent_patterns/output_guardrails.py
```
