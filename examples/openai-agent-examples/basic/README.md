# Basic Examples

Ports of the `basic/` examples from openai-agents-python. These cover the simplest patterns.

## hello_world.py

**Original pattern:** Instantiate one agent with a system instruction, run it once, print the output.

**Port:** One `agskill` with `system_prompt`, `tools=[]`. A single `ag.run()` call blocks until the skill resolves.

```
Agent(instructions=...) + Runner.run(agent, input)
→ agskill(system_prompt=...) + ag.run(skill, agdata(...))
```

**Run:**
```bash
python examples/openai-agent-examples/basic/hello_world.py
```

---

## tools.py

**Original pattern:** Agent is given a `@function_tool`-decorated Python function. The LLM calls it as a JSON function, the framework executes it, and the result is fed back.

**Port:** `agtool(name, description, fn, params)` replaces `@function_tool`. The `fn` receives an `agdata` (from the LLM's JSON arguments) and returns an `agdata`. The skill's `tools=[get_weather]` adds the weather tool on top of any agent-level tools already available.

```
@function_tool def get_weather(...) → Weather
→ agtool(name="get_weather", fn=lambda arg: agdata(...), params={...})
```

**Run:**
```bash
python examples/openai-agent-examples/basic/tools.py
```
