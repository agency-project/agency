# Agency

Agency is a Python framework for running AI agents in containers and combining
them into workflows. Use the same task and data interface to switch agent
harnesses or models, run agents in parallel, and inspect their work.

## Quick Start

You need **Linux x86-64**, **Python 3.12+**, [uv](https://docs.astral.sh/uv/),
a working local **Docker or Podman** runtime, and an OpenAI API key.
Run Agency on the Linux host where the containers run.

```bash
git clone https://github.com/agency-project/agency.git
cd agency
uv sync --python 3.12

export OPENAI_API_KEY="your-api-key"
AGENCY_HARNESS=native uv run python examples/01_basic_agent.py
```

This runs one agent that summarizes a topic and prints a `summary:` line,
the result as JSON, and the artifact directory. It uses the tutorial's OpenAI
`gpt-6-luna` model; model calls use your API account. The first run pulls the
Python sandbox image and installs its dependencies, so allow extra time.
The native harness is included; no separate agent CLI is needed.

To use another model or provider, see [configuration](../guides/configuration.md).

## What You Can Do

- Run coding tasks with shell commands, file search, and file edits in a container.
- Choose native, Claude Code, Codex, Grok Build, Kimi Code, or OpenCode harnesses.
- Use OpenAI, Anthropic, Bedrock, or an OpenAI-compatible model endpoint.
- Run multiple agents, pass results between tasks, and save or restore agents.
- Pause, resume, cancel, and inspect executions in a browser.

## Basic Usage

A **skill** defines a task and its input/output fields. An **agent** runs it and
keeps conversation context between calls. Save this as `first_agent.py`:

```python
import os

from agency import Agent, agdata, agskill
from agency.configs.agconfig import agconfig, llmconfig

cfg = agconfig(llmconfig(
    provider="openai",
    base_url="https://api.openai.com/v1",
    model="gpt-6-luna",
    api_key=os.environ["OPENAI_API_KEY"],
    reasoning_effort="none",
    max_completion_tokens=4096,
))

summarize = agskill(
    name="summarize",
    prompt="Summarize the supplied topic in one sentence.",
    input_schema=agdata(topic=str),
    output_schema=agdata(summary=str),
)

worker = Agent("writer", agconfig=cfg)
result = worker.run(summarize, agdata(topic="sandboxed coding agents"))
print(result.summary)
```

Run it from the checkout with `uv run python first_agent.py`.
`run()` returns immediately; reading a result field waits for completion.
Calls on the same agent run in order. Different agents can run concurrently:

```python
workers = [Agent(agconfig=cfg) for _ in range(2)]
results = [w.run(summarize, agdata(topic=t)) for w, t in zip(
    workers, ["code review", "test generation"]
)]
print([r.summary for r in results])
```

For an existing `worker` and its `result`:

| Call | Use |
| --- | --- |
| `result.is_pending()` | Check whether work is still pending. |
| `result.wait(timeout=300)` | Wait explicitly; async code can use `await result`. |
| `worker.queue_message("Keep answers brief")` | Add context for the next submission. |
| `worker.pause()` / `worker.resume()` | Pause or resume the agent. |
| `worker.cancel(result)` | Cancel that execution. |

## Configuration

Pass model settings through `agconfig(llmconfig(...))`, as above.
OpenAI backends require an explicit `base_url` and API key. Python agents use
the native harness and `python:3.12-slim` sandbox image by default; Docker or
Podman is detected automatically. No GPU or privileged host setup is needed
for the quick start.

The example scripts have their own environment settings, including
`AGENCY_HARNESS` and `AGENCY_LLM_MODEL`. These are tutorial conveniences;
your own scripts configure agents through Python.

## Next Steps

- [Examples](../../examples/README.md): tools, files, workflows, and checkpoints.
- [Configuration](../guides/configuration.md): models, harnesses, and project mounts.
- [View runs](../guides/profiling.md): open the dashboard and profiler.
