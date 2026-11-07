# Agency

Run, control, and inspect coding agents from Python.

Agency is a Python library for developers building coding workflows and
researchers comparing agent behavior. Give an agent a task, let it work in
its own container, and read the result from your Python program. Use it when
you want to automate code changes, coordinate several agents, or understand
what happened during an execution.

## Why use Agency?

- **Compose tasks in Python.** Define input and output fields with a skill,
  keep context across calls, and pass results between agents.
- **Choose the execution parts.** Use Agency's included native harness or
  supported external CLI harnesses with separately configured model backends.
- **Inspect and control work.** Keep execution logs, pause or resume agents,
  and open a run in the browser dashboard and profiler.

A *harness* is the program that executes the agent's task. Start with native;
external CLIs have additional installation requirements. Agency provides
container execution and tools for shell commands, file search, and edits.
Your script decides what task to submit and when to use its result.

## Quick start

You need **Linux x86-64**, **Python 3.12+**, Git,
[uv](https://docs.astral.sh/uv/getting-started/installation/), and a working
local **Docker or Podman engine**. Run the Python driver on the same Linux
host as the containers. The Python Podman client installed as a dependency
does not install the engine or CLI. Check `docker info` or `podman info`
as your user before starting.

Install this repository from source:

```bash
git clone https://github.com/agency-project/agency.git
cd agency
uv sync --locked --python 3.12

export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-6-luna"
uv run python examples/quickstart.py
```

The example uses OpenAI Chat Completions with function tools and
`reasoning_effort="none"`. You need API access to the selected model;
requests use your account and incur its normal charges. `OPENAI_MODEL` is
optional and defaults to the model shown. Overrides must accept these
settings. The script reads the exported key explicitly and does not load
`.env` automatically.

The first run pulls a Python sandbox image and installs missing dependencies
inside it. Allow extra time and outbound network access. This example disables
GPU passthrough and needs no GPU, ZFS, CRIU, or host provisioning. See
[getting started](../getting-started.md) for installation checks and the
most likely setup failures.

## Basic usage

A *skill* defines the task and its input/output fields. This complete example
is the workload in [quickstart.py](../../examples/quickstart.py). Save it as
`first_agent.py` in the checkout and run `uv run python first_agent.py`:

```python
import os
from pathlib import Path

from agency import Agent, agdata, agskill
from agency.configs.agconfig import agconfig, llmconfig, resourcesconfig, sandboxconfig

cfg = agconfig(
    llmconfig(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model=os.environ.get("OPENAI_MODEL", "gpt-6-luna"),
        api_key=os.environ["OPENAI_API_KEY"],
        reasoning_effort="none",
        max_completion_tokens=4096,
    ),
    sandboxconfig(gpu_passthrough=False),
    resourcesconfig(idle_cpus=1),
)
task = agskill(
    name="code_and_test",
    prompt="Complete the request in /workspace. Run the code, then summarize the result.",
    input_schema=agdata(request=str),
    output_schema=agdata(summary=str),
)
worker = Agent("coder", agconfig=cfg, harness="native")
result = worker.run(
    task,
    agdata(request="Write sum_even(numbers) in Python and run assertions for empty and mixed lists."),
)
result.wait()
print(result.summary)
print(f"Run directory: {Path(worker.data_logger.db_path).parent.parent}")
```

`run()` returns a pending result immediately. `wait()` blocks until it resolves;
reading `summary` retrieves the output or raises an error if execution failed.
The function and assertions run inside `/workspace` in the container. The
script prints the model's summary and a host run directory containing logs.
Agency cleans up the sandbox on process exit; the logs remain available.

## Next steps

- [Examples](../../examples/README.md): tools, context, parallel workflows, and files.
- [Profiler startup](../guides/profiling.md): run this example with `--webui`
  and reopen completed traces.
- [Documentation](../index.md): configuration, providers, project mounts,
  and existing API and architecture references.
- [Contributing](../../CONTRIBUTING.md): set up development, run checks, and find
  a place to start.
