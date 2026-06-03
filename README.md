# Agency

A sandboxed multi-agent framework. Each agent runs inside its own Docker container with isolated filesystem, GPU access control, and automatic background-process tracking.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — package manager
- Docker (or Podman)

## Install

```bash
git clone https://github.com/agency-project/agency
cd agency

# Build the sandbox base image (once)
docker build -t agency-sandbox:latest docker/

# Install the package in development mode
uv pip install -e .

# Install dev dependencies (for tests)
uv pip install -e ".[dev]"
```

## Quick start

```python
from agency import agent, agskill, agdata

summarise = agskill(
    name="summarise",
    system_prompt="Summarise the given text in one sentence.",
    input_schema=agdata(text="str"),
    output_schema=agdata(summary="str"),
    tools=[],
)

ag = agent(
    llm_config={
        "base_url": "http://localhost:8000/v1",
        "api_key":  "EMPTY",
        "model":    "meta-llama/Llama-3.1-8B-Instruct",
    },
    agskills=[summarise],
)

result = ag.run("summarise", agdata(text="The quick brown fox jumps over the lazy dog."))
print(result.summary)   # blocks until done
```

## Examples

All examples connect to an OpenAI-compatible endpoint. Set the environment variables for your server before running.

| Variable | Default | Description |
|---|---|---|
| `VLLM_BASE_URL` | `https://kimi.js-park.info:18000/v1` | API endpoint |
| `VLLM_API_KEY` | _(empty)_ | API key |
| `VLLM_MODEL` | `moonshotai/Kimi-K2.6` | Model name |

### base_example — file I/O + shared history

Demonstrates writing and reading files inside the sandbox, and passing conversation history between two skills.

```bash
python examples/base_example.py

# Custom endpoint
VLLM_BASE_URL=http://localhost:8000/v1 VLLM_MODEL=llama3 python examples/base_example.py
```

### parallel_exec — sequential chain and fork fan-out

Demonstrates the two parallelism patterns:
- **Sequential chain** — two calls on one agent, automatically ordered via the history chain
- **Fork fan-out** — multiple `agent(parent).run()` calls fire concurrently

```bash
python examples/parallel_exec.py
```

### paper_crawler — parallel summarisation + report

Searches arXiv for papers, summarises each in parallel using forked agents, then compiles a markdown report.

```bash
python examples/paper_crawler.py
python examples/paper_crawler.py "speculative decoding"
MAX_PAPERS=6 python examples/paper_crawler.py "flash attention"
```

The report is written to `runs/<timestamp>_paper_crawler/agent_output/<agname>/report.md`.

## Running tests

```bash
pytest
```

The test suite requires Docker. Tests that spin up containers are marked `@pytest.mark.docker` and skipped automatically if Docker is unavailable.

## Docs

Detailed engineering documentation is in `docs/`:

| File | Topic |
|---|---|
| `agent.md` | Agent construction, `run()`, forking, class-level config |
| `container.md` | Sandbox container — lifecycle, GPU access, output dir, exec wrapper, PID tracking |
| `execution_loop.md` | Full execution path — outer monitoring loop, inner ReAct loop, input/output handling |
| `execution_process_control.md` | Per-scenario trace: background job, foreground job, daemon |
| `skills.md` | Skills — ReAct loop, schemas, validation, retries |
| `tools.md` | Built-in tools, sandboxed factories, tool logging, `daemon_release` |
| `resource_control.md` | GPU/CPU/memory resource pool |
| `logging.md` | Structured log — skill entries, tool entries, lifecycle events |
