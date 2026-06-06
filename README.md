# Agency

A multi-agent framework where each agent runs inside its own sandbox container with isolated filesystem, GPU access control, and automatic background-process tracking.

Agents are non-blocking by default. `agent.run()` returns a pending `agdata` immediately; reading any field on it blocks until the result is ready. Multiple agents run concurrently in a thread pool. Tool calls are offloaded to a process pool so CPU-bound work never blocks the main interpreter.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — package manager
- Docker or Podman

## Install

```bash
git clone https://github.com/agency-project/agency
cd agency

# Build the sandbox base image (once)
docker build -t agency-sandbox:latest images/
# or Podman:
podman build -t localhost/agency-sandbox:latest images/

uv pip install -e .
uv pip install -e ".[dev]"   # dev dependencies (pytest etc.)
```

## Quick start

```python
from agency import agent, agskill, agdata

summarise = agskill(
    name="summarise",
    system_prompt="Summarise the given text in one sentence.",
    input_schema=agdata(text=str),
    output_schema=agdata(summary=str),
)

ag = agent(
    llm_config={
        "base_url": "http://localhost:8000/v1",
        "api_key":  "EMPTY",
        "model":    "meta-llama/Llama-3.1-8B-Instruct",
    },
)

result = ag.run(summarise, agdata(text="The quick brown fox jumps over the lazy dog."))
print(result.summary)   # blocks until done
```

## Core concepts

**`agdata`** — a lightweight dict wrapper that travels between agents, skills, and tools. Fields are accessed as attributes (`result.summary`). Supports JSON serialisation and schema validation.

**`agtype`** — base class for typed agdata field values. Subclass to control how a schema field is serialised, transferred to/from the sandbox filesystem, represented in the system prompt, and cleaned up. `agfile` is the built-in subclass for file-backed fields.

**`agskill`** — a named ReAct loop with its own system prompt, optional input/output schemas, and an optional tool list. The LLM calls tools, inspects results, and iterates until it produces a final JSON answer. Output is validated against the schema; failures inject a correction message and retry.

**`agtool`** — a named callable an LLM can invoke via function calling. Every tool call is offloaded to a `ProcessPoolExecutor` worker so CPU-bound tools don't block other agents. Tools are serialised with `cloudpickle`, so bound methods work without any extra machinery.

**`agent`** — holds an LLM config, sandboxed tools, and a conversation history. `agent.run(skill, input)` accepts an `agskill` object directly and is non-blocking; the result is a pending `agdata` that resolves lazily. Sequential calls on the same agent are automatically serialised through the history chain. Forking via `agent(parent)` deep-copies the history and runs the new task concurrently.

**`agteam`** — coordinates multiple agents or tasks via a `ThreadPoolExecutor`. Subclass, define `setup()` to wire up agents and skills, override `run()` with your workflow.

**`agUI`** — a terminal UI (Textual) that shows all live agents, their current state, streaming token output, tool calls, and a human-in-the-loop interaction pane.

## Parallelism model

| Layer | Mechanism | Notes |
|---|---|---|
| Agents / team tasks | `ThreadPoolExecutor` (256 threads) | Threads release the GIL during LLM I/O |
| LLM streaming | Background drain thread + 100 ms batch queue | Reduces GIL acquisitions from O(tokens) to O(tokens/batch) |
| Tool execution | `ProcessPoolExecutor` (256 workers) | Each tool call gets its own GIL |

See [docs/parallelization.md](docs/parallelization.md) for the full design.

## Examples

All examples read LLM config from environment variables:

| Variable | Default | Description |
|---|---|---|
| `VLLM_BASE_URL` | `https://kimi.js-park.info:18000/v1` | API endpoint |
| `VLLM_API_KEY` | _(empty)_ | API key |
| `VLLM_MODEL` | `moonshotai/Kimi-K2.6` | Model name |

### base_example — file I/O and sandboxed tools

An agent writes a file inside its container, then reads it back. Demonstrates sandboxed tool use and typed skill schemas.

```bash
uv run python examples/base_example.py
```

### parallel_exec — sequential chain and fork fan-out

Two parallelism patterns side by side:

- **Sequential chain** — two `agent.run()` calls on the same agent; the second waits for the first automatically via the history chain
- **Fork fan-out** — `agent(parent).run()` creates an independent copy per input; all run concurrently, results resolve lazily

```bash
uv run python examples/parallel_exec.py
```

### paper_crawler — parallel summarisation pipeline

Searches arXiv for papers on a topic, summarises each in parallel with forked agents, then compiles a markdown report inside the sandbox.

```bash
uv run python examples/paper_crawler.py
uv run python examples/paper_crawler.py "speculative decoding"
MAX_PAPERS=6 uv run python examples/paper_crawler.py "flash attention"
```

The report lands at `runs/<timestamp>_paper_crawler/agent_output/<agname>/report.md`.

## Common skills

`agency.common_skills` provides ready-made skill classes:

| Class | What it does |
|---|---|
| `SummariserSkill` | One-sentence summary, no tools |
| `WriterSkill` | Writes content to a file path in the sandbox |
| `FindPapersSkill` | Searches Hugging Face Papers; returns title, URL, abstract |
| `SummarisePaperSkill` | Fetches full arxiv HTML and writes a technical summary |
| `CompileReportSkill` | Writes a structured markdown report to the sandbox |

## Running tests

```bash
pytest
```

Most tests mock the OpenAI client and run entirely in-process (no container needed). Tests that require a live container are marked and skipped if Docker/Podman is unavailable. Tool calls run through the real process pool in all tests — the same code path as production.

## Docs

| File | Topic |
|---|---|
| [parallelization.md](docs/parallelization.md) | Parallelism design — threads, processes, GIL, limitations |
| [agent.md](docs/agent.md) | Agent construction, `run()`, forking, history, UI callbacks |
| [skills.md](docs/skills.md) | ReAct loop, schemas, `agtype`/`agfile` typed fields, input offloading, validation, retries |
| [agtype.md](docs/agtype.md) | `agtype` interface — typed field values, `agfile`, custom subclasses |
| [tools.md](docs/tools.md) | Built-in tools, process offloading, sandboxed factories, `ask_human` |
| [agteam.md](docs/agteam.md) | Team coordination, `setup()` / `run()`, `agsync` |
| [container.md](docs/container.md) | Sandbox lifecycle, GPU access, exec wrapper, PID tracking |
| [execution_loop.md](docs/execution_loop.md) | Outer monitoring loop, inner ReAct loop, inbox drain, compaction |
| [execution_process_control.md](docs/execution_process_control.md) | Trace: background job, foreground job, daemon |
| [resource_control.md](docs/resource_control.md) | GPU/CPU/memory resource pool |
| [logging.md](docs/logging.md) | Structured JSONL log — skills, tools, lifecycle, compaction |
| [compaction.md](docs/compaction.md) | Auto-compaction — trigger, algorithm, incremental summaries |
| [ui.md](docs/ui.md) | Terminal UI — layout, keyboard bindings, interaction pane |
| [agsync.md](docs/agsync.md) | `agsync` — block until all pending agent results resolve |
| [deadlock.md](docs/deadlock.md) | Deadlock patterns — shared agents across parallel threads, diagnosis, and fixes |
