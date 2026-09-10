# Agency

A multi-agent framework with sandboxed execution, isolated filesystems, GPU access control, and automatic background-process tracking. Submission is lazy: dependency-blocked, pause-gated, and host-only context-message requests create no engine or sandbox infrastructure before dispatch. A sandbox is created only after an engine-backed request is admitted, then its transaction is committed or discarded before output context and results are published.

Agents are non-blocking by default. `agent.run()` returns a pending `agdata` immediately. It can be awaited, passed into another skill, or read through attribute access; Agency privately retains the execution identity needed by `redirect()` and `cancel()`. A process-wide, event-driven orchestrator holds dependency-blocked work without occupying a worker or execution slot, then dispatches eligible work onto a reusable worker pool. Every dispatch still receives a fresh `AgentEngine`. Concurrency is unlimited by default and can be capped globally.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) — package manager ([installation](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer))
- Docker or Podman
- GPU (optional): NVIDIA (CUDA) or AMD (ROCm)

## Install

```bash
git clone https://github.com/agency-project/agency
cd agency

# Build the sandbox base image (once).
# Auto-detects the host GPU (NVIDIA / AMD / CPU-only):
./images/build.sh

uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate

uv pip install -e .
uv pip install -e ".[dev]"   # dev dependencies (pytest, ruff, pre-commit)

pre-commit install   # one-time; runs ruff (lint + format) and hygiene checks on every commit
```

The sandbox image installs `torch torchvision transformers datasets accelerate numpy scipy matplotlib`. The build script selects the NVIDIA, AMD, or CPU path automatically and runs an inline PyTorch smoke check.

## Quick start

Supply LLM backends to agents by building an `agconfig` object and passing it to the agents:

```python
cfg = agconfig(llmconfig(provider="vllm", model="...", api_key="..."))
ag = agent(agconfig=cfg)
```

`agconfig` is one typed object holding every tunable in the system, organized one level deep into small per-domain namespaces — `cfg.llm` (LLM fields), `cfg.sandbox` (sandbox fields), `cfg.orchestrator`, `cfg.agent`, and so on. Construction is positional and dispatches by type — `agconfig(llmconfig(...), sandboxconfig(...))` — so you never spell out a keyword name for the namespace; any namespace you don't pass gets a fresh, all-default instance. Set `provider` on `llmconfig` to pick your backend (`"vllm"`, `"openai"`, `"anthropic"`, `"bedrock"`, ...); only the fields relevant to that provider are read, and an unknown field name raises immediately (`TypeError` on construction, `AttributeError` on later assignment) so typos are caught right away. Need config for more than one thing, such as an LLM backend and a sandbox mount? Pass both namespace instances to the same `agconfig(...)` call (mounts are added via `cfg.sandbox.add_mount(...)`).

**OpenAI-compatible serving endpoint (vLLM, local, etc.)**

```python
from agency import agent, agskill, agdata
from agency.configs.agconfig import agconfig, llmconfig

continuation = agskill(
    name="continuation",
    system_prompt="Continue the sentence.",
    input_schema=agdata(text=str),
    output_schema=agdata(summary=str),
)

cfg = agconfig(
    llmconfig(
        provider="vllm",
        base_url="http://localhost:8000/v1", # Your serving API URL
        model="YOUR_SERVED_MODEL",
        api_key="YOUR_API_KEY" # Leave blank ("") if unused
    )
)

ag = agent(agconfig=cfg)

result = ag.run(continuation, agdata(text="Fly me to the moon and let me "))
print(result.summary)   # blocks until the execution succeeds or fails
```

## Submissions and lifecycle

A skill call returns a pending `agdata` result. Calls from the same agent execute in context order.

```python
first = ag.run(skill, agdata(topic="one"))
second = ag.run(other_skill, first)
ag.queue_message("Keep citations next to the claims they support")
third = ag.run(skill, agdata(topic="three"))
ag.redirect(third, "Use only primary sources")
third.wait()
```

`ag.redirect(result, message)` targets the skill execution that produced that result. Active Claude Code, Codex, Grok Build, and OpenCode harnesses accept redirects through their interactive PTYs. If the target has not started, has finished, or cannot accept the redirect, the message becomes `queue_message()` exactly once. Native keeps its existing non-PTY execution and queued redirects. A late redirect for an earlier result never interrupts a later run. See [redirect lifecycle](docs/Redirect.md) and [PTY architecture](docs/PTY.md).

`queue_message()` appends future context in submission order and returns `None`. It does not change already submitted work; the next submission after the enqueue receives that context.

## Agent and result API reference

`Agent` is the public alias of `agent`.

| API | Purpose |
|---|---|
| `ag.run(skill, skill_input, max_steps=None)` | Submit work and return its pending `agdata`. |
| `await ag.asyncio_run(skill, skill_input, max_steps=None)` | Submit work and await the resolved result. |
| `ag.redirect(result, message)` | Deliver to that active execution or queue future context. |
| `ag.queue_message(message)` | Enqueue ordered future context. |
| `ag.pause()` | Freeze the current harness process; pause future launches until resumed. |
| `ag.resume()` | Resume the harness and clear the persistent pause request. |
| `ag.is_paused()` | Report whether pause was requested. |
| `ag.cancel(result)` | Cancel the execution that produced that result. |
| `ag.context` | The agent's authoritative context chain. |
| `ag.history` | Read the committed transcript as `agdata`. |

### Agent configuration, cloning, and checkpoints

| API | Purpose |
|---|---|
| `ag.change_config(new_config)` | Replace the agent's live configuration with a clone of `new_config`. |
| `ag.get_config_copy()` | Return an independent copy of the current configuration. |
| `Agent.fork(source, agname=None)` | Create an independent agent from the source's resolved context, configuration, and sandbox snapshot. |
| `ag.save(path)` | Save one agent checkpoint. |
| `Agent.load(path, agconfig=None)` | Restore one agent checkpoint. |
| `Agent.save_all(directory)` | Save every live agent and return the checkpoint paths. |
| `Agent.load_all(directory, agconfig=None)` | Restore all checkpoints in a directory. |
| `Agent.all()` | Return all live agents in this process. |

Useful agent metadata includes `ag.agname`, `ag.harness`, `ag.output_path`, and `ag.container_output_path`. `ag.record_state(state, skill=None, tool=None)` is available for runtime logging and UI reporting.

### Results

A result is an awaitable `agdata`. Both `result.wait(timeout=None)` and `await result` resolve it. Field access, `to_dict()`, and `to_json()` wait automatically. `result.is_pending()` checks without blocking. Execution identity stays private and survives resolution, so the same result remains a valid redirect target afterward.

**OpenAI**

```python
from agency.configs.agconfig import agconfig, llmconfig

cfg = agconfig(
    llmconfig(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="YOUR_OPENAI_MODEL",
        api_key="YOUR_API_KEY",
    )
)

ag = agent(agconfig=cfg)
```

This is the same underlying backend used for vLLM/local endpoints above — omit `base_url` and it talks to `https://api.openai.com/v1`. Unlike the other providers, the API key isn't picked up from an environment variable automatically; pass it explicitly.

**Anthropic**

```python
from agency.configs.agconfig import agconfig, llmconfig

cfg = agconfig(
    llmconfig(
        provider="anthropic",
        model="claude-sonnet-5",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
)

ag = agent(agconfig=cfg)
```

Requires the `anthropic` package (`pip install anthropic`).
For Claude on Bedrock, use `provider="bedrock"` instead (see below) — it's picked automatically for `anthropic.*` model IDs.
For Claude via AWS's direct Anthropic-on-AWS API, use `provider="anthropicAWS"`.

**Amazon Bedrock**

Credentials are picked up automatically from the environment (IAM role, `~/.aws/credentials`, SSO, etc.). `aws_bedrock_token_generator` (included in dependencies) exchanges them for a bearer token on each request.

```python
from agency.configs.agconfig import agconfig, llmconfig

cfg = agconfig(
    llmconfig(
        provider="bedrock",
        region="us-east-1",
        model="nvidia.nemotron-super-3-120b",
    )
)

ag = agent(agconfig=cfg)
```

Pass `api_key="bedrock-api-key-..."` to `llmconfig(provider="bedrock", ...)` to use a static Bedrock API key instead of IAM credentials. For Claude models on Bedrock, stick to the fields listed under **Anthropic** above — other generation params aren't supported there.

## Usage Examples

| Example | What it shows |
| --- | --- |
| [`01_basic_agent.py`](examples/01_basic_agent.py) | The smallest complete typed agent submission and pending result. |
| [`02_context_and_lifecycle.py`](examples/02_context_and_lifecycle.py) | Context ordering, dependencies, async calls, redirect, pause, resume, and cancel. |
| [`03_tools_and_policy.py`](examples/03_tools_and_policy.py) | Direct, host MCP, and sandbox MCP tools plus policy. |
| [`04_files_images_and_types.py`](examples/04_files_images_and_types.py) | Every built-in specialized data type and a custom `agtype`. |
| [`05_parallel_workflows.py`](examples/05_parallel_workflows.py) | `agmap`, forks, scheduler fan-in, synchronization, and teams. |
| [`06_configuration_and_resources.py`](examples/06_configuration_and_resources.py) | All configuration namespaces, mounts, and the resource pool. |
| [`07_sandbox_api.py`](examples/07_sandbox_api.py) | Direct sandbox execution, files, limits, checkpoints, restore, and fork. |
| [`08_checkpoints.py`](examples/08_checkpoints.py) | Single-agent and registry-wide save/load. |
| [`09_observability.py`](examples/09_observability.py) | Orchestrator snapshots, event logs, state, and profiling. |
| [`10_harnesses_and_webui.py`](examples/10_harnesses_and_webui.py) | Codex/native harness interchangeability and the Web UI. |

See [`examples/README.md`](examples/README.md) for more details on each example.

## Core concepts

**`agent` / `Agent`** — a state container with LLM config, sandboxed tools, one authoritative conversation-context chain (`agcontext`), and a name. `run()` and `queue_message()` atomically reserve positions in that chain. Engine-backed requests receive a fresh `AgentEngine` only when the global orchestrator dispatches them; reusable orchestrator workers provide cross-agent concurrency while preserving one active engine-backed request per agent. Sandboxes and harness services are created lazily. `Agent` is the public alias of `agent`.

**`agskill`** — a named skill with its own system prompt, optional input/output schemas, and an optional tool list. `agskill.run(agent, input)` uses the same orchestrator path and returns the same pending `agdata` shape as `agent.run()`. Harness execution begins only after scheduler admission.

**`GlobalAgentOrchestrator`** -- the process-wide scheduler, dependency resolver, context-only message executor, ready queue, and reusable execution-worker owner. Submission, control changes, dependency completion, engine completion, and shutdown push events to one condition-backed scheduler thread; there is no completion polling. Its asynchronous global collector stores scheduler, team, and resource events plus a lightweight agent-database catalog in `agency.sqlite3`; detailed agent history remains in each agent's own database. Configure active engine capacity with `agconfig(orchestratorconfig(max_concurrent_engines=...))`, and inspect, flush, or stop it through `get_orchestrator().snapshot()`, `.flush()`, and `.shutdown()`.

**`agtool`** — a named callable an LLM can invoke via function calling. It exposes an OpenAI-compatible tool schema and calls its function directly in the execution-owning process and thread, preserving closures over live host state. The optional timeout argument is retained for call-site compatibility but is not enforced by `agtool` itself. Sandbox commit or rollback belongs to the enclosing request transaction.

**`agdata`** — a lightweight dict wrapper that travels between agents, skills, and tools. A pending result resolves on field access, `wait()`, `to_dict()`, `to_json()`, or `await`, and can be passed anywhere pending `agdata` is accepted. It also supports JSON serialisation and schema validation.

**`agtype`** — base class for typed agdata field values. Subclass to control how a schema field is serialised, transferred to/from the sandbox filesystem, represented in the system prompt, and cleaned up. `agfile` is the built-in subclass for file-backed fields. `agimage` is the built-in subclass for multimodal image inputs — local files are base64-encoded automatically; the image is injected into the message content array so the model sees it visually. `agrawstring` bypasses JSON formatting entirely — the input string is sent as raw text and the model's full response is captured as-is, skipping JSON parsing and the retry loop.

**`agteam`** — coordinates multiple agents or tasks. Subclass, define `setup()` to wire up agents and skills, override `run()` with your workflow. Each `run()` call executes in its own daemon thread.

**`agwebui`** -- a browser-based dashboard whose FastAPI server runs in a separate process. Its execution-side emitter enqueues lightweight global events for the asynchronous `agency.sqlite3` writer; the standalone server polls that database and pushes updates to browsers over WebSocket. Detailed state, configuration, and context history are read from a selected agent's SQLite database only when the UI requests that agent.

**GPU support** — NVIDIA and AMD (ROCm) GPUs are both supported. `agResourcePool` auto-detects GPUs via `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD) and issues leases to prevent two agents from sharing a device. The sandbox container receives `--gpus all` (NVIDIA) or `--device /dev/kfd --device /dev/dri` (AMD) at startup. GPU access uses *lazy physical allocation*: `reserve_gpu` sets a virtual flag with no physical cost; a physical GPU is claimed from the pool only when a bash command actually runs, and returned as soon as the command's processes finish. Between bash calls the GPU is free for other agents. `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` are set to the assigned device ID for the duration of each bash execution.

## Running tests

```bash
uv run pytest
```

Most tests mock the model client and run entirely in-process without a container. Tests that require a live container are marked and skipped when Docker or Podman is unavailable. Tool tests use the same direct-call path as production.

## Linting

```bash
uv run pre-commit run --all-files
```

Runs the same checks as the `pre-commit` git hook and the CI `pre-commit` job: `ruff check --fix` (unused imports/variables, undefined names), `ruff format`, and hygiene hooks (trailing whitespace, end-of-file, YAML/TOML syntax, merge-conflict markers). Config lives in `.pre-commit-config.yaml` and `pyproject.toml`'s `[tool.ruff]`.

## Docs

| File | Topic |
|---|---|
| [Invocation_API.md](docs/Invocation_API.md) | Pending results, ordered context messages, and exact-result controls |
| [Design_orchestrator.md](docs/Design_orchestrator.md) | Atomic context-chain publication, event scheduling, reusable workers, and shutdown |
| [Design_execution_loop.md](docs/Design_execution_loop.md) | Safe-boundary control delivery, transactions, attempt isolation, and retained cursors |
