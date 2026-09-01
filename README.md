# Agency

A multi-agent framework with sandboxed execution, isolated filesystems, GPU access control, and automatic background-process tracking. Submission is lazy: PREPARED, dependency-blocked, suspension-gated, and host-only message requests create no engine or sandbox infrastructure before dispatch. A sandbox is created only after an engine-backed invocation is admitted, then its transaction is committed or discarded before output context and results are published.

Agents are non-blocking by default. `agent.run()` returns an `Invocation` immediately. The invocation is both the exact lifecycle handle controlled by the global orchestrator and an agdata-compatible pending result: it can be awaited, passed into another skill, or read through attribute access. A process-wide, event-driven orchestrator holds PREPARED and dependency-blocked work without occupying a worker or execution slot, then dispatches eligible work onto a reusable worker pool. Every dispatch still receives a fresh `AgentEngine`. Concurrency is unlimited by default and can be capped globally.

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
uv pip install -e ".[dev]"   # dev dependencies (pytest, ruff, pre-commit) + the profiler extra the tests need

pre-commit install   # one-time; runs ruff (lint + format) and hygiene checks on every commit
```

The sandbox image installs `torch torchvision transformers datasets accelerate numpy scipy matplotlib`. The build script selects the NVIDIA, AMD, or CPU path automatically and runs an inline PyTorch smoke check.

## Quick start

Supply LLM backends to agents by building an `agConfig` object and passing it to the agents:

```python
cfg = agConfig(agVLLMBackendConfig(model="...", api_key="..."))
ag = agent(agconfig=cfg)
```

Pick the config class for your backend — `agVLLMBackendConfig`, `agOpenAIBackendConfig`, `agAnthropicBackendConfig`, or `agBedrockBackendConfig` — and it only accepts the fields that backend actually uses, catching typos and unsupported options immediately. Need config for more than one thing, such as an LLM backend and a sandbox mount? Pass several config views to the same `agConfig(...)` call.

**OpenAI-compatible serving endpoint (vLLM, local, etc.)**

```python
from agency import agent, agskill, agdata
from agency.agconfig import agConfig
from agency.llm import agVLLMBackendConfig

continuation = agskill(
    name="continuation",
    system_prompt="Continue the sentence.",
    input_schema=agdata(text=str),
    output_schema=agdata(summary=str),
)

cfg = agConfig(
    agVLLMBackendConfig(
        base_url="http://localhost:8000/v1", # Your serving API URL
        model="YOUR_SERVED_MODEL",
        api_key="YOUR_API_KEY" # Leave blank ("") if unused
    )
)

ag = agent(agconfig=cfg)

invocation = ag.run(continuation, agdata(text="Fly me to the moon and let me "))
print(invocation.summary)   # blocks until the invocation succeeds or fails
```

## Submissions and lifecycle

All submissions reserve one position in the agent's authoritative context chain. Later work from the same agent cannot overtake that position.

```python
first = ag.run(skill, agdata(topic="one"))
second = ag.run(other_skill, first)  # Invocation is a pending data dependency

held = ag.prepare(skill, agdata(topic="three"))
message = ag.send("Keep citations next to the claims they support")

# `held` intentionally blocks the later message until its original position opens.
held.start()       # starts only this prepared invocation
# ag.start()       # alternatively starts a snapshot of all currently prepared work

close = ag.destroy()  # rejects new submissions immediately
close.wait()          # waits for asynchronous cleanup; repeated destroy() returns this handle
```

While an invocation is active, `inv.steer(...)`, `inv.pause()`, `inv.resume()`, and `inv.cancel()` control only that invocation. They are observed at safe boundaries, never halfway through a model request or tool call. Steering closes at the final-answer fence, so callers should handle `RuntimeError` if execution has already crossed it.

`ag.suspend()` is the independent agent-wide gate: it prevents new engine-backed dispatch and parks active work at its next safe boundary if it has not crossed the closing/completion fence. Queued work held by the gate occupies no worker or global slot; an already-running invocation retains its slot while parked. `ag.resume()` clears only that gate. It neither starts PREPARED work nor clears an invocation-specific pause. `ag.pause()` remains a compatibility alias for `ag.suspend()`.

`ag.send()` returns a `MessageSubmission`. It copies its predecessor context and appends retained host-only context without creating an engine, sandbox, harness, or model request. See [Invocation API](docs/Invocation_API.md) for result compatibility and lifecycle details.

**OpenAI**

```python
from agency.agconfig import agConfig
from agency.llm import agOpenAIBackendConfig

cfg = agConfig(
        agOpenAIBackendConfig(
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
from agency.agconfig import agConfig
from agency.llm import agAnthropicBackendConfig

cfg = agConfig(
        agAnthropicBackendConfig(
        model="claude-sonnet-5",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
)

ag = agent(agconfig=cfg)
```

Requires the `anthropic` package (`pip install anthropic`).
For Claude on Bedrock, use `agBedrockBackendConfig` instead (see below) — it's picked automatically for `anthropic.*` model IDs.
For Claude via AWS's direct Anthropic-on-AWS API, use the generic `agLLMBackendConfig(provider="anthropicAWS", ...)` — there's no dedicated class for it yet.

**Amazon Bedrock**

Credentials are picked up automatically from the environment (IAM role, `~/.aws/credentials`, SSO, etc.). `aws_bedrock_token_generator` (included in dependencies) exchanges them for a bearer token on each request.

```python
from agency.agconfig import agConfig
from agency.llm import agBedrockBackendConfig

cfg = agConfig(
        agBedrockBackendConfig(
        region="us-east-1",
        model="nvidia.nemotron-super-3-120b",
    )
)

ag = agent(agconfig=cfg)
```

Pass `api_key="bedrock-api-key-..."` to `agBedrockBackendConfig(...)` to use a static Bedrock API key instead of IAM credentials. For Claude models on Bedrock, stick to the fields listed under **Anthropic** above — other generation params aren't supported there.

## Usage Examples

| Example | What it shows |
| --- | --- |
| [`base_example.py`](examples/base_example.py) | The simplest complete agent — one agent, two skills, shared history. |
| [`parallel_exec.py`](examples/parallel_exec.py) | The two natural parallelism patterns: sequential chaining on one agent, and fork fan-out across multiple agents/containers. |
| [`custom_tools.py`](examples/custom_tools.py) | A multi-step, multi-agent research pipeline combining a custom host-side tool, parallel summarisation forks, and a shared output directory. |
| [`image_processing.py`](examples/image_processing.py) | `agimage`, the multimodal image input field type, across single-image, multi-image, and URL-image forms. |
| [`sandbox_handoff.py`](examples/sandbox_handoff.py) | Reading and driving an agent's `agSandbox` directly from the host, and handing one sandbox off between two agents. |
| [`dynamic_config_example.py`](examples/dynamic_config_example.py) | Composing an `agConfig` from two owners' fields, then updating a `DynamicConfigParam` field on the same config between two skill calls. |

See [`examples/README.md`](examples/README.md) for more details on each example.

## Core concepts

**`agent` / `Agent`** — a state container with LLM config, sandboxed tools, one authoritative conversation-context chain (`agcontext`), and a name. `run()`, `prepare()`, and `send()` atomically reserve positions in that chain. Engine-backed requests receive a fresh `AgentEngine` only when the global orchestrator dispatches them; reusable orchestrator workers provide cross-agent concurrency while preserving one active engine-backed request per agent. Sandboxes and harness services are created lazily. `Agent` is the public alias of `agent`.

**`agskill`** — a named skill with its own system prompt, optional input/output schemas, and an optional tool list. `agskill.run(agent, input)` uses the same orchestrator path and returns the same `Invocation` shape as `agent.run()`. Harness execution begins only after scheduler admission.

**`GlobalAgentOrchestrator`** — the process-wide scheduler, dependency resolver, context-only message executor, ready queue, and reusable execution-worker owner. Submission, control changes, dependency completion, engine completion, and shutdown push events to one condition-backed scheduler thread; there is no completion polling. Configure active engine capacity with `agOrchestratorConfig(max_concurrent_engines=...)`, and inspect or stop it through `get_orchestrator().snapshot()` and `.shutdown()`.

**`agtool`** — a named callable an LLM can invoke via function calling. It exposes an OpenAI-compatible tool schema and calls its function directly in the execution-owning process and thread, preserving closures over live host state. The optional timeout argument is retained for call-site compatibility but is not enforced by `agtool` itself. Sandbox commit or rollback belongs to the enclosing invocation transaction.

**`agdata`** — a lightweight dict wrapper that travels between agents, skills, and tools. An `Invocation` exposes its pending output through `inv.result`, proxies unknown attributes to that output, and can be passed anywhere pending agdata is accepted. Supports JSON serialisation and schema validation.

**`agtype`** — base class for typed agdata field values. Subclass to control how a schema field is serialised, transferred to/from the sandbox filesystem, represented in the system prompt, and cleaned up. `agfile` is the built-in subclass for file-backed fields. `agimage` is the built-in subclass for multimodal image inputs — local files are base64-encoded automatically; the image is injected into the message content array so the model sees it visually. `agrawstring` bypasses JSON formatting entirely — the input string is sent as raw text and the model's full response is captured as-is, skipping JSON parsing and the retry loop.

**`agteam`** — coordinates multiple agents or tasks. Subclass, define `setup()` to wire up agents and skills, override `run()` with your workflow. Each `run()` call executes in its own daemon thread.

**`agwebui`** — a browser-based dashboard whose FastAPI server runs in a separate process. Its execution-side emitter writes structured events to `ui_events.db`; the standalone server polls that SQLite database and pushes updates to browsers over WebSocket.

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
| [Invocation_API.md](docs/Invocation_API.md) | `Invocation`, PREPARED work, ordered messages, controls, and destruction |
| [Design_orchestrator.md](docs/Design_orchestrator.md) | Atomic context-chain publication, event scheduling, reusable workers, and shutdown |
| [Design_execution_loop.md](docs/Design_execution_loop.md) | Safe-boundary control delivery, transactions, attempt isolation, and retained cursors |
