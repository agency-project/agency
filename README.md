# Agency

A multi-agent framework with sandboxed execution, isolated filesystems, GPU access control, and automatic background-process tracking. Containers are created lazily — only when a task actually calls a sandboxed tool (bash, file I/O, etc.). Tasks that complete using only host-side tools (web fetch, paper search, …) never start a container at all. When a container is started, it is committed to a checkpoint image at the end of the task and destroyed, so containers exist only while sandboxed work is actively running.

Agents are non-blocking by default. `agent.run()` returns a pending `agdata` immediately; reading any field on it blocks until the result is ready. Each `run()` call executes in its own daemon thread so agents run concurrently without any shared pool to exhaust. Tool calls are offloaded to a process pool so CPU-bound work never blocks the main interpreter.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) — package manager ([installation](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer))
- Docker or Podman
- GPU (optional): NVIDIA (CUDA) or AMD (ROCm)
- Profiling: Linux only (cgroups v2 and Linux `/proc` kernel interfaces);
  other operating systems are currently unsupported

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
uv pip install -e ".[profiler]"   # profiling (PyTorch traces + NVIDIA GPU sampling)
uv pip install -e ".[dev]"   # dev dependencies (pytest, ruff, pre-commit)

pre-commit install   # one-time; runs ruff (lint + format) and hygiene checks on every commit
```

The sandbox image comes with `torch torchvision transformers datasets accelerate numpy scipy matplotlib` pre-installed, and the `Qwen/Qwen3.5-4B` model weights and `wikitext-2-raw-v1` dataset pre-cached. Run `python /opt/model_smoke.py` inside any container to verify the setup.

The profiler runs in the host Python environment, not inside the sandbox
image. Install the `profiler` extra before using it; this provides `torch` for
trace collection and `nvidia-ml-py` for NVIDIA GPU sampling. Environment-enabled
profiling places the complete benchmark and its Docker containers in a
dedicated transient cgroup, which requires `systemd-run`, `setpriv`, and
non-interactive `sudo` permission for `systemd-run`. See
[`agency/profiler/README.md`](agency/profiler/README.md) for usage.
For an unmodified non-Web-UI application, set
`AGENCY_PROFILE_SCOPE=process`; workload scope is opened automatically by
`agwebui.run(...)` or explicitly with `agprof.workload()`.

Within that workload cgroup, the profiler discovers PIDs recursively and adds
an independent TensorBoard process group for each stable PID/start-time
identity. CPU, RSS, virtual memory, and permitted per-process I/O counters are
therefore shown separately; the cgroup aggregate is labeled
`workload_total`.

## Quick start

Supply LLM backends to agents by building an `agConfig` object and passing it to the agents:

```python
cfg = agConfig(agVLLMBackendConfig(model="...", api_key="..."))
ag = agent(agconfig=cfg)
```

Pick the config class for your backend — `agVLLMBackendConfig`, `agOpenAIBackendConfig`, `agAnthropicBackendConfig`, or `agBedrockBackendConfig` — and it only accepts the fields that backend actually uses, catching typos and unsupported options immediately. Need config for more than one thing (an LLM backend and a sandbox mount, say)? Pass several to the same `agConfig(...)` call. See [`agconfig.md`](docs/agconfig.md) and [`Design_configuration.md`](docs/Design_configuration.md) for the full picture, and [`agllm.md`](docs/agllm.md) for the complete LLM field reference.

**OpenAI-compatible serving endpoint (vLLM, local, etc.)**

```python
from agency import agent, agskill, agdata
from agency.agconfig import agConfig
from agency.agllm_backends import agVLLMBackendConfig

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

result = ag.run(continuation, agdata(text="Fly me to the moon and let me "))
print(result.summary)   # blocks until done
```

**OpenAI**

```python
from agency.agconfig import agConfig
from agency.agllm_backends import agOpenAIBackendConfig

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
from agency.agllm_backends import agAnthropicBackendConfig

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
from agency.agllm_backends import agBedrockBackendConfig

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
| [`human_in_the_loop.py`](examples/human_in_the_loop.py) | Driving approval loops from Python so `ask_human` is guaranteed to be called, with re-plan/re-write cycles on rejection. |
| [`sandbox_handoff.py`](examples/sandbox_handoff.py) | Reading and driving an agent's `agSandbox` directly from the host, and handing one sandbox off between two agents. |
| [`dynamic_config_example.py`](examples/dynamic_config_example.py) | Composing an `agConfig` from two owners' fields, then updating a `DynamicConfigParam` field on the same config between two skill calls. |

See [`examples/README.md`](examples/README.md) for more details on each example.

## Core concepts

**`agent`** — a pure state container: holds an LLM config, sandboxed tools, conversation context (`agcontext`), and a name. It does not own an execution loop. `agent.run(skill, input)` is a thin dispatch call; the scheduling wrapper is `agskill.run()`, which spawns a daemon thread, and the ReAct loop is `agskill.execute_react()`. Each `run()` call is non-blocking and returns a pending `agdata` that resolves lazily. Sequential calls on the same agent are automatically serialised through the history chain. Between tasks `ag.sandbox` is `None`; containers exist only while a task is executing. Forking via `agent(parent)` deep-copies the context and copies the parent's checkpoint image (via `docker tag`, or a directory copy for a chroot-backed sandbox — see [agsandbox.md](docs/agsandbox.md)); the fork's container/jail is created lazily on its first `run()`.

**`agskill`** — a named skill with its own system prompt, optional input/output schemas, and an optional tool list. `agskill.run()` is a non-blocking scheduling wrapper: it spawns a daemon thread and returns a pending `agdata` immediately. The actual synchronous ReAct loop is `agskill.execute_react()`. The LLM calls tools, inspects results, and iterates until it has registered all required output fields. Output is collected via per-field tools (`return_summary`, `return_score`, etc.) generated dynamically from the output schema — each with a typed `value` parameter — rather than a single JSON blob. Each field is validated immediately on registration; missing fields trigger a targeted reprompt.

**`agtool`** — a named callable an LLM can invoke via function calling. Every tool call is offloaded to a `ProcessPoolExecutor` worker so CPU-bound tools don't block other agents. Tools are serialised with `cloudpickle`, so bound methods work without any extra machinery. Before each sandboxed tool call the container is checkpointed; on tool failure the sandbox is automatically rolled back to that checkpoint and the LLM is told the workspace was reverted. Agents can pass `"timeout": <seconds>` in any tool call's arguments to override the default 30 s watchdog.

**`agdata`** — a lightweight dict wrapper that travels between agents, skills, and tools. Fields are accessed as attributes (`result.summary`). Supports JSON serialisation and schema validation.

**`agtype`** — base class for typed agdata field values. Subclass to control how a schema field is serialised, transferred to/from the sandbox filesystem, represented in the system prompt, and cleaned up. `agfile` is the built-in subclass for file-backed fields. `agimage` is the built-in subclass for multimodal image inputs — local files are base64-encoded automatically; the image is injected into the message content array so the model sees it visually. `agrawstring` bypasses JSON formatting entirely — the input string is sent as raw text and the model's full response is captured as-is, skipping JSON parsing and the retry loop.

**`agteam`** — coordinates multiple agents or tasks. Subclass, define `setup()` to wire up agents and skills, override `run()` with your workflow. Each `run()` call executes in its own daemon thread.

**`agwebui`** — a browser-based dashboard that runs in a separate process. Writes structured events to a JSONL file; a standalone FastAPI server tails it and pushes updates to connected browsers over WebSocket. See [docs/agwebui.md](docs/agwebui.md).

**GPU support** — NVIDIA and AMD (ROCm) GPUs are both supported. `agResourcePool` auto-detects GPUs via `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD) and issues leases to prevent two agents from sharing a device. The sandbox container receives `--gpus all` (NVIDIA) or `--device /dev/kfd --device /dev/dri` (AMD) at startup. GPU access uses *lazy physical allocation*: `reserve_gpu` sets a virtual flag with no physical cost; a physical GPU is claimed from the pool only when a bash command actually runs, and returned as soon as the command's processes finish. Between bash calls the GPU is free for other agents. `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` are set to the assigned device ID for the duration of each bash execution.

## Running tests

```bash
pytest
```

Most tests mock the OpenAI client and run entirely in-process (no container needed). Tests that require a live container are marked and skipped if Docker/Podman is unavailable. Tool calls run through the real process pool in all tests — the same code path as production.

## Linting

```bash
pre-commit run --all-files
```

Runs the same checks as the `pre-commit` git hook and the CI `pre-commit` job: `ruff check --fix` (unused imports/variables, undefined names), `ruff format`, and hygiene hooks (trailing whitespace, end-of-file, YAML/TOML syntax, merge-conflict markers). Config lives in `.pre-commit-config.yaml` and `pyproject.toml`'s `[tool.ruff]`.

## Docs

### Design

| File | Topic |
|---|---|
| [Design_execution_loop.md](docs/Design_execution_loop.md) | Outer monitoring loop, inner ReAct loop, inbox drain, compaction |
| [Design_sandbox_lifecycle.md](docs/Design_sandbox_lifecycle.md) | Trace: background job, foreground job, daemon |
| [Design_compaction.md](docs/Design_compaction.md) | Auto-compaction — trigger, algorithm, incremental summaries |
| [Design_deadlock.md](docs/Design_deadlock.md) | Deadlock patterns — shared agents across parallel threads, diagnosis, and fixes |
| [Design_parallelization.md](docs/Design_parallelization.md) | Parallelism model — threads, GIL, process pool, LLM streaming |
| [Design_resource_control.md](docs/Design_resource_control.md) | All semaphores and locks — what each guards and how it is acquired |
| [Design_error_handling.md](docs/Design_error_handling.md) | All try/except blocks, retry loops, error emissions, and propagation paths |
| [Design_configuration.md](docs/Design_configuration.md) | Configuring agents/teams, the tiered parameter system, adding custom config params |


### Implementation

| File | Topic |
|---|---|
| [agent.md](docs/agent.md) | Agent construction, `run()`, forking, context, UI callbacks |
| [agconfig.md](docs/agconfig.md) | `agConfig` storage model, `ConfigParam` tiers, `FIELD_REGISTRY`, `_AgConfigViewBase`, `_ALLOWED_FIELDS` |
| [agcontext.md](docs/agcontext.md) | Persistent conversation state — message history, token counts, compaction summary |
| [agdata.md](docs/agdata.md) | Data container — pending results, schema types, serialization, error handling |
| [agllm.md](docs/agllm.md) | LLM wrapper — streaming calls, message construction, compaction, Bedrock support |
| [agskill.md](docs/agskill.md) | `run()` scheduling wrapper, `execute_react()` ReAct loop, schemas, `agtype`/`agfile` typed fields, input offloading, validation, retries |
| [agtype.md](docs/agtype.md) | `agtype` interface — typed field values, `agfile`, `agimage` (multimodal), `agrawstring` (raw bypass), custom subclasses |
| [agtools.md](docs/agtools.md) | Built-in tools, process offloading, sandboxed factories, `ask_human` |
| [agteam.md](docs/agteam.md) | Team coordination, `setup()` / `run()`, `agsync` |
| [agsandbox.md](docs/agsandbox.md) | Backend selection (docker/podman/chroot), sandbox lifecycle, GPU access, exec wrapper, PID tracking |
| [agresources.md](docs/agresources.md) | GPU/CPU/memory resource pool |
| [aglog.md](docs/aglog.md) | Structured JSONL log — skills, tools, lifecycle, compaction |
| [agterm.md](docs/agterm.md) | Color-coded terminal logger — event labels, color palette, webui routing |
| [agwebui.md](docs/agwebui.md) | Web UI — browser dashboard, event stream, WebSocket, ask_human path |
| [agsync.md](docs/agsync.md) | `agsync` — block until all pending agent results resolve |
| [agmap.md](docs/agmap.md) | `agmap` — run non-agent functions over items concurrently, sync or async |
| [agname.md](docs/agname.md) | Agent naming — auto-generated unique names for agents and run directories |
| [agutil.md](docs/agutil.md) | Shared utilities — helpers used across the framework |
| [agschema.md](docs/agschema.md) | Schema validation — output field validation, type error fixes, field handler construction |
