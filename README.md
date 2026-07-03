# Agency

A multi-agent framework with sandboxed execution, isolated filesystems, GPU access control, and automatic background-process tracking. Containers are created lazily — only when a task actually calls a sandboxed tool (bash, file I/O, etc.). Tasks that complete using only host-side tools (web fetch, paper search, …) never start a container at all. When a container is started, it is committed to a checkpoint image at the end of the task and destroyed, so containers exist only while sandboxed work is actively running.

Agents are non-blocking by default. `agent.run()` returns a pending `agdata` immediately; reading any field on it blocks until the result is ready. Each `run()` call executes in its own daemon thread so agents run concurrently without any shared pool to exhaust. Tool calls are offloaded to a process pool so CPU-bound work never blocks the main interpreter.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — package manager
- Docker or Podman
- GPU (optional): NVIDIA (CUDA) or AMD (ROCm)

## Install

```bash
git clone https://github.com/agency-project/agency
cd agency

# Build the sandbox base image (once).
# Auto-detects the host GPU (NVIDIA / AMD / CPU-only):
./images/build.sh

# Override GPU type explicitly:
GPU_TYPE=rocm ./images/build.sh    # AMD ROCm 7.2
GPU_TYPE=nvidia ./images/build.sh  # NVIDIA CUDA
GPU_TYPE=cpu ./images/build.sh     # CPU only

uv venv
source .venv/bin/activate

uv pip install -e .
uv pip install -e ".[dev]"   # dev dependencies (pytest etc.)
```

The sandbox image comes with `torch torchvision transformers datasets accelerate numpy scipy matplotlib` pre-installed, and the `Qwen/Qwen3.5-4B` model weights and `wikitext-2-raw-v1` dataset pre-cached. Run `python /opt/model_smoke.py` inside any container to verify the setup.

## Quick start

**OpenAI-compatible endpoint (vLLM, local, etc.)**

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

**Amazon Bedrock**

Credentials are picked up automatically from the environment (IAM role, `~/.aws/credentials`, SSO, etc.). `aws_bedrock_token_generator` (included in dependencies) exchanges them for a bearer token on each request.

```python
ag = agent(
    llm_config={
        "provider": "bedrock",
        "region":   "us-east-2",
        "model":    "nvidia.nemotron-super-3-120b",
    },
)
```

Pass `"api_key": "bedrock-api-key-..."` to use a static Bedrock API key instead of IAM credentials.

## Core concepts

**`agdata`** — a lightweight dict wrapper that travels between agents, skills, and tools. Fields are accessed as attributes (`result.summary`). Supports JSON serialisation and schema validation.

**`agtype`** — base class for typed agdata field values. Subclass to control how a schema field is serialised, transferred to/from the sandbox filesystem, represented in the system prompt, and cleaned up. `agfile` is the built-in subclass for file-backed fields. `agimage` is the built-in subclass for multimodal image inputs — local files are base64-encoded automatically; the image is injected into the message content array so the model sees it visually. `agrawstring` bypasses JSON formatting entirely — the input string is sent as raw text and the model's full response is captured as-is, skipping JSON parsing and the retry loop.

**`agskill`** — a named skill with its own system prompt, optional input/output schemas, and an optional tool list. `agskill.run()` is a non-blocking scheduling wrapper: it spawns a daemon thread and returns a pending `agdata` immediately. The actual synchronous ReAct loop is `agskill.execute_react()`. The LLM calls tools, inspects results, and iterates until it has registered all required output fields. Output is collected via per-field tools (`return_summary`, `return_score`, etc.) generated dynamically from the output schema — each with a typed `value` parameter — rather than a single JSON blob. Each field is validated immediately on registration; missing fields trigger a targeted reprompt.

**`agtool`** — a named callable an LLM can invoke via function calling. Every tool call is offloaded to a `ProcessPoolExecutor` worker so CPU-bound tools don't block other agents. Tools are serialised with `cloudpickle`, so bound methods work without any extra machinery. Before each sandboxed tool call the container is checkpointed; on tool failure the sandbox is automatically rolled back to that checkpoint and the LLM is told the workspace was reverted. Agents can pass `"timeout": <seconds>` in any tool call's arguments to override the default 30 s watchdog.

**`agent`** — a pure state container: holds an LLM config, sandboxed tools, conversation context (`agcontext`), and a name. It does not own an execution loop. `agent.run(skill, input)` is a thin dispatch call; the scheduling wrapper is `agskill.run()`, which spawns a daemon thread, and the ReAct loop is `agskill.execute_react()`. Each `run()` call is non-blocking and returns a pending `agdata` that resolves lazily. Sequential calls on the same agent are automatically serialised through the history chain. Between tasks `ag.sandbox` is `None`; containers exist only while a task is executing. Forking via `agent(parent)` deep-copies the context and copies the parent's checkpoint image via `docker tag`; the fork's container is created lazily on its first `run()`.

**GPU support** — NVIDIA and AMD (ROCm) GPUs are both supported. `agResourcePool` auto-detects GPUs via `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD) and issues leases to prevent two agents from sharing a device. The sandbox container receives `--gpus all` (NVIDIA) or `--device /dev/kfd --device /dev/dri` (AMD) at startup. GPU access uses *lazy physical allocation*: `reserve_gpu` sets a virtual flag with no physical cost; a physical GPU is claimed from the pool only when a bash command actually runs, and returned as soon as the command's processes finish. Between bash calls the GPU is free for other agents. `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` are set to the assigned device ID for the duration of each bash execution.

**`agteam`** — coordinates multiple agents or tasks. Subclass, define `setup()` to wire up agents and skills, override `run()` with your workflow. Each `run()` call executes in its own daemon thread.

**`agwebui`** — a browser-based dashboard that runs in a separate process. Writes structured events to a JSONL file; a standalone FastAPI server tails it and pushes updates to connected browsers over WebSocket. See [docs/agwebui.md](docs/agwebui.md).

## Parallelism model

| Layer | Mechanism | Notes |
|---|---|---|
| Agents / team tasks | One daemon thread per `run()` call | Threads release the GIL during LLM I/O; no shared pool to exhaust |
| LLM streaming | Background drain thread + 100 ms batch queue | Reduces GIL acquisitions from O(tokens) to O(tokens/batch) |
| Tool execution | `ProcessPoolExecutor` (256 workers) | Each tool call gets its own GIL |

See [docs/Design_parallelization.md](docs/Design_parallelization.md) for the full design.

## Examples

End-to-end examples now live in the separate [`agency-examples`](https://github.com/agency-project/agency-examples) repository.

The examples repository includes runnable workflows for sandboxed file I/O, shared agent history, forked parallel execution, paper crawling, multimodal image input, human-in-the-loop writing, sandbox handoff, bug-fixing loops, and ports of selected OpenAI Agents examples.

For local development, clone `agency` and `agency-examples` side by side, install `agency` in editable mode, and run the examples from the `agency-examples` checkout.

## Common skills

`agency.common_skills` provides two base skill subclasses for structured agent workflows:

| Class | Mode | Tools available | Workflow |
|---|---|---|---|
| `agplan` | Plan | `read`, `grep`, `glob`, `webfetch` (read-only; no bash or write) | UNDERSTAND → DESIGN → REVIEW → OUTPUT |
| `agbuild` | Build | Full sandbox tool set (bash, read, write, grep, glob, …) | UNDERSTAND → PLAN → IMPLEMENT → VERIFY → OUTPUT |

Both classes prepend a structured workflow prompt to the skill's system prompt and override `_build_tools()` to enforce the correct tool set. Subclass them the same way as `agskill`:

```python
from agency.common_skills import agplan, agbuild

analysis = agplan(
    name="analyse_codebase",
    system_prompt="Analyse the repository structure and identify the main entry points.",
    output_schema=agdata(summary=str, entry_points=list),
)

implementation = agbuild(
    name="add_feature",
    system_prompt="Implement the feature described in the plan.",
    input_schema=agdata(plan=str),
    output_schema=agdata(files_changed=list, tests_passed=bool),
)
```

`agplan` skills are well-suited to research, code review, gap analysis, and structured report generation. `agbuild` skills are suited to code generation, refactoring, running experiments, and any task that requires writing files or executing commands.

## Running tests

```bash
pytest
```

Most tests mock the OpenAI client and run entirely in-process (no container needed). Tests that require a live container are marked and skipped if Docker/Podman is unavailable. Tool calls run through the real process pool in all tests — the same code path as production.

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


### Implementation

| File | Topic |
|---|---|
| [agent.md](docs/agent.md) | Agent construction, `run()`, forking, context, UI callbacks |
| [agcontext.md](docs/agcontext.md) | Persistent conversation state — message history, token counts, compaction summary |
| [agdata.md](docs/agdata.md) | Data container — pending results, schema types, serialization, error handling |
| [agllm.md](docs/agllm.md) | LLM wrapper — streaming calls, message construction, compaction, Bedrock support |
| [agskill.md](docs/agskill.md) | `run()` scheduling wrapper, `execute_react()` ReAct loop, schemas, `agtype`/`agfile` typed fields, input offloading, validation, retries |
| [agtype.md](docs/agtype.md) | `agtype` interface — typed field values, `agfile`, `agimage` (multimodal), `agrawstring` (raw bypass), custom subclasses |
| [agtools.md](docs/agtools.md) | Built-in tools, process offloading, sandboxed factories, `ask_human` |
| [agteam.md](docs/agteam.md) | Team coordination, `setup()` / `run()`, `agsync` |
| [agsandbox.md](docs/agsandbox.md) | Sandbox lifecycle, GPU access, exec wrapper, PID tracking |
| [agresources.md](docs/agresources.md) | GPU/CPU/memory resource pool |
| [aglog.md](docs/aglog.md) | Structured JSONL log — skills, tools, lifecycle, compaction |
| [agterm.md](docs/agterm.md) | Color-coded terminal logger — event labels, color palette, webui routing |
| [agwebui.md](docs/agwebui.md) | Web UI — browser dashboard, event stream, WebSocket, ask_human path |
| [agsync.md](docs/agsync.md) | `agsync` — block until all pending agent results resolve |
| [agname.md](docs/agname.md) | Agent naming — auto-generated unique names for agents and run directories |
| [agutil.md](docs/agutil.md) | Shared utilities — helpers used across the framework |
| [agschema.md](docs/agschema.md) | Schema validation — output field validation, type error fixes, field handler construction |