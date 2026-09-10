# Agency tutorial

These examples are a progressive, executable tour of Agency's current
architecture. They follow the same path as the architecture diagram:

1. a user submits skills and data;
2. the process-wide orchestrator resolves dependencies and schedules work;
3. a fresh per-request engine exposes host services and the configured LLM;
4. the selected harness and sandbox tools execute inside an isolated sandbox.

Run the examples in numeric order. Each script is standalone, writes artifacts
under `runs/tutorials/`, cleans up its own sandboxes, and exits normally. The
web UI is optional and does not linger in the automated run.

## Configure a model and harness

The default is the Codex harness routed through Agency to OpenAI Luna:

```bash
export OPENAI_API_KEY="..."
export AGENCY_LLM_PROVIDER="openai"
export AGENCY_LLM_MODEL="gpt-5.6-luna"
export AGENCY_HARNESS="codex"
export AGENCY_REASONING_EFFORT="none"
```

External CLI harnesses must also be runnable inside the sandbox image. If your
default image does not include the CLI's interpreter (Node for Codex), select a
compatible image with `AGENCY_SANDBOX_IMAGE`.

The harness and model are independent. For example, the same tutorials can run
Agency's native harness against an OpenAI-compatible vLLM endpoint:

```bash
export AGENCY_HARNESS="native"
export AGENCY_LLM_PROVIDER="vllm"
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_MODEL="your-served-model"
export LLM_API_KEY=""
```

Run everything:

```bash
uv run python examples/run_all.py
```

Run one lesson:

```bash
uv run python examples/01_basic_agent.py
```

## Lessons

| Lesson | Public surface demonstrated |
|---|---|
| [`01_basic_agent.py`](01_basic_agent.py) | `agent`/`Agent`, `agskill`, `agschema`, `agdata`, pending results, serialization, `agerror`, `agcanceled`, `AgError` |
| [`02_context_and_lifecycle.py`](02_context_and_lifecycle.py) | `agcontext`, ordered submissions, pending dependencies, `history`, `queue_message`, async execution, completed-result redirect fallback, `pause`, `resume`, `cancel` |
| [`03_tools_and_policy.py`](03_tools_and_policy.py) | `agtool`, host MCP tools, sandbox MCP tools, `agpolicy`, tool schemas and direct calls |
| [`04_files_images_and_types.py`](04_files_images_and_types.py) | `agtype`, `agfile`, `agbinary`, `agpath`, `agimage`, `agrawstring` |
| [`05_parallel_workflows.py`](05_parallel_workflows.py) | `Agent.fork`, scheduler dependency fan-in, `agteam`, `agmap`, `agtask`, `agsync`, `agdata.wait_all` |
| [`06_configuration_and_resources.py`](06_configuration_and_resources.py) | every `agconfig` namespace, cloning/updating/redaction, `change_config`, mounts, resource tools, `agResourcePool` |
| [`07_sandbox_api.py`](07_sandbox_api.py) | lazy sandbox creation, `agSandbox`, text/binary I/O, exec, detached processes, commit/restore/fork, runtime detection |
| [`08_checkpoints.py`](08_checkpoints.py) | `save`, `load`, `save_all`, `load_all`, `all`, checkpointed context and filesystem state |
| [`09_observability.py`](09_observability.py) | global orchestrator, `ExecutionScheduler`, `OrchestratorSnapshot`, `agDataLogger`, `record_state`, `agprof` |
| [`10_harnesses_and_webui.py`](10_harnesses_and_webui.py) | interchangeable native/Codex harnesses, `agwebui`, graceful process shutdown |

Set `AGENCY_KEEP_EXAMPLE_CHECKPOINTS=1` to retain lesson 8's full sandbox
checkpoint after its restore check. Set `AGENCY_WEBUI_LINGER=1` to keep lesson
10's dashboard running for interactive inspection; the acceptance runner uses
the non-lingering default.

`run_all.py` is the acceptance runner used for the EC2 validation. Files with a
leading underscore are support modules and are deliberately excluded from the
lesson glob.

## What is intentionally not a tutorial API

The private engine, adapter, daemon, protocol, and syscall-tracer modules are
implementation seams. Model-compatibility matrices, raw `/v1/messages` probes,
and full profiler workloads belong in tests or benchmarks rather than here.
