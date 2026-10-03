# Examples

Start with [installation and credentials](../docs/getting-started.md), then
run scripts from the repository root:

```bash
export OPENAI_API_KEY="your-api-key"
uv run python examples/quickstart.py
```

[quickstart.py](quickstart.py) uses the included native harness, disables GPU
passthrough, waits for the coding task to finish, and prints a summary and run
directory under `agency_runs/`. Add `--webui` using the
[profiler startup guide](../docs/guides/profiling.md).

## Pick a task

| Example | Use it to… | Additional setup |
| --- | --- | --- |
| [quickstart.py](quickstart.py) | Write and check a Python function with one native agent. | OpenAI key; optional `OPENAI_MODEL`. |
| [01_basic_agent.py](01_basic_agent.py) | Define a typed skill and read a pending result. | Numbered tutorial settings below. |
| [02_context_and_lifecycle.py](02_context_and_lifecycle.py) | Chain calls, use async results, pause, resume, and cancel. | Numbered tutorial settings. |
| [03_tools_and_policy.py](03_tools_and_policy.py) | Add Python tools via MCP and control tool permissions. | Also runs Codex explicitly. |
| [04_files_images_and_types.py](04_files_images_and_types.py) | Pass files, images, paths, and text. | Model must support the input types used. |
| [05_parallel_workflows.py](05_parallel_workflows.py) | Run agents in parallel and combine results. | Container resources for several agents. |
| [06_configuration_and_resources.py](06_configuration_and_resources.py) | Customize mounts and configuration, and inspect CPU/GPU resources. | Numbered tutorial settings. |
| [07_sandbox_api.py](07_sandbox_api.py) | Execute commands and read/write sandbox files directly. | Local container engine; no LLM key needed. |
| [08_checkpoints.py](08_checkpoints.py) | Save and restore agent state. | Numbered tutorial settings; no ZFS requirement. |
| [09_observability.py](09_observability.py) | Record an explicit profile and inspect timing reports. | Linux profiler prerequisites. |
| [10_harnesses_and_webui.py](10_harnesses_and_webui.py) | Run native and Codex agents with a dashboard. | Codex executable and Perfetto build. |
| [11_live_trajectory.py](11_live_trajectory.py) | Generate real local file/subprocess telemetry without an agent. | No model or container; [startup instructions](../docs/guides/profiling/live-trajectory.md#run-it-locally). |
| [12_swebench_live.py](12_swebench_live.py) | Run a prepared SWE-bench task through Codex. | Prepared dataset row, instance image, env file, and compatible Codex binary; inspect `--help`. |

## Numbered tutorial settings

Examples 01–10 share `_common.py`. They default to **Codex**, unlike the native
quickstart. Set `AGENCY_HARNESS=native` for lessons that do not select a harness
explicitly:

```bash
AGENCY_HARNESS=native uv run python examples/01_basic_agent.py
```

Their default provider/model is OpenAI Chat Completions with `gpt-6-luna`,
`reasoning_effort="none"`, and a 4096-token completion budget. They print
artifact paths under `runs/tutorials/` and clean up sandboxes on exit. Examples
03 and 10 select Codex in addition to native, so the environment override
alone does not remove their [CLI requirement](../docs/guides/configuration.md#agent-harnesses).
The shared helper leaves GPU passthrough at the library default; on a CPU-only
host use `quickstart.py` first. The tutorial CPU policy is a recorded
[follow-up](../docs/archive/documentation-refactor.md#follow-ups).

To use a tool-capable Responses model available to your account:

```bash
export AGENCY_LLM_PROVIDER=openai_responses
export AGENCY_LLM_MODEL="your-model-id"
export AGENCY_REASONING_EFFORT=medium
AGENCY_HARNESS=native uv run python examples/01_basic_agent.py
```

Choose a reasoning effort accepted by that model. For Anthropic, set
`AGENCY_LLM_PROVIDER=anthropic`, `AGENCY_LLM_MODEL`, and `ANTHROPIC_API_KEY`.
For vLLM, set `AGENCY_LLM_PROVIDER=vllm`, `LLM_BASE_URL`, `LLM_MODEL`, and
optionally `LLM_API_KEY`. `AGENCY_SANDBOX_IMAGE` selects a tutorial image;
`AGENCY_EXAMPLE_WAIT_TIMEOUT_SECONDS` increases the default 300-second result
wait. These are helper-specific variables, not automatic Agency configuration.
The quickstart reads only `OPENAI_API_KEY` and `OPENAI_MODEL` for its model.
See [configuration](../docs/guides/configuration.md) for Python settings.

## Run selected lessons

With the required model credentials and external harnesses installed:

```bash
uv run python examples/run_all.py --only 01 02 03 04 05 06 07 08 09 10
uv run python examples/smoke_test_models.py --help
```

The first command runs those tutorials sequentially. The model smoke test
makes real API calls when run without `--help`. See
[contributing](../CONTRIBUTING.md) for import and unit checks.
