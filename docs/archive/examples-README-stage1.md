# Examples

Run these from the repository root after [installation](../../README.md#quick-start).
Start with one example; each script is standalone.

```bash
export OPENAI_API_KEY="your-api-key"
export AGENCY_HARNESS=native
uv run python examples/01_basic_agent.py
```

Examples 01–10 use OpenAI `gpt-6-luna` with `reasoning_effort="none"` by
default. They print their artifact directory under `runs/tutorials/` and
clean up their sandboxes on exit. Examples 03 and 10 also run Codex explicitly,
so those need [Codex installed](../guides/configuration.md#agent-harnesses).

## Pick a task

| Example | Try it when you want to… |
| --- | --- |
| [01_basic_agent.py](../../examples/01_basic_agent.py) | Run a skill and read a typed result. |
| [02_context_and_lifecycle.py](../../examples/02_context_and_lifecycle.py) | Chain tasks, use async calls, and control executions. |
| [03_tools_and_policy.py](../../examples/03_tools_and_policy.py) | Add Python tools via MCP and control which calls are allowed. |
| [04_files_images_and_types.py](../../examples/04_files_images_and_types.py) | Pass files, images, paths, or plain text. |
| [05_parallel_workflows.py](../../examples/05_parallel_workflows.py) | Run agents in parallel and combine results. |
| [06_configuration_and_resources.py](../../examples/06_configuration_and_resources.py) | Customize settings, mounts, and CPU/GPU resources. |
| [07_sandbox_api.py](../../examples/07_sandbox_api.py) | Run commands and read/write sandbox files directly. |
| [08_checkpoints.py](../../examples/08_checkpoints.py) | Save and restore agents. |
| [09_observability.py](../../examples/09_observability.py) | Record a profile and inspect timing reports. |
| [10_harnesses_and_webui.py](../../examples/10_harnesses_and_webui.py) | Compare native and Codex agents with a dashboard. |

## Change the model

These environment variables are read by the tutorial helper, not by arbitrary
Agency scripts. See [configuration](../guides/configuration.md) for Python settings.

```bash
# OpenAI Responses API with a reasoning model available to your account
export AGENCY_LLM_PROVIDER=openai_responses
export AGENCY_LLM_MODEL="your-model-id"
export AGENCY_REASONING_EFFORT=medium
uv run python examples/01_basic_agent.py
```

For Anthropic, set `AGENCY_LLM_PROVIDER=anthropic`, `AGENCY_LLM_MODEL` to your
model ID, and `ANTHROPIC_API_KEY`. For vLLM, set `AGENCY_LLM_PROVIDER=vllm`,
`LLM_BASE_URL`, `LLM_MODEL`, and optionally `LLM_API_KEY`.
`AGENCY_SANDBOX_IMAGE` selects an image with any additional tools you need.

## Viewing examples

- [11_live_trajectory.py](../../examples/11_live_trajectory.py) runs local file and subprocess
  work without a model or agent. See [live trajectory setup](../guides/profiling/live-trajectory.md#run-it-locally).
- [12_swebench_live.py](../../examples/12_swebench_live.py) runs a SWE-bench task with live
  agents; inspect `uv run python examples/12_swebench_live.py --help` for setup
  and options.

## Developer checks

With model credentials and the harnesses needed by the selected lessons:

```bash
uv run python examples/run_all.py --only 01 02 03 04 05 06 07 08 09 10
uv run python examples/smoke_test_models.py --help
```

The first command runs the original tutorials sequentially. The model smoke
test makes real API calls when run without `--help`. For contributors,
see [development checks](../guides/development.md).
