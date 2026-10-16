# Examples

Run scripts from the repository root after following
[installation and credentials](../docs/getting-started.md):

```bash
export OPENAI_API_KEY="your-api-key"
uv run python examples/quickstart.py
uv run python examples/run_all.py
```

Each example demonstrates a small workflow and prints the results. The runner
executes lessons 01–11, 13 and 14 in separate processes, continues after failures, and
returns a nonzero exit code if any lesson fails. The SWE-bench example requires
its own prepared inputs and runs separately.

| Example | Demonstrates |
| --- | --- |
| [quickstart.py](quickstart.py) | Write a Python function and print sample outputs. |
| [01_basic_agent.py](01_basic_agent.py) | Run one typed skill and wait for its result. |
| [02_context_and_lifecycle.py](02_context_and_lifecycle.py) | Chain results, make async calls, pause, resume, and cancel. |
| [03_tools_and_policy.py](03_tools_and_policy.py) | Call a Python tool directly, on the host, and in a sandbox. |
| [04_files_images_and_types.py](04_files_images_and_types.py) | Pass files, binary data, paths, raw text, and an image. |
| [05_parallel_workflows.py](05_parallel_workflows.py) | Map tasks, fork agents, combine results, and use a team. |
| [06_configuration_and_resources.py](06_configuration_and_resources.py) | Clone configuration, mount a directory, and inspect resources. |
| [07_sandbox_api.py](07_sandbox_api.py) | Read and write sandbox files and execute a command without a model. |
| [08_checkpoints.py](08_checkpoints.py) | Save an agent and restore it in a fresh process. |
| [09_observability.py](09_observability.py) | Record a profile and print orchestrator and timing information. |
| [10_harnesses_and_webui.py](10_harnesses_and_webui.py) | Run native and Codex agents with a temporary dashboard. |
| [11_live_trajectory.py](11_live_trajectory.py) | Record real local file edits and subprocess telemetry without a model. |
| [12_swebench_live.py](12_swebench_live.py) | Run one prepared SWE-bench issue and export its patch. |
| [13_context_manipulation.py](13_context_manipulation.py) | Rewrite message blocks sent to the model with `patch_history`, between skill runs and at an exact LLM call with `pause`/`resume`. |
| [14_self_context_editing.py](14_self_context_editing.py) | Let the model list and edit its own context with `context_tools`, for one skill run or with `persist=True`. |

Lessons 01–10 share `_common.py`. They default to the included **native harness**,
OpenAI Chat Completions with `gpt-6-luna`, `reasoning_effort="none"`, and a
4096-token completion budget. GPU passthrough is disabled, and idle sandboxes use one CPU. Artifact directories
are printed under `runs/tutorials/`, and sandboxes are cleaned up on exit.

Lesson 03 uses a Python image matching the host interpreter because sandbox
tools serialize Python functions.

Lessons 03 and 10 also use Codex. Set `AGENCY_CODEX_BINARY` to a compatible
executable if the one on your PATH differs from the version required by Agency:

```bash
export AGENCY_CODEX_BINARY="/absolute/path/to/compatible/codex"
uv run python examples/run_all.py --only 01 03 07 10
```

Set `AGENCY_HARNESS` to change the default harness. Set `AGENCY_LLM_PROVIDER`,
`AGENCY_LLM_MODEL`, and `AGENCY_REASONING_EFFORT` to change the tutorial model.
The helper supports OpenAI, OpenAI Responses, Anthropic, LiteLLM, vLLM, and
Bedrock; configure the selected provider's credentials and endpoint as described
in [configuration](../docs/guides/configuration.md).

`AGENCY_SANDBOX_IMAGE` selects a container image. `AGENCY_EXAMPLE_RUN_ROOT`
changes the artifact directory. `AGENCY_EXAMPLE_WAIT_TIMEOUT_SECONDS` changes
the result wait timeout. Keep a checkpoint with `AGENCY_KEEP_EXAMPLE_CHECKPOINTS=1`.
Quickstart uses `OPENAI_API_KEY` and optional `OPENAI_MODEL`; add `--webui` to
show its dashboard.

Lesson 11 creates a fresh temporary log directory by default. If you supply
`--run-dir`, choose a fresh directory on every run. Its first subprocess
intentionally returns a failure code, then the example edits the file and reruns
it successfully to show both outcomes in the trace.

For SWE-bench, prepare an official task row and a matching instance image, then
inspect its required arguments:

```bash
uv run python examples/12_swebench_live.py --help
```

An instance image needs Agency's Python runtime and the repository's test
environment. Use `--source-dir` for a prepared host checkout, `--test-python`
for its test interpreter, and `--harness-python` for the harness interpreter.
You can also pass these arguments through the runner with
`--only 12 --example-args ...`.

[smoke_test_models.py](smoke_test_models.py) is a separate model compatibility
utility. It makes real API calls and reads unset credentials from the repository
`.env` file. Run a small check with:

```bash
uv run python examples/smoke_test_models.py --provider openai --model gpt-6-luna
```
