# Configuration

For exact namespace fields/defaults, cloning and update scopes, see the
[configuration API reference](../api/configuration.md).

An `agconfig` combines model settings with optional harness, sandbox, and
other settings. Pass it to `Agent(agconfig=cfg)`. Each agent gets its own copy.

## Models

The [README](../../README.md#basic-usage) shows the tutorial's OpenAI Chat
Completions configuration. To use the Responses API instead:

```python
import os
from agency.configs.agconfig import agconfig, llmconfig

cfg = agconfig(llmconfig(
    provider="openai_responses",
    base_url="https://api.openai.com/v1",
    model=os.environ["OPENAI_MODEL"],
    api_key=os.environ["OPENAI_API_KEY"],
    max_completion_tokens=4096,
))
```

Set `OPENAI_MODEL` to a model available to your account that supports tools.
Set `reasoning_effort` only to a value that model accepts. The repository's
Chat Completions examples use `gpt-6-luna` with `"none"`; for a model that
needs reasoning alongside tools, use `openai_responses` with an accepted effort.
Responses does not accept Chat-only fields such as `seed`, `stop`, or `n`.
Both OpenAI backends require `base_url` and an explicitly passed API key.

For Anthropic:

```python
cfg = agconfig(llmconfig(
    provider="anthropic",
    model=os.environ["ANTHROPIC_MODEL"],
    api_key=os.environ["ANTHROPIC_API_KEY"],
    max_completion_tokens=4096,
))
```

For a tool-capable OpenAI-compatible endpoint, such as vLLM:

```python
cfg = agconfig(llmconfig(
    provider="vllm",
    base_url=os.environ["LLM_BASE_URL"],  # e.g. http://localhost:8000/v1
    model=os.environ["LLM_MODEL"],
    api_key=os.environ.get("LLM_API_KEY", ""),
    max_completion_tokens=4096,
))
```

For Amazon Bedrock, supply a Bedrock bearer-token API key:

```python
cfg = agconfig(llmconfig(
    provider="bedrock",
    region=os.environ["AWS_REGION"],
    model=os.environ["BEDROCK_MODEL"],
    api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
    max_completion_tokens=4096,
))
```

### Responses that are not a finished turn

A successful HTTP status does not mean the model produced a turn. Agency checks
every response before a harness sees it:

| Response | What Agency does | `llmconfig` field |
|---|---|---|
| No text and no tool call (an empty 200, or reasoning with no answer) | asks again, then reports a retryable error | `unusable_response_retries` (2), `fail_on_empty_response` (true) |
| Stream that ends without its terminating event, or breaks after it started | asks again, then reports a retryable error | `unusable_response_retries`, `strict_stream_termination` (true) |
| Tool call whose arguments are not a JSON object | asks again, then reports a retryable error | `unusable_response_retries` |
| Cut off at the output limit | optionally asks again with a larger limit; otherwise passes it on with its stop reason, its incomplete tool calls marked | `truncation_retries` (0), `truncation_output_multiplier` (2), `truncation_max_output_tokens` |

Asking again is safe: no part of a response reaches the harness, and no tool
runs, before the response is complete. A rate limit, a rejected request and a
failure before any response arrived are never retried here; the SDK and the
harness handle those. Each discarded response is recorded as an `llm_retry`
event and on the delivered exchange's metadata.

`max_completion_tokens` limits reasoning and visible output together. A
reasoning model needs far more than a few thousand: with a limit that is too
low, responses stop inside a tool call's arguments. Empty content next to a
valid tool call is an ordinary tool turn and is never treated as empty.

The Codex adapter reports a cut-off response to Codex as `response.incomplete`
and sends none of its items, since Codex starts a tool as soon as its item
arrives. With `provider="openai_responses"`, reasoning items, tool-call ids and
message phases round-trip through Codex unchanged; the Chat Completions
backend cannot send reasoning state back and says so once when it drops it.

## Agent harnesses

A harness is the program that executes the agent's task. Select it independently
of the model:

```python
from agency import Agent

worker = Agent(agconfig=cfg, harness="native")
```

| `harness` | Program required in the sandbox |
| --- | --- |
| `native` | Included with Agency; no separate CLI. |
| `claude_code` | `claude` |
| `codex` | `codex` **0.147.0**; other versions are rejected. |
| `grok` | `grok` |
| `kimi` | `kimi` |
| `opencode` | `opencode` |

Use Linux executables compatible with the sandbox. Agency can mount a host
CLI's dedicated installation directory, or use a CLI installed in the image.
An npm installation also needs Node in the image; system interpreters and
shared libraries are not copied from the host. The default Python image does
not include Node or these external CLIs.

The [harness image recipe](../../tests/host/fixtures/Dockerfile.harnesses) shows
pinned CLI installations used in validation. These integrations are sensitive
to CLI versions; start with native for the shortest setup.

## Work on a project

Add a mount before creating the agent. This example preserves all settings
from `cfg` and exposes the current project at `/workspace/project`:

```python
from pathlib import Path
project_cfg = cfg.clone()
project_cfg.sandbox.add_mount(
    "project", Path.cwd(), "/workspace/project", mode="rw"
)
worker = Agent(agconfig=project_cfg)
```

Tell your skill to work in `/workspace/project`. Writes to this mount change
host files directly and are outside sandbox rollback/checkpoints. Use
`mode="ro"` for read-only access. Set `sandboxconfig(base_image="your-image")`
to provide project dependencies. The default is `docker.io/library/python:3.12-slim`.

## Advanced setup

- [Configuration and resources example](../../examples/06_configuration_and_resources.py):
  resource limits, concurrency settings, and updating an agent's configuration.
- [Optional Ubuntu host setup](host-setup.md): ZFS checkpoints and CRIU resume.
  Ordinary container runs use image checkpoints and need no `setup-host` command.
