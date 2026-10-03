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

Add a mount before creating the agent. This example keeps the model settings
from `cfg` and exposes the current project at `/workspace/project`:

```python
from pathlib import Path
from agency.configs.agconfig import sandboxconfig

project_cfg = agconfig(cfg.llm, sandboxconfig())
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
