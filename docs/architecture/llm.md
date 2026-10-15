# llm — provider backends

`agency/llm/` translates between Agency's model-message representation and provider APIs. It lets the host select a provider independently of the harness's native model protocol.

## Backend selection and translation

[agllm.py](../../agency/llm/agllm.py) defines the backend interface and `agllm.for_config()` selection. Backends consume Agency requests, format provider payloads, dispatch them and convert responses or streaming events back to Agency message blocks. The interface also supplies model lookup, context-limit discovery and generation-parameter handling.

| Source | Provider path |
| --- | --- |
| [openai.py](../../agency/llm/openai.py) | OpenAI-compatible chat completions, including compatible self-hosted endpoints. |
| [openai_responses.py](../../agency/llm/openai_responses.py) | OpenAI Responses API. |
| [anthropic.py](../../agency/llm/anthropic.py) | Anthropic API. |
| [bedrock.py](../../agency/llm/bedrock.py) | AWS model routes selected according to provider and model. |
| [mock.py](../../agency/llm/mock.py) | Replay recorded successful model exchanges with configurable timing. |

The common representation carries text, tool calls/results, reasoning and metadata blocks. Provider-specific conversion remains in the backend because APIs differ in supported parameters and message semantics. A common interface does not make every model accept every option.

## Connection to execution

The [engine's LLM handler](../../agency/engine/host_servers/llm_handler_server.py) owns host request handling, streaming lifecycle, retries and exchange recording. It calls this package for provider dispatch. The sandbox [harness gateway](harness.md) translates CLI-facing protocols to that host interface; provider credentials remain on the host.

Context-limit discovery first respects an explicit override, then tries provider model information and known limits, finally falling back to configured defaults. The native harness uses this information for its own compaction rather than delegating the agent loop to this package.

[usage_tracker.py](../../agency/llm/usage_tracker.py) tracks newly introduced prompt tokens across matching conversation branches. This separates incremental input from the full prompt size reported on every exchange. Usage accounting and mock replay describe observed model traffic; they do not evaluate task correctness. Provider settings are documented in the [configuration guide](../guides/configuration.md).
