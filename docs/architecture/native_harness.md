# native_harness — Agency's coding-agent loop

`agency/native_harness/` implements Agency's own harness. It is a standalone command-line program with its own model client, tools and session files. When Agency selects the native harness, the [native adapter](../../agency/harness/adapters/native.py) launches this program inside the sandbox, using the same daemon and host-service boundary as external harnesses.

## Model-and-tool loop

[react_loop.py](../../agency/native_harness/react_loop.py) runs a ReAct loop: ask the model for its next action, execute requested tools, append their results and ask again. Tool calls in a response execute serially. A response without tool calls ends the loop; exhausting the model-turn budget returns an error.

Built-in coding tools live in [tools.py](../../agency/native_harness/tools.py) and use local filesystem and subprocess operations. Additional tools come through [mcp_client.py](../../agency/native_harness/mcp_client.py), using MCP (Model Context Protocol) discovery and calls. When connected to Agency, [bridge_client.py](../../agency/native_harness/bridge_client.py) performs tool-policy checks and reports activity to the host.

## Context and continuity

[compaction.py](../../agency/native_harness/compaction.py) prunes large tool output and summarizes older messages near the context limit. Large tool results can also be saved to files so the conversation carries a reference. [session.py](../../agency/native_harness/session.py) saves a JSON message snapshot for a session ID, allowing a fresh CLI process to continue the conversation. Progress files expose liveness and partial text to the adapter.

[cli.py](../../agency/native_harness/cli.py) assembles the clients, prompt, session and loop. [llm_client.py](../../agency/native_harness/llm_client.py) supports direct standalone model access or Agency gateway access. [profiling.py](../../agency/native_harness/profiling.py) and [annotations.py](../../agency/native_harness/annotations.py) add activity reporting and optional tool-annotation instrumentation.

## Separation from the host runtime

The launcher puts `agency/` on `PYTHONPATH` and invokes `native_harness` as a top-level package. As explained in [__init__.py](../../agency/native_harness/__init__.py), this avoids importing Agency's root package and its host dependencies inside the harness process.

The loop owns model reasoning and tool sequencing. Scheduling, request cancellation arbitration, resource allocation and filesystem checkpoint publication stay in the surrounding runtime. Its JSON conversation snapshot is distinct from an exported agent or a CRIU process image.
