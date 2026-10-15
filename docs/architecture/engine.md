# engine — one request's execution lifecycle

`agency/engine/` connects a scheduled skill request to the sandbox harness. An `AgentEngine` belongs to one request and owns its attempts, host services, output handling and cleanup. The [orchestrator](orchestrator.md) decides when it starts and settles its result afterward.

## Internal responsibilities

| Source | Responsibility |
| --- | --- |
| [engine.py](../../agency/engine/engine.py) | Sandbox lock, prompt construction, attempts, output validation, commit and discard. |
| [harness_daemon_launcher.py](../../agency/engine/harness_daemon_launcher.py) | Prepare and start a sandbox daemon, check readiness and obtain a control handle. |
| [clients/](../../agency/engine/clients/harness_interaction_client.py) | Host-to-daemon RPC for execution and request controls. |
| [host_servers/](../../agency/engine/host_servers/host_server_manager.py) | Host-side model, MCP tool, policy and profiling services. |

## The host service boundary

`HostServerManager` assembles three services behind a Unix-domain socket: `LlmHandlerServer` dispatches models through [llm](llm.md), `HostMcpServer` exposes the skill's host tools and output submission, and `HostInteractionServer` evaluates policy callbacks and records harness activity. The [sandbox harness gateway](harness.md) forwards to these services.

Each attempt binds a fresh token. Admission checks reject stale callback traffic, and a lease tracks callbacks already admitted, including streaming responses. Retiring the token closes admission and waits for those callbacks to drain. This prevents callbacks from one attempt being attributed to its successor.

## Execute, then publish

The engine holds the sandbox lock across execution and filesystem teardown. This also serializes requests from different agents that share the same sandbox facade. It prepares schema-backed input files and turns skill instructions, user content and retained messages into a prompt.

A request can contain several harness attempts. Missing structured output can trigger repair prompts in the same working sandbox; successful attempts can supply session state for the next attempt. The engine validates and recovers final output before considering the request successful.

Before committing, it checks cancellation and claims completion through the orchestrator. Host services close before filesystem commit. Harness session snapshots and retained-message cursors remain staged until commit returns; failure drops staging and removes the working runtime. The engine then reports its outcome to the scheduler for public settlement.

The current implementation propagates commit exceptions but does not check a false return from `sandbox.commit()`. Host-tool, provider and external mount effects also lie outside this transaction. [Engine tests](../../tests/engine/test_engine.py) cover execution and staging; [host service tests](../../tests/engine/host_servers_tests/test_host_server_manager.py) cover token lifetimes and callback drain.
