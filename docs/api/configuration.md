# Configuration reference

```python
from agency.configs.agconfig import (
    agconfig, llmconfig, sandboxconfig, orchestratorconfig, resourcesconfig,
    agentconfig, schemaconfig, skillconfig, toolconfig, harnessadapterconfig,
    ptraceconfig, dataloggerconfig, hostserverconfig, confignamespace,
)
```

`from agency import agconfig` is also supported. Each namespace is a slots dataclass with keyword fields/defaults shown below. `agconfig(*namespaces)` creates fresh defaults for all namespaces and recognizes supplied namespace objects by type, not argument order. Duplicate or unknown namespaces raise `TypeError`. **Supplied namespace objects are stored directly**; clone before sharing mutable configuration. Unknown constructor fields raise `TypeError`.

Installation and provider/harness setup are in [getting started](../getting-started.md) and the [configuration guide](../guides/configuration.md).

## Common options

| Namespace.attribute | Purpose and effect |
| --- | --- |
| `llm.provider`, `model`, `api_key`, `base_url` | Select backend/model and authentication/endpoint. Defaults None, empty model, None, None. Use explicit credentials as in onboarding; not all providers infer them from host environment. |
| `llm.context_limit`, `max_completion_tokens` | Model input/output budgeting. Defaults None; fallback limits 200,000/32,000 respectively. `max_tokens` is a deprecated output-limit alias. Provider-specific acceptance varies. |
| `llm.temperature`, `reasoning_effort`, `extra_body` | Optional generation/provider controls, all None. Unsupported controls may be ignored or rejected by the backend/model. |
| `agent.harness` | Constructor selection, default `native`; explicit Agent harness argument overrides. Changing config later does not change the agent's chosen harness. |
| `sandbox.base_image`, `backend` | Sandbox seed image and backend selection (default python:3.12-slim and auto). Apply before sandbox creation; see checkpoint/backend limits. |
| `sandbox.gpu_passthrough`, `cpuset_cpus`, `cpuset_mems` | GPU visibility policy and initial container limits. GPU passthrough defaults true; disable for an explicit CPU-only application. CPU/node affinity constraints default None; idle resource limits set initial CPU/memory caps. |
| `sandbox.mounts` | Named host-to-sandbox mappings; fresh dict by default. Mounted writes survive rollback/save. Use add_mount/remove_mount before starting sandbox. |
| `orchestrator.max_concurrent_engines` | Process admission ceiling, default None (unbounded admission). Positive int required if supplied. Configure before first orchestrator initialization. |
| `resources.idle_cpus`, `idle_memory` | Idle sandbox limits (8.0 CPUs, no memory cap). These are per-sandbox resting limits, not spare host capacity. See the initial resource-pool issue below. |
| `skill.react_max_steps` | Default request step budget, 4096; per-run positive max_steps overrides. |
| `agent.log_dir`, `output_dir`; logger/orchestrator `db_path` | Host artifacts and SQLite destinations, defaults computed by owning objects when None. Configure before construction. |
| `tool.timeout_s` | Advertised default 1800 seconds; **direct agtool timeout is not enforced**, nor is this a universal MCP execution deadline. |
| `sandbox.checkpoint_backend` | `image_commit` default; `cow_zfs` needs explicit local host capability/setup and restricts portability. |

## Mutation and scope

`clone()` deep-copies a namespace or full config. `update(**fields)` on a namespace validates field names and returns self; `agconfig.update(llm={...}, sandbox={...})` validates namespace names, then updates each and returns self. Unknown fields/namespaces raise `TypeError`. Values are not comprehensively validated/coerced just because their annotations declare a type.

`safe_snapshot()` returns JSON-safe public values. It omits llm api_key, aws_access_key, aws_secret_key and aws_session_token plus private/non-JSON-safe values. This is configuration metadata, not a sanitizer for arbitrary prompts, files or session data.

| Operation | Existing object/new-object effect |
| --- | --- |
| `Agent(..., agconfig=cfg)` | Deep clone for that agent; later changes to cfg do not affect it. |
| `agent.change_config(cfg)` | Deep clone and cascade to its logger/sandbox/engine. Does not change harness or global orchestrator. Output-path properties read the new config, but an existing output mount stays at its original location. Apply between runs. |
| `agteam(..., agconfig=cfg)` / `team.change_config(cfg)` | Clone; change cascades to tracked agents and affects new team agents. |
| `agSandbox(..., agconfig=cfg)` / `sandbox.change_config(cfg)` | Clone and cascade backend settings. Does not recreate an already-running container to retrofit mounts/image/backend. |
| `agResourcePool(..., agconfig=cfg)` / pool.change_config | Clone; idle limits are read when applied. Hardware totals are construction-time values. |
| `get_orchestrator(cfg)` | Initializes a process singleton once; later arguments are ignored. Direct orchestrator construction/change_config retains supplied config instead of cloning. change_config cascades scoped logger and pool settings. |
| `agDataLogger(cfg)` / logger.change_config | Retains supplied config; a change can fill an omitted db_path in that supplied object from the existing path. Requires explicit db_path on first use. It does not transparently move an open connection. |

Initial orchestrator pool construction currently omits the supplied config; its resources namespace starts with defaults until an explicit orchestrator/pool change_config. Increasing max_concurrent_engines after construction does not resize the executor's original worker ceiling. [Known issues](contracts.md) captures evidence; choose stable process config before starting work.

`sandboxconfig.add_mount(name, host_path, container_path, mode="rw")` stores `(str(host_path), container_path, mode)` and returns self; no path/mode validation here. `remove_mount(name)` returns self, unknown names harmless. Don't rely on post-start mount edits to alter the live filesystem.

## Complete namespace fields and defaults

The tables below are generated from dataclass source, including fresh mutable default factories. Per-backend/provider options are intentionally preserved rather than reduced to a universal settings model.

- llm: selection, generation controls, credential fields, retry/network limits and mock replay controls. Retry/stream fields govern gateway requests; max_tokens is deprecated.
- sandbox: filesystem/runtime/checkpoint selection, initial limits, command/file/image timeouts and retry/hibernation settings. Persistent containers keep working state; checkpoint mounts stay external. Host-profile defaults apply through the supported launcher.
- orchestrator: process admission and global persistence batching.
- resources: each idle sandbox's CPU/memory limits.
- agent: harness, paths, checkpoint archive timeouts, native annotation/experiment fields.
- schema: long-input offloading thresholds, model context fraction and character/token estimate.
- skill: step budget, binary validation probe timeout and error/output log truncation.
- tool: advertised timeout, large output offloading and generated identifier length.
- harness_adapter: CLI path, session resume hint and subagent permission.
- ptrace: traced syscall names, opt-in file-access metadata, native sandbox switch; profiler field is reserved and has no current heavyweight-profiler effect.
- data_logger: per-object SQLite path/batching.
- host_server: per-agent UDS path and service startup/shutdown deadlines.

<!-- CONFIG_TABLES -->

Process helper constants in the configuration module (e.g. Docker semaphore and minimum CPU/memory floors) are implementation constants, not configuration namespaces or tunable per-agent fields.

## Source signatures

[Source: agconfig.py](../../agency/configs/agconfig.py)

::: agency.configs.agconfig.agconfig
    options:
      members: ["__init__", "clone", "update", "safe_snapshot"]

[Source: agconfig.py](../../agency/configs/agconfig.py)

::: agency.configs.agconfig.confignamespace
    options:
      members: ["clone", "update", "safe_snapshot"]

[Source: agconfig.py](../../agency/configs/agconfig.py)

::: agency.configs.agconfig.llmconfig
    options:
      members: []

[Source: agconfig.py](../../agency/configs/agconfig.py)

::: agency.configs.agconfig.sandboxconfig
    options:
      members: ["add_mount", "remove_mount"]

[Source: agconfig.py](../../agency/configs/agconfig.py)

::: agency.configs.agconfig.orchestratorconfig
    options:
      members: []

[Source: agconfig.py](../../agency/configs/agconfig.py)

::: agency.configs.agconfig.resourcesconfig
    options:
      members: []

[Source: agconfig.py](../../agency/configs/agconfig.py)

::: agency.configs.agconfig.agentconfig
    options:
      members: []

[Source: agconfig.py](../../agency/configs/agconfig.py)

::: agency.configs.agconfig.schemaconfig
    options:
      members: []

[Source: agconfig.py](../../agency/configs/agconfig.py)

::: agency.configs.agconfig.skillconfig
    options:
      members: []

[Source: agconfig.py](../../agency/configs/agconfig.py)

::: agency.configs.agconfig.toolconfig
    options:
      members: []

[Source: agconfig.py](../../agency/configs/agconfig.py)

::: agency.configs.agconfig.harnessadapterconfig
    options:
      members: []

[Source: agconfig.py](../../agency/configs/agconfig.py)

::: agency.configs.agconfig.ptraceconfig
    options:
      members: []

[Source: agconfig.py](../../agency/configs/agconfig.py)

::: agency.configs.agconfig.dataloggerconfig
    options:
      members: []

[Source: agconfig.py](../../agency/configs/agconfig.py)

::: agency.configs.agconfig.hostserverconfig
    options:
      members: []
