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

The tables below list dataclass fields and defaults, including fresh mutable default factories. Check the linked source when updating configuration. Per-backend/provider options are intentionally preserved rather than reduced to a universal settings model.

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

### llmconfig

| Field | Type | Default |
| --- | --- | --- |
| `provider` | `str &#124; None` | `None` |
| `model` | `str` | `''` |
| `api_key` | `str &#124; None` | `None` |
| `base_url` | `str &#124; None` | `None` |
| `region` | `str &#124; None` | `None` |
| `context_limit` | `int &#124; None` | `None` |
| `temperature` | `float &#124; None` | `None` |
| `reasoning_effort` | `str &#124; None` | `None` |
| `max_completion_tokens` | `int &#124; None` | `None` |
| `max_tokens` | `int &#124; None` | `None` |
| `top_p` | `float &#124; None` | `None` |
| `frequency_penalty` | `float &#124; None` | `None` |
| `presence_penalty` | `float &#124; None` | `None` |
| `n` | `int &#124; None` | `None` |
| `stop` | `str &#124; list &#124; None` | `None` |
| `logprobs` | `bool &#124; int &#124; None` | `None` |
| `seed` | `int &#124; None` | `None` |
| `extra_body` | `dict &#124; None` | `None` |
| `top_k` | `int &#124; None` | `None` |
| `repetition_penalty` | `float &#124; None` | `None` |
| `min_p` | `float &#124; None` | `None` |
| `min_tokens` | `int &#124; None` | `None` |
| `guided_json` | `dict &#124; None` | `None` |
| `guided_regex` | `str &#124; None` | `None` |
| `workspace_id` | `str &#124; None` | `None` |
| `aws_access_key` | `str &#124; None` | `None` |
| `aws_secret_key` | `str &#124; None` | `None` |
| `aws_session_token` | `str &#124; None` | `None` |
| `aws_profile` | `str &#124; None` | `None` |
| `aws_region` | `str &#124; None` | `None` |
| `model_listing_timeout_seconds` | `float` | `10.0` |
| `default_max_tokens` | `int` | `32000` |
| `sdk_max_retries` | `int` | `2` |
| `codex_request_max_retries` | `int &#124; None` | `None` |
| `codex_stream_max_retries` | `int &#124; None` | `None` |
| `max_retries` | `int` | `12` |
| `idle_timeout` | `float` | `900.0` |
| `stream_timeout` | `float` | `1200.0` |
| `retry_sleep_s` | `float` | `2` |
| `http_connect_timeout` | `float` | `10.0` |
| `http_write_timeout` | `float` | `10.0` |
| `http_pool_timeout` | `float` | `10.0` |
| `live_redraw_char_threshold` | `int` | `100` |
| `rate_limit_base_backoff_s` | `float` | `5.0` |
| `rate_limit_max_backoff_s` | `float` | `80.0` |
| `rate_limit_retry_after_jitter_s` | `float` | `5.0` |
| `default_context_limit` | `int` | `200000` |
| `unusable_response_retries` | `int` | `2` |
| `unusable_response_backoff_s` | `float` | `1.0` |
| `fail_on_empty_response` | `bool` | `True` |
| `strict_stream_termination` | `bool` | `True` |
| `truncation_retries` | `int` | `0` |
| `truncation_output_multiplier` | `float` | `2.0` |
| `truncation_max_output_tokens` | `int &#124; None` | `None` |
| `reasoning_summary` | `str &#124; None` | `None` |
| `replay_db_path` | `str &#124; None` | `None` |
| `timing_mode` | `str` | `'exact'` |
| `constant_ttft_s` | `float` | `0.3` |
| `constant_tpot_s` | `float` | `0.02` |
| `poisson_rate_hz` | `float` | `10.0` |
| `poisson_ttft_mean_s` | `float` | `0.3` |
| `poisson_seed` | `int &#124; None` | `None` |
| `timing_fn` | `Callable &#124; None` | `None` |
| `replay_dispatch_hook` | `Callable[[dict], None] &#124; None` | `None` |

### sandboxconfig

| Field | Type | Default |
| --- | --- | --- |
| `base_image` | `str` | `'docker.io/library/python:3.12-slim'` |
| `persistent` | `bool` | `False` |
| `hibernation_diagnostics` | `bool` | `False` |
| `checkpoint_diagnostics` | `bool` | `False` |
| `checkpoint_diagnostics_extended` | `bool` | `False` |
| `mounts` | `dict[str, tuple[str, str, str]]` | `fresh dict()` |
| `flags` | `list[str]` | `fresh list()` |
| `gpu_passthrough` | `bool` | `True` |
| `harness_python_path` | `str &#124; None` | `None` |
| `backend` | `str` | `'auto'` |
| `checkpoint_backend` | `str` | `'image_commit'` |
| `checkpoint_fast_resume` | `bool` | `False` |
| `checkpoint_zfs_parent` | `str &#124; None` | `None` |
| `checkpoint_setup_timeout_s` | `float` | `900` |
| `checkpoint_irmap_paths` | `tuple[str, ...]` | `('/tmp/agency-0', '/workspace', '/testbed')` |
| `cpuset_cpus` | `str &#124; None` | `None` |
| `cpuset_mems` | `str &#124; None` | `None` |
| `inspect_timeout_s` | `float` | `120` |
| `exec_quick_timeout_s` | `float` | `120` |
| `docker_run_timeout_s` | `float` | `120` |
| `docker_rm_timeout_s` | `float` | `120` |
| `file_io_timeout_s` | `float` | `120` |
| `image_timeout_s` | `float` | `120` |
| `commit_timeout_s` | `float` | `120` |
| `keyring_wait_timeout_s` | `float` | `120` |
| `unkillable_child_grace_s` | `float` | `10` |
| `container_limit_floor` | `int` | `4` |
| `container_limit_buffer` | `int` | `5` |
| `container_limit_fallback` | `int` | `200` |
| `conflict_retry_max_attempts` | `int` | `8` |
| `keyring_poll_interval_s` | `float` | `5` |
| `container_removal_wait_s` | `float` | `10` |
| `container_removal_poll_interval_s` | `float` | `0.5` |
| `conflict_retry_backoff_base_s` | `float` | `0.5` |
| `docker_stop_timeout_s` | `float` | `120` |
| `docker_stop_grace_s` | `float` | `0` |
| `docker_start_timeout_s` | `float` | `120` |
| `stop_inspect_timeout_s` | `float` | `30` |
| `commit_retry_attempts` | `int` | `3` |
| `commit_retry_backoff_s` | `float` | `1` |
| `stop_ps_check_timeout_s` | `float` | `10` |
| `rm_retry_attempts` | `int` | `3` |
| `rm_retry_backoff_s` | `float` | `1` |
| `checkpoint_squash_max_depth` | `int` | `100` |
| `squash_timeout_s` | `float` | `600` |

### orchestratorconfig

| Field | Type | Default |
| --- | --- | --- |
| `max_concurrent_engines` | `int &#124; None` | `None` |
| `db_path` | `str &#124; None` | `None` |
| `flush_batch_size` | `int` | `500` |
| `flush_interval_s` | `float` | `1.0` |

### resourcesconfig

| Field | Type | Default |
| --- | --- | --- |
| `idle_cpus` | `float &#124; None` | `8.0` |
| `idle_memory` | `str &#124; None` | `None` |

### agentconfig

| Field | Type | Default |
| --- | --- | --- |
| `log_dir` | `str &#124; None` | `None` |
| `output_dir` | `str &#124; None` | `None` |
| `checkpoint_save_timeout_s` | `int` | `600` |
| `checkpoint_load_timeout_s` | `int` | `600` |
| `harness` | `str` | `'native'` |
| `annotation_arm` | `str` | `'baseline'` |
| `native_trace_file` | `str &#124; None` | `None` |
| `experiment_run_id` | `str &#124; None` | `None` |
| `experiment_agent_id` | `str &#124; None` | `None` |

### schemaconfig

| Field | Type | Default |
| --- | --- | --- |
| `input_offload_chars` | `int` | `40000` |
| `offload_context_fraction` | `float` | `0.1` |
| `chars_per_token` | `int` | `4` |

### skillconfig

| Field | Type | Default |
| --- | --- | --- |
| `react_max_steps` | `int` | `4096` |
| `agbinary_validate_exec_timeout` | `float` | `5` |
| `error_log_truncate` | `int` | `300` |
| `last_output_log_truncate` | `int` | `2000` |

### toolconfig

| Field | Type | Default |
| --- | --- | --- |
| `timeout_s` | `float` | `1800` |
| `output_offload_chars` | `int` | `40000` |
| `offload_id_prefix_len` | `int` | `12` |

### harnessadapterconfig

| Field | Type | Default |
| --- | --- | --- |
| `session_resume_id` | `str &#124; None` | `None` |
| `binary_path` | `str &#124; None` | `None` |
| `allow_subagents` | `bool` | `False` |
| `stream_keepalive_s` | `float` | `15.0` |

### ptraceconfig

| Field | Type | Default |
| --- | --- | --- |
| `syscalls` | `tuple[str, ...]` | `('execve', 'execveat', 'connect', 'bind', 'sendto')` |
| `file_access` | `bool` | `False` |
| `profiler` | `str &#124; None` | `None` |
| `disable_harness_native_sandbox` | `bool` | `True` |

### dataloggerconfig

| Field | Type | Default |
| --- | --- | --- |
| `db_path` | `str &#124; None` | `None` |
| `flush_batch_size` | `int` | `20` |
| `flush_interval_s` | `float` | `0.2` |

### hostserverconfig

| Field | Type | Default |
| --- | --- | --- |
| `uds_path` | `str &#124; None` | `None` |
| `startup_timeout_s` | `float` | `10.0` |
| `shutdown_timeout_s` | `float` | `10.0` |


Process helper constants in the configuration module (e.g. Docker semaphore and minimum CPU/memory floors) are implementation constants, not configuration namespaces or tunable per-agent fields.

## Source reference

[Source: agconfig.py](../../agency/configs/agconfig.py)

[`agency.configs.agconfig.agconfig`](../../agency/configs/agconfig.py#L354): `__init__`, `clone`, `update`, `safe_snapshot`.

[Source: agconfig.py](../../agency/configs/agconfig.py)

[`agency.configs.agconfig.confignamespace`](../../agency/configs/agconfig.py#L42): `clone`, `update`, `safe_snapshot`.

[Source: agconfig.py](../../agency/configs/agconfig.py)

[`agency.configs.agconfig.llmconfig`](../../agency/configs/agconfig.py#L78)

[Source: agconfig.py](../../agency/configs/agconfig.py)

[`agency.configs.agconfig.sandboxconfig`](../../agency/configs/agconfig.py#L172): `add_mount`, `remove_mount`.

[Source: agconfig.py](../../agency/configs/agconfig.py)

[`agency.configs.agconfig.orchestratorconfig`](../../agency/configs/agconfig.py#L249)

[Source: agconfig.py](../../agency/configs/agconfig.py)

[`agency.configs.agconfig.resourcesconfig`](../../agency/configs/agconfig.py#L259)

[Source: agconfig.py](../../agency/configs/agconfig.py)

[`agency.configs.agconfig.agentconfig`](../../agency/configs/agconfig.py#L267)

[Source: agconfig.py](../../agency/configs/agconfig.py)

[`agency.configs.agconfig.schemaconfig`](../../agency/configs/agconfig.py#L282)

[Source: agconfig.py](../../agency/configs/agconfig.py)

[`agency.configs.agconfig.skillconfig`](../../agency/configs/agconfig.py#L291)

[Source: agconfig.py](../../agency/configs/agconfig.py)

[`agency.configs.agconfig.toolconfig`](../../agency/configs/agconfig.py#L301)

[Source: agconfig.py](../../agency/configs/agconfig.py)

[`agency.configs.agconfig.harnessadapterconfig`](../../agency/configs/agconfig.py#L310)

[Source: agconfig.py](../../agency/configs/agconfig.py)

[`agency.configs.agconfig.ptraceconfig`](../../agency/configs/agconfig.py#L322)

[Source: agconfig.py](../../agency/configs/agconfig.py)

[`agency.configs.agconfig.dataloggerconfig`](../../agency/configs/agconfig.py#L333)

[Source: agconfig.py](../../agency/configs/agconfig.py)

[`agency.configs.agconfig.hostserverconfig`](../../agency/configs/agconfig.py#L344)
