# Library API

Start with [getting started](../getting-started.md) for installation, provider credentials and a native first run. This reference describes the current checkout, including demonstrated limitations. Each group links to its Python source; behavioral explanations and reference tables are maintained in Markdown.

| Task | Reference |
| --- | --- |
| Submit work, manage history, save or fork agents | [Agents and lifecycle](agents.md) |
| Wait, inspect failures, control requests | [Results and cancellation](results.md) |
| Define skills and executable tools | [Skills, tools and policies](skills.md) |
| Validate shapes and move files or images | [Schemas and types](types.md) |
| Compose parallel work and teams | [Workflows](workflows.md) |
| Set or update configuration | [Configuration](configuration.md) |
| Execute commands and own checkpoints | [Sandboxes and checkpoints](sandboxes.md) |
| Inspect scheduling, resources, logs and traces | [Advanced runtime and observability](runtime.md) |

Application entry points are `Agent`/`agent`, `agdata`, skills, tools, schemas, types, workflows and namespaced configuration. Direct sandbox use requires owning its cleanup. Scheduler, context internals and instrumentation APIs are advanced: avoid modifying them during submitted work. [Known contract issues](contracts.md) records differences between intended and demonstrated behavior.

## Root exports

The [inventory manifest](public-api.json) maps every current root export and additional supported import to a destination. Update the manifest and reference pages when public exports or supported members change. `Agent` is the exact alias of `agent`, not a subclass.

| Root import | Reference |
| --- | --- |
| `from agency import agconfig` | [agency.configs.agconfig.agconfig](configuration.md) |
| `from agency import agdata` | [agency.agdata.agdata](results.md) |
| `from agency import agerror` | [agency.agdata.agerror](results.md) |
| `from agency import agcanceled` | [agency.agdata.agcanceled](results.md) |
| `from agency import agcontext` | [agency.agcontext.agcontext](runtime.md) |
| `from agency import MessageBlock` | [agency.llm.message_store.MessageBlock](agents.md) |
| `from agency import agschema` | [agency.agschema.agschema](types.md) |
| `from agency import agtype` | [agency.agtype.agtype](types.md) |
| `from agency import agfile` | [agency.agtype.agfile](types.md) |
| `from agency import agbinary` | [agency.agtype.agbinary](types.md) |
| `from agency import agimage` | [agency.agtype.agimage](types.md) |
| `from agency import agrawstring` | [agency.agtype.agrawstring](types.md) |
| `from agency import agpath` | [agency.agtype.agpath](types.md) |
| `from agency import AgError` | [agency.agdata.AgError](results.md) |
| `from agency import agskill` | [agency.agskill.agskill](skills.md) |
| `from agency import agtool` | [agency.agtool.agtool](skills.md) |
| `from agency import agDataLogger` | [agency.observability.agdatalogger.agDataLogger](runtime.md) |
| `from agency import agent` | [agency.agent.agent](agents.md) |
| `from agency import Agent` | [agency.agent.agent](agents.md) |
| `from agency import agmap` | [agency.utils.agmap.agmap](workflows.md) |
| `from agency import agtask` | [agency.utils.agmap.agtask](workflows.md) |
| `from agency import agteam` | [agency.agteam.agteam](workflows.md) |
| `from agency import agsync` | [agency.utils.agsync.agsync](workflows.md) |
| `from agency import agSandbox` | [agency.sandbox.agsandbox.agSandbox](sandboxes.md) |
| `from agency import agResourcePool` | [agency.orchestrator.agresources.agResourcePool](runtime.md) |
| `from agency import GlobalAgentOrchestrator` | [agency.orchestrator.orchestrator.GlobalAgentOrchestrator](runtime.md) |
| `from agency import ExecutionScheduler` | [agency.orchestrator.scheduler.ExecutionScheduler](runtime.md) |
| `from agency import OrchestratorSnapshot` | [agency.orchestrator.orchestrator.OrchestratorSnapshot](runtime.md) |
| `from agency import get_orchestrator` | [agency.orchestrator.orchestrator.get_orchestrator](runtime.md) |
| `from agency import get_container_runtime` | [agency.sandbox.container.get_container_runtime](sandboxes.md) |
| `from agency import sigterm_as_exit` | [agency.utils.agutil.sigterm_as_exit](workflows.md) |
| `from agency import agprof` | [agency.observability.profiler.agprof](runtime.md) |

## Additional imports

| Import path | Reference |
| --- | --- |
| `agency.agpolicy.agpolicy` | [skills](skills.md) |
| `agency.harness._syscall_event.agsyscallevent` | [skills](skills.md) |
| `agency.configs.agconfig.confignamespace` | [configuration](configuration.md) |
| `agency.configs.agconfig.llmconfig` | [configuration](configuration.md) |
| `agency.configs.agconfig.sandboxconfig` | [configuration](configuration.md) |
| `agency.configs.agconfig.orchestratorconfig` | [configuration](configuration.md) |
| `agency.configs.agconfig.resourcesconfig` | [configuration](configuration.md) |
| `agency.configs.agconfig.agentconfig` | [configuration](configuration.md) |
| `agency.configs.agconfig.schemaconfig` | [configuration](configuration.md) |
| `agency.configs.agconfig.skillconfig` | [configuration](configuration.md) |
| `agency.configs.agconfig.toolconfig` | [configuration](configuration.md) |
| `agency.configs.agconfig.harnessadapterconfig` | [configuration](configuration.md) |
| `agency.configs.agconfig.ptraceconfig` | [configuration](configuration.md) |
| `agency.configs.agconfig.dataloggerconfig` | [configuration](configuration.md) |
| `agency.configs.agconfig.hostserverconfig` | [configuration](configuration.md) |
| `agency.sandbox.checkpoint.CheckpointHandle` | [sandboxes](sandboxes.md) |
| `agency.sandbox.checkpoint.CheckpointCapabilityError` | [sandboxes](sandboxes.md) |
| `agency.orchestrator.orchestrator.peek_orchestrator` | [runtime](runtime.md) |
| `agency.orchestrator.agresources.GpuWaitCancelled` | [runtime](runtime.md) |
| `agency.observability.agdatalogger.read_llm_exchanges` | [runtime](runtime.md) |
| `agency.observability.agdatalogger.reconstruct_llm_exchanges` | [runtime](runtime.md) |
| `agency.observability.agdatalogger.resolve_global_db_path` | [runtime](runtime.md) |
| `agency.observability.agwebui.agwebui` | [runtime](runtime.md) |

## Subpackage exports and aliases

Public-facing subpackages below are inventoried separately. Backend/engine extension protocols have explicit exclusions.

| Import path | Destination or rationale |
| --- | --- |
| `agency.configs.agconfig` | [configuration](configuration.md) |
| `agency.orchestrator.GlobalAgentOrchestrator` | [runtime](runtime.md) |
| `agency.orchestrator.ExecutionScheduler` | [runtime](runtime.md) |
| `agency.orchestrator.OrchestratorSnapshot` | [runtime](runtime.md) |
| `agency.orchestrator.get_orchestrator` | [runtime](runtime.md) |
| `agency.orchestrator.peek_orchestrator` | [runtime](runtime.md) |
| `agency.sandbox.AgSandboxBackendFields` | Backend implementation, selection helper, cache/provisioning or orphan-reaper utility. Application sandbox operations use agSandbox and get_container_runtime; not a normal application API. |
| `agency.sandbox.agsandbox_backend` | Backend implementation, selection helper, cache/provisioning or orphan-reaper utility. Application sandbox operations use agSandbox and get_container_runtime; not a normal application API. |
| `agency.sandbox.backend_for_image_kind` | Backend implementation, selection helper, cache/provisioning or orphan-reaper utility. Application sandbox operations use agSandbox and get_container_runtime; not a normal application API. |
| `agency.sandbox._ContainerAlreadyRunning` | Backend implementation, selection helper, cache/provisioning or orphan-reaper utility. Application sandbox operations use agSandbox and get_container_runtime; not a normal application API. |
| `agency.sandbox._ContainerBackendBase` | Backend implementation, selection helper, cache/provisioning or orphan-reaper utility. Application sandbox operations use agSandbox and get_container_runtime; not a normal application API. |
| `agency.sandbox._RUN_ID` | Backend implementation, selection helper, cache/provisioning or orphan-reaper utility. Application sandbox operations use agSandbox and get_container_runtime; not a normal application API. |
| `agency.sandbox.get_container_runtime` | [sandboxes](sandboxes.md) |
| `agency.sandbox.keyring_quota` | Backend implementation, selection helper, cache/provisioning or orphan-reaper utility. Application sandbox operations use agSandbox and get_container_runtime; not a normal application API. |
| `agency.sandbox.reap_orphaned_containers` | Backend implementation, selection helper, cache/provisioning or orphan-reaper utility. Application sandbox operations use agSandbox and get_container_runtime; not a normal application API. |
| `agency.sandbox.seed_cache_from_image` | Backend implementation, selection helper, cache/provisioning or orphan-reaper utility. Application sandbox operations use agSandbox and get_container_runtime; not a normal application API. |
| `agency.sandbox._DockerBackend` | Backend implementation, selection helper, cache/provisioning or orphan-reaper utility. Application sandbox operations use agSandbox and get_container_runtime; not a normal application API. |
| `agency.sandbox._PodmanBackend` | Backend implementation, selection helper, cache/provisioning or orphan-reaper utility. Application sandbox operations use agSandbox and get_container_runtime; not a normal application API. |
| `agency.sandbox._ChrootBackend` | Backend implementation, selection helper, cache/provisioning or orphan-reaper utility. Application sandbox operations use agSandbox and get_container_runtime; not a normal application API. |
| `agency.sandbox.chroot_available` | Backend implementation, selection helper, cache/provisioning or orphan-reaper utility. Application sandbox operations use agSandbox and get_container_runtime; not a normal application API. |
| `agency.observability.profiler.agprof` | [runtime](runtime.md) |
| `agency.observability.profiler.annotate` | [runtime](runtime.md) |
| `agency.observability.profiler.cancel_external_span` | Advanced external-span instrumentation protocol; use agprof.span/session for application tracing. Explicitly excluded in the profiler member inventory. |
| `agency.observability.profiler.enabled` | [runtime](runtime.md) |
| `agency.observability.profiler.interrupt_external_span` | Advanced external-span instrumentation protocol; use agprof.span/session for application tracing. Explicitly excluded in the profiler member inventory. |
| `agency.observability.profiler.next_index` | [runtime](runtime.md) |
| `agency.observability.profiler.profile_records` | [runtime](runtime.md) |
| `agency.observability.profiler.profile_scope` | [runtime](runtime.md) |
| `agency.observability.profiler.session` | [runtime](runtime.md) |
| `agency.observability.profiler.spawn_traced` | [runtime](runtime.md) |
| `agency.observability.profiler.span` | [runtime](runtime.md) |
| `agency.observability.profiler.start` | [runtime](runtime.md) |
| `agency.observability.profiler.start_external_span` | Advanced external-span instrumentation protocol; use agprof.span/session for application tracing. Explicitly excluded in the profiler member inventory. |
| `agency.observability.profiler.stop` | [runtime](runtime.md) |
| `agency.observability.profiler.summary_metrics` | [runtime](runtime.md) |
| `agency.observability.profiler.summary_table` | [runtime](runtime.md) |
| `agency.observability.profiler.thread_name` | [runtime](runtime.md) |
| `agency.observability.profiler.workload` | [runtime](runtime.md) |
| `agency.llm.agllm` | Backend/engine/adapter extension protocol. Applications select implementations through namespaced configuration and Agent; diagnostic examples may use these internals directly. |
| `agency.llm.BAD_REQUEST_EXCS` | Backend/engine/adapter extension protocol. Applications select implementations through namespaced configuration and Agent; diagnostic examples may use these internals directly. |
| `agency.llm.API_CONN_EXCS` | Backend/engine/adapter extension protocol. Applications select implementations through namespaced configuration and Agent; diagnostic examples may use these internals directly. |
| `agency.llm.RATE_LIMIT_EXCS` | Backend/engine/adapter extension protocol. Applications select implementations through namespaced configuration and Agent; diagnostic examples may use these internals directly. |
| `agency.llm.API_ERROR_EXCS` | Backend/engine/adapter extension protocol. Applications select implementations through namespaced configuration and Agent; diagnostic examples may use these internals directly. |
| `agency.llm.ProviderResponseError` | Backend response validation exception handled by the engine and harness adapters; application failures are exposed through agerror. |
| `agency.llm.StreamInterruptedError` | Backend response validation exception handled by the engine and harness adapters; application failures are exposed through agerror. |
| `agency.llm.UnusableResponseError` | Backend response validation exception handled by the engine and harness adapters; application failures are exposed through agerror. |
| `agency.engine.AgentEngine` | Backend/engine/adapter extension protocol. Applications select implementations through namespaced configuration and Agent; diagnostic examples may use these internals directly. |
| `agency.engine.HostInteractionServer` | Backend/engine/adapter extension protocol. Applications select implementations through namespaced configuration and Agent; diagnostic examples may use these internals directly. |
| `agency.harness.adapters.HarnessAdapter` | Backend/engine/adapter extension protocol. Applications select implementations through namespaced configuration and Agent; diagnostic examples may use these internals directly. |

## Exclusions

These diagnostic integrations are used by live examples but are not application APIs.

| Example import | Rationale |
| --- | --- |
| `agency.engine.host_servers.host_interaction_server.HostInteractionServer` | Advanced diagnostic example integration; implementation adapter, transport, server, or private test hook. Not a supported application API; application entry points are Agent, agpolicy, agwebui and agprof. |
| `agency.engine.host_servers.llm_handler_server.LlmHandlerServer` | Advanced diagnostic example integration; implementation adapter, transport, server, or private test hook. Not a supported application API; application entry points are Agent, agpolicy, agwebui and agprof. |
| `agency.harness.adapters.claude_code.ClaudeCodeAdapter` | Advanced diagnostic example integration; implementation adapter, transport, server, or private test hook. Not a supported application API; application entry points are Agent, agpolicy, agwebui and agprof. |
| `agency.harness.adapters.codex.CodexAdapter` | Advanced diagnostic example integration; implementation adapter, transport, server, or private test hook. Not a supported application API; application entry points are Agent, agpolicy, agwebui and agprof. |
| `agency.harness.adapters.native.NativeAdapter` | Advanced diagnostic example integration; implementation adapter, transport, server, or private test hook. Not a supported application API; application entry points are Agent, agpolicy, agwebui and agprof. |
| `agency.harness.clients.host_services_client.HostServicesClient` | Advanced diagnostic example integration; implementation adapter, transport, server, or private test hook. Not a supported application API; application entry points are Agent, agpolicy, agwebui and agprof. |
| `agency.llm.usage_tracker.LlmUsageTracker` | Advanced diagnostic example integration; implementation adapter, transport, server, or private test hook. Not a supported application API; application entry points are Agent, agpolicy, agwebui and agprof. |
| `agency.native_harness.llm_client.LLMClient` | Advanced diagnostic example integration; implementation adapter, transport, server, or private test hook. Not a supported application API; application entry points are Agent, agpolicy, agwebui and agprof. |
| `agency.llm.agllm.agllm` | Advanced diagnostic example integration; implementation adapter, transport, server, or private test hook. Not a supported application API; application entry points are Agent, agpolicy, agwebui and agprof. |
| `agency.llm.openai._OpenAICompatibleBackend` | Advanced diagnostic example integration; implementation adapter, transport, server, or private test hook. Not a supported application API; application entry points are Agent, agpolicy, agwebui and agprof. |

The following public-looking members belong to implementation protocols. Their exclusions are explicit so changes still receive review.

| Object/member | Rationale |
| --- | --- |
| `agency.agtype.agfile.schema_type` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agfile.needs_sandbox` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agfile.prepare` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agfile.recover` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agfile.extra_input_prompt` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agfile.extra_output_prompt` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agfile.get_return_tool_description` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agfile.get_return_tool_value_description` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agfile.validate_output` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agpath.schema_type` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agpath.needs_sandbox` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agpath.extra_input_prompt` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agpath.extra_output_prompt` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agpath.get_return_tool_description` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agpath.get_return_tool_value_description` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agpath.validate_output` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agpath.validate_input_value` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agbinary.schema_type` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agbinary.needs_sandbox` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agbinary.prepare` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agbinary.recover` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agbinary.extra_input_prompt` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agbinary.extra_output_prompt` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agbinary.get_return_tool_description` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agbinary.get_return_tool_value_description` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agbinary.validate_output` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agimage.schema_type` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agimage.needs_sandbox` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agimage.prepare` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agimage.extra_input_prompt` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agimage.build_content_prompt` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agimage.get_return_tool_description` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agimage.get_return_tool_value_description` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agrawstring.schema_type` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.agtype.agrawstring.needs_sandbox` | Concrete type protocol override; shared signatures are documented on agtype, and marker-specific behavior in the type table. |
| `agency.orchestrator.scheduler.ExecutionScheduler.resolve_dependency` | Scheduler protocol consumes private _ExecutionRequest records; use Agent.run(). |
| `agency.orchestrator.scheduler.ExecutionScheduler.set_agent_blocked` | Scheduler protocol consumes private _ExecutionRequest records; use snapshots for inspection. |
| `agency.observability.profiler.agprof.active_output_dir` | Runtime telemetry helper used to locate the active profiling session's output directory; not an application profiling entry point. |
| `agency.observability.profiler.agprof.container_cgroup_parent` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.profiler.agprof.container_started` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.profiler.agprof.container_stopped` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.profiler.agprof.container_registered` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.profiler.agprof.detail_metadata` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.profiler.agprof.register_engine` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.profiler.agprof.telemetry_error` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.profiler.agprof.ingest_auto_samples` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.profiler.agprof.sandbox_identity` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.profiler.agprof.start_external_span` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.profiler.agprof.interrupt_external_span` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.profiler.agprof.cancel_external_span` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.profiler.agprof.gpu_lease_begin` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.profiler.agprof.gpu_lease_end` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.profiler.agprof.current_execution_attributes` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.profiler.agprof.current_span_id` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.profiler.agprof.session_identity` | Runtime instrumentation callback/helper used by harnesses, samplers, or container backends; not an application profiling entry point. |
| `agency.observability.agwebui.agwebui.__init__` | Server instance lifecycle is managed by agwebui.run(); use the callable entry point or standalone CLI. |
