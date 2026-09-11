"""agconfig — one config class for the whole framework"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from typing import Any, Callable, ClassVar

# ---------------------------------------------------------------------------
# Process-wide constants:
#   - DOCKER_SEMAPHORE_LIMIT: a literal module-level threading.Semaphore in
#     sandbox/container.py, throttling every docker/podman subprocess call
#     process-wide regardless of agent.
#   - GPU/SYSCTL/MEMORY detection + MARKER_MB: host hardware detection runs
#     once per process, shared by agResourcePool and the sandbox backends.
#   - MIN_CPUS/MIN_MEMORY_MB: safety floors applied to any cpu/memory limit
#     before it's ever applied to a running sandbox -- intentionally never
#     per-agent overridable.
#   - IDLE_CHECK_INTERVAL_S: read by a free function (_iter_batched) with no
#     agconfig threaded through it at all.
# ---------------------------------------------------------------------------
DOCKER_SEMAPHORE_LIMIT: int = 16
GPU_DETECT_TIMEOUT_S: float = 10
SYSCTL_DETECT_TIMEOUT_S: float = 5
MEMORY_DETECT_FALLBACK_MB: int = 4096
MARKER_MB: int = 128
MIN_CPUS: float = 1.0
MIN_MEMORY_MB: int = 1024
IDLE_CHECK_INTERVAL_S: float = 1.0


def _is_json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_safe(v) for k, v in value.items())
    return False


class confignamespace:
    """Base class for every one-level namespace config dataclass"""

    __slots__: "ClassVar[tuple]" = ()
    _SENSITIVE_FIELDS: ClassVar[frozenset] = frozenset()

    def clone(self):
        """Independent copy -- mutating the clone never affects the original."""
        return copy.deepcopy(self)

    def update(self, **values: Any):
        """Set several fields at once, e.g. from a partial webui-editor
        payload. Raises on an unknown field name."""
        known = {f.name for f in fields(self) if not f.name.startswith("_")}
        unknown = set(values) - known
        if unknown:
            raise TypeError(f"{type(self).__name__} has no field(s) {sorted(unknown)}")
        for name, value in values.items():
            setattr(self, name, value)
        return self

    def safe_snapshot(self) -> "dict[str, Any]":
        """JSON-safe, secret-redacted snapshot of every field on this namespace."""
        sensitive = self._SENSITIVE_FIELDS
        result: "dict[str, Any]" = {}
        for f in fields(self):
            if f.name.startswith("_") or f.name in sensitive:
                continue
            value = getattr(self, f.name)
            if not _is_json_safe(value):
                continue
            result[f.name] = value
        return result


@dataclass(slots=True)
class llmconfig(confignamespace):
    """LLM backend config"""

    provider: "str | None" = None
    model: str = ""
    api_key: "str | None" = None
    base_url: "str | None" = None
    region: "str | None" = None
    context_limit: "int | None" = None
    temperature: "float | None" = None
    reasoning_effort: "str | None" = None
    max_completion_tokens: "int | None" = None
    max_tokens: "int | None" = None  # deprecated alias for max_completion_tokens
    top_p: "float | None" = None
    frequency_penalty: "float | None" = None
    presence_penalty: "float | None" = None
    n: "int | None" = None
    stop: "str | list | None" = None
    logprobs: "bool | int | None" = None
    seed: "int | None" = None
    extra_body: "dict | None" = None
    top_k: "int | None" = None
    repetition_penalty: "float | None" = None
    min_p: "float | None" = None
    min_tokens: "int | None" = None
    guided_json: "dict | None" = None
    guided_regex: "str | None" = None
    workspace_id: "str | None" = None
    aws_access_key: "str | None" = None
    aws_secret_key: "str | None" = None
    aws_session_token: "str | None" = None
    aws_profile: "str | None" = None
    aws_region: "str | None" = None
    model_listing_timeout_seconds: float = 10.0
    default_max_tokens: int = 128000
    max_retries: int = 12
    idle_timeout: float = 900.0  # seconds to wait for first chunk
    stream_timeout: float = 1200.0  # seconds to wait between chunks mid-stream
    retry_sleep_s: float = 2
    http_connect_timeout: float = 10.0
    http_write_timeout: float = 10.0
    http_pool_timeout: float = 10.0
    live_redraw_char_threshold: int = 100
    rate_limit_base_backoff_s: float = 5.0
    rate_limit_max_backoff_s: float = 80.0
    rate_limit_retry_after_jitter_s: float = 5.0
    default_context_limit: int = 200_000

    # Mock/replay LLM backend (agency/llm/mock.py)
    replay_db_path: "str | None" = None
    timing_mode: str = "exact"
    constant_ttft_s: float = 0.3
    constant_tpot_s: float = 0.02
    poisson_rate_hz: float = 10.0
    poisson_ttft_mean_s: float = 0.3
    poisson_seed: "int | None" = None
    timing_fn: "Callable | None" = None  # a plain callable -- not JSON-safe,
    # so safe_snapshot() silently omits it, same as any other non-JSON-safe value.
    # Test synchronization/inspection before either replay dispatch path emits
    # a response. Use a function closure so config clones share its gate.
    replay_dispatch_hook: "Callable[[dict], None] | None" = None

    _SENSITIVE_FIELDS: ClassVar[frozenset] = frozenset(
        {"api_key", "aws_access_key", "aws_secret_key", "aws_session_token"}
    )


@dataclass(slots=True)
class sandboxconfig(confignamespace):
    """Sandbox facade (agency/sandbox/agsandbox.py) and backend mechanics
    (agency/sandbox/base.py) -- docker/podman daemon-call timeouts and retry
    policy. docker_semaphore_limit moved to the process-wide
    DOCKER_SEMAPHORE_LIMIT constant above (it gates an actual shared
    semaphore, not a per-agent tunable)."""

    base_image: str = "agency-sandbox:latest"
    persistent: bool = False
    hibernation_diagnostics: bool = False
    mounts: "dict[str, tuple[str, str, str]]" = field(default_factory=dict)

    backend: str = "auto"  # podman | docker | chroot | auto
    # Container creation constraints, retained across hibernation and rollback.
    # Changing these on an existing container requires recreating that container.
    cpuset_cpus: "str | None" = None
    cpuset_mems: "str | None" = None
    inspect_timeout_s: float = 120
    exec_quick_timeout_s: float = 120
    docker_run_timeout_s: float = 120
    docker_rm_timeout_s: float = 120
    file_io_timeout_s: float = 120
    image_timeout_s: float = 120
    commit_timeout_s: float = 120
    keyring_wait_timeout_s: float = 120
    unkillable_child_grace_s: float = 10
    container_limit_floor: int = 4
    container_limit_buffer: int = 5
    container_limit_fallback: int = 200
    conflict_retry_max_attempts: int = 8
    keyring_poll_interval_s: float = 5
    container_removal_wait_s: float = 10
    container_removal_poll_interval_s: float = 0.5
    conflict_retry_backoff_base_s: float = 0.5
    docker_stop_timeout_s: float = 120
    docker_stop_grace_s: float = 0
    docker_start_timeout_s: float = 120
    stop_inspect_timeout_s: float = 30
    commit_retry_attempts: int = 3
    commit_retry_backoff_s: float = 1
    stop_ps_check_timeout_s: float = 10
    rm_retry_attempts: int = 3
    rm_retry_backoff_s: float = 1
    checkpoint_squash_max_depth: int = 100
    squash_timeout_s: float = 600

    def add_mount(
        self, name: str, host_path, container_path: str, mode: str = "rw"
    ) -> "sandboxconfig":
        self.mounts[name] = (str(host_path), container_path, mode)
        return self

    def remove_mount(self, name: str) -> "sandboxconfig":
        self.mounts.pop(name, None)
        return self


@dataclass(slots=True)
class orchestratorconfig(confignamespace):
    """The global orchestrator (agency/orchestrator/orchestrator.py)"""

    max_concurrent_engines: "int | None" = None
    db_path: "str | None" = None
    flush_batch_size: int = 500
    flush_interval_s: float = 1.0


@dataclass(slots=True)
class resourcesconfig(confignamespace):
    """Resource pool (agency/orchestrator/agresources.py)"""

    idle_cpus: "float | None" = 8.0
    idle_memory: "str | None" = None


@dataclass(slots=True)
class agentconfig(confignamespace):
    """agency/agent.py"""

    log_dir: "str | None" = None
    output_dir: "str | None" = None
    checkpoint_save_timeout_s: int = 600
    checkpoint_load_timeout_s: int = 600
    harness: str = "native"


@dataclass(slots=True)
class schemaconfig(confignamespace):
    """agency/agschema.py."""

    input_offload_chars: int = 40_000
    offload_context_fraction: float = 0.1
    chars_per_token: int = 4


@dataclass(slots=True)
class skillconfig(confignamespace):
    """agency/agskill.py."""

    react_max_steps: int = 4096
    agbinary_validate_exec_timeout: float = 5
    error_log_truncate: int = 300
    last_output_log_truncate: int = 2000


@dataclass(slots=True)
class toolconfig(confignamespace):
    """agency/agtool.py."""

    timeout_s: float = 1800
    output_offload_chars: int = 40_000
    offload_id_prefix_len: int = 12


@dataclass(slots=True)
class harnessadapterconfig(confignamespace):
    """agency/harness/adapters/agharness_backend.py."""

    session_resume_id: "str | None" = None
    binary_path: "str | None" = None
    mediation_mode: str = "auto"  # ptrace | native_hooks | auto


@dataclass(slots=True)
class ptraceconfig(confignamespace):
    """ptrace harness supervisor (agency/harness/ptrace/supervisor.py)."""

    syscalls: "tuple[str, ...]" = ("execve", "execveat")
    profiler: "str | None" = None  # reserved for a future heavyweight profiler (e.g. perf)
    disable_harness_native_sandbox: bool = True


@dataclass(slots=True)
class dataloggerconfig(confignamespace):
    """Per-agent data logger (agency/observability/agdatalogger.py). db_path
    is computed once per agent (log_dir/<agname>_data.sqlite3) if not
    already set."""

    db_path: "str | None" = None
    flush_batch_size: int = 20
    flush_interval_s: float = 0.2


@dataclass(slots=True)
class hostserverconfig(confignamespace):
    """Host server manager (agency/engine/host_servers/host_server_manager.py).
    uds_path is computed once per agent if not already set."""

    uds_path: "str | None" = None
    startup_timeout_s: float = 10.0
    shutdown_timeout_s: float = 10.0


@dataclass(slots=True, init=False)
class agconfig:
    llm: llmconfig
    sandbox: sandboxconfig
    orchestrator: orchestratorconfig
    resources: resourcesconfig
    agent: agentconfig
    schema: schemaconfig
    skill: skillconfig
    tool: toolconfig
    harness_adapter: harnessadapterconfig
    ptrace: ptraceconfig
    data_logger: dataloggerconfig
    host_server: hostserverconfig

    # Maps each namespace dataclass to the agconfig field it lives on --
    # __init__ dispatches by the *type* of each positional argument, so
    # callers never spell out a keyword: agconfig(llmconfig(...), sandboxconfig(...)).
    _FIELD_BY_NAMESPACE_TYPE: ClassVar[dict] = {
        llmconfig: "llm",
        sandboxconfig: "sandbox",
        orchestratorconfig: "orchestrator",
        resourcesconfig: "resources",
        agentconfig: "agent",
        schemaconfig: "schema",
        skillconfig: "skill",
        toolconfig: "tool",
        harnessadapterconfig: "harness_adapter",
        ptraceconfig: "ptrace",
        dataloggerconfig: "data_logger",
        hostserverconfig: "host_server",
    }

    def __init__(self, *namespaces: Any) -> None:
        self.llm = llmconfig()
        self.sandbox = sandboxconfig()
        self.orchestrator = orchestratorconfig()
        self.resources = resourcesconfig()
        self.agent = agentconfig()
        self.schema = schemaconfig()
        self.skill = skillconfig()
        self.tool = toolconfig()
        self.harness_adapter = harnessadapterconfig()
        self.ptrace = ptraceconfig()
        self.data_logger = dataloggerconfig()
        self.host_server = hostserverconfig()

        seen: "set[str]" = set()
        for namespace in namespaces:
            field_name = self._FIELD_BY_NAMESPACE_TYPE.get(type(namespace))
            if field_name is None:
                raise TypeError(
                    f"agconfig() received an unrecognized namespace object: "
                    f"{type(namespace).__name__!r}"
                )
            if field_name in seen:
                raise TypeError(f"agconfig() received more than one {field_name!r} namespace")
            seen.add(field_name)
            setattr(self, field_name, namespace)

    def clone(self) -> "agconfig":
        """Independent copy"""
        return copy.deepcopy(self)

    def update(self, **namespace_updates: "dict[str, Any]") -> "agconfig":
        """Merge partial field updates into one or more namespaces at once."""
        known = {f.name for f in fields(self)}
        unknown = set(namespace_updates) - known
        if unknown:
            raise TypeError(f"agconfig has no namespace(s) {sorted(unknown)}")
        for namespace_name, values in namespace_updates.items():
            getattr(self, namespace_name).update(**values)
        return self

    def safe_snapshot(self) -> "dict[str, dict[str, Any]]":
        """JSON-safe, secret-redacted snapshot of every namespace, for the
        webui config editor and checkpoints/event logs."""
        return {f.name: getattr(self, f.name).safe_snapshot() for f in fields(self)}
