"""Types shared across the agent-engine and harness-manager boundaries.

The harness wire types contain JSON-compatible data only. They may cross
the host/sandbox process boundary and must not contain Agency objects,
callbacks, credentials, or tool implementations. ``CompletedResult`` is an
in-process result and is deliberately allowed to contain Agency objects.

ExecutionBuilder compiles an ``agskill`` and its prepared input/output
contract into ``PromptPayload``. The manager receives only that compiled
prompt, a JSON-safe projection of the existing unified ``agConfig``, a
short-lived capability token. Tools and policies stay host-side and are
resolved from the capability token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, NewType, TypeAlias

if TYPE_CHECKING:
    from ..agcontext import agcontext
    from ..agdata import agdata


# ---------------------------------------------------------------------------
# Wire primitives
# ---------------------------------------------------------------------------
# Everything below is safe to serialize across the host/sandbox boundary.
# NewType gives identifiers distinct meanings to a type checker while keeping
# their runtime and JSON representation as ordinary strings.

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
ConfigPayload: TypeAlias = JsonObject

ExecutionId = NewType("ExecutionId", str)
CapabilityToken = NewType("CapabilityToken", str)
ManagerId = NewType("ManagerId", str)


# ---------------------------------------------------------------------------
# Per-run request: ExecutionBuilder -> Harness Manager
# ---------------------------------------------------------------------------
# ExecutionBuilder turns Agency domain objects into these JSON-only values.
# The capability token is the only reference back to host-owned tools,
# policies, resources, output collection, and telemetry.

# A string is the normal CLI input surface. The list form preserves
# multimodal content until an adapter decides how its frontend supports it.
PromptPayload: TypeAlias = str | list[JsonObject]


@dataclass(frozen=True, slots=True)
class SkillExecutionRequest:
    """Complete per-execution request sent to the harness manager."""

    execution_id: ExecutionId
    capability_token: CapabilityToken
    config: ConfigPayload
    prompt: PromptPayload


# ---------------------------------------------------------------------------
# Manager lifecycle: ExecutionBuilder <-> Harness Manager daemon
# ---------------------------------------------------------------------------
# These types describe the long-lived manager process, not an individual
# skill executions. One manager may handle multiple SkillExecutionRequests.


@dataclass(frozen=True, slots=True)
class HarnessManagerLaunchConfig:
    """Process-level configuration used to start a manager daemon.

    The host UDS is established once for the manager's lifetime and therefore
    does not need to be repeated in every run request.
    """

    manager_id: ManagerId
    host_uds_path: str
    workspace_path: str = "/workspace"
    listen_host: str = "127.0.0.1"
    listen_port: int = 0
    startup_timeout_s: float = 10.0


@dataclass(frozen=True, slots=True)
class HarnessManagerHandle:
    """Addressable handle returned after launching a manager daemon."""

    manager_id: ManagerId
    pid: int
    base_url: str


# ---------------------------------------------------------------------------
# Per-run response: Harness Manager -> ExecutionBuilder
# ---------------------------------------------------------------------------
# Every harness adapter normalizes its native exit/session/usage information
# into this common result. Rich Agency output recovery happens afterward on
# the host and is intentionally not represented here.


@dataclass(frozen=True, slots=True)
class SkillExecutionResponse:
    """Normalized result returned by every harness adapter."""

    execution_id: ExecutionId
    status: SkillExecutionStatus
    final_text: str = ""
    exit_code: int | None = None
    error: HarnessError | None = None

    @property
    def ok(self) -> bool:
        return self.status is SkillExecutionStatus.SUCCEEDED


class SkillExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class HarnessErrorKind(StrEnum):
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_HARNESS = "unsupported_harness"
    STARTUP_FAILED = "startup_failed"
    HARNESS_EXITED = "harness_exited"
    CONTEXT_CONFLICT = "context_conflict"
    PROTOCOL_ERROR = "protocol_error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class HarnessError:
    kind: HarnessErrorKind
    message: str
    retryable: bool = False
    details: JsonObject = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Temporary bridge compatibility
# ---------------------------------------------------------------------------
# Remove this alias once HarnessManagerBridge and ExecutionBuilder speak in
# SkillExecutionRequest/SkillExecutionResponse directly.

# Compatibility name for the existing bridge skeleton. The bridge should use
# SkillExecutionResponse directly when its implementation lands.
HarnessAttemptResult = SkillExecutionResponse


# ---------------------------------------------------------------------------
# Host-only Agency result
# ---------------------------------------------------------------------------
# Unlike the wire types above, this object deliberately contains agdata and
# agcontext instances. It is returned by the engine to Agency and must never
# be serialized or sent to the sandbox-side manager.


@dataclass
class CompletedResult:
    """In-process Agency result; never serialized to the harness manager."""

    output: "agdata"
    context: "agcontext"
    delta: "list[dict]"
    ok: bool = True
    error_message: str = ""


__all__ = [
    "CapabilityToken",
    "CompletedResult",
    "ConfigPayload",
    "ExecutionId",
    "HarnessAttemptResult",
    "HarnessError",
    "HarnessErrorKind",
    "HarnessManagerHandle",
    "HarnessManagerLaunchConfig",
    "JsonObject",
    "JsonScalar",
    "JsonValue",
    "ManagerId",
    "PromptPayload",
    "SkillExecutionRequest",
    "SkillExecutionResponse",
    "SkillExecutionStatus",
]
