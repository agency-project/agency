"""Harness backend base class, shared config, and backend selection.

Mirrors `llm/base.py`'s shape: one base (`agharness_backend`)
concrete backends subclass, a shared `Fields` class so every backend reads
its tunables as plain attributes, and a `for_config()` selector with lazy,
function-local imports of the concrete backends (avoids a circular import,
same reasoning as llm).

Selection dispatches on the `harness` string itself (e.g. `ag.harness ==
"claude_code"`, set via `agent(harness=...)` -- see docs/agent.md's harness
seam) rather than a separate `provider` config field: `ag.harness` is
already the authoritative "which harness" signal (Component 4 of
docs/Design_harness_integration.md), so there is no second place a user
would need to keep in sync with it.

Every concrete backend implements the sandbox-daemon
`run_daemon_attempt(AdapterRuntime, ...)` seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ...agconfig import DynamicConfigParam, _AgConfigViewBase

if TYPE_CHECKING:
    from ...agconfig import agConfig


class AgHarnessFields:
    """Every harness config field used by any backend, as
    DynamicConfigParam descriptors -- see llm/base.py's
    AgLLMBackendFields for the same pattern."""

    gateway_mode = DynamicConfigParam(
        "agharness", default="passthrough"
    )  # "passthrough" (the configured agllm backend already speaks the
    # harness's wire format, forward requests unmodified through
    # agproxy_llm) | "translate" (reshape requests/responses -- not
    # implemented yet; only the passthrough route exists so far).
    session_resume_id = DynamicConfigParam(
        "agharness", default=None
    )  # the harness's own session id from a prior run on this agent, for
    # multi-turn resume -- set by a concrete backend after its first execute().
    binary_path = DynamicConfigParam(
        "agharness", default=None
    )  # override the harness CLI's resolved path; None = look up the
    # backend-specific default binary name via PATH.
    mediation_mode = DynamicConfigParam(
        "agharness", default="auto"
    )  # "ptrace" | "native_hooks" | "auto" (ptrace if ptrace_available(),
    # else native_hooks with a logged reduced-coverage warning) -- see
    # `harness/_native_hooks.py`.

    def __init__(self, agconfig=None) -> None:
        self._agconfig = agconfig


class agHarnessConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting agharness fields in one call::

        cfg = agConfig(agHarnessConfig(gateway_mode="translate"))
        ag = agent(agconfig=cfg, harness="codex")

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics.
    """

    _OWNER = "agharness"


@dataclass
class AttemptResult:
    """Normalized result of one harness CLI invocation."""

    ok: bool
    final_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    session_id: "str | None" = None
    session_blob: "bytes | None" = None
    error_message: str = ""


class AdapterSandbox(Protocol):
    """Filesystem/command surface available to an adapter inside the daemon."""

    def exec(self, cmd: str, workdir: str = "/workspace", timeout: int = 600): ...

    def read_file(self, path: str) -> str: ...

    def read_file_bytes(self, path: str) -> bytes: ...

    def write_file_bytes(self, path: str, data: bytes) -> None: ...


@dataclass(frozen=True)
class AdapterRuntime:
    """Explicit sandbox-daemon dependencies for one harness attempt.

    This deliberately is not an ``agent``. Host-owned agent, skill, manager,
    logging, and UI state must not leak across the host/sandbox boundary.
    """

    agconfig: "agConfig"
    model: str
    engine_name: str
    harness_base_url: str
    token: str
    syscall_policy: object
    sandbox: "AdapterSandbox | None" = None
    suppress_builtin_tools: bool = False


class agharness_backend(AgHarnessFields):
    """One backend instance per agconfig -- drives one off-the-shelf
    harness CLI in place of agskill's native ReAct loop. Use
    `agharness_backend.for_config(engine, agconfig)` to get the right
    subclass; don't instantiate a subclass directly."""

    def __init__(self, agconfig: "agConfig") -> None:
        self._agconfig = agconfig.clone()

    def change_config(self, agconfig: "agConfig") -> None:
        self._agconfig = agconfig.clone()

    def get_config_copy(self) -> "agConfig":
        return self._agconfig.clone()

    def run_daemon_attempt(
        self,
        runtime: AdapterRuntime,
        *,
        prompt: str,
        resume_session_id: "str | None",
        prior_session_blob: "bytes | None",
        max_steps: "int | None",
    ) -> AttemptResult:
        """Run one CLI attempt through the narrow sandbox-daemon seam."""
        raise NotImplementedError

    @staticmethod
    def for_config(harness: str, agconfig: "agConfig") -> "agharness_backend":
        from .claude_code import _ClaudeCodeBackend
        from .codex import _CodexBackend
        from .grok import _GrokBackend
        from .native import _NativeBackend
        from .opencode import _OpencodeBackend

        if harness == "native":
            return _NativeBackend(agconfig)
        if harness == "opencode":
            return _OpencodeBackend(agconfig)
        if harness == "claude_code":
            return _ClaudeCodeBackend(agconfig)
        if harness == "codex":
            return _CodexBackend(agconfig)
        if harness == "grok":
            return _GrokBackend(agconfig)
        raise ValueError(
            f"Unknown harness {harness!r} -- set agent(harness=...) to one of "
            f"'native', 'opencode', 'claude_code', 'codex', 'grok'"
        )
