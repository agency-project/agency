"""Harness backend base class, shared config, and backend selection.

Mirrors `llm/base.py`'s shape: one base (`agharness_backend`)
concrete backends subclass, a shared `Fields` class so every backend reads
its tunables as plain attributes, and a `for_config()` selector with lazy,
function-local imports of the concrete backends (avoids a circular import,
same reasoning as llm).

Selection dispatches on the `harness` string itself (e.g. `ag.harness ==
"claude_code"`, set via `agent(harness=...)`) rather than a separate
`provider` config field: `ag.harness` is already the authoritative "which
harness" signal, so there is no second place a user
would need to keep in sync with it.

Every concrete backend implements the sandbox-daemon
`run_daemon_attempt(AdapterRuntime, ...)` seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    from ...configs.agconfig import agconfig as agconfig_cls


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

    agconfig: "agconfig_cls"
    model: str
    engine_name: str
    harness_base_url: str
    token: str
    syscall_policy: object
    sandbox: "AdapterSandbox | None" = None
    has_sandbox_mcp_tools: bool = False


class agharness_backend:
    """One backend instance per agconfig -- drives one off-the-shelf
    harness CLI in place of agskill's native ReAct loop. Use
    `agharness_backend.for_config(engine, agconfig)` to get the right
    subclass; don't instantiate a subclass directly."""

    # Fields with no viable fallback for a given agconfig.agent.harness --
    # checked eagerly by _validate_config() on every construction/
    # change_config() call. Empty today: every concrete adapter's
    # binary_path falls back to its own _DEFAULT_BINARY when unset, so
    # nothing about agconfig.harness_adapter is strictly required from
    # agconfig alone. Kept as a real, populated mechanism -- not a stub --
    # for the day a harness-specific field with no safe default is added.
    _REQUIRED_FIELDS_BY_HARNESS: "ClassVar[dict[str, tuple[str, ...]]]" = {}

    def __init__(self, agconfig: "agconfig_cls") -> None:
        self.change_config(agconfig)

    def change_config(self, agconfig: "agconfig_cls") -> None:
        self.agconfig = agconfig.clone()
        self._validate_config()

    def _validate_config(self) -> None:
        harness = self.agconfig.agent.harness
        required = self._REQUIRED_FIELDS_BY_HARNESS.get(harness, ())
        missing = [name for name in required if not getattr(self.agconfig.harness_adapter, name)]
        if missing:
            raise ValueError(
                f"agconfig.harness_adapter with harness={harness!r} is missing "
                f"required field(s): {', '.join(missing)}"
            )

    def get_config_copy(self) -> "agconfig_cls":
        return self.agconfig.clone()

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

    def register(self, app, router) -> None:
        raise NotImplementedError

    def _format_context_harness_to_agency(self, raw_request: dict) -> dict:
        raise NotImplementedError

    def _format_context_agency_to_harness(self, agency_response: dict) -> dict:
        raise NotImplementedError

    def _format_agency_stream_to_harness(self, agency_stream):
        raise NotImplementedError

    @staticmethod
    def for_config(harness: str, agconfig: "agconfig_cls") -> "agharness_backend":
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
