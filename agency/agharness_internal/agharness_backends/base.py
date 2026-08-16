"""Harness backend base class, shared config, and backend selection.

Mirrors `agllm_backends/base.py`'s shape: one abstract base
(`agharness_backend`) concrete backends subclass, a shared `Fields` class
so every backend reads its tunables as plain attributes, and a
`for_config()` selector with lazy, function-local imports of the concrete
backends (avoids a circular import, same reasoning as agllm_backends).

Selection dispatches on the `engine` string itself (e.g. `ag.engine ==
"claude_code"`, set via `agent(engine=...)` -- see docs/agent.md's engine
seam) rather than a separate `provider` config field: `ag.engine` is
already the authoritative "which engine" signal (Component 4 of
docs/Design_harness_integration.md), so there is no second place a user
would need to keep in sync with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...agconfig import DynamicConfigParam, _AgConfigViewBase

if TYPE_CHECKING:
    from ...agharness import HarnessMessages
    from ...agconfig import agConfig
    from ...agent import agent
    from ...agcontext import agcontext
    from ...agdata import agdata
    from ...agskill import agskill


class AgHarnessFields:
    """Every harness config field used by any backend, as
    DynamicConfigParam descriptors -- see agllm_backends/base.py's
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
    # agharness_backends/_native_hooks.py.

    def __init__(self, agconfig=None) -> None:
        self._agconfig = agconfig


class agHarnessConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting agharness fields in one call::

        cfg = agConfig(agHarnessConfig(gateway_mode="translate"))
        ag = agent(agconfig=cfg, engine="codex")

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics.
    """

    _OWNER = "agharness"


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

    def execute(
        self,
        ag: "agent",
        prev_ctx: "agcontext",
        skill_input: "agdata",
        max_steps: "int | None",
        *,
        skill: "agskill",
        extra_system: "str | None" = None,
        canonical_input: "HarnessMessages | None" = None,
    ) -> "tuple[agdata, agcontext, list[dict]]":
        """Same return contract as `agskill.execute_react()`/
        `execute_harness()`: `ctx` is the SAME `prev_ctx` object, mutated in
        place (`.messages`/`.total_input_tokens`/`.total_output_tokens`);
        `delta` is `[system_prompt_message] + every message appended since
        this call started`. Concrete backends implement this.

        `canonical_input` is built once by `execute_harness()` and contains
        system instructions, prior context, current input, file notices,
        attachments, and output guidance. `extra_system` remains only as a
        compatibility input for direct backend calls and the native backend."""
        raise NotImplementedError

    @staticmethod
    def for_config(engine: str, agconfig: "agConfig") -> "agharness_backend":
        from .claude_code import _ClaudeCodeBackend
        from .codex import _CodexBackend
        from .grok import _GrokBackend
        from .native import _NativeBackend
        from .opencode import _OpencodeBackend

        if engine == "native":
            return _NativeBackend(agconfig)
        if engine == "opencode":
            return _OpencodeBackend(agconfig)
        if engine == "claude_code":
            return _ClaudeCodeBackend(agconfig)
        if engine == "codex":
            return _CodexBackend(agconfig)
        if engine == "grok":
            return _GrokBackend(agconfig)
        raise ValueError(
            f"Unknown harness engine {engine!r} -- set agent(engine=...) to one of "
            f"'native', 'opencode', 'claude_code', 'codex', 'grok'"
        )
