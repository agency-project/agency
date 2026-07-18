"""Thin, engine-agnostic glue shared by every `agharness_backends/*`
concrete backend.

Deliberately small -- per-harness config-file format and CLI argv
construction stay in each concrete backend, not here. This module only
holds what's genuinely shared: an isolated per-launch config-home
directory (so concurrent harness-driven agents never see each other's
token/base_url, and a run leaves no trace in the user's own `~/.claude`/
`~/.codex`/`~/.config/opencode`), and prompt construction that reuses
agskill's own existing code rather than re-implementing it -- the skill's
task is delivered to the harness as a plain user-turn prompt, never
injected as the harness's own system prompt or as a tool (see
docs/Design_harness_integration.md).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import agent
    from .agdata import agdata
    from .agskill import agskill


def materialize_config_home(ag: "agent", token: str, base_url: str) -> Path:
    """Create a fresh, isolated directory for one harness launch's config
    home. Concrete backends write their own harness-specific config files
    (env vars, provider blocks, etc. pointing at *base_url* with *token*)
    into this directory -- what to write is backend-specific, only the
    "give me an isolated directory" part is shared."""
    return Path(tempfile.mkdtemp(prefix=f"agharness-{ag.agname}-"))


def cleanup_config_home(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def build_user_turn_prompt(skill: "agskill", skill_input: "agdata") -> "str | list":
    """The skill's task, delivered as a plain user-turn prompt -- reuses
    agskill's own prompt-construction code so a harness sees exactly the
    same JSON-input convention the native ReAct loop's first user message
    uses."""
    return skill._build_user_content(skill_input)


def build_output_format_instruction(skill: "agskill") -> "str | None":
    """A plain-text instruction describing the required JSON response shape,
    appended to the prompt for skills with a structured output_schema --
    the harness-driven counterpart to the native loop's `return_<field>`
    tools, without adding a tool to the harness's tool list. Returns None
    for a raw-text/no-schema skill, which needs no such instruction."""
    if skill.output_schema is None or skill.output_schema.raw_key() is not None:
        return None
    return (
        "\n\nWhen you are done, respond with a final message containing ONLY a single "
        "JSON object (no surrounding prose, no markdown code fence) matching this shape:\n"
        f"{skill.output_schema.to_json()}"
    )


class _LoggingAllowAllPolicy:
    """Default mediation for harness-driven agents until the real policy
    retrofit (docs/Design_harness_integration.md's later build phase) picks
    a real allow/deny/rewrite implementation: allow everything, but log
    every intercepted syscall through this agent's own `aglog`, the same
    structured log native tool calls go through -- so a harness-driven
    agent's execution is observable in the webui/log files exactly like a
    native one's, even with no real security policy wired up yet."""

    def __init__(self, ag: "agent") -> None:
        self._ag = ag

    def check(self, ag, event):
        from .agpolicy import agdecision

        try:
            self._ag.log._tool_call(
                event.syscall,
                {"argv": event.argv, "path": event.path},
                {},
                0,
            )
        except Exception:
            pass  # logging is best-effort; never let it block the traced process
        return agdecision.allow()


def default_policy(ag: "agent"):
    return _LoggingAllowAllPolicy(ag)


__all__ = [
    "materialize_config_home",
    "cleanup_config_home",
    "build_user_turn_prompt",
    "build_output_format_instruction",
    "default_policy",
]
