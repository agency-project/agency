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
    from ..agent import agent
    from ..agdata import agdata
    from ..agskill import agskill


def _runtime_name(owner: "agent | str") -> str:
    return owner if isinstance(owner, str) else owner.agname


def materialize_config_home(ag: "agent | str", token: str, base_url: str) -> Path:
    """Create a fresh, isolated directory for one harness launch's config
    home. Concrete backends write their own harness-specific config files
    (env vars, provider blocks, etc. pointing at *base_url* with *token*)
    into this directory -- what to write is backend-specific, only the
    "give me an isolated directory" part is shared."""
    return Path(tempfile.mkdtemp(prefix=f"agharness-{_runtime_name(ag)}-"))


def cleanup_config_home(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def is_container_backed(sandbox) -> bool:
    """True for a docker/podman-backed sandbox (`IMAGE_KIND == "container"`),
    False for chroot or no sandbox at all. See
    docs/Design_harness_integration.md's "Prerequisites": a container-backed
    harness launch needs the in-container ptrace bridge and in-container
    config-home materialization below; chroot's harness launch already runs
    on the bare host (the jail IS a real host directory) and needs neither."""
    return sandbox is not None and getattr(sandbox._backend, "IMAGE_KIND", "") == "container"


def materialize_config_home_in_container(ag: "agent | str", sandbox, token: str) -> str:
    """In-container counterpart to `materialize_config_home` -- creates a
    fresh, isolated directory INSIDE *sandbox*'s own container filesystem
    instead of a host tempdir. Required once the harness process itself
    runs inside the container: a host tempdir is invisible to a process in
    the container's own mount namespace, so `cwd`/`CLAUDE_CONFIG_DIR` (or
    each other harness's equivalent) must point somewhere the harness can
    actually see. Returns the in-container path; paired with
    `cleanup_config_home_in_container` in the caller's `finally`, mirroring
    `materialize_config_home`/`cleanup_config_home`'s own pairing."""
    import shlex

    path = f"/tmp/agharness-{_runtime_name(ag)}-{token}"
    sandbox.exec(f"mkdir -p {shlex.quote(path)}", workdir="/")
    return path


def cleanup_config_home_in_container(sandbox, path: str) -> None:
    import shlex

    sandbox.exec(f"rm -rf {shlex.quote(path)}", workdir="/")


# ---------------------------------------------------------------------------
# Manager/bridge lifecycle -- owned by the HOST PROCESS (agskill.py's
# execute_harness(), the one place backend.execute() is ever called from),
# not by any concrete backend. These are the only two functions that ever
# construct/launch anything; a backend only ever receives the already-built
# `host_manager`/`harness_base_url` as `execute()` parameters and consumes
# them (see agharness_backends/base.py's execute() docstring) -- it never
# calls these itself. This is deliberately narrower than the equivalent
# lazy-singleton getters the old design had (get_shared_terminus() etc.,
# one per process): a manager here is one PER AGENT, cached on the agent
# object itself, same convention as the existing `ag._harness_sessions`.
# ---------------------------------------------------------------------------


def get_or_create_host_manager(ag: "agent", agconfig=None):
    """This agent's `agHostAgentManager` -- constructed and started at
    most once per agent, reused across every subsequent skill call. Cached
    on `ag` itself (`ag._host_agent_manager`), the same place
    `ag._harness_sessions` already lives, rather than in a module-level
    registry: there is no need for a second lookup structure keyed by
    agent when the agent object is already right there."""
    existing = getattr(ag, "_host_agent_manager", None)
    if existing is not None:
        return existing
    from ..manager.agmanager_host import agHostAgentManager

    manager = agHostAgentManager(ag, agconfig)
    manager.start()
    ag._host_agent_manager = manager
    return manager


def ensure_harness_bridge(sandbox, host_manager) -> str:
    """The base URL a harness should point its LLM/MCP/policy/profiler
    traffic at -- ALWAYS an `agmanager_harness` instance, container-backed
    or not:
    - Container-backed: launches `agmanager_harness.py` as a real process
      INSIDE the container if it isn't already running (idempotent per
      sandbox, cached on `sandbox._agmanager_harness_url`).
    - Bare-host/chroot (`sandbox` not container-backed, including
      `sandbox is None`): runs the exact same `agmanager_harness` app
      in-process on this host instead (idempotent per agent, cached on
      `host_manager._local_harness_url`) -- see `agmanager_harness.
      launcher.ensure_launched_locally`'s docstring for why that's not a
      second implementation, just a different transport to the same
      bridge."""
    from ..manager.agmanager_harness import ensure_launched, ensure_launched_locally

    if is_container_backed(sandbox):
        return ensure_launched(sandbox, host_manager)
    return ensure_launched_locally(host_manager)


def mcp_config_for(harness_base_url: str, token: str) -> dict:
    """The `--mcp-config`-shaped dict every container-backed harness needs
    to reach `agmanager_harness`'s `/mcp` mount -- one shared shape, since
    every harness (Claude Code today, native_harness and eventually Codex/
    opencode/Grok tomorrow) wants the identical
    resource-control/submit_output/ask_human tool surface at the same
    bridged URL."""
    return {
        "mcpServers": {
            "agency": {
                "type": "http",
                "url": f"{harness_base_url}/mcp",
                "headers": {"Authorization": f"Bearer {token}"},
            }
        }
    }


# [REFACTOR] Why do we have this wrapper?
def build_user_turn_prompt(skill: "agskill", skill_input: "agdata") -> "str | list":
    """The skill's task, delivered as a plain user-turn prompt -- reuses
    agskill's own prompt-construction code so a harness sees exactly the
    same JSON-input convention the native ReAct loop's first user message
    uses."""
    return skill._build_user_content(skill_input)


def build_output_format_instruction(skill: "agskill") -> "str | None":
    """A plain-text instruction describing the required JSON response shape,
    appended to the prompt for skills with a structured output_schema,
    parsed post-hoc by `agschema.validate_and_recover`. Used by every
    harness backend that hasn't been wired to the shared MCP server's
    `submit_output` tool yet (codex/opencode/grok -- see
    build_mcp_output_format_instruction's docstring for the ones that
    have). Returns None for a raw-text/no-schema skill, which needs no such
    instruction."""
    if skill.output_schema is None or skill.output_schema.raw_key() is not None:
        return None
    return (
        "\n\nWhen you are done, respond with a final message containing ONLY a single "
        "JSON object (no surrounding prose, no markdown code fence) matching this shape:\n"
        f"{skill.output_schema.to_json()}"
    )


def build_mcp_output_format_instruction(skill: "agskill") -> "str | None":
    """A plain-text instruction directing the harness to call the shared
    MCP server's `submit_output` tool (see harness/agmcp_server.py,
    Phase 4) once per required output field -- the harness-driven
    counterpart to the native loop's `return_<field>` tools, reusing the
    same MCP server every engine already gets `reserve_cpu`/`cpu_release`/
    `daemon_release` from rather than a second, harness-only mechanism.
    Only for backends that actually register this skill's tokens against
    that server and wire `--mcp-config` (today: claude_code.py only --
    codex/opencode/grok still use `build_output_format_instruction` above
    until they get the same wiring, task #11). Returns None for a
    raw-text/no-schema skill, which needs no such instruction."""
    if skill.output_schema is None or skill.output_schema.raw_key() is not None:
        return None
    field_lines = "\n".join(
        f"  - {f}: {skill.output_schema.field_desc(f)}" for f in skill.output_schema._data
    )
    return (
        "\n\nThis task requires structured output. Call the `submit_output` tool once for "
        f"each of the following required fields (do not respond with a JSON object in your "
        f"final message instead):\n{field_lines}\n\n"
        "- Call submit_output separately for each field -- one field per call.\n"
        "- `value` must be the field's value encoded as a JSON literal (a quoted string for "
        "a string field, a bare number for int/float, true/false for bool).\n"
        "- Only call submit_output once you have the final value ready for that field."
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
        try:
            self._ag.log._tool_call(
                event.syscall,
                {"argv": event.argv, "path": event.path},
                {},
                0,
            )
        except Exception as _e:
            # logging is best-effort; never let it block the traced process
            print(f"[agharness] WARNING: failed to log syscall {event.syscall!r}: {_e}")
        return True

    def check_tool(self, ag, tool_name: str, tool_input: dict):
        try:
            self._ag.log._tool_call(tool_name, tool_input, {}, 0)
        except Exception as _e:
            # logging is best-effort; never let it block the harness
            print(f"[agharness] WARNING: failed to log tool call {tool_name!r}: {_e}")
        return True


def default_policy(ag: "agent"):
    syscall_policy = getattr(ag, "__dict__", {}).get("syscall_policy")
    if syscall_policy is not None:
        return syscall_policy
    return _LoggingAllowAllPolicy(ag)


__all__ = [
    "materialize_config_home",
    "cleanup_config_home",
    "is_container_backed",
    "materialize_config_home_in_container",
    "cleanup_config_home_in_container",
    "get_or_create_host_manager",
    "ensure_harness_bridge",
    "mcp_config_for",
    "build_user_turn_prompt",
    "build_output_format_instruction",
    "build_mcp_output_format_instruction",
    "default_policy",
]
