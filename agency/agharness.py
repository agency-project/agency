"""Thin, engine-agnostic glue shared by every `agharness_backends/*`
concrete backend.

Concrete backends retain only config-file, argv, and output parsing details.
This module owns the engine-neutral input envelope, isolated config homes,
and the common supervised CLI lifecycle.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import agent
    from .agdata import agdata
    from .agskill import agskill
    from .agcontext import agcontext


@dataclass(frozen=True)
class HarnessMessages:
    """Engine-neutral input for one external harness invocation."""

    system_instructions: str
    previous_context: tuple[dict, ...]
    current_user_input: str
    file_notices: tuple[str, ...] = ()
    attachments: tuple[dict, ...] = ()
    output_guidance: "str | None" = None


def build_harness_messages(
    skill: "agskill",
    previous_context: "agcontext",
    skill_input: "agdata",
    *,
    file_notice: "str | None" = None,
) -> HarnessMessages:
    """Build the complete Agency task before any engine is selected.

    Multimodal blocks stay typed in ``attachments`` instead of being flattened
    into adapter-specific prompt syntax.  The adapters only choose how this
    canonical value is transported to their CLI.
    """
    content = skill._build_user_content(skill_input)
    attachments: list[dict] = []

    # Seperate text instructions from multi modal attachment
    if isinstance(content, str):
        user_input = content
    else:
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
            elif isinstance(block, dict):
                attachments.append(block)
        user_input = "\n".join(part for part in text_parts if part)

    return HarnessMessages(
        system_instructions=skill._build_system_prompt(include_output_guidance=False),
        previous_context=tuple(previous_context.messages),
        current_user_input=user_input,
        file_notices=(file_notice.strip(),) if file_notice and file_notice.strip() else (),
        attachments=tuple(attachments),
        output_guidance=build_output_format_instruction(skill),
    )


def render_harness_messages(
    messages: HarnessMessages, *, output_guidance: "str | None" = None
) -> str:
    """Render the canonical task as the CLI-independent full-task envelope."""
    sections = [("SYSTEM INSTRUCTIONS", messages.system_instructions)]
    if messages.previous_context:
        sections.append(
            (
                "PREVIOUS CONTEXT",
                json.dumps(list(messages.previous_context), ensure_ascii=False, default=str),
            )
        )
    sections.append(("CURRENT USER INPUT", messages.current_user_input))
    if messages.file_notices:
        sections.append(("FILE NOTICES", "\n".join(messages.file_notices)))
    if messages.attachments:
        sections.append(
            (
                "ATTACHMENTS",
                json.dumps(list(messages.attachments), ensure_ascii=False, default=str),
            )
        )
    guidance = messages.output_guidance if output_guidance is None else output_guidance
    if guidance:
        sections.append(("OUTPUT GUIDANCE", guidance.strip()))
    return "\n\n".join(f"[{title}]\n{body}" for title, body in sections)


def harness_user_message(messages: HarnessMessages) -> dict:
    """Return the canonical current turn in Agency's context format."""
    content: "str | list[dict]" = messages.current_user_input
    if messages.attachments:
        content = [
            {"type": "text", "text": messages.current_user_input},
            *messages.attachments,
        ]
    return {"role": "user", "content": content}


def run_harness_cli(
    ag: "agent",
    argv: list[str],
    envp: dict[str, str],
    *,
    stdin: "str | bytes | None" = None,
    timeout_s: float = 600,
    cwd: str = "/workspace",
    sandbox=None,
) -> "tuple[str, str, int]":
    """Launch, feed, supervise, and reap one external harness process."""
    from .agharness_internal.agproxy_ptrace import agProxyPtrace, wire_to_sandbox

    target_sandbox = ag.sandbox if sandbox is None else sandbox
    handle = agProxyPtrace(ag.agconfig).launch(
        argv,
        envp,
        cwd=cwd,
        policy=default_policy(ag),
        ag=ag,
        sandbox=target_sandbox,
        stdin=stdin,
    )
    if target_sandbox is not None:
        wire_to_sandbox(handle, target_sandbox)

    stdout, stderr, rc = handle.wait(timeout=timeout_s)
    if rc != -1:
        return stdout, stderr, rc

    # ``wait`` uses -1 exclusively for deadline expiry. Own termination and
    # reap here so no adapter can accidentally leave a process tree behind.
    handle.kill()
    final_stdout, final_stderr, _killed_rc = handle.wait()
    return final_stdout or stdout, final_stderr or stderr, -1


def materialize_config_home(ag: "agent", token: str, base_url: str) -> Path:
    """Create a fresh, isolated directory for one harness launch's config
    home. Concrete backends write their own harness-specific config files
    (env vars, provider blocks, etc. pointing at *base_url* with *token*)
    into this directory -- what to write is backend-specific, only the
    "give me an isolated directory" part is shared."""
    return Path(tempfile.mkdtemp(prefix=f"agharness-{ag.agname}-"))


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


def resolve_harness_binary_in_container(sandbox, binary: str) -> "str | None":
    """Resolve a target-compatible CLI without copying a host binary.

    The image PATH wins, followed by the read-only harness binary cache mount.
    Avoiding an automatic host copy is essential when host and sandbox ABIs
    differ (for example an ARM macOS host and an x86_64 Linux container).
    """
    import shlex

    out, rc = sandbox.exec(f"which {shlex.quote(binary)}", workdir="/")
    if rc == 0 and out.strip():
        return out.strip()
    cached_path = f"/opt/agency_harness_bin/{binary}"
    out, rc = sandbox.exec(f"test -x {shlex.quote(cached_path)}", workdir="/")
    return cached_path if rc == 0 else None


def materialize_config_home_in_container(ag: "agent", sandbox, token: str) -> str:
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

    # The gateway bearer token must never appear in a filesystem path (Grok
    # references its private task file by path in argv).
    path = f"/tmp/agharness-{ag.agname}-{uuid.uuid4().hex}"
    sandbox.exec(f"mkdir -m 700 -p {shlex.quote(path)}", workdir="/")
    return path


def cleanup_config_home_in_container(sandbox, path: str) -> None:
    import shlex

    sandbox.exec(f"rm -rf {shlex.quote(path)}", workdir="/")


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
    MCP server's `submit_output` tool (see agharness_internal/agmcp_server.py,
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
        from .agpolicy import agdecision

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
        return agdecision.allow()

    def check_tool(self, ag, tool_name: str, tool_input: dict):
        from .agpolicy import agdecision

        try:
            self._ag.log._tool_call(tool_name, tool_input, {}, 0)
        except Exception as _e:
            # logging is best-effort; never let it block the harness
            print(f"[agharness] WARNING: failed to log tool call {tool_name!r}: {_e}")
        return agdecision.allow()


def default_policy(ag: "agent"):
    return _LoggingAllowAllPolicy(ag)


__all__ = [
    "materialize_config_home",
    "cleanup_config_home",
    "is_container_backed",
    "resolve_harness_binary_in_container",
    "materialize_config_home_in_container",
    "cleanup_config_home_in_container",
    "HarnessMessages",
    "build_harness_messages",
    "render_harness_messages",
    "harness_user_message",
    "run_harness_cli",
    "build_output_format_instruction",
    "build_mcp_output_format_instruction",
    "default_policy",
]
