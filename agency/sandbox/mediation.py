"""Harness-native hook fallback.

`ptrace` is the default
mediation path everywhere it's usable (Linux with a working ptrace), and
this module exists only for the case it explicitly isn't (non-Linux, or a
sandboxed environment where `ptrace_available()` genuinely returns False).

Scope deliberately narrow: this provides the hook-JSON <-> `agsyscallevent`/
`PtraceDecision` translation (the part worth writing once, correctly, since
Claude Code's and Codex's PreToolUse/PostToolUse hook payload shapes are
near-identical), and `resolve_mediation_mode()` to decide which path a
launch should take. It is NOT wired into `claude_code.py`/`opencode.py`/
`codex.py`'s `execute()` methods as an actual alternate code path -- doing
that for all three, plus standing up each harness's own hook-registration
config, is a real second implementation of Component 3 mediation, which is
out of scope for what this phase calls a "reduced-coverage fallback."

This module has no adapter-specific state. It lives in `agency.sandbox`
beside the syscall event and ptrace implementations so the active harness
boundary can depend on sandbox mediation without importing archived code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .events import agsyscallevent

if TYPE_CHECKING:
    from .ptrace import PtraceDecision


def resolve_mediation_mode(mediation_mode: str) -> str:
    """Resolve "auto" to a concrete mode; "ptrace"/"native_hooks" pass
    through unchanged (an explicit request is never silently overridden)."""
    if mediation_mode != "auto":
        return mediation_mode
    from .ptrace import ptrace_available

    return "ptrace" if ptrace_available() else "native_hooks"


def hook_payload_to_syscallevent(payload: dict):
    """Translate a `PreToolUse`/`PostToolUse`-shaped hook payload (Claude
    Code and Codex use near-identical JSON here: `tool_name`, `tool_input`)
    into the same `agsyscallevent` shape `ptrace` delivers to
    the policy service, so one policy implementation can back both
    mediation paths without knowing which one is active."""
    tool_input = payload.get("tool_input") or {}
    argv = None
    if "command" in tool_input:
        # Bash-family tools: the harness's own shell-parsing already
        # happened, so this is a best-effort split, not a real argv array
        # the way ptrace's execve resolution gives you.
        argv = str(tool_input["command"]).split()
    return agsyscallevent(
        syscall=payload.get("tool_name", "unknown"),
        pid=payload.get("pid", -1),
        tid=payload.get("pid", -1),
        argv=argv,
        envp=None,
        path=tool_input.get("path") or tool_input.get("file_path"),
        timestamp=0.0,
        tool_name=payload.get("tool_name") or "unknown",
        tool_args=tool_input,
    )


def decision_to_hook_response(decision: "PtraceDecision") -> dict:
    """Translate a `PtraceDecision` back into the `PreToolUse` JSON response
    shape Claude Code/Codex hooks expect (`hookSpecificOutput` with
    `permissionDecision` + optional `updatedInput`)."""
    if decision.kind == "deny":
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": decision.reason or "denied by agpolicy",
            }
        }
    if decision.kind == "rewrite" and decision.new_args:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": {"command": " ".join(decision.new_args)},
            }
        }
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}


__all__ = [
    "resolve_mediation_mode",
    "hook_payload_to_syscallevent",
    "decision_to_hook_response",
]
