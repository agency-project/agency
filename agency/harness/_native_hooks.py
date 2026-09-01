"""Harness-native hook fallback with reduced coverage: `agproxy_ptrace` is the default
mediation path everywhere it's usable (Linux with a working ptrace), and
this module exists only for the case it explicitly isn't (non-Linux, or a
sandboxed environment where `ptrace_available()` genuinely returns False).

Scope deliberately narrow: this provides the hook-JSON <-> `agsyscallevent`/
allow-or-deny translation (the part worth writing once, correctly, since
Claude Code's and Codex's PreToolUse/PostToolUse hook payload shapes are
near-identical), and `resolve_mediation_mode()` to decide which path a
launch should take. It is NOT wired into `claude_code.py`/`opencode.py`/
`codex.py`'s `execute()` methods as an actual alternate code path -- doing
that for all three, plus standing up each harness's own hook-registration
config, is a real second implementation of Component 3 mediation, which is
out of scope for what this phase calls a "reduced-coverage fallback."

**Lives at the `harness` top level, not inside
`agharness_backends/`** (moved from there -- see the conversation that
caught this): `agprof_ingest.py`/`agmanager_host/profiler_ingest.py` (both
top-level-ish "service" modules) need to import this for hook-payload
parsing, while every concrete backend in `agharness_backends/` imports
`agprof_ingest.py`. Nesting this module inside `agharness_backends/` made
that a real circular dependency between the two layers (a service module
reaching down into the backends package, while the backends package reaches
back up into the service module) -- not a hard `ImportError` (every import
site is function-local/lazy), but a real layering violation. This module
has no backend-specific state or logic of its own (pure hook-JSON
translation), so it belongs as a peer of `_syscall_event.py`, not nested
under the backends it happens to currently only be used to support.
"""

from __future__ import annotations

from ._syscall_event import agsyscallevent


def resolve_mediation_mode(mediation_mode: str) -> str:
    """Resolve "auto" to a concrete mode; "ptrace"/"native_hooks" pass
    through unchanged (an explicit request is never silently overridden)."""
    if mediation_mode != "auto":
        return mediation_mode
    from .ptrace.supervisor import ptrace_available

    return "ptrace" if ptrace_available() else "native_hooks"


def hook_payload_to_syscallevent(payload: dict):
    """Translate a `PreToolUse`/`PostToolUse`-shaped hook payload (Claude
    Code and Codex use near-identical JSON here: `tool_name`, `tool_input`)
    into the same `agsyscallevent` shape `agproxy_ptrace` delivers to
    `agpolicy.check()`, so one policy implementation can back both
    mediation paths without knowing which one is active."""
    tool_input = payload.get("tool_input") or {}
    argv = None
    if "command" in tool_input:
        # Bash-family tools: the harness's own shell-parsing already
        # happened, so this is a best-effort split, not a real argv array
        # the way agproxy_ptrace's execve resolution gives you.
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


def decision_to_hook_response(decision: "bool | tuple[bool, str]") -> dict:
    """Translate the shared allow-or-deny result into hook JSON."""
    allowed, reason = decision if isinstance(decision, tuple) else (decision, None)
    if not allowed:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason or "denied by agpolicy",
            }
        }
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}


__all__ = [
    "resolve_mediation_mode",
    "hook_payload_to_syscallevent",
    "decision_to_hook_response",
]
