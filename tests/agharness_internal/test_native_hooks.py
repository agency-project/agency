"""Tests for the harness-native hook fallback's translation logic.
Deliberately narrow scope -- see _native_hooks.py's module docstring:
this is a reduced-coverage fallback, not wired into any concrete backend's
execute() path yet."""

from __future__ import annotations

from unittest.mock import patch

from agency.agpolicy import agdecision
from agency.agharness_internal._native_hooks import (
    decision_to_hook_response,
    hook_payload_to_syscallevent,
    resolve_mediation_mode,
)


def test_resolve_mediation_mode_passes_through_explicit_ptrace():
    assert resolve_mediation_mode("ptrace") == "ptrace"


def test_resolve_mediation_mode_passes_through_explicit_native_hooks():
    assert resolve_mediation_mode("native_hooks") == "native_hooks"


def test_resolve_mediation_mode_auto_resolves_to_ptrace_when_available():
    with patch("agency.agharness_internal.agproxy_ptrace.ptrace_available", return_value=True):
        assert resolve_mediation_mode("auto") == "ptrace"


def test_resolve_mediation_mode_auto_resolves_to_native_hooks_when_unavailable():
    with patch("agency.agharness_internal.agproxy_ptrace.ptrace_available", return_value=False):
        assert resolve_mediation_mode("auto") == "native_hooks"


def test_hook_payload_to_syscallevent_bash_command():
    payload = {"tool_name": "Bash", "tool_input": {"command": "rm -rf /tmp/x"}, "pid": 123}
    event = hook_payload_to_syscallevent(payload)
    assert event.syscall == "Bash"
    assert event.pid == 123
    assert event.argv == ["rm", "-rf", "/tmp/x"]
    assert event.path is None
    assert event.tool_name == "Bash"
    assert event.tool_args == {"command": "rm -rf /tmp/x"}


def test_hook_event_uses_shared_architecture_neutral_policy_type():
    from agency.agharness_internal._syscall_event import agsyscallevent
    from agency.agharness_internal import _native_hooks

    event = hook_payload_to_syscallevent({"tool_name": "Read", "tool_input": {}})
    assert type(event) is agsyscallevent
    assert _native_hooks.agsyscallevent is agsyscallevent
    try:
        from agency.agharness_internal import agproxy_ptrace
    except RuntimeError:
        # The ptrace implementation itself is x86_64-only, but both import
        # sites still resolve their class from this neutral module.
        return
    assert agproxy_ptrace.agsyscallevent is agsyscallevent


def test_hook_payload_to_syscallevent_file_path():
    payload = {"tool_name": "Read", "tool_input": {"file_path": "/etc/passwd"}}
    event = hook_payload_to_syscallevent(payload)
    assert event.syscall == "Read"
    assert event.argv is None
    assert event.path == "/etc/passwd"


def test_decision_to_hook_response_allow():
    resp = decision_to_hook_response(agdecision.allow())
    assert resp["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert "updatedInput" not in resp["hookSpecificOutput"]


def test_decision_to_hook_response_deny():
    resp = decision_to_hook_response(agdecision.deny("nope"))
    assert resp["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert resp["hookSpecificOutput"]["permissionDecisionReason"] == "nope"


def test_decision_to_hook_response_rewrite():
    resp = decision_to_hook_response(agdecision.rewrite(["echo", "safe"]))
    assert resp["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert resp["hookSpecificOutput"]["updatedInput"]["command"] == "echo safe"
