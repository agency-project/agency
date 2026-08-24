"""Tests for the declarative tool and syscall policy configuration."""

from __future__ import annotations

from agency.agpolicy import agpolicy


def test_policy_defaults_to_allow_without_hooks():
    policy = agpolicy()

    assert policy.tool_hooks is None
    assert policy.syscall_hooks is None
    assert policy.default_to_deny is False


def test_policy_stores_tool_and_syscall_hooks():
    tool_hook = lambda _input: (False, "blocked tool")
    syscall_hook = lambda _event: True

    policy = agpolicy(
        tool_hooks={"bash": tool_hook},
        syscall_hooks={"execve": syscall_hook},
        default_to_deny=True,
    )

    assert policy.tool_hooks == {"bash": tool_hook}
    assert policy.syscall_hooks == {"execve": syscall_hook}
    assert policy.default_to_deny is True
