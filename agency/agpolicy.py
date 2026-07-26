"""Mediation interface for externally-driven (harness) execution.

`agpolicy` is the decision point `agproxy_ptrace` (syscall-level interception
for harness-driven agents) calls on every intercepted event, and the point
`agtool.dispatch_tools()` will call for native tool calls once the retrofit
in the design doc's later build phase lands. Same interface, two different
event sources -- a syscall trace stop for a harness-driven agent, a tool-call
dispatch for a native one -- so one policy implementation can govern a mixed
team of both.

Deliberately has exactly one bootstrapping implementation here
(`agAllowAllPolicy`) -- this module only defines the contract; anything with
real allow/deny/rewrite logic belongs in the caller's own code, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import agent
    from .agharness_internal.agproxy_ptrace import agsyscallevent


@dataclass
class agdecision:
    kind: str  # "allow" | "deny" | "rewrite"
    reason: "str | None" = None
    new_args: "list[str] | None" = None

    @staticmethod
    def allow() -> "agdecision":
        return agdecision(kind="allow")

    @staticmethod
    def deny(reason: str) -> "agdecision":
        return agdecision(kind="deny", reason=reason)

    @staticmethod
    def rewrite(new_args: "list[str]") -> "agdecision":
        return agdecision(kind="rewrite", new_args=new_args)


class agpolicy:
    """Base class for a mediation policy. Subclass and override `check()`
    and `check_tool()`."""

    def check(self, ag: "agent", event: "agsyscallevent") -> agdecision:
        raise NotImplementedError

    def check_tool(self, ag: "agent", tool_name: str, tool_input: dict) -> agdecision:
        """Mediate a harness's own tool call, reported through its native
        permission-check mechanism (e.g. Claude Code's `PreToolUse` hook)
        rather than observed at the syscall level -- a harness's tools
        (Write, Bash, ...) don't map 1:1 onto individual syscalls, so this
        is a second, harness-native mediation point alongside `check()`,
        not a replacement for it. See `agproxy_llm.py`'s
        `/agpolicy/check_tool` route, which is what a harness's hook
        script actually calls."""
        raise NotImplementedError


class agAllowAllPolicy(agpolicy):
    """Allows every event unmediated. Exists to bootstrap
    agproxy_ptrace's launch/trace/policy loop before any real allow/deny/
    rewrite logic is wired up -- not a real security boundary."""

    def check(self, ag: "agent", event: "agsyscallevent") -> agdecision:
        return agdecision.allow()

    def check_tool(self, ag: "agent", tool_name: str, tool_input: dict) -> agdecision:
        return agdecision.allow()
