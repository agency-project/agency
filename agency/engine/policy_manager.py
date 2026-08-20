from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agconfig import agConfig


@dataclass
class EngineDecision:
    kind: str  # "allow" | "deny" | "rewrite"
    reason: "str | None" = None
    new_args: "list[str] | None" = None

    @staticmethod
    def allow() -> "EngineDecision":
        raise NotImplementedError

    @staticmethod
    def deny(reason: str) -> "EngineDecision":
        raise NotImplementedError

    @staticmethod
    def rewrite(new_args: "list[str]") -> "EngineDecision":
        raise NotImplementedError


class AgentHarnessPolicyManager:
    def __init__(self, agconfig: "agConfig") -> None:
        raise NotImplementedError

    def check_syscall(self, event: "object") -> EngineDecision:
        raise NotImplementedError

    def check_tool(self, tool_name: str, tool_input: dict) -> EngineDecision:
        raise NotImplementedError
