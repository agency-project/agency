from __future__ import annotations

from typing import TYPE_CHECKING

from .host_servers.host_server_manager import HostServerManager
from .types import ExecutionResult

if TYPE_CHECKING:
    from ..agcontext import agcontext
    from ..agdata import agdata
    from ..agent import agent
    from ..agresources import agResourcePool
    from ..agskill import agskill


class agentEngine:
    def __init__(
        self,
        agent: "agent",
        context: "agcontext",
        skill: "agskill",
        skill_input: "agdata",
        resource_pool: "agResourcePool",
    ) -> None:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def run(self) -> ExecutionResult:
        raise NotImplementedError

    @property
    def host_server_manager(self) -> HostServerManager:
        raise NotImplementedError
