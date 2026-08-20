from __future__ import annotations

from typing import TYPE_CHECKING

from .host_server_manager.host_server_manager import HostServerManager
from .policy_manager import AgentHarnessPolicyManager
from .types import ExecutionResult

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agcontext import agcontext
    from ..agdata import agdata
    from ..agskill import agskill
    from ..sandbox.agsandbox import agSandbox


class agentEngine:
    def __init__(
        self,
        context: "agcontext",
        sandbox: "agSandbox",
        agconfig: "agConfig",
        skill: "agskill",
        skill_input: "agdata",
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

    @property
    def policy_manager(self) -> AgentHarnessPolicyManager:
        raise NotImplementedError
