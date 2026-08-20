from __future__ import annotations

from typing import TYPE_CHECKING

from .llm_api_handler import LlmApiHandler
from .mcp_tools import HostMcpTools

if TYPE_CHECKING:
    from ...agconfig import agConfig
    from ...agcontext import agcontext
    from ...agskill import agskill
    from ...sandbox.agsandbox import agSandbox


class HostServerManager:
    def __init__(
        self,
        context: "agcontext",
        sandbox: "agSandbox",
        agconfig: "agConfig",
        skill: "agskill",
    ) -> None:
        raise NotImplementedError

    def start(self) -> str:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    @property
    def llm_api_handler(self) -> LlmApiHandler:
        raise NotImplementedError

    @property
    def mcp_tools(self) -> HostMcpTools:
        raise NotImplementedError
