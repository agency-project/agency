from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .host_server_base import HostServerBase

if TYPE_CHECKING:
    from ...agconfig import agConfig
    from ...agent import agent
    from ...agskill import agskill
    from ...harness._syscall_event import agsyscallevent


class HarnessInteractionServer(HostServerBase):
    def __init__(self, agent: "agent", skill: "agskill") -> None:
        self._agent = agent
        self._policy = skill.policy
        self.set_config(agent.agconfig)

    def set_config(self, agconfig: "agConfig") -> None:
        self._agconfig = agconfig

    def check_tool(self, tool_name: str, tool_input: dict) -> "tuple[bool, str | None]":
        hook = (self._policy.tool_hooks or {}).get(tool_name)
        if hook is None:
            return (not self._policy.default_to_deny, None)
        result = hook(tool_input)
        return result if isinstance(result, tuple) else (result, None)

    def check_syscall(self, syscall: "agsyscallevent") -> "tuple[bool, str | None]":
        hook = (self._policy.syscall_hooks or {}).get(syscall.syscall)
        if hook is None:
            return (not self._policy.default_to_deny, None)
        result = hook(syscall)
        return result if isinstance(result, tuple) else (result, None)

    def check_inbox(self) -> "list[dict]":
        raise NotImplementedError

    def build_app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/check_tool")
        def _check_tool(request: dict) -> JSONResponse:
            allowed, reason = self.check_tool(request["tool_name"], request["tool_input"])
            return JSONResponse({"allowed": allowed, "reason": reason})

        @app.post("/check_inbox")
        def _check_inbox() -> JSONResponse:
            return JSONResponse({"messages": self.check_inbox()})

        return app
