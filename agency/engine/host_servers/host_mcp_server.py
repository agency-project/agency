from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .host_server_base import HostServerBase

if TYPE_CHECKING:
    from ...agconfig import agConfig
    from ...agresources import agResourcePool
    from ...agskill import agskill
    from ...sandbox.agsandbox import agSandbox


class HostMcpServer(HostServerBase):
    def __init__(
        self, sandbox: "agSandbox", skill: "agskill", resource_pool: "agResourcePool"
    ) -> None:
        self._sandbox = sandbox
        self._skill = skill
        self._resource_pool = resource_pool

    def set_config(self, agconfig: "agConfig") -> None:
        self._agconfig = agconfig

    def reserve_cpu(self, count: float) -> bool:
        raise NotImplementedError

    def cpu_release(self, count: float) -> None:
        raise NotImplementedError

    def daemon_release(self, pid: int) -> None:
        raise NotImplementedError

    def submit_output(self, field: str, value: "object") -> None:
        raise NotImplementedError

    def collected_output(self) -> dict:
        raise NotImplementedError

    def build_app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/reserve_cpu")
        def _reserve_cpu(request: dict) -> JSONResponse:
            return JSONResponse({"ok": self.reserve_cpu(request["count"])})

        @app.post("/cpu_release")
        def _cpu_release(request: dict) -> JSONResponse:
            self.cpu_release(request["count"])
            return JSONResponse({"ok": True})

        @app.post("/daemon_release")
        def _daemon_release(request: dict) -> JSONResponse:
            self.daemon_release(request["pid"])
            return JSONResponse({"ok": True})

        @app.post("/submit_output")
        def _submit_output(request: dict) -> JSONResponse:
            self.submit_output(request["field"], request["value"])
            return JSONResponse({"ok": True})

        @app.get("/collected_output")
        def _collected_output() -> JSONResponse:
            return JSONResponse(self.collected_output())

        return app
