from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .host_server_base import HostServerBase

if TYPE_CHECKING:
    from ...agconfig import agConfig


class LlmHandlerServer(HostServerBase):
    def __init__(self, agconfig: "agConfig") -> None:
        self.set_config(agconfig)

    def dispatch(self, request: dict) -> dict:
        raise NotImplementedError

    def resolve_model(self) -> str:
        raise NotImplementedError

    def context_limit(self) -> int:
        raise NotImplementedError

    def build_app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/dispatch")
        def _dispatch(request: dict) -> JSONResponse:
            return JSONResponse(self.dispatch(request))

        @app.get("/resolve_model")
        def _resolve_model() -> JSONResponse:
            return JSONResponse({"model": self.resolve_model()})

        @app.get("/context_limit")
        def _context_limit() -> JSONResponse:
            return JSONResponse({"context_limit": self.context_limit()})

        return app
