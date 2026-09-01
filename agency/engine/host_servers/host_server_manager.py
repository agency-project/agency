from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI

from ...agutil import new_uds_path
from .host_interaction_server import HostInteractionServer
from .host_mcp_server import HostMcpServer
from .llm_handler_server import LlmHandlerServer

if TYPE_CHECKING:
    from ...agconfig import agConfig
    from ...agent import agent
    from ...agresources import agResourcePool
    from ...agskill import agskill
    from ...sandbox.agsandbox import agSandbox


@dataclass
class HostServerManagerConfigs:
    uds_path: str
    startup_timeout_s: float = 10.0
    shutdown_timeout_s: float = 10.0


class HostServerManager:
    def __init__(
        self,
        agent: "agent",
        sandbox: "agSandbox",
        skill: "agskill",
        resource_pool: "agResourcePool",
    ) -> None:
        from ...profiler import agprof

        self._ensure_runtime_configs(agent.agconfig)
        self._data_collector = agent.data_collector
        self._llm_handler_server = LlmHandlerServer(
            agent.agconfig, self._data_collector, parent_context=agprof.current_span_context()
        )
        self._host_mcp_server = HostMcpServer(sandbox, skill, resource_pool, self._data_collector)
        self._interaction_server = HostInteractionServer(skill, self._data_collector)
        self.set_config(agent.agconfig)

        self._server: "uvicorn.Server | None" = None
        self._server_thread: "threading.Thread | None" = None

    @property
    def interaction_server(self) -> "HostInteractionServer":
        return self._interaction_server

    @property
    def host_mcp_server(self) -> "HostMcpServer":
        return self._host_mcp_server

    @property
    def llm_handler_server(self) -> "LlmHandlerServer":
        return self._llm_handler_server

    def set_config(self, agconfig: "agConfig") -> None:
        self._ensure_runtime_configs(agconfig)
        self._configs = agconfig.HostServerManagerConfigs
        self._llm_handler_server.set_config(agconfig)

    def _ensure_runtime_configs(self, agconfig: "agConfig") -> None:
        manager_configs = agconfig.__dict__.get("HostServerManagerConfigs")
        if manager_configs is None:
            manager_configs = getattr(self, "_configs", None)
        if manager_configs is None:
            manager_configs = HostServerManagerConfigs(uds_path=new_uds_path("host"))
        agconfig.HostServerManagerConfigs = manager_configs

    def start(self) -> str:
        if self._server is not None:
            return self._configs.uds_path

        Path(self._configs.uds_path).parent.mkdir(parents=True, exist_ok=True)
        mcp_app = self._host_mcp_server.build_app()
        sub_apps = [
            ("/llm", self._llm_handler_server.build_app()),
            (
                "/interaction",
                self._interaction_server.build_app(),
            ),
            # MCP's Streamable HTTP app defines the exact route /mcp.
            # Mount it last at the root so that route remains /mcp rather
            # than becoming /mcp/mcp or redirecting to /mcp/.
            ("/", mcp_app),
        ]

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            async with self._host_mcp_server.lifespan_context(mcp_app):
                yield

        app = FastAPI(lifespan=lifespan)
        for prefix, sub_app in sub_apps:
            app.mount(prefix, sub_app)
        config = uvicorn.Config(app, uds=self._configs.uds_path, log_level="warning")
        server = uvicorn.Server(config)
        self._server = server
        self._server_thread = threading.Thread(
            target=server.run, daemon=True, name="host-server-manager"
        )
        self._server_thread.start()

        deadline = time.monotonic() + self._configs.startup_timeout_s
        while time.monotonic() < deadline and not server.started:
            time.sleep(0.01)
        assert server.started, "HostServerManager did not start within the configured timeout"
        return self._configs.uds_path

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=self._configs.shutdown_timeout_s)
        self._server = None
        self._server_thread = None

        self._llm_handler_server.stop()
