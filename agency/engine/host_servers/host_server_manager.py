from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI

from .harness_interaction_server import HarnessInteractionServer
from .host_mcp_server import HostMcpServer
from .host_server_base import HostServerBase
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


class HostServerManager(HostServerBase):
    def __init__(
        self,
        agent: "agent",
        sandbox: "agSandbox",
        skill: "agskill",
        resource_pool: "agResourcePool",
    ) -> None:
        self._llm_handler_server = LlmHandlerServer(agent.agconfig)
        self._host_mcp_server = HostMcpServer(sandbox, skill, resource_pool)
        self._harness_interaction_server = HarnessInteractionServer(agent, skill)
        self._server_instances: "list[HostServerBase]" = [
            self._llm_handler_server,
            self._host_mcp_server,
            self._harness_interaction_server,
        ]
        self.set_config(agent.agconfig)

        self._server: "uvicorn.Server | None" = None
        self._server_thread: "threading.Thread | None" = None

    def set_config(self, agconfig: "agConfig") -> None:
        self._configs = agconfig.HostServerManagerConfigs
        for server_instance in self._server_instances:
            server_instance.set_config(agconfig)

    def start(self) -> str:
        if self._server is not None:
            return self._configs.uds_path

        for server_instance in self._server_instances:
            server_instance.start()

        Path(self._configs.uds_path).parent.mkdir(parents=True, exist_ok=True)
        app = FastAPI()
        for server_instance in self._server_instances:
            app.mount(f"/{type(server_instance).__name__}", server_instance.build_app())
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

        for server_instance in self._server_instances:
            server_instance.stop()

    @property
    def harness_interaction_server(self) -> HarnessInteractionServer:
        return self._harness_interaction_server
