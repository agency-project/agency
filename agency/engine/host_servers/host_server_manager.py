from __future__ import annotations

import threading
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI

from ..agDataCollector import agDataCollector, agDataCollectorConfigs
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
        configured_manager = getattr(agent.agconfig, "HostServerManagerConfigs", None)
        if configured_manager is None:
            from ...agutil import new_uds_path

            configured_manager = HostServerManagerConfigs(
                uds_path=new_uds_path(f"host-server-{agent.agname}")
            )

        configured_collector = getattr(agent.agconfig, "agDataCollectorConfigs", None)
        if configured_collector is None:
            configured_collector = agDataCollectorConfigs(
                db_path=str(Path(configured_manager.uds_path).with_suffix(".db"))
            )

        # These two dataclass configs predate agConfig's registered-field
        # system. Keep one runtime view so real agents (which do not expose
        # them as attributes) receive safe per-execution defaults while the
        # existing explicit-config tests remain supported.
        self._runtime_config = SimpleNamespace(
            HostServerManagerConfigs=configured_manager,
            agDataCollectorConfigs=configured_collector,
        )
        self._data_collector = agDataCollector(self._runtime_config)
        self._llm_handler_server = LlmHandlerServer(agent.agconfig)
        self._host_mcp_server = HostMcpServer(sandbox, skill, resource_pool)
        self._harness_interaction_server = HarnessInteractionServer(
            agent, skill, self._data_collector
        )
        self._server_instances: "list[HostServerBase]" = [
            self._llm_handler_server,
            self._host_mcp_server,
            self._harness_interaction_server,
        ]
        self.set_config(agent.agconfig)

        self._server: "uvicorn.Server | None" = None
        self._server_thread: "threading.Thread | None" = None
        # Mark a component before entering its start() method: a start failure
        # can still leave partially initialized state that needs stop().  The
        # lists are also cleared component-by-component after successful
        # shutdown so a later stop() is idempotent while failed cleanup can be
        # retried.
        self._data_collector_needs_stop = False
        self._server_instances_to_stop: "list[HostServerBase]" = []

    def set_config(self, agconfig: "agConfig") -> None:
        manager_configs = getattr(agconfig, "HostServerManagerConfigs", None)
        if manager_configs is not None:
            self._runtime_config.HostServerManagerConfigs = manager_configs
        collector_configs = getattr(agconfig, "agDataCollectorConfigs", None)
        if collector_configs is not None:
            self._runtime_config.agDataCollectorConfigs = collector_configs

        self._configs = self._runtime_config.HostServerManagerConfigs
        self._data_collector.set_config(self._runtime_config)
        for server_instance in self._server_instances:
            server_instance.set_config(agconfig)

    def start(self) -> str:
        if self._server is not None:
            return self._configs.uds_path

        self._data_collector_needs_stop = True
        self._data_collector.start()
        for server_instance in self._server_instances:
            self._server_instances_to_stop.append(server_instance)
            server_instance.start()

        Path(self._configs.uds_path).parent.mkdir(parents=True, exist_ok=True)
        sub_apps = [
            (f"/{type(server_instance).__name__}", server_instance.build_app())
            for server_instance in self._server_instances
        ]

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            async with AsyncExitStack() as stack:
                for server_instance, (_, sub_app) in zip(self._server_instances, sub_apps):
                    ctx = server_instance.lifespan_context(sub_app)
                    if ctx is not None:
                        await stack.enter_async_context(ctx)
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
        first_error: "BaseException | None" = None

        def remember_failure(label: str, exc: BaseException) -> None:
            nonlocal first_error
            if first_error is None:
                first_error = exc
            else:
                cleanup_notes = tuple(getattr(exc, "__notes__", ()))
                first_error.add_note(f"{label} also failed: {exc}")
                if first_error is not exc:
                    for note in cleanup_notes:
                        first_error.add_note(f"{label}: {note}")

        server = self._server
        thread = self._server_thread
        if server is not None:
            try:
                server.should_exit = True
            except BaseException as exc:
                remember_failure("signalling the host server", exc)

        thread_is_alive = False
        if thread is not None:
            try:
                thread.join(timeout=self._configs.shutdown_timeout_s)
            except BaseException as exc:
                remember_failure("joining the host-server thread", exc)
            try:
                thread_is_alive = thread.is_alive()
            except BaseException as exc:
                # If liveness cannot be established, retain the handles so a
                # later stop() can retry instead of claiming shutdown.
                thread_is_alive = True
                remember_failure("checking the host-server thread", exc)

            if thread_is_alive:
                remember_failure(
                    "waiting for the host-server thread",
                    TimeoutError(
                        "HostServerManager did not stop within the configured "
                        f"{self._configs.shutdown_timeout_s}s timeout"
                    ),
                )

        if not thread_is_alive:
            self._server = None
            self._server_thread = None

        try:
            Path(self._configs.uds_path).unlink(missing_ok=True)
        except BaseException as exc:
            remember_failure("removing the host UDS", exc)

        still_needing_stop: "list[HostServerBase]" = []
        for server_instance in self._server_instances_to_stop:
            try:
                server_instance.stop()
            except BaseException as exc:
                still_needing_stop.append(server_instance)
                remember_failure(f"stopping {type(server_instance).__name__}", exc)
        self._server_instances_to_stop = still_needing_stop

        if self._data_collector_needs_stop:
            try:
                self._data_collector.stop()
            except BaseException as exc:
                remember_failure("stopping the data collector", exc)
            else:
                self._data_collector_needs_stop = False

        if first_error is not None:
            raise first_error
