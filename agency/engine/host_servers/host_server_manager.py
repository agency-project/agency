from __future__ import annotations

import hmac
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers

from ...harness.protocol import ATTEMPT_TOKEN_HEADER
from ...utils.agutil import new_uds_path
from .host_interaction_server import HostInteractionServer
from .host_mcp_server import HostMcpServer, bind_data_logger_for_current_thread
from .llm_handler_server import LlmHandlerServer

if TYPE_CHECKING:
    from ...configs.agconfig import agconfig as agconfig_cls
    from ...agent import agent
    from ...orchestrator.agresources import agResourcePool
    from ...agskill import agskill
    from ...sandbox.agsandbox import agSandbox


class _AttemptFenceMiddleware:
    """Hold an attempt lease through the complete mounted ASGI response.

    FastAPI's function middleware releases after ``call_next`` returns, which
    can precede consumption of a streaming response body.  Awaiting the inner
    ASGI app directly keeps the lease until its final body and cleanup finish.
    """

    def __init__(self, app, *, manager: "HostServerManager") -> None:
        self._app = app
        self._manager = manager

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        token = Headers(scope=scope).get(ATTEMPT_TOKEN_HEADER)
        if not self._manager._acquire_attempt_lease(token):
            response = JSONResponse(
                {"error": "unknown or inactive harness attempt token"},
                status_code=401,
            )
            await response(scope, receive, send)
            return

        try:
            await self._app(scope, receive, send)
        finally:
            self._manager._release_attempt_lease(token)


class HostServerManager:
    def __init__(
        self,
        agent: "agent",
        sandbox: "agSandbox",
        skill: "agskill",
        resource_pool: "agResourcePool",
        *,
        is_cancelled: "Callable[[], bool] | None" = None,
        request_id: "str | None" = None,
        recent_transcript: "list[dict] | None" = None,
    ) -> None:
        from ...observability.profiler import agprof

        agprof.register_engine(getattr(agent, "harness", "unknown"))
        self._ensure_runtime_configs(agent.agconfig)
        self._data_logger = agent.data_logger
        self._llm_handler_server = LlmHandlerServer(
            agent.agconfig,
            self._data_logger,
            agent.llm_usage_tracker,
            parent_context=agprof.current_span_context(),
            request_id=request_id,
            skill_name=skill.name,
            recent_transcript=recent_transcript,
        )
        self._interaction_server = HostInteractionServer(
            skill,
            self._data_logger,
            agent.agname,
            sandbox=sandbox,
            is_cancelled=is_cancelled,
            parent_context=agprof.current_span_context(),
            profile_attributes={
                **agprof.current_span_attributes(),
                "harness": getattr(agent, "harness", "unknown"),
            },
        )
        self._llm_handler_server._profile_context_provider = (
            self._interaction_server.current_open_context
        )
        self._host_mcp_server = HostMcpServer(
            sandbox,
            skill,
            resource_pool,
            self._data_logger,
            live_session=agent.agconfig.sandbox.checkpoint_fast_resume,
            # Native tools have already passed the loop's admission fence.
            is_cancelled=is_cancelled if getattr(agent, "harness", None) != "native" else None,
        )
        self.change_config(agent.agconfig)

        self._server: "uvicorn.Server | None" = None
        self._server_thread: "threading.Thread | None" = None
        self._lifecycle_lock = threading.RLock()
        self._attempt_token_condition = threading.Condition()
        # A fresh manager may be exercised without starting its UDS server by
        # focused callers/tests. Once shutdown begins, however, admission stays
        # closed until a later start has successfully brought up a new server.
        self._attempt_admission_closed = False
        self._active_attempt_token: "str | None" = None
        self._retiring_attempt_token: "str | None" = None
        self._attempt_inflight: "dict[str, int]" = {}

    @property
    def interaction_server(self) -> "HostInteractionServer":
        return self._interaction_server

    @property
    def host_mcp_server(self) -> "HostMcpServer":
        return self._host_mcp_server

    @property
    def llm_handler_server(self) -> "LlmHandlerServer":
        return self._llm_handler_server

    def change_config(self, agconfig: "agconfig_cls") -> None:
        self._ensure_runtime_configs(agconfig)
        self._configs = agconfig
        self._llm_handler_server.change_config(agconfig)

    def bind_attempt_token(self, token: str) -> None:
        """Authorize exactly one in-flight harness attempt."""
        if not isinstance(token, str) or not token:
            raise ValueError("attempt token must be a non-empty string")
        with self._attempt_token_condition:
            if self._attempt_admission_closed:
                raise RuntimeError("harness attempt admission is closed")
            if (
                self._active_attempt_token is not None
                or self._retiring_attempt_token is not None
                or self._attempt_inflight
            ):
                raise RuntimeError("another harness attempt token is already bound")
            self._active_attempt_token = token

    def clear_attempt_token(self, token: str) -> bool:
        """Revoke only the matching token so stale cleanup cannot clear its successor."""
        if not isinstance(token, str) or not token:
            return False
        with self._attempt_token_condition:
            active = self._active_attempt_token
            if active is None or not self._attempt_tokens_match(active, token):
                return False
            self._active_attempt_token = None
            self._retiring_attempt_token = token
            self._attempt_token_condition.wait_for(
                lambda: self._attempt_inflight.get(token, 0) == 0
            )
            self._attempt_inflight.pop(token, None)
            self._retiring_attempt_token = None
            self._interaction_server.finalize_profile()
            self._attempt_token_condition.notify_all()
            return True

    def _allows_attempt_token(self, token: "str | None") -> bool:
        if not isinstance(token, str) or not token:
            return False
        with self._attempt_token_condition:
            active = self._active_attempt_token
            return active is not None and self._attempt_tokens_match(active, token)

    def _acquire_attempt_lease(self, token: "str | None") -> bool:
        if not isinstance(token, str) or not token:
            return False
        with self._attempt_token_condition:
            active = self._active_attempt_token
            if active is None or not self._attempt_tokens_match(active, token):
                return False
            self._attempt_inflight[token] = self._attempt_inflight.get(token, 0) + 1
            return True

    @staticmethod
    def _attempt_tokens_match(active: str, candidate: str) -> bool:
        # compare_digest rejects non-ASCII ``str`` values. Comparing encoded
        # bytes keeps malformed or adversarial header values fail-closed.
        try:
            return hmac.compare_digest(active.encode(), candidate.encode())
        except UnicodeEncodeError:
            return False

    def _release_attempt_lease(self, token: str) -> None:
        with self._attempt_token_condition:
            count = self._attempt_inflight.get(token, 0)
            if count <= 0:
                raise RuntimeError("attempt lease released without a matching acquisition")
            if count == 1:
                self._attempt_inflight.pop(token, None)
            else:
                self._attempt_inflight[token] = count - 1
            self._attempt_token_condition.notify_all()

    def _ensure_runtime_configs(self, agconfig: "agconfig_cls") -> None:
        if agconfig.host_server.uds_path is None:
            existing = getattr(self, "_configs", None)
            agconfig.host_server.uds_path = (
                existing.host_server.uds_path
                if existing is not None and existing.host_server.uds_path is not None
                else new_uds_path("host")
            )

    def start(self) -> str:
        with self._lifecycle_lock:
            if self._server is not None:
                thread = self._server_thread
                if thread is not None and thread.is_alive() and self._server.started:
                    with self._attempt_token_condition:
                        if self._attempt_admission_closed:
                            raise RuntimeError("HostServerManager shutdown is still in progress")
                    return self._configs.host_server.uds_path
                if thread is not None and thread.is_alive():
                    raise RuntimeError("HostServerManager startup is already in progress")
                # A prior failed worker must never masquerade as a live server.
                self._server = None
                self._server_thread = None

            # Keep token admission closed throughout startup. A failed or
            # half-started server must never authorize a new attempt.
            with self._attempt_token_condition:
                self._attempt_admission_closed = True

            Path(self._configs.host_server.uds_path).parent.mkdir(parents=True, exist_ok=True)
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
            app.add_middleware(_AttemptFenceMiddleware, manager=self)

            for prefix, sub_app in sub_apps:
                app.mount(prefix, sub_app)
            config = uvicorn.Config(
                app, uds=self._configs.host_server.uds_path, log_level="warning"
            )
            server = uvicorn.Server(config)

            data_logger = self._data_logger

            def _run_server() -> None:
                bind_data_logger_for_current_thread(data_logger)
                server.run()

            thread = threading.Thread(target=_run_server, daemon=True, name="host-server-manager")
            self._server = server
            self._server_thread = thread
            try:
                thread.start()
            except BaseException:
                self._server = None
                self._server_thread = None
                raise

            deadline = time.monotonic() + self._configs.host_server.startup_timeout_s
            while time.monotonic() < deadline and not server.started and thread.is_alive():
                time.sleep(0.01)
            if not server.started:
                server.should_exit = True
                server.force_exit = True
                thread.join(timeout=self._configs.host_server.shutdown_timeout_s)
                if thread.is_alive():
                    # Keep both references so stop() can retry the shutdown.
                    raise RuntimeError(
                        "HostServerManager failed to start and its worker did not stop"
                    )
                self._server = None
                self._server_thread = None
                raise RuntimeError("HostServerManager did not start within the configured timeout")
            with self._attempt_token_condition:
                self._attempt_admission_closed = False
                self._attempt_token_condition.notify_all()
            return self._configs.host_server.uds_path

    def stop(self) -> None:
        with self._lifecycle_lock:
            with self._attempt_token_condition:
                # Revoke new admissions before beginning teardown. Existing
                # leases remain counted until their complete ASGI cleanup.
                self._attempt_admission_closed = True
                self._active_attempt_token = None
                self._attempt_token_condition.notify_all()

            llm_error: "BaseException | None" = None
            server_error: "BaseException | None" = None
            lease_error: "BaseException | None" = None

            # Closing provider streams first unblocks StreamingResponse tasks,
            # allowing Uvicorn's graceful shutdown to drain their ASGI leases.
            try:
                self._llm_handler_server.stop()
            except BaseException as exc:
                llm_error = exc
                from ...utils.agutil import format_exception

                print(
                    f"[host_server_manager] WARNING: llm_handler_server.stop() failed: "
                    f"{format_exception(exc)}"
                )

            server = self._server
            thread = self._server_thread
            if server is not None:
                server.should_exit = True
            if thread is not None:
                thread.join(timeout=self._configs.host_server.shutdown_timeout_s)
                if thread.is_alive():
                    if server is not None:
                        server.force_exit = True
                    thread.join(timeout=self._configs.host_server.shutdown_timeout_s)
                if thread.is_alive():
                    server_error = RuntimeError(
                        "HostServerManager worker did not stop within the configured timeout"
                    )
                else:
                    self._server = None
                    self._server_thread = None
            elif server is not None:
                self._server = None

            if server_error is None:
                with self._attempt_token_condition:
                    drained = self._attempt_token_condition.wait_for(
                        lambda: not self._attempt_inflight and self._retiring_attempt_token is None,
                        timeout=self._configs.host_server.shutdown_timeout_s,
                    )
                    if not drained:
                        lease_error = RuntimeError(
                            "HostServerManager attempt traffic did not drain during shutdown"
                        )

            self._interaction_server.finalize_profile()
            if server_error is not None:
                raise server_error
            if lease_error is not None:
                raise lease_error
            if llm_error is not None:
                raise llm_error
