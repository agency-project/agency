"""Long-lived sandbox-side Harness Manager process.

The daemon owns orchestration and transport only. Harness selection remains
in ``harness.adapters`` and final attempt results are returned on the same
HTTP-over-UDS request that submitted them.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import signal
import subprocess
import threading
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Callable

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..configs.agconfig import agconfig as agconfig_cls, harnessadapterconfig, ptraceconfig
from . import interaction_router, mcp_proxy, sandbox_mcp
from .adapters.agharness_backend import AdapterRuntime, AttemptResult, agharness_backend
from .clients.host_services_client import HostServicesClient
from .common import extract_bearer_token
from .protocol import HarnessAttemptRequest, HarnessAttemptResult
from .servers import HarnessInteractionServer

_HARNESS_API_PORT = 8766


class _HostSyscallPolicy:
    """Ptrace policy adapter backed by the host interaction service."""

    def __init__(self, host_services: HostServicesClient, attempt_token: str) -> None:
        self._host_services = host_services
        self._attempt_token = attempt_token

    def check(self, _agent, syscall):
        return self._host_services.check_syscall_policy(self._attempt_token, syscall)

    def check_completion(self, _agent, call_id: "str | None", return_value: int) -> None:
        # A denied (never admitted) syscall has no call_id -- the ptrace
        # exit-hook only fires for admitted syscalls anyway (see
        # _tracer_loop.py), but stay defensive here too.
        if not call_id:
            return
        self._host_services.complete_syscall_policy(self._attempt_token, call_id, return_value)


class _LocalSandboxBackend:
    IMAGE_KIND = "container"

    def ingest_ptrace_pids(self, **_changes) -> None:
        return None


class _LocalSandbox:
    """The adapter sandbox surface when it already runs inside the sandbox."""

    def __init__(self) -> None:
        self._backend = _LocalSandboxBackend()

    def exec(self, cmd: str, workdir: str = "/workspace", timeout: int = 600):
        completed = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            cwd=workdir,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        output = (completed.stdout + completed.stderr).decode("utf-8", errors="replace")
        return output, completed.returncode

    def read_file(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8")

    def read_file_bytes(self, path: str) -> bytes:
        return Path(path).read_bytes()

    def write_file_bytes(self, path: str, data: bytes) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


class _HarnessApiServer:
    def __init__(self, host_uds_path: str, port: int, harness_backend: agharness_backend) -> None:
        self._bridge = HostServicesClient(host_uds_path)
        self._port = port
        self._harness_backend = harness_backend
        self._server: "uvicorn.Server | None" = None
        self._thread: "threading.Thread | None" = None
        self._loop: "asyncio.AbstractEventLoop | None" = None
        self._sandbox_mcp_app = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def change_config(self, agconfig: "agconfig_cls") -> None:
        self._harness_backend.change_config(agconfig)

    def start(self, timeout_s: float = 10.0) -> None:
        @asynccontextmanager
        async def lifespan(app):
            self._loop = asyncio.get_running_loop()
            try:
                yield
            finally:
                self._loop = None

        app = FastAPI(lifespan=lifespan)
        self._harness_backend.register(app, self._bridge)
        app.include_router(interaction_router.build_router(self._bridge))
        app.include_router(mcp_proxy.build_router(self._bridge))
        app.mount("/sandbox", self._sandbox_mcp_endpoint)
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self._port, log_level="warning")
        )
        self._server = server
        self._thread = threading.Thread(
            target=server.run,
            daemon=True,
            name="harness-facing-api",
        )
        self._thread.start()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not server.started:
            time.sleep(0.01)
        if not server.started:
            self.stop()
            raise RuntimeError("harness-facing API did not start within timeout")
        self._port = server.servers[0].sockets[0].getsockname()[1]

    async def _sandbox_mcp_endpoint(self, scope, receive, send) -> None:
        token = extract_bearer_token(Request(scope))
        app = self._sandbox_mcp_app
        if app is None or not token or not self._bridge.validate_token(token):
            response = JSONResponse({"error": "inactive sandbox MCP attempt"}, status_code=401)
            await response(scope, receive, send)
            return
        await app(scope, receive, send)

    async def run_sandbox_attempt(self, request, handler) -> HarnessAttemptResult:
        # Enter and exit MCP's task-group lifespan in the same API-loop task.
        # Closing it drains admitted tool calls before another attempt can start.
        async with AsyncExitStack() as stack:
            try:
                app = sandbox_mcp.build_app(
                    request.sandbox_mcp_tools_b64,
                    lambda: (
                        self._sandbox_mcp_app is app
                        and self._bridge.validate_token(request.attempt_token)
                    ),
                )
                await stack.enter_async_context(app.router.lifespan_context(app))
            except Exception as exc:
                message = (
                    str(exc)
                    if isinstance(exc, sandbox_mcp.SandboxMcpSetupError)
                    else (f"sandbox MCP setup failed starting server ({type(exc).__name__})")
                )
                return HarnessAttemptResult(ok=False, error_message=message)
            self._sandbox_mcp_app = app
            try:
                return await asyncio.to_thread(handler, request)
            finally:
                self._sandbox_mcp_app = None

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._bridge.close()
        self._server = None
        self._thread = None

    def register_attempt_token(self, token: str) -> None:
        self._bridge.register_attempt_token(token)

    def clear_attempt_token(self, token: str) -> bool:
        return self._bridge.clear_attempt_token(token)

    def resolve_model(self, token: str) -> str:
        return self._bridge.resolve_model(token)

    def syscall_policy(self, token: str) -> _HostSyscallPolicy:
        return _HostSyscallPolicy(self._bridge, token)


def _render_attempt_prompt(request: HarnessAttemptRequest) -> str:
    user_content = request.prompt.user_content
    if not isinstance(user_content, str):
        user_content = json.dumps(user_content)
    parts = [request.prompt.system_instruction, user_content]
    if request.prompt.output_instruction:
        parts.append(request.prompt.output_instruction)
    return "\n\n".join(part for part in parts if part)


def _run_adapter_attempt(
    request: HarnessAttemptRequest,
    agconfig: agconfig_cls,
    harness_base_url: str,
    model: str,
    engine_name: str,
    syscall_policy,
    register_control_handle: "Callable[[object], None]",
    register_redirect: "Callable[[Callable[[str], bool]], None]",
) -> HarnessAttemptResult:
    attempt_token = request.attempt_token
    if not isinstance(attempt_token, str) or not attempt_token:
        return HarnessAttemptResult(ok=False, error_message="missing harness attempt token")
    try:
        adapter = agharness_backend.for_config(request.harness, agconfig)
        if type(adapter).run_daemon_attempt is agharness_backend.run_daemon_attempt:
            return HarnessAttemptResult(
                ok=False,
                error_message=(
                    f"harness adapter {request.harness!r} has not adopted the single-attempt "
                    "daemon seam yet"
                ),
            )

        # The daemon is already inside the sandbox. Native uses a local
        # filesystem facade; external CLIs launch directly in this process's
        # namespace. No host-side agent or skill object crosses this boundary.
        runtime = AdapterRuntime(
            agconfig=agconfig,
            model=model,
            engine_name=engine_name,
            harness_base_url=harness_base_url,
            token=attempt_token,
            syscall_policy=syscall_policy,
            sandbox=_LocalSandbox() if request.harness == "native" else None,
            has_sandbox_mcp_tools=request.sandbox_mcp_tools_b64 is not None,
            register_control_handle=register_control_handle,
            register_redirect=register_redirect,
        )
        result: AttemptResult = adapter.run_daemon_attempt(
            runtime,
            prompt=_render_attempt_prompt(request),
            resume_session_id=request.resume_session_id,
            prior_session_blob=(
                base64.b64decode(request.prior_session_blob_b64)
                if request.prior_session_blob_b64 is not None
                else None
            ),
            max_steps=request.max_steps,
        )
    except Exception as exc:
        return HarnessAttemptResult(ok=False, error_message=f"{type(exc).__name__}: {exc}")
    return HarnessAttemptResult(
        ok=result.ok,
        final_text=result.final_text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        session_id=result.session_id,
        session_blob_b64=(
            base64.b64encode(result.session_blob).decode("ascii")
            if result.session_blob is not None
            else None
        ),
        error_message=result.error_message,
    )


class HarnessManager:
    def __init__(
        self,
        sandbox_uds_path: str,
        host_uds_path: str,
        engine_name: str,
        harness: str,
        *,
        agconfig: "agconfig_cls | None" = None,
        attempt_handler: "Callable[[HarnessAttemptRequest], HarnessAttemptResult] | None" = None,
        harness_api_port: int = _HARNESS_API_PORT,
    ) -> None:
        self._agconfig = agconfig if agconfig is not None else agconfig_cls()
        self._engine_name = engine_name
        harness_backend = agharness_backend.for_config(harness, self._agconfig)
        self._harness_api = _HarnessApiServer(host_uds_path, harness_api_port, harness_backend)
        self._attempt_handler = (
            attempt_handler if attempt_handler is not None else self._run_adapter_request
        )
        self._attempt_lock = threading.Lock()
        self._current_attempt_token: "str | None" = None
        # Separate from _attempt_lock, which is held for an attempt's ENTIRE
        # duration (see _dispatch_attempt()) -- a /control/* request must be
        # served concurrently with an in-flight attempt, not block behind it.
        self._control_lock = threading.Lock()
        self._current_control_handle: object = None
        self._current_request_id: "str | None" = None
        self._redirect_handler: "Callable[[str], bool] | None" = None
        # Sticky: persists across attempts for this daemon's whole life, so
        # a pause requested between attempts (or before the first one ever
        # ran) still applies the instant the next harness process exists --
        # see _register_control_handle().
        self._agent_paused = False
        self._interaction_server = HarnessInteractionServer(
            sandbox_uds_path,
            self._dispatch_attempt,
            control_handler=self.control,
            redirect_handler=self.redirect,
        )

    def change_config(self, agconfig: "agconfig_cls") -> None:
        self._agconfig = agconfig
        self._harness_api.change_config(self._agconfig)

    def _register_control_handle(self, handle: object) -> None:
        """Passed to each adapter as AdapterRuntime.register_control_handle
        -- called with this attempt's launch handle immediately after it
        starts. Applies a pause requested before this handle existed."""
        with self._control_lock:
            self._current_control_handle = handle
            if self._agent_paused:
                try:
                    handle.pause()
                except Exception as exc:
                    print(f"[harness_daemon] WARNING: pause() on new attempt failed: {exc}")

    def _clear_control_handle(self) -> None:
        with self._control_lock:
            self._current_control_handle = None
            self._current_request_id = None
            self._redirect_handler = None

    def _register_redirect(self, handler: "Callable[[str], bool]") -> None:
        with self._control_lock:
            self._redirect_handler = handler

    def redirect(self, request_id: str, message: str) -> bool:
        # Hold the existing control lock through delivery and terminal cleanup.
        # _attempt_lock keeps the next attempt from starting before that cleanup.
        # The adapter also fences its native completion against prompt acceptance.
        with self._control_lock:
            if request_id != self._current_request_id or self._redirect_handler is None:
                return False
            try:
                return self._redirect_handler(message)
            except Exception as exc:
                print(f"[harness_daemon] WARNING: redirect delivery failed: {exc}")
                return False

    def control(self, action: str) -> None:
        """Backs HarnessInteractionServer's /control/{action} route.
        No-op (not an error) when nothing is currently registered -- see
        agent.py's cancel()/pause()/resume() for why that's safe."""
        with self._control_lock:
            if action == "pause":
                self._agent_paused = True
            elif action == "resume":
                self._agent_paused = False
            handle = self._current_control_handle
            # A pause cannot begin midway through redirect's native input wait:
            # resume would otherwise block on this lock while input stayed frozen.
            if handle is None:
                return
            if action == "pause":
                handle.pause()
            elif action == "resume":
                handle.resume()
            elif action == "cancel":
                handle.kill()
            else:
                raise ValueError(f"unknown harness control action {action!r}")

    def _dispatch_attempt(self, request: HarnessAttemptRequest) -> HarnessAttemptResult:
        token = request.attempt_token
        if not isinstance(token, str) or not token:
            return HarnessAttemptResult(ok=False, error_message="missing harness attempt token")
        with self._attempt_lock:
            try:
                self._harness_api.register_attempt_token(token)
            except Exception as exc:
                return HarnessAttemptResult(
                    ok=False,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            self._current_attempt_token = token
            with self._control_lock:
                self._current_request_id = request.request_id
                self._redirect_handler = None
            try:
                if request.sandbox_mcp_tools_b64 is not None:
                    return asyncio.run_coroutine_threadsafe(
                        self._harness_api.run_sandbox_attempt(request, self._attempt_handler),
                        self._harness_api._loop,
                    ).result()
                return self._attempt_handler(request)
            finally:
                self._clear_control_handle()
                self._current_attempt_token = None
                self._harness_api.clear_attempt_token(token)

    def _run_adapter_request(self, request: HarnessAttemptRequest) -> HarnessAttemptResult:
        token = self._current_attempt_token
        if token is None or request.attempt_token != token:
            return HarnessAttemptResult(ok=False, error_message="no active harness attempt token")
        try:
            return _run_adapter_attempt(
                request,
                self._agconfig,
                self._harness_api.base_url,
                self._harness_api.resolve_model(token),
                self._engine_name,
                self._harness_api.syscall_policy(token),
                self._register_control_handle,
                self._register_redirect,
            )
        except Exception as exc:
            return HarnessAttemptResult(ok=False, error_message=f"{type(exc).__name__}: {exc}")

    def start(self) -> str:
        self._harness_api.start()
        try:
            return self._interaction_server.start()
        except BaseException:
            self._harness_api.stop()
            raise

    def stop(self) -> None:
        self._interaction_server.stop()
        self._harness_api.stop()


def _parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agency sandbox Harness Manager")
    parser.add_argument("--sandbox-uds", required=True)
    parser.add_argument("--host-uds", required=True)
    parser.add_argument("--engine-name", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--config-json", default="{}")
    parser.add_argument("--harness-api-port", type=int, default=_HARNESS_API_PORT)
    return parser.parse_args(argv)


def main(argv: "list[str] | None" = None) -> None:
    args = _parse_args(argv)
    payload = json.loads(args.config_json)
    manager = HarnessManager(
        args.sandbox_uds,
        args.host_uds,
        args.engine_name,
        args.harness,
        agconfig=agconfig_cls(
            harnessadapterconfig(**payload.get("harness_adapter", {})),
            ptraceconfig(**payload.get("ptrace", {})),
        ),
        harness_api_port=args.harness_api_port,
    )
    stopped = threading.Event()

    def _stop(_signum, _frame) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    manager.start()
    try:
        stopped.wait()
    finally:
        manager.stop()


if __name__ == "__main__":
    main()


__all__ = ["HarnessManager", "main"]
