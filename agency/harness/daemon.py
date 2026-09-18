"""Long-lived sandbox-side Harness Manager process.

The daemon owns orchestration and transport only. Harness selection remains
in ``harness.adapters`` and final attempt results are returned on the same
HTTP-over-UDS request that submitted them.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import json
import os
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

from ..configs.agconfig import (
    agconfig as agconfig_cls,
    harnessadapterconfig,
    ptraceconfig,
    sandboxconfig,
)
from . import interaction_router, mcp_proxy, sandbox_mcp
from .adapters.base import AdapterRuntime, AttemptResult, HarnessAdapter
from .clients.host_services_client import HostServicesClient
from .common import extract_bearer_token
from .executable import prepare_harness_executable_local
from .protocol import HarnessAttemptRequest, HarnessAttemptResult
from .servers import HarnessInteractionServer

_HARNESS_API_PORT = 8766

# Checked (and installed, if the harness Python isn't pinned) by a
# dependency-free pre-exec step *before* this module is ever imported -- see
# `_package_bootstrap_command()` in `engine/harness_daemon_launcher.py`.
# Importing this module already needs all of these transitively (fastapi/
# uvicorn directly, the rest via `agency`'s own package `__init__`), so by
# the time any code here runs, they're already guaranteed present.
_REQUIRED_HARNESS_PACKAGES = (
    "fastapi",
    "uvicorn",
    "openai",
    "httpx",
    "mcp",
    "pyseccomp",
    "cloudpickle",
    "pyte",
)

# Backs _HostSyscallPolicy's fire-and-forget admission logging (see its
# docstring). Module-level and shared for the daemon process's whole life --
# a pool per attempt would leak idle worker threads, since
# ThreadPoolExecutor workers never self-terminate.
_SYSCALL_LOG_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="syscall-admission-log"
)


class _HostSyscallPolicy:
    """Ptrace policy adapter backed by the host interaction service.

    Only syscalls this attempt's policy actually hooks (plus exec-family
    ones, see below) need the synchronous host round-trip -- every other
    syscall has a decision that is fully known before the harness even
    launches (``not default_to_deny``), so it never sits on the traced
    process's critical path. The same ``/interaction/check_syscall`` call
    still fires for those, just in the background and with its result
    discarded, purely so the admission still gets logged (tool_call/
    agent_state events, the "SYSCALL" line) exactly like a hooked call's
    does. Its returned ``call_id`` (the host stashes it as a pending call
    awaiting completion) is immediately closed out with an "unknown"
    outcome, since nothing here ever observes the syscall's real return
    value for an admission that was never actually gated on that host
    round-trip.

    ``execve``/``execveat`` are always forced onto the real synchronous
    path regardless of ``hooked_syscalls`` -- the host uses exactly this
    admission to gate a pending GPU reservation (blocking until one is
    actually free) and to scope CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES
    onto the process about to run, neither of which the short-circuit's
    "decision known before launch" premise holds for.
    """

    _ALWAYS_SYNCHRONOUS = frozenset({"execve", "execveat"})

    def __init__(
        self,
        host_services: HostServicesClient,
        attempt_token: str,
        *,
        default_to_deny: bool = False,
        hooked_syscalls: "frozenset[str] | None" = None,
    ) -> None:
        self._host_services = host_services
        self._attempt_token = attempt_token
        self._default_to_deny = default_to_deny
        self._hooked_syscalls = hooked_syscalls or frozenset()

    def record_file_access(self, payload):
        self._host_services.record_file_access(self._attempt_token, payload)

    def check(self, _agent, syscall):
        if syscall.syscall in self._hooked_syscalls or syscall.syscall in self._ALWAYS_SYNCHRONOUS:
            try:
                return self._host_services.check_syscall_policy(self._attempt_token, syscall)
            except Exception:
                if self._host_services.validate_token(self._attempt_token):
                    raise
                # A retained CLI can reach another exec stop as its completed
                # attempt is revoked. Deny it without killing the tracer that
                # must still quiesce the tree for checkpointing.
                return False, "inactive harness attempt", None, None
        _SYSCALL_LOG_POOL.submit(self._log_admission_best_effort, syscall)
        return (not self._default_to_deny, None, None, None)

    def _log_admission_best_effort(self, syscall) -> None:
        try:
            _allowed, _reason, call_id = self._host_services.check_syscall_policy(
                self._attempt_token, syscall
            )
        except Exception:
            return  # best-effort logging only -- never affects the syscall's outcome
        if not call_id:
            return
        try:
            self._host_services.complete_syscall_policy(
                self._attempt_token, call_id, return_value=None
            )
        except Exception:  # noqa: S110 - best-effort logging cleanup only, never the syscall's outcome
            pass

    def check_completion(self, _agent, call_id: "str | None", return_value: int) -> None:
        # A denied (never admitted) syscall has no call_id -- the ptrace
        # exit-hook only fires for admitted syscalls anyway (see
        # _tracer_loop.py), but stay defensive here too.
        if not call_id:
            return
        try:
            self._host_services.complete_syscall_policy(self._attempt_token, call_id, return_value)
        except Exception:
            if self._host_services.validate_token(self._attempt_token):
                raise
            # The syscall was already admitted. Its late completion cannot
            # report into a retired host attempt or fail the retained tracer.


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
    def __init__(
        self,
        host_uds_path: str,
        port: int,
        harness_backend: HarnessAdapter,
        *,
        live_session: bool = False,
    ) -> None:
        self._bridge = HostServicesClient(host_uds_path)
        self._port = port
        self._harness_backend = harness_backend
        self._server: "uvicorn.Server | None" = None
        self._thread: "threading.Thread | None" = None
        self._loop: "asyncio.AbstractEventLoop | None" = None
        self._sandbox_mcp_app = None
        self._persistent = live_session

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

    async def run_sandbox_attempt(
        self, request, handler, *, local_token: "str | None" = None
    ) -> HarnessAttemptResult:
        # Enter and exit MCP's task-group lifespan in the same API-loop task.
        # Closing it drains admitted tool calls before another attempt can start.
        accepted_token = local_token if local_token is not None else request.attempt_token
        async with AsyncExitStack() as stack:
            try:
                app = sandbox_mcp.build_app(
                    request.sandbox_mcp_tools_b64,
                    lambda: (
                        self._sandbox_mcp_app is app and self._bridge.validate_token(accepted_token)
                    ),
                    live_session=self._persistent,
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

    def register_attempt_token(self, token: str, *, local_token: "str | None" = None) -> None:
        self._bridge.register_attempt_token(token, local_token=local_token)

    def clear_attempt_token(self, token: str) -> bool:
        return self._bridge.clear_attempt_token(token)

    def prepare_fast_checkpoint(self) -> None:
        # Drop idle host-UDS keepalive sockets.  A restored daemon reconnects
        # lazily through the same mounted path when the next attempt starts.
        self._bridge.quiesce_connections()

    def complete_fast_restore(self) -> None:
        return None

    def resolve_model(self, token: str) -> str:
        return self._bridge.resolve_model(token)

    def syscall_policy(
        self,
        token: str,
        *,
        default_to_deny: bool = False,
        hooked_syscalls: "frozenset[str] | None" = None,
    ) -> _HostSyscallPolicy:
        return _HostSyscallPolicy(
            self._bridge,
            token,
            default_to_deny=default_to_deny,
            hooked_syscalls=hooked_syscalls,
        )


def _render_attempt_prompt(request: HarnessAttemptRequest) -> str:
    user_content = request.prompt.user_content
    if not isinstance(user_content, str):
        user_content = json.dumps(user_content)
    parts = (
        [user_content]
        if request.resume_session_id
        else [request.prompt.system_instruction, user_content]
    )
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
    run_pty_execution=None,
    local_attempt_token: "str | None" = None,
) -> HarnessAttemptResult:
    attempt_token = local_attempt_token or request.attempt_token
    if not isinstance(attempt_token, str) or not attempt_token:
        return HarnessAttemptResult(ok=False, error_message="missing harness attempt token")
    try:
        adapter = HarnessAdapter.for_config(request.harness, agconfig)
        if type(adapter).run_daemon_attempt is HarnessAdapter.run_daemon_attempt:
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
        if run_pty_execution is None:
            from .adapters.base import _run_one_pty_execution

            run_pty_execution = _run_one_pty_execution
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
            run_pty_execution=run_pty_execution,
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
        self._harness = harness
        self._bootstrapped = False
        self._persistent = bool(self._agconfig.sandbox.checkpoint_fast_resume)
        harness_backend = HarnessAdapter.for_config(harness, self._agconfig)
        self._harness_api = _HarnessApiServer(
            host_uds_path, harness_api_port, harness_backend, live_session=self._persistent
        )
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
        self._current_request_cancelled = False
        self._current_request_id: "str | None" = None
        self._redirect_handler: "Callable[[str], bool] | None" = None
        self._live_execution = None
        self._live_execution_key = None
        self._live_local_token: "str | None" = None
        self._fast_checkpoint_prepared = False
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
            lifecycle_handler=self.lifecycle,
        )

    def _retire_live_execution(self) -> None:
        execution = getattr(self, "_live_execution", None)
        self._live_execution = None
        self._live_execution_key = None
        self._live_local_token = None
        if execution is not None:
            execution.close()

    def _run_pty_execution(self, key, factory, prompt):
        """Run one prompt, retaining an idle PTY only for CRIU fast resume."""
        if not self._persistent:
            return factory().run(prompt)
        execution = self._live_execution
        if execution is not None and (
            self._live_execution_key != key
            or execution.handle is None
            or execution.handle.returncode is not None
        ):
            self._retire_live_execution()
            execution = None
        if execution is None:
            execution = factory()
            self._live_execution = execution
            self._live_execution_key = key
        try:
            return execution.run(prompt, keep_alive=True)
        except BaseException:
            self._retire_live_execution()
            raise

    def lifecycle(self, action: str) -> dict:
        """Quiesce or resume the retained ptrace tree around runtime CRIU."""
        with self._attempt_lock:
            execution = self._live_execution
            handle = execution.handle if execution is not None else None

            def status():
                return {
                    "ok": True,
                    "live_pty": execution is not None,
                    "daemon_pid": os.getpid(),
                    "root_pid": handle.root_pid if handle is not None else None,
                    "pids": sorted(handle.pids()) if handle is not None else [],
                }

            if action == "prepare_fast_checkpoint":
                if self._fast_checkpoint_prepared:
                    return status()
                self._harness_api.prepare_fast_checkpoint()
                if execution is not None:
                    execution.prepare_fast_checkpoint()
                self._fast_checkpoint_prepared = True
                return status()
            if action == "seize_fast_restore":
                if not self._fast_checkpoint_prepared:
                    raise RuntimeError("fast checkpoint handoff is not prepared")
                if execution is not None:
                    execution.seize_fast_restore()
                return status()
            if action in {"complete_fast_restore", "abort_fast_checkpoint"}:
                if self._fast_checkpoint_prepared:
                    if execution is not None:
                        execution.complete_fast_restore()
                    self._harness_api.complete_fast_restore()
                    self._fast_checkpoint_prepared = False
                return status()
            raise ValueError(f"unknown lifecycle action {action!r}")

    def change_config(self, agconfig: "agconfig_cls") -> None:
        self._agconfig = agconfig
        self._harness_api.change_config(self._agconfig)

    def _register_control_handle(self, handle: object) -> None:
        """Passed to each adapter as AdapterRuntime.register_control_handle
        -- called with this attempt's launch handle immediately after it
        starts. Applies a pause requested before this handle existed."""
        with self._control_lock:
            self._current_control_handle = handle
            if self._current_request_cancelled:
                handle.kill()
            elif self._agent_paused:
                try:
                    handle.pause()
                except Exception as exc:
                    print(f"[harness_daemon] WARNING: pause() on new attempt failed: {exc}")
            elif self._persistent:
                # A restored tracee remains parked until the next invocation
                # has installed its token and syscall policy.
                handle.resume()

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

    def control(self, action: str, *, request_id: str | None = None) -> None:
        """Backs HarnessInteractionServer's /control/{action} route.
        No-op (not an error) when nothing is currently registered -- see
        agent.py's cancel()/pause()/resume() for why that's safe."""
        with self._control_lock:
            if action == "cancel" and (
                request_id is None or request_id != self._current_request_id
            ):
                return
            if action == "cancel":
                self._current_request_cancelled = True
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
            local_token = getattr(self, "_live_local_token", None) or token
            try:
                if local_token == token:
                    self._harness_api.register_attempt_token(token)
                else:
                    self._harness_api.register_attempt_token(token, local_token=local_token)
            except Exception as exc:
                return HarnessAttemptResult(
                    ok=False,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            self._current_attempt_token = token
            with self._control_lock:
                self._current_request_id = request.request_id
                self._current_request_cancelled = False
                self._redirect_handler = None
            try:
                if request.sandbox_mcp_tools_b64 is not None:
                    return asyncio.run_coroutine_threadsafe(
                        self._harness_api.run_sandbox_attempt(
                            request, self._attempt_handler, local_token=local_token
                        ),
                        self._harness_api._loop,
                    ).result()
                return self._attempt_handler(request)
            finally:
                self._clear_control_handle()
                self._current_attempt_token = None
                self._harness_api.clear_attempt_token(token)
                if (
                    getattr(self, "_live_execution", None) is not None
                    and getattr(self, "_live_local_token", None) is None
                ):
                    self._live_local_token = local_token

    def _run_adapter_request(self, request: HarnessAttemptRequest) -> HarnessAttemptResult:
        token = self._current_attempt_token
        if token is None or request.attempt_token != token:
            return HarnessAttemptResult(ok=False, error_message="no active harness attempt token")
        local_token = getattr(self, "_live_local_token", None) or token
        try:
            if not self._bootstrapped:
                # Runs once, on the first real attempt, so a failure here is
                # reported the exact same way any other attempt failure is:
                # caught below and returned as this request's
                # HarnessAttemptResult.
                prepare_harness_executable_local(self._harness, self._agconfig)
                self._bootstrapped = True
            arguments = (
                request,
                self._agconfig,
                self._harness_api.base_url,
                self._harness_api.resolve_model(local_token),
                self._engine_name,
                self._harness_api.syscall_policy(
                    local_token,
                    default_to_deny=request.syscall_default_to_deny,
                    hooked_syscalls=frozenset(request.syscall_hooked_names or ()),
                ),
                self._register_control_handle,
                self._register_redirect,
            )
            if getattr(self, "_persistent", False):
                policy_key = (
                    bool(request.syscall_default_to_deny),
                    tuple(sorted(request.syscall_hooked_names or ())),
                )
                runner = lambda key, factory, prompt: self._run_pty_execution(
                    (policy_key, key), factory, prompt
                )
                return _run_adapter_attempt(*arguments, runner, local_token)
            return _run_adapter_attempt(*arguments)
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
        self._retire_live_execution()
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
            sandboxconfig(**payload.get("sandbox", {})),
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
