"""Long-lived sandbox-side Harness Manager process.

The daemon owns orchestration and transport only. Harness selection remains
in ``harness.adapters`` and final attempt results are returned on the same
HTTP-over-UDS request that submitted them.
"""

from __future__ import annotations

import argparse
import base64
import json
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

import uvicorn
from fastapi import FastAPI

from ..agconfig import agConfig
from . import interaction_router, llm_router, mcp_proxy
from .adapters.base import AdapterRuntime, AttemptResult, agharness_backend
from .clients.host_services_client import HostServicesClient
from .protocol import HarnessAttemptRequest, HarnessAttemptResult
from .servers import SandboxInteractionServer

_HARNESS_API_PORT = 8766


class _HostSyscallPolicy:
    """Ptrace policy adapter backed by the host interaction service."""

    def __init__(self, host_services: HostServicesClient) -> None:
        self._host_services = host_services

    def check(self, _agent, syscall):
        return self._host_services.check_syscall_policy(syscall)


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
    def __init__(self, host_uds_path: str, port: int) -> None:
        self._bridge = HostServicesClient(host_uds_path, None)
        self.syscall_policy = _HostSyscallPolicy(self._bridge)
        self.request_budget = llm_router.LlmRequestBudget()
        self._port = port
        self._server: "uvicorn.Server | None" = None
        self._thread: "threading.Thread | None" = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def start(self, timeout_s: float = 10.0) -> None:
        app = FastAPI()
        app.include_router(llm_router.build_router(self._bridge, self.request_budget))
        app.include_router(interaction_router.build_router(self._bridge))
        app.include_router(mcp_proxy.build_router(self._bridge))
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

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._bridge.client.close()
        self._server = None
        self._thread = None

    def resolve_model(self) -> str:
        return self._bridge.resolve_model("harness-daemon")


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
    agconfig: agConfig,
    harness_base_url: str,
    model: str,
    engine_name: str,
    syscall_policy,
) -> HarnessAttemptResult:
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
            token=f"daemon-{uuid.uuid4().hex}",
            syscall_policy=syscall_policy,
            sandbox=_LocalSandbox() if request.harness == "native" else None,
            suppress_builtin_tools=request.suppress_builtin_tools,
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
        *,
        agconfig: "agConfig | None" = None,
        attempt_handler: "Callable[[HarnessAttemptRequest], HarnessAttemptResult] | None" = None,
        harness_api_port: int = _HARNESS_API_PORT,
    ) -> None:
        self._agconfig = agconfig if agconfig is not None else agConfig()
        self._engine_name = engine_name
        self._harness_api = _HarnessApiServer(host_uds_path, harness_api_port)
        handler = attempt_handler if attempt_handler is not None else self._dispatch_attempt
        self._interaction_server = SandboxInteractionServer(sandbox_uds_path, handler)

    def _dispatch_attempt(self, request: HarnessAttemptRequest) -> HarnessAttemptResult:
        self._harness_api.request_budget.reset(request.max_steps)
        try:
            return _run_adapter_attempt(
                request,
                self._agconfig,
                self._harness_api.base_url,
                self._harness_api.resolve_model(),
                self._engine_name,
                self._harness_api.syscall_policy,
            )
        except Exception as exc:
            return HarnessAttemptResult(ok=False, error_message=f"{type(exc).__name__}: {exc}")
        finally:
            self._harness_api.request_budget.reset(None)

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
    parser.add_argument("--config-json", default="{}")
    parser.add_argument("--harness-api-port", type=int, default=_HARNESS_API_PORT)
    return parser.parse_args(argv)


def main(argv: "list[str] | None" = None) -> None:
    args = _parse_args(argv)
    manager = HarnessManager(
        args.sandbox_uds,
        args.host_uds,
        args.engine_name,
        agconfig=agConfig(json.loads(args.config_json)),
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
