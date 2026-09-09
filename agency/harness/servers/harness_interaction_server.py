"""Mockable sandbox-side RPC server for one harness attempt."""

from __future__ import annotations

import threading
import os
import time
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..protocol import HarnessAttemptRequest, HarnessAttemptResult, PromptPayload


def _unlink_socket(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise FileExistsError(f"refusing to replace non-socket path: {path}")
    path.unlink()


class HarnessInteractionServer:
    def __init__(
        self,
        uds_path: str,
        attempt_handler: Callable[[HarnessAttemptRequest], HarnessAttemptResult],
        *,
        control_handler: "Callable[[str], None] | None" = None,
        startup_timeout_s: float = 10.0,
        shutdown_timeout_s: float = 10.0,
    ) -> None:
        self.uds_path = uds_path
        self._attempt_handler = attempt_handler
        self._control_handler = control_handler
        self._startup_timeout_s = startup_timeout_s
        self._shutdown_timeout_s = shutdown_timeout_s
        self._server: "uvicorn.Server | None" = None
        self._server_thread: "threading.Thread | None" = None

    def build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/health")
        def _health() -> JSONResponse:
            # Keep the health payload compatible while identifying the owned
            # daemon in its sandbox PID namespace for process monitoring.
            headers = {"X-Agency-Daemon-Pid": str(os.getpid())}
            try:
                raw = Path("/proc/self/stat").read_text()
                headers["X-Agency-Daemon-Start-Ticks"] = raw.rsplit(")", 1)[1].split()[19]
            except (OSError, IndexError):
                pass  # Non-Linux health checks remain supported; identity is unavailable.
            return JSONResponse({"ready": True}, headers=headers)

        @app.post("/harness_attempt")
        def _harness_attempt(payload: dict) -> JSONResponse:
            prompt = PromptPayload(**payload["prompt"])
            request = HarnessAttemptRequest(
                prompt=prompt,
                harness=payload["harness"],
                max_steps=payload.get("max_steps"),
                resume_session_id=payload.get("resume_session_id"),
                prior_session_blob_b64=payload.get("prior_session_blob_b64"),
                attempt_token=payload.get("attempt_token"),
                sandbox_mcp_tools_b64=payload.get("sandbox_mcp_tools_b64"),
                syscall_default_to_deny=payload.get("syscall_default_to_deny", False),
                syscall_hooked_names=payload.get("syscall_hooked_names"),
            )
            return JSONResponse(asdict(self._attempt_handler(request)))

        @app.post("/control/{action}")
        def _control(action: str) -> JSONResponse:
            # A sync route -- like /harness_attempt above -- runs in
            # Starlette's threadpool, so this is served concurrently even
            # while /harness_attempt's handler is still blocked in its own
            # worker thread for a different in-flight attempt.
            if self._control_handler is None:
                return JSONResponse({"error": "no control handler configured"}, status_code=501)
            if action not in ("pause", "resume", "cancel"):
                return JSONResponse({"error": f"unknown action {action!r}"}, status_code=404)
            self._control_handler(action)
            return JSONResponse({"ok": True})

        return app

    def start(self) -> str:
        if self._server is not None:
            return self.uds_path

        socket_path = Path(self.uds_path)
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        _unlink_socket(socket_path)
        server = uvicorn.Server(
            uvicorn.Config(self.build_app(), uds=self.uds_path, log_level="warning")
        )
        self._server = server
        self._server_thread = threading.Thread(
            target=server.run,
            daemon=True,
            name="sandbox-interaction-server",
        )
        self._server_thread.start()

        deadline = time.monotonic() + self._startup_timeout_s
        while time.monotonic() < deadline and not server.started:
            time.sleep(0.01)
        if not server.started:
            self.stop()
            raise RuntimeError("HarnessInteractionServer did not start within timeout")
        return self.uds_path

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=self._shutdown_timeout_s)
        self._server = None
        self._server_thread = None
        _unlink_socket(Path(self.uds_path))


__all__ = ["HarnessInteractionServer"]
