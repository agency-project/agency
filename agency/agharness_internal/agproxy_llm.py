"""Local HTTP server routing a harness's LLM traffic to an agent's `agllm`.

Agency's LLM client (`agllm.py`) is server-less -- it only ever makes
outbound calls. Wiring a harness's `ANTHROPIC_BASE_URL`/`model_providers.
base_url`/custom `provider` block at agency's own `agConfig` backend choice
requires becoming a server for the harness to connect *to*. This module is
that server: one process-wide FastAPI app, started on demand, routing each
harness's requests to the specific agent that launched it (via a per-run
bearer token), and to the exact wire format that agent's own backend
already speaks -- no protocol translation for this phase, see
docs/Design_harness_integration.md's Component 1.

Chat-completions passthrough only (Phase 3): this already matches agllm's
internal format 1:1, so the route below just forwards the request body to
the agent's own backend client and streams the response back unmodified --
no request/response reshaping. An Anthropic Messages / OpenAI Responses
adapter (for harnesses that don't speak chat-completions natively) is
future work, not built here.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..agconfig import GlobalConfigParam, DynamicConfigParam, _AgConfigViewBase

if TYPE_CHECKING:
    from ..agconfig import agConfig
    from ..agent import agent

# NOTE: FastAPI/Starlette resolve a route handler's parameter annotations
# (e.g. `request: Request`) from the function's *module-level* globals at
# runtime, via typing.get_type_hints() -- this matters because
# `from __future__ import annotations` (above) turns every annotation into
# a plain string, so `Request` must be importable from THIS module's
# globals for FastAPI to recognize it as the special injected-Request type
# rather than guessing it's a query parameter. A function-local/nested
# import of `Request` (e.g. inside _build_app()) breaks this silently --
# the route still "works" but every call 422s with "field required:
# request" instead of dispatching -- hit and fixed during development, see
# docs/agproxy_llm.md.


class _AgProxyLLMFields:
    bind_host = GlobalConfigParam("agproxy_llm", default="127.0.0.1")
    port = DynamicConfigParam(
        "agproxy_llm", default=0
    )  # 0 = OS-assigned ephemeral port; read back the real one from start()'s
    # return value, never assume this value is what actually got bound.
    request_timeout_s = GlobalConfigParam("agproxy_llm", default=300)

    def __init__(self, agconfig: "agConfig | None" = None) -> None:
        self._agconfig = agconfig


class agProxyLLMConfig(_AgConfigViewBase):
    _OWNER = "agproxy_llm"


def _extract_bearer_token(request) -> "str | None":
    auth = request.headers.get("authorization") or request.headers.get("x-api-key")
    if not auth:
        return None
    if auth.lower().startswith("bearer "):
        return auth[len("Bearer ") :].strip()
    return auth.strip()


class agProxyLLM:
    """One local HTTP server, shared across every harness-driven agent in
    this process. Each `launch()` (agharness_backends) mints a token via
    `register()` before starting its harness subprocess, and calls
    `unregister()` once that subprocess exits."""

    def __init__(self, agconfig: "agConfig | None" = None) -> None:
        self._agconfig = agconfig
        self._agents_by_token: "dict[str, agent]" = {}
        self._lock = threading.Lock()
        self._app = self._build_app()
        self._server = None
        self._thread: "threading.Thread | None" = None
        self.base_url: "str | None" = None

    # -- token <-> agent registry ---------------------------------------

    def register(self, token: str, ag: "agent") -> None:
        with self._lock:
            self._agents_by_token[token] = ag

    def unregister(self, token: str) -> None:
        with self._lock:
            self._agents_by_token.pop(token, None)

    def _agent_for_token(self, token: "str | None"):
        if token is None:
            return None
        with self._lock:
            return self._agents_by_token.get(token)

    # -- app / routes -----------------------------------------------------

    def _build_app(self):
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            token = _extract_bearer_token(request)
            ag = self._agent_for_token(token)
            if ag is None:
                return JSONResponse(
                    {"error": {"message": "unknown or missing bearer token"}}, status_code=401
                )
            body = await request.json()
            timeout_s = _AgProxyLLMFields(self._agconfig).request_timeout_s
            client = ag.llm.backend.make_client(httpx.Timeout(timeout_s))
            if body.get("stream"):

                def sse_gen():
                    for chunk in client.chat.completions.create(**body):
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(sse_gen(), media_type="text/event-stream")
            result = client.chat.completions.create(**body)
            return JSONResponse(result.model_dump())

        return app

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> str:
        """Start the server on a background thread; returns its base URL
        (e.g. "http://127.0.0.1:54321"). Idempotent -- calling twice returns
        the same URL without starting a second server."""
        if self.base_url is not None:
            return self.base_url

        fields = _AgProxyLLMFields(self._agconfig)
        config = uvicorn.Config(
            self._app, host=fields.bind_host, port=fields.port, log_level="warning"
        )
        server = uvicorn.Server(config)
        self._server = server

        self._thread = threading.Thread(target=server.run, daemon=True, name="agproxy_llm")
        self._thread.start()

        # uvicorn.Server.run() creates its sockets asynchronously on the
        # thread it's running on -- poll `started` rather than assuming any
        # fixed delay is enough.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not server.started:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("agProxyLLM server did not start within 10s")

        port = fields.port
        if port == 0 and server.servers:
            port = server.servers[0].sockets[0].getsockname()[1]
        self.base_url = f"http://{fields.bind_host}:{port}"
        return self.base_url

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None
        self._thread = None
        self.base_url = None


_shared_gateway: "agProxyLLM | None" = None
_shared_gateway_lock = threading.Lock()


def get_shared_gateway(agconfig: "agConfig | None" = None) -> agProxyLLM:
    """One `agProxyLLM` per process, lazily started on first use and shared
    by every harness-driven agent -- each launch registers its own bearer
    token against the same running server rather than each agent starting
    its own (which would mean N ports, N servers, for no benefit: routing
    is already per-token, not per-port)."""
    global _shared_gateway
    if _shared_gateway is not None:
        return _shared_gateway
    with _shared_gateway_lock:
        if _shared_gateway is None:
            gateway = agProxyLLM(agconfig)
            gateway.start()
            _shared_gateway = gateway
        return _shared_gateway


__all__ = ["agProxyLLM", "agProxyLLMConfig", "get_shared_gateway"]
