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

Three routes: `/v1/chat/completions` is a straight passthrough (already
matches agllm's internal format 1:1, used by opencode/Grok Build, whose
providers already speak chat-completions). `/v1/messages` (Anthropic
Messages API, used by Claude Code) and `/v1/responses` (OpenAI Responses
API, used by Codex) are translated: the incoming request is reshaped into
chat-completions kwargs, dispatched through the exact same
`client.chat.completions.create()` every backend uniformly exposes, and the
(possibly streaming) response reshaped back into the harness's native
format -- see `agproxy_llm_adapters.py` for the conversion functions
themselves, kept in a separate module so this file stays about routing, not
wire-format detail.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import TYPE_CHECKING

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..agconfig import GlobalConfigParam, DynamicConfigParam, _AgConfigViewBase
from .agproxy_llm_adapters import (
    anthropic_messages_to_openai,
    openai_response_to_anthropic_message,
    openai_chunks_to_anthropic_sse,
    responses_request_to_openai,
    openai_response_to_responses_api,
    openai_chunks_to_responses_sse,
)

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


_DEBUG_CAPTURE_MARKERS = ("Available agent types", "task tools haven't been used", "gentle reminder")


def _debug_log_anthropic_messages_body(path: str, body: dict) -> None:
    """Temporary diagnostic: dump raw incoming /v1/messages shape (roles,
    mid-array system messages, reminder markers) to `path` for direct
    inspection of what the harness actually sent, before any adapter
    transformation. Opt-in only, via AGENCY_DEBUG_CAPTURE_LOG."""
    messages = body.get("messages", [])
    roles = [m.get("role") for m in messages]
    mid_array_system = "system" in roles[1:] if roles else False
    lines = [f"\n=== num_messages={len(roles)} roles={roles} "
             f"{'!!! MID-ARRAY SYSTEM !!!' if mid_array_system else ''}"]
    sys_field = body.get("system")
    if sys_field:
        sys_text = sys_field if isinstance(sys_field, str) else json.dumps(sys_field)
        lines.append(f"    top-level system ({len(sys_text)} chars)")
        for marker in _DEBUG_CAPTURE_MARKERS:
            if marker in sys_text:
                lines.append(f"    >>> top-level system contains marker: {marker!r}")
    else:
        lines.append("    top-level system: ABSENT")
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            content = m.get("content")
            text = content if isinstance(content, str) else json.dumps(content)
            lines.append(f"    >>> MID-ARRAY SYSTEM MESSAGE at index {i} ({len(text)} chars): {text[:300]!r}")
            for marker in _DEBUG_CAPTURE_MARKERS:
                if marker in text:
                    lines.append(f"        >>> contains marker: {marker!r}")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def _warn_mid_array_system_messages(ag: "agent", body: dict) -> None:
    """Claude Code's generic (ANTHROPIC_BASE_URL) client sometimes emits its
    own dynamic reminders (e.g. an Agent-tool-availability nudge) as a
    `role: "system"` entry inside `messages`, not the top-level `system`
    field -- a shape the real Anthropic Messages API and AWS Bedrock's
    Anthropic-invoke endpoint both reject outright (confirmed directly
    against both). `anthropic_messages_to_openai` below folds every such
    occurrence into the one leading system message so it never reaches a
    real backend, but that's a silent correctness workaround for what looks
    like an inconsistency in Claude Code's own request serialization on
    this client path -- surface it instead of absorbing it invisibly."""
    n = sum(1 for m in body.get("messages", []) if m.get("role") == "system")
    if n:
        ag.terminal.log(
            "WARNING  ",
            f"harness emitted {n} mid-conversation system-role message(s) in "
            "its /v1/messages request -- not valid per the Anthropic Messages "
            "API (system must be the top-level `system` field, never a "
            "`messages` entry); folding into the leading system message "
            "before forwarding to the real backend",
        )


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
        # Separate from the TCP listener above -- a docker/podman-backed
        # harness launch runs inside the container's own network namespace,
        # where the TCP listener's host-bound address (127.0.0.1 by default)
        # is unreachable, and this host's actual container-to-host
        # networking (rootless Docker: neither the bridge gateway IP nor
        # `host.docker.internal` reached a host-bound port here) can't be
        # relied on either. A Unix domain socket sidesteps that entirely: it
        # crosses the container boundary as a bind-mounted filesystem
        # object (agsandbox's existing, universally-supported mechanism),
        # not a network hop, so it works the same regardless of the
        # container runtime's networking mode. See
        # docs/Design_harness_integration.md and the in-container relay
        # script (`agharness_internal/agproxy_ptrace_internal/
        # _tcp_to_uds_relay.py`) that bridges a container-local TCP port to
        # this socket. Started lazily, independent of `start()` -- a process
        # that never runs a container-backed harness never pays for this.
        self._uds_server = None
        self._uds_thread: "threading.Thread | None" = None
        self.uds_path: "str | None" = None
        # Every successfully-authenticated request across all three routes,
        # appended as {"route", "token", "model"} -- this is what lets a
        # test against a REAL harness binary prove its traffic actually
        # transited this gateway, rather than just checking the final
        # answer looks right (which a harness falling back to its own real
        # credentials could produce too). See test_claude_code.py's
        # `real_claude`-marked tests.
        self.request_log: "list[dict]" = []

    def _log_request(self, route: str, token: str, model: str) -> None:
        with self._lock:
            self.request_log.append({"route": route, "token": token, "model": model})

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
            self._log_request("/v1/chat/completions", token, body.get("model", ""))
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

        @app.post("/v1/messages")
        async def anthropic_messages(request: Request):
            token = _extract_bearer_token(request)
            ag = self._agent_for_token(token)
            if ag is None:
                return JSONResponse(
                    {
                        "type": "error",
                        "error": {
                            "type": "authentication_error",
                            "message": "unknown or missing bearer token",
                        },
                    },
                    status_code=401,
                )
            body = await request.json()
            _debug_capture_path = os.environ.get("AGENCY_DEBUG_CAPTURE_LOG")
            if _debug_capture_path:
                _debug_log_anthropic_messages_body(_debug_capture_path, body)
            # Route to the model THIS agent is configured for, not whatever
            # model name the harness itself happened to request -- Claude
            # Code's own default model id has no reason to match this
            # agent's configured backend/model (e.g. a Bedrock inference-
            # profile id), so trusting the harness's choice here would 400
            # against the real backend rather than actually routing through
            # agency's own configured LLM. Hit and fixed against the real
            # `claude` CLI during development.
            #
            # Deliberately does NOT fall back to body.get("model") when
            # this agent's own model is unset -- that would mean an empty
            # config silently starts trusting the harness's own guess,
            # exactly the thing this whole gateway exists to prevent (hit
            # for real: an empty agVLLMBackendConfig(model="") launch
            # forwarded Claude Code's own internal model alias to a vLLM
            # server that had no such model, 404). Native's own call sites
            # (agllm.py:541,790) never had this fallback either -- they
            # just send `backend.model or ""` unconditionally and let the
            # real server do whatever it does with an empty value (a vLLM
            # server serving exactly one model uses it regardless). This
            # route now matches that exactly, instead of being the one
            # place that second-guesses an intentionally-empty model.
            model = ag.llm.backend.model or ""
            request_id = f"msg_{uuid.uuid4().hex}"
            self._log_request("/v1/messages", token, model)
            timeout_s = _AgProxyLLMFields(self._agconfig).request_timeout_s
            client = ag.llm.backend.make_client(httpx.Timeout(timeout_s))
            _warn_mid_array_system_messages(ag, body)
            openai_kwargs = anthropic_messages_to_openai(body)
            openai_kwargs["model"] = model

            if body.get("stream"):

                def sse_gen():
                    chunks = client.chat.completions.create(**openai_kwargs)
                    for frame in openai_chunks_to_anthropic_sse(chunks, model, request_id):
                        yield frame

                return StreamingResponse(sse_gen(), media_type="text/event-stream")

            resp = client.chat.completions.create(**openai_kwargs)
            return JSONResponse(openai_response_to_anthropic_message(resp, model, request_id))

        @app.post("/agpolicy/check_tool")
        async def agpolicy_check_tool(request: Request):
            # Bridges a harness's own native permission-check mechanism
            # (e.g. Claude Code's `PreToolUse` hook, invoked as a subprocess
            # that can't reach into this process's Python state directly)
            # to `agpolicy` -- the same mediation interface `agproxy_ptrace`
            # already calls for syscall-level events, now also reachable
            # over the one channel a hook subprocess actually has: HTTP,
            # through the same per-run bearer token already used for LLM
            # traffic. See docs/Design_harness_integration.md.
            token = _extract_bearer_token(request)
            ag = self._agent_for_token(token)
            if ag is None:
                return JSONResponse(
                    {"decision": "deny", "reason": "unknown or missing bearer token"},
                    status_code=401,
                )
            body = await request.json()
            tool_name = body.get("tool_name", "")
            tool_input = body.get("tool_input") or {}

            from .. import agharness

            policy = agharness.default_policy(ag)
            decision = policy.check_tool(ag, tool_name, tool_input)
            return JSONResponse({"decision": decision.kind, "reason": decision.reason})

        @app.post("/v1/messages/count_tokens")
        async def anthropic_count_tokens(request: Request):
            # No tokenizer wired up here -- a rough heuristic (chars / 4) is
            # good enough for Claude Code's own context-usage estimates,
            # which this endpoint only feeds informationally.
            token = _extract_bearer_token(request)
            if self._agent_for_token(token) is None:
                return JSONResponse(
                    {
                        "type": "error",
                        "error": {
                            "type": "authentication_error",
                            "message": "unknown or missing bearer token",
                        },
                    },
                    status_code=401,
                )
            body = await request.json()
            approx_chars = len(str(body.get("system", ""))) + sum(
                len(str(m.get("content", ""))) for m in body.get("messages", [])
            )
            return JSONResponse({"input_tokens": max(1, approx_chars // 4)})

        @app.post("/v1/responses")
        async def openai_responses(request: Request):
            token = _extract_bearer_token(request)
            ag = self._agent_for_token(token)
            if ag is None:
                return JSONResponse(
                    {"error": {"message": "unknown or missing bearer token"}}, status_code=401
                )
            body = await request.json()
            # Same reasoning as /v1/messages above: route to this agent's
            # own configured model, not whatever Codex's own default
            # happened to request -- including when this agent's own model
            # is unset, in which case pass that through as-is (matching
            # native's `backend.model or ""`), never substitute Codex's own
            # guess.
            model = ag.llm.backend.model or ""
            request_id = f"resp_{uuid.uuid4().hex}"
            self._log_request("/v1/responses", token, model)
            timeout_s = _AgProxyLLMFields(self._agconfig).request_timeout_s
            client = ag.llm.backend.make_client(httpx.Timeout(timeout_s))
            openai_kwargs = responses_request_to_openai(body)
            openai_kwargs["model"] = model

            if body.get("stream"):

                def sse_gen():
                    chunks = client.chat.completions.create(**openai_kwargs)
                    for frame in openai_chunks_to_responses_sse(chunks, model, request_id):
                        yield frame

                return StreamingResponse(sse_gen(), media_type="text/event-stream")

            resp = client.chat.completions.create(**openai_kwargs)
            return JSONResponse(openai_response_to_responses_api(resp, model, request_id))

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
        self.stop_uds()

    def ensure_uds_started(self) -> str:
        """Start (idempotently) a second listener for the same `self._app`
        bound to a Unix domain socket instead of TCP, and return its path.
        Independent of `start()`/`base_url` -- a docker/podman-backed
        harness launch uses this path via the in-container TCP-to-UDS
        relay; a bare host-level/chroot launch never calls this at all and
        never pays for it."""
        if self.uds_path is not None:
            return self.uds_path

        import uuid

        from ..agutil import agharness_llm_gateway_dir

        sock_path = str(agharness_llm_gateway_dir() / f"agproxy_llm-{uuid.uuid4().hex}.sock")
        config = uvicorn.Config(self._app, uds=sock_path, log_level="warning")
        server = uvicorn.Server(config)
        self._uds_server = server

        self._uds_thread = threading.Thread(
            target=server.run, daemon=True, name="agproxy_llm-uds"
        )
        self._uds_thread.start()

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not server.started:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("agProxyLLM UDS server did not start within 10s")

        self.uds_path = sock_path
        return sock_path

    def stop_uds(self) -> None:
        if self._uds_server is not None:
            self._uds_server.should_exit = True
        if self._uds_thread is not None:
            self._uds_thread.join(timeout=10)
        self._uds_server = None
        self._uds_thread = None
        self.uds_path = None


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
