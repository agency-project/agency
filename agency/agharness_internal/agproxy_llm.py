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

import asyncio
import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from ..agconfig import GlobalConfigParam, DynamicConfigParam, _AgConfigViewBase
from .agllm_terminus import agLLMTerminus, get_shared_terminus
from .agproxy_llm_adapters import (
    UnsupportedResponsesRequest,
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

_MAX_PROFILER_HOOK_BODY_BYTES = 256 * 1024
_MAX_CONCURRENT_PROFILER_FORWARDS = 8
_PROFILER_FORWARD_TIMEOUT_S = 0.25
_MAX_PROFILER_HOOK_EVENTS_PER_TOKEN_PER_SECOND = 128
_MAX_PROFILER_RATE_TOKENS = 1024

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


_DEBUG_CAPTURE_MARKERS = (
    "Available agent types",
    "task tools haven't been used",
    "gentle reminder",
)


def _debug_log_anthropic_messages_body(path: str, body: dict) -> None:
    """Temporary diagnostic: dump raw incoming /v1/messages shape (roles,
    mid-array system messages, reminder markers) to `path` for direct
    inspection of what the harness actually sent, before any adapter
    transformation. Opt-in only, via AGENCY_DEBUG_CAPTURE_LOG."""
    messages = body.get("messages", [])
    roles = [m.get("role") for m in messages]
    mid_array_system = "system" in roles[1:] if roles else False
    lines = [
        f"\n=== num_messages={len(roles)} roles={roles} "
        f"{'!!! MID-ARRAY SYSTEM !!!' if mid_array_system else ''}"
    ]
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
            lines.append(
                f"    >>> MID-ARRAY SYSTEM MESSAGE at index {i} ({len(text)} chars): {text[:300]!r}"
            )
            for marker in _DEBUG_CAPTURE_MARKERS:
                if marker in text:
                    lines.append(f"        >>> contains marker: {marker!r}")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def _warn_mid_array_system_messages(log_fn: "Callable[[str], None]", body: dict) -> None:
    """Claude Code's generic (ANTHROPIC_BASE_URL) client sometimes emits its
    own dynamic reminders (e.g. an Agent-tool-availability nudge) as a
    `role: "system"` entry inside `messages`, not the top-level `system`
    field -- a shape the real Anthropic Messages API and AWS Bedrock's
    Anthropic-invoke endpoint both reject outright (confirmed directly
    against both). `anthropic_messages_to_openai` below folds every such
    occurrence into the one leading system message so it never reaches a
    real backend, but that's a silent correctness workaround for what looks
    like an inconsistency in Claude Code's own request serialization on
    this client path -- surface it instead of absorbing it invisibly.

    Takes a `log_fn` callable rather than a live `ag` object -- routed
    through `agProxyLLM._log_warning()` (itself forwarding to
    `agllm_terminus`'s `/internal/log_warning`), not `ag.terminal.log`
    directly, since `ag` isn't reachable from wherever this routing/
    translation layer eventually runs (see agllm_terminus.py)."""
    n = sum(1 for m in body.get("messages", []) if m.get("role") == "system")
    if n:
        log_fn(
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
    `unregister()` once that subprocess exits.

    Routing/translation only -- real backend credentials are never touched
    here, and neither is the token<->agent registry: unlike an earlier
    version of this class, there is no local `_agents_by_token` dict at
    all. Every route's auth check and every piece of per-agent state
    (model, policy, logging, credentialed dispatch) goes through
    `agllm_terminus.agLLMTerminus` over real HTTP -- constructed either
    from a live `agLLMTerminus` object (`terminus=`, when both live in the
    same host process) or a bind-mounted UDS path (`terminus_uds_path=`,
    when this class itself runs inside the sandbox container and the
    terminus is a separate host-side process reachable only over that
    socket). Removing the local registry is what makes those two
    constructions genuinely interchangeable: a same-process Python dict
    lookup is simply impossible once this class runs in a different
    process from whatever called `register()`, so EVERY request always
    asks the terminus, regardless of where this instance happens to run."""

    def __init__(
        self,
        agconfig: "agConfig | None" = None,
        terminus: "agLLMTerminus | None" = None,
        terminus_uds_path: "str | None" = None,
        profiler_uds_path: "str | None" = None,
    ) -> None:
        self._agconfig = agconfig
        self._lock = threading.Lock()
        # terminus_uds_path takes precedence when both are somehow given --
        # it signals "this instance runs somewhere `terminus` (a live
        # Python object) cannot be shared to," which get_shared_terminus()'s
        # default would silently violate by spinning up a SEPARATE terminus
        # in this process instead of reaching the real one.
        if terminus_uds_path is not None:
            self._terminus: "agLLMTerminus | None" = None
        else:
            self._terminus = terminus if terminus is not None else get_shared_terminus(agconfig)
        self._terminus_uds_path = terminus_uds_path
        self._terminus_client: "httpx.Client | None" = None
        # Profiler traffic deliberately bypasses the LLM terminus.  The
        # gateway is only a container-reachable HTTP bridge; it forwards
        # each authenticated hook event to agProfilerIngest's independent
        # framed-JSON UDS so telemetry cannot contend with dispatch/TTFT on
        # the terminus event loop (Design_profiler_harness_integration §5.5).
        if profiler_uds_path is not None and terminus_uds_path is not None:
            profiler_path = Path(profiler_uds_path)
            if (
                profiler_path.parent != Path("/var/run/agency_llm_gateway")
                or not profiler_path.name.startswith("agprof-ingest-")
                or profiler_path.suffix != ".sock"
            ):
                raise ValueError("invalid in-container profiler UDS bridge path")
        self._profiler_uds_path = profiler_uds_path
        self._profiler_sync_lock = threading.Lock()
        # asyncio.to_thread() bounds workers but not its pending-work queue.
        # Admit profiler work before token validation and UDS forwarding so
        # a valid-token request loop cannot enqueue unbounded blocking jobs.
        self._profiler_forward_slots = threading.BoundedSemaphore(_MAX_CONCURRENT_PROFILER_FORWARDS)
        self._profiler_rate_lock = threading.Lock()
        self._profiler_rate_windows: "dict[str, tuple[float, int]]" = {}
        # In-container gateways are long-lived while launch tokens are
        # unique, so keep this cache explicitly bounded. Re-syncing an
        # evicted active token is harmless.
        self._profiler_synced_tokens: dict[str, None] = {}
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
        # Survives stop_uds() (which clears uds_path) so a restart rebinds the
        # SAME path -- see agutil.reserve_uds_path.
        self._uds_reserved_path: "str | None" = None
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

    # -- token <-> agent registry -- lives entirely on the terminus now ----

    def register(self, token: str, ag: "agent") -> None:
        # A no-op when constructed with terminus_uds_path= (this instance
        # runs somewhere `ag` -- a live Python object -- can't be shared
        # to): nothing calls register() on THIS instance in that case
        # anyway, since the caller registers directly on the real
        # host-side terminus before ever launching the process this
        # instance runs inside. See claude_code.py's execute() for the
        # concrete split.
        if self._terminus is not None:
            self._terminus.register(token, ag)

    def unregister(self, token: str) -> None:
        with self._profiler_sync_lock:
            self._profiler_synced_tokens.pop(token, None)
        with self._profiler_rate_lock:
            self._profiler_rate_windows.pop(token, None)
        if self._terminus is not None:
            self._terminus.unregister(token)

    def _token_valid(self, token: "str | None") -> bool:
        if token is None:
            return False
        client = self._terminus_http_client()
        resp = client.post("/internal/validate_token", json={"token": token})
        if resp.status_code != 200:
            return False
        return bool(resp.json().get("valid"))

    # -- dispatch, via the terminus, never in-process ----------------------

    def _terminus_http_client(self) -> httpx.Client:
        if self._terminus_client is None:
            timeout_s = _AgProxyLLMFields(self._agconfig).request_timeout_s
            if self._terminus_uds_path is not None:
                transport = httpx.HTTPTransport(uds=self._terminus_uds_path)
                self._terminus_client = httpx.Client(
                    transport=transport, base_url="http://agllm-terminus", timeout=timeout_s
                )
            else:
                base_url = self._terminus.start()
                self._terminus_client = httpx.Client(base_url=base_url, timeout=timeout_s)
        return self._terminus_client

    def _dispatch(self, token: "str | None", kwargs: dict):
        """POST `kwargs` (already-uniform chat-completions arguments) to the
        terminus's `/internal/dispatch`, keyed by `token`, and reconstruct
        real `ChatCompletion`/`ChatCompletionChunk` SDK objects from its
        response -- so `agproxy_llm_adapters.py`'s contract (attribute
        access like `resp.choices[0]`, not dict indexing) is unaffected by
        the fact that the real backend call now happens in a different
        process. Non-streaming: returns a `ChatCompletion`. Streaming:
        returns a generator of `ChatCompletionChunk`."""
        client = self._terminus_http_client()

        if kwargs.get("stream"):

            def gen():
                with client.stream(
                    "POST", "/internal/dispatch", json={"token": token, "kwargs": kwargs}
                ) as resp:
                    if resp.status_code != 200:
                        resp.read()
                        raise RuntimeError(
                            f"terminus dispatch failed: {resp.status_code} {resp.text}"
                        )
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        payload = line[len("data: ") :]
                        if payload == "[DONE]":
                            return
                        yield ChatCompletionChunk.model_validate(json.loads(payload))

            return gen()

        resp = client.post("/internal/dispatch", json={"token": token, "kwargs": kwargs})
        if resp.status_code != 200:
            raise RuntimeError(f"terminus dispatch failed: {resp.status_code} {resp.text}")
        return ChatCompletion.model_validate(resp.json())

    # -- everything else that needs the real `ag` object, routed through the
    # terminus rather than this class's own registry -- the prerequisite for
    # this routing/translation layer to run somewhere `ag` isn't reachable
    # at all (e.g. inside the sandbox container). See agllm_terminus.py's
    # resolve_model/log_warning/check_tool_policy routes.

    def _resolve_model(self, token: "str | None") -> str:
        client = self._terminus_http_client()
        resp = client.post("/internal/resolve_model", json={"token": token})
        if resp.status_code != 200:
            raise RuntimeError(f"terminus resolve_model failed: {resp.status_code} {resp.text}")
        return resp.json()["model"]

    def _log_warning(self, token: "str | None", message: str) -> None:
        client = self._terminus_http_client()
        client.post("/internal/log_warning", json={"token": token, "message": message})

    def _check_tool_policy(self, token: "str | None", tool_name: str, tool_input: dict) -> dict:
        client = self._terminus_http_client()
        resp = client.post(
            "/internal/check_tool_policy",
            json={"token": token, "tool_name": tool_name, "tool_input": tool_input},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"terminus check_tool_policy failed: {resp.status_code} {resp.text}")
        return resp.json()

    def _profiler_socket_path(self) -> str:
        if self._profiler_uds_path is not None:
            return self._profiler_uds_path
        if self._terminus_uds_path is not None:
            raise RuntimeError("in-container proxy has no profiler UDS bridge")
        # Host-resident gateways can resolve the host service lazily.
        from .agprof_ingest import get_shared_profiler_ingest

        self._profiler_uds_path = get_shared_profiler_ingest().ensure_uds_started()
        return self._profiler_uds_path

    def _forward_profiler_hook(self, token: str, event: dict) -> dict:
        """Send one hook event to agProfilerIngest's separate UDS.

        The HTTP bearer token is injected only into the private framed
        envelope used for host correlation; it is never accepted from the
        request body or copied into span metadata.
        """
        from ..profiler.agprof_emit import _recv_framed, _send_framed

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(_PROFILER_FORWARD_TIMEOUT_S)
        try:
            sock.connect(self._profiler_socket_path())
            # The hook and this gateway share one container/host clock
            # domain. Synchronize that domain once per launch token before
            # forwarding the hook's captured timestamp.
            with self._profiler_sync_lock:
                if token not in self._profiler_synced_tokens:
                    wall_0 = time.time_ns()
                    perf_0 = time.perf_counter_ns()
                    _send_framed(
                        sock,
                        {
                            "token": token,
                            "ev": "clock_sync",
                            "wall_ns": wall_0,
                            "perf_ns": perf_0,
                        },
                    )
                    sync = _recv_framed(sock)
                    wall_1 = time.time_ns()
                    perf_1 = time.perf_counter_ns()
                    if not sync.get("ok"):
                        return sync
                    _send_framed(
                        sock,
                        {
                            "token": token,
                            "ev": "clock_offset",
                            "wall_offset_ns": int(sync["host_wall_ns"] - (wall_0 + wall_1) / 2),
                            "perf_offset_ns": int(sync["host_perf_ns"] - (perf_0 + perf_1) / 2),
                        },
                    )
                    offset = _recv_framed(sock)
                    if not offset.get("ok"):
                        return offset
                    if len(self._profiler_synced_tokens) >= 1024:
                        self._profiler_synced_tokens.pop(next(iter(self._profiler_synced_tokens)))
                    self._profiler_synced_tokens[token] = None

            _send_framed(sock, {**event, "token": token, "ev": "hook"})
            return _recv_framed(sock)
        finally:
            sock.close()

    def _profiler_hook_rate_allowed(self, token: str, *, now: "float | None" = None) -> bool:
        """Apply a bounded, per-launch fixed-window profiler event limit."""
        if now is None:
            now = time.monotonic()
        with self._profiler_rate_lock:
            window_start, count = self._profiler_rate_windows.get(token, (now, 0))
            if now - window_start >= 1.0 or now < window_start:
                window_start, count = now, 0
            if count >= _MAX_PROFILER_HOOK_EVENTS_PER_TOKEN_PER_SECOND:
                return False
            if token not in self._profiler_rate_windows and (
                len(self._profiler_rate_windows) >= _MAX_PROFILER_RATE_TOKENS
            ):
                self._profiler_rate_windows.pop(next(iter(self._profiler_rate_windows)))
            self._profiler_rate_windows[token] = (window_start, count + 1)
            return True

    # -- app / routes -----------------------------------------------------

    def _build_app(self):
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            token = _extract_bearer_token(request)
            if not self._token_valid(token):
                return JSONResponse(
                    {"error": {"message": "unknown or missing bearer token"}}, status_code=401
                )
            body = await request.json()
            self._log_request("/v1/chat/completions", token, body.get("model", ""))
            if body.get("stream"):

                def sse_gen():
                    for chunk in self._dispatch(token, body):
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(sse_gen(), media_type="text/event-stream")
            result = self._dispatch(token, body)
            return JSONResponse(result.model_dump())

        @app.post("/v1/messages")
        async def anthropic_messages(request: Request):
            token = _extract_bearer_token(request)
            if not self._token_valid(token):
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
            model = self._resolve_model(token)
            request_id = f"msg_{uuid.uuid4().hex}"
            self._log_request("/v1/messages", token, model)
            _warn_mid_array_system_messages(lambda msg: self._log_warning(token, msg), body)
            openai_kwargs = anthropic_messages_to_openai(body)
            openai_kwargs["model"] = model

            if body.get("stream"):

                def sse_gen():
                    chunks = self._dispatch(token, openai_kwargs)
                    for frame in openai_chunks_to_anthropic_sse(chunks, model, request_id):
                        yield frame

                return StreamingResponse(sse_gen(), media_type="text/event-stream")

            resp = self._dispatch(token, openai_kwargs)
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
            if not self._token_valid(token):
                return JSONResponse(
                    {"decision": "deny", "reason": "unknown or missing bearer token"},
                    status_code=401,
                )
            body = await request.json()
            tool_name = body.get("tool_name", "")
            tool_input = body.get("tool_input") or {}
            decision = self._check_tool_policy(token, tool_name, tool_input)
            return JSONResponse(decision)

        @app.post("/agprof/hook")
        async def agprof_hook(request: Request):
            # Claude's hook subprocess can reach this gateway's local HTTP
            # port from inside the sandbox, but not a host-only UDS by URL.
            # Hop directly to agProfilerIngest's separate framed listener,
            # which owns the authoritative token registration and rejects an
            # unknown bearer.  Do not first round-trip through the LLM
            # terminus merely to validate the same token: that redundant hop
            # pushed concurrent PostToolUse hooks past their tight telemetry
            # deadline.  Body/rate/slot bounds still apply before forwarding.
            # The blocking UDS round trip runs in a worker so it cannot stall
            # this gateway's async LLM routes.
            slot_acquired = self._profiler_forward_slots.acquire(blocking=False)
            try:
                if not slot_acquired:
                    return JSONResponse(
                        {"ok": False, "error": "profiler hook bridge saturated"},
                        status_code=429,
                    )
                token = _extract_bearer_token(request)
                if not token:
                    return JSONResponse({"ok": False, "error": "unknown or missing token"}, 401)
                if not self._profiler_hook_rate_allowed(token):
                    return JSONResponse(
                        {"ok": False, "error": "profiler hook rate limit exceeded"},
                        status_code=429,
                    )
                body = bytearray()
                async for chunk in request.stream():
                    if len(body) + len(chunk) > _MAX_PROFILER_HOOK_BODY_BYTES:
                        return JSONResponse(
                            {"ok": False, "error": "profiler hook body too large"},
                            status_code=413,
                        )
                    body.extend(chunk)
                try:
                    event = json.loads(body)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return JSONResponse(
                        {"ok": False, "error": "invalid profiler hook JSON"},
                        status_code=400,
                    )
                if not isinstance(event, dict):
                    return JSONResponse(
                        {"ok": False, "error": "invalid profiler hook event"},
                        status_code=400,
                    )
                try:
                    result = await asyncio.to_thread(self._forward_profiler_hook, token, event)
                except Exception as exc:
                    return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)
                if result.get("error") == "unknown or missing token":
                    return JSONResponse(result, status_code=401)
                return JSONResponse(result, status_code=200 if result.get("ok") else 400)
            finally:
                if slot_acquired:
                    self._profiler_forward_slots.release()

        @app.get("/agprof/status")
        async def agprof_status():
            return JSONResponse({"configured": self._profiler_uds_path is not None})

        @app.post("/v1/messages/count_tokens")
        async def anthropic_count_tokens(request: Request):
            # No tokenizer wired up here -- a rough heuristic (chars / 4) is
            # good enough for Claude Code's own context-usage estimates,
            # which this endpoint only feeds informationally.
            token = _extract_bearer_token(request)
            if not self._token_valid(token):
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
            if not self._token_valid(token):
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
            model = self._resolve_model(token)
            request_id = f"resp_{uuid.uuid4().hex}"
            self._log_request("/v1/responses", token, model)
            try:
                openai_kwargs = responses_request_to_openai(
                    body, warning_handler=lambda message: self._log_warning(token, message)
                )
            except UnsupportedResponsesRequest as exc:
                return JSONResponse(
                    {
                        "error": {
                            "message": str(exc),
                            "type": "invalid_request_error",
                            "code": "unsupported_responses_translation",
                        }
                    },
                    status_code=400,
                )
            openai_kwargs["model"] = model

            if body.get("stream"):

                def sse_gen():
                    chunks = self._dispatch(token, openai_kwargs)
                    for frame in openai_chunks_to_responses_sse(chunks, model, request_id):
                        yield frame

                return StreamingResponse(sse_gen(), media_type="text/event-stream")

            resp = self._dispatch(token, openai_kwargs)
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
        # Only closes this instance's own client to the terminus -- the
        # terminus itself is a separate, possibly-shared object (see
        # get_shared_terminus) and is never stopped as a side effect of
        # stopping this routing layer.
        if self._terminus_client is not None:
            self._terminus_client.close()
            self._terminus_client = None

    def ensure_uds_started(self) -> str:
        """Start (idempotently) a second listener for the same `self._app`
        bound to a Unix domain socket instead of TCP, and return its path.
        Independent of `start()`/`base_url` -- a docker/podman-backed
        harness launch uses this path via the in-container TCP-to-UDS
        relay; a bare host-level/chroot launch never calls this at all and
        never pays for it."""
        from ..agutil import reserve_uds_path, uds_listener_is_live

        # "Idempotently" must mean *still working*, not merely *started once*:
        # an external cleanup can delete a live socket file (a socket's mtime
        # never updates, so age-based reapers see every long-lived one as
        # stale) and a dead server thread takes its socket with it, since
        # uvicorn unlinks on shutdown. Either leaves this method handing out a
        # path nothing listens on. Rebuild at the same reserved path instead.
        if self.uds_path is not None:
            if uds_listener_is_live(self.uds_path, self._uds_thread):
                return self.uds_path
            self.stop_uds()

        sock_path = reserve_uds_path(self._uds_reserved_path, "agproxy_llm")
        self._uds_reserved_path = sock_path
        config = uvicorn.Config(self._app, uds=sock_path, log_level="warning")
        server = uvicorn.Server(config)
        self._uds_server = server

        self._uds_thread = threading.Thread(target=server.run, daemon=True, name="agproxy_llm-uds")
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
