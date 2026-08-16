"""LLM wire-format translation routes for `agmanager_harness`.

`/v1/chat/completions` passthrough (opencode/Grok, already chat-completions-
shaped), `/v1/messages` (Anthropic Messages, Claude Code), `/v1/responses`
(OpenAI Responses, Codex), `/v1/messages/count_tokens` -- replaces the old
`agproxy_llm.py`'s three-route translation layer (and the separate
`agproxy_llm_in_container.py` launcher that stood up a second copy of that
SAME class inside the container for docker/podman-backed launches -- this
design doesn't need that split at all, since the one process here already
runs in-container unconditionally). See `agmanager_harness.py`'s module
docstring for the full design and its reuse policy for
`agproxy_llm_adapters.py`."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..agproxy_llm_adapters import (
    anthropic_messages_to_openai,
    openai_response_to_anthropic_message,
    openai_chunks_to_anthropic_sse,
    responses_request_to_openai,
    openai_response_to_responses_api,
    openai_chunks_to_responses_sse,
)
from .common import extract_bearer_token

if TYPE_CHECKING:
    from .host_bridge import _HostBridge


def _mid_array_system_warning(body: dict) -> "str | None":
    """Same check the old `agproxy_llm.py`'s `_warn_mid_array_system_messages`
    did -- duplicated (short enough) rather than imported, see
    `agmanager_harness.py`'s module docstring's reuse policy. Claude
    Code's generic ANTHROPIC_BASE_URL client sometimes emits its own
    dynamic reminders as a `role: "system"` entry inside `messages` rather
    than the top-level `system` field, a shape the real Anthropic Messages
    API rejects outright; `anthropic_messages_to_openai` folds it away, but
    that's a silent correctness workaround worth surfacing rather than
    absorbing invisibly."""
    n = sum(1 for m in body.get("messages", []) if m.get("role") == "system")
    if not n:
        return None
    return (
        f"harness emitted {n} mid-conversation system-role message(s) in its "
        "/v1/messages request -- not valid per the Anthropic Messages API "
        "(system must be the top-level `system` field, never a `messages` "
        "entry); folding into the leading system message before forwarding "
        "to the real backend"
    )


def build_router(bridge: "_HostBridge") -> APIRouter:
    import asyncio

    router = APIRouter()

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        token = extract_bearer_token(request)
        if not token or not bridge.validate_token(token):
            return JSONResponse(
                {"error": {"message": "unknown or missing bearer token"}}, status_code=401
            )
        body = await request.json()
        if body.get("stream"):

            def sse_gen():
                for chunk in bridge.dispatch(token, body):
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse_gen(), media_type="text/event-stream")
        result = await asyncio.to_thread(bridge.dispatch, token, body)
        return JSONResponse(result.model_dump())

    @router.post("/v1/messages")
    async def anthropic_messages(request: Request):
        token = extract_bearer_token(request)
        if not token or not bridge.validate_token(token):
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
        # Route to THIS agent's own configured model, never whatever
        # default Claude Code itself requested -- same reasoning as the
        # old agproxy_llm.py's identical routing rule.
        model = bridge.resolve_model(token)
        request_id = f"msg_{uuid.uuid4().hex}"
        warning = _mid_array_system_warning(body)
        if warning:
            bridge.log_warning(token, warning)
        openai_kwargs = anthropic_messages_to_openai(body)
        openai_kwargs["model"] = model

        if body.get("stream"):

            def sse_gen():
                chunks = bridge.dispatch(token, openai_kwargs)
                for frame in openai_chunks_to_anthropic_sse(chunks, model, request_id):
                    yield frame

            return StreamingResponse(sse_gen(), media_type="text/event-stream")

        resp = await asyncio.to_thread(bridge.dispatch, token, openai_kwargs)
        return JSONResponse(openai_response_to_anthropic_message(resp, model, request_id))

    @router.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request):
        token = extract_bearer_token(request)
        if not token or not bridge.validate_token(token):
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
        # No tokenizer wired up -- a chars/4 heuristic is good enough for a
        # harness's own informational context-usage estimate.
        approx_chars = len(str(body.get("system", ""))) + sum(
            len(str(m.get("content", ""))) for m in body.get("messages", [])
        )
        return JSONResponse({"input_tokens": max(1, approx_chars // 4)})

    @router.post("/v1/responses")
    async def openai_responses(request: Request):
        token = extract_bearer_token(request)
        if not token or not bridge.validate_token(token):
            return JSONResponse(
                {"error": {"message": "unknown or missing bearer token"}}, status_code=401
            )
        body = await request.json()
        model = bridge.resolve_model(token)
        request_id = f"resp_{uuid.uuid4().hex}"
        openai_kwargs = responses_request_to_openai(body)
        openai_kwargs["model"] = model

        if body.get("stream"):

            def sse_gen():
                chunks = bridge.dispatch(token, openai_kwargs)
                for frame in openai_chunks_to_responses_sse(chunks, model, request_id):
                    yield frame

            return StreamingResponse(sse_gen(), media_type="text/event-stream")

        resp = await asyncio.to_thread(bridge.dispatch, token, openai_kwargs)
        return JSONResponse(openai_response_to_responses_api(resp, model, request_id))

    return router


__all__ = ["build_router"]
