"""Harness-facing policy, profiler, and context-limit bridge routes.

None of these decide anything themselves -- each forwards to
`agmanager_host` over the bridged UDS, which is the only place `ag`/real
policy/real profiler run-context exist. They exist as LOCAL endpoints
because the callers (a subprocess-based permission hook, a future
harness-independent pause-check loop) can only reach a local
`http://127.0.0.1:<port>`, never a host-side address or a bind-mounted UDS
path directly. See `agmanager_harness.py`'s module docstring for the
full design."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .common import extract_bearer_token

if TYPE_CHECKING:
    from .clients.host_services_client import HostServicesClient


def build_router(bridge: "HostServicesClient") -> APIRouter:
    router = APIRouter()

    @router.post("/agpolicy/check_tool")
    async def agpolicy_check_tool(request: Request):
        token = extract_bearer_token(request)
        if not token or not bridge.validate_token(token):
            return JSONResponse(
                {"decision": "deny", "reason": "unknown or missing bearer token"}, status_code=401
            )
        body = await request.json()
        decision = await asyncio.to_thread(
            bridge.check_tool_policy, token, body.get("tool_name", ""), body.get("tool_input") or {}
        )
        return JSONResponse(decision)

    @router.post("/agpolicy/complete_tool")
    async def agpolicy_complete_tool(request: Request):
        token = extract_bearer_token(request)
        if not token or not bridge.validate_token(token):
            return JSONResponse({"error": "unknown or missing bearer token"}, status_code=401)
        body = await request.json()
        call_id = body.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            return JSONResponse({"error": "invalid call_id"}, status_code=400)
        await asyncio.to_thread(
            bridge.complete_tool_policy,
            token,
            call_id,
            body.get("result"),
            body.get("error"),
            **({"duration_ns": body["duration_ns"]} if "duration_ns" in body else {}),
            **({"started_wall_ns": body["started_wall_ns"]} if "started_wall_ns" in body else {}),
        )
        return JSONResponse({"ok": True})

    @router.get("/agprof/status")
    async def agprof_status(request: Request):
        token = extract_bearer_token(request)
        if not token or not bridge.validate_token(token):
            return JSONResponse({"error": "unknown or missing token"}, status_code=401)
        return JSONResponse(await asyncio.to_thread(bridge.profiler_settings, token))

    @router.post("/agprof/span")
    async def agprof_span(request: Request):
        token = extract_bearer_token(request)
        if not token or not bridge.validate_token(token):
            return JSONResponse({"ok": False, "error": "unknown or missing token"}, status_code=401)
        body = await request.json()
        return JSONResponse(await asyncio.to_thread(bridge.record_profiler_span, token, body))

    @router.post("/agprof/samples")
    async def agprof_samples(request: Request):
        token = extract_bearer_token(request)
        if not token or not bridge.validate_token(token):
            return JSONResponse({"ok": False, "error": "unknown or missing token"}, status_code=401)
        if len(await request.body()) > 1_048_576:
            return JSONResponse({"error": "sample batch too large"}, status_code=413)
        body = await request.json()
        return JSONResponse(
            await asyncio.to_thread(bridge.record_profiler_samples, token, body.get("samples", []))
        )

    # This agent's model's context window, for a caller (native_harness's
    # own compaction, see that package's `compaction.py`) that runs its own
    # ReAct loop and needs to know when to compact -- reached over this
    # bridge rather than a direct UDS connection to the terminus.
    @router.post("/internal/context_limit")
    async def context_limit(request: Request):
        body = await request.json()
        token = body.get("token")
        if not token or not bridge.validate_token(token):
            return JSONResponse({"error": "unknown or missing token"}, status_code=401)
        limit = await asyncio.to_thread(bridge.context_limit, token)
        return JSONResponse({"context_limit": limit})

    return router


__all__ = ["build_router"]
