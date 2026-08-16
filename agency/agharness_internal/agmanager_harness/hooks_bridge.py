"""Policy-check, profiler-hook, and pause/inbox-check-in bridge routes for
`agmanager_harness`.

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
    from .host_bridge import _HostBridge


def build_router(bridge: "_HostBridge") -> APIRouter:
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

    @router.post("/agprof/hook")
    async def agprof_hook(request: Request):
        token = extract_bearer_token(request)
        if not token:
            return JSONResponse({"ok": False, "error": "unknown or missing token"}, status_code=401)
        body = await request.json()
        result = await asyncio.to_thread(
            bridge.forward_profiler_event, token, {**body, "ev": "hook"}
        )
        if result.get("error") == "unknown or missing token":
            return JSONResponse(result, status_code=401)
        return JSONResponse(result, status_code=200 if result.get("ok") else 400)

    @router.get("/agprof/status")
    async def agprof_status():
        return JSONResponse({"configured": bridge.profiler_uds_path is not None})

    # The endpoint a harness-independent loop polls before every turn (the
    # standalone `native_harness` package's own react loop, see that
    # package's `bridge_client.py`); forwards to agmanager_host, which is
    # the only place ag._check_pause()/ag._drain_inbox() exist.
    @router.post("/internal/check_in")
    async def check_in(request: Request):
        body = await request.json()
        token = body.get("token")
        if not token:
            return JSONResponse({"error": "missing token"}, status_code=401)
        messages = await asyncio.to_thread(bridge.check_in, token)
        return JSONResponse({"messages": messages})

    # This agent's model's context window, for a caller (native_harness's
    # own compaction, see that package's `compaction.py`) that runs its own
    # ReAct loop and needs to know when to compact -- mirrors the old
    # `_native_in_container_entrypoint.py`'s `_fetch_context_limit`, just
    # reached over this bridge instead of a direct UDS connection to the
    # terminus.
    @router.post("/internal/context_limit")
    async def context_limit(request: Request):
        body = await request.json()
        token = body.get("token")
        if not token:
            return JSONResponse({"error": "missing token"}, status_code=401)
        limit = await asyncio.to_thread(bridge.context_limit, token)
        return JSONResponse({"context_limit": limit})

    return router


__all__ = ["build_router"]
