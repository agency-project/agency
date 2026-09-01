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
from contextlib import suppress
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

    @router.post("/agprof/hook")
    async def agprof_hook(request: Request):
        token = extract_bearer_token(request)
        if not token or not bridge.validate_token(token):
            return JSONResponse({"ok": False, "error": "unknown or missing token"}, status_code=401)
        body = await request.json()
        result = await asyncio.to_thread(
            bridge.forward_profiler_event, token, {**body, "ev": "hook"}
        )
        if result.get("error") == "unknown or missing token":
            return JSONResponse(result, status_code=401)
        return JSONResponse(result, status_code=200 if result.get("ok") else 400)

    @router.get("/agprof/status")
    async def agprof_status(request: Request):
        token = extract_bearer_token(request)
        if not token or not bridge.validate_token(token):
            return JSONResponse({"error": "unknown or missing token"}, status_code=401)
        return JSONResponse({"configured": bridge.profiler_uds_path is not None})

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
        if not token or not bridge.validate_token(token):
            return JSONResponse({"error": "unknown or missing token"}, status_code=401)
        limit = await asyncio.to_thread(bridge.context_limit, token)
        return JSONResponse({"context_limit": limit})

    @router.post("/internal/checkpoint")
    async def checkpoint(request: Request):
        body = await request.json()
        token = extract_bearer_token(request) or body.get("token")
        if not token or not bridge.validate_token(token):
            return JSONResponse({"error": "unknown or missing token"}, status_code=401)
        boundary_id = body.get("boundary_id")
        phase = body.get("phase")
        allow_steering = body.get("allow_steering")
        if (
            not isinstance(boundary_id, str)
            or not boundary_id
            or not isinstance(phase, str)
            or not phase
            or not isinstance(allow_steering, bool)
        ):
            return JSONResponse({"error": "invalid lifecycle checkpoint"}, status_code=400)
        checkpoint_task = asyncio.create_task(
            bridge.checkpoint_async(
                token,
                boundary_id,
                allow_steering=allow_steering,
                phase=phase,
            )
        )

        async def wait_for_disconnect() -> None:
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    return

        disconnect_task = asyncio.create_task(wait_for_disconnect())
        try:
            done, _pending = await asyncio.wait(
                (checkpoint_task, disconnect_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if checkpoint_task in done:
                try:
                    return JSONResponse(checkpoint_task.result())
                except Exception as exc:
                    return JSONResponse({"error": str(exc)}, status_code=502)

            checkpoint_task.cancel()
            with suppress(asyncio.CancelledError):
                await checkpoint_task
            return JSONResponse(
                {"error": "lifecycle checkpoint client disconnected"},
                status_code=499,
            )
        finally:
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task
            if not checkpoint_task.done():
                checkpoint_task.cancel()
                with suppress(asyncio.CancelledError):
                    await checkpoint_task

    return router


__all__ = ["build_router"]
