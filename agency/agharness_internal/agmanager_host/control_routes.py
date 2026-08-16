"""General per-agent control-plane routes for `agmanager_host`: token
validation, warning logging, tool-policy checks, and pause/inbox check-in.

None of these are LLM- or MCP-specific -- they exist purely because a
container-side caller (`agmanager_harness`) has no live `ag` object of its
own. See `agmanager_host.py`'s module docstring for the full design."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from ...agent import agent
    from .launch_state import LaunchRegistry


class _CheckInRequest(BaseModel):
    token: str


def build_router(ag: "agent", registry: "LaunchRegistry") -> APIRouter:
    router = APIRouter()

    @router.post("/internal/validate_token")
    async def validate_token(request: Request):
        body = await request.json()
        return JSONResponse({"valid": registry.get(body.get("token")) is not None})

    @router.post("/internal/log_warning")
    async def log_warning(request: Request):
        body = await request.json()
        if registry.get(body.get("token")) is None:
            return JSONResponse({"error": "unknown or missing token"}, status_code=401)
        ag.terminal.log("WARNING  ", body.get("message", ""))
        return JSONResponse({"ok": True})

    @router.post("/internal/check_tool_policy")
    async def check_tool_policy(request: Request):
        body = await request.json()
        if registry.get(body.get("token")) is None:
            return JSONResponse(
                {"decision": "deny", "reason": "unknown or missing token"}, status_code=401
            )
        from ... import agharness

        policy = agharness.default_policy(ag)
        decision = policy.check_tool(ag, body.get("tool_name", ""), body.get("tool_input") or {})
        return JSONResponse({"decision": decision.kind, "reason": decision.reason})

    # Same reasoning as the old agharness_messenger.py: a plain `def`, not
    # `async def` -- `ag._check_pause()` can block for as long as the agent
    # stays paused, and FastAPI/Starlette runs a sync route in its worker
    # thread pool rather than the single event loop, so this doesn't stall
    # this agent's own concurrent dispatch/tool routes. The caller is
    # agmanager_harness's container-side check-in endpoint, forwarding here
    # since `ag._check_pause()`/`ag._drain_inbox()` only exist on this
    # live, host-only object.
    @router.post("/internal/check_in")
    def check_in(payload: _CheckInRequest):
        if registry.get(payload.token) is None:
            return JSONResponse({"error": "unknown or missing token"}, status_code=401)
        ag._check_pause()
        drained: "list[dict]" = []
        ag._drain_inbox(drained)
        return JSONResponse({"messages": drained})

    return router


__all__ = ["build_router"]
