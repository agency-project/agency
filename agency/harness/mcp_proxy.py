"""Generic reverse proxy for the MCP tool surface, for `agmanager_harness`.

Forwards to `agmanager_host`'s own `/mcp` mount over the bridged UDS using
an ordinary HTTP reverse-proxy route on this same process. See
`agmanager_harness.py`'s module docstring for the full design.

Responses are streamed incrementally.  In particular, streamable HTTP may
keep a GET open for the lifetime of an MCP client; buffering that response
would prevent the sandbox-side request from ever completing and would leave
the host attempt lease held during attempt cleanup.

Response headers, not just the body, must be forwarded back -- confirmed
the hard way (a real end-to-end run against `native_harness`'s MCP client):
the streamable-HTTP protocol returns a fresh `Mcp-Session-Id` header on
`initialize`, which every subsequent request in that same client session
(`list_tools`, `call_tool`) must echo back. Dropping it (an earlier version
of this proxy returned only `media_type` from the response, silently
discarding every other header) breaks a session after its first request:
the second request arrives at agmanager_host's real MCP server with no
session id at all and gets rejected with 400, which nothing in this proxy
surfaces as an error -- it just looks like the tool call failed."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .common import extract_bearer_token

if TYPE_CHECKING:
    from .clients.host_services_client import HostServicesClient


class _ClosingStreamingResponse(StreamingResponse):
    """Close the upstream HTTP context on completion or client disconnect."""

    def __init__(self, *args, upstream_stack: AsyncExitStack, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._upstream_stack = upstream_stack

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._upstream_stack.aclose()


def build_router(bridge: "HostServicesClient") -> APIRouter:
    router = APIRouter()

    # Exact path, not a wildcard sub-path: agmanager_host's own MCP mount
    # (`streamable_http_app()`) defines exactly one route, "/mcp" itself --
    # session/state management happens via headers (`Mcp-Session-Id`), not
    # sub-paths, so there is nothing under "/mcp/*" to proxy.
    @router.api_route("/mcp", methods=["GET", "POST", "DELETE"])
    async def mcp_proxy(request: Request):
        token = extract_bearer_token(request)
        if not token or not bridge.validate_token(token):
            return JSONResponse(
                {"error": "unknown or missing bearer token"},
                status_code=401,
            )
        body = await request.body()
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")
        }

        upstream_stack = AsyncExitStack()
        try:
            resp = await upstream_stack.enter_async_context(
                bridge.forward_mcp_request(
                    token,
                    request.method,
                    content=body,
                    headers=headers,
                    params=dict(request.query_params),
                )
            )
        except BaseException:
            await upstream_stack.aclose()
            raise
        # Forward every response header EXCEPT the hop-by-hop ones Starlette
        # recomputes for its own response (content-length/transfer-encoding
        # depend on how *this* proxy resends the body, not the upstream
        # response; connection is a raw-socket concern, not meaningful to
        # re-send at all) -- most importantly `Mcp-Session-Id`, see module
        # docstring for why dropping it breaks every request after the
        # first in an MCP client session.
        response_headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "connection")
        }
        return _ClosingStreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=response_headers,
            media_type=resp.headers.get("content-type"),
            upstream_stack=upstream_stack,
        )

    return router


__all__ = ["build_router"]
