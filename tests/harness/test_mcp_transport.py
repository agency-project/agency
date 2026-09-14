"""Retained CLIs must not pin attempt leases with optional MCP event streams."""

import asyncio
import base64
from types import SimpleNamespace

import cloudpickle
import httpx
import pytest

from agency.agdata import agdata
from agency.agtool import agtool
from agency.engine.host_servers.host_mcp_server import HostMcpServer
from agency.harness.sandbox_mcp import build_app


def make_app(surface, live):
    if surface == "host":
        return HostMcpServer(
            None, SimpleNamespace(host_mcp_tools=[]), None, None, live_session=live
        ).build_app()
    tool = agtool("probe", "probe", lambda data: agdata(reply="ok"))
    return build_app(
        base64.b64encode(cloudpickle.dumps([tool])).decode(), lambda: True, live_session=live
    )


@pytest.mark.parametrize("surface", ["host", "sandbox"])
@pytest.mark.parametrize("live", [False, True])
def test_mcp_live_sessions_decline_event_streams_but_accept_requests(surface, live):
    async def check():
        app = make_app(surface, live)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                if live:
                    response = await asyncio.wait_for(
                        client.get("/mcp", headers={"accept": "text/event-stream"}), 2
                    )
                    assert response.status_code == 405
                    assert response.headers["allow"] == "POST"
                response = await client.post(
                    "/mcp",
                    headers={"accept": "application/json, text/event-stream"},
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-03-26",
                            "capabilities": {},
                            "clientInfo": {"name": "checkpoint-test", "version": "1"},
                        },
                    },
                )
                assert response.status_code == 200
                assert bool(response.headers.get("mcp-session-id")) is (not live)
                if live:
                    response = await client.post(
                        "/mcp",
                        headers={"accept": "application/json, text/event-stream"},
                        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                    )
                    assert response.status_code == 200
                    assert "tools" in response.json()["result"]

    asyncio.run(check())
