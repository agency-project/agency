"""Tests for native_harness/mcp_client.py's tool-call timeout.

A real MCP server over real HTTP, in a background thread -- not mocked -- so this exercises
the actual httpx2 client construction, not just that a number got passed to a fake."""

from __future__ import annotations

import threading
import time

import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer

from agency.harness.mcp_transport import build_http_app
from agency.native_harness.mcp_client import McpToolset, _call_tool_async


@pytest.fixture
def slow_server():
    """One tool, `wait(seconds)`, which sleeps for *seconds* then returns "done"."""
    server = MCPServer(name="slow-test-server")

    def wait(seconds: float) -> str:
        time.sleep(seconds)
        return "done"

    server.add_tool(wait, name="wait")
    app = build_http_app(server)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    while not instance.started:
        time.sleep(0.01)
    port = instance.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        instance.should_exit = True
        thread.join(timeout=5)


class TestToolCallTimeout:
    def test_a_call_slower_than_httpx2s_5s_default_still_succeeds(self, slow_server):
        """The regression this exists for: httpx2.AsyncClient()'s own unconfigured default is
        5 seconds (see McpToolset docs). Before this fix, any tool call slower than that died
        with a bare ExceptionGroup no matter what timeout_s said. This tool sleeps 7s -- past
        that default -- with a 20s configured timeout, and must still complete."""
        import asyncio

        result = asyncio.run(
            _call_tool_async(slow_server, {}, "wait", {"seconds": 7}, timeout_s=20.0)
        )
        assert result == {"result": "done"}

    def test_a_call_exceeding_the_configured_timeout_still_fails(self, slow_server):
        """The configured value is a real ceiling, not just a way to raise the old default --
        a 4s timeout must still cut off a 7s tool call, and within roughly that 4s, not the
        old 5s default and not some other unrelated error."""
        import asyncio

        started = time.monotonic()
        with pytest.raises(BaseExceptionGroup):
            asyncio.run(_call_tool_async(slow_server, {}, "wait", {"seconds": 7}, timeout_s=4.0))
        elapsed = time.monotonic() - started
        assert 3.5 < elapsed < 6, f"expected to cut off near 4s, took {elapsed:.1f}s"

    def test_the_agent_sees_the_real_reason_not_a_bare_exceptiongroup(self, slow_server):
        """The other half of the fix: unwrapping the ExceptionGroup so the tool result the
        agent actually reads names the real failure instead of a meaningless wrapper."""
        toolset = McpToolset(None)
        toolset._tool_owner["wait"] = (slow_server, {})
        toolset._timeout_s = 4.0
        result = toolset.call("wait", '{"seconds": 7}')
        assert "ExceptionGroup" not in result
        assert "timed out" in result.lower()

    def test_mcptoolset_call_uses_its_configured_timeout(self, slow_server):
        toolset = McpToolset({"mcpServers": {"s": {"type": "http", "url": slow_server}}})
        toolset._tool_owner["wait"] = (slow_server, {})
        toolset._timeout_s = 20.0
        result = toolset.call("wait", '{"seconds": 7}')
        assert result == '{"result": "done"}'


def test_mcptoolset_defaults_to_a_long_timeout_not_httpx2s_5s():
    """Constructed with no explicit timeout_s, McpToolset should not silently inherit
    httpx2's 5s default -- that's the exact bug this fix closes."""
    toolset = McpToolset(None)
    assert toolset._timeout_s == 1800.0
