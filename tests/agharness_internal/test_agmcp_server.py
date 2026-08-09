"""Tests for agmcp_server -- the shared host-side MCP server exposing
resource-control and output-submission tools identically to every engine
(native and harness-driven alike). Real MCP-client-over-the-wire tests,
not just calling the Python methods directly -- the whole point of this
server is that Claude Code/Codex/opencode/Grok's OWN native MCP clients
must be able to discover and call these tools, so at least one test here
must prove that with an actual MCP client, not a shortcut.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock


def _make_session(sandbox=None, pool=None):
    """A minimal object exposing exactly the (sandbox, agresource_pool)
    surface agmcp_server's tools read -- not a full `agent`/`agskill`."""
    ag = MagicMock()
    ag.sandbox = sandbox or MagicMock()
    ag.sandbox._cpu_acquired = 0.0
    ag.sandbox._memory_acquired_mb = 0
    ag.agresource_pool = pool or MagicMock()
    ag.agresource_pool.idle_cpus = 8.0
    ag.agresource_pool.idle_memory = None
    skill = MagicMock()
    return ag, skill


async def _call_tool_over_http(base_url, token, tool_name, arguments):
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    http_client = httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"})
    async with streamable_http_client(f"{base_url}/mcp", http_client=http_client) as (
        read,
        write,
    ):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool_name, arguments)


def _run(coro):
    return asyncio.run(coro)


class TestAgMCPServerRealClient:
    @classmethod
    def setup_class(cls):
        from agency.agharness_internal.agmcp_server import agMCPServer

        cls.server = agMCPServer()
        cls.base_url = cls.server.start(timeout_s=10)

    @classmethod
    def teardown_class(cls):
        cls.server.stop()

    def test_tool_discovery_lists_expected_tools(self):
        async def go():
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client

            async with streamable_http_client(f"{self.base_url}/mcp") as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.list_tools()

        result = _run(go())
        names = {t.name for t in result.tools}
        assert {
            "reserve_cpu",
            "cpu_release",
            "daemon_release",
            "submit_output",
            "ask_human",
        } <= names
        # The known, documented gap -- see agmcp_server.py's module
        # docstring -- reserve_gpu is deliberately NOT exposed yet.
        assert "reserve_gpu" not in names

    def test_unknown_token_returns_error(self):
        result = _run(
            _call_tool_over_http(self.base_url, "totally-unknown-token", "cpu_release", {})
        )
        payload = result.structured_content or result.content[0].text
        assert "unknown or missing token" in str(payload)

    def test_reserve_cpu_and_release_round_trip(self):
        ag, skill = _make_session()
        token = "tok-cpu"
        self.server.register(token, ag, skill)
        try:
            result = _run(
                _call_tool_over_http(
                    self.base_url, token, "reserve_cpu", {"cpus": 4.0, "memory": "8g"}
                )
            )
            assert "error" not in str(result.structured_content)
            ag.sandbox.update_limits.assert_called_once_with(cpus=4.0, memory="8g")
            assert ag.sandbox._cpu_acquired == 4.0
            assert ag.sandbox._memory_acquired_mb == 8192
            ag.agresource_pool.notify_cpu_acquired.assert_called_once_with(4.0, 8192)

            _run(_call_tool_over_http(self.base_url, token, "cpu_release", {}))
            ag.sandbox.update_limits.assert_called_with(cpus=8.0, memory=None)
            assert ag.sandbox._cpu_acquired == 0.0
        finally:
            self.server.unregister(token)

    def test_daemon_release_calls_sandbox(self):
        ag, skill = _make_session()
        token = "tok-daemon"
        self.server.register(token, ag, skill)
        try:
            _run(_call_tool_over_http(self.base_url, token, "daemon_release", {"pid": 4242}))
            ag.sandbox.release_daemon.assert_called_once_with(4242)
        finally:
            self.server.unregister(token)

    def test_submit_output_records_value_and_reports_still_missing(self):
        from agency.agdata import agdata
        from agency.agschema import agschema

        ag, skill = _make_session()
        skill.output_schema = agschema(agdata(greeting=str, word_count=int))
        token = "tok-output"
        self.server.register(token, ag, skill)
        try:
            result = _run(
                _call_tool_over_http(
                    self.base_url,
                    token,
                    "submit_output",
                    {"field": "greeting", "value": '"hello there"'},
                )
            )
            payload = result.structured_content
            assert payload["still_missing"] == ["word_count"]
            assert self.server.collected_output(token) == {"greeting": "hello there"}

            _run(
                _call_tool_over_http(
                    self.base_url, token, "submit_output", {"field": "word_count", "value": "2"}
                )
            )
            assert self.server.collected_output(token) == {
                "greeting": "hello there",
                "word_count": 2,
            }
        finally:
            self.server.unregister(token)

    def test_submit_output_rejects_wrong_type(self):
        from agency.agdata import agdata
        from agency.agschema import agschema

        ag, skill = _make_session()
        skill.output_schema = agschema(agdata(word_count=int))
        token = "tok-output-bad"
        self.server.register(token, ag, skill)
        try:
            result = _run(
                _call_tool_over_http(
                    self.base_url,
                    token,
                    "submit_output",
                    {"field": "word_count", "value": '"not a number"'},
                )
            )
            payload = result.structured_content
            assert "error" in payload
            assert self.server.collected_output(token) == {}
        finally:
            self.server.unregister(token)

    def test_ask_human_calls_shared_blocking_helper_with_live_agent(self):
        """Proves the MCP wiring (token -> live ag, question passthrough) --
        the actual blocking/webui-vs-stdin mechanics are `ask_human_and_
        wait`'s own responsibility, already covered by
        tests/agwebui/test_agwebui_hooks.py against the host-side agtool
        that shares this same implementation."""
        from unittest.mock import patch

        ag, skill = _make_session()
        ag.agname = "tok-human-agent"
        token = "tok-human"
        self.server.register(token, ag, skill)
        try:
            with patch("agency.tools.human.ask_human_and_wait", return_value="42") as mock_wait:
                result = _run(
                    _call_tool_over_http(
                        self.base_url, token, "ask_human", {"question": "how many?"}
                    )
                )
            assert result.structured_content == {"reply": "42"}
            mock_wait.assert_called_once()
            call_args = mock_wait.call_args
            assert call_args.args[0] == "tok-human-agent"
            assert call_args.args[1] == "how many?"
            assert call_args.kwargs["ag"] is ag
        finally:
            self.server.unregister(token)

    def test_uds_transport_real_round_trip(self):
        """Proves the UDS listener works too -- the transport
        native.py's in-container entrypoint and any harness with a
        UDS-capable MCP client would actually use, since the TCP listener
        alone isn't reachable from inside a container (same reasoning as
        agllm_terminus's UDS listener)."""

        async def go():
            import httpx2
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client

            sock_path = self.server.ensure_uds_started(timeout_s=10)
            transport = httpx2.AsyncHTTPTransport(uds=sock_path)
            http_client = httpx2.AsyncClient(transport=transport, base_url="http://127.0.0.1")
            async with streamable_http_client("http://127.0.0.1/mcp", http_client=http_client) as (
                read,
                write,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.list_tools()

        result = _run(go())
        names = {t.name for t in result.tools}
        assert {
            "reserve_cpu",
            "cpu_release",
            "daemon_release",
            "submit_output",
            "ask_human",
        } <= names

    def test_submit_output_unknown_field_rejected(self):
        from agency.agdata import agdata
        from agency.agschema import agschema

        ag, skill = _make_session()
        skill.output_schema = agschema(agdata(greeting=str))
        token = "tok-output-unknownfield"
        self.server.register(token, ag, skill)
        try:
            result = _run(
                _call_tool_over_http(
                    self.base_url,
                    token,
                    "submit_output",
                    {"field": "not_a_real_field", "value": '"x"'},
                )
            )
            payload = result.structured_content
            assert "unknown output field" in str(payload.get("error", ""))
        finally:
            self.server.unregister(token)
