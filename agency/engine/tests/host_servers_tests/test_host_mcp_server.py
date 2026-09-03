# Tests for host_mcp_server.py

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from agency.agdata import agdata
from agency.agskill import agskill
from agency.agtool import agtool
from agency.engine.host_servers.host_mcp_server import HostMcpServer

_FIXED_TOOL_NAMES = {
    "reserve_resource",
    "release_resource",
    "get_current_resources",
    "daemon_release",
    "submit_output",
    "submitted_output",
}


class _FakeSandbox:
    def __init__(self):
        self._agname = "test_0000"
        self._cpu_acquired = 0.0
        self._memory_acquired_mb = 0
        self._gpu_ids: "list[int]" = []
        self._gpu_count_requested = 0
        self._gpu_acquire_fn = None
        self._gpu_release_fn = None

    def update_limits(self, cpus=None, memory=None):
        self.last_update = (cpus, memory)

    def release_daemon(self, pid):
        self.released_pid = pid


class _FakeResourcePool:
    idle_cpus = 8.0
    idle_memory = None
    total_cpus = 8
    total_memory_mb = 16384
    gpus = [0, 1, 2, 3]

    def acquire_cpu_mem(self, sandbox, cpus=None, memory_mb=None):
        self.acquire_cpu_mem_called_with = (cpus, memory_mb)

    def release_cpu_mem(self, sandbox, cpu=False, memory=False):
        self.release_cpu_mem_called_with = (cpu, memory)

    def acquire_gpus(self, sandbox, count, timeout=None):
        self.acquire_gpus_called_with = count
        ids = list(range(count))
        sandbox._gpu_ids = ids
        return ids

    def release_gpus(self, sandbox, gpu_ids):
        self.release_gpus_called_with = list(gpu_ids)
        sandbox._gpu_ids = []


class _FakeDataLogger:
    def __init__(self):
        self.events = []

    def record_event(self, type, payload, call_label=None, update_latest_snapshot=False, **_kw):
        self.events.append((type, payload, call_label, update_latest_snapshot))


def _make_server(add_host_mcp_tools=None, sandbox=None, resource_pool=None, output_schema=None):
    sandbox = sandbox if sandbox is not None else SimpleNamespace()
    resource_pool = resource_pool if resource_pool is not None else SimpleNamespace()
    skill = agskill(
        name="s",
        system_prompt="p",
        add_host_mcp_tools=add_host_mcp_tools,
        output_schema=output_schema,
    )
    server = HostMcpServer(sandbox, skill, resource_pool, _FakeDataLogger())
    server.build_app()
    return server, sandbox, resource_pool


def _call(server, name, args):
    return asyncio.run(server._mcp_server.call_tool(name, args))


def _tool_names(server):
    return {t.name for t in server._mcp_server._tool_manager.list_tools()}


def _tool(server, name):
    return server._mcp_server._tool_manager.get_tool(name)


def test_build_app_registers_the_default_host_tools_with_no_extras():
    server, _, _ = _make_server(add_host_mcp_tools=None)
    assert _tool_names(server) == _FIXED_TOOL_NAMES


def test_build_app_registers_add_host_mcp_tools_alongside_the_defaults():
    echo = agtool(
        name="echo",
        description="Echo back the message.",
        fn=lambda d: agdata(msg=d._data["msg"]),
        params={
            "type": "object",
            "properties": {"msg": {"type": "string", "description": "text to echo"}},
            "required": ["msg"],
        },
    )
    server, _, _ = _make_server(add_host_mcp_tools=[echo])
    assert _tool_names(server) == _FIXED_TOOL_NAMES | {"echo"}


def test_dynamic_tool_preserves_the_original_json_schema():
    echo = agtool(
        name="echo",
        description="Echo back the message.",
        fn=lambda d: agdata(msg=d._data["msg"]),
        params={
            "type": "object",
            "properties": {"msg": {"type": "string", "description": "text to echo"}},
            "required": ["msg"],
        },
    )
    server, _, _ = _make_server(add_host_mcp_tools=[echo])
    schema = _tool(server, "echo").parameters
    assert schema["properties"]["msg"]["type"] == "string"
    assert schema["properties"]["msg"]["description"] == "text to echo"
    assert schema["required"] == ["msg"]


def test_dynamic_tool_defaults_to_requiring_every_property_when_unspecified():
    tool = agtool(
        name="noreq",
        description="no explicit required list",
        fn=lambda d: agdata(),
        params={"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}},
    )
    server, _, _ = _make_server(add_host_mcp_tools=[tool])
    schema = _tool(server, "noreq").parameters
    assert set(schema["required"]) == {"a", "b"}


def test_dynamic_tool_with_no_params_takes_no_arguments():
    calls = []
    tool = agtool(
        name="noargs",
        description="takes nothing",
        fn=lambda d: calls.append(dict(d._data)) or agdata(),
    )
    server, _, _ = _make_server(add_host_mcp_tools=[tool])
    result = asyncio.run(server._mcp_server.call_tool("noargs", {}))
    assert calls == [{}]
    assert result.is_error is False


def test_dynamic_tool_dispatch_invokes_the_agtool_fn():
    echo = agtool(
        name="echo",
        description="Echo back the message.",
        fn=lambda d: agdata(msg=d._data["msg"]),
        params={
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"],
        },
    )
    server, _, _ = _make_server(add_host_mcp_tools=[echo])
    result = asyncio.run(server._mcp_server.call_tool("echo", {"msg": "hello"}))
    assert result.is_error is False
    assert "hello" in result.content[0].text


def test_call_tool_records_tool_result_with_arguments_and_result():
    """Tool arguments/results aren't in any llm_block row (only the model's
    own tool_use block is) -- this is the only place a tool's actual return
    value ever gets persisted, needed for live per-agent transcript
    reconstruction in the webui."""
    echo = agtool(
        name="echo",
        description="Echo back the message.",
        fn=lambda d: agdata(msg=d._data["msg"]),
        params={
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"],
        },
    )
    server, _, _ = _make_server(add_host_mcp_tools=[echo])
    asyncio.run(server._mcp_server.call_tool("echo", {"msg": "hello"}))

    tool_result_events = [e for e in server._data_logger.events if e[0] == "tool_result"]
    assert len(tool_result_events) == 1
    _type, payload, _call_label, _snapshot = tool_result_events[0]
    assert payload["tool"] == "echo"
    assert payload["arguments"] == {"msg": "hello"}
    assert payload["result"]["msg"] == "hello"


def test_dynamic_tool_missing_required_field_raises_tool_error():
    echo = agtool(
        name="echo",
        description="Echo back the message.",
        fn=lambda d: agdata(msg=d._data["msg"]),
        params={
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"],
        },
    )
    server, _, _ = _make_server(add_host_mcp_tools=[echo])
    with pytest.raises(ToolError):
        asyncio.run(server._mcp_server.call_tool("echo", {}))


# ---------------------------------------------------------------------------
# Default host tools -- real behavior
# ---------------------------------------------------------------------------


def test_reserve_resource_cpu_only_leaves_memory_and_gpu_untouched():
    sandbox = _FakeSandbox()
    pool = _FakeResourcePool()
    server, sandbox, pool = _make_server(sandbox=sandbox, resource_pool=pool)
    result = _call(server, "reserve_resource", {"cpus": 2.0})
    assert result.is_error is False
    assert pool.acquire_cpu_mem_called_with == (2.0, None)
    assert sandbox._gpu_count_requested == 0
    assert not hasattr(pool, "acquire_gpus_called_with")


def test_reserve_resource_cpu_and_memory_together():
    sandbox = _FakeSandbox()
    pool = _FakeResourcePool()
    server, sandbox, pool = _make_server(sandbox=sandbox, resource_pool=pool)
    result = _call(server, "reserve_resource", {"cpus": 2.0, "memory_mb": 512})
    assert result.is_error is False
    assert pool.acquire_cpu_mem_called_with == (2.0, 512)


def test_reserve_resource_rejects_cpus_exceeding_pool_total():
    server, sandbox, pool = _make_server(sandbox=_FakeSandbox(), resource_pool=_FakeResourcePool())
    result = _call(server, "reserve_resource", {"cpus": 100.0})
    assert "cpus" in result.content[0].text
    assert not hasattr(pool, "acquire_cpu_mem_called_with")


def test_reserve_resource_rejects_memory_exceeding_pool_total():
    server, sandbox, pool = _make_server(sandbox=_FakeSandbox(), resource_pool=_FakeResourcePool())
    result = _call(server, "reserve_resource", {"memory_mb": 999999})
    assert "MB" in result.content[0].text
    assert not hasattr(pool, "acquire_cpu_mem_called_with")


def test_reserve_resource_gpu_only_arms_lazy_acquire_without_calling_the_pool():
    sandbox = _FakeSandbox()
    pool = _FakeResourcePool()
    server, sandbox, pool = _make_server(sandbox=sandbox, resource_pool=pool)
    result = _call(server, "reserve_resource", {"gpu": 2})
    assert result.is_error is False
    assert sandbox._gpu_count_requested == 2
    assert not hasattr(pool, "acquire_gpus_called_with")
    assert sandbox._cpu_acquired == 0.0

    # Lazy acquire/release fns are wrapped (bound to the requesting sandbox)
    # so they can be invoked later with just (count) / (gpu_ids).
    sandbox._gpu_acquire_fn(2)
    assert pool.acquire_gpus_called_with == 2
    sandbox._gpu_release_fn([0, 1])
    assert pool.release_gpus_called_with == [0, 1]


def test_reserve_resource_rejects_gpu_count_exceeding_pool_size():
    server, sandbox, pool = _make_server(sandbox=_FakeSandbox(), resource_pool=_FakeResourcePool())
    result = _call(server, "reserve_resource", {"gpu": 100})
    assert "gpus" in result.content[0].text
    assert sandbox._gpu_count_requested == 0


def test_reserve_resource_with_nothing_given_returns_an_error():
    server, _, _ = _make_server(sandbox=_FakeSandbox(), resource_pool=_FakeResourcePool())
    result = _call(server, "reserve_resource", {})
    assert "nothing to reserve" in result.content[0].text


def test_release_resource_cpu_only_leaves_gpu_untouched():
    sandbox = _FakeSandbox()
    sandbox._gpu_count_requested = 1
    sandbox._gpu_ids = [0]
    pool = _FakeResourcePool()
    server, sandbox, pool = _make_server(sandbox=sandbox, resource_pool=pool)
    result = _call(server, "release_resource", {"cpu": True})
    assert result.is_error is False
    # The MCP tool call fills the unset "memory" param with an explicit None
    # (not omitted), so arg._data.get("memory", False) resolves to None here
    # -- falsy either way, which is all release_cpu_mem's `if memory:` cares about.
    assert pool.release_cpu_mem_called_with == (True, None)
    assert sandbox._gpu_count_requested == 1
    assert sandbox._gpu_ids == [0]


def test_release_resource_cpu_and_memory_together():
    sandbox = _FakeSandbox()
    pool = _FakeResourcePool()
    server, sandbox, pool = _make_server(sandbox=sandbox, resource_pool=pool)
    result = _call(server, "release_resource", {"cpu": True, "memory": True})
    assert result.is_error is False
    assert pool.release_cpu_mem_called_with == (True, True)


def test_release_resource_gpu_releases_held_ids_and_disarms_lazy_acquire():
    sandbox = _FakeSandbox()
    sandbox._gpu_count_requested = 2
    sandbox._gpu_ids = [0, 1]
    pool = _FakeResourcePool()
    server, sandbox, pool = _make_server(sandbox=sandbox, resource_pool=pool)
    result = _call(server, "release_resource", {"gpu": True})
    assert result.is_error is False
    assert pool.release_gpus_called_with == [0, 1]
    assert sandbox._gpu_ids == []
    assert sandbox._gpu_count_requested == 0
    assert sandbox._gpu_acquire_fn is None
    assert sandbox._gpu_release_fn is None


def test_release_resource_gpu_with_none_reserved_reports_nothing_held():
    server, sandbox, pool = _make_server(sandbox=_FakeSandbox(), resource_pool=_FakeResourcePool())
    result = _call(server, "release_resource", {"gpu": True})
    assert result.is_error is False
    assert "no gpu was reserved" in result.content[0].text


def test_release_resource_with_nothing_given_returns_an_error():
    server, _, _ = _make_server(sandbox=_FakeSandbox(), resource_pool=_FakeResourcePool())
    result = _call(server, "release_resource", {})
    assert "nothing to release" in result.content[0].text


def test_get_current_resources_reports_sandbox_and_pool_state():
    sandbox = _FakeSandbox()
    sandbox._cpu_acquired = 2.0
    sandbox._memory_acquired_mb = 512
    sandbox._gpu_count_requested = 1
    sandbox._gpu_ids = [0]
    pool = _FakeResourcePool()
    server, sandbox, pool = _make_server(sandbox=sandbox, resource_pool=pool)
    result = _call(server, "get_current_resources", {})
    assert result.is_error is False
    body = json.loads(result.content[0].text)
    assert body == {
        "cpus_acquired": 2.0,
        "memory_mb_acquired": 512,
        "gpu_count_requested": 1,
        "gpu_ids_held": [0],
        "total_cpus": pool.total_cpus,
        "total_memory_mb": pool.total_memory_mb,
        "total_gpus": len(pool.gpus),
    }


def test_daemon_release_calls_sandbox_release_daemon():
    sandbox = _FakeSandbox()
    server, sandbox, _ = _make_server(sandbox=sandbox)
    result = _call(server, "daemon_release", {"pid": 4321})
    assert result.is_error is False
    assert sandbox.released_pid == 4321


def test_submit_output_with_no_output_schema_returns_an_error():
    server, _, _ = _make_server(output_schema=None)
    result = _call(server, "submit_output", {"field": "x", "value": "y"})
    assert "no output_schema" in result.content[0].text


def test_submit_output_rejects_an_unknown_field():
    server, _, _ = _make_server(output_schema=agdata(summary=str))
    result = _call(server, "submit_output", {"field": "bogus", "value": "x"})
    assert "unknown output field" in result.content[0].text


def test_submit_output_rejects_a_value_of_the_wrong_type():
    server, _, _ = _make_server(output_schema=agdata(count=int))
    result = _call(server, "submit_output", {"field": "count", "value": "not-an-int"})
    assert result.content[0].text  # check_field's own error message, not asserting exact text
    body = result.content[0].text
    assert "error" in body


def test_submit_output_records_a_valid_field_and_reports_still_missing():
    server, _, _ = _make_server(output_schema=agdata(summary=str, count=int))
    result = _call(server, "submit_output", {"field": "summary", "value": "hello"})
    assert result.is_error is False
    assert "count" in result.content[0].text  # still missing


def test_submit_output_accumulates_across_calls_and_submitted_output_reflects_it():
    server, _, _ = _make_server(output_schema=agdata(summary=str, count=int))
    _call(server, "submit_output", {"field": "summary", "value": "hello"})
    _call(server, "submit_output", {"field": "count", "value": 42})
    result = _call(server, "submitted_output", {})
    assert result.is_error is False
    assert "hello" in result.content[0].text
    assert "42" in result.content[0].text


def test_submitted_output_starts_empty():
    server, _, _ = _make_server(output_schema=agdata(summary=str))
    result = _call(server, "submitted_output", {})
    assert result.is_error is False
    assert result.content[0].text.strip() == "{}"


def test_collected_output_starts_empty():
    server, _, _ = _make_server(output_schema=agdata(summary=str))
    assert server.collected_output() == {}


def test_collected_output_reflects_submitted_fields():
    server, _, _ = _make_server(output_schema=agdata(summary=str, count=int))
    _call(server, "submit_output", {"field": "summary", "value": "hello"})
    _call(server, "submit_output", {"field": "count", "value": 42})
    assert server.collected_output() == {"summary": "hello", "count": 42}


def test_collected_output_returns_a_copy_not_the_live_store():
    server, _, _ = _make_server(output_schema=agdata(summary=str))
    _call(server, "submit_output", {"field": "summary", "value": "hello"})
    snapshot = server.collected_output()
    snapshot["summary"] = "mutated"
    assert server.collected_output() == {"summary": "hello"}


def test_host_tool_receives_the_servers_sandbox_and_resource_pool_as_context():
    seen = {}

    def _probe(arg, sandbox, resource_pool):
        seen["sandbox"] = sandbox
        seen["resource_pool"] = resource_pool
        return agdata(ok=True)

    probe_tool = agtool(name="probe", description="d", fn=_probe)
    sandbox = SimpleNamespace(marker="sandbox")
    resource_pool = SimpleNamespace(marker="resource_pool")
    server, sandbox, resource_pool = _make_server(
        add_host_mcp_tools=[probe_tool], sandbox=sandbox, resource_pool=resource_pool
    )
    result = asyncio.run(server._mcp_server.call_tool("probe", {}))
    assert result.is_error is False
    assert seen["sandbox"] is sandbox
    assert seen["resource_pool"] is resource_pool


def test_lifespan_context_is_usable_as_an_async_context_manager():
    server, _, _ = _make_server()
    app = server.build_app()
    ctx = server.lifespan_context(app)
    assert ctx is not None

    async def _enter_and_exit():
        async with ctx:
            pass

    asyncio.run(_enter_and_exit())
