"""Materialize an attempt's explicit sandbox tools in the Harness Manager."""

from __future__ import annotations

import base64
import inspect
from typing import Annotated, Any, Callable

import cloudpickle
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.utilities.func_metadata import WithJsonSchema
from mcp.server.transport_security import TransportSecuritySettings

from ..agdata import agdata
from ..agtool import agtool


class SandboxMcpSetupError(RuntimeError):
    """Setup diagnostics safe to return across the attempt RPC."""


def build_app(payload: str, is_active: Callable[[], bool]):
    try:
        tools = cloudpickle.loads(base64.b64decode(payload, validate=True))
        if (
            not isinstance(tools, list)
            or not tools
            or any(not isinstance(tool, agtool) for tool in tools)
        ):
            raise ValueError("expected a nonempty list of agtool objects")
    except Exception as exc:
        raise SandboxMcpSetupError(
            f"sandbox MCP setup failed during deserialization ({type(exc).__name__})"
        ) from None

    server = MCPServer(name="agency-sandbox-mcp-server")
    persistent_vars: dict[str, object] = {}
    names = set()
    for tool in tools:
        try:
            if tool.name in names:
                raise ValueError("duplicate sandbox tool name")
            names.add(tool.name)
            _register_tool(server, tool, persistent_vars, is_active)
        except Exception as exc:
            raise SandboxMcpSetupError(
                f"sandbox MCP setup failed registering tool {tool.name!r} ({type(exc).__name__})"
            ) from None

    return server.streamable_http_app(
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False)
    )


def _register_tool(server, tool, persistent_vars, is_active) -> None:
    # Each wrapper captures its own reconstructed tool; no host execution
    # context or reverse proxy participates in these calls.
    def call_tool(**kwargs) -> dict:
        if not is_active():
            raise RuntimeError("inactive sandbox MCP attempt")
        persistent = {}
        for name, factory in tool.persistent_vars.items():
            if name not in persistent_vars:
                persistent_vars[name] = factory()
            persistent[name] = persistent_vars[name]
        return tool(agdata(**kwargs), **persistent).to_dict()

    properties = tool.params.get("properties", {})
    required = set(tool.params.get("required", list(properties)))
    call_tool.__name__ = tool.name
    call_tool.__signature__ = inspect.Signature(
        [
            inspect.Parameter(
                name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=Annotated[Any, WithJsonSchema(schema)],
                default=inspect.Parameter.empty if name in required else None,
            )
            for name, schema in properties.items()
        ]
    )
    server.add_tool(call_tool, name=tool.name, description=tool.description)
