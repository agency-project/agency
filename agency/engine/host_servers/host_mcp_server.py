from __future__ import annotations

import inspect
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.utilities.func_metadata import WithJsonSchema
from mcp.server.transport_security import TransportSecuritySettings

from ...agdata import agdata

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from ...agresources import agResourcePool
    from ...agskill import agskill
    from ...agtool import agtool
    from ...sandbox.agsandbox import agSandbox


class HostMcpServer:
    def __init__(
        self, sandbox: "agSandbox", skill: "agskill", resource_pool: "agResourcePool"
    ) -> None:
        self._sandbox = sandbox
        self._skill = skill
        self._resource_pool = resource_pool
        self._persistent_vars: "dict[str, object]" = {}
        self._mcp_server: "MCPServer | None" = None

    def _register_tool(self, server: MCPServer, tool: "agtool") -> None:
        properties = (tool.params or {}).get("properties", {})
        required = set((tool.params or {}).get("required", list(properties.keys())))

        def call_tool(**kwargs: "object") -> dict:
            persistent = {
                var_name: self._persistent_vars.setdefault(var_name, factory())
                for var_name, factory in tool.persistent_vars.items()
            }
            return tool(
                agdata(**kwargs),
                sandbox=self._sandbox,
                resource_pool=self._resource_pool,
                output_schema=self._skill.output_schema,
                **persistent,
            ).to_dict()

        call_tool.__name__ = tool.name
        call_tool.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    key,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    annotation=Annotated[Any, WithJsonSchema(schema)],
                    default=inspect.Parameter.empty if key in required else None,
                )
                for key, schema in properties.items()
            ]
        )
        server.add_tool(call_tool, name=tool.name, description=tool.description)

    def collected_output(self) -> dict:
        return dict(self._persistent_vars.get("submitted_output_store", {}))

    def build_app(self) -> "Starlette":
        server = MCPServer(name="agency-host-mcp-server")

        for tool in self._skill.host_mcp_tools:
            self._register_tool(server, tool)

        self._mcp_server = server
        return server.streamable_http_app(
            transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False)
        )

    def lifespan_context(self, app: "Starlette") -> "AbstractAsyncContextManager[None] | None":
        return app.router.lifespan_context(app)
