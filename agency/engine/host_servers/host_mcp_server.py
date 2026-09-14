from __future__ import annotations

import inspect
import logging
import threading
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Annotated, Any, Callable

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.utilities.func_metadata import WithJsonSchema
from mcp.server.transport_security import TransportSecuritySettings

from ...agdata import agdata

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from ...observability.agdatalogger import agDataLogger
    from ...orchestrator.agresources import agResourcePool
    from ...agskill import agskill
    from ...agtool import agtool
    from ...sandbox.agsandbox import agSandbox

_current_data_logger = threading.local()


def bind_data_logger_for_current_thread(data_logger: "agDataLogger") -> None:
    """Route this thread's `mcp.server.*` logging to *data_logger* -- one
    dedicated thread per agent's MCP server makes a thread-local sufficient."""
    _current_data_logger.data_logger = data_logger


class _ThreadRoutedLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        data_logger = getattr(_current_data_logger, "data_logger", None)
        if data_logger is None:
            return
        payload = {
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        try:
            data_logger.record_event("mcp_server_log", payload)
        except Exception:
            self.handleError(record)


_mcp_server_logger = logging.getLogger("mcp.server")
_mcp_server_logger.addHandler(_ThreadRoutedLogHandler())
_mcp_server_logger.propagate = False

# MCPServer.__init__() (below) unconditionally calls mcp's own
# configure_logging(), which does logging.basicConfig(level="INFO",
# handlers=[RichHandler(...)]) the first time any MCPServer is constructed
# in this process -- reconfiguring the process-wide root logger. Any
# library using stdlib logging with no level of its own (httpx, boto3, the
# LLM SDKs' HTTP transport, ...) then inherits that INFO level and starts
# printing every request through Rich. Pin the noisy ones back down
# explicitly here; harmless regardless of whether that basicConfig() has
# already fired or fires later, since an explicit level always wins over
# inherited effective level.
for _noisy_logger_name in ("httpx", "httpcore", "boto3", "botocore", "urllib3"):
    logging.getLogger(_noisy_logger_name).setLevel(logging.WARNING)


class HostMcpServer:
    def __init__(
        self,
        sandbox: "agSandbox",
        skill: "agskill",
        resource_pool: "agResourcePool",
        data_logger: "agDataLogger",
        *,
        is_cancelled: "Callable[[], bool] | None" = None,
    ) -> None:
        self._sandbox = sandbox
        self._skill = skill
        self._resource_pool = resource_pool
        self._data_logger = data_logger
        self._is_cancelled = is_cancelled if is_cancelled is not None else (lambda: False)
        self._persistent_vars: "dict[str, object]" = {}
        self._persistent_lock = threading.Lock()
        self._mcp_server: "MCPServer | None" = None

    def _register_tool(self, server: MCPServer, tool: "agtool") -> None:
        properties = (tool.params or {}).get("properties", {})
        required = set((tool.params or {}).get("required", list(properties.keys())))

        def call_tool(**kwargs: "object") -> dict:
            if self._is_cancelled():
                return {"error": "agent invocation stopped"}
            # No admit_tool_call() here -- the caller (PreToolUse hook or
            # react_loop.py's bridge check) already admitted; doing it again
            # would double-admit every MCP tool call.
            persistent = {}
            # Factories can open resources or build expensive state. Initialize
            # once across MCP worker threads, then release before calling tools.
            with self._persistent_lock:
                for var_name, factory in tool.persistent_vars.items():
                    if var_name not in self._persistent_vars:
                        self._persistent_vars[var_name] = factory()
                    persistent[var_name] = self._persistent_vars[var_name]
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
