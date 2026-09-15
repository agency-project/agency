"""MCP transport settings shared by host and sandbox tool servers."""

from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import Response


class _RequestOnlyMcp:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # Stateless/json-response settings still permit a standalone GET SSE
        # stream in the SDK. A retained CLI would pin its host attempt lease
        # forever. MCP permits 405 when the optional event stream is not offered.
        if scope["type"] == "http" and scope["method"] == "GET":
            await Response(status_code=405, headers={"Allow": "POST"})(scope, receive, send)
            return
        await self.app(scope, receive, send)


def build_http_app(server, *, live_session=False):
    app = server.streamable_http_app(
        **({"stateless_http": True, "json_response": True} if live_session else {}),
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    if live_session:
        app.add_middleware(_RequestOnlyMcp)
    return app
