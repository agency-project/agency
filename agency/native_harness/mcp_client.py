"""MCP client for the standalone native harness's extension tools.

Per E.'s decision: built-in tools ship with the harness (`tools.py`);
everything else is an MCP server, configured via `--mcp-config` -- the
exact same convention Claude Code already uses (a JSON blob shaped like
`{"mcpServers": {"name": {"type": "http", "url": ..., "headers": {...}}}}`),
so a real, standalone MCP server works identically whether native_harness
is launched by agency (pointed at `agmanager_harness`'s own `/mcp` mount)
or run by a person from a bash prompt (pointed at any MCP server they
choose).

Uses the real `mcp` client library, the same one every harness's own
native MCP client speaks -- no hand-rolled JSON-RPC client, no risk of
diverging from the actual protocol. `httpx2` (not `httpx`) is used for the
transport here, matching the old `_native_in_container_entrypoint.py`'s
own MCP client code -- kept for consistency with that proven-working
pattern rather than switched to plain `httpx` without a reason to."""

from __future__ import annotations

import asyncio
import json


def _decode_tool_result(result) -> dict:
    """Recover a mapping from MCP structured content or JSON text content."""
    if result.structured_content is not None:
        return result.structured_content
    content = list(result.content or [])
    if len(content) == 1 and isinstance(getattr(content[0], "text", None), str):
        text = content[0].text
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        else:
            if isinstance(decoded, dict):
                return decoded
        return {"result": text}
    if not content:
        return {}

    # MCP content is a tagged union, not text-only. Preserve every image,
    # audio, resource, and text block when the result cannot use the legacy
    # single-text compatibility shape above.
    blocks = []
    for block in content:
        model_dump = getattr(block, "model_dump", None)
        if callable(model_dump):
            blocks.append(model_dump(mode="json", by_alias=True, exclude_none=True))
        elif isinstance(block, dict):
            blocks.append(dict(block))
        else:
            blocks.append(
                {
                    "type": str(getattr(block, "type", type(block).__name__)),
                    "value": str(block),
                }
            )
    return {"content": blocks}


def _mcp_server_configs(mcp_config: "dict | None") -> "list[tuple[str, str, dict]]":
    """`(server_name, url, headers)` for every `type: "http"` server in an
    already-parsed `--mcp-config` blob. Non-HTTP server types (a local
    stdio command, say) aren't supported here -- same scope the old
    entrypoint's own MCP client had (only ever spoke to agmcp_server's own
    HTTP-transport server)."""
    if not mcp_config:
        return []
    servers = mcp_config.get("mcpServers") or {}
    out = []
    for name, cfg in servers.items():
        if cfg.get("type") != "http" or not cfg.get("url"):
            continue
        out.append((name, cfg["url"], cfg.get("headers") or {}))
    return out


async def _list_tools_async(url: str, headers: dict) -> list:
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    http_client = httpx2.AsyncClient(headers=headers)
    async with streamable_http_client(url, http_client=http_client) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.input_schema,
                    },
                }
                for t in result.tools
            ]


async def _call_tool_async(url: str, headers: dict, tool_name: str, arguments: dict) -> dict:
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    http_client = httpx2.AsyncClient(headers=headers)
    async with streamable_http_client(url, http_client=http_client) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return _decode_tool_result(result)


class McpToolset:
    """Discovers every configured MCP server's tools once, and dispatches
    a tool call by name back to whichever server owns it."""

    def __init__(self, mcp_config: "dict | None") -> None:
        self._servers = _mcp_server_configs(mcp_config)
        # tool_name -> (url, headers), last server registered for a given
        # name wins on collision -- same precedence built-ins/MCP/custom
        # tools already use elsewhere in this codebase (first-registered
        # wins there; here there's no built-in-vs-MCP collision to begin
        # with, since tools.py's names are reserved and never repeated in
        # an MCP server's own tool set by convention).
        self._tool_owner: "dict[str, tuple[str, dict]]" = {}

    def discover(self) -> "list[dict]":
        schemas: "list[dict]" = []
        for _name, url, headers in self._servers:
            try:
                tools = asyncio.run(_list_tools_async(url, headers))
            except Exception as e:
                print(f"[native_harness] WARNING: MCP server {url!r} unreachable: {e}")
                continue
            for schema in tools:
                tool_name = schema["function"]["name"]
                self._tool_owner[tool_name] = (url, headers)
                schemas.append(schema)
        return schemas

    def call(self, tool_name: str, arguments_json: str) -> str:
        try:
            arguments = json.loads(arguments_json) if arguments_json else {}
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        url, headers = self._tool_owner[tool_name]
        try:
            result = asyncio.run(_call_tool_async(url, headers, tool_name, arguments))
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
        return json.dumps(result)


__all__ = ["McpToolset"]
