"""Sandbox-side client for host interaction, policy, and control RPCs."""

from __future__ import annotations

import httpx


class HostInteractionClient:
    def __init__(self, uds_path: str, timeout_s: float = 300.0) -> None:
        self._client = httpx.Client(
            transport=httpx.HTTPTransport(uds=uds_path),
            base_url="http://agency-host",
            timeout=timeout_s,
        )

    def check_tool(self, tool_name: str, tool_input: dict) -> "tuple[bool, str | None]":
        response = self._client.post(
            "/interaction/check_tool",
            json={"tool_name": tool_name, "tool_input": tool_input},
        )
        response.raise_for_status()
        result = response.json()
        return bool(result["allowed"]), result.get("reason")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HostInteractionClient":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


__all__ = ["HostInteractionClient"]
