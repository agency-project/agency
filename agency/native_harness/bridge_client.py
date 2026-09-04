"""Optional bridge client for the standalone native harness.

When native_harness is launched BY agency, it's given a `--bridge-base-url`
pointed at this run's own `agmanager_harness` instance and a `--bridge-
token` bearer credential -- the same single (base_url, token) pair that
already serves LLM dispatch (`llm_client.py` points its OpenAI-compatible
client's `base_url` at `<bridge-base-url>` too), so ONE bridge configures
policy checks, lifecycle checkpoints, and context-limit lookup all at once,
mirroring how Claude Code's own single `ANTHROPIC_BASE_URL`/
`ANTHROPIC_AUTH_TOKEN` pair already serves its LLM traffic, its permission
hook, and its profiler hook through the very same `agmanager_harness`
process.

When native_harness is run fully standalone (no `--bridge-base-url`), this
client is simply never constructed -- every one of these checks is an
agency-side control-plane concern with no meaning outside agency, so
`react_loop.py` treats "no bridge configured" as "allow every tool, never
pause, never compact" (context_limit=None), not an error."""

from __future__ import annotations

import httpx


CONTROL_PHASE_BOUNDARY = "boundary"
CONTROL_PHASE_CLOSING = "closing"
CONTROL_PHASE_MODEL = "model"


class BridgeClient:
    def __init__(self, base_url: str, token: str, timeout_s: float = 300) -> None:
        self.token = token
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s)

    def check_tool_policy(self, tool_name: str, tool_input: dict) -> dict:
        try:
            resp = self._client.post(
                "/agpolicy/check_tool",
                json={"tool_name": tool_name, "tool_input": tool_input},
                headers={"Authorization": f"Bearer {self.token}"},
            )
            if resp.status_code != 200:
                return {"decision": "deny", "reason": f"policy bridge returned {resp.status_code}"}
            return resp.json()
        except Exception as e:
            # Fail closed, not open: an unreachable policy bridge must not
            # silently become "allow everything."
            return {"decision": "deny", "reason": f"policy bridge unreachable: {e}"}

    def checkpoint(
        self,
        boundary_id: str,
        *,
        allow_messages: bool,
        phase: str,
    ) -> dict:
        try:
            resp = self._client.post(
                "/internal/checkpoint",
                json={
                    "boundary_id": boundary_id,
                    "allow_messages": allow_messages,
                    "phase": phase,
                },
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=None,
            )
            if resp.status_code != 200:
                return {
                    "cancelled": True,
                    "destroyed": False,
                    "invocation_messages": [],
                    "error": f"control bridge returned {resp.status_code}",
                }
            result = resp.json()
            return {
                "cancelled": bool(result.get("cancelled")),
                "destroyed": bool(result.get("destroyed")),
                "invocation_messages": result.get("invocation_messages") or [],
                "action_admitted": bool(result.get("action_admitted")),
            }
        except Exception as exc:
            # A configured control bridge is authoritative; continuing when
            # it is unreachable could run another model call or tool after a
            # pause/cancel that Agency can no longer deliver.
            return {
                "cancelled": True,
                "destroyed": False,
                "invocation_messages": [],
                "error": f"control bridge unreachable: {exc}",
            }

    def context_limit(self) -> "int | None":
        try:
            resp = self._client.post("/internal/context_limit", json={"token": self.token})
            if resp.status_code != 200:
                return None
            return resp.json().get("context_limit")
        except Exception:
            return None

    def close(self) -> None:
        self._client.close()


__all__ = [
    "BridgeClient",
    "CONTROL_PHASE_BOUNDARY",
    "CONTROL_PHASE_CLOSING",
    "CONTROL_PHASE_MODEL",
]
