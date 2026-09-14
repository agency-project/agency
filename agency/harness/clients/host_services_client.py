"""Client for the host services exposed to this sandbox over ``host.sock``.

Everything the container-side process cannot decide on its own (real LLM
dispatch, policy decisions, logging, pause/inbox state) goes through
`HostServicesClient`, never a second, separately-credentialed path. Used by the
harness-facing LLM, interaction, and MCP routes."""

from __future__ import annotations

import hmac
import json
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import TYPE_CHECKING, AsyncIterator

import httpx

from ..protocol import ATTEMPT_TOKEN_HEADER

if TYPE_CHECKING:
    from .._syscall_event import agsyscallevent


class HostDispatchError(RuntimeError):
    """Preserve retryability across the host/daemon protocol boundary."""

    def __init__(self, item):
        super().__init__(item["message"])
        self.status_code = item.get("status_code", 500)
        self.transient = item.get("transient", False)


class HostServicesClient:
    """Thin client wrapping this agent's one bridged connection to its
    `agmanager_host` instance -- the single UDS path any harness (or a
    harness's own profiler) uses to reach the host."""

    def __init__(self, uds_path: str, timeout_s: float = 300) -> None:
        self._uds_path = uds_path
        self._timeout_s = timeout_s
        transport = httpx.HTTPTransport(uds=uds_path)
        self.client = httpx.Client(
            transport=transport, base_url="http://agmanager-host", timeout=timeout_s
        )
        self._attempt_token_lock = threading.Lock()
        self._active_attempt_token: "str | None" = None
        self._upstream_attempt_token: "str | None" = None

    def register_attempt_token(self, token: str, *, local_token: "str | None" = None) -> None:
        """Map one process-local credential to the current host attempt.

        A CRIU-restored CLI cannot change its environment, so it keeps the
        credential created with the process.  The host credential remains
        attempt scoped and is replaced on every dispatch.
        """
        if not isinstance(token, str) or not token:
            raise ValueError("attempt token must be a non-empty string")
        accepted = local_token if local_token is not None else token
        if not isinstance(accepted, str) or not accepted:
            raise ValueError("local attempt token must be a non-empty string")
        with self._attempt_token_lock:
            if self._active_attempt_token is not None:
                raise RuntimeError("another harness attempt token is already active")
            self._active_attempt_token = accepted
            self._upstream_attempt_token = token

    def clear_attempt_token(self, token: str) -> bool:
        """Revoke only the matching credential, making stale cleanup harmless."""
        if not isinstance(token, str) or not token:
            return False
        with self._attempt_token_lock:
            upstream = self._upstream_attempt_token
            if upstream is None or not self._attempt_tokens_match(upstream, token):
                return False
            self._active_attempt_token = None
            self._upstream_attempt_token = None
            return True

    def validate_token(self, token: str) -> bool:
        if not isinstance(token, str) or not token:
            return False
        with self._attempt_token_lock:
            active = self._active_attempt_token
            return active is not None and self._attempt_tokens_match(active, token)

    @staticmethod
    def _attempt_tokens_match(active: str, candidate: str) -> bool:
        try:
            return hmac.compare_digest(active.encode(), candidate.encode())
        except UnicodeEncodeError:
            return False

    def _attempt_headers(self, token: str) -> dict[str, str]:
        if not self.validate_token(token):
            raise RuntimeError("unknown or missing bearer token")
        with self._attempt_token_lock:
            upstream = self._upstream_attempt_token
        if upstream is None:
            raise RuntimeError("no active upstream attempt token")
        return {ATTEMPT_TOKEN_HEADER: upstream}

    def quiesce_connections(self) -> None:
        """Remove host-external sockets before a container-level CRIU dump."""
        self.client.close()
        self.client = httpx.Client(
            transport=httpx.HTTPTransport(uds=self._uds_path),
            base_url="http://agmanager-host",
            timeout=self._timeout_s,
        )

    def resolve_model(self, token: str) -> str:
        resp = self.client.get("/llm/resolve_model", headers=self._attempt_headers(token))
        resp.raise_for_status()
        return resp.json()["model"]

    def context_limit(self, token: str) -> "int | None":
        """This agent's model's context window, for a caller that runs its
        own ReAct loop and needs to know when to compact (native_harness's
        `compaction.py`)."""
        try:
            resp = self.client.get("/llm/context_limit", headers=self._attempt_headers(token))
            if resp.status_code != 200:
                return None
            return resp.json().get("context_limit")
        except Exception:
            return None

    def log_warning(self, token: str, message: str) -> None:
        # DATACOLLECTOR: append -- the one existing production call already wired through
        # record_event(type="warning"); model other emission points after this shape.
        self.client.post(
            "/interaction/record_event",
            json={"type": "warning", "payload": {"message": message}},
            headers=self._attempt_headers(token),
        )

    def record_file_access(self, token: str, payload: dict) -> None:
        response = self.client.post(
            "/interaction/record_event",
            json={"type": "checkpoint_file_access", "payload": payload},
            headers=self._attempt_headers(token),
            timeout=2,
        )
        response.raise_for_status()

    def check_tool_policy(self, token: str, tool_name: str, tool_input: dict) -> dict:
        # A GPU-gated tool call can legitimately block on the host for as
        # long as another agent holds the GPU it reserved -- unbounded by
        # design (see agResourcePool.acquire_gpus), so this one call, unlike
        # the rest of this client's (should-be-fast) calls, must not race a
        # generic client timeout. Same reasoning as engine.py's
        # `handle.client(timeout_s=None)` for run_harness_attempt.
        resp = self.client.post(
            "/interaction/check_tool",
            json={"tool_name": tool_name, "tool_input": tool_input},
            headers=self._attempt_headers(token),
            timeout=None,
        )
        resp.raise_for_status()
        result = resp.json()
        return {
            "decision": "allow" if result.get("allowed") else "deny",
            "reason": result.get("reason"),
            "call_id": result.get("call_id"),
        }

    def complete_tool_policy(
        self,
        token: str,
        call_id: str,
        result: object = None,
        error: "str | None" = None,
        duration_ns: int | None = None,
        started_wall_ns: int | None = None,
    ) -> None:
        response = self.client.post(
            "/interaction/complete_tool",
            json={
                "call_id": call_id,
                "result": result,
                "error": error,
                **({"duration_ns": duration_ns} if duration_ns is not None else {}),
                **({"started_wall_ns": started_wall_ns} if started_wall_ns is not None else {}),
            },
            headers=self._attempt_headers(token),
        )
        response.raise_for_status()

    def check_syscall_policy(
        self, token: str, syscall: "agsyscallevent"
    ) -> "tuple[bool, str | None, str | None, dict[str, str] | None]":
        # See check_tool_policy's comment above -- execve/execveat are
        # exactly the syscalls a GPU reservation gates, so this call can
        # also legitimately block on the host for an unbounded time.
        response = self.client.post(
            "/interaction/check_syscall",
            json=asdict(syscall),
            headers=self._attempt_headers(token),
            timeout=None,
        )
        response.raise_for_status()
        result = response.json()
        return (
            bool(result.get("allowed")),
            result.get("reason"),
            result.get("call_id"),
            result.get("env_overrides"),
        )

    def complete_syscall_policy(
        self,
        token: str,
        call_id: str,
        return_value: "int | None" = None,
        error: "str | None" = None,
    ) -> None:
        response = self.client.post(
            "/interaction/complete_syscall",
            json={"call_id": call_id, "return_value": return_value, "error": error},
            headers=self._attempt_headers(token),
        )
        response.raise_for_status()

    def dispatch(self, token: str, agency_context: dict) -> dict:
        resp = self.client.post(
            "/llm/dispatch",
            json=agency_context,
            headers=self._attempt_headers(token),
        )
        if resp.status_code != 200:
            raise RuntimeError(f"host dispatch failed: {resp.status_code} {resp.text}")
        return resp.json()

    def dispatch_stream(self, token: str, agency_context: dict):
        with self.client.stream(
            "POST",
            "/llm/dispatch",
            json={**agency_context, "stream": True},
            headers=self._attempt_headers(token),
        ) as resp:
            if resp.status_code != 200:
                resp.read()
                raise RuntimeError(f"host dispatch failed: {resp.status_code} {resp.text}")
            for line in resp.iter_lines():
                if not line:
                    continue
                item = json.loads(line)
                if item["type"] == "error":
                    raise RuntimeError(f"host dispatch failed: {item['message']}")
                yield item
                if item["type"] == "done":
                    return

    async def dispatch_stream_async(self, token: str, agency_context: dict):
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=self._uds_path),
            base_url="http://agency-host",
            timeout=None,
        ) as client:
            async with client.stream(
                "POST",
                "/llm/dispatch",
                json={**agency_context, "stream": True},
                headers=self._attempt_headers(token),
            ) as response:
                if response.is_error:
                    await response.aread()
                    try:
                        error = response.json()["error"]
                    except (ValueError, KeyError, TypeError):
                        response.raise_for_status()
                    raise HostDispatchError(
                        {
                            "message": error.get("message", "host dispatch failed"),
                            "status_code": response.status_code,
                            "transient": error.get("transient", response.status_code >= 500),
                        }
                    )
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    item = json.loads(line)
                    if item["type"] == "error":
                        raise HostDispatchError(item)
                    yield item
                    if item["type"] == "done":
                        return

    @asynccontextmanager
    async def forward_mcp_request(
        self,
        token: str,
        method: str,
        *,
        content: bytes,
        headers: dict[str, str],
        params: dict[str, str],
    ) -> AsyncIterator[httpx.Response]:
        """Stream MCP while replacing any sandbox-supplied attempt header.

        A streamable-HTTP GET may stay open for the lifetime of an MCP client.
        Keeping the async response context open lets the proxy forward chunks
        immediately and, critically, cancel the UDS request when that client
        disconnects.
        """
        upstream_headers = {
            key: value
            for key, value in headers.items()
            if key.lower() != ATTEMPT_TOKEN_HEADER.lower()
        }
        upstream_headers.update(self._attempt_headers(token))
        async with self._new_mcp_async_client() as client:
            async with client.stream(
                method,
                "/mcp",
                content=content,
                headers=upstream_headers,
                params=params,
            ) as response:
                yield response

    def _new_mcp_async_client(self) -> httpx.AsyncClient:
        transport = httpx.AsyncHTTPTransport(uds=self._uds_path)
        return httpx.AsyncClient(
            transport=transport,
            base_url="http://agmanager-host",
            timeout=self._timeout_s,
        )

    def record_profiler_span(self, token: str, payload: dict) -> dict:
        """Report one span -- open (no ``end_ts``), closing, or already
        complete -- via the same canonical route any host-observed span
        goes through. See ``HostInteractionServer.record_span``."""
        response = self.client.post(
            "/interaction/record_span",
            json=payload,
            headers=self._attempt_headers(token),
            timeout=2.0,
        )
        response.raise_for_status()
        return response.json()

    def record_profiler_samples(self, token: str, samples: list) -> dict:
        """Report a batch of already-measured function-call samples."""
        response = self.client.post(
            "/interaction/record_samples",
            json={"samples": samples},
            headers=self._attempt_headers(token),
            timeout=2.0,
        )
        response.raise_for_status()
        return response.json()

    def profiler_settings(self, token: str) -> dict:
        """Whether profiling is on, and automatic-function-sampling settings."""
        response = self.client.post(
            "/interaction/profile_settings",
            json={},
            headers=self._attempt_headers(token),
            timeout=2.0,
        )
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        with self._attempt_token_lock:
            self._active_attempt_token = None
            self._upstream_attempt_token = None
        self.client.close()


__all__ = ["HostServicesClient"]
