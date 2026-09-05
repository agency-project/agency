"""Client for the host services exposed to this sandbox over ``host.sock``.

Everything the container-side process cannot decide on its own (real LLM
dispatch, policy decisions, logging, pause/inbox state) goes through
`HostServicesClient`, never a second, separately-credentialed path. Used by the
harness-facing LLM, interaction, and MCP routes."""

from __future__ import annotations

import hmac
import json
import socket
import struct
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import TYPE_CHECKING, AsyncIterator

import httpx

from ..protocol import ATTEMPT_TOKEN_HEADER

if TYPE_CHECKING:
    from .._syscall_event import agsyscallevent


def _recv_exactly(sock, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(f"connection closed with {remaining} bytes still expected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_framed(sock) -> dict:
    (length,) = struct.unpack(">Q", _recv_exactly(sock, 8))
    return json.loads(_recv_exactly(sock, length).decode("utf-8"))


def _send_framed(sock, payload: dict) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack(">Q", len(body)) + body)


class HostServicesClient:
    """Thin client wrapping this agent's one bridged connection to its
    `agmanager_host` instance."""

    def __init__(
        self, uds_path: str, profiler_uds_path: "str | None", timeout_s: float = 300
    ) -> None:
        self._uds_path = uds_path
        self._timeout_s = timeout_s
        transport = httpx.HTTPTransport(uds=uds_path)
        self.client = httpx.Client(
            transport=transport, base_url="http://agmanager-host", timeout=timeout_s
        )
        self.profiler_uds_path = profiler_uds_path
        self._profiler_synced_tokens: "set[str]" = set()
        self._attempt_token_lock = threading.Lock()
        self._active_attempt_token: "str | None" = None

    def register_attempt_token(self, token: str) -> None:
        """Activate exactly one credential for the current daemon attempt."""
        if not isinstance(token, str) or not token:
            raise ValueError("attempt token must be a non-empty string")
        with self._attempt_token_lock:
            if self._active_attempt_token is not None:
                raise RuntimeError("another harness attempt token is already active")
            self._active_attempt_token = token

    def clear_attempt_token(self, token: str) -> bool:
        """Revoke only the matching credential, making stale cleanup harmless."""
        if not isinstance(token, str) or not token:
            return False
        with self._attempt_token_lock:
            active = self._active_attempt_token
            if active is None or not self._attempt_tokens_match(active, token):
                return False
            self._active_attempt_token = None
            self._profiler_synced_tokens.discard(active)
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
        return {ATTEMPT_TOKEN_HEADER: token}

    def resolve_model(self, token: str) -> str:
        resp = self.client.get("/llm/resolve_model", headers=self._attempt_headers(token))
        resp.raise_for_status()
        return resp.json()["model"]

    def context_limit(self, token: str) -> "int | None":
        """This agent's model's context window, for a caller that runs its
        own ReAct loop and needs to know when to compact (native_harness's
        `compaction.py`, mirroring the old `_native_in_container_
        entrypoint.py`'s `_fetch_context_limit`). Returns None on any
        failure -- compaction just never triggers in that case, the same
        graceful-when-unknown behavior `native_harness/compaction.py`'s
        `maybe_compact()` already has."""
        try:
            resp = self.client.get("/llm/context_limit", headers=self._attempt_headers(token))
            if resp.status_code != 200:
                return None
            return resp.json().get("context_limit")
        except Exception:
            return None

    def checkpoint(
        self,
        token: str,
        boundary_id: str,
        *,
        allow_messages: bool,
        phase: str,
    ) -> dict:
        """Wait at an invocation-bound host safe boundary."""
        response = self.client.post(
            "/interaction/checkpoint",
            json={
                "boundary_id": boundary_id,
                "allow_messages": allow_messages,
                "phase": phase,
            },
            headers=self._attempt_headers(token),
            # A pause intentionally outlives the ordinary LLM transport timeout.
            timeout=None,
        )
        response.raise_for_status()
        return response.json()

    async def checkpoint_async(
        self,
        token: str,
        boundary_id: str,
        *,
        allow_messages: bool,
        phase: str,
    ) -> dict:
        """Wait at a checkpoint on a cancellable per-request UDS client.

        The native harness reaches this method through an async sandbox route.
        Cancelling that route closes only this request's UDS connection, which
        lets the host observe ``http.disconnect`` without disturbing concurrent
        LLM, MCP, profiler, or policy traffic on the shared synchronous client.
        """
        async with self._new_checkpoint_async_client() as client:
            response = await client.post(
                "/interaction/checkpoint",
                json={
                    "boundary_id": boundary_id,
                    "allow_messages": allow_messages,
                    "phase": phase,
                },
                headers=self._attempt_headers(token),
                timeout=None,
            )
            response.raise_for_status()
            return response.json()

    def _new_checkpoint_async_client(self) -> httpx.AsyncClient:
        transport = httpx.AsyncHTTPTransport(uds=self._uds_path)
        return httpx.AsyncClient(
            transport=transport,
            base_url="http://agmanager-host",
            timeout=self._timeout_s,
        )

    def log_warning(self, token: str, message: str) -> None:
        # DATACOLLECTOR: append -- the one existing production call already wired through
        # record_event(type="warning"); model other emission points after this shape.
        self.client.post(
            "/interaction/record_event",
            json={"type": "warning", "payload": {"message": message}},
            headers=self._attempt_headers(token),
        )

    def check_tool_policy(self, token: str, tool_name: str, tool_input: dict) -> dict:
        resp = self.client.post(
            "/interaction/check_tool",
            json={"tool_name": tool_name, "tool_input": tool_input},
            headers=self._attempt_headers(token),
        )
        resp.raise_for_status()
        result = resp.json()
        return {
            "decision": "allow" if result.get("allowed") else "deny",
            "reason": result.get("reason"),
            "call_id": result.get("call_id"),
        }

    def complete_tool_policy(
        self, token: str, call_id: str, result: object = None, error: "str | None" = None
    ) -> None:
        response = self.client.post(
            "/interaction/complete_tool",
            json={"call_id": call_id, "result": result, "error": error},
            headers=self._attempt_headers(token),
        )
        response.raise_for_status()

    def check_syscall_policy(
        self, token: str, syscall: "agsyscallevent"
    ) -> "tuple[bool, str | None, str | None]":
        response = self.client.post(
            "/interaction/check_syscall",
            json=asdict(syscall),
            headers=self._attempt_headers(token),
        )
        response.raise_for_status()
        result = response.json()
        return bool(result.get("allowed")), result.get("reason"), result.get("call_id")

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

    def forward_profiler_event(self, token: str, event: dict) -> dict:
        if not self.validate_token(token):
            return {"ok": False, "error": "unknown or missing token"}
        if self.profiler_uds_path is None:
            return {"ok": False, "error": "no profiler bridge configured for this launch"}
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        try:
            sock.connect(self.profiler_uds_path)
            # Synchronize this container's clock against the host's, once
            # per token -- same handshake the old agproxy_llm.py's
            # `_forward_profiler_hook` did, needed so a hook's captured
            # in-container timestamp can be translated into the host's
            # clock domain before it's used as a span boundary.
            if token not in self._profiler_synced_tokens:
                wall_0 = time.time_ns()
                perf_0 = time.perf_counter_ns()
                _send_framed(
                    sock, {"token": token, "ev": "clock_sync", "wall_ns": wall_0, "perf_ns": perf_0}
                )
                sync = _recv_framed(sock)
                wall_1 = time.time_ns()
                perf_1 = time.perf_counter_ns()
                if not sync.get("ok"):
                    return sync
                _send_framed(
                    sock,
                    {
                        "token": token,
                        "ev": "clock_offset",
                        "wall_offset_ns": int(sync["host_wall_ns"] - (wall_0 + wall_1) / 2),
                        "perf_offset_ns": int(sync["host_perf_ns"] - (perf_0 + perf_1) / 2),
                    },
                )
                offset = _recv_framed(sock)
                if not offset.get("ok"):
                    return offset
                self._profiler_synced_tokens.add(token)
            _send_framed(sock, {**event, "token": token})
            return _recv_framed(sock)
        finally:
            sock.close()

    def close(self) -> None:
        with self._attempt_token_lock:
            self._active_attempt_token = None
            self._profiler_synced_tokens.clear()
        self.client.close()


__all__ = ["HostServicesClient"]
