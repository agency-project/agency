"""This agent's one bridged connection to its `agmanager_host` instance.

Everything the container-side process cannot decide on its own (real LLM
dispatch, policy decisions, logging, pause/inbox state) goes through
`_HostBridge`, never a second, separately-credentialed path. Used by
`llm_routing.py`, `hooks_bridge.py`, and `mcp_proxy.py` -- see
`agmanager_harness.py`'s module docstring for the full design."""

from __future__ import annotations

import json
import socket
import struct
import time
import uuid

import httpx
from openai.types.chat import ChatCompletion, ChatCompletionChunk


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


class _HostBridge:
    """Thin client wrapping this agent's one bridged connection to its
    `agmanager_host` instance."""

    def __init__(
        self, uds_path: str, profiler_uds_path: "str | None", timeout_s: float = 300
    ) -> None:
        transport = httpx.HTTPTransport(uds=uds_path)
        self.client = httpx.Client(
            transport=transport, base_url="http://agmanager-host", timeout=timeout_s
        )
        self.profiler_uds_path = profiler_uds_path
        self._profiler_synced_tokens: "set[str]" = set()

    def validate_token(self, token: str) -> bool:
        # The host UDS is private to one agent. Bearer-token validation is
        # therefore owned by the sandbox-side daemon, not a host routing
        # registry shared by multiple agents.
        return bool(token)

    def resolve_model(self, token: str) -> str:
        resp = self.client.get("/llm/resolve_model")
        resp.raise_for_status()
        return resp.json()["model"]

    def context_limit(self, token: str) -> "int | None":
        """This agent's model's context window, for a caller that runs its
        own ReAct loop and needs to know when to compact (native_harness's
        `compaction.py`, mirroring the old `_native_in_container_
        entrypoint.py`'s `_fetch_context_limit`). Returns None on any
        failure -- compaction just never triggers in that case, same
        graceful-when-unknown behavior `agllm.py`'s own `maybe_compact()`
        already has."""
        try:
            resp = self.client.get("/llm/context_limit")
            if resp.status_code != 200:
                return None
            return resp.json().get("context_limit")
        except Exception:
            return None

    def log_warning(self, token: str, message: str) -> None:
        self.client.post(
            "/interaction/record_event",
            json={"type": "warning", "payload": {"message": message}},
        )

    def check_tool_policy(self, token: str, tool_name: str, tool_input: dict) -> dict:
        resp = self.client.post(
            "/interaction/check_tool",
            json={"tool_name": tool_name, "tool_input": tool_input},
        )
        result = resp.json()
        return {
            "decision": "allow" if result.get("allowed") else "deny",
            "reason": result.get("reason"),
        }

    def dispatch(self, token: str, kwargs: dict):
        """Non-streaming: returns a real `ChatCompletion`. Streaming:
        returns a generator of `ChatCompletionChunk` -- same contract the
        old `agproxy_llm.py`'s `_dispatch()` gave its own adapters, so
        `agproxy_llm_adapters.py`'s functions work unchanged here."""
        if kwargs.get("stream"):

            def gen():
                completion_id = f"chatcmpl_{uuid.uuid4().hex}"
                with self.client.stream("POST", "/llm/dispatch", json=kwargs) as resp:
                    if resp.status_code != 200:
                        resp.read()
                        raise RuntimeError(f"host dispatch failed: {resp.status_code} {resp.text}")
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        item = json.loads(line)
                        if item["type"] == "error":
                            raise RuntimeError(f"host dispatch failed: {item['message']}")
                        if item["type"] == "delta":
                            yield ChatCompletionChunk.model_validate(
                                {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": kwargs.get("model", ""),
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": item["content"]},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                            continue
                        if item["type"] == "done":
                            message = item.get("message") or {}
                            delta = {}
                            if message.get("tool_calls"):
                                delta["tool_calls"] = [
                                    {"index": index, **tool_call}
                                    for index, tool_call in enumerate(message["tool_calls"])
                                ]
                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": kwargs.get("model", ""),
                                "choices": [{"index": 0, "delta": delta, "finish_reason": "stop"}],
                            }
                            if item.get("usage") is not None:
                                chunk["usage"] = item["usage"]
                            yield ChatCompletionChunk.model_validate(chunk)
                            return

            return gen()

        resp = self.client.post("/llm/dispatch", json=kwargs)
        if resp.status_code != 200:
            raise RuntimeError(f"host dispatch failed: {resp.status_code} {resp.text}")
        result = resp.json()
        completion = {
            "id": f"chatcmpl_{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": kwargs.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": result["message"],
                    "finish_reason": result.get("stop_reason"),
                }
            ],
        }
        if result.get("usage") is not None:
            completion["usage"] = result["usage"]
        return ChatCompletion.model_validate(completion)

    def forward_profiler_event(self, token: str, event: dict) -> dict:
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


__all__ = ["_HostBridge"]
