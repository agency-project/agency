"""LLM dispatch for the standalone native harness.

Speaks one fixed wire format -- OpenAI-compatible chat-completions --
against a configurable `base_url` + API key. This is what makes "bridged"
and "fully standalone" the same code path with zero special-casing:
- Bridged (launched by agency): `base_url` points at this run's own
  `agmanager_harness` instance (`<bridge-base-url>/v1/chat/completions`,
  see that package's `llm_routing.py`), `api_key` is the per-launch bearer
  token. Every dispatch lands on `agmanager_host`'s `/internal/dispatch`,
  which records this token's running transcript on every turn, so live
  visibility needs no special streaming-output CLI flag.
- Standalone: `base_url`/`api_key` point at a real OpenAI-compatible
  provider directly.

**Always dispatches with `stream=True` and reassembles client-side, never
`stream=False`** -- ported from the old `_native_in_container_entrypoint.py`'s
`_dispatch_via_terminus`, which discovered a real, confirmed gap: some
backends' non-streaming code path (Anthropic/Bedrock's compatibility shim,
reached when this loop is bridged through agency to one of those backends)
doesn't reliably support tool calls at all. Streaming is the path every
backend actually supports fully, so this loop takes it unconditionally
rather than needing two code paths.

Retry policy: this loop owns its own bounded retry (503 / connection
failure only, never after a chunk has already been reassembled -- retrying
past that point would silently corrupt the conversation) -- same reasoning
as the old entrypoint's identical retry, needed because there is no
harness CLI underneath THIS loop the way there is for Claude Code/Codex
(who have their own resilience); when bridged, `agmanager_host`'s own
dispatch route deliberately makes exactly one attempt and classifies
failures for exactly this reason (see that module's docstring)."""

from __future__ import annotations

import json
import random
import time

import httpx

_DISPATCH_MAX_RETRIES = 4
_DISPATCH_BASE_BACKOFF_S = 0.5
_DISPATCH_MAX_BACKOFF_S = 8.0


def _retry_backoff_s(attempt: int) -> float:
    return random.uniform(0, min(_DISPATCH_MAX_BACKOFF_S, _DISPATCH_BASE_BACKOFF_S * (2**attempt)))


class LLMClient:
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 300) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    def dispatch(self, model: str, messages: list, tools: "list[dict] | None" = None) -> dict:
        """Returns `{"message": {...}, "usage": {...} | None}` on success,
        `{"error": "..."}` on failure (exhausted retries or a non-retryable
        status)."""
        kwargs = {"model": model, "messages": messages, "stream": True}
        if tools:
            kwargs["tools"] = tools

        last_error = "dispatch failed with no attempts made"
        for attempt in range(_DISPATCH_MAX_RETRIES):
            content_parts: "list[str]" = []
            tool_calls_raw: "dict[int, dict]" = {}
            usage: "dict | None" = None
            try:
                with self._client.stream("POST", "/v1/chat/completions", json=kwargs) as resp:
                    if resp.status_code == 503:
                        resp.read()
                        last_error = f"dispatch failed: {resp.status_code} {resp.text}"
                        if attempt < _DISPATCH_MAX_RETRIES - 1:
                            time.sleep(_retry_backoff_s(attempt))
                            continue
                        return {"error": last_error}
                    if resp.status_code != 200:
                        resp.read()
                        return {"error": f"dispatch failed: {resp.status_code} {resp.text}"}

                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        payload = line[len("data: ") :]
                        if payload == "[DONE]":
                            break
                        chunk = json.loads(payload)
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                        for tc_delta in delta.get("tool_calls") or []:
                            idx = tc_delta.get("index", 0)
                            slot = tool_calls_raw.setdefault(
                                idx,
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            if tc_delta.get("id"):
                                slot["id"] = tc_delta["id"]
                            fn_delta = tc_delta.get("function") or {}
                            if fn_delta.get("name"):
                                slot["function"]["name"] += fn_delta["name"]
                            if fn_delta.get("arguments"):
                                slot["function"]["arguments"] += fn_delta["arguments"]

                message = {"role": "assistant", "content": "".join(content_parts) or None}
                if tool_calls_raw:
                    message["tool_calls"] = [tool_calls_raw[i] for i in sorted(tool_calls_raw)]
                return {"message": message, "usage": usage}

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = f"llm endpoint unreachable: {e}"
                if attempt < _DISPATCH_MAX_RETRIES - 1:
                    time.sleep(_retry_backoff_s(attempt))
                    continue
                return {"error": last_error}

        return {"error": last_error}


__all__ = ["LLMClient"]
