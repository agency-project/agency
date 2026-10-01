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
`stream=False`**: some backends' non-streaming code path (Anthropic/
Bedrock's compatibility shim, reached when this loop is bridged through
agency to one of those backends) doesn't reliably support tool calls at
all. Streaming is the path every backend actually supports fully, so this
loop takes it unconditionally rather than needing two code paths.

Retry policy: this loop owns its own bounded retry (503 / connection
failure only, never after a chunk has already been reassembled -- retrying
past that point would silently corrupt the conversation), needed because
there is no harness CLI underneath THIS loop the way there is for Claude
Code/Codex (who have their own resilience); when bridged, `agmanager_host`'s
own dispatch route deliberately makes exactly one attempt and classifies
failures for exactly this reason (see that module's docstring)."""

from __future__ import annotations

import json
import random
import time

import httpx
from .profiling import span as profile_span

_DISPATCH_MAX_RETRIES = 4
_DISPATCH_BASE_BACKOFF_S = 0.5
_DISPATCH_MAX_BACKOFF_S = 8.0


def _retry_backoff_s(attempt: int) -> float:
    return random.uniform(0, min(_DISPATCH_MAX_BACKOFF_S, _DISPATCH_BASE_BACKOFF_S * (2**attempt)))


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: float = 300,
        *,
        model_settings: dict | None = None,
        observer=None,
    ) -> None:
        self.model_settings = dict(model_settings or {})
        self.observer = observer
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    def dispatch(
        self,
        model: str,
        messages: list,
        tools: "list[dict] | None" = None,
        *,
        internal_kind: "str | None" = None,
    ) -> dict:
        """Returns `{"message": {...}, "usage": {...} | None}` on success,
        `{"error": "..."}` on failure (exhausted retries or a non-retryable
        status)."""
        kwargs = {**self.model_settings, "model": model, "messages": messages, "stream": True}
        if tools:
            kwargs["tools"] = tools
        if internal_kind is not None:
            kwargs["agency_internal_kind"] = internal_kind

        last_error = "dispatch failed with no attempts made"
        for attempt in range(_DISPATCH_MAX_RETRIES):
            if self.observer is not None:
                self.observer("model_attempt", {"attempt": attempt, "internal_kind": internal_kind})
            content_parts: "list[str]" = []
            tool_calls_raw: "dict[int, dict]" = {}
            usage: "dict | None" = None
            finish_reason = None
            try:
                with self._client.stream("POST", "/v1/chat/completions", json=kwargs) as resp:
                    if resp.status_code == 503:
                        resp.read()
                        last_error = f"dispatch failed: {resp.status_code} {resp.text}"
                        if attempt < _DISPATCH_MAX_RETRIES - 1:
                            with profile_span(self, f"llm:retry_backoff[{attempt}]"):
                                time.sleep(_retry_backoff_s(attempt))
                            continue
                        return self._dispatch_error(last_error, attempt + 1, internal_kind)
                    if resp.status_code != 200:
                        resp.read()
                        return self._dispatch_error(
                            f"dispatch failed: {resp.status_code} {resp.text}",
                            attempt + 1,
                            internal_kind,
                        )

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
                        finish_reason = choices[0].get("finish_reason") or finish_reason
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
                result = {"message": message, "usage": usage}
                if self.observer is not None:
                    self.observer(
                        "model_dispatch_complete",
                        {
                            "usage": usage,
                            "finish_reason": finish_reason,
                            "attempts": attempt + 1,
                            "internal_kind": internal_kind,
                        },
                    )
                return result

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = f"llm endpoint unreachable: {e}"
                if attempt < _DISPATCH_MAX_RETRIES - 1:
                    with profile_span(self, f"llm:retry_backoff[{attempt}]"):
                        time.sleep(_retry_backoff_s(attempt))
                    continue
                return self._dispatch_error(last_error, attempt + 1, internal_kind)

        return self._dispatch_error(last_error, _DISPATCH_MAX_RETRIES, internal_kind)

    def _dispatch_error(self, error, attempts, internal_kind):
        if self.observer is not None:
            self.observer(
                "model_dispatch_complete",
                {
                    "usage": None,
                    "finish_reason": "error",
                    "attempts": attempts,
                    "internal_kind": internal_kind,
                },
            )
        return {"error": error}

    def close(self) -> None:
        self._client.close()


__all__ = ["LLMClient"]
