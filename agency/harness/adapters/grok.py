"""Grok Build interactive PTY adapter and model-protocol translation."""

from __future__ import annotations

import json
import shutil
import time
import uuid

from fastapi import Request

from .agharness_backend import AdapterRuntime, AttemptResult, agharness_backend
from ..common import extract_bearer_token
from .pty_drivers import run_pty_attempt
from .pty_session import stream_response


def grok_available() -> bool:
    return shutil.which("grok") is not None


_STOP_REASON_TO_OPENAI = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "pause_turn": "stop",
}


def _stop_reason_to_openai(stop_reason: "str | None") -> "str | None":
    return _STOP_REASON_TO_OPENAI.get(stop_reason, stop_reason)


_CHATCOMPLETIONS_TYPE_PREFIX = "openai_chatcompletions_"


def _chatcompletions_native_block_type(native_type: str) -> str:
    return f"{_CHATCOMPLETIONS_TYPE_PREFIX}{native_type}"


def _flatten_unknown_data(data):
    fragments = data if isinstance(data, list) else [data]
    merged = None
    for fragment in fragments:
        if isinstance(fragment, dict) and ("start" in fragment or "deltas" in fragment):
            flat = dict(fragment.get("start") or {})
            for delta in fragment.get("deltas") or []:
                if not isinstance(delta, dict):
                    continue
                for k, v in delta.items():
                    if isinstance(v, str) and isinstance(flat.get(k), str):
                        flat[k] += v
                    else:
                        flat[k] = v
            fragment = flat
        if merged is None:
            merged = fragment
        elif isinstance(merged, str) and isinstance(fragment, str):
            merged += fragment
        else:
            merged = fragment
    return merged


def _toml_string(value: str) -> str:
    """Quote a string for a hand-written TOML file -- only the escapes
    this module's config values actually need."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class _GrokBackend(agharness_backend):
    _DEFAULT_BINARY = "grok"
    _MODEL_NAME = "agency-proxy"

    def run_daemon_attempt(
        self,
        runtime: AdapterRuntime,
        *,
        prompt: str,
        resume_session_id: "str | None",
        prior_session_blob: "bytes | None",
        max_steps: "int | None",
    ) -> AttemptResult:
        return run_pty_attempt(
            self,
            runtime,
            prompt=prompt,
            resume_session_id=resume_session_id,
            prior_session_blob=prior_session_blob,
            max_steps=max_steps,
        )

    def _write_grok_config(self, config_home, base_url: str, token: str, model: str) -> None:
        # config.toml, per docs.x.ai/build's configuration guide: a
        # [model.<name>] block with base_url/api_key/api_backend, and a
        # [models] table selecting the default model -- api_backend =
        # "chat_completions" is what makes this usable via agproxy_llm's
        # existing passthrough route with no translation, the same as
        # opencode's @ai-sdk/openai-compatible provider.
        config_toml = (
            f"[models]\n"
            f"default = {_toml_string(self._MODEL_NAME)}\n\n"
            f"[model.{self._MODEL_NAME}]\n"
            f"model = {_toml_string(model)}\n"
            f"base_url = {_toml_string(f'{base_url}/v1')}\n"
            f"api_key = {_toml_string(token)}\n"
            f'api_backend = "chat_completions"\n'
        )
        (config_home / "config.toml").write_text(config_toml)

    def register(self, app, router) -> None:
        from fastapi.responses import JSONResponse, StreamingResponse

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            token = extract_bearer_token(request)
            if not token or not router.validate_token(token):
                return JSONResponse(
                    {"error": {"message": "unknown or missing bearer token"}}, status_code=401
                )
            body = await request.json()
            model = router.resolve_model(token)
            # Grok requests UI titles and dashboard text separately from turns.
            # Like Claude's title request, these must not consume invocation
            # model calls or acknowledge redirects meant for the real turn.
            is_title = any(
                message.get("role") == "system"
                and isinstance(message.get("content"), str)
                and message["content"].startswith(
                    "You are tasked with generating the session title."
                )
                and "Just generate the session_title and nothing else" in message["content"]
                for message in body.get("messages", [])
            )
            messages = body.get("messages", [])
            last = messages[-1] if messages else {}
            is_dashboard = (
                last.get("role") == "user"
                and isinstance(last.get("content"), str)
                and last["content"].startswith(
                    "<system-reminder>Write an ultra-short dashboard line that captures "
                    "the AGENT'S REPLY for the last turn only"
                )
                and last["content"].endswith("</system-reminder>")
            )
            if is_title or is_dashboard:
                title_response = {
                    "type": "done",
                    "message": {
                        "role": "assistant",
                        "blocks": [
                            {
                                "type": "text",
                                "index": 0,
                                "text": "Agency session" if is_title else "Agency turn",
                            }
                        ],
                    },
                    "stop_reason": "stop",
                    "usage": {},
                }
                if body.get("stream"):
                    return StreamingResponse(
                        self._format_agency_stream_to_harness(iter([title_response]), model),
                        media_type="text/event-stream",
                    )
                return JSONResponse(self._format_context_agency_to_harness(title_response, model))
            agency_context = self._format_context_harness_to_agency(body)
            if body.get("stream"):
                return StreamingResponse(
                    stream_response(
                        router, token, agency_context, model, self._format_agency_stream_to_harness
                    ),
                    media_type="text/event-stream",
                )
            agency_response = router.dispatch(token, agency_context)
            return JSONResponse(self._format_context_agency_to_harness(agency_response, model))

    def _format_context_harness_to_agency(self, raw_request: dict) -> dict:
        messages: "list[dict]" = []
        for m in raw_request.get("messages", []):
            role = m.get("role")
            if role == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "blocks": [
                            {
                                "type": "tool_result",
                                "index": 0,
                                "tool_call_id": m.get("tool_call_id", ""),
                                "text": m.get("content") or "",
                            }
                        ],
                    }
                )
                continue

            blocks: "list[dict]" = []
            content = m.get("content")
            if isinstance(content, str):
                if content:
                    blocks.append({"type": "text", "index": len(blocks), "text": content})
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type")
                    if ptype == "text":
                        blocks.append(
                            {"type": "text", "index": len(blocks), "text": part.get("text", "")}
                        )
                    else:
                        blocks.append(
                            {
                                "type": _chatcompletions_native_block_type(ptype),
                                "index": len(blocks),
                                "data": part,
                            }
                        )
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "index": len(blocks),
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}"),
                    }
                )
            function_call = m.get("function_call")
            if function_call:
                blocks.append(
                    {
                        "type": "tool_use",
                        "index": len(blocks),
                        "id": "",
                        "name": function_call.get("name", ""),
                        "arguments": function_call.get("arguments", "{}"),
                    }
                )
            refusal = m.get("refusal")
            if refusal:
                blocks.append(
                    {
                        "type": _chatcompletions_native_block_type("refusal"),
                        "index": len(blocks),
                        "data": refusal,
                    }
                )
            messages.append({"role": role, "blocks": blocks})

        return {
            "messages": messages,
            "tools": raw_request.get("tools"),
            "tool_choice": raw_request.get("tool_choice"),
        }

    def _format_context_agency_to_harness(self, agency_response: dict, model: str) -> dict:
        message = agency_response["message"]
        text_parts = []
        tool_calls = []
        reasoning_parts = []
        for b in message.get("blocks", []):
            if b["type"] == "text":
                text_parts.append(b["text"])
            elif b["type"] == "thinking":
                reasoning_parts.append(b["text"])
            elif b["type"] == "tool_use":
                tool_calls.append(
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {"name": b["name"], "arguments": b["arguments"]},
                    }
                )

        response_message = {"role": "assistant", "content": "".join(text_parts) or None}
        if tool_calls:
            response_message["tool_calls"] = tool_calls
        if reasoning_parts:
            response_message["reasoning_content"] = "".join(reasoning_parts)
        for b in message.get("blocks", []):
            if b["type"].startswith(_CHATCOMPLETIONS_TYPE_PREFIX):
                field = b["type"][len(_CHATCOMPLETIONS_TYPE_PREFIX) :]
                response_message[field] = _flatten_unknown_data(b.get("data"))

        usage = agency_response.get("usage") or {}
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "finish_reason": _stop_reason_to_openai(agency_response.get("stop_reason")),
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

    def _format_agency_stream_to_harness(self, agency_stream, model: str):
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        def _chunk(delta: dict, finish_reason: "str | None" = None) -> str:
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(payload)}\n\n"

        for item in agency_stream:
            if item["type"] == "delta":
                # A redirect can replace this draft. SSE cannot retract text;
                # publish the authoritative message after the host checkpoint.
                continue

            tool_call_index = 0
            for b in item["message"].get("blocks", []):
                if b["type"] == "text":
                    yield _chunk({"content": b.get("text", "")})
                elif b["type"] == "tool_use":
                    yield _chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": tool_call_index,
                                    "id": b["id"],
                                    "type": "function",
                                    "function": {
                                        "name": b["name"],
                                        "arguments": b.get("arguments", "{}"),
                                    },
                                }
                            ]
                        }
                    )
                    tool_call_index += 1
                elif b["type"] == "thinking":
                    yield _chunk({"reasoning_content": b["text"]})
                elif b["type"].startswith(_CHATCOMPLETIONS_TYPE_PREFIX):
                    field = b["type"][len(_CHATCOMPLETIONS_TYPE_PREFIX) :]
                    yield _chunk({field: _flatten_unknown_data(b.get("data"))})

            yield _chunk({}, finish_reason=_stop_reason_to_openai(item.get("stop_reason")))

            usage = item.get("usage") or {}
            usage_payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }
            yield f"data: {json.dumps(usage_payload)}\n\n"
            yield "data: [DONE]\n\n"
            return


__all__ = ["_GrokBackend", "grok_available"]
