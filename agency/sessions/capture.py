"""Opt-in, wire-preserving HTTP capture for the real interactive CLIs.

The gateway never translates, retries, fits, or edits a provider request. Raw
bodies are authoritative; Agency blocks are an additional inspection projection.
LiteLLM mode replaces authentication; native capture forwards the CLI's own
authentication to its original provider. Credentials are never archived.
"""

from __future__ import annotations

import asyncio
import copy
import gzip
import hashlib
import json
import math
import os
import re
import ssl
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import anyio
import httpx
from fastapi import Request, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.websockets import WebSocketDisconnect
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

from agency.llm.message_store import MessageStore, with_tool_schema

_PRIVATE_HEADERS = {
    "authorization",
    "x-api-key",
    "api-key",
    "cookie",
    "set-cookie",
    "x-agency-token",
}
_HOP_HEADERS = {
    "host",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
}
_PATH = re.compile(
    r"v1/(?:messages(?:/count_tokens)?|responses(?:/compact|/[A-Za-z0-9_-]+(?:/cancel)?)?|models)"
)


@dataclass(frozen=True)
class CaptureConfig:
    model: str
    base_url: str = "https://router.js-park.info"
    key_env: str = "LITE_LLM_KEY"
    native = False

    def __post_init__(self):
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("capture upstream must be an HTTP(S) URL without credentials")
        if url.query or url.fragment or not self.model or not self.key_env:
            raise ValueError("invalid capture configuration")


@dataclass(frozen=True)
class NativeCaptureConfig:
    """Fixed upstreams, selected by Codex's native account header, never a URL
    supplied by an incoming request. Loopback HTTP is useful for offline tests.
    Login, token persistence and refresh remain entirely owned by Codex.
    """

    api_base_url: str = "https://api.openai.com/v1"
    chatgpt_base_url: str = "https://chatgpt.com/backend-api/codex"
    native = True
    harness = "codex"
    model = None
    key_env = None

    def __post_init__(self):
        for base in (self.api_base_url, self.chatgpt_base_url):
            _validate_native_upstream(base)

    def upstream(self, headers, path, query):
        base = self.chatgpt_base_url if headers.get("chatgpt-account-id") else self.api_base_url
        return base.rstrip("/") + "/" + path.removeprefix("v1/") + ("?" + query if query else "")


def _validate_native_upstream(base):
    if not isinstance(base, str):
        raise ValueError("native capture upstream must be a URL string")
    url = urlsplit(base)
    if (
        not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or not (
            url.scheme == "https"
            or url.scheme == "http"
            and url.hostname in {"127.0.0.1", "::1", "localhost"}
        )
    ):
        raise ValueError(
            "native capture upstream must be HTTPS without credentials (or loopback HTTP)"
        )
    # Accessing port also validates malformed or out-of-range authorities.
    _ = url.port


def _headers(headers, *, archive=False):
    excluded = _HOP_HEADERS | (_PRIVATE_HEADERS if archive else set())
    # Connection can nominate additional hop-by-hop headers.
    excluded = excluded | {s.strip().lower() for s in headers.get("connection", "").split(",")}
    pairs = [(k.decode("latin-1"), v.decode("latin-1")) for k, v in headers.raw]
    return [(k, v) for k, v in pairs if k.lower() not in excluded]


def _private_file(path: Path, mode="wb"):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(fd, mode)


def _decode(body: bytes, encoding: str):
    if encoding.lower() == "gzip":
        body = gzip.decompress(body)
    elif encoding.lower() == "deflate":
        body = zlib.decompress(body)
    elif encoding and encoding.lower() != "identity":
        raise ValueError("unsupported capture projection encoding")
    return body


def _events(raw: bytes):
    """SSE framing, including CRLF, multiline data, and split UTF-8 bytes."""
    # A cancelled stream may end inside a UTF-8 character. Earlier complete
    # frames still project; the authoritative bytes are never decoded/replaced.
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    for frame in text.split("\n\n"):
        data = "\n".join(
            line[5:].removeprefix(" ") for line in frame.splitlines() if line.startswith("data:")
        )
        if not data or data == "[DONE]":
            continue
        try:
            value = json.loads(data)
        except ValueError:
            continue  # Unparseable frames remain intact in response.bin.
        if isinstance(value, dict):
            yield value


def _response(raw: bytes, content_type: str, protocol: str, *, events=None):
    if events is None and "text/event-stream" not in content_type:
        value = json.loads(raw)
        failed = isinstance(value, dict) and (
            value.get("error") or value.get("status") in {"failed", "incomplete"}
        )
        return value, True, bool(failed)
    result, blocks, arguments = {}, {}, {}
    terminal = failed = False
    for event in _events(raw) if events is None else events:
        kind = event.get("type")
        if protocol == "responses":
            if isinstance(event.get("response"), dict):
                result = copy.deepcopy(event["response"])
            index = event.get("output_index", 0)
            if kind in {"response.output_item.added", "response.output_item.done"}:
                blocks[index] = copy.deepcopy(event["item"])
            elif kind == "response.output_text.delta":
                item = blocks.setdefault(
                    index, {"type": "message", "role": "assistant", "content": []}
                )
                content = item.setdefault("content", [])
                ci = event.get("content_index", 0)
                while len(content) <= ci:
                    content.append({"type": "output_text", "text": ""})
                content[ci]["text"] = content[ci].get("text", "") + event.get("delta", "")
            elif kind == "response.function_call_arguments.delta":
                arguments[index] = arguments.get(index, "") + event.get("delta", "")
                blocks.setdefault(index, {"type": "function_call"})["arguments"] = arguments[index]
            if kind in {
                "response.completed",
                "response.failed",
                "response.incomplete",
                "response.cancelled",
                "error",
            }:
                terminal = True
                failed = kind not in {"response.completed", "response.cancelled"}
                if kind == "error":
                    result["error"] = event.get("error", event)
                elif kind == "response.cancelled":
                    result["status"] = "cancelled"
        else:
            if kind == "message_start":
                result = copy.deepcopy(event.get("message", {}))
            elif kind == "content_block_start":
                blocks[event["index"]] = copy.deepcopy(event["content_block"])
            elif kind == "content_block_delta":
                index, delta = event["index"], event.get("delta", {})
                block = blocks.setdefault(index, {})
                for key in ("text", "thinking", "signature"):
                    if key in delta:
                        block[key] = block.get(key, "") + delta[key]
                if "partial_json" in delta:
                    arguments[index] = arguments.get(index, "") + delta["partial_json"]
            elif kind == "message_delta":
                result.update(event.get("delta", {}))
                result.setdefault("usage", {}).update(event.get("usage", {}))
            elif kind in {"message_stop", "error"}:
                terminal = True
                failed = kind == "error"
                if failed:
                    result["error"] = event.get("error", event)
    if protocol == "responses":
        if not result.get("output"):
            result["output"] = [blocks[i] for i in sorted(blocks)]
    else:
        for index, value in arguments.items():
            try:
                blocks[index]["input"] = json.loads(value)
            except ValueError:
                blocks[index]["partial_input_json"] = value
        result["content"] = [blocks[i] for i in sorted(blocks)] or result.get("content", [])
    return result, terminal, failed


def _content(content, protocol):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [] if content is None else [{"type": "capture_unknown", "data": content}]
    converted = []
    for block in content or []:
        if not isinstance(block, dict):
            converted.append({"type": "capture_unknown", "data": block})
            continue
        kind = block.get("type", "unknown")
        if kind in {"text", "input_text", "output_text"}:
            converted.append({**block, "type": "text"})
        elif kind == "tool_use":
            converted.append({**block, "arguments": json.dumps(block.get("input", {}))})
        elif kind == "tool_result":
            converted.append(
                {
                    **block,
                    "tool_call_id": block.get("tool_use_id"),
                    "text": block.get("content", ""),
                }
            )
        elif kind in {"thinking", "redacted_thinking"}:
            converted.append({**block, "text": block.get("thinking", "")})
        else:
            converted.append({"type": f"{protocol}_{kind}", "data": block})
    return converted


def _items(items):
    """Lossless native envelopes are stored separately; these are UI blocks."""
    if isinstance(items, str):
        return [{"role": "user", "blocks": [{"type": "text", "text": items}]}]
    if items is not None and not isinstance(items, list):
        return [{"role": "unknown", "blocks": [{"type": "capture_unknown", "data": items}]}]
    messages = []
    for item in items or []:
        if not isinstance(item, dict):
            messages.append(
                {"role": "unknown", "blocks": [{"type": "capture_unknown", "data": item}]}
            )
            continue
        kind = item.get("type", "message")
        if kind == "message":
            messages.append(
                {
                    **{k: value for k, value in item.items() if k not in {"role", "content"}},
                    "role": item.get("role", "user"),
                    "blocks": _content(item.get("content"), "openai_responses"),
                }
            )
        elif kind in {"function_call", "custom_tool_call"}:
            messages.append(
                {
                    "role": "assistant",
                    "blocks": [
                        {
                            "type": "tool_use",
                            "id": item.get("call_id", item.get("id")),
                            "name": item.get("name"),
                            "arguments": item.get("arguments", item.get("input", "")),
                        }
                    ],
                }
            )
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            messages.append(
                {
                    "role": "tool",
                    "blocks": [
                        {
                            "type": "tool_result",
                            "tool_call_id": item.get("call_id"),
                            "text": item.get("output", ""),
                        }
                    ],
                }
            )
        elif kind == "reasoning":
            messages.append(
                {
                    "role": "assistant",
                    "blocks": [
                        {
                            "type": "thinking",
                            "signature": item.get("encrypted_content"),
                            "data": item,
                        }
                    ],
                }
            )
        else:
            messages.append(
                {
                    "role": item.get("role", "assistant"),
                    "blocks": [{"type": f"openai_responses_{kind}", "data": item}],
                }
            )
    return messages


def _prompt(body, protocol):
    if protocol == "responses":
        messages = _items(body.get("input"))
        if body.get("instructions"):
            messages.insert(
                0, {"role": "developer", "blocks": _content(body["instructions"], protocol)}
            )
    else:
        native_messages = body.get("messages") or []
        if not isinstance(native_messages, list):
            native_messages = [{"role": "unknown", "content": native_messages}]
        messages = [
            {
                **{k: value for k, value in m.items() if k not in {"role", "content"}},
                "role": m.get("role", "user"),
                "blocks": _content(m.get("content"), "anthropic"),
            }
            for m in native_messages
            if isinstance(m, dict)
        ]
        if body.get("system"):
            messages.insert(0, {"role": "system", "blocks": _content(body["system"], "anthropic")})
    extra = {
        k: value
        for k, value in body.items()
        if k not in {"messages", "system", "input", "instructions", "tools", "tool_choice"}
    }
    tools = body.get("tools")
    if tools is not None and not isinstance(tools, list):
        extra["tools"], tools = tools, None
    return with_tool_schema(messages, tools, body.get("tool_choice"), extra)


class _Recording:
    def __init__(self, session, config, method, path, query, headers, raw, body):
        self.session, self.config, self.body = session, config, body
        self.model = body.get("model") or config.model
        self.actor = session._main
        self.label = uuid.uuid4().hex
        self.started = time.time()
        self.path = path
        self.protocol = "responses" if path.startswith("v1/responses") else "messages"
        self.is_model = method == "POST" and path in {
            "v1/messages",
            "v1/responses",
            "v1/responses/compact",
        }
        self.finished = False
        self.cancelled = False
        self.error = None
        self.status = None
        self.response_headers = httpx.Headers()
        self.frames = None
        self.frame_index = None
        self.directory = session.run_dir / "capture" / self.label
        self.directory.mkdir(parents=True, mode=0o700)
        (session.run_dir / "capture").chmod(0o700)
        self.manifest = {
            "schema_version": 1,
            "call_label": self.label,
            "protocol": self.protocol,
            "path": path,
            "method": method,
            "query": query,
            "request_headers": _headers(headers, archive=True),
            "started_at": self.started,
            "request_sha256": hashlib.sha256(raw).hexdigest(),
        }
        with _private_file(self.directory / "request.bin") as f:
            f.write(raw)
        self.output = _private_file(self.directory / "response.bin")
        self._save_manifest()
        if self.is_model:
            self.actor.logger.record_event(
                "agent_state",
                {"state": "waiting_llm", "model": self.model},
                call_label=self.label,
                timestamp=self.started,
                flush=True,
            )

    def _save_manifest(self):
        temporary = self.directory / "manifest.tmp"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(self.manifest, f)
        temporary.replace(self.directory / "manifest.json")

    def websocket_frame(self, data):
        raw = data.encode() if isinstance(data, str) else data
        frame = {
            "offset": self.output.tell(),
            "length": len(raw),
            "binary": isinstance(data, bytes),
        }
        self.frames.append(frame)
        self.output.write(raw)
        self.output.flush()
        if self.frame_index is None:
            self.frame_index = _private_file(self.directory / "response.frames.jsonl", "w")
        self.frame_index.write(json.dumps(frame) + "\n")
        self.frame_index.flush()

    def _websocket_events(self, raw):
        for frame in self.frames:
            try:
                value = json.loads(raw[frame["offset"] : frame["offset"] + frame["length"]])
            except (ValueError, UnicodeError):
                continue
            if isinstance(value, dict):
                yield value

    def finish(self):
        if self.finished:
            return
        self.finished = True
        self.output.close()
        if self.frame_index is not None:
            self.frame_index.close()
        ended = time.time()
        raw = (self.directory / "response.bin").read_bytes()
        response, terminal, failed = {}, False, False
        projection_error = None
        try:
            decoded = _decode(raw, self.response_headers.get("content-encoding", ""))
            if decoded:
                response, terminal, failed = _response(
                    decoded,
                    self.response_headers.get("content-type", ""),
                    self.protocol,
                    events=self._websocket_events(raw) if self.frames is not None else None,
                )
        except (
            ValueError,
            UnicodeError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
            zlib.error,
            OSError,
        ):
            projection_error = "unparsed_native_response"
        if failed:
            self.error = self.error or "provider_response_failed_or_incomplete"
        failed = failed or bool(self.status and self.status >= 400) or bool(self.error)
        if self.is_model and not terminal and not self.cancelled:
            failed = True
            self.error = self.error or "upstream_response_incomplete"
        native_response = response
        if not isinstance(response, dict):
            response = {}
            projection_error = "native_response_is_not_an_object"
        client_disconnected = self.cancelled
        # Native CLIs close HTTP as soon as the terminal SSE event arrives;
        # LiteLLM may still be sending trailing framing. That disconnect is
        # transport evidence, not cancellation of the completed model call.
        if terminal:
            self.cancelled = False
        inputs = self.body.get("input") or []
        inputs = inputs if isinstance(inputs, list) else []
        outputs = response.get("output") or []
        outputs = outputs if isinstance(outputs, list) else []
        if response.get("status") == "cancelled":
            self.cancelled = True
        coverage = {
            "complete": not bool(
                self.body.get("previous_response_id") or self.body.get("conversation")
            ),
            "previous_response_id": self.body.get("previous_response_id"),
            "server_state": bool(
                self.body.get("previous_response_id") or self.body.get("conversation")
            ),
            "compaction": self.path.endswith("/compact")
            or any(
                i.get("type") == "compaction" for i in [*inputs, *outputs] if isinstance(i, dict)
            ),
            "context_editing": False,
        }
        if coverage["compaction"]:
            coverage["complete"] = False
        self.manifest.update(
            ended_at=ended,
            status=self.status,
            response_headers=_headers(self.response_headers, archive=True),
            response_sha256=hashlib.sha256(raw).hexdigest(),
            response_bytes=len(raw),
            cancelled=self.cancelled,
            client_disconnected=client_disconnected,
            error=self.error,
            projection_error=projection_error,
            coverage=coverage,
        )
        if self.frames is not None:
            self.manifest.update(transport="websocket", response_frames=self.frames)
        self._save_manifest()
        if not self.is_model:
            self.actor.logger.record_event(
                "capture_auxiliary", {**self.manifest, "archive": str(self.directory)}, flush=True
            )
            return
        usage = response.get("usage") or {}
        usage = usage if isinstance(usage, dict) else {}
        if self.protocol == "messages":
            usage = {
                **usage,
                "prompt_tokens": sum(
                    usage[k]
                    if isinstance(usage.get(k), (int, float)) and math.isfinite(usage[k])
                    else 0
                    for k in (
                        "input_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    )
                ),
                "completion_tokens": usage.get("output_tokens"),
            }
            messages = [
                {"role": "assistant", "blocks": _content(response.get("content"), "anthropic")}
            ]
        else:
            usage = {
                **usage,
                "prompt_tokens": usage.get("input_tokens"),
                "completion_tokens": usage.get("output_tokens"),
            }
            messages = _items(response.get("output"))
        if not messages or messages[-1]["role"] != "assistant":
            messages.append({"role": "assistant", "blocks": []})
        blocks = messages[-1]["blocks"]
        metadata = {
            "type": "metadata",
            "model": self.model,
            "usage": usage,
            "request_id": self.response_headers.get("x-request-id")
            or self.response_headers.get("request-id"),
            "source": "interactive_capture",
            "capture": coverage,
            "data": {
                "request": self.body,
                "response": native_response,
                "archive": str(self.directory),
                "status": self.status,
                "projection_error": projection_error,
            },
            "attribution": "session_gateway; subagent identity not inferred",
        }
        blocks.append(metadata)
        if self.cancelled:
            blocks.append({"type": "capture_status", "cancelled": True})
        elif failed:
            blocks.append(
                {"type": "capture_status", "error": self.error or f"upstream HTTP {self.status}"}
            )
        store = MessageStore()
        prompt = _prompt(self.body, self.protocol)
        # Invalid provider fields must not prevent archiving a rejected request.
        # The native envelopes stay exact; only this inspection role is repaired.
        for message in [*prompt, *messages]:
            role = message.get("role")
            if not isinstance(role, str) or not role:
                message["capture_native_role"], message["role"] = role, "unknown"
        prompt_hashes = store.add_messages(prompt)
        response_hashes = store.add_messages(messages)
        self.actor.logger.record_llm_exchange(
            self.label,
            exchange_type="llm_stream_cancelled"
            if self.cancelled
            else "llm_stream_error"
            if failed
            else "llm_block",
            message_store=store,
            harness_message_hashes=prompt_hashes,
            patched_message_hashes=prompt_hashes,
            response_message_hashes=response_hashes,
            timestamp=ended,
            print_to_terminal=False,
        )
        self.actor.host.record_span(
            f"llm:{self.model}",
            self.started,
            ended,
            {
                "model": self.model,
                "provenance": "interactive_capture",
                "outcome": "cancelled" if self.cancelled else "failure" if failed else "success",
            },
            call_label=self.label,
            span_id=self.label,
        )
        # Auxiliary generations (for example a CLI's title request) can end
        # after its Stop hook. Restore the hook-owned lifecycle state instead
        # of leaving the column waiting for an already-finished model call.
        self.session._set_state(self.actor.state, self.actor)


class _CaptureResponse(StreamingResponse):
    def __init__(self, recording, upstream, client):
        self.recording, self.upstream, self.client = recording, upstream, client

        async def chunks():
            try:
                async for chunk in upstream.aiter_raw():
                    recording.output.write(chunk)
                    recording.output.flush()
                    yield chunk
            except httpx.HTTPError:
                recording.error = "upstream_stream_error"
                raise

        super().__init__(chunks(), status_code=upstream.status_code)
        self.raw_headers = [
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in _headers(upstream.headers)
        ]

    async def __call__(self, scope, receive, send):
        body_complete = False

        async def tracked_send(message):
            nonlocal body_complete
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                body_complete = True
            await send(message)

        try:
            # Watch disconnects while the upstream is silent, too. ASGI 2.4's
            # send-only detection otherwise waits for another upstream token.
            async with anyio.create_task_group() as group:

                async def stream():
                    try:
                        await self.stream_response(tracked_send)
                    except OSError:
                        self.recording.cancelled = True
                        raise
                    finally:
                        group.cancel_scope.cancel()

                group.start_soon(stream)
                await self.listen_for_disconnect(receive)
                self.recording.cancelled = not body_complete
                group.cancel_scope.cancel()
        except (asyncio.CancelledError, OSError):
            self.recording.cancelled = True
            raise
        finally:
            with anyio.CancelScope(shield=True):
                await self.upstream.aclose()
                await self.client.aclose()
                self.recording.finish()


class CaptureGateway:
    def __init__(self, session, config, *, tls_context=None):
        self.session, self.config = session, config
        self._key = os.environ.get(config.key_env) if config.key_env else None
        if not config.native and not self._key:
            raise ValueError(f"capture requires {config.key_env}")
        self._tls = tls_context
        if config.native and self._tls is None:
            try:
                self._tls = ssl.create_default_context(
                    cafile=os.environ.get("CODEX_CA_CERTIFICATE") or os.environ.get("SSL_CERT_FILE")
                )
            except OSError as exc:
                raise ValueError("cannot load the Codex capture CA bundle") from exc

    def _authorized(self, headers):
        token = (
            headers.get("x-agency-token")
            if self.config.native
            else (
                headers.get("authorization", "").removeprefix("Bearer ") or headers.get("x-api-key")
            )
        )
        return self.session._authorized(token)

    def _upstream(self, headers, path, query):
        if self.config.native:
            return self.config.upstream(headers, path, query)
        base = self.config.base_url.rstrip("/").removesuffix("/v1")
        return base + "/" + path + ("?" + query if query else "")

    def _forward_headers(self, headers):
        if self.config.native:
            return [(k, v) for k, v in _headers(headers) if k.lower() != "x-agency-token"]
        result = [
            (k, v)
            for k, v in _headers(headers)
            if k.lower() not in _PRIVATE_HEADERS | {"accept-encoding"}
        ]
        return result + [("authorization", "Bearer " + self._key), ("accept-encoding", "identity")]

    def _client_options(self, url):
        return (
            {"trust_env": True, "verify": self._tls} if self.config.native else {"trust_env": False}
        )

    async def forward(self, request: Request, path: str):
        if not self._authorized(request.headers):
            return JSONResponse({"error": {"message": "invalid capture token"}}, status_code=401)
        native_paths = (
            ("v1/messages",)
            if getattr(self.config, "harness", None) == "claude_code"
            else ("v1/responses", "v1/models")
        )
        if not _PATH.fullmatch(path) or self.config.native and not path.startswith(native_paths):
            return JSONResponse(
                {"error": {"message": "unsupported capture endpoint"}}, status_code=404
            )
        raw = await request.body()
        try:
            body = (
                json.loads(_decode(raw, request.headers.get("content-encoding", ""))) if raw else {}
            )
            if not isinstance(body, dict):
                raise ValueError("request must be an object")
        except (ValueError, UnicodeError, zlib.error, OSError):
            record = _Recording(
                self.session,
                self.config,
                request.method,
                path,
                request.url.query,
                request.headers,
                raw,
                {},
            )
            record.status, record.error = 400, "invalid_native_request"
            record.finish()
            return JSONResponse(
                {"error": {"message": "unsupported request encoding or JSON"}}, status_code=400
            )
        record = _Recording(
            self.session,
            self.config,
            request.method,
            path,
            request.url.query,
            request.headers,
            raw,
            body,
        )
        if (
            not self.config.native
            and (record.is_model or body.get("model"))
            and body.get("model") != self.config.model
        ):
            record.status, record.error = 400, "capture_model_mismatch"
            record.finish()
            return JSONResponse(
                {"error": {"message": "model differs from the configured capture model"}},
                status_code=400,
            )
        url = self._upstream(request.headers, path, request.url.query)
        headers = self._forward_headers(request.headers)
        if not self.config.native and path.startswith("v1/messages"):
            headers.append(("x-api-key", self._key))
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(1200, connect=10),
            follow_redirects=False,
            **self._client_options(url),
        )
        try:
            upstream, error = None, None
            async with anyio.create_task_group() as group:

                async def connect():
                    nonlocal upstream, error
                    try:
                        upstream = await client.send(
                            client.build_request(request.method, url, headers=headers, content=raw),
                            stream=True,
                        )
                    except httpx.HTTPError as exc:
                        error = exc
                    finally:
                        group.cancel_scope.cancel()

                group.start_soon(connect)
                while (await request.receive())["type"] != "http.disconnect":
                    pass
                record.cancelled = True
                group.cancel_scope.cancel()
            if error is not None:
                raise error
            if upstream is None:
                await client.aclose()
                record.finish()
                return JSONResponse(
                    {"error": {"message": "capture client disconnected"}}, status_code=499
                )
        except asyncio.CancelledError:
            record.cancelled = True
            with anyio.CancelScope(shield=True):
                await client.aclose()
                record.finish()
            raise
        except httpx.HTTPError:
            await client.aclose()
            record.status, record.error = 502, "capture_upstream_connection_error"
            record.finish()
            return JSONResponse(
                {"error": {"message": "capture upstream connection failed"}}, status_code=502
            )
        record.status, record.response_headers = upstream.status_code, upstream.headers
        return _CaptureResponse(record, upstream, client)

    async def websocket(self, downstream: WebSocket, path: str):
        """Relay native Responses messages without changing their JSON or opcode.

        Each create request has an exchange; response frames are concatenated
        byte-for-byte with offsets in its manifest. Unmatched protocol messages
        are auxiliary captures. Native token refresh/reconnect stays with Codex.
        """
        if (
            not self.config.native
            or getattr(self.config, "harness", None) != "codex"
            or not self._authorized(downstream.headers)
        ):
            await downstream.send_denial_response(Response(status_code=401))
            return
        if path != "v1/responses":
            await downstream.send_denial_response(Response(status_code=404))
            return
        url = self._upstream(downstream.headers, path, downstream.url.query)
        parts = urlsplit(url)
        url = parts._replace(scheme="wss" if parts.scheme == "https" else "ws").geturl()
        headers = [
            (k, v)
            for k, v in self._forward_headers(downstream.headers)
            if not k.lower().startswith("sec-websocket-")
        ]
        records, by_id = [], {}
        last_model = None
        auxiliary = None
        client_left = False

        def recording(raw, body, *, model_call):
            nonlocal last_model
            record = _Recording(
                self.session,
                self.config,
                "POST" if model_call else "WS",
                path,
                downstream.url.query,
                downstream.headers,
                raw,
                body,
            )
            record.frames = []
            record.manifest["transport"] = "websocket"
            record.manifest["method"] = "WS"
            record.status = 101
            record.response_headers = httpx.Headers(list(upstream.response.headers.raw_items()))
            last_model = body.get("model") or last_model
            record.model = record.model or last_model
            return record

        def event(data):
            try:
                value = json.loads(data)
                return value if isinstance(value, dict) else {}
            except (ValueError, UnicodeError):
                return {}

        try:
            async with websocket_connect(
                url,
                additional_headers=headers,
                proxy=True
                if urlsplit(url).hostname not in {"127.0.0.1", "::1", "localhost"}
                else None,
                max_size=None,
                subprotocols=downstream.scope.get("subprotocols") or None,
                **({"ssl": self._tls} if url.startswith("wss:") else {}),
            ) as upstream:
                response_headers = httpx.Headers(list(upstream.response.headers.raw_items()))
                await downstream.accept(
                    subprotocol=upstream.subprotocol,
                    headers=[
                        (k.lower().encode("latin-1"), v.encode("latin-1"))
                        for k, v in _headers(response_headers)
                        if not k.lower().startswith("sec-websocket-")
                    ],
                )
                async with anyio.create_task_group() as group:

                    async def client_messages():
                        nonlocal client_left
                        try:
                            while True:
                                message = await downstream.receive()
                                if message["type"] == "websocket.disconnect":
                                    client_left = True
                                    break
                                data = message.get("text")
                                data = data if data is not None else message["bytes"]
                                body = event(data)
                                raw = data.encode() if isinstance(data, str) else data
                                model_call = (
                                    body.get("type") == "response.create"
                                    and body.get("generate") is not False
                                )
                                record = recording(raw, body, model_call=model_call)
                                if body.get("type") == "response.create":
                                    records.append(record)
                                else:
                                    record.finish()
                                await upstream.send(data)
                        except WebSocketDisconnect:
                            client_left = True
                        except ConnectionClosed:
                            pass
                        finally:
                            group.cancel_scope.cancel()

                    async def provider_messages():
                        nonlocal auxiliary, client_left
                        try:
                            async for data in upstream:
                                value = event(data)
                                response = value.get("response")
                                response_id = (
                                    response.get("id") if isinstance(response, dict) else None
                                ) or value.get("response_id")
                                record = by_id.get(response_id) or next(
                                    (r for r in records if not r.finished), None
                                )
                                if record is None:
                                    if auxiliary is None:
                                        auxiliary = recording(b"", {}, model_call=False)
                                    record = auxiliary
                                elif response_id:
                                    by_id[response_id] = record
                                record.websocket_frame(data)
                                if value.get("type") in {
                                    "response.completed",
                                    "response.failed",
                                    "response.incomplete",
                                    "response.cancelled",
                                    "error",
                                }:
                                    record.finish()
                                    for key in [k for k, r in by_id.items() if r is record]:
                                        del by_id[key]
                                    if record in records:
                                        records.remove(record)
                                    if record is auxiliary:
                                        auxiliary = None
                                if isinstance(data, str):
                                    await downstream.send_text(data)
                                else:
                                    await downstream.send_bytes(data)
                            for record in records:
                                record.error = record.error or "upstream_response_incomplete"
                            await downstream.close(code=upstream.close_code or 1000)
                        except ConnectionClosed as exc:
                            code = exc.rcvd.code if exc.rcvd else 1006
                            if code not in (1000, 1001):
                                for record in records:
                                    if not record.finished:
                                        record.error = "upstream_websocket_closed"
                            await downstream.close(
                                code=code if code not in (1005, 1006, 1015) else 1011
                            )
                        except (WebSocketDisconnect, OSError):
                            client_left = True
                        finally:
                            group.cancel_scope.cancel()

                    group.start_soon(client_messages)
                    group.start_soon(provider_messages)
        except InvalidStatus as exc:
            response = exc.response
            record = _Recording(
                self.session,
                self.config,
                "GET",
                path,
                downstream.url.query,
                downstream.headers,
                b"",
                {},
            )
            record.status = response.status_code
            record.response_headers = httpx.Headers(list(response.headers.raw_items()))
            record.output.write(response.body or b"")
            record.finish()
            await downstream.send_denial_response(
                Response(
                    content=bytes(response.body or b""),
                    status_code=response.status_code,
                    headers=dict(_headers(record.response_headers)),
                )
            )
        except (OSError, TimeoutError, InvalidHandshake):
            record = _Recording(
                self.session,
                self.config,
                "GET",
                path,
                downstream.url.query,
                downstream.headers,
                b"",
                {},
            )
            record.status, record.error = 502, "capture_upstream_connection_error"
            record.finish()
            await downstream.send_denial_response(Response(status_code=502))
        finally:
            for record in [*records, auxiliary]:
                if record is not None and not record.finished:
                    record.cancelled = client_left and not record.error
                    record.finish()
