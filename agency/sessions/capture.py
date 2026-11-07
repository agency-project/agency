"""Opt-in, wire-preserving HTTP capture for the real interactive CLIs.

The gateway never translates, retries, fits, or edits a provider request. Raw
bodies are authoritative; Agency blocks are an additional inspection projection.
Authentication is replaced at the boundary and is never written to the archive.
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
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import anyio
import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

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

    def __post_init__(self):
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("capture upstream must be an HTTP(S) URL without credentials")
        if url.query or url.fragment or not self.model or not self.key_env:
            raise ValueError("invalid capture configuration")


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


def _response(raw: bytes, content_type: str, protocol: str):
    if "text/event-stream" not in content_type:
        value = json.loads(raw)
        failed = isinstance(value, dict) and (
            value.get("error") or value.get("status") in {"failed", "incomplete"}
        )
        return value, True, bool(failed)
    result, blocks, arguments = {}, {}, {}
    terminal = failed = False
    for event in _events(raw):
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
                failed = kind != "response.completed"
                if kind == "error":
                    result["error"] = event.get("error", event)
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
                {"state": "waiting_llm", "model": config.model},
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

    def finish(self):
        if self.finished:
            return
        self.finished = True
        self.output.close()
        ended = time.time()
        raw = (self.directory / "response.bin").read_bytes()
        response, terminal, failed = {}, False, False
        projection_error = None
        try:
            decoded = _decode(raw, self.response_headers.get("content-encoding", ""))
            if decoded:
                response, terminal, failed = _response(
                    decoded, self.response_headers.get("content-type", ""), self.protocol
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
            "model": self.config.model,
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
            f"llm:{self.config.model}",
            self.started,
            ended,
            {
                "model": self.config.model,
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
    def __init__(self, session, config: CaptureConfig):
        self.session, self.config = session, config
        self._key = os.environ.get(config.key_env)
        if not self._key:
            raise ValueError(f"capture requires {config.key_env}")

    async def forward(self, request: Request, path: str):
        token = request.headers.get("authorization", "").removeprefix(
            "Bearer "
        ) or request.headers.get("x-api-key")
        if not self.session._authorized(token):
            return JSONResponse({"error": {"message": "invalid capture token"}}, status_code=401)
        if not _PATH.fullmatch(path):
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
        if (record.is_model or body.get("model")) and body.get("model") != self.config.model:
            record.status, record.error = 400, "capture_model_mismatch"
            record.finish()
            return JSONResponse(
                {"error": {"message": "model differs from the configured capture model"}},
                status_code=400,
            )
        base = self.config.base_url.rstrip("/").removesuffix("/v1")
        url = base + "/" + path + ("?" + request.url.query if request.url.query else "")
        headers = [
            (k, v)
            for k, v in _headers(request.headers)
            if k.lower() not in _PRIVATE_HEADERS | {"accept-encoding"}
        ]
        headers += [("authorization", "Bearer " + self._key), ("accept-encoding", "identity")]
        if path.startswith("v1/messages"):
            headers.append(("x-api-key", self._key))
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(1200, connect=10), follow_redirects=False, trust_env=False
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
