"""codex interactive PTY adapter and model-protocol translation."""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from functools import partial

from fastapi import Request

from .base import AdapterRuntime, AttemptResult, HarnessAdapter
from ..common import extract_bearer_token
from .pty.driver import _HookPtyDriver, run_pty_attempt
from .pty.execution import stream_response


def codex_available() -> bool:
    return shutil.which("codex") is not None


_RESPONSES_TYPE_PREFIX = "openai_responses_"


def _responses_native_block_type(native_type: str) -> str:
    return f"{_RESPONSES_TYPE_PREFIX}{native_type}"


def _responses_content_to_blocks(content) -> "list[dict]":
    if isinstance(content, str):
        return [{"type": "text", "index": 0, "text": content}]
    if not isinstance(content, list):
        return []
    blocks = []
    for item in content:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype in ("input_text", "output_text", "text"):
            blocks.append({"type": "text", "index": len(blocks), "text": item.get("text", "")})
        else:
            blocks.append(
                {"type": _responses_native_block_type(itype), "index": len(blocks), "data": item}
            )
    return blocks


def _flatten_unknown_fragment(fragment) -> dict:
    if not isinstance(fragment, dict):
        return {}
    if "start" in fragment or "deltas" in fragment:
        flat = dict(fragment.get("start") or {})
        for delta in fragment.get("deltas") or []:
            if not isinstance(delta, dict):
                continue
            for k, v in delta.items():
                if isinstance(v, str) and isinstance(flat.get(k), str):
                    flat[k] += v
                else:
                    flat[k] = v
        return flat
    return dict(fragment)


def _unknown_block_to_responses(b: dict) -> dict:
    data = b.get("data")
    fragments = data if isinstance(data, list) else [data]
    merged: dict = {}
    for fragment in fragments:
        flat = _flatten_unknown_fragment(fragment)
        for k, v in flat.items():
            if isinstance(v, str) and isinstance(merged.get(k), str):
                merged[k] += v
            else:
                merged[k] = v
    merged["type"] = b["type"][len(_RESPONSES_TYPE_PREFIX) :]
    return merged


def _responses_content_to_text(content) -> str:
    return "".join(b["text"] for b in _responses_content_to_blocks(content))


_TOOL_WIRE_NAME_MAX_LEN = 64  # Chat Completions' tool-name cap; no namespace field.


def _tool_wire_name(kind: str, name: str, namespace: "str | None" = None) -> str:
    if kind == "function" and not namespace:
        return name
    segments = ["mcp", *(namespace or "").split("."), name]
    if kind != "function":
        segments.append(kind)
    wire_name = "__".join(s for s in segments if s)
    if len(wire_name) > _TOOL_WIRE_NAME_MAX_LEN:
        # Truncate from the front: the tool name (and innermost namespace
        # segment) at the tail is the readable part worth keeping intact.
        wire_name = wire_name[-_TOOL_WIRE_NAME_MAX_LEN:]
    return wire_name


def _responses_tool_routes(request: dict) -> dict:
    routes = {}

    def visit(tools, namespace=None):
        for tool in tools or []:
            kind = tool.get("type")
            if kind == "namespace":
                scope = ".".join(filter(None, (namespace, tool["name"])))
                visit(tool.get("tools"), scope)
            elif kind in ("function", "custom"):
                name = _tool_wire_name(kind, tool["name"], namespace)
                routes[name] = {"tool": tool, "namespace": namespace}

    visit(request.get("tools"))
    inputs = request.get("input")
    if isinstance(inputs, list):
        for item in inputs:
            if isinstance(item, dict) and item.get("type") == "additional_tools":
                visit(item.get("tools"))
    return routes


def _responses_tools_to_agency(routes) -> "list[dict] | None":
    converted = []
    for name, route in routes.items():
        t, namespace = route["tool"], route["namespace"]
        description = t.get("description", "")
        parameters = t.get("parameters") or {"type": "object", "properties": {}}
        if namespace:
            description = f"{namespace}.{t['name']}: {description}"
        if t["type"] == "custom":
            # JSON-capable providers need an envelope; Codex receives the exact
            # string again, not JSON text masquerading as JavaScript/a patch.
            description += "\nSupply the raw tool input as the input string."
            if t.get("format"):
                description += "\nTool input format: " + json.dumps(t["format"])
            parameters = {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
                "additionalProperties": False,
            }
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return converted or None


def _tool_use_to_responses(block: dict, routes: "dict | None") -> dict:
    route = (routes or {}).get(block["name"])
    name = route["tool"]["name"] if route else block["name"]
    kind = route["tool"]["type"] if route else "function"
    item = {"id": f"fc_{uuid.uuid4().hex}", "call_id": block["id"], "name": name}
    if route and route["namespace"]:
        item["namespace"] = route["namespace"]
    if kind == "custom":
        payload = json.loads(block.get("arguments", "{}"))
        if not isinstance(payload, dict) or not isinstance(payload.get("input"), str):
            raise ValueError("Codex custom tool requires a string input")
        item.update(type="custom_tool_call", input=payload["input"])
    else:
        item.update(
            type="function_call", arguments=block.get("arguments", "{}"), status="completed"
        )
    return item


def _responses_tool_choice_to_agency(tool_choice):
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice if tool_choice in ("auto", "required", "none") else None
    if isinstance(tool_choice, dict) and tool_choice.get("type") in ("function", "custom"):
        name = _tool_wire_name(
            tool_choice["type"], tool_choice.get("name", ""), tool_choice.get("namespace")
        )
        return {"type": "function", "function": {"name": name}}
    if isinstance(tool_choice, dict) and tool_choice.get("type"):
        return {"type": _responses_native_block_type(tool_choice["type"]), "data": tool_choice}
    return None


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


class CodexDriver(_HookPtyDriver):
    name = "codex"
    _HOOK_EVENTS = ("SessionStart", "UserPromptSubmit", "Stop")
    _HOOK_PATH = ("hooks.json",)

    def __init__(self, adapter, runtime, root, session_id, blob, max_steps):
        self._trusted_directory = False
        super().__init__(adapter, runtime, root, session_id, blob, max_steps)

    def _configure(self, adapter, runtime, max_steps):
        adapter._write_codex_config(
            self.root,
            runtime.harness_base_url,
            runtime.model or "default",
            has_sandbox_mcp_tools=runtime.has_sandbox_mcp_tools,
        )
        with (self.root / "config.toml").open("a") as config:
            config.write('\n[projects."/workspace"]\ntrust_level = "trusted"\n')
        self.env.update(CODEX_HOME=str(self.root), AGENCY_PROXY_API_KEY=runtime.token)
        # Codex has no CLI step-limit flag, as with its previous adapter.
        self.argv += [
            "--no-alt-screen",
            "--dangerously-bypass-approvals-and-sandbox",
            "--dangerously-bypass-hook-trust",
        ]
        if self.session_id:
            self.argv += ["resume", self.session_id]
        self._write_hooks()

    def ready(self, handle):
        lines, x, y, _generation = handle.terminal_screen()
        if not self._trusted_directory and any(
            "Do you trust the contents of this directory?" in line for line in lines
        ):
            if any(self.cwd in line for line in lines) and any(
                "1. Yes, continue" in line for line in lines
            ):
                self._trusted_directory = True
                handle.write_terminal(b"\r")
            return False
        # Require the actual composer, never silence or an old message in history.
        return lines[y].strip().startswith("›") and x <= 3

    def _stop_event(self, event, payload):
        event = super()._stop_event(event, payload)
        # Codex reports JSON null when a turn ends immediately after a
        # successful MCP submission. The protocol represents that as
        # an empty final string so the engine can consume the output
        # collected by submit_output.
        if event["text"] is None:
            event["text"] = ""
        return event

    def _scan_dialect_row(self, row, pending):
        payload = row.get("payload", {})
        if row.get("type") != "event_msg":
            return
        event_type = payload.get("type")
        if event_type in {"task_started", "turn_started"}:
            self._transcript_turn_id = payload.get("turn_id")
        elif event_type == "error" or (event_type == "task_complete" and payload.get("error")):
            # Native errors can omit turn_id; associate only with an observed
            # transcript start, never the host's current prompt (which may
            # have changed).
            pending.append(
                {
                    "kind": "error",
                    "turn_id": payload.get("turn_id") or self._transcript_turn_id,
                    "error": "Codex terminal error: "
                    + str(payload.get("error") or payload.get("message", "request failed")),
                }
            )

    def completed(self, event):
        super().completed(event)
        for row in self._rows():
            payload = row.get("payload", {})
            if (
                row.get("type") == "event_msg"
                and payload.get("type") == "task_complete"
                and payload.get("turn_id") == event["turn_id"]
            ):
                return True
        return False


class CodexAdapter(HarnessAdapter):
    _DEFAULT_BINARY = "codex"
    _PTY_DRIVER = CodexDriver
    _PROVIDER_NAME = "agency-proxy"
    _ENV_KEY_NAME = "AGENCY_PROXY_API_KEY"

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

    def _write_codex_config(
        self,
        config_home,
        base_url: str,
        model: str,
        *,
        has_sandbox_mcp_tools: bool = False,
    ) -> None:
        toml_text = (
            f'model = "{model}"\n'
            f'model_provider = "{self._PROVIDER_NAME}"\n'
            f"\n"
            f"[model_providers.{self._PROVIDER_NAME}]\n"
            f'name = "Agency Proxy"\n'
            f'base_url = "{base_url}/v1"\n'
            f'env_key = "{self._ENV_KEY_NAME}"\n'
            f'wire_api = "responses"\n'
            "\n"
            "[mcp_servers.agency]\n"
            f'url = "{base_url}/mcp"\n'
            f'bearer_token_env_var = "{self._ENV_KEY_NAME}"\n'
            "required = true\n"
            'default_tools_approval_mode = "approve"\n'
        )
        if has_sandbox_mcp_tools:
            toml_text += (
                "\n"
                '[mcp_servers."agency-sandbox"]\n'
                f'url = "{base_url}/sandbox/mcp"\n'
                f'bearer_token_env_var = "{self._ENV_KEY_NAME}"\n'
                "required = true\n"
                'default_tools_approval_mode = "approve"\n'
            )
        (config_home / "config.toml").write_text(toml_text)

    def register(self, app, router) -> None:
        from fastapi.responses import JSONResponse, StreamingResponse

        def _auth_error():
            return JSONResponse(
                {"error": {"message": "unknown or missing bearer token"}}, status_code=401
            )

        @app.post("/v1/responses")
        async def responses(request: Request):
            token = extract_bearer_token(request)
            if not token or not router.validate_token(token):
                return _auth_error()
            body = await request.json()
            model = router.resolve_model(token)
            agency_context = self._format_context_harness_to_agency(body)
            # Keep routing request-local: one adapter serves multiple tokens.
            tool_routes = _responses_tool_routes(body)
            if body.get("stream"):
                from ..clients.host_services_client import HostDispatchError

                frames = stream_response(
                    router,
                    token,
                    agency_context,
                    model,
                    partial(self._format_agency_stream_to_harness, tool_routes=tool_routes),
                )

                # This adapter already buffers through `done`. Fetch the first
                # frame before committing HTTP 200 so a permanent upstream 400
                # is not turned into a retryable truncated SSE connection.
                async def wait_for_disconnect():
                    while (await request.receive())["type"] != "http.disconnect":
                        pass

                first_task = asyncio.create_task(anext(frames))
                disconnect_task = asyncio.create_task(wait_for_disconnect())
                first_received = False
                try:
                    done, _ = await asyncio.wait(
                        {first_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if disconnect_task in done:
                        return JSONResponse({"error": "client disconnected"}, status_code=499)
                    first = first_task.result()
                    first_received = True
                except HostDispatchError as exc:
                    return JSONResponse(
                        {
                            "error": {
                                "message": str(exc),
                                "type": "upstream_error",
                                "transient": exc.transient,
                            }
                        },
                        status_code=exc.status_code,
                    )
                finally:
                    import anyio

                    # A disconnect before HTTP headers must cancel the blocked
                    # model read just as a later StreamingResponse disconnect does.
                    with anyio.CancelScope(shield=True):
                        first_task.cancel()
                        disconnect_task.cancel()
                        await asyncio.gather(first_task, disconnect_task, return_exceptions=True)
                        if not first_received:
                            await frames.aclose()

                async def response_frames():
                    try:
                        yield first
                        async for frame in frames:
                            yield frame
                    finally:
                        await frames.aclose()

                return StreamingResponse(
                    response_frames(),
                    media_type="text/event-stream",
                )
            agency_response = router.dispatch(token, agency_context)
            return JSONResponse(
                self._format_context_agency_to_harness(
                    agency_response, model, tool_routes=tool_routes
                )
            )

    def _format_context_harness_to_agency(self, raw_request: dict) -> dict:
        messages: "list[dict]" = []

        instructions = raw_request.get("instructions")
        if instructions:
            messages.append(
                {"role": "system", "blocks": [{"type": "text", "index": 0, "text": instructions}]}
            )

        raw_input = raw_request.get("input")
        if isinstance(raw_input, str):
            messages.append(
                {"role": "user", "blocks": [{"type": "text", "index": 0, "text": raw_input}]}
            )
        elif isinstance(raw_input, list):
            for item in raw_input:
                if not isinstance(item, dict):
                    continue
                itype = item.get("type")
                if itype == "additional_tools":
                    continue
                if itype in ("function_call", "custom_tool_call"):
                    kind = "custom" if itype == "custom_tool_call" else "function"
                    arguments = (
                        json.dumps({"input": item.get("input", "")})
                        if kind == "custom"
                        else item.get("arguments", "{}")
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "blocks": [
                                {
                                    "type": "tool_use",
                                    "index": 0,
                                    "id": item.get("call_id", ""),
                                    "name": _tool_wire_name(
                                        kind, item.get("name", ""), item.get("namespace")
                                    ),
                                    "arguments": arguments,
                                }
                            ],
                        }
                    )
                elif itype in ("function_call_output", "custom_tool_call_output"):
                    messages.append(
                        {
                            "role": "tool",
                            "blocks": [
                                {
                                    "type": "tool_result",
                                    "index": 0,
                                    "tool_call_id": item.get("call_id", ""),
                                    "text": _responses_content_to_text(item.get("output")),
                                }
                            ],
                        }
                    )
                elif itype is None or itype == "message":
                    role = item.get("role", "user")
                    blocks = _responses_content_to_blocks(item.get("content"))
                    messages.append({"role": role, "blocks": blocks})
                elif itype == "reasoning":
                    summary_text = "".join(
                        part.get("text", "")
                        for part in item.get("summary") or []
                        if isinstance(part, dict)
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "blocks": [
                                {
                                    "type": "thinking",
                                    "index": 0,
                                    "text": summary_text,
                                    "signature": item.get("encrypted_content") or "",
                                }
                            ],
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "blocks": [
                                {
                                    "type": _responses_native_block_type(itype),
                                    "index": 0,
                                    "data": item,
                                }
                            ],
                        }
                    )

        # Responses represents parallel calls as separate output items. Chat
        # Completions requires them in ONE assistant message before any replies.
        # Preserve all blocks (including interleaved reasoning/commentary), not
        # separate assistant messages that orphan the preceding tool call.
        grouped = []
        for message in messages:
            if message["role"] == "assistant" and grouped and grouped[-1]["role"] == "assistant":
                grouped[-1]["blocks"].extend(message["blocks"])
            else:
                grouped.append({**message, "blocks": list(message["blocks"])})
        for message in grouped:
            for index, block in enumerate(message["blocks"]):
                block["index"] = index
        return {
            "messages": grouped,
            "tools": _responses_tools_to_agency(_responses_tool_routes(raw_request)),
            "tool_choice": _responses_tool_choice_to_agency(raw_request.get("tool_choice")),
        }

    def _format_context_agency_to_harness(
        self, agency_response: dict, model: str, *, tool_routes=None
    ) -> dict:
        message = agency_response["message"]
        output: "list[dict]" = []
        for b in message.get("blocks", []):
            if b["type"] == "text":
                output.append(
                    {
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex}",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": b["text"], "annotations": []}],
                    }
                )
            elif b["type"] == "thinking":
                output.append(
                    {
                        "type": "reasoning",
                        "id": f"rs_{uuid.uuid4().hex}",
                        "summary": [{"type": "summary_text", "text": b["text"]}],
                        "encrypted_content": b.get("signature", ""),
                    }
                )
            elif b["type"] == "tool_use":
                output.append(_tool_use_to_responses(b, tool_routes))
            elif b["type"].startswith(_RESPONSES_TYPE_PREFIX):
                output.append(_unknown_block_to_responses(b))

        usage = agency_response.get("usage") or {}
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        return {
            "id": f"resp_{uuid.uuid4().hex}",
            "object": "response",
            "status": "completed",
            "model": model,
            "output": output,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

    def _format_agency_stream_to_harness(self, agency_stream, model: str, *, tool_routes=None):
        request_id = f"resp_{uuid.uuid4().hex}"
        yield _sse(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": request_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": model,
                },
            },
        )

        output_index = 0
        text_item_id = None
        for item in agency_stream:
            if item["type"] == "delta":
                # A redirect can supersede draft deltas. Responses SSE cannot
                # retract them, so wait for the authoritative host checkpoint.
                continue

            text_parts = [
                block.get("text", "")
                for block in item["message"].get("blocks", [])
                if block["type"] == "text"
            ]
            if any(text_parts):
                text_item_id = f"msg_{uuid.uuid4().hex}"
                yield _sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {
                            "type": "message",
                            "id": text_item_id,
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                )
                yield _sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": text_item_id,
                        "output_index": output_index,
                        "delta": "".join(text_parts),
                    },
                )

            if text_item_id is not None:
                yield _sse(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": {
                            "type": "message",
                            "id": text_item_id,
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "".join(text_parts),
                                    "annotations": [],
                                }
                            ],
                        },
                    },
                )
                output_index += 1

            for b in item["message"].get("blocks", []):
                if b["type"] == "text":
                    continue
                idx = output_index
                output_index += 1
                if b["type"] == "thinking":
                    reasoning_item = {
                        "type": "reasoning",
                        "id": f"rs_{uuid.uuid4().hex}",
                        "summary": [{"type": "summary_text", "text": b["text"]}],
                        "encrypted_content": b.get("signature", ""),
                    }
                    yield _sse(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": idx,
                            "item": reasoning_item,
                        },
                    )
                    yield _sse(
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "output_index": idx,
                            "item": reasoning_item,
                        },
                    )
                elif b["type"] == "tool_use":
                    call_item = _tool_use_to_responses(b, tool_routes)
                    yield _sse(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": idx,
                            "item": call_item,
                        },
                    )
                    yield _sse(
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "output_index": idx,
                            "item": call_item,
                        },
                    )
                elif b["type"].startswith(_RESPONSES_TYPE_PREFIX):
                    native_item = _unknown_block_to_responses(b)
                    yield _sse(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": idx,
                            "item": native_item,
                        },
                    )
                    yield _sse(
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "output_index": idx,
                            "item": native_item,
                        },
                    )

            usage = item.get("usage") or {}
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            yield _sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": request_id,
                        "object": "response",
                        "status": "completed",
                        "model": model,
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        },
                    },
                },
            )
            return


__all__ = ["CodexAdapter", "codex_available"]
