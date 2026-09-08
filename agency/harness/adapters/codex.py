"""Codex CLI harness adapter.

Builds an isolated Codex configuration, launches the CLI through the sandbox
daemon, and normalizes its JSONL output.
"""

from __future__ import annotations

import json
import shutil
import uuid

from fastapi import Request

from .agharness_backend import AdapterRuntime, AttemptResult, agharness_backend
from ..common import extract_bearer_token
from ..executable import HARNESS_PATH


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


def _responses_tools_to_agency(tools) -> "list[dict] | None":
    if not tools:
        return None
    converted = []
    for t in tools:
        if t.get("type") != "function":
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return converted or None


def _responses_tool_choice_to_agency(tool_choice):
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice if tool_choice in ("auto", "required", "none") else None
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    if isinstance(tool_choice, dict) and tool_choice.get("type"):
        return {"type": _responses_native_block_type(tool_choice["type"]), "data": tool_choice}
    return None


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


class _CodexBackend(agharness_backend):
    _DEFAULT_BINARY = "codex"
    _DEFAULT_TIMEOUT_S = 600
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
        from pathlib import Path

        from .. import agharness
        from ..ptrace.supervisor import agProxyPtrace

        # The host preparation layer supplies an executable in this namespace.
        resolved = self.agconfig.harness_adapter.binary_path or self._DEFAULT_BINARY

        config_home = agharness.materialize_config_home(
            runtime.engine_name, runtime.token, runtime.harness_base_url
        )
        try:
            self._write_codex_config(
                config_home, runtime.harness_base_url, runtime.model or "default"
            )

            # Bridge Codex's own PreToolUse/PostToolUse hooks (near-identical
            # payload shape to Claude Code's -- same shared hook script, see
            # _harness_permission_hook.py) to agpolicy admission and the
            # host's tool-call completion endpoint. Unverified against a
            # live `codex` binary (none available in this environment, see
            # this module's docstring) -- modeled on Claude Code's own
            # settings.json `hooks` shape and CODEX_HOME redirecting the
            # whole config directory the same way it does for config.toml.
            hook_src = (Path(__file__).parent.parent / "_harness_permission_hook.py").read_bytes()
            hook_path = config_home / "agpolicy_hook.py"
            hook_path.write_bytes(hook_src)
            hook_command = {"hooks": [{"type": "command", "command": f"python3 {hook_path}"}]}
            (config_home / "hooks.json").write_text(
                json.dumps({"hooks": {"PreToolUse": [hook_command], "PostToolUse": [hook_command]}})
            )

            argv = [resolved, "exec", "--json", prompt]
            envp = {
                "PATH": HARNESS_PATH,
                "CODEX_HOME": str(config_home),
                # Referenced by config.toml's `env_key` -- Codex reads the
                # provider's API key from the env var *named* there, not
                # from an inline value in config.toml.
                self._ENV_KEY_NAME: runtime.token,
                # Read by agpolicy_hook.py (registered above).
                "AGPOLICY_BASE_URL": runtime.harness_base_url,
                "AGPOLICY_TOKEN": runtime.token,
                "AGPOLICY_STATE_DIR": str(config_home),
            }

            px = agProxyPtrace(runtime.agconfig, allow_initial_exec=True)
            handle = px.launch(
                argv,
                envp,
                cwd="/workspace",
                policy=runtime.syscall_policy,
                ag=None,
            )
            stdout, stderr, rc = handle.wait(timeout=self._DEFAULT_TIMEOUT_S)
        finally:
            agharness.cleanup_config_home(config_home)

        if rc != 0:
            return AttemptResult(
                ok=False, error_message=f"codex exited with code {rc}: {stderr or stdout}"
            )

        final_text = self._parse_output_events(stdout)
        return AttemptResult(ok=True, final_text=final_text)

    def _write_codex_config(self, config_home, base_url: str, model: str) -> None:
        toml_text = (
            f'model = "{model}"\n'
            f'model_provider = "{self._PROVIDER_NAME}"\n'
            f"\n"
            f"[model_providers.{self._PROVIDER_NAME}]\n"
            f'name = "Agency Proxy"\n'
            f'base_url = "{base_url}/v1"\n'
            f'env_key = "{self._ENV_KEY_NAME}"\n'
            f'wire_api = "responses"\n'
        )
        (config_home / "config.toml").write_text(toml_text)

    @staticmethod
    def _parse_output_events(stdout: str) -> str:
        """Best-effort extraction of the final `agent_message` item's text
        from `codex exec --json`'s NDJSON event stream -- unverified
        against a live run (no `codex` binary available), see this
        module's docstring."""
        last_text = ""
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            item = event.get("item") or {}
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text:
                    last_text = text
        if last_text:
            return last_text
        return stdout.strip()

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
            if body.get("stream"):

                def gen():
                    yield from self._format_agency_stream_to_harness(
                        router.dispatch_stream(token, agency_context), model
                    )

                return StreamingResponse(gen(), media_type="text/event-stream")
            agency_response = router.dispatch(token, agency_context)
            return JSONResponse(self._format_context_agency_to_harness(agency_response, model))

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
                if itype == "function_call":
                    messages.append(
                        {
                            "role": "assistant",
                            "blocks": [
                                {
                                    "type": "tool_use",
                                    "index": 0,
                                    "id": item.get("call_id", ""),
                                    "name": item.get("name", ""),
                                    "arguments": item.get("arguments", "{}"),
                                }
                            ],
                        }
                    )
                elif itype == "function_call_output":
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
                                    "signature": item.get("encrypted_content", ""),
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

        return {
            "messages": messages,
            "tools": _responses_tools_to_agency(raw_request.get("tools")),
            "tool_choice": _responses_tool_choice_to_agency(raw_request.get("tool_choice")),
        }

    def _format_context_agency_to_harness(self, agency_response: dict, model: str) -> dict:
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
                    }
                )
            elif b["type"] == "tool_use":
                output.append(
                    {
                        "type": "function_call",
                        "id": f"fc_{uuid.uuid4().hex}",
                        "call_id": b["id"],
                        "name": b["name"],
                        "arguments": b["arguments"],
                        "status": "completed",
                    }
                )
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

    def _format_agency_stream_to_harness(self, agency_stream, model: str):
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
        text_parts: "list[str]" = []
        for item in agency_stream:
            if item["type"] == "delta":
                content = item.get("content")
                if content:
                    if text_item_id is None:
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
                    text_parts.append(content)
                    yield _sse(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": text_item_id,
                            "output_index": output_index,
                            "delta": content,
                        },
                    )
                continue

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
                    call_item = {
                        "type": "function_call",
                        "id": f"fc_{uuid.uuid4().hex}",
                        "call_id": b["id"],
                        "name": b["name"],
                        "arguments": b.get("arguments", "{}"),
                        "status": "completed",
                    }
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


__all__ = ["_CodexBackend", "codex_available"]
