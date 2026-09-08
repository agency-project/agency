"""Grok Build (xAI) backend -- https://grok.com/build, binary name `grok`.

Like opencode (and unlike Claude Code/Codex), Grok Build's model config
supports `api_backend = "chat_completions"` per `[model.<name>]` block in
its `config.toml` -- i.e. it can be pointed at an arbitrary OpenAI-
compatible endpoint speaking plain chat-completions, which is exactly
`agproxy_llm`'s existing passthrough route with zero translation. This is
the second backend (after opencode) that routes its LLM traffic through
`agproxy_llm` rather than leaving the harness's endpoint untouched.

CAVEAT: no `grok` binary was installed in the environment this was
developed in (installing it requires running xAI's `curl | bash` install
script, a real download-and-execute-from-the-internet action deliberately
not taken without being asked first) -- this
backend's orchestration logic (config-home isolation, agproxy_ptrace
launch, agproxy_llm token registration, output-schema recovery) follows
the exact same tested shape as `opencode.py`/`claude_code.py`, but the
config.toml schema and `--output-format json`'s exact field names are
implemented from xAI's published docs (docs.x.ai/build, the xai-org/
grok-build repo's user guide), not verified against a live run.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid

from fastapi import Request

from .agharness_backend import AdapterRuntime, AttemptResult, agharness_backend
from ..common import extract_bearer_token
from ..executable import HARNESS_PATH


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
    """Quote a string for inclusion in a hand-written TOML file -- only
    the escapes actually needed for the values this module writes
    (prompt text never goes through here; only config values do)."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class _GrokBackend(agharness_backend):
    _DEFAULT_BINARY = "grok"
    _MODEL_NAME = "agency-proxy"
    _DEFAULT_TIMEOUT_S = 600

    def run_daemon_attempt(
        self,
        runtime: AdapterRuntime,
        *,
        prompt: str,
        resume_session_id: "str | None",
        prior_session_blob: "bytes | None",
        max_steps: "int | None",
    ) -> AttemptResult:
        from .. import agharness
        from ..ptrace.supervisor import agProxyPtrace

        # The host preparation layer supplies an executable in this namespace.
        resolved = self.agconfig.harness_adapter.binary_path or self._DEFAULT_BINARY

        config_home = agharness.materialize_config_home(
            runtime.engine_name, runtime.token, runtime.harness_base_url
        )
        try:
            self._write_grok_config(
                config_home,
                runtime.harness_base_url,
                runtime.token,
                runtime.model or "default",
            )
            self._write_grok_hooks(config_home)
            prompt_path = config_home / "prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")

            argv = [
                resolved,
                "--prompt-file",
                str(prompt_path),
                "--output-format",
                "json",
                "--yolo",
            ]
            if max_steps is not None:
                argv += ["--max-turns", str(max_steps)]
            envp = {
                "PATH": HARNESS_PATH,
                # GROK_HOME redirects the *entire* config directory (config.toml,
                # auth.json, sessions/) -- the closest analog to Codex's CODEX_HOME,
                # and the documented isolation mechanism here: xAI's docs don't
                # expose a Claude-Code-style "--setting-sources ''"/"--ignore-user-
                # config" flag, so this is what actually keeps a scripted run from
                # touching (or reading) the caller's real ~/.grok. Per docs.x.ai/
                # build/features/hooks, project/personal hooks normally live under
                # <home>/hooks/*.json -- assumed (unverified, no live binary, see
                # module docstring) to follow GROK_HOME the same way config.toml
                # does.
                "GROK_HOME": str(config_home),
                # Read by agpolicy_hook.py (registered by _write_grok_hooks).
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

        parsed_result = self._parse_structured_result(stdout)
        max_turn_exhausted = (
            max_steps is not None
            and "max turns reached" in stderr.lower()
            and parsed_result is not None
            and parsed_result.get("type") != "error"
            and isinstance(parsed_result.get("text"), str)
            and parsed_result.get("stopReason") in {"cancelled", "max_turn_requests"}
        )
        if rc != 0 and not max_turn_exhausted:
            return AttemptResult(
                ok=False, error_message=f"grok exited with code {rc}: {stderr or stdout}"
            )

        final_text, usage, session_id = self._parse_result_json(stdout)
        return AttemptResult(
            ok=True,
            final_text=final_text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            session_id=session_id,
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

    def _write_grok_hooks(self, config_home) -> None:
        """Bridge Grok Build's own PreToolUse/PostToolUse hooks (per docs.
        x.ai/build/features/hooks, near-identical payload shape to Claude
        Code's/Codex's -- same shared hook script, see
        _harness_permission_hook.py) to agpolicy admission and the host's
        tool-call completion endpoint. Unverified against a live `grok`
        binary (none available in this environment, see this module's
        docstring)."""
        from pathlib import Path

        hook_src = (Path(__file__).parent.parent / "_harness_permission_hook.py").read_bytes()
        hooks_dir = config_home / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook_path = hooks_dir / "agpolicy_hook.py"
        hook_path.write_bytes(hook_src)
        hook_command = {"hooks": [{"type": "command", "command": f"python3 {hook_path}"}]}
        (hooks_dir / "agpolicy.json").write_text(
            json.dumps({"hooks": {"PreToolUse": [hook_command], "PostToolUse": [hook_command]}})
        )

    @staticmethod
    def _parse_structured_result(stdout: str) -> "dict | None":
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _parse_result_json(stdout: str) -> "tuple[str, dict, str | None]":
        """Parse `grok --prompt-file ... --output-format json`'s JSON result
        object -- `{"text": "...", "usage": {...}, "sessionId": "...", ...}`
        per xAI's published headless-mode docs (not verified against a
        live run -- see this module's docstring)."""
        payload = _GrokBackend._parse_structured_result(stdout)
        if payload is None:
            return stdout.strip(), {}, None
        text = payload.get("text", "")
        usage = payload.get("usage", {}) or {}
        session_id = payload.get("sessionId")
        return text, usage, session_id

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
            # Grok's headless CLI still requests UI titles and dashboard text.
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

                def gen():
                    yield from self._format_agency_stream_to_harness(
                        router.dispatch_stream(token, agency_context), model
                    )

                return StreamingResponse(gen(), media_type="text/event-stream")
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
