"""opencode backend -- the reference concrete `agharness_backend`
implementation and the smallest real end-to-end harness slice: opencode's
`@ai-sdk/openai-compatible` provider already speaks plain OpenAI
chat-completions, matching `agproxy_llm`'s passthrough route with zero
translation, and `opencode run --format json` is a simple headless
invocation.

CAVEAT: no `opencode` binary is installable in the environment this was
developed in (opencode requires Node/Bun, neither available) -- this
backend's orchestration logic (config-home isolation, agproxy_ptrace
launch, agproxy_llm token registration, output-schema recovery) is real
and tested (see tests/harness/agharness_backends/test_opencode.py's mocked-launch
tests), but the exact shape of `--format json`'s event stream and the
`opencode.json` provider-block schema are implemented from documented
behavior, not verified against a live run. `_parse_output_events` is
deliberately isolated and defensive (best-effort per-line JSON parsing,
falls back to raw text) so CLI drift or a wrong assumption here is a
contained, fixable gap rather than a crash.
"""

from __future__ import annotations

import json
import shutil
import uuid

from fastapi import Request

from .agharness_backend import AdapterRuntime, AttemptResult, agharness_backend
from ..common import extract_bearer_token
from ..executable import HARNESS_PATH


def opencode_available() -> bool:
    return shutil.which("opencode") is not None


# See _OpencodeBackend._write_agpolicy_plugin's docstring for the caveats
# on this plugin's exact hook shape/denial mechanism.
_AGPOLICY_PLUGIN_JS = """\
export const AgpolicyPlugin = async () => {
  const baseUrl = process.env.AGPOLICY_BASE_URL
  const token = process.env.AGPOLICY_TOKEN
  const pending = new Map()

  async function post(path, body) {
    const response = await fetch(baseUrl.replace(/\\/$/, "") + path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + token,
      },
      body: JSON.stringify(body),
    })
    return response.json()
  }

  return {
    "tool.execute.before": async (input, output) => {
      if (!baseUrl || !token) return
      let decision
      try {
        decision = await post("/agpolicy/check_tool", {
          tool_name: input.tool,
          tool_input: output.args,
        })
      } catch (err) {
        throw new Error("Cannot check invocation admission: agpolicy request failed: " + err)
      }
      if (decision.call_id) pending.set(input.callID, decision.call_id)
      if (decision.decision === "deny") {
        throw new Error(decision.reason || "denied by agpolicy")
      }
    },
    "tool.execute.after": async (input, output) => {
      if (!baseUrl || !token) return
      const callId = pending.get(input.callID)
      pending.delete(input.callID)
      if (!callId) return
      try {
        await post("/agpolicy/complete_tool", {
          call_id: callId,
          result: output.output,
          error: null,
        })
      } catch (err) {
        console.error("[agpolicy plugin] complete_tool request failed:", err)
      }
    },
  }
}
"""


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


class _OpencodeBackend(agharness_backend):
    _DEFAULT_BINARY = "opencode"
    _PROVIDER_NAME = "agency-proxy"
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
            plugin_path = self._write_agpolicy_plugin(config_home)
            self._write_opencode_config(
                config_home,
                runtime.harness_base_url,
                runtime.token,
                runtime.model or "default",
                plugin_path,
            )

            argv = [resolved, "run", "--format", "json", prompt]
            envp = {
                "PATH": HARNESS_PATH,
                "HOME": str(config_home),
                "OPENCODE_CONFIG": str(config_home / "opencode.json"),
                # Read directly (via process.env) by agpolicy_plugin.js --
                # unlike the subprocess-per-hook-firing CLIs (Claude Code/
                # Codex/Grok), an opencode plugin is one long-lived JS
                # module for the whole run, so it can just keep admitted
                # call_ids in an in-memory Map between tool.execute.before
                # and tool.execute.after -- no on-disk state dir needed.
                "AGPOLICY_BASE_URL": runtime.harness_base_url,
                "AGPOLICY_TOKEN": runtime.token,
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
                ok=False,
                error_message=f"opencode exited with code {rc}: {stderr or stdout}",
            )

        final_text = self._parse_output_events(stdout)
        return AttemptResult(ok=True, final_text=final_text)

    def _write_opencode_config(
        self, config_home, base_url: str, token: str, model: str, plugin_path
    ) -> None:
        config = {
            "provider": {
                self._PROVIDER_NAME: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Agency Proxy",
                    "options": {"baseURL": f"{base_url}/v1", "apiKey": token},
                    "models": {model: {"name": model}},
                }
            },
            "model": f"{self._PROVIDER_NAME}/{model}",
            # Per opencode.ai/docs/config's `plugin` array: local plugins are
            # referenced by file:// URL alongside npm-package/version specs.
            "plugin": [f"file://{plugin_path}"],
        }
        (config_home / "opencode.json").write_text(json.dumps(config))

    def _write_agpolicy_plugin(self, config_home):
        """Bridge opencode's own `tool.execute.before`/`tool.execute.after`
        plugin hooks (per opencode.ai/docs/plugins) to agpolicy admission
        and the host's tool-call completion endpoint. Unverified against a
        live `opencode` binary (none available in this environment, see
        this module's docstring) -- in particular, throwing from
        `tool.execute.before` is this plugin's best-effort mechanism for
        denying a call; opencode's docs describe the hook as able to
        "modify or block" execution but don't spell out the exact
        denial API."""
        plugin_dir = config_home / "plugin"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        plugin_path = plugin_dir / "agpolicy_plugin.js"
        plugin_path.write_text(_AGPOLICY_PLUGIN_JS)
        return plugin_path

    @staticmethod
    def _parse_output_events(stdout: str) -> str:
        """Best-effort extraction of the final assistant text from
        `opencode run --format json`'s output -- see this module's
        docstring on why this is deliberately defensive rather than a
        strict schema parse."""
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
            for key in ("text", "content", "result", "message"):
                value = event.get(key)
                if isinstance(value, str) and value:
                    last_text = value
        if last_text:
            return last_text
        # Fall back to the raw stdout itself (e.g. a plain-text response
        # with no JSON structure at all) rather than silently returning
        # an empty string.
        return stdout.strip()

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

        def _chunk(delta: dict, finish_reason: "str | None" = None) -> str:
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(payload)}\n\n"

        for item in agency_stream:
            if item["type"] == "delta":
                content = item.get("content")
                if content:
                    yield _chunk({"content": content})
                continue

            tool_call_index = 0
            for b in item["message"].get("blocks", []):
                if b["type"] == "tool_use":
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


__all__ = ["_OpencodeBackend", "opencode_available"]
