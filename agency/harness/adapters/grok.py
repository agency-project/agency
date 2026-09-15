"""Grok Build interactive PTY adapter and model-protocol translation."""

from __future__ import annotations

import shutil

from fastapi import Request

from .base import AdapterRuntime, AttemptResult, HarnessAdapter
from ..common import extract_bearer_token
from .pty.driver import _HookPtyDriver, run_pty_attempt
from .openai_chat_completions import ChatCompletionsProtocol
from .pty.execution import stream_response


def grok_available() -> bool:
    return shutil.which("grok") is not None












def _toml_string(value: str) -> str:
    """Quote a string for a hand-written TOML file -- only the escapes
    this module's config values actually need."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class GrokDriver(_HookPtyDriver):
    name = "grok"
    interrupt_key = b"\x03"
    _HOOK_EVENTS = ("SessionStart", "UserPromptSubmit", "Stop", "StopFailure", "StopCancelled")
    _HOOK_PATH = ("hooks", "agency.json")

    def _configure(self, adapter, runtime, max_steps):
        adapter._write_grok_config(
            self.root, runtime.harness_base_url, runtime.token, runtime.model or "default"
        )
        self.env["GROK_HOME"] = str(self.root)
        self.argv += ["--no-alt-screen", "--always-approve", "--no-memory", "--no-plan"]
        if max_steps is not None:
            self.argv += ["--max-turns", str(max_steps)]
        if self.session_id:
            self.argv += ["--resume", self.session_id]
        self._write_hooks()

    def ready(self, handle):
        lines, _x, y, _generation = handle.terminal_screen()
        current = lines[y].strip()
        return current.startswith("│ ❯") and not current.removeprefix("│ ❯").rstrip("│ ").strip()

    def _submitted_prompt(self, prompt):
        if (
            isinstance(prompt, str)
            and prompt.startswith("<user_query>\n")
            and prompt.endswith("\n</user_query>")
        ):
            return prompt[len("<user_query>\n") : -len("\n</user_query>")]
        return prompt

    def _stop_event(self, event, payload):
        if payload.get("reason") != "end_turn":
            return None
        return super()._stop_event(event, payload)

    def interrupted(self, handle):
        """Recognize Grok's restored composer draft as interrupt evidence."""
        if self._last_prompt is None:
            return False
        marker = self._last_prompt.split("\n", 1)[0]
        pasted = f"│ ❯ [Pasted: {len(self._last_prompt.splitlines())} lines]"
        lines, _x, y, _generation = handle.terminal_screen()
        # Grok collapses a restored multiline paste into a token. The prompt
        # marker then appears only in the old transcript.
        return (
            self.ready(handle)
            or any("│ ❯ " + marker in line for line in lines)
            or pasted in lines[y]
        )

    def clear_input(self, handle, wait_until):
        # Grok restores a pre-response cancellation as a draft. Its idle
        # Ctrl+C clears that draft; Ctrl+U would trigger self-update.
        if self._last_prompt is None:
            return
        wait_until(lambda: self.interrupted(handle), "restored Grok input")
        if not self.ready(handle):
            handle.write_terminal(b"\x03")

    def _scan_dialect_row(self, row, pending):
        update = row.get("params", {}).get("update", {})
        if update.get("sessionUpdate") != "turn_completed":
            return
        reason = update.get("stop_reason")
        if (
            reason == "cancelled"
            and row["params"].get("_meta", {}).get("cancelTrigger") == "ctrl_c"
        ):
            pending.append({"kind": "interrupt", "turn_id": update.get("prompt_id")})
        elif reason != "end_turn":
            pending.append(
                {
                    "kind": "error",
                    "turn_id": update.get("prompt_id"),
                    "error": f"Grok turn ended: {reason}",
                }
            )

    def completed(self, event):
        super().completed(event)
        text = []
        for row in self._rows():
            params = row.get("params", {})
            update = params.get("update", {})
            if (
                params.get("_meta", {}).get("promptId") == event["turn_id"]
                and update.get("sessionUpdate") == "agent_message_chunk"
            ):
                content = update.get("content", {})
                if content.get("type") == "text":
                    text.append(content["text"])
            if (
                update.get("sessionUpdate") == "turn_completed"
                and update.get("prompt_id") == event["turn_id"]
            ):
                event["text"] = "".join(text)
                usage = update.get("usage", {})
                event.update(
                    input_tokens=usage.get("inputTokens", 0),
                    output_tokens=usage.get("outputTokens", 0),
                )
                return True
        return False



class GrokAdapter(ChatCompletionsProtocol, HarnessAdapter):
    _DEFAULT_BINARY = "grok"
    _PTY_DRIVER = GrokDriver
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





__all__ = ["GrokAdapter", "grok_available"]
