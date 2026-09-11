"""Claude Code harness adapter.

Builds the Claude CLI invocation, routes it through the sandbox daemon's
policy-aware runtime, and owns native PTY input and completion.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import Request

from .agharness_backend import AdapterRuntime, AttemptResult, agharness_backend
from ..common import extract_bearer_token
from ..executable import HARNESS_PATH


def claude_code_available() -> bool:
    return shutil.which("claude") is not None


_DEFAULT_TIMEOUT_S = 600


# -- Native session continuity ----------------------------------------------
#
# Claude Code's own conversation transcript, stored as
# `<config_home>/projects/<slug>/<session_id>.jsonl` where <slug> is the
# launch's cwd (== config_home, same value passed to px.launch(cwd=...))
# with every non-alphanumeric character replaced by '-'. Confirmed
# empirically against the real CLI (v2.1.220): copying that file into a
# fresh directory and resuming with --resume <session_id> from there
# recovers the exact original conversation, with real prompt-cache hits;
# without the file, --resume fails cleanly ("No conversation found").
#
# Deliberately NOT the source of truth for history -- ag.context.recent_transcript
# stays that. This is a per-engine, opt-in optimization: extracted from and
# reinjected into whatever sandbox handles the next call, stored on
# ag.context.harness_sessions (and agent.save()/load()'s state.json), never on
# the sandbox's own filesystem.
_SESSION_SLUG_RE = re.compile(r"[^a-zA-Z0-9]")


def _session_slug(cwd: str) -> str:
    return _SESSION_SLUG_RE.sub("-", str(cwd))


def _session_path(config_home: str, session_id: str) -> str:
    return f"{config_home}/projects/{_session_slug(config_home)}/{session_id}.jsonl"


def _write_session_blob(path: str, data: bytes) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)


def _stringify_anthropic_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return "" if content is None else str(content)


def _text_block_from_anthropic(block: dict, index: int) -> dict:
    agency_block = {"type": "text", "index": index, "text": block.get("text", "")}
    if block.get("citations"):
        agency_block["citations"] = block["citations"]
    return agency_block


def _text_block_to_anthropic(b: dict) -> dict:
    block = {"type": "text", "text": b["text"]}
    if b.get("citations"):
        block["citations"] = b["citations"]
    return block


def _anthropic_system_to_text(system) -> "str | None":
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        joined = "".join(
            b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"
        )
        return joined or None
    return None


_SESSION_TITLE_PROMPT_MARKER = "Generate a concise, sentence-case title (3-7 words)"
_HEADLESS_SESSION_TITLE = '{"title":"Agency session"}'


def _is_session_title_request(raw_request: dict) -> bool:
    """Identify Claude Code's internal session-title generation request.

    Headless harness runs do not display the title, so sending this auxiliary
    request through the invocation's model endpoint only adds latency and can
    incorrectly compete with the user-visible turn's final-answer checkpoint.
    """
    parts = [_anthropic_system_to_text(raw_request.get("system")) or ""]
    parts.extend(
        _stringify_anthropic_content(message.get("content"))
        for message in raw_request.get("messages", [])
        if isinstance(message, dict)
    )
    return any(
        _SESSION_TITLE_PROMPT_MARKER in part
        or (
            "The session content is provided inside <session> tags." in part
            and 'Return JSON with a single "title" field.' in part
        )
        for part in parts
    )


def _anthropic_tools_to_agency(tools) -> "list[dict] | None":
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def _anthropic_tool_choice_to_agency(tool_choice):
    if not tool_choice:
        return None
    kind = tool_choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    return None


_ANTHROPIC_TYPE_PREFIX = "anthropic_"


def _anthropic_native_block_type(native_type: str) -> str:
    return f"{_ANTHROPIC_TYPE_PREFIX}{native_type}"


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


def _unknown_block_to_anthropic(b: dict) -> dict:
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
    merged["type"] = b["type"][len(_ANTHROPIC_TYPE_PREFIX) :]
    return merged


_STOP_REASON_TO_ANTHROPIC = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


def _stop_reason_to_anthropic(stop_reason: "str | None") -> str:
    return _STOP_REASON_TO_ANTHROPIC.get(stop_reason, stop_reason or "end_turn")


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


class _ClaudeCodeBackend(agharness_backend):
    _DEFAULT_BINARY = "claude"

    def run_daemon_attempt(
        self,
        runtime: AdapterRuntime,
        *,
        prompt: str,
        resume_session_id: str | None,
        prior_session_blob: bytes | None,
        max_steps: int | None,
    ) -> AttemptResult:
        execution = _ClaudePtyExecution(
            self,
            runtime,
            resume_session_id=resume_session_id,
            prior_session_blob=prior_session_blob,
            max_steps=max_steps,
        )
        return execution.run(prompt)

    def prepare_pty(self, runtime, *, resume_session_id=None, prior_session_blob=None):
        """Prepare isolated native configuration; one process belongs to one attempt."""
        from .. import agharness

        resolved = self.agconfig.harness_adapter.binary_path or self._DEFAULT_BINARY
        config_home = agharness.materialize_config_home(runtime.engine_name)
        try:
            if resume_session_id and prior_session_blob is not None:
                _write_session_blob(
                    _session_path(str(config_home), resume_session_id),
                    prior_session_blob,
                )

            hook_src = (Path(__file__).parent.parent / "_harness_permission_hook.py").read_bytes()
            hook_path = f"{config_home}/agpolicy_hook.py"
            Path(hook_path).write_bytes(hook_src)
            hook_command = {"hooks": [{"type": "command", "command": f"python3 {hook_path}"}]}
            lifecycle_path = config_home / "claude_pty_hook.py"
            lifecycle_path.write_bytes(Path(__file__).with_name("_claude_pty_hook.py").read_bytes())
            lifecycle_hook = {
                "hooks": [{"type": "command", "command": f"python3 {lifecycle_path}"}]
            }
            (config_home / "events").mkdir()
            hooks_settings = json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [hook_command],
                        "PostToolUse": [hook_command],
                        "PostToolUseFailure": [hook_command],
                        **{
                            name: [lifecycle_hook]
                            for name in (
                                "SessionStart",
                                "UserPromptSubmit",
                                "Stop",
                                "StopFailure",
                                "SessionEnd",
                                "Notification",
                            )
                        },
                    }
                }
            )

            mcp_config = json.dumps(
                agharness.mcp_config_for(
                    runtime.harness_base_url,
                    runtime.token,
                    has_sandbox_mcp_tools=runtime.has_sandbox_mcp_tools,
                )
            )

            envp = {
                "PATH": HARNESS_PATH,
                "TERM": "xterm-256color",
                # The daemon runs inside Agency's sandbox, including when its
                # container user is root. Claude requires this marker to allow
                # bypass mode there; Agency's hooks and ptrace still apply.
                "IS_SANDBOX": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_AUTOUPDATER": "1",
                "AGENCY_CLAUDE_STATE": str(config_home),
                "ANTHROPIC_BASE_URL": runtime.harness_base_url,
                "ANTHROPIC_AUTH_TOKEN": runtime.token,
                "AGPOLICY_BASE_URL": runtime.harness_base_url,
                "AGPOLICY_TOKEN": runtime.token,
                "AGPOLICY_STATE_DIR": str(config_home),
                "CLAUDE_CODE_USE_BEDROCK": "0",
                "CLAUDE_CONFIG_DIR": str(config_home),
            }
            if "HOME" in os.environ:
                envp["HOME"] = os.environ["HOME"]

            (config_home / "agency-turn.json").write_text(json.dumps({"turn_id": None}))
            # This is Agency's generated sandbox workspace. Acknowledge startup
            # dialogs here; Agency's hooks and tracer govern unattended tool use.
            (config_home / ".claude.json").write_text(
                json.dumps(
                    {
                        "hasCompletedOnboarding": True,
                        "theme": "dark",
                        "bypassPermissionsModeAccepted": True,
                        "projects": {str(config_home): {"hasTrustDialogAccepted": True}},
                    }
                )
            )
            argv = [
                resolved,
                "--permission-mode",
                "bypassPermissions",
                "--setting-sources",
                "",
                "--settings",
                hooks_settings,
                "--mcp-config",
                mcp_config,
                "--strict-mcp-config",
                "--model",
                runtime.model,
            ]
            if resume_session_id:
                argv += ["--resume", resume_session_id]
            return argv, envp, config_home
        except BaseException:
            agharness.cleanup_config_home(config_home)
            raise

    def register(self, app, router) -> None:
        from fastapi.responses import JSONResponse, StreamingResponse

        def _auth_error():
            return JSONResponse(
                {
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "unknown or missing bearer token",
                    },
                },
                status_code=401,
            )

        @app.post("/v1/messages")
        async def messages(request: Request):
            token = extract_bearer_token(request)
            if not token or not router.validate_token(token):
                return _auth_error()
            body = await request.json()
            warning = _mid_array_system_warning(body)
            if warning:
                router.log_warning(token, warning)
            model = router.resolve_model(token)
            if _is_session_title_request(body):
                title_text = '{"title":"Agency session"}'
                title_response = {
                    "type": "done",
                    "message": {
                        "role": "assistant",
                        "blocks": [{"type": "text", "index": 0, "text": title_text}],
                    },
                    "stop_reason": "stop",
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                }
                if body.get("stream"):

                    def title_gen():
                        yield from self._format_agency_stream_to_harness(
                            [{"type": "delta", "content": title_text}, title_response], model
                        )

                    return StreamingResponse(title_gen(), media_type="text/event-stream")
                return JSONResponse(self._format_context_agency_to_harness(title_response, model))
            agency_context = self._format_context_harness_to_agency(body)
            if body.get("stream"):

                async def gen():
                    # Closing Claude's request closes the upstream UDS stream,
                    # including while the host is still generating its batch.
                    from contextlib import aclosing
                    import anyio

                    stream = router.dispatch_stream_async(token, agency_context)
                    try:
                        async with aclosing(stream):
                            async for item in stream:
                                if item["type"] == "done":
                                    for event in self._format_agency_stream_to_harness(
                                        [item], model
                                    ):
                                        yield event
                    finally:
                        with anyio.CancelScope(shield=True):
                            await stream.aclose()

                return StreamingResponse(gen(), media_type="text/event-stream")
            agency_response = router.dispatch(token, agency_context)
            return JSONResponse(self._format_context_agency_to_harness(agency_response, model))

        @app.post("/v1/messages/count_tokens")
        async def count_tokens(request: Request):
            token = extract_bearer_token(request)
            if not token or not router.validate_token(token):
                return _auth_error()
            body = await request.json()
            approx_chars = len(str(body.get("system", ""))) + sum(
                len(str(m.get("content", ""))) for m in body.get("messages", [])
            )
            return JSONResponse({"input_tokens": max(1, approx_chars // 4)})

    def _format_context_harness_to_agency(self, raw_request: dict) -> dict:
        messages: "list[dict]" = []
        system_text = _anthropic_system_to_text(raw_request.get("system"))
        extra_system_parts: "list[str]" = []

        for m in raw_request.get("messages", []):
            role = m.get("role")
            content = m.get("content")
            if role == "system":
                extra_text = _anthropic_system_to_text(content)
                if extra_text:
                    extra_system_parts.append(extra_text)
                continue
            if isinstance(content, str):
                messages.append(
                    {"role": role, "blocks": [{"type": "text", "index": 0, "text": content}]}
                )
                continue
            if not isinstance(content, list):
                messages.append({"role": role, "blocks": []})
                continue

            if role == "user":
                blocks: "list[dict]" = []
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        blocks.append(_text_block_from_anthropic(block, len(blocks)))
                    elif btype == "tool_result":
                        if blocks:
                            messages.append({"role": "user", "blocks": blocks})
                            blocks = []
                        result_content = block.get("content")
                        tool_result_block = {
                            "type": "tool_result",
                            "index": 0,
                            "tool_call_id": block.get("tool_use_id", ""),
                            "text": _stringify_anthropic_content(result_content),
                        }
                        if isinstance(result_content, list) and any(
                            not (isinstance(c, dict) and c.get("type") == "text")
                            for c in result_content
                        ):
                            tool_result_block["raw_content"] = result_content
                        messages.append({"role": "tool", "blocks": [tool_result_block]})
                    else:
                        blocks.append(
                            {
                                "type": _anthropic_native_block_type(btype),
                                "index": len(blocks),
                                "data": block,
                            }
                        )
                if blocks:
                    messages.append({"role": "user", "blocks": blocks})
            elif role == "assistant":
                blocks = []
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        blocks.append(_text_block_from_anthropic(block, len(blocks)))
                    elif btype == "thinking":
                        blocks.append(
                            {
                                "type": "thinking",
                                "index": len(blocks),
                                "text": block.get("thinking", ""),
                                "signature": block.get("signature", ""),
                            }
                        )
                    elif btype == "tool_use":
                        blocks.append(
                            {
                                "type": "tool_use",
                                "index": len(blocks),
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            }
                        )
                    else:
                        blocks.append(
                            {
                                "type": _anthropic_native_block_type(btype),
                                "index": len(blocks),
                                "data": block,
                            }
                        )
                messages.append({"role": "assistant", "blocks": blocks})

        combined_system = (
            "\n\n".join([system_text] + extra_system_parts)
            if system_text
            else "\n\n".join(extra_system_parts)
        )
        if combined_system:
            messages.insert(
                0,
                {
                    "role": "system",
                    "blocks": [{"type": "text", "index": 0, "text": combined_system}],
                },
            )

        return {
            "messages": messages,
            "tools": _anthropic_tools_to_agency(raw_request.get("tools")),
            "tool_choice": _anthropic_tool_choice_to_agency(raw_request.get("tool_choice")),
        }

    def _format_context_agency_to_harness(self, agency_response: dict, model: str) -> dict:
        message = agency_response["message"]
        content_blocks: "list[dict]" = []
        for b in message.get("blocks", []):
            if b["type"] == "text":
                content_blocks.append(_text_block_to_anthropic(b))
            elif b["type"] == "thinking":
                block = {"type": "thinking", "thinking": b["text"]}
                if b.get("signature"):
                    block["signature"] = b["signature"]
                content_blocks.append(block)
            elif b["type"] == "tool_use":
                try:
                    tool_input = json.loads(b["arguments"] or "{}")
                except ValueError:
                    tool_input = {}
                content_blocks.append(
                    {"type": "tool_use", "id": b["id"], "name": b["name"], "input": tool_input}
                )
            elif b["type"].startswith(_ANTHROPIC_TYPE_PREFIX):
                content_blocks.append(_unknown_block_to_anthropic(b))
        usage = agency_response.get("usage") or {}
        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": model,
            "stop_reason": _stop_reason_to_anthropic(agency_response.get("stop_reason")),
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }

    def _format_agency_stream_to_harness(self, agency_stream, model: str):
        request_id = f"msg_{uuid.uuid4().hex}"
        yield _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": request_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )
        next_index = 0
        for item in agency_stream:
            if item["type"] == "delta":
                # Draft deltas can be superseded by an in-flight redirect.
                # Anthropic SSE cannot retract text already sent to Claude;
                # emit only the authoritative message after the checkpoint.
                continue
            for b in item["message"].get("blocks", []):
                idx = next_index
                next_index += 1
                if b["type"] == "text":
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "text_delta", "text": b.get("text", "")},
                        },
                    )
                    for citation in b.get("citations") or []:
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {"type": "citations_delta", "citation": citation},
                            },
                        )
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
                    continue
                if b["type"] == "thinking":
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {"type": "thinking", "thinking": ""},
                        },
                    )
                    if b.get("text"):
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {"type": "thinking_delta", "thinking": b["text"]},
                            },
                        )
                    if b.get("signature"):
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {"type": "signature_delta", "signature": b["signature"]},
                            },
                        )
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
                elif b["type"] == "tool_use":
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": b["id"],
                                "name": b["name"],
                                "input": {},
                            },
                        },
                    )
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": b.get("arguments", "{}"),
                            },
                        },
                    )
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
                elif b["type"].startswith(_ANTHROPIC_TYPE_PREFIX):
                    native_block = _unknown_block_to_anthropic(b)
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": native_block,
                        },
                    )
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
            usage = item.get("usage") or {}
            yield _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": _stop_reason_to_anthropic(item.get("stop_reason")),
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": usage.get("completion_tokens", 0)},
                },
            )
            yield _sse("message_stop", {"type": "message_stop"})
            return


def _mid_array_system_warning(body: dict) -> "str | None":
    n = sum(1 for m in body.get("messages", []) if m.get("role") == "system")
    if not n:
        return None
    return (
        f"harness emitted {n} mid-conversation system-role message(s) in its "
        "/v1/messages request -- not valid per the Anthropic Messages API "
        "(system must be the top-level `system` field, never a `messages` "
        "entry); folding into the leading system message before forwarding"
    )


class _ClaudePtyExecution:
    """One attempt's PTY, native acknowledgments, and terminal boundary.

    The lock covers both final completion and redirect submission. The process
    is never reused by another attempt, even when its native transcript resumes.
    """

    INPUT_TIMEOUT = 20.0
    START_TIMEOUT = 30.0

    def __init__(self, adapter, runtime, *, resume_session_id, prior_session_blob, max_steps):
        self.runtime = runtime
        self.argv, self.env, self.config_home = adapter.prepare_pty(
            runtime, resume_session_id=resume_session_id, prior_session_blob=prior_session_blob
        )
        if max_steps is not None:
            self.argv += ["--max-turns", str(max_steps)]
        self.handle = None
        self._lock = threading.RLock()
        self._active = False
        self._session_id = resume_session_id
        self._started = False
        self._turn_id = None
        self._expected_prompt = None
        self._acknowledged = False
        self._stop = None
        self._failure = None
        self._submission_offset = 0
        self._attempt_offset = 0
        self._deadline = time.monotonic() + _DEFAULT_TIMEOUT_S

    @property
    def _transcript_path(self):
        if self._session_id is None:
            return None
        return Path(_session_path(str(self.config_home), self._session_id))

    def _transcript(self):
        path = self._transcript_path
        if path is None or not path.exists():
            return b""
        return path.read_bytes()

    def _poll(self):
        for path in sorted((self.config_home / "events").glob("*.json")):
            event = json.loads(path.read_text())
            path.unlink()
            payload = event["payload"]
            kind = payload.get("hook_event_name")
            if kind == "SessionStart":
                self._session_id = payload["session_id"]
                self._started = True
            if event["turn_id"] != self._turn_id:
                continue
            if kind == "UserPromptSubmit" and payload.get("prompt") == self._expected_prompt:
                self._acknowledged = True
            elif kind == "Stop":
                self._stop = payload.get("last_assistant_message")
            elif kind in {"StopFailure", "SessionEnd"}:
                self._failure = f"Claude {kind}: {payload.get('error', 'session ended')}"

    def _check_alive(self):
        if self._failure:
            raise RuntimeError(self._failure)
        if self.handle.returncode is not None:
            raise RuntimeError(f"Claude process exited ({self.handle.returncode})")

    def _wait_until(self, predicate, description, timeout=None):
        deadline = time.monotonic() + (self.INPUT_TIMEOUT if timeout is None else timeout)
        last_poll = time.monotonic()
        while True:
            now = time.monotonic()
            if self.handle.is_paused():
                deadline += now - last_poll
            last_poll = now
            self._poll()
            if predicate():
                return
            self._check_alive()
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Claude timed out waiting for {description}")
            time.sleep(0.025)

    def _editable_input(self):
        # Claude 2.1.251's input box, with the cursor inside it. History and
        # terminal silence are not evidence that input can safely be submitted.
        lines, _x, y, generation = self.handle.terminal_screen()
        upper = next((i for i in range(y - 1, -1, -1) if lines[i].strip().startswith("───")), None)
        lower = next(
            (i for i in range(y + 1, len(lines)) if lines[i].strip().startswith("───")), None
        )
        if upper is None or lower is None:
            return None
        first = lines[upper + 1].lstrip()
        if not first.startswith("❯"):
            return None
        draft = "\n".join(
            [first[1:].strip(), *[s.strip() for s in lines[upper + 2 : lower]]]
        ).strip()
        return draft, generation

    @staticmethod
    def _validate_prompt(prompt):
        if any((ord(c) < 32 and c not in "\n\t") or ord(c) == 127 for c in prompt):
            raise ValueError("native prompt contains terminal control characters")

    def _submit(self, prompt):
        self._validate_prompt(prompt)
        self._turn_id = uuid.uuid4().hex
        self._expected_prompt = prompt
        self._acknowledged = False
        self._stop = None
        self._submission_offset = len(self._transcript())
        state = self.config_home / "agency-turn.json"
        temporary = state.with_suffix(".tmp")
        temporary.write_text(json.dumps({"turn_id": self._turn_id}))
        temporary.replace(state)
        self.handle.write_terminal(b"\x1b[200~" + prompt.encode() + b"\x1b[201~")
        self.handle.write_terminal(b"\r")
        self._wait_until(lambda: self._acknowledged, "UserPromptSubmit acknowledgment")
        self._deadline = time.monotonic() + _DEFAULT_TIMEOUT_S

    @staticmethod
    def _rows(blob):
        for line in blob.splitlines(keepends=True):
            if line.endswith(b"\n"):
                yield json.loads(line)

    @staticmethod
    def _text(row):
        content = row.get("message", {}).get("content", [])
        if isinstance(content, str):
            return content
        return "\n".join(b.get("text", "") for b in content if b.get("type") == "text")

    def _completed_snapshot(self):
        if self._stop is None:
            return None
        blob = self._transcript()
        submitted = False
        for row in self._rows(blob[self._submission_offset :]):
            text = self._text(row)
            if row.get("type") == "user" and text == self._expected_prompt:
                submitted = True
            if submitted and row.get("type") == "assistant" and text == self._stop:
                return blob
        return None

    def _interrupt(self):
        offset = len(self._transcript())
        generation = self.handle.terminal_screen()[3]
        self.handle.write_terminal(b"\x1b")

        def interrupted():
            # Native completion can win just after the caller's active check.
            # Preserve that result and queue the redirect instead of treating
            # the idle CLI's lack of an interruption record as a failed run.
            if self._stop is not None:
                return True
            for row in self._rows(self._transcript()[offset:]):
                if row.get("type") == "user" and self._text(row) in {
                    "[Request interrupted by user]",
                    "[Request interrupted by user for tool use]",
                }:
                    return True
            editable = self._editable_input()
            if editable and editable[0] and editable[1] > generation:
                # Before its first response Claude restores the original prompt
                # as a draft. Clear that known draft, then require an empty box.
                self.handle.write_terminal(b"\x1b\x1b")
                self._wait_until(
                    lambda: (self._editable_input() or (None,))[0] == "", "empty input"
                )
                return True
            return False

        self._wait_until(interrupted, "native interruption")
        return self._stop is None

    def redirect(self, message: str) -> bool:
        with self._lock:
            if not self._active:
                return False
            if self.handle.is_paused():
                return False
            self._poll()
            if self._stop is not None:
                return False
            try:
                self._validate_prompt(message)
            except ValueError:
                return False
            try:
                self._check_alive()
                if not self._interrupt():
                    return False
                self._submit("[Agency redirect]\n" + message)
                return True
            except (OSError, RuntimeError) as exc:
                # A partial or unacknowledged submission cannot remain alive:
                # terminate this attempt before the caller queues its fallback.
                self._active = False
                self._failure = str(exc)
                self.handle.kill()
                return False

    def run(self, prompt):
        from ..ptrace.supervisor import agProxyPtrace
        from ..agharness import cleanup_config_home

        try:
            self.handle = agProxyPtrace(self.runtime.agconfig, allow_initial_exec=True).launch(
                self.argv,
                self.env,
                cwd=str(self.config_home),
                pty_size=(120, 36),
                policy=self.runtime.syscall_policy,
                ag=None,
            )
            self.runtime.register_control_handle(self.handle)
            self.runtime.register_redirect(self.redirect)
            # Startup does not hold the delivery lock: early redirects return
            # False immediately instead of waiting for a harness to become ready.
            self._wait_until(
                lambda: self._started and (self._editable_input() or (None,))[0] == "",
                "startup input",
                self.START_TIMEOUT,
            )
            self._attempt_offset = len(self._transcript())
            self._submit("[Agency run]\n" + prompt)
            with self._lock:
                self._active = True
            last_poll = time.monotonic()
            while True:
                with self._lock:
                    now = time.monotonic()
                    if self.handle.is_paused():
                        self._deadline += now - last_poll
                    last_poll = now
                    self._poll()
                    self._check_alive()
                    snapshot = self._completed_snapshot()
                    if snapshot is not None:
                        self._active = False
                        usage = {"input_tokens": 0, "output_tokens": 0}
                        for row in self._rows(snapshot[self._attempt_offset :]):
                            for key in usage:
                                usage[key] += row.get("message", {}).get("usage", {}).get(key, 0)
                        return AttemptResult(
                            ok=True,
                            final_text=self._stop,
                            session_id=self._session_id,
                            session_blob=snapshot,
                            **usage,
                        )
                    if time.monotonic() > self._deadline:
                        raise RuntimeError("Claude attempt timed out")
                time.sleep(0.025)
        finally:
            with self._lock:
                self._active = False
                if self.handle is not None:
                    self.handle.close()
            cleanup_config_home(self.config_home)


__all__ = ["_ClaudeCodeBackend", "claude_code_available"]
