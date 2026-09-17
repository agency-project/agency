"""Claude Code harness adapter.

Builds the Claude CLI invocation, routes it through the sandbox daemon's
policy-aware runtime, and owns native PTY input and completion.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path

from fastapi import Request

from .base import AdapterRuntime, AttemptResult, HarnessAdapter
from .pty.driver import PtyDriver, run_pty_attempt
from ..common import extract_bearer_token
from ..executable import HARNESS_PATH


def claude_code_available() -> bool:
    return shutil.which("claude") is not None


_DEFAULT_TIMEOUT_S = 300


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


class ClaudeDriver(PtyDriver):
    """Claude Code: Agency-assigned turn identity, hook-driven completion.

    Claude does not report a turn identity of its own, so this driver commits
    one to `agency-turn.json` and the lifecycle hook stamps events with it.
    Once a Stop event carries that turn_id, the turn is done; the transcript
    is only consulted afterward, to pull usage totals and a resumable
    session snapshot.
    """

    name = "claude"
    INPUT_TIMEOUT = 20.0
    START_TIMEOUT = 30.0
    ATTEMPT_TIMEOUT = _DEFAULT_TIMEOUT_S
    activity_extends_deadline = True

    def __init__(self, adapter, runtime, root, session_id, blob, max_steps):
        self.started = False
        self._prior_blob = blob
        self._expected_prompt = None
        self._attempt_offset = None
        self._snapshot = None
        self._interrupt_offset = 0
        self._interrupt_generation = None
        self._draft_cleared = False
        super().__init__(adapter, runtime, root, session_id, blob, max_steps)

    def _restore(self, session_id, blob):
        """prepare_pty owns the isolated config, including any resumed transcript."""

    def _configure(self, adapter, runtime, max_steps):
        # Claude's transcript path is derived from the launch cwd, so the
        # isolated config root has to be the working directory too.
        self.cwd = str(self.root)
        self.argv, self.env = adapter.prepare_pty(
            runtime,
            self.root,
            resume_session_id=self.session_id,
            prior_session_blob=self._prior_blob,
        )
        if max_steps is not None:
            self.argv += ["--max-turns", str(max_steps)]

    def _transcript(self):
        if self.session_id is None:
            return b""
        path = Path(_session_path(str(self.root), self.session_id))
        return path.read_bytes() if path.exists() else b""

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

    def _editable_input(self, handle):
        # Claude 2.1.251's input box, with the cursor inside it. History and
        # terminal silence are not evidence that input can safely be submitted.
        lines, _x, y, generation = handle.terminal_screen()
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

    def ready(self, handle):
        return self.started and (self._editable_input(handle) or (None,))[0] == ""

    def submission_marker(self, label):
        # This driver fences turns by identity, so the marker needs no nonce.
        return f"[Agency {label}]"

    def begin_turn(self, prompt):
        self._expected_prompt = prompt
        if self._attempt_offset is None:
            self._attempt_offset = len(self._transcript())
        turn_id = uuid.uuid4().hex
        state = self.root / "agency-turn.json"
        temporary = state.with_suffix(".tmp")
        temporary.write_text(json.dumps({"turn_id": turn_id}))
        temporary.replace(state)
        return turn_id

    def _event_from_payload(self, record):
        payload = record["payload"]
        kind = payload.get("hook_event_name")
        if kind == "SessionStart":
            self.session_id = payload["session_id"]
            self.started = True
            return None
        event = {"turn_id": record["turn_id"]}
        if kind == "UserPromptSubmit":
            event.update(kind="submit", prompt=payload.get("prompt"))
        elif kind == "Stop":
            event.update(kind="stop", text=payload.get("last_assistant_message") or "")
        elif kind in {"StopFailure", "SessionEnd"}:
            error = payload.get("error", "session ended")
            event.update(kind="error", error=f"Claude {kind}: {error}")
        else:
            return None
        return event

    def clear_input(self, handle, wait_until):
        """Claude's restored draft is cleared during interruption; send no key."""

    def begin_interrupt(self, handle):
        self._interrupt_offset = len(self._transcript())
        self._interrupt_generation = handle.terminal_screen()[3]
        self._draft_cleared = False

    def interrupted(self, handle):
        for row in self._rows(self._transcript()[self._interrupt_offset :]):
            if row.get("type") == "user" and self._text(row) in {
                "[Request interrupted by user]",
                "[Request interrupted by user for tool use]",
            }:
                return True
        editable = self._editable_input(handle)
        if editable and editable[0] and editable[1] > self._interrupt_generation:
            # Before its first response Claude restores the original prompt as a
            # draft. Clear it here; the caller then requires an empty box.
            if not self._draft_cleared:
                self._draft_cleared = True
                handle.write_terminal(b"\x1b\x1b")
            return True
        return False

    def reap(self, handle):
        handle.kill()

    def completed(self, event):
        super().completed(event)
        self._record_usage(self._transcript(), event)
        return True

    def _record_usage(self, blob, event):
        self._snapshot = blob
        usage = {"input_tokens": 0, "output_tokens": 0}
        for row in self._rows(blob[self._attempt_offset :]):
            for key in usage:
                usage[key] += row.get("message", {}).get("usage", {}).get(key, 0)
        event.update(usage)

    def snapshot(self):
        return self._snapshot


class ClaudeCodeAdapter(HarnessAdapter):
    _DEFAULT_BINARY = "claude"
    _PTY_DRIVER = ClaudeDriver

    def run_daemon_attempt(
        self,
        runtime: AdapterRuntime,
        *,
        prompt: str,
        resume_session_id: str | None,
        prior_session_blob: bytes | None,
        max_steps: int | None,
    ) -> AttemptResult:
        return run_pty_attempt(
            self,
            runtime,
            prompt=prompt,
            resume_session_id=resume_session_id,
            prior_session_blob=prior_session_blob,
            max_steps=max_steps,
        )

    def prepare_pty(self, runtime, config_home, *, resume_session_id=None, prior_session_blob=None):
        """Prepare isolated native configuration; one process belongs to one attempt."""
        from .. import agharness

        resolved = self.agconfig.harness_adapter.binary_path or self._DEFAULT_BINARY
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
            lifecycle_path = config_home / "claude_lifecycle_hook.py"
            lifecycle_path.write_bytes(
                (Path(__file__).parent / "pty" / "_claude_pty_hook.py").read_bytes()
            )
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
            if not self.agconfig.harness_adapter.allow_subagents:
                argv += ["--disallowedTools", "Agent"]
            if resume_session_id:
                argv += ["--resume", resume_session_id]
            return argv, envp
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

        for m in raw_request.get("messages", []):
            role = m.get("role")
            content = m.get("content")
            if role == "system":
                extra_text = _anthropic_system_to_text(content)
                if extra_text:
                    messages.append(
                        {
                            "role": "system",
                            "blocks": [{"type": "text", "index": 0, "text": extra_text}],
                        }
                    )
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

        if system_text:
            messages.insert(
                0,
                {
                    "role": "system",
                    "blocks": [{"type": "text", "index": 0, "text": system_text}],
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


__all__ = ["ClaudeCodeAdapter", "claude_code_available"]
