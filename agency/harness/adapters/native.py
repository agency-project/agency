"""Native harness adapter.

Runs the standalone native harness package inside the sandbox and normalizes
its result through the common daemon adapter seam.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid

from fastapi import Request

from .base import AdapterRuntime, AttemptResult, HarnessAdapter
from ..common import extract_bearer_token
from ..executable import HARNESS_PATH

# An idle deadline, not a flat one: reset whenever the react loop's progress
# checkpoint advances (see react_loop.py's _write_progress), the same
# activity-driven contract the PTY-based harnesses use. A genuinely slow but
# progressing run is never punished for its total wall-clock time; only
# actual silence for this long ends the attempt.
_DEFAULT_TIMEOUT_S = 300
_PROGRESS_POLL_INTERVAL_S = 1.0
# Bound on reaping an already-exited process tree, matching
# agProxyPtraceHandle.close()'s own kill()-then-join(timeout=10) sequence.
_REAP_TIMEOUT_S = 10.0

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


def _session_file_path(session_dir: str, session_id: str) -> str:
    """Must match `native_harness/session.py`'s own `session_path()`."""
    return f"{session_dir}/{session_id}.json"


class NativeAdapter(HarnessAdapter):
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

        from ...utils.agutil import AGENCY_PACKAGE_CONTAINER_MOUNT, ensure_python_packages_locally

        sandbox = runtime.sandbox
        assert sandbox is not None
        ensure_python_packages_locally(["httpx", "httpx2", "mcp", "html2text"], timeout_s=180)

        # Fresh scratch space per ATTEMPT, never reused across a
        # structured-output retry: session continuity across attempts flows
        # through AttemptResult.session_blob/resume_session_id, not
        # directory reuse, so each attempt can be fully self-contained.
        scratch_dir = agharness.materialize_config_home_in_container(
            runtime.engine_name, sandbox, uuid.uuid4().hex
        )
        offload_dir = f"{scratch_dir}/long_tool_call_outputs"
        progress_path = f"{scratch_dir}/progress.json"
        handle = None
        try:
            if resume_session_id and prior_session_blob is not None:
                sandbox.write_file_bytes(
                    _session_file_path(scratch_dir, resume_session_id), prior_session_blob
                )

            mcp_config = agharness.mcp_config_for(
                runtime.harness_base_url,
                runtime.token,
                has_sandbox_mcp_tools=runtime.has_sandbox_mcp_tools,
            )
            pkg_pythonpath = f"{AGENCY_PACKAGE_CONTAINER_MOUNT}/agency"

            argv = [
                sys.executable,
                "-m",
                "native_harness.cli",
                "-p",
                prompt,
                "--model",
                runtime.model or "",
                "--max-steps",
                str(runtime.agconfig.skill.react_max_steps if max_steps is None else max_steps),
                "--output-format",
                "json",
                "--bridge-base-url",
                runtime.harness_base_url,
                "--bridge-token",
                runtime.token,
                "--mcp-config",
                json.dumps(mcp_config),
                "--session-dir",
                scratch_dir,
                "--offload-dir",
                offload_dir,
                "--progress-file",
                progress_path,
            ]
            if resume_session_id:
                argv += ["--resume", resume_session_id]

            envp = {
                "PATH": HARNESS_PATH,
                "PYTHONPATH": pkg_pythonpath,
            }
            if "HOME" in os.environ:
                envp["HOME"] = os.environ["HOME"]

            # Same ptrace-supervised launch every other adapter uses (see
            # agProxyPtrace.launch()'s docstring) -- gives native the exact
            # same OS-level pause/resume/kill control as claude_code/codex/
            # opencode/grok, with no harness-specific control mechanism.
            px = agProxyPtrace(runtime.agconfig, allow_initial_exec=True)
            handle = px.launch(
                argv,
                envp,
                cwd="/workspace",
                policy=runtime.syscall_policy,
                ag=None,
            )
            runtime.register_control_handle(handle)
            deadline = time.monotonic() + _DEFAULT_TIMEOUT_S
            last_progress_mtime = None
            while handle.returncode is None:
                now = time.monotonic()
                if now > deadline:
                    return self._partial_result_from_progress(sandbox, progress_path)
                try:
                    mtime = os.stat(progress_path).st_mtime
                except OSError:
                    mtime = None
                if mtime is not None and mtime != last_progress_mtime:
                    last_progress_mtime = mtime
                    deadline = now + _DEFAULT_TIMEOUT_S
                time.sleep(_PROGRESS_POLL_INTERVAL_S)
            # Already finished -- kill() (a no-op if it already exited on its
            # own) then join with a bound, the same guaranteed-terminating
            # sequence handle.close() uses for the PTY drivers' teardown. A
            # bare handle.wait() here previously could hang forever: if the
            # tracer's "everything this launch spawned has exited"
            # bookkeeping ever misses an edge case (observed after a
            # mid-turn backend crash killed the react loop), nothing bounds
            # the wait, and the whole attempt is stuck with no recovery.
            handle.kill()
            stdout, stderr, rc = handle.wait(timeout=_REAP_TIMEOUT_S)

            if rc != 0:
                return AttemptResult(
                    ok=False,
                    error_message=f"native_harness exited with code {rc}: {stderr or stdout}",
                )

            payload = json.loads(stdout)
            session_id = payload.get("session_id")
            session_blob = None
            if session_id:
                try:
                    session_blob = sandbox.read_file_bytes(
                        _session_file_path(scratch_dir, session_id)
                    )
                except Exception:  # noqa: S110 - session persistence is best-effort
                    session_blob = None

            usage = payload.get("usage") or {}
            return AttemptResult(
                ok=True,
                final_text=payload.get("result", ""),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                session_id=session_id,
                session_blob=session_blob,
            )
        finally:
            try:
                if handle is not None:
                    # Idling out on the progress deadline returns while the
                    # process is still running. Reap it before deleting files
                    # it may still be using.
                    handle.close()
            finally:
                agharness.cleanup_config_home_in_container(sandbox, scratch_dir)

    @staticmethod
    def _partial_result_from_progress(sandbox, progress_path: str) -> AttemptResult:
        """The react loop idled past its deadline with no completion signal.
        Recover whatever it last checkpointed instead of failing an attempt
        that may still be genuinely working -- mirrors the PTY-based
        harnesses' own timeout-as-completion fallback (see
        `harness/adapters/pty/execution.py`), since the underlying reason is
        the same: a step-driven completion signal that can arrive late (or
        not at all) is not evidence the run itself failed.
        """
        try:
            progress = json.loads(sandbox.read_file_bytes(progress_path))
        except Exception:  # noqa: S110 - no checkpoint yet is not an error
            progress = {}
        return AttemptResult(
            ok=True,
            final_text=progress.get("final_text", ""),
            input_tokens=progress.get("total_input_tokens", 0),
            output_tokens=progress.get("total_output_tokens", 0),
        )

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
            "agency_internal_kind": raw_request.get("agency_internal_kind"),
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
                # Publish only the authoritative response after host finalization.
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


__all__ = ["NativeAdapter"]
