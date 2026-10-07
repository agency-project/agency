"""Tandem harness adapter.

Runs the standalone tandem harness package (`tandem_harness/`, a fork of
`native_harness/` -- see that package's own docstrings for the two-model
design) inside the sandbox and normalizes its result through the common
daemon adapter seam, exactly like `native.py`'s `NativeAdapter` does for
`native_harness`.

Subclasses `NativeAdapter` rather than duplicating it wholesale: the wire
format between this adapter and its own subprocess
(`/v1/chat/completions`, OpenAI-compatible chat-completions) is identical
to native's, since `tandem_harness/llm_client.py` is byte-for-byte the same
dispatch client -- so `_format_context_harness_to_agency()`/
`_format_context_agency_to_harness()`/`_format_agency_stream_to_harness()`
are inherited unchanged. `run_daemon_attempt()` differs since it launches
`tandem_harness.cli` with two models (`--worker-model`/`--supervisor-model`)
instead of `native_harness.cli`'s single `--model`, and `register()` adds
a second route forwarding to the host's `/llm_supervisor` handler.
"""

from __future__ import annotations

import anyio
import json
import os
import sys
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

from .base import AdapterRuntime, AttemptResult
from .native import NativeAdapter
from .openai_chat_completions import CHAT_KEEPALIVE_FRAME, chat_error_frame
from .streaming import start_streaming_response, stream_response
from ..common import extract_bearer_token
from ..executable import HARNESS_PATH

# Same activity-driven idle deadline as native.py -- see that module's own
# comment on _DEFAULT_TIMEOUT_S for the rationale.
_DEFAULT_TIMEOUT_S = 0  # 0 = no idle deadline
_PROGRESS_POLL_INTERVAL_S = 1.0
_REAP_TIMEOUT_S = 10.0


def _session_file_path(session_dir: str, session_id: str) -> str:
    """Must match `tandem_harness/session.py`'s own `session_path()` -- the
    same on-disk layout native_harness uses (see native.py's own copy of
    this helper)."""
    return f"{session_dir}/{session_id}.json"


class TandemAdapter(NativeAdapter):
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

        scratch_dir = agharness.materialize_config_home_in_container(
            runtime.engine_name, sandbox, uuid.uuid4().hex
        )
        offload_dir = f"{scratch_dir}/long_tool_call_outputs"
        progress_path = f"{scratch_dir}/progress.json"
        # Decided here, not left for the subprocess to invent internally --
        # this way we always know which file to read a session out of,
        # even from an attempt that never reaches its own clean exit (an
        # idle timeout, say). Continuing a prior session keeps that same
        # id; a fresh attempt still gets one up front rather than only
        # learning it after the fact from the subprocess's own stdout.
        session_id = resume_session_id or uuid.uuid4().hex
        handle = None
        try:
            if resume_session_id and prior_session_blob is not None:
                sandbox.write_file_bytes(
                    _session_file_path(scratch_dir, session_id), prior_session_blob
                )

            mcp_config = agharness.mcp_config_for(
                runtime.harness_base_url,
                runtime.token,
                has_sandbox_mcp_tools=runtime.has_sandbox_mcp_tools,
            )
            pkg_pythonpath = f"{AGENCY_PACKAGE_CONTAINER_MOUNT}/agency"

            ha = runtime.agconfig.harness_adapter
            if not ha.supervisor_model:
                raise ValueError(
                    "tandem harness_adapter.supervisor_model is unset -- refusing to launch a "
                    "'tandem' attempt that would silently run the worker model for both roles"
                )
            if not runtime.model:
                raise ValueError("tandem runtime.model (worker model) is unset")
            argv = [
                sys.executable,
                "-m",
                "tandem_harness.cli",
                "-p",
                prompt,
                # runtime.model is the harness-wide model field every adapter
                # already gets (agconfig.agent.model) -- tandem treats it as
                # the worker model, the one that actually calls tools.
                "--worker-model",
                runtime.model,
                "--supervisor-model",
                ha.supervisor_model,
                "--segment-step-cap",
                str(ha.segment_step_cap),
                "--worker-history-turns",
                str(ha.worker_history_turns),
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
                # Always passed now, resuming or not -- see session_id's
                # own comment above. cli.py's _resolve_session() already
                # treats --session-id and --resume identically (both just
                # seed which id to look up), so this alone covers both
                # cases; --resume is no longer needed.
                "--session-id",
                session_id,
                "--supervisor-llm-base-url",
                f"{runtime.harness_base_url}/supervisor",
                "--supervisor-llm-api-key",
                runtime.token,
            ]
            if ha.canonicalize_run_output:
                argv.append("--canonicalize-run-output")
            if ha.compact_tables:
                argv.append("--compact-tables")
            if ha.brief_reports:
                argv.append("--brief-reports")
            if ha.workflow_prompt:
                argv.append("--workflow-prompt")
            if ha.outline_reports:
                argv.append("--outline-reports")
            if ha.batch_mode:
                argv += ["--batch-mode", ha.batch_mode]
            if ha.review_reports:
                argv.append("--review-reports")
            if ha.coverage_check:
                argv.append("--coverage-check")
            if ha.dual_mode:
                argv.append("--dual-mode")
            if ha.delta_reports:
                argv.append("--delta-reports")
            if ha.typed_reports:
                argv += ["--typed-reports", ha.typed_reports]
            if ha.compact_tests:
                argv.append("--compact-tests")
            if ha.end_on_submit:
                argv.append("--end-on-submit")
            if ha.submit_with_check:
                argv.append("--submit-with-check")
            if ha.drop_tools:
                argv += ["--drop-tools", ha.drop_tools]

            envp = {
                "PATH": HARNESS_PATH,
                "PYTHONPATH": pkg_pythonpath,
            }
            if "HOME" in os.environ:
                envp["HOME"] = os.environ["HOME"]

            # Same ptrace-supervised launch every other adapter uses -- see
            # native.py's own comment on this for the rationale.
            px = agProxyPtrace(runtime.agconfig, allow_initial_exec=True)
            handle = px.launch(
                argv,
                envp,
                cwd="/workspace",
                policy=runtime.syscall_policy,
                ag=None,
                output_callback=runtime.output_sink,
            )
            runtime.register_control_handle(handle)
            deadline = time.monotonic() + _DEFAULT_TIMEOUT_S
            last_progress_mtime = None
            while handle.returncode is None:
                now = time.monotonic()
                if _DEFAULT_TIMEOUT_S and now > deadline:
                    return self._partial_result_from_progress(
                        sandbox, progress_path, scratch_dir=scratch_dir, session_id=session_id
                    )
                try:
                    mtime = os.stat(progress_path).st_mtime
                except OSError:
                    mtime = None
                if mtime is not None and mtime != last_progress_mtime:
                    last_progress_mtime = mtime
                    deadline = now + _DEFAULT_TIMEOUT_S
                time.sleep(_PROGRESS_POLL_INTERVAL_S)
            handle.kill()
            stdout, stderr, rc = handle.wait(timeout=_REAP_TIMEOUT_S)

            if rc != 0:
                return AttemptResult(
                    ok=False,
                    error_message=f"tandem_harness exited with code {rc}: {stderr or stdout}",
                )

            payload = json.loads(stdout)
            # session_id is ours from the start now (see above) -- no need
            # to trust payload's own echo of it back.
            try:
                session_blob = sandbox.read_file_bytes(_session_file_path(scratch_dir, session_id))
            except Exception:  # swallow-ok: session persistence is best-effort
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
                    handle.close()
            finally:
                agharness.cleanup_config_home_in_container(sandbox, scratch_dir)

    def register(self, app, router) -> None:
        super().register(app, router)

        @app.post("/supervisor/v1/chat/completions")
        async def supervisor_chat_completions(request: Request):
            token = extract_bearer_token(request)
            if not token or not router.validate_token(token):
                return JSONResponse(
                    {"error": {"message": "unknown or missing bearer token"}}, status_code=401
                )
            body = await request.json()
            model = router.resolve_model(token, mount="llm_supervisor")
            agency_context = self._format_context_harness_to_agency(body)
            if body.get("stream"):
                frames = stream_response(
                    router,
                    token,
                    agency_context,
                    model,
                    self._format_agency_stream_to_harness,
                    keepalive_frame=CHAT_KEEPALIVE_FRAME,
                    keepalive_s=self.agconfig.harness_adapter.stream_keepalive_s,
                    error_frame=chat_error_frame,
                    mount="llm_supervisor",
                )
                return await start_streaming_response(request, frames)
            agency_response = await anyio.to_thread.run_sync(
                lambda: router.dispatch(token, agency_context, mount="llm_supervisor")
            )
            return JSONResponse(self._format_context_agency_to_harness(agency_response, model))


__all__ = ["TandemAdapter"]
