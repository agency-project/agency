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
dispatch client -- so `register()`/`_format_context_harness_to_agency()`/
`_format_context_agency_to_harness()`/`_format_agency_stream_to_harness()`
are inherited unchanged. Only `run_daemon_attempt()` differs, since it
launches `tandem_harness.cli` with two models (`--worker-model`/
`--supervisor-model`) instead of `native_harness.cli`'s single `--model`.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid

from .base import AdapterRuntime, AttemptResult
from .native import NativeAdapter
from ..executable import HARNESS_PATH

# Same activity-driven idle deadline as native.py -- see that module's own
# comment on _DEFAULT_TIMEOUT_S for the rationale.
_DEFAULT_TIMEOUT_S = 300
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
        output_instruction: "str | None" = None,
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

            ha = runtime.agconfig.harness_adapter
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
                runtime.model or "",
                "--supervisor-model",
                ha.supervisor_model or "",
                "--segment-step-cap",
                str(ha.segment_step_cap),
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
            if ha.supervisor_base_url:
                argv += ["--supervisor-llm-base-url", ha.supervisor_base_url]
            if ha.supervisor_api_key:
                argv += ["--supervisor-llm-api-key", ha.supervisor_api_key]
            if output_instruction:
                # The submit_output tool-usage block -- goes to the worker's
                # own system prompt, since it's the worker that holds the
                # tool, not the supervisor (see daemon.py's
                # _render_attempt_prompt/_run_adapter_attempt).
                argv += ["--worker-output-instruction", output_instruction]
            if resume_session_id:
                argv += ["--resume", resume_session_id]

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
            handle.kill()
            stdout, stderr, rc = handle.wait(timeout=_REAP_TIMEOUT_S)

            if rc != 0:
                return AttemptResult(
                    ok=False,
                    error_message=f"tandem_harness exited with code {rc}: {stderr or stdout}",
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
                    handle.close()
            finally:
                agharness.cleanup_config_home_in_container(sandbox, scratch_dir)


__all__ = ["TandemAdapter"]
