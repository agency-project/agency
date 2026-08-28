"""Native harness adapter.

Runs the standalone native harness package inside the sandbox and normalizes
its result through the common daemon adapter seam.
"""

from __future__ import annotations

import json
import shlex
import uuid
from .base import AdapterRuntime, AttemptResult, agharness_backend

_DEFAULT_TIMEOUT_S = 600


def _session_file_path(session_dir: str, session_id: str) -> str:
    """Must match `native_harness/session.py`'s own `session_path()`."""
    return f"{session_dir}/{session_id}.json"


class _NativeBackend(agharness_backend):
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

        # Do not infer this from the bridge URL: native_harness genuinely needs a container
        # (PYTHONPATH/package bind-mount, ensure_python_packages_in_container,
        # sandbox.exec() itself), so it checks the sandbox's own kind
        # directly, a real, current limitation, not a silently-accepted no-op.
        if not agharness.is_container_backed(runtime.sandbox):
            return AttemptResult(
                ok=False,
                error_message=(
                    "native's standalone harness requires a container-backed sandbox -- "
                    "bare-host/chroot support is future work"
                ),
            )

        from ...agutil import AGENCY_PACKAGE_CONTAINER_MOUNT, ensure_python_packages_in_container

        sandbox = runtime.sandbox
        assert sandbox is not None
        ensure_python_packages_in_container(
            sandbox, ["httpx", "httpx2", "mcp", "html2text"], timeout_s=180
        )

        # Fresh scratch space per ATTEMPT (not reused across a structured-
        # output retry the way the pre-refactor version reused one across
        # the whole execute() call) -- session continuity across attempts
        # flows through AttemptResult.session_blob/resume_session_id now,
        # not directory reuse, so each attempt can be fully self-contained.
        scratch_dir = agharness.materialize_config_home_in_container(
            runtime.engine_name, sandbox, uuid.uuid4().hex
        )
        offload_dir = f"{scratch_dir}/long_tool_call_outputs"
        try:
            if resume_session_id and prior_session_blob is not None:
                sandbox.write_file_bytes(
                    _session_file_path(scratch_dir, resume_session_id), prior_session_blob
                )

            mcp_config = agharness.mcp_config_for(runtime.harness_base_url, runtime.token)
            pkg_pythonpath = f"{AGENCY_PACKAGE_CONTAINER_MOUNT}/agency"
            run_id = uuid.uuid4().hex[:8]
            stdout_path = f"{scratch_dir}/stdout-{run_id}.json"
            stderr_path = f"{scratch_dir}/stderr-{run_id}.log"

            argv = [
                "python3",
                "-m",
                "native_harness.cli",
                "-p",
                shlex.quote(prompt),
                "--model",
                shlex.quote(runtime.model or ""),
                "--max-steps",
                str(max_steps or 20),
                "--output-format",
                "json",
                "--bridge-base-url",
                shlex.quote(runtime.harness_base_url),
                "--bridge-token",
                shlex.quote(runtime.token),
                "--mcp-config",
                shlex.quote(json.dumps(mcp_config)),
                "--session-dir",
                shlex.quote(scratch_dir),
                "--offload-dir",
                shlex.quote(offload_dir),
            ]
            if resume_session_id:
                argv += ["--resume", shlex.quote(resume_session_id)]

            cmd = (
                f"cd /workspace && PYTHONPATH={shlex.quote(pkg_pythonpath)} "
                + " ".join(argv)
                + f" > {shlex.quote(stdout_path)} 2> {shlex.quote(stderr_path)}"
            )
            _, rc = sandbox.exec(cmd, workdir="/workspace", timeout=_DEFAULT_TIMEOUT_S)
            stdout = sandbox.read_file(stdout_path)
            stderr = sandbox.read_file(stderr_path) if rc != 0 else ""

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
            agharness.cleanup_config_home_in_container(sandbox, scratch_dir)


__all__ = ["_NativeBackend"]
