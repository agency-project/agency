"""Native engine backend -- launches the standalone `native_harness`
package (see `agency/native_harness/`) as a real, one-shot harness CLI
inside the sandbox container, exactly the way `claude_code.py` launches
the real `claude` binary. This replaces the previous design (a persistent
in-container RPC server this module launched once per sandbox and talked
to over a private UDS protocol, dispatching LLM calls through
`agllm_terminus`/`agmcp_server`/`agharness_messenger`/`agprof_ingest`
directly) -- see this module's own git history, or the conversation that
produced `agmanager_host`/`agmanager_harness`/`native_harness`, for the
full design rationale.

**Implements `_run_attempt()`, not `execute()`.** Everything generic
across every harness-driven engine -- prompt building, the structured-
output reprompt-retry loop, session-blob bookkeeping, live per-turn
transcript polling, building the final `(result, ctx, delta)` -- lives in
`agharness_backends/base.py`'s shared template method now. This module
only builds `native_harness`'s argv/env from `harness_base_url`/
`launch.token`, runs it via a plain `sandbox.exec()` (no ptrace tracing
needed -- see below), and parses its one-line JSON result.

**No `agProxyPtrace` needed, unlike every other harness backend**:
`native_harness`'s own loop already asks `agmanager_harness`'s
`/agpolicy/check_tool` before executing each built-in tool call (see that
package's `react_loop.py`) -- an explicit, in-process check, not something
that needs syscall-level interception the way an opaque third-party CLI
(Claude Code, Codex, ...) does. A plain `sandbox.exec()` is enough.

**Current limitation, not silently accepted**: only container-backed
sandboxes are supported (`harness_base_url` is None otherwise, and
`_run_attempt()` returns a clear failure rather than guessing) -- matching
what this backend already exclusively supported before this migration.
`native_harness` itself runs standalone on a bare host fine (see that
package's own docstring); wiring THIS backend up to launch it there too,
without a container to bridge into, is future work.

**`skill.add_tools`/`replace_tools` (cloudpickled custom tools) are no
longer supported** -- a deliberate scope decision (see the conversation
this design came out of): built-in tools ship with `native_harness` itself,
any extension goes through MCP (`--mcp-config`), never a second,
harness-specific mechanism. A skill that sets either one gets a clear
failure, not a silently-ignored tool set. `skill.replace_tools == []`
(the `plan_mode` case: suppress built-ins, add nothing) still works, via
`native_harness`'s own `--no-builtin-tools` flag.

**Session continuity**: `native_harness` persists a single JSON snapshot
per session (see that package's `session.py`) rather than Claude Code's
JSONL transcript -- `_run_attempt()` writes `prior_session_blob` into a
fresh scratch directory before launching (if resuming) and reads back
whatever `native_harness` wrote after a successful run; `execute()` in
`base.py` handles the actual `ag._harness_sessions["native"]` bookkeeping.

**Structured output**: `native_harness`'s own loop already discovers and
calls `submit_output` (exposed by `agmanager_host`'s MCP surface, reached
via `--mcp-config`) like any other MCP tool -- the bounded reprompt-retry
loop on top (Phase 6's "Outer" layer) is `base.py`'s shared job now, not
this module's.

**Known gap from this migration, not silently dropped**: no profiler span
emission from inside `native_harness` yet (the old in-container entrypoint
had `RemoteProfilerEmitter` wired into every turn/tool span). Would need an
HTTP-based profiler-hook mechanism in `native_harness` mirroring how
`claude_code.py`'s permission hook already feeds `/agprof/hook` -- not
built in this pass.

**Test suite fallout, not silently left broken**: `tests/harness/
agharness_backends/test_native.py` and `_native_loop_harness.py` exercise
the OLD design directly (`launch_in_container_entrypoint`, `ping`,
`_ensure_entrypoint`, the old `_LiveTranscriptPusher` against a real
`agLLMTerminus`) and now fail against this module -- porting or replacing
them is a necessary follow-up this pass does not include.
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
    engine_key = "native"

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

        # NOT a `harness_base_url is None` check -- `ensure_harness_bridge()`
        # always returns a real URL now (bare-host mode runs agmanager_harness
        # in-process on the host instead of inside a container, see that
        # function's docstring). native_harness genuinely needs a container
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

        suppress_builtins = runtime.suppress_builtin_tools

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
            if suppress_builtins:
                argv.append("--no-builtin-tools")

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
