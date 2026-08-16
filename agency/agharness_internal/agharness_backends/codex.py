"""Production adapter for the Codex non-interactive CLI.

Codex remains a normal :class:`agharness_backend`: Agency owns the process,
proxy, MCP, profiler, sandbox, result validation, and portable history.  This
module owns only Codex's config/argv, JSONL protocol, and native rollout-file
session format.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from ...agdata import agdata, agerror
from .base import agharness_backend

if TYPE_CHECKING:
    from ...agent import agent
    from ...agcontext import agcontext
    from ...agharness import HarnessMessages
    from ...agskill import agskill


_ENGINE_KEY = "codex"
_VERSION_RE = re.compile(r"(?:codex-cli\s+)?([^\s]+)")


def codex_available() -> bool:
    return shutil.which("codex") is not None


def _toml_string(value: object) -> str:
    """JSON string syntax is a valid TOML basic string and escapes safely."""
    return json.dumps(str(value), ensure_ascii=False)


def _read_file(sandbox, in_container: bool, path: str) -> bytes | None:
    try:
        if in_container:
            return sandbox.read_file_bytes(path)
        return Path(path).read_bytes()
    except (FileNotFoundError, OSError):
        return None


def _write_file(sandbox, in_container: bool, path: str, data: bytes) -> None:
    if in_container:
        sandbox.write_file_bytes(path, data)
    else:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _remove_file(sandbox, in_container: bool, path: str) -> None:
    try:
        if in_container:
            sandbox.exec(f"rm -f {shlex.quote(path)}", workdir="/")
        else:
            Path(path).unlink(missing_ok=True)
    except (FileNotFoundError, OSError):
        return


def _valid_rollout_relative_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None
    if path.parts[0] != "sessions" or path.suffix != ".jsonl":
        return None
    return path.as_posix()


def _canonical_session_id(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        canonical = str(uuid.UUID(value))
    except (ValueError, AttributeError):
        return None
    return canonical if value.lower() == canonical else None


def _rollout_session_metadata(blob: bytes) -> dict | None:
    """Read the authoritative metadata emitted at the start of a rollout."""
    for raw_line in blob.splitlines()[:32]:
        try:
            event = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(event, dict) or event.get("type") != "session_meta":
            continue
        payload = event.get("payload")
        return payload if isinstance(payload, dict) else None
    return None


def _rollout_matches_session(
    blob: bytes,
    session_id: str,
    *,
    workspace: str,
    rollout_cli_version: str | None = None,
) -> bool:
    """Reject corrupt, swapped, or workspace-incompatible native state."""
    metadata = _rollout_session_metadata(blob)
    if metadata is None:
        return False
    if metadata.get("id") != session_id or metadata.get("cwd") != workspace:
        return False
    metadata_version = metadata.get("cli_version")
    if not isinstance(metadata_version, str) or not metadata_version:
        return False
    return rollout_cli_version is None or metadata_version == rollout_cli_version


def _rollout_paths(sandbox, in_container: bool, config_home: str) -> list[str]:
    if in_container:
        command = (
            f"find {shlex.quote(config_home + '/sessions')} -type f "
            "-name '*.jsonl' -print 2>/dev/null"
        )
        output, rc = sandbox.exec(command, workdir="/")
        return (
            sorted(line.strip() for line in output.splitlines() if line.strip()) if rc == 0 else []
        )

    root = Path(config_home) / "sessions"
    if not root.is_dir():
        return []
    return sorted(str(path) for path in root.rglob("*.jsonl") if path.is_file())


def _relative_rollout_path(config_home: str, path: str) -> str | None:
    try:
        relative = PurePosixPath(path).relative_to(PurePosixPath(config_home)).as_posix()
    except ValueError:
        return None
    return _valid_rollout_relative_path(relative)


def _find_rollout(sandbox, in_container: bool, config_home: str, session_id: str) -> str | None:
    candidates = [
        path
        for path in _rollout_paths(sandbox, in_container, config_home)
        if session_id in PurePosixPath(path).name
    ]
    # The timestamped rollout filename sorts chronologically.  An isolated
    # CODEX_HOME normally has one match; choosing the newest is defensive.
    return candidates[-1] if candidates else None


def _restore_native_session(
    sandbox,
    in_container: bool,
    config_home: str,
    prior: object,
    *,
    context_revision: int,
    codex_version: str | None,
    workspace: str,
) -> str | None:
    if not isinstance(prior, dict):
        return None
    session_id = _canonical_session_id(prior.get("session_id"))
    relative_path = _valid_rollout_relative_path(prior.get("rollout_path"))
    encoded = prior.get("blob_b64")
    rollout_cli_version = prior.get("rollout_cli_version")
    if (
        session_id is None
        or relative_path is None
        or not isinstance(encoded, str)
        or not isinstance(rollout_cli_version, str)
        or not rollout_cli_version
        or prior.get("agcontext_revision") != context_revision
        or codex_version is None
        or prior.get("codex_version") != codex_version
        or rollout_cli_version != codex_version
    ):
        return None
    try:
        blob = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return None
    if not _rollout_matches_session(
        blob,
        session_id,
        workspace=workspace,
        rollout_cli_version=rollout_cli_version,
    ):
        return None
    try:
        _write_file(
            sandbox,
            in_container,
            f"{config_home}/{relative_path}",
            blob,
        )
    except (OSError, RuntimeError):
        return None
    return session_id


def _capture_native_session(
    sandbox,
    in_container: bool,
    config_home: str,
    session_id: str,
    *,
    next_context_revision: int,
    codex_version: str | None,
    workspace: str,
) -> dict | None:
    session_id = _canonical_session_id(session_id)
    if codex_version is None or session_id is None:
        return None
    path = _find_rollout(sandbox, in_container, config_home, session_id)
    if path is None:
        return None
    relative_path = _relative_rollout_path(config_home, path)
    blob = _read_file(sandbox, in_container, path)
    metadata = _rollout_session_metadata(blob) if blob is not None else None
    rollout_cli_version = metadata.get("cli_version") if metadata is not None else None
    if (
        relative_path is None
        or blob is None
        or not isinstance(rollout_cli_version, str)
        or not rollout_cli_version
        or rollout_cli_version != codex_version
        or not _rollout_matches_session(blob, session_id, workspace=workspace)
    ):
        return None
    return {
        "session_id": session_id,
        "rollout_path": relative_path,
        "blob_b64": base64.b64encode(blob).decode("ascii"),
        "agcontext_revision": next_context_revision,
        "codex_version": codex_version,
        "rollout_cli_version": rollout_cli_version,
    }


def _detect_codex_version(resolved: str, sandbox, in_container: bool) -> str | None:
    try:
        if in_container:
            output, rc = sandbox.exec(f"{shlex.quote(resolved)} --version", workdir="/")
            if rc != 0:
                return None
        else:
            completed = subprocess.run(
                [resolved, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
            )
            if completed.returncode != 0:
                return None
            output = completed.stdout or completed.stderr
    except (OSError, subprocess.SubprocessError, RuntimeError):
        return None
    match = _VERSION_RE.search(output.strip())
    return match.group(1) if match else None


def _render_codex_task(
    messages: "HarnessMessages",
    *,
    include_previous_context: bool,
    output_guidance: str | None,
) -> str:
    """Render only user-task material; system text lives in developer_instructions."""
    sections: list[tuple[str, str]] = []
    if include_previous_context and messages.previous_context:
        sections.append(
            (
                "PREVIOUS CONTEXT",
                json.dumps(list(messages.previous_context), ensure_ascii=False, default=str),
            )
        )
    sections.append(("CURRENT USER INPUT", messages.current_user_input))
    if messages.file_notices:
        sections.append(("FILE NOTICES", "\n".join(messages.file_notices)))
    if output_guidance:
        sections.append(("OUTPUT GUIDANCE", output_guidance.strip()))
    return "\n\n".join(f"[{title}]\n{body}" for title, body in sections)


def _error_text(event: dict) -> str:
    error = event.get("error")
    if isinstance(error, dict):
        for key in ("message", "detail", "code"):
            if error.get(key):
                return str(error[key])
        return json.dumps(error, ensure_ascii=False, default=str)
    if error:
        return str(error)
    for key in ("message", "detail"):
        if event.get(key):
            return str(event[key])
    return event.get("type", "unknown Codex error")


def _token_count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


@dataclass(frozen=True)
class _CodexEventSummary:
    session_id: str | None
    final_text: str
    input_tokens: int
    output_tokens: int
    completed: bool
    error: str | None


def _parse_codex_jsonl(stdout: str) -> _CodexEventSummary:
    session_id = None
    final_text = ""
    input_tokens = 0
    output_tokens = 0
    completed = False
    terminal_error = None
    malformed_lines: list[int] = []

    for line_number, raw_line in enumerate(stdout.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed_lines.append(line_number)
            continue
        if not isinstance(event, dict):
            malformed_lines.append(line_number)
            continue

        event_type = event.get("type")
        if event_type == "thread.started" and isinstance(event.get("thread_id"), str):
            session_id = event["thread_id"]
        elif event_type == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str):
                    final_text = text
        elif event_type == "turn.completed":
            completed = True
            usage = event.get("usage")
            if isinstance(usage, dict):
                # output_tokens already includes any reasoning-token portion;
                # never add a separate reasoning count on top.
                input_tokens = _token_count(usage.get("input_tokens"))
                output_tokens = _token_count(usage.get("output_tokens"))
        elif event_type in {
            "error",
            "turn.failed",
            "turn.cancelled",
            "turn.interrupted",
        }:
            terminal_error = _error_text(event)

    if malformed_lines:
        joined = ", ".join(str(number) for number in malformed_lines[:8])
        suffix = "…" if len(malformed_lines) > 8 else ""
        terminal_error = f"malformed Codex JSONL at line(s) {joined}{suffix}"
    elif terminal_error is None and not completed:
        terminal_error = "Codex JSONL ended without a turn.completed event"

    return _CodexEventSummary(
        session_id=session_id,
        final_text=final_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        completed=completed,
        error=terminal_error,
    )


def _looks_like_resume_state_failure(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "no saved session",
            "no conversation found",
            "session not found",
            "no session found",
            "failed to load rollout",
            "failed to resume",
            "unable to resume",
            "rollout file",
        )
    )


def _diagnostic(stdout: str, stderr: str, limit: int = 8000) -> str:
    text = (stderr or stdout or "no diagnostic output").strip()
    return text if len(text) <= limit else text[:limit] + "…"


class _CodexBackend(agharness_backend):
    _DEFAULT_BINARY = "codex"
    _DEFAULT_TIMEOUT_S = 600
    _PROVIDER_NAME = "agency-proxy"
    _PROXY_ENV_KEY = "AGENCY_PROXY_API_KEY"
    _MCP_ENV_KEY = "AGENCY_MCP_TOKEN"

    def execute(
        self,
        ag: "agent",
        prev_ctx: "agcontext",
        skill_input: agdata,
        max_steps: "int | None",
        *,
        skill: "agskill",
        extra_system: "str | None" = None,
        canonical_input=None,
    ) -> "tuple[agdata, agcontext, list[dict]]":
        from ... import agharness

        messages = canonical_input or agharness.build_harness_messages(
            skill, prev_ctx, skill_input, file_notice=extra_system
        )
        sys_msg = {"role": "system", "content": messages.system_instructions}

        if max_steps is not None:
            return (
                agerror("Codex does not expose a reliable max_steps limit; pass max_steps=None"),
                prev_ctx,
                [sys_msg],
            )
        if messages.attachments:
            return (
                agerror(
                    "Codex harness attachments are not yet materialized as CLI --image files; "
                    "use file paths in the task until the shared attachment contract supports it"
                ),
                prev_ctx,
                [sys_msg],
            )

        binary = self.binary_path or self._DEFAULT_BINARY
        in_container = agharness.is_container_backed(ag.sandbox)
        if in_container:
            resolved = agharness.resolve_harness_binary_in_container(ag.sandbox, binary)
        else:
            resolved = shutil.which(binary)
        if resolved is None:
            where = "inside the sandbox container" if in_container else "on PATH"
            return agerror(f"codex binary {binary!r} not found {where}"), prev_ctx, [sys_msg]

        codex_version = _detect_codex_version(resolved, ag.sandbox, in_container)
        token = uuid.uuid4().hex
        gateway = None
        terminus = None
        profiler_ingest = None
        mcp_server = None
        mcp_relay_proc = None
        config_home = None
        gateway_registered = False
        terminus_registered = False
        profiler_registered = False
        mcp_registered = False
        transcript = None
        setup_error: Exception | None = None
        stdout = ""
        stderr = ""
        rc: int | None = None
        protocol_error: str | None = None
        final_text = ""
        collected_output: dict = {}
        pending_session: dict | None = None
        total_input_tokens = 0
        total_output_tokens = 0
        output_schema_retries_left = skill.max_output_schema_retries
        use_structured_output = (
            skill.output_schema is not None and skill.output_schema.raw_key() is None
        )

        try:
            if in_container:
                from ..agllm_terminus import get_shared_terminus
                from ..agproxy_llm_in_container import ensure_agproxy_llm_in_container

                terminus = get_shared_terminus(ag.agconfig)
                terminus.register(token, ag)
                terminus_registered = True
                base_url = ensure_agproxy_llm_in_container(ag.sandbox, ag.agconfig)
            else:
                from ..agllm_terminus import get_shared_terminus
                from ..agproxy_llm import get_shared_gateway

                gateway = get_shared_gateway(ag.agconfig)
                terminus = get_shared_terminus(ag.agconfig)
                gateway.register(token, ag)
                gateway_registered = True
                base_url = gateway.base_url

            from ..agprof_ingest import get_shared_profiler_ingest

            profiler_ingest = get_shared_profiler_ingest()
            profiler_ingest.register(token, ag, exact_tool_events=False)
            profiler_registered = True

            from ..agmcp_server import get_shared_mcp_server

            mcp_server = get_shared_mcp_server(ag.agconfig)
            mcp_server.register(token, ag, skill)
            mcp_registered = True
            if in_container:
                from ..agproxy_ptrace_internal._in_container_launcher import start_tcp_relay

                mcp_relay_proc, mcp_relay_port = start_tcp_relay(
                    ag.sandbox, mcp_server.ensure_uds_started()
                )
                mcp_base_url = f"http://127.0.0.1:{mcp_relay_port}"
            else:
                mcp_base_url = mcp_server.start()

            if in_container:
                config_home = agharness.materialize_config_home_in_container(ag, ag.sandbox, token)
            else:
                config_home = agharness.materialize_config_home(ag, token, base_url)

            workspace = "/workspace" if ag.sandbox is not None else os.getcwd()
            sandbox_mode = "danger-full-access" if ag.sandbox is not None else "workspace-write"
            model = getattr(ag.llm.backend, "model", "") or "default"
            self._write_codex_config(
                config_home,
                base_url,
                model,
                developer_instructions=messages.system_instructions,
                mcp_base_url=mcp_base_url,
                sandbox_mode=sandbox_mode,
                workspace=workspace,
                sandbox=ag.sandbox if in_container else None,
            )

            sessions = getattr(ag, "_harness_sessions", None)
            prior = sessions.get(_ENGINE_KEY) if isinstance(sessions, dict) else None
            resume_session_id = _restore_native_session(
                ag.sandbox,
                in_container,
                str(config_home),
                prior,
                context_revision=prev_ctx.revision,
                codex_version=codex_version,
                workspace=workspace,
            )
            if prior is not None and resume_session_id is None and isinstance(sessions, dict):
                sessions.pop(_ENGINE_KEY, None)

            output_guidance = agharness.build_mcp_output_format_instruction(skill)
            first_prompt = _render_codex_task(
                messages,
                include_previous_context=resume_session_id is None,
                output_guidance=output_guidance,
            )
            prompt = first_prompt
            restored_session_id = resume_session_id
            resume_fallback_used = False
            final_message_path = f"{config_home}/last-message.txt"

            while True:
                _remove_file(ag.sandbox, in_container, final_message_path)
                argv = [resolved, "-C", workspace, "exec"]
                if resume_session_id:
                    argv += ["resume", resume_session_id]
                argv += [
                    "--json",
                    "--strict-config",
                    "--skip-git-repo-check",
                    "--ignore-rules",
                    "--output-last-message",
                    final_message_path,
                    "-",
                ]
                envp = {
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "CODEX_HOME": str(config_home),
                    self._PROXY_ENV_KEY: token,
                    self._MCP_ENV_KEY: token,
                }

                stdout, stderr, rc = agharness.run_harness_cli(
                    ag,
                    argv,
                    envp,
                    stdin=prompt,
                    timeout_s=self._DEFAULT_TIMEOUT_S,
                    cwd=workspace,
                )

                if rc != 0:
                    failure_text = f"{stderr}\n{stdout}"
                    if (
                        resume_session_id == restored_session_id
                        and not resume_fallback_used
                        and _looks_like_resume_state_failure(failure_text)
                    ):
                        # The portable Agency context is authoritative.  A native
                        # rollout rejected by Codex is an optimization miss, not
                        # a failed skill call.
                        resume_fallback_used = True
                        resume_session_id = None
                        pending_session = None
                        prompt = _render_codex_task(
                            messages,
                            include_previous_context=True,
                            output_guidance=output_guidance,
                        )
                        if isinstance(sessions, dict):
                            sessions.pop(_ENGINE_KEY, None)
                        continue
                    break

                summary = _parse_codex_jsonl(stdout)
                # A completed Codex turn is billable even when a later
                # protocol problem or structured-output retry makes the
                # overall skill call fail.  Account for each parsed attempt
                # before taking any error/fallback branch.
                total_input_tokens += summary.input_tokens
                total_output_tokens += summary.output_tokens
                if summary.error is not None:
                    if (
                        resume_session_id == restored_session_id
                        and not resume_fallback_used
                        and _looks_like_resume_state_failure(f"{summary.error}\n{stderr}\n{stdout}")
                    ):
                        resume_fallback_used = True
                        resume_session_id = None
                        pending_session = None
                        prompt = _render_codex_task(
                            messages,
                            include_previous_context=True,
                            output_guidance=output_guidance,
                        )
                        if isinstance(sessions, dict):
                            sessions.pop(_ENGINE_KEY, None)
                        continue
                    protocol_error = summary.error
                    break

                latest_file = _read_file(ag.sandbox, in_container, final_message_path)
                if latest_file is not None and latest_file.strip():
                    final_text = latest_file.decode("utf-8", errors="replace").strip()
                else:
                    final_text = summary.final_text

                completed_session_id = _canonical_session_id(summary.session_id)
                if completed_session_id:
                    resume_session_id = completed_session_id
                    pending_session = _capture_native_session(
                        ag.sandbox,
                        in_container,
                        str(config_home),
                        completed_session_id,
                        next_context_revision=prev_ctx.revision + 1,
                        codex_version=codex_version,
                        workspace=workspace,
                    )

                collected_output = mcp_server.collected_output(token)
                if not use_structured_output:
                    break
                required = set(skill.output_schema._data.keys())
                missing = sorted(required - set(collected_output.keys()))
                if not missing or output_schema_retries_left <= 0:
                    break
                if resume_session_id is None or pending_session is None:
                    protocol_error = (
                        "Codex completed but its rollout could not be captured for the "
                        "structured-output retry"
                    )
                    break
                output_schema_retries_left -= 1
                prompt = (
                    "[HARNESS SYSTEM] You have not yet provided all required output fields. "
                    f"Still missing: {missing}. Call the submit_output tool once for each."
                )

        except Exception as exc:
            setup_error = exc
        finally:
            if terminus is not None and (gateway_registered or terminus_registered):
                try:
                    transcript = terminus.transcript_for_token(token)
                except Exception as exc:
                    print(f"[agharness] WARNING: failed to read Codex transcript: {exc}")
            if gateway_registered:
                try:
                    gateway.unregister(token)
                except Exception as exc:
                    print(f"[agharness] WARNING: failed to unregister Codex gateway: {exc}")
            if terminus_registered:
                try:
                    terminus.unregister(token)
                except Exception as exc:
                    print(f"[agharness] WARNING: failed to unregister Codex terminus: {exc}")
            if profiler_registered:
                try:
                    profiler_ingest.unregister(token)
                except Exception as exc:
                    print(f"[agharness] WARNING: failed to unregister Codex profiler: {exc}")
            if mcp_registered:
                try:
                    mcp_server.unregister(token)
                except Exception as exc:
                    print(f"[agharness] WARNING: failed to unregister Codex MCP: {exc}")
            if mcp_relay_proc is not None:
                try:
                    from ..agproxy_ptrace_internal._in_container_launcher import stop_tcp_relay

                    stop_tcp_relay(mcp_relay_proc)
                except Exception as exc:
                    print(f"[agharness] WARNING: failed to stop Codex MCP relay: {exc}")
            if config_home is not None:
                try:
                    if in_container:
                        agharness.cleanup_config_home_in_container(ag.sandbox, str(config_home))
                    else:
                        agharness.cleanup_config_home(config_home)
                except Exception as exc:
                    # Teardown diagnostics must not replace the Codex result
                    # (or the original setup/process failure) already in hand.
                    print(f"[agharness] WARNING: failed to clean up Codex config home: {exc}")

        if setup_error is not None:
            prev_ctx.total_input_tokens += total_input_tokens
            prev_ctx.total_output_tokens += total_output_tokens
            return agerror(f"Codex setup/execution failed: {setup_error}"), prev_ctx, [sys_msg]
        prev_ctx.total_input_tokens += total_input_tokens
        prev_ctx.total_output_tokens += total_output_tokens
        if rc == -1:
            return (
                agerror(
                    f"Codex timed out after {self._DEFAULT_TIMEOUT_S}s: "
                    f"{_diagnostic(stdout, stderr)}"
                ),
                prev_ctx,
                [sys_msg],
            )
        if rc != 0:
            return (
                agerror(f"Codex exited with code {rc}: {_diagnostic(stdout, stderr)}"),
                prev_ctx,
                [sys_msg],
            )
        if protocol_error is not None:
            return agerror(f"Codex protocol failure: {protocol_error}"), prev_ctx, [sys_msg]

        if use_structured_output:
            required = set(skill.output_schema._data.keys())
            missing = sorted(required - set(collected_output.keys()))
            if missing:
                result = agerror(
                    "structured output incomplete after "
                    f"{skill.max_output_schema_retries - output_schema_retries_left} "
                    "retry/retries -- submit_output was never called for: " + ", ".join(missing)
                )
            else:
                result = agharness.finalize_harness_result(
                    agharness.HarnessResult(submitted_fields=collected_output),
                    skill,
                    ag.sandbox,
                )
        else:
            result = agharness.finalize_harness_result(
                agharness.HarnessResult(final_text=final_text), skill, ag.sandbox
            )

        if not isinstance(result, agerror) and pending_session is not None:
            sessions = getattr(ag, "_harness_sessions", None)
            if isinstance(sessions, dict):
                sessions[_ENGINE_KEY] = pending_session
            self.session_resume_id = pending_session["session_id"]

        if isinstance(transcript, list) and transcript:
            # A Codex Responses request normally becomes two system messages
            # at the Chat Completions boundary: top-level ``instructions``
            # plus the developer message in ``input``.  Neither belongs in
            # Agency's portable conversation history; the next invocation
            # regenerates its own developer instructions from the skill.
            history = [message for message in transcript if message.get("role") != "system"]
            prev_ctx.messages = history
            delta = [sys_msg, *history]
        else:
            user_msg = agharness.harness_user_message(messages)
            assistant_msg = {"role": "assistant", "content": final_text}
            prev_ctx.messages = [*messages.previous_context, user_msg, assistant_msg]
            delta = [sys_msg, user_msg, assistant_msg]
        return result, prev_ctx, delta

    def _write_codex_config(
        self,
        config_home,
        base_url: str,
        model: str,
        *,
        developer_instructions: str,
        mcp_base_url: str,
        sandbox_mode: str,
        workspace: str,
        sandbox=None,
    ) -> None:
        shell_path = "/usr/local/bin:/usr/bin:/bin"
        home = str(config_home)
        toml_text = "\n".join(
            [
                f"model = {_toml_string(model)}",
                f"model_provider = {_toml_string(self._PROVIDER_NAME)}",
                'approval_policy = "never"',
                f"sandbox_mode = {_toml_string(sandbox_mode)}",
                f"developer_instructions = {_toml_string(developer_instructions)}",
                "check_for_update_on_startup = false",
                'web_search = "disabled"',
                "project_doc_max_bytes = 0",
                "project_doc_fallback_filenames = []",
                "",
                f"[model_providers.{self._PROVIDER_NAME}]",
                'name = "Agency Proxy"',
                f"base_url = {_toml_string(base_url + '/v1')}",
                f"env_key = {_toml_string(self._PROXY_ENV_KEY)}",
                'wire_api = "responses"',
                "supports_websockets = false",
                "request_max_retries = 0",
                "stream_max_retries = 0",
                "",
                "[mcp_servers.agency]",
                f"url = {_toml_string(mcp_base_url + '/mcp')}",
                f"bearer_token_env_var = {_toml_string(self._MCP_ENV_KEY)}",
                "required = true",
                # The server is per-run bearer authenticated and Agency owns
                # policy.  Pre-approve its tools so headless Codex never
                # stalls on an interactive MCP confirmation.
                'default_tools_approval_mode = "approve"',
                "",
                "[shell_environment_policy]",
                'inherit = "none"',
                "ignore_default_excludes = false",
                "experimental_use_profile = false",
                "set = { "
                f"PATH = {_toml_string(shell_path)}, HOME = {_toml_string(home)}, "
                'TMPDIR = "/tmp", LANG = "C.UTF-8" }',
                "",
                "[features]",
                "apps = false",
                "plugins = false",
                "multi_agent = false",
                "memories = false",
                "standalone_web_search = false",
                "browser_use = false",
                "in_app_browser = false",
                "computer_use = false",
                "image_generation = false",
                "goals = false",
                "",
                "[tools]",
                "web_search = false",
                "",
                f"[projects.{_toml_string(workspace)}]",
                'trust_level = "untrusted"',
                "",
            ]
        )
        path = f"{config_home}/config.toml"
        if sandbox is not None:
            sandbox.write_file(path, toml_text)
        else:
            Path(path).write_text(toml_text)

    @staticmethod
    def _parse_output_events(stdout: str) -> str:
        """Compatibility helper for callers wanting best-effort final text."""
        summary = _parse_codex_jsonl(stdout)
        return summary.final_text or stdout.strip()

    @staticmethod
    def _parse_output(stdout: str) -> "tuple[str, str | None]":
        summary = _parse_codex_jsonl(stdout)
        return summary.final_text or stdout.strip(), summary.session_id


__all__ = ["_CodexBackend", "codex_available"]
