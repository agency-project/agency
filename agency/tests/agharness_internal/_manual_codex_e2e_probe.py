"""Manual, real Codex harness end-to-end probe (intentionally not pytest).

This exercises the production ``agent.run(..., engine="codex")`` path with a
real Codex CLI in a real Docker sandbox.  It proves all of the integration
boundaries that mocks cannot:

* Codex reaches Agency's Responses proxy and the configured LLM backend;
* Codex uses its shell in ``/workspace`` without forwarding host/API secrets;
* structured results arrive through the shared MCP ``submit_output`` tool;
* a captured rollout resumes the same native Codex thread in a fresh container;
* a deliberately stale native session falls back to portable Agency history;
* per-run auth/MCP/profiler registrations, config homes, and containers are
  cleaned up.

The probe never prints credential values or ephemeral bearer tokens.  It only
records whether selected variable *names* were visible to Codex's child shell.

Run from the repository root with an OpenAI-compatible, tool-capable model::

    set -a
    source /path/to/.llm_env
    set +a
    uv run python agency/tests/agharness_internal/_manual_codex_e2e_probe.py

Required environment variables: ``LLM_BASE_URL``, ``LLM_MODEL``, and
``LLM_API_KEY``.  ``LLM_CONTEXT_LIMIT`` and ``AGENCY_SANDBOX_IMAGE`` are
optional.  The Linux Codex executable must be available as ``codex`` in the
sandbox image or at ``~/.cache/agency_harness_bin/codex`` on the host (that
cache is mounted read-only at ``/opt/agency_harness_bin``).
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath

from agency import agSandbox, agdata, agent, agskill
from agency.agconfig import agConfig
from agency.agllm_backends import agVLLMBackendConfig
from agency.agsandbox import agSandboxConfig
from agency.agsandbox_backends import agSandboxBackendConfig
from agency.agharness_internal.agllm_terminus import get_shared_terminus
from agency.agharness_internal.agmcp_server import get_shared_mcp_server
from agency.agharness_internal.agprof_ingest import get_shared_profiler_ingest


_REQUIRED_ENV = ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY")
_CONTAINER_CODEX_CACHE = "/opt/agency_harness_bin/codex"
_SHELL_SECRET_NAMES = (
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AGENCY_PROXY_API_KEY",
    "AGENCY_MCP_TOKEN",
)


class ProbeFailure(RuntimeError):
    """Failure with a deliberately credential-free diagnostic."""


@dataclass(frozen=True)
class _RegistrySnapshot:
    terminus_agents: frozenset[str]
    mcp_sessions: frozenset[str]
    mcp_outputs: frozenset[str]
    profiler_registrations: frozenset[str]


@dataclass
class _TrackedSandbox:
    sandbox: agSandbox
    runtime: str
    container_name: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeFailure(message)


def _required_environment() -> dict[str, str]:
    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    _require(not missing, "missing required environment variable(s): " + ", ".join(missing))
    return {name: os.environ[name] for name in _REQUIRED_ENV}


def _build_config(env: dict[str, str]) -> agConfig:
    llm_fields: dict[str, object] = {
        "base_url": env["LLM_BASE_URL"],
        "model": env["LLM_MODEL"],
        "api_key": env["LLM_API_KEY"],
    }
    raw_context_limit = os.environ.get("LLM_CONTEXT_LIMIT")
    if raw_context_limit:
        try:
            llm_fields["context_limit"] = int(raw_context_limit)
        except ValueError as exc:
            raise ProbeFailure("LLM_CONTEXT_LIMIT must be an integer") from exc

    cfg = agConfig(
        agVLLMBackendConfig(**llm_fields),
        agSandboxBackendConfig(backend="docker"),
    )
    if image := os.environ.get("AGENCY_SANDBOX_IMAGE"):
        agSandboxConfig(cfg).set_base_image(image)
    return cfg


def _registry_snapshot(terminus, mcp_server, profiler) -> _RegistrySnapshot:
    with terminus._lock:
        terminus_agents = frozenset(terminus._agents_by_token)
    with mcp_server._lock:
        mcp_sessions = frozenset(mcp_server._sessions_by_token)
        mcp_outputs = frozenset(mcp_server._collected_outputs)
    with profiler._lock:
        profiler_registrations = frozenset(profiler._registrations)
    return _RegistrySnapshot(
        terminus_agents=terminus_agents,
        mcp_sessions=mcp_sessions,
        mcp_outputs=mcp_outputs,
        profiler_registrations=profiler_registrations,
    )


def _assert_registry_cleanup(terminus, mcp_server, profiler, baseline: _RegistrySnapshot) -> None:
    _require(
        _registry_snapshot(terminus, mcp_server, profiler) == baseline,
        "a per-run terminus, MCP, or profiler registration leaked",
    )


def _terminus_request_count(terminus) -> int:
    with terminus._lock:
        return len(terminus.request_log)


def _assert_proxy_dispatch(terminus, previous_count: int, expected_model: str) -> int:
    with terminus._lock:
        new_entries = list(terminus.request_log[previous_count:])
        next_count = len(terminus.request_log)
    _require(new_entries, "Codex completed without an authenticated Agency proxy dispatch")
    _require(
        any(entry.get("model") == expected_model for entry in new_entries),
        "Agency proxy dispatch did not use the configured model",
    )
    return next_count


def _new_sandbox(cfg: agConfig, name: str, tracked: list[_TrackedSandbox]) -> agSandbox:
    sandbox = agSandbox(name, agconfig=cfg)
    tracked.append(
        _TrackedSandbox(
            sandbox=sandbox,
            runtime=sandbox._backend._runtime,
            container_name=sandbox._backend._container_name(),
        )
    )
    # Starting it here gives an early, clear binary/ABI failure instead of a
    # later generic harness result.  It also lets cleanup track the exact
    # container even if the skill call itself fails.
    sandbox._backend._ensure_started()
    return sandbox


def _codex_version_in(sandbox: agSandbox) -> str:
    resolve = (
        "if command -v codex >/dev/null 2>&1; then command -v codex; "
        f"elif test -x {_CONTAINER_CODEX_CACHE}; then echo {_CONTAINER_CODEX_CACHE}; "
        "else exit 1; fi"
    )
    binary, rc = sandbox.exec(resolve, workdir="/")
    _require(
        rc == 0 and binary.strip(),
        "no Linux Codex executable was found in the image or mounted harness cache",
    )
    version, rc = sandbox.exec(f"{shlex.quote(binary.strip())} --version", workdir="/")
    _require(rc == 0 and version.strip(), "the in-container Codex executable could not run")
    return version.strip().splitlines()[0]


def _resolve_result(pending: agdata, label: str) -> dict:
    result = pending.wait().to_dict()
    if result.get("error"):
        # Backend errors should not contain credentials, but do not echo their
        # free-form text from a probe whose contract is to print no secrets.
        raise ProbeFailure(f"{label} returned an Agency error (details intentionally suppressed)")
    return result


def _assistant_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            if parts:
                return "\n".join(parts).strip()
    return ""


def _is_json_document(text: str) -> bool:
    try:
        json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return True


def _validated_session_record(ag: agent) -> dict:
    ag.ctx.resolve_prev_dependencies()
    record = ag._harness_sessions.get("codex")
    _require(isinstance(record, dict), "Codex did not persist native session state")
    required = {
        "session_id",
        "rollout_path",
        "blob_b64",
        "agcontext_revision",
        "codex_version",
    }
    _require(required <= set(record), "the persisted Codex session record is incomplete")
    _require(
        record["agcontext_revision"] == ag.ctx.revision,
        "the Codex session record does not match Agency context revision",
    )
    _require(
        isinstance(record["session_id"], str) and bool(record["session_id"]),
        "the Codex session record has no thread id",
    )
    _require(isinstance(record["rollout_path"], str), "the Codex rollout path is not text")
    rollout_path = PurePosixPath(record["rollout_path"])
    _require(
        not rollout_path.is_absolute()
        and ".." not in rollout_path.parts
        and rollout_path.parts[0] == "sessions"
        and rollout_path.suffix == ".jsonl",
        "the Codex rollout path is not a safe relative session path",
    )
    _require(isinstance(record["blob_b64"], str), "the persisted Codex rollout is not text")
    try:
        rollout = base64.b64decode(record["blob_b64"], validate=True)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise ProbeFailure("the persisted Codex rollout is not valid base64") from exc
    _require(
        record["session_id"].encode() in rollout,
        "the persisted Codex rollout does not belong to its recorded thread",
    )
    _require(
        isinstance(record["codex_version"], str) and bool(record["codex_version"]),
        "the Codex session record has no CLI version",
    )
    return record


def _assert_no_config_home(sandbox: agSandbox) -> None:
    output, rc = sandbox.exec(
        "find /tmp -maxdepth 1 -type d -name 'agharness-*' -print",
        workdir="/",
    )
    _require(rc == 0, "could not inspect temporary Codex config homes")
    _require(not output.strip(), "a temporary Codex config home leaked inside the sandbox")


def _destroy_all(tracked: list[_TrackedSandbox]) -> None:
    failures: list[str] = []
    for item in reversed(tracked):
        try:
            item.sandbox.destroy()
        except Exception as exc:
            failures.append(f"destroy:{type(exc).__name__}")
    for item in tracked:
        try:
            inspected = subprocess.run(
                [item.runtime, "inspect", item.container_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(f"inspect:{type(exc).__name__}")
            continue
        if inspected.returncode == 0:
            failures.append("container-still-exists")
    _require(not failures, "sandbox cleanup failed: " + ", ".join(failures))


def main() -> None:
    env = _required_environment()
    cfg = _build_config(env)
    nonce = uuid.uuid4().hex[:12]
    marker = f"AGENCY_CODEX_E2E_MARKER={nonce}"
    codename = f"EMBER-{nonce.upper()}"
    artifact_path = f"/workspace/codex-e2e-{nonce}.txt"
    env_status_path = f"/workspace/codex-e2e-env-{nonce}.txt"
    artifact_contents = f"{marker}\nDEPLOYMENT_CODENAME={codename}\n"

    terminus = get_shared_terminus(cfg)
    mcp_server = get_shared_mcp_server(cfg)
    profiler = get_shared_profiler_ingest()
    registry_baseline = _registry_snapshot(terminus, mcp_server, profiler)
    request_count = _terminus_request_count(terminus)
    tracked: list[_TrackedSandbox] = []

    print("Codex E2E: starting three isolated Docker turns")
    try:
        first_sandbox = _new_sandbox(cfg, f"codex_e2e_first_{nonce}", tracked)
        codex_version = _codex_version_in(first_sandbox)
        print(f"Codex E2E: using {codex_version}")

        structured_skill = agskill(
            name="codex_e2e_workspace_and_mcp",
            system_prompt=(
                "You are a deterministic integration-test worker. Use real shell commands in "
                "/workspace, verify files by reading them, never reveal environment values, and "
                "submit every required output field through the Agency MCP submit_output tool."
            ),
            input_schema=agdata(instruction=str),
            output_schema=agdata(
                status=str,
                artifact_path=str,
                artifact_contents=str,
                env_status=str,
                codename=str,
            ),
            max_output_schema_retries=2,
        )
        recall_skill = agskill(
            name="codex_e2e_recall",
            system_prompt=(
                "Answer from the conversation state supplied by Agency. Be concise and do not "
                "invent a deployment codename."
            ),
            input_schema=agdata(instruction=str),
        )

        ag = agent(
            f"codex_e2e_agent_{nonce}",
            agconfig=cfg,
            sandbox=first_sandbox,
            engine="codex",
        )
        secret_names_json = json.dumps(list(_SHELL_SECRET_NAMES))
        first_prompt = f"""
Complete these steps with actual shell commands:

1. Create {artifact_path} with exactly these UTF-8 contents, including the final newline:
{artifact_contents}
2. In a child shell, inspect only whether any of the environment variable names in
   {secret_names_json} are present. Never print or write their values. Write exactly `clean` plus
   a newline to {env_status_path} when none are present; otherwise write `present:` followed only
   by comma-separated variable names and a newline.
3. Read both files back before reporting success.
4. Use the Agency MCP `submit_output` tool once for each required field. Submit status `created`,
   artifact_path {artifact_path!r}, artifact_contents {artifact_contents!r}, env_status `clean`,
   and codename {codename!r}. Encode each tool value as the required JSON literal.
5. After all tool calls succeed, make the final assistant message plain text containing
   `CODEX_E2E_COMPLETE`; do not return a JSON document.
""".strip()

        before_tokens = (ag.ctx.total_input_tokens, ag.ctx.total_output_tokens)
        first_result = _resolve_result(
            ag.run(structured_skill, agdata(instruction=first_prompt)), "structured turn"
        )
        first_history = ag.ctx.get_resolved_messages()
        _require(first_result.get("status") == "created", "submit_output returned wrong status")
        _require(
            first_result.get("artifact_path") == artifact_path,
            "submit_output returned the wrong artifact path",
        )
        _require(
            first_result.get("artifact_contents") == artifact_contents,
            "submit_output returned contents that differ from the workspace file contract",
        )
        _require(first_result.get("env_status") == "clean", "Codex reported secret env exposure")
        _require(first_result.get("codename") == codename, "submit_output returned wrong codename")
        _require(
            first_sandbox.read_file(artifact_path) == artifact_contents,
            "independent sandbox read found incorrect artifact contents",
        )
        _require(
            first_sandbox.read_file(env_status_path) == "clean\n",
            "Codex's child shell observed a protected environment variable name",
        )
        final_message = _assistant_text(first_history)
        _require(final_message, "the structured turn produced no final assistant text")
        _require(
            "CODEX_E2E_COMPLETE" in final_message and not _is_json_document(final_message),
            "structured MCP output was not followed by the requested non-JSON final message",
        )
        first_record = _validated_session_record(ag)
        first_thread_id = first_record["session_id"]
        _require(
            ag.ctx.total_input_tokens > before_tokens[0]
            and ag.ctx.total_output_tokens > before_tokens[1],
            "Codex token usage was not propagated into Agency context",
        )
        request_count = _assert_proxy_dispatch(terminus, request_count, env["LLM_MODEL"])
        _assert_registry_cleanup(terminus, mcp_server, profiler, registry_baseline)
        _assert_no_config_home(first_sandbox)
        first_sandbox.destroy()
        print("Codex E2E: workspace, proxy, MCP, auth isolation, and session capture passed")

        # A fresh container has no prior CODEX_HOME.  Reusing the exact same
        # thread id therefore proves Agency restored the captured rollout.
        second_sandbox = _new_sandbox(cfg, f"codex_e2e_resume_{nonce}", tracked)
        _require(
            _codex_version_in(second_sandbox) == codex_version,
            "the fresh container has a different Codex CLI version",
        )
        ag.sandbox = second_sandbox
        second_result = _resolve_result(
            ag.run(
                recall_skill,
                agdata(
                    instruction=(
                        "What deployment codename did I give you in the previous turn? Reply "
                        "with the exact codename and no explanation."
                    )
                ),
            ),
            "native resume turn",
        )
        _require(codename in second_result.get("result", ""), "native resume lost the codename")
        second_record = _validated_session_record(ag)
        _require(
            second_record["session_id"] == first_thread_id,
            "the second turn started a new Codex thread instead of resuming the captured one",
        )
        request_count = _assert_proxy_dispatch(terminus, request_count, env["LLM_MODEL"])
        _assert_registry_cleanup(terminus, mcp_server, profiler, registry_baseline)
        _assert_no_config_home(second_sandbox)
        second_sandbox.destroy()
        print("Codex E2E: native rollout resume across a fresh container passed")

        # Deliberately invalidate the compatibility guard.  The adapter must
        # discard this optimization, start a new Codex thread, and include
        # Agency's portable context so recall still succeeds.
        stale_thread_id = second_record["session_id"]
        stale_version = f"deliberately-stale-{nonce}"
        second_record["codex_version"] = stale_version
        third_sandbox = _new_sandbox(cfg, f"codex_e2e_fallback_{nonce}", tracked)
        ag.sandbox = third_sandbox
        third_result = _resolve_result(
            ag.run(
                recall_skill,
                agdata(
                    instruction=(
                        "Again, state the exact deployment codename from our earlier context and "
                        "nothing else."
                    )
                ),
            ),
            "portable-history fallback turn",
        )
        _require(
            codename in third_result.get("result", ""),
            "portable Agency history did not preserve the codename",
        )
        third_record = _validated_session_record(ag)
        _require(
            third_record["session_id"] != stale_thread_id,
            "a deliberately stale Codex session was resumed instead of discarded",
        )
        _require(
            third_record["codex_version"] != stale_version,
            "the stale session compatibility marker was not replaced",
        )
        request_count = _assert_proxy_dispatch(terminus, request_count, env["LLM_MODEL"])
        _assert_registry_cleanup(terminus, mcp_server, profiler, registry_baseline)
        _assert_no_config_home(third_sandbox)
        print("Codex E2E: stale-session portable-history fallback passed")
        print(f"Codex E2E: PASS ({request_count} total process-local proxy dispatches observed)")
    finally:
        _destroy_all(tracked)
        _assert_registry_cleanup(terminus, mcp_server, profiler, registry_baseline)


if __name__ == "__main__":
    main()
