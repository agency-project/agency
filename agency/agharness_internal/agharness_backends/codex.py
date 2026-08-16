"""Codex CLI backend.

Structural implementation only -- no `codex` binary was installable in the
environment this was developed in (no network path to it was verified),
so this backend's `execute()` follows the exact same shape as
`_ClaudeCodeBackend`/`_OpencodeBackend` (isolated config home,
`agproxy_ptrace` launch, output-schema recovery) but is exercised only by
mocked tests (see tests/agharness_internal/agharness_backends/test_codex.py), never against
a live process.

Codex speaks the OpenAI Responses API only (`wire_api="chat"` was removed
upstream), routed at `agproxy_llm`'s `/v1/responses` route (`gateway_mode=
"translate"` -- see agproxy_llm.py/agproxy_llm_adapters.py's Responses-API
adapter). This backend writes a `[model_providers.agency-proxy]` block into
an isolated `CODEX_HOME/config.toml` pointing `base_url` at the gateway and
selects it as the active provider; the gateway token is passed via an env
var named by `env_key` (Codex's config schema wants an env-var *name*, not
an inline key, per its documented config.toml shape) -- the Responses-API
wire adapter itself is unverified against a live `codex` binary (same
"no binary available" caveat as the rest of this module).
"""

from __future__ import annotations

import json
import shutil
import uuid
from typing import TYPE_CHECKING

from ...agdata import agdata, agerror
from .base import agharness_backend

if TYPE_CHECKING:
    from ...agent import agent
    from ...agcontext import agcontext
    from ...agskill import agskill


def codex_available() -> bool:
    return shutil.which("codex") is not None


class _CodexBackend(agharness_backend):
    _DEFAULT_BINARY = "codex"
    _DEFAULT_TIMEOUT_S = 600
    _PROVIDER_NAME = "agency-proxy"
    _ENV_KEY_NAME = "AGENCY_PROXY_API_KEY"

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
        from ..agproxy_llm import get_shared_gateway

        messages = canonical_input or agharness.build_harness_messages(
            skill, prev_ctx, skill_input, file_notice=extra_system
        )
        sys_msg = {"role": "system", "content": messages.system_instructions}

        binary = self.binary_path or self._DEFAULT_BINARY
        in_container = agharness.is_container_backed(ag.sandbox)
        if in_container:
            resolved = agharness.resolve_harness_binary_in_container(ag.sandbox, binary)
        else:
            resolved = shutil.which(binary)
        if resolved is None:
            return agerror(f"codex binary {binary!r} not found on PATH"), prev_ctx, [sys_msg]

        gateway = get_shared_gateway(ag.agconfig)
        from ..agprof_ingest import get_shared_profiler_ingest

        profiler_ingest = get_shared_profiler_ingest()
        token = uuid.uuid4().hex
        gateway.register(token, ag)
        profiler_ingest.register(token, ag)

        if in_container:
            from ..agproxy_llm_in_container import ensure_agproxy_llm_in_container

            base_url = ensure_agproxy_llm_in_container(ag.sandbox, ag.agconfig)
            config_home = agharness.materialize_config_home_in_container(ag, ag.sandbox, token)
        else:
            base_url = gateway.base_url
            config_home = agharness.materialize_config_home(ag, token, base_url)
        try:
            model = getattr(ag.llm.backend, "model", "") or "default"
            self._write_codex_config(
                config_home, base_url, model, sandbox=ag.sandbox if in_container else None
            )

            sessions = getattr(ag, "_harness_sessions", None)
            prior_session = sessions.get("codex", {}) if isinstance(sessions, dict) else {}
            resume_session_id = (
                prior_session.get("session_id")
                if prior_session.get("agcontext_revision") == prev_ctx.revision
                else None
            )
            prompt = agharness.render_harness_messages(
                messages,
                include_previous_context=resume_session_id is None,
            )
            argv = [resolved, "exec", "--json", "--strict-config", "--cd", "/workspace"]
            if resume_session_id:
                argv += ["--resume", resume_session_id]
            argv.append("-")
            envp = {
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "CODEX_HOME": str(config_home),
                # Referenced by config.toml's `env_key` -- Codex reads the
                # provider's API key from the env var *named* there, not
                # from an inline value in config.toml.
                self._ENV_KEY_NAME: token,
            }

            stdout, stderr, rc = agharness.run_harness_cli(
                ag,
                argv,
                envp,
                stdin=prompt,
                timeout_s=self._DEFAULT_TIMEOUT_S,
                cwd="/workspace" if ag.sandbox is not None else str(config_home),
            )
        finally:
            gateway.unregister(token)
            profiler_ingest.unregister(token)
            if in_container:
                agharness.cleanup_config_home_in_container(ag.sandbox, config_home)
            else:
                agharness.cleanup_config_home(config_home)

        if rc != 0:
            return (
                agerror(f"codex exited with code {rc}: {stderr or stdout}"),
                prev_ctx,
                [sys_msg],
            )

        final_text, session_id = self._parse_output(stdout)
        user_msg = agharness.harness_user_message(messages)
        assistant_msg = {"role": "assistant", "content": final_text}

        result = agharness.finalize_harness_result(
            agharness.HarnessResult(final_text=final_text), skill, ag.sandbox
        )

        if session_id:
            if isinstance(getattr(ag, "_harness_sessions", None), dict):
                ag._harness_sessions["codex"] = {
                    "session_id": session_id,
                    "agcontext_revision": prev_ctx.revision + 1,
                }
            self.session_resume_id = session_id

        prev_ctx.messages = [*messages.previous_context, user_msg, assistant_msg]
        return result, prev_ctx, [sys_msg, user_msg, assistant_msg]

    def _write_codex_config(self, config_home, base_url: str, model: str, *, sandbox=None) -> None:
        toml_text = (
            f'model = "{model}"\n'
            f'model_provider = "{self._PROVIDER_NAME}"\n'
            f"\n"
            f"[model_providers.{self._PROVIDER_NAME}]\n"
            f'name = "Agency Proxy"\n'
            f'base_url = "{base_url}/v1"\n'
            f'env_key = "{self._ENV_KEY_NAME}"\n'
            f'wire_api = "responses"\n'
        )
        path = f"{config_home}/config.toml"
        if sandbox is not None:
            sandbox.write_file(path, toml_text)
        else:
            (config_home / "config.toml").write_text(toml_text)

    @staticmethod
    def _parse_output_events(stdout: str) -> str:
        """Best-effort extraction of the final `agent_message` item's text
        from `codex exec --json`'s NDJSON event stream -- unverified
        against a live run (no `codex` binary available), see this
        module's docstring."""
        return _CodexBackend._parse_output(stdout)[0]

    @staticmethod
    def _parse_output(stdout: str) -> "tuple[str, str | None]":
        last_text = ""
        session_id = None
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str):
                session_id = event["thread_id"]
            item = event.get("item") or {}
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text:
                    last_text = text
        if last_text:
            return last_text, session_id
        return stdout.strip(), session_id


__all__ = ["_CodexBackend", "codex_available"]
