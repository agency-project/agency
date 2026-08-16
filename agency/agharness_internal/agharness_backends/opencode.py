"""opencode backend -- the reference concrete `agharness_backend`
implementation, and the smallest real end-to-end harness slice per
docs/Design_harness_integration.md's build order: opencode's
`@ai-sdk/openai-compatible` provider already speaks plain OpenAI
chat-completions, matching `agproxy_llm`'s passthrough route with zero
translation, and `opencode run --format json` is a simple headless
invocation.

CAVEAT: no `opencode` binary is installable in the environment this was
developed in (opencode requires Node/Bun, neither available) -- this
backend's orchestration logic (config-home isolation, agproxy_ptrace
launch, agproxy_llm token registration, output-schema recovery) is real
and tested (see tests/agharness_internal/agharness_backends/test_opencode.py's mocked-launch
tests), but the exact shape of `--format json`'s event stream and the
`opencode.json` provider-block schema are implemented from documented
behavior, not verified against a live run. `_parse_output_events` is
deliberately isolated and defensive (best-effort per-line JSON parsing,
falls back to raw text) so CLI drift or a wrong assumption here is a
contained, fixable gap rather than a crash.
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


def opencode_available() -> bool:
    return shutil.which("opencode") is not None


class _OpencodeBackend(agharness_backend):
    _DEFAULT_BINARY = "opencode"
    _PROVIDER_NAME = "agency-proxy"
    _DEFAULT_TIMEOUT_S = 600

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
            return agerror(f"opencode binary {binary!r} not found on PATH"), prev_ctx, [sys_msg]

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
            self._write_opencode_config(
                config_home, base_url, token, model, sandbox=ag.sandbox if in_container else None
            )

            sessions = getattr(ag, "_harness_sessions", None)
            prior_session = sessions.get("opencode", {}) if isinstance(sessions, dict) else {}
            resume_session_id = (
                prior_session.get("session_id")
                if prior_session.get("agcontext_revision") == prev_ctx.revision
                else None
            )
            prompt = agharness.render_harness_messages(
                messages,
                include_previous_context=resume_session_id is None,
            )
            argv = [resolved, "run", "--format", "json"]
            if resume_session_id:
                argv += ["--session", resume_session_id]
            envp = {
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "HOME": str(config_home),
                "OPENCODE_CONFIG": f"{config_home}/opencode.json",
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
                agerror(f"opencode exited with code {rc}: {stderr or stdout}"),
                prev_ctx,
                [sys_msg],
            )

        final_text, session_id = self._parse_output(stdout)
        user_msg = agharness.harness_user_message(messages)
        assistant_msg = {"role": "assistant", "content": final_text}

        if skill.output_schema is not None and skill.output_schema.raw_key() is None:
            result, _paths = skill.output_schema.validate_and_recover(final_text, ag.sandbox)
        else:
            out_key = skill.output_schema.raw_key() if skill.output_schema is not None else "result"
            result = agdata(**{out_key: final_text})

        if session_id:
            if isinstance(getattr(ag, "_harness_sessions", None), dict):
                ag._harness_sessions["opencode"] = {
                    "session_id": session_id,
                    "agcontext_revision": prev_ctx.revision + 1,
                }
            self.session_resume_id = session_id

        prev_ctx.messages = [*messages.previous_context, user_msg, assistant_msg]
        return result, prev_ctx, [sys_msg, user_msg, assistant_msg]

    def _write_opencode_config(
        self, config_home, base_url: str, token: str, model: str, *, sandbox=None
    ) -> None:
        config = {
            "provider": {
                self._PROVIDER_NAME: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Agency Proxy",
                    "options": {"baseURL": f"{base_url}/v1", "apiKey": token},
                    "models": {model: {"name": model}},
                }
            },
            "model": f"{self._PROVIDER_NAME}/{model}",
        }
        import json

        content = json.dumps(config)
        path = f"{config_home}/opencode.json"
        if sandbox is not None:
            sandbox.write_file(path, content)
        else:
            (config_home / "opencode.json").write_text(content)

    @staticmethod
    def _parse_output_events(stdout: str) -> str:
        """Best-effort extraction of the final assistant text from
        `opencode run --format json`'s output -- see this module's
        docstring on why this is deliberately defensive rather than a
        strict schema parse."""
        return _OpencodeBackend._parse_output(stdout)[0]

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
            for key in ("sessionID", "session_id", "sessionId"):
                value = event.get(key)
                if isinstance(value, str) and value:
                    session_id = value
            for key in ("text", "content", "result", "message"):
                value = event.get(key)
                if isinstance(value, str) and value:
                    last_text = value
        if last_text:
            return last_text, session_id
        # Fall back to the raw stdout itself (e.g. a plain-text response
        # with no JSON structure at all) rather than silently returning
        # an empty string.
        return stdout.strip(), session_id


__all__ = ["_OpencodeBackend", "opencode_available"]
