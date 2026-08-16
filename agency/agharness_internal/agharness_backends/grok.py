"""Grok Build (xAI) backend -- https://grok.com/build, binary name `grok`.

Like opencode (and unlike Claude Code/Codex), Grok Build's model config
supports `api_backend = "chat_completions"` per `[model.<name>]` block in
its `config.toml` -- i.e. it can be pointed at an arbitrary OpenAI-
compatible endpoint speaking plain chat-completions, which is exactly
`agproxy_llm`'s existing passthrough route with zero translation. This is
the second backend (after opencode) that routes its LLM traffic through
`agproxy_llm` rather than leaving the harness's endpoint untouched.

CAVEAT: no `grok` binary was installed in the environment this was
developed in (installing it requires running xAI's `curl | bash` install
script, a real download-and-execute-from-the-internet action deliberately
not taken without being asked first -- see docs/agharness.md) -- this
backend's orchestration logic (config-home isolation, agproxy_ptrace
launch, agproxy_llm token registration, output-schema recovery) follows
the exact same tested shape as `opencode.py`/`claude_code.py`, but the
config.toml schema and `--output-format json`'s exact field names are
implemented from xAI's published docs (docs.x.ai/build, the xai-org/
grok-build repo's user guide), not verified against a live run.
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


def grok_available() -> bool:
    return shutil.which("grok") is not None


def _toml_string(value: str) -> str:
    """Quote a string for inclusion in a hand-written TOML file -- only
    the escapes actually needed for the values this module writes
    (prompt text never goes through here; only config values do)."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class _GrokBackend(agharness_backend):
    _DEFAULT_BINARY = "grok"
    _MODEL_NAME = "agency-proxy"
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
            return agerror(f"grok binary {binary!r} not found on PATH"), prev_ctx, [sys_msg]

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
            self._write_grok_config(
                config_home, base_url, token, model, sandbox=ag.sandbox if in_container else None
            )

            prompt = agharness.render_harness_messages(messages)
            prompt_path = f"{config_home}/task.txt"
            if in_container:
                ag.sandbox.write_file(prompt_path, prompt)
                import shlex

                ag.sandbox.exec(f"chmod 600 {shlex.quote(prompt_path)}", workdir="/")
            else:
                from pathlib import Path

                Path(prompt_path).write_text(prompt)
                Path(prompt_path).chmod(0o600)
            argv = [
                resolved,
                "-p",
                f"Read the complete task from {prompt_path} and follow it.",
                "--output-format",
                "json",
            ]
            sessions = getattr(ag, "_harness_sessions", None)
            prior_session = sessions.get("grok", {}) if isinstance(sessions, dict) else {}
            resume_session_id = prior_session.get("session_id") or self.session_resume_id
            if resume_session_id:
                argv += ["--resume", resume_session_id]
            envp = {
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                # GROK_HOME redirects the *entire* config directory (config.toml,
                # auth.json, sessions/) -- the closest analog to Codex's CODEX_HOME,
                # and the documented isolation mechanism here: xAI's docs don't
                # expose a Claude-Code-style "--setting-sources ''"/"--ignore-user-
                # config" flag, so this is what actually keeps a scripted run from
                # touching (or reading) the caller's real ~/.grok.
                "GROK_HOME": str(config_home),
            }

            stdout, stderr, rc = agharness.run_harness_cli(
                ag,
                argv,
                envp,
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
                agerror(f"grok exited with code {rc}: {stderr or stdout}"),
                prev_ctx,
                [sys_msg],
            )

        final_text, usage, session_id = self._parse_result_json(stdout)
        user_msg = agharness.harness_user_message(messages)
        assistant_msg = {"role": "assistant", "content": final_text}

        if skill.output_schema is not None and skill.output_schema.raw_key() is None:
            result, _paths = skill.output_schema.validate_and_recover(final_text, ag.sandbox)
        else:
            out_key = skill.output_schema.raw_key() if skill.output_schema is not None else "result"
            result = agdata(**{out_key: final_text})

        if usage:
            prev_ctx.total_input_tokens += usage.get("input_tokens", 0)
            prev_ctx.total_output_tokens += usage.get("output_tokens", 0)
        if session_id:
            # Multi-turn resume (`grok -r <id>`) is future work -- not
            # threaded through to a second execute() call yet, just
            # recorded so that wiring is a config-read away rather than a
            # new field.
            self.session_resume_id = session_id
            if isinstance(getattr(ag, "_harness_sessions", None), dict):
                ag._harness_sessions["grok"] = {"session_id": session_id}
        prev_ctx.messages = [*messages.previous_context, user_msg, assistant_msg]
        return result, prev_ctx, [sys_msg, user_msg, assistant_msg]

    def _write_grok_config(
        self, config_home, base_url: str, token: str, model: str, *, sandbox=None
    ) -> None:
        # config.toml, per docs.x.ai/build's configuration guide: a
        # [model.<name>] block with base_url/api_key/api_backend, and a
        # top-level `model` key selecting the active one -- api_backend =
        # "chat_completions" is what makes this usable via agproxy_llm's
        # existing passthrough route with no translation, the same as
        # opencode's @ai-sdk/openai-compatible provider.
        config_toml = (
            f"model = {_toml_string(self._MODEL_NAME)}\n\n"
            f"[model.{self._MODEL_NAME}]\n"
            f"model = {_toml_string(model)}\n"
            f"base_url = {_toml_string(f'{base_url}/v1')}\n"
            f"api_key = {_toml_string(token)}\n"
            f'api_backend = "chat_completions"\n'
        )
        path = f"{config_home}/config.toml"
        if sandbox is not None:
            sandbox.write_file(path, config_toml)
        else:
            (config_home / "config.toml").write_text(config_toml)

    @staticmethod
    def _parse_result_json(stdout: str) -> "tuple[str, dict, str | None]":
        """Parse `grok -p ... --output-format json`'s single JSON result
        object -- `{"text": "...", "usage": {...}, "sessionId": "...", ...}`
        per xAI's published headless-mode docs (not verified against a
        live run -- see this module's docstring)."""
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return stdout.strip(), {}, None
        if not isinstance(payload, dict):
            return stdout.strip(), {}, None
        text = payload.get("text", "")
        usage = payload.get("usage", {}) or {}
        session_id = payload.get("sessionId")
        return text, usage, session_id


__all__ = ["_GrokBackend", "grok_available"]
