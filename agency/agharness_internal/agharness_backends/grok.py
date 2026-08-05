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
    ) -> "tuple[agdata, agcontext, list[dict]]":
        from ... import agharness
        from ..agproxy_llm import get_shared_gateway
        from ..agproxy_ptrace import agProxyPtrace, wire_to_sandbox

        sys_msg = {"role": "system", "content": skill._build_system_prompt(extra_system)}

        binary = self.binary_path or self._DEFAULT_BINARY
        resolved = shutil.which(binary)
        if resolved is None:
            return agerror(f"grok binary {binary!r} not found on PATH"), prev_ctx, [sys_msg]

        gateway = get_shared_gateway(ag.agconfig)
        token = uuid.uuid4().hex
        gateway.register(token, ag)

        config_home = agharness.materialize_config_home(ag, token, gateway.base_url)
        try:
            model = getattr(ag.llm.backend, "model", "") or "default"
            self._write_grok_config(config_home, gateway.base_url, token, model)

            prompt = agharness.build_user_turn_prompt(skill, skill_input)
            if not isinstance(prompt, str):
                prompt = json.dumps(prompt)
            extra = agharness.build_output_format_instruction(skill)
            if extra:
                prompt = prompt + extra

            argv = [resolved, "-p", prompt, "--output-format", "json"]
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

            px = agProxyPtrace(ag.agconfig)
            policy = agharness.default_policy(ag)
            handle = px.launch(argv, envp, cwd=str(config_home), policy=policy, ag=ag)
            if ag.sandbox is not None:
                wire_to_sandbox(handle, ag.sandbox)

            stdout, stderr, rc = handle.wait(timeout=self._DEFAULT_TIMEOUT_S)
        finally:
            gateway.unregister(token)
            agharness.cleanup_config_home(config_home)

        if rc != 0:
            return (
                agerror(f"grok exited with code {rc}: {stderr or stdout}"),
                prev_ctx,
                [sys_msg],
            )

        final_text, usage, session_id = self._parse_result_json(stdout)
        user_msg = {"role": "user", "content": prompt}
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
        prev_ctx.messages = [user_msg, assistant_msg]
        return result, prev_ctx, [sys_msg, user_msg, assistant_msg]

    def _write_grok_config(self, config_home, base_url: str, token: str, model: str) -> None:
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
            f'base_url = {_toml_string(f"{base_url}/v1")}\n'
            f"api_key = {_toml_string(token)}\n"
            f'api_backend = "chat_completions"\n'
        )
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
