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
    ) -> "tuple[agdata, agcontext, list[dict]]":
        from ... import agharness
        from ..agproxy_llm import get_shared_gateway
        from ..agproxy_ptrace import agProxyPtrace, wire_to_sandbox

        sys_msg = {"role": "system", "content": skill._build_system_prompt(extra_system)}

        binary = self.binary_path or self._DEFAULT_BINARY
        resolved = shutil.which(binary)
        if resolved is None:
            return agerror(f"codex binary {binary!r} not found on PATH"), prev_ctx, [sys_msg]

        gateway = get_shared_gateway(ag.agconfig)
        token = uuid.uuid4().hex
        gateway.register(token, ag)

        config_home = agharness.materialize_config_home(ag, token, gateway.base_url)
        try:
            model = getattr(ag.llm.backend, "model", "") or "default"
            self._write_codex_config(config_home, gateway.base_url, model)

            prompt = agharness.build_user_turn_prompt(skill, skill_input)
            if not isinstance(prompt, str):
                prompt = json.dumps(prompt)
            extra = agharness.build_output_format_instruction(skill)
            if extra:
                prompt = prompt + extra

            # --ignore-user-config keeps this run from inheriting the
            # caller's own ~/.codex/config.toml, matching the same
            # isolated-config-home intent as the other two backends.
            argv = [resolved, "exec", "--json", "--ignore-user-config", prompt]
            envp = {
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "CODEX_HOME": str(config_home),
                # Referenced by config.toml's `env_key` -- Codex reads the
                # provider's API key from the env var *named* there, not
                # from an inline value in config.toml.
                self._ENV_KEY_NAME: token,
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
                agerror(f"codex exited with code {rc}: {stderr or stdout}"),
                prev_ctx,
                [sys_msg],
            )

        final_text = self._parse_output_events(stdout)
        user_msg = {"role": "user", "content": prompt}
        assistant_msg = {"role": "assistant", "content": final_text}

        if skill.output_schema is not None and skill.output_schema.raw_key() is None:
            result, _paths = skill.output_schema.validate_and_recover(final_text, ag.sandbox)
        else:
            out_key = skill.output_schema.raw_key() if skill.output_schema is not None else "result"
            result = agdata(**{out_key: final_text})

        prev_ctx.messages = [user_msg, assistant_msg]
        return result, prev_ctx, [sys_msg, user_msg, assistant_msg]

    def _write_codex_config(self, config_home, base_url: str, model: str) -> None:
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
        (config_home / "config.toml").write_text(toml_text)

    @staticmethod
    def _parse_output_events(stdout: str) -> str:
        """Best-effort extraction of the final `agent_message` item's text
        from `codex exec --json`'s NDJSON event stream -- unverified
        against a live run (no `codex` binary available), see this
        module's docstring."""
        last_text = ""
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
            item = event.get("item") or {}
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text:
                    last_text = text
        if last_text:
            return last_text
        return stdout.strip()


__all__ = ["_CodexBackend", "codex_available"]
