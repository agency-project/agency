"""Claude Code backend.

LLM routing: `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` are pointed at
`agproxy_llm`'s `/v1/messages` route (Anthropic Messages API,
`gateway_mode="translate"` -- see agproxy_llm.py/agproxy_llm_adapters.py),
which reshapes the request into the exact `client.chat.completions.create()`
call every other backend uses and routes it through this agent's own
configured `agConfig` backend. The host's own real Anthropic credentials
(API key, OAuth login, Bedrock env) are deliberately NOT forwarded to the
launched process -- every `claude`-driven agent's LLM traffic goes through
agency's own backend choice, not whatever this host happens to have lying
around. Everything else (isolated config home, agproxy_ptrace launch +
tracing, output-schema recovery) works the same as the opencode backend and
was verified against the real `claude` CLI (v2.1.212) during development.
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


def claude_code_available() -> bool:
    return shutil.which("claude") is not None


class _ClaudeCodeBackend(agharness_backend):
    _DEFAULT_BINARY = "claude"
    _DEFAULT_TIMEOUT_S = 600

    def execute(
        self,
        ag: "agent",
        prev_ctx: "agcontext",
        skill_input: agdata,
        max_steps: "int | None",
        *,
        skill: "agskill",
    ) -> "tuple[agdata, agcontext, list[dict]]":
        from ... import agharness
        from ..agproxy_llm import get_shared_gateway
        from ..agproxy_ptrace import agProxyPtrace, wire_to_sandbox

        sys_msg = {"role": "system", "content": skill._build_system_prompt()}

        binary = self.binary_path or self._DEFAULT_BINARY
        resolved = shutil.which(binary)
        if resolved is None:
            return agerror(f"claude binary {binary!r} not found on PATH"), prev_ctx, [sys_msg]

        gateway = get_shared_gateway(ag.agconfig)
        token = uuid.uuid4().hex
        gateway.register(token, ag)

        config_home = agharness.materialize_config_home(ag, token, gateway.base_url)
        try:
            prompt = agharness.build_user_turn_prompt(skill, skill_input)
            if not isinstance(prompt, str):
                prompt = json.dumps(prompt)
            extra = agharness.build_output_format_instruction(skill)
            if extra:
                prompt = prompt + extra

            # --setting-sources "" -- load none of the user/project/local
            # settings that would normally apply, so this run doesn't
            # inherit the caller's own Claude Code configuration (matching
            # the same isolated-config-home intent as opencode's own
            # OPENCODE_CONFIG, just via a flag here instead of a file,
            # since --settings/--setting-sources are what Claude Code
            # itself provides for this).
            argv = [
                resolved,
                "-p",
                "--output-format",
                "json",
                "--setting-sources",
                "",
                prompt,
            ]
            import os

            envp = {
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                # Point Claude Code's own LLM traffic at agproxy_llm's
                # translated Anthropic Messages route instead of any real
                # Anthropic endpoint -- ANTHROPIC_AUTH_TOKEN sends this
                # token as a bearer `Authorization` header, which
                # `_extract_bearer_token` (agproxy_llm.py) reads to look up
                # this launch's agent. Real host credentials (API key,
                # OAuth login, Bedrock env) are deliberately NOT forwarded:
                # every claude-driven agent's LLM calls must go through
                # this agent's own configured agConfig backend, not
                # whatever this host happens to have lying around.
                "ANTHROPIC_BASE_URL": gateway.base_url,
                "ANTHROPIC_AUTH_TOKEN": token,
                # Also explicitly unset so the CLI can't fall back to a
                # locally-configured Bedrock/API-key credential path.
                "CLAUDE_CODE_USE_BEDROCK": "0",
            }
            # Deliberately does NOT override HOME: Claude Code's OAuth
            # credentials live under the real $HOME (~/.claude/
            # .credentials.json), and --setting-sources "" above is
            # already what provides the "don't inherit CLAUDE.md/settings"
            # isolation this backend needs -- overriding HOME too would
            # additionally (and unintentionally) cut off the real login,
            # forcing "Not logged in" for every run (hit and fixed during
            # development against the real CLI). This doesn't matter for
            # authentication anymore since ANTHROPIC_AUTH_TOKEN above
            # always takes precedence over OAuth login, but HOME is still
            # left alone since other CLI state may expect it.
            if "HOME" in os.environ:
                envp["HOME"] = os.environ["HOME"]

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
                agerror(f"claude exited with code {rc}: {stderr or stdout}"),
                prev_ctx,
                [sys_msg],
            )

        final_text, usage = self._parse_result_json(stdout)
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
        prev_ctx.messages = [user_msg, assistant_msg]
        return result, prev_ctx, [sys_msg, user_msg, assistant_msg]

    @staticmethod
    def _parse_result_json(stdout: str) -> "tuple[str, dict]":
        """Parse `claude -p --output-format json`'s single JSON result
        object -- `{"result": "...", "usage": {...}, ...}` (verified
        directly against the real CLI, v2.1.212, during development)."""
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return stdout.strip(), {}
        if not isinstance(payload, dict):
            return stdout.strip(), {}
        text = payload.get("result", "")
        usage = payload.get("usage", {}) or {}
        return text, usage


__all__ = ["_ClaudeCodeBackend", "claude_code_available"]
