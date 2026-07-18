"""Claude Code backend.

CAVEAT: `agproxy_llm` (see agproxy_llm.py) only implements a
chat-completions passthrough route today -- Claude Code speaks the
Anthropic Messages API (`POST /v1/messages`), a different wire format, so
`gateway_mode="translate"`'s Messages-API adapter is NOT implemented yet.
Until it is, this backend does not override `ANTHROPIC_BASE_URL`/
`ANTHROPIC_AUTH_TOKEN` at all -- the launched `claude` process uses
whatever credentials/endpoint it's already configured with on this host
(its own `~/.claude/.credentials.json` or its own env), same as running it
by hand. LLM routing through agency's own `agConfig` backend choice is a
real, open gap for this backend specifically -- see
docs/Design_harness_integration.md's Component 1. Everything else
(isolated config home, agproxy_ptrace launch + tracing, output-schema
recovery) works the same as the opencode backend and was verified against
the real `claude` CLI (v2.1.212) during development.
"""

from __future__ import annotations

import json
import shutil
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
        from ..agproxy_ptrace import agProxyPtrace, wire_to_sandbox

        sys_msg = {"role": "system", "content": skill._build_system_prompt()}

        binary = self.binary_path or self._DEFAULT_BINARY
        resolved = shutil.which(binary)
        if resolved is None:
            return agerror(f"claude binary {binary!r} not found on PATH"), prev_ctx, [sys_msg]

        config_home = agharness.materialize_config_home(ag, token="", base_url="")
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

            envp = {"PATH": "/usr/bin:/bin:/usr/local/bin"}
            # Deliberately does NOT override HOME: Claude Code's OAuth
            # credentials live under the real $HOME (~/.claude/
            # .credentials.json), and --setting-sources "" above is
            # already what provides the "don't inherit CLAUDE.md/settings"
            # isolation this backend needs -- overriding HOME too would
            # additionally (and unintentionally) cut off the real login,
            # forcing "Not logged in" for every run (hit and fixed during
            # development against the real CLI).
            if "HOME" in os.environ:
                envp["HOME"] = os.environ["HOME"]
            # Carry over the real credential env vars this host's `claude`
            # already relies on (API key / OAuth token paths, etc.) -- see
            # this module's docstring: LLM routing through agproxy_llm is
            # not implemented for this backend yet, so the harness must be
            # left free to authenticate exactly as it would run by hand.
            for key in (
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN",
                "ANTHROPIC_BASE_URL",
                "CLAUDE_CODE_USE_BEDROCK",
                "AWS_REGION",
                "AWS_BEARER_TOKEN_BEDROCK",
            ):
                if key in os.environ:
                    envp[key] = os.environ[key]

            px = agProxyPtrace(ag.agconfig)
            policy = agharness.default_policy(ag)
            handle = px.launch(argv, envp, cwd=str(config_home), policy=policy, ag=ag)
            if ag.sandbox is not None:
                wire_to_sandbox(handle, ag.sandbox)

            stdout, stderr, rc = handle.wait(timeout=self._DEFAULT_TIMEOUT_S)
        finally:
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
