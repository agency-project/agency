"""Codex CLI backend.

Structural implementation only -- no `codex` binary was installable in the
environment this was developed in (no network path to it was verified),
so this backend's `execute()` follows the exact same shape as
`_ClaudeCodeBackend`/`_OpencodeBackend` (isolated config home,
`agproxy_ptrace` launch, output-schema recovery) but is exercised only by
mocked tests (see tests/agharness_internal/agharness_backends/test_codex.py), never against
a live process.

Codex speaks the OpenAI Responses API only (`wire_api="chat"` was removed
upstream) -- `agproxy_llm`'s only implemented route is chat-completions
passthrough, so `gateway_mode="translate"` is mandatory here, not optional
the way it is for opencode/Claude Code's matched-format cases. That
Responses-API adapter is NOT implemented (same documented gap as Claude
Code's Messages-API adapter) -- see docs/Design_harness_integration.md's
Component 1 and agproxy_llm.py's module docstring. Until it exists, this
backend does not override the harness's LLM endpoint at all.
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


def codex_available() -> bool:
    return shutil.which("codex") is not None


class _CodexBackend(agharness_backend):
    _DEFAULT_BINARY = "codex"
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
            return agerror(f"codex binary {binary!r} not found on PATH"), prev_ctx, [sys_msg]

        config_home = agharness.materialize_config_home(ag, token="", base_url="")
        try:
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
            }

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
