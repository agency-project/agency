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
    ) -> "tuple[agdata, agcontext, list[dict]]":
        from ... import agharness
        from ..agproxy_llm import get_shared_gateway
        from ..agproxy_ptrace import agProxyPtrace, wire_to_sandbox

        sys_msg = {"role": "system", "content": skill._build_system_prompt(extra_system)}

        binary = self.binary_path or self._DEFAULT_BINARY
        resolved = shutil.which(binary)
        if resolved is None:
            return agerror(f"opencode binary {binary!r} not found on PATH"), prev_ctx, [sys_msg]

        gateway = get_shared_gateway(ag.agconfig)
        from ..agprof_ingest import get_shared_profiler_ingest

        profiler_ingest = get_shared_profiler_ingest()
        token = uuid.uuid4().hex
        gateway.register(token, ag)
        profiler_ingest.register(token, ag)

        config_home = agharness.materialize_config_home(ag, token, gateway.base_url)
        try:
            model = getattr(ag.llm.backend, "model", "") or "default"
            self._write_opencode_config(config_home, gateway.base_url, token, model)

            prompt = agharness.build_user_turn_prompt(skill, skill_input)
            if not isinstance(prompt, str):
                prompt = json.dumps(prompt)
            extra = agharness.build_output_format_instruction(skill)
            if extra:
                prompt = prompt + extra

            argv = [resolved, "run", "--format", "json", prompt]
            envp = {
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "HOME": str(config_home),
                "OPENCODE_CONFIG": str(config_home / "opencode.json"),
            }

            px = agProxyPtrace(ag.agconfig)
            policy = agharness.default_policy(ag)
            handle = px.launch(argv, envp, cwd=str(config_home), policy=policy, ag=ag)
            if ag.sandbox is not None:
                wire_to_sandbox(handle, ag.sandbox)

            stdout, stderr, rc = handle.wait(timeout=self._DEFAULT_TIMEOUT_S)
        finally:
            gateway.unregister(token)
            profiler_ingest.unregister(token)
            agharness.cleanup_config_home(config_home)

        if rc != 0:
            return (
                agerror(f"opencode exited with code {rc}: {stderr or stdout}"),
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

    def _write_opencode_config(self, config_home, base_url: str, token: str, model: str) -> None:
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
        (config_home / "opencode.json").write_text(json.dumps(config))

    @staticmethod
    def _parse_output_events(stdout: str) -> str:
        """Best-effort extraction of the final assistant text from
        `opencode run --format json`'s output -- see this module's
        docstring on why this is deliberately defensive rather than a
        strict schema parse."""
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
            for key in ("text", "content", "result", "message"):
                value = event.get(key)
                if isinstance(value, str) and value:
                    last_text = value
        if last_text:
            return last_text
        # Fall back to the raw stdout itself (e.g. a plain-text response
        # with no JSON structure at all) rather than silently returning
        # an empty string.
        return stdout.strip()


__all__ = ["_OpencodeBackend", "opencode_available"]
