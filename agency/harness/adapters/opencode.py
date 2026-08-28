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
and tested (see tests/harness/agharness_backends/test_opencode.py's mocked-launch
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
from .base import AdapterRuntime, AttemptResult, agharness_backend


def opencode_available() -> bool:
    return shutil.which("opencode") is not None


class _OpencodeBackend(agharness_backend):
    _DEFAULT_BINARY = "opencode"
    _PROVIDER_NAME = "agency-proxy"
    _DEFAULT_TIMEOUT_S = 600

    def run_daemon_attempt(
        self,
        runtime: AdapterRuntime,
        *,
        prompt: str,
        resume_session_id: "str | None",
        prior_session_blob: "bytes | None",
        max_steps: "int | None",
    ) -> AttemptResult:
        from .. import agharness
        from ..ptrace.supervisor import agProxyPtrace

        binary = self.binary_path or self._DEFAULT_BINARY
        resolved = shutil.which(binary)
        if resolved is None:
            return AttemptResult(
                ok=False, error_message=f"opencode binary {binary!r} not found on PATH"
            )

        config_home = agharness.materialize_config_home(
            runtime.engine_name, runtime.token, runtime.harness_base_url
        )
        try:
            self._write_opencode_config(
                config_home,
                runtime.harness_base_url,
                runtime.token,
                runtime.model or "default",
            )

            argv = [resolved, "run", "--format", "json", prompt]
            envp = {
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "HOME": str(config_home),
                "OPENCODE_CONFIG": str(config_home / "opencode.json"),
            }

            px = agProxyPtrace(runtime.agconfig)
            handle = px.launch(
                argv,
                envp,
                cwd=str(config_home),
                policy=runtime.syscall_policy,
                ag=None,
            )
            stdout, stderr, rc = handle.wait(timeout=self._DEFAULT_TIMEOUT_S)
        finally:
            agharness.cleanup_config_home(config_home)

        if rc != 0:
            return AttemptResult(
                ok=False,
                error_message=f"opencode exited with code {rc}: {stderr or stdout}",
            )

        final_text = self._parse_output_events(stdout)
        return AttemptResult(ok=True, final_text=final_text)

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
