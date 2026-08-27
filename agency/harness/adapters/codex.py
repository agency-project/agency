"""Codex CLI harness adapter.

Builds an isolated Codex configuration, launches the CLI through the sandbox
daemon, and normalizes its JSONL output.
"""

from __future__ import annotations

import json
import shutil
from .base import AdapterRuntime, AttemptResult, agharness_backend


def codex_available() -> bool:
    return shutil.which("codex") is not None


class _CodexBackend(agharness_backend):
    _DEFAULT_BINARY = "codex"
    _DEFAULT_TIMEOUT_S = 600
    _PROVIDER_NAME = "agency-proxy"
    _ENV_KEY_NAME = "AGENCY_PROXY_API_KEY"

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

        if runtime.suppress_builtin_tools:
            return AttemptResult(
                ok=False, error_message="codex does not support replace_tools=[] in daemon mode"
            )

        binary = self.binary_path or self._DEFAULT_BINARY
        resolved = shutil.which(binary)
        if resolved is None:
            return AttemptResult(
                ok=False, error_message=f"codex binary {binary!r} not found on PATH"
            )

        config_home = agharness.materialize_config_home(
            runtime.engine_name, runtime.token, runtime.harness_base_url
        )
        try:
            self._write_codex_config(
                config_home, runtime.harness_base_url, runtime.model or "default"
            )

            # CODEX_HOME already isolates both config and state. Loading that
            # config is required for the Agency provider and MCP server.
            argv = [
                resolved,
                "exec",
                "--json",
                "--strict-config",
                "--skip-git-repo-check",
                "--ignore-rules",
                "--ephemeral",
                prompt,
            ]
            envp = {
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "CODEX_HOME": str(config_home),
                # Referenced by config.toml's `env_key` -- Codex reads the
                # provider's API key from the env var *named* there, not
                # from an inline value in config.toml.
                self._ENV_KEY_NAME: runtime.token,
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
                ok=False, error_message=f"codex exited with code {rc}: {stderr or stdout}"
            )

        unsupported = self._parse_unsupported_tool_events(stdout)
        if unsupported:
            return AttemptResult(
                ok=False,
                error_message=(
                    "codex used tools outside the harness contract: "
                    + ", ".join(sorted(unsupported))
                ),
            )

        final_text = self._parse_output_events(stdout)
        usage = self._parse_usage_events(stdout)
        return AttemptResult(
            ok=True,
            final_text=final_text,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
        )

    def _write_codex_config(self, config_home, base_url: str, model: str) -> None:
        quote = json.dumps
        toml_text = (
            f"model = {quote(model)}\n"
            f"model_provider = {quote(self._PROVIDER_NAME)}\n"
            f'approval_policy = "never"\n'
            f'sandbox_mode = "read-only"\n'
            f'web_search = "disabled"\n'
            f"\n"
            f"[agents]\n"
            f"enabled = false\n"
            f"\n"
            f"[features]\n"
            f"remote_plugin = false\n"
            f"shell_tool = false\n"
            f"unified_exec = false\n"
            f"\n"
            f"[model_providers.{self._PROVIDER_NAME}]\n"
            f'name = "Agency Proxy"\n'
            f"base_url = {quote(f'{base_url}/v1')}\n"
            f"env_key = {quote(self._ENV_KEY_NAME)}\n"
            f'wire_api = "responses"\n'
            f"\n"
            f"[mcp_servers.agency]\n"
            f"url = {quote(f'{base_url}/mcp')}\n"
            f"bearer_token_env_var = {quote(self._ENV_KEY_NAME)}\n"
            f"required = true\n"
            f'default_tools_approval_mode = "approve"\n'
        )
        (config_home / "config.toml").write_text(toml_text)

    @staticmethod
    def _parse_usage_events(stdout: str) -> dict[str, int]:
        usage = {"input_tokens": 0, "output_tokens": 0}
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "turn.completed":
                continue
            reported = event.get("usage") or {}
            usage["input_tokens"] += int(reported.get("input_tokens") or 0)
            usage["output_tokens"] += int(reported.get("output_tokens") or 0)
        return usage

    @staticmethod
    def _parse_unsupported_tool_events(stdout: str) -> set[str]:
        unsupported = set()
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item") if isinstance(event, dict) else None
            item_type = item.get("type") if isinstance(item, dict) else None
            if item_type in {"command_execution", "file_change", "web_search"}:
                unsupported.add(item_type)
        return unsupported

    @staticmethod
    def _parse_output_events(stdout: str) -> str:
        """Best-effort extraction of the final `agent_message` item's text
        from `codex exec --json`'s NDJSON event stream."""
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
