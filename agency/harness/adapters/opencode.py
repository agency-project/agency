"""opencode interactive PTY adapter; the model protocol is shared."""

from __future__ import annotations

import json
import shutil

from .agharness_backend import AdapterRuntime, AttemptResult, agharness_backend
from .openai_protocol import ChatCompletionsBackend
from .pty_drivers import OpencodeDriver, run_pty_attempt


def opencode_available() -> bool:
    return shutil.which("opencode") is not None


class _OpencodeBackend(ChatCompletionsBackend, agharness_backend):
    _DEFAULT_BINARY = "opencode"
    _PTY_DRIVER = OpencodeDriver
    _PROVIDER_NAME = "agency-proxy"

    def run_daemon_attempt(
        self,
        runtime: AdapterRuntime,
        *,
        prompt: str,
        resume_session_id: "str | None",
        prior_session_blob: "bytes | None",
        max_steps: "int | None",
    ) -> AttemptResult:
        return run_pty_attempt(
            self,
            runtime,
            prompt=prompt,
            resume_session_id=resume_session_id,
            prior_session_blob=prior_session_blob,
            max_steps=max_steps,
        )

    def _write_opencode_config(
        self,
        config_home,
        base_url: str,
        token: str,
        model: str,
        plugin_path,
        max_steps: "int | None",
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
            # Agency owns session titles. Keep
            # that auxiliary generation out of the invocation's model channel
            # and final-answer checkpoints (OpenCode's built-in title agent).
            "agent": {"title": {"disable": True}},
            # Per opencode.ai/docs/config's `plugin` array: local plugins are
            # referenced by file:// URL alongside npm-package/version specs.
            "plugin": [f"file://{plugin_path}"],
        }
        if max_steps is not None:
            config["agent"]["build"] = {"steps": max_steps}
        (config_home / "opencode.json").write_text(json.dumps(config))

    def _write_agpolicy_plugin(self, config_home):
        """Install the native lifecycle and policy observer."""
        plugin_dir = config_home / "plugin"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        plugin_path = plugin_dir / "agpolicy_plugin.js"
        from pathlib import Path

        plugin_path.write_bytes((Path(__file__).parent / "_opencode_pty_plugin.js").read_bytes())
        return plugin_path


__all__ = ["_OpencodeBackend", "opencode_available"]
