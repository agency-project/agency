"""opencode interactive PTY adapter; the model protocol is shared."""

from __future__ import annotations

import json
import base64
import sqlite3
from .pty.execution import MAX_SESSION_BYTES
import shutil

from .base import AdapterRuntime, AttemptResult, HarnessAdapter
from .openai_chat_completions import ChatCompletionsProtocol
from .pty.driver import PtyDriver, run_pty_attempt


def opencode_available() -> bool:
    return shutil.which("opencode") is not None


class OpencodeDriver(PtyDriver):
    name = "opencode"
    confirm_interrupt = True

    def _configure(self, adapter, runtime, max_steps):
        plugin = adapter._write_agpolicy_plugin(self.root)
        adapter._write_opencode_config(
            self.root,
            runtime.harness_base_url,
            runtime.token,
            runtime.model or "default",
            plugin,
            max_steps,
        )
        self.env.update(
            OPENCODE_CONFIG=str(self.root / "opencode.json"),
            OPENCODE_DISABLE_AUTOUPDATE="true",
            OPENCODE_DISABLE_DEFAULT_PLUGINS="true",
        )
        self.argv += ["--auto"]
        if self.session_id:
            self.argv += ["--session", self.session_id]

    def ready(self, handle):
        lines, _x, y, _generation = handle.terminal_screen()
        current = lines[y].strip()
        empty_composer = current == "┃" or current.startswith(
            ("┃  Ask anything...", "┃  Ask anything…")
        )
        return empty_composer and any(
            "Build" in line and "Agency Proxy" in line for line in lines[max(0, y - 1) : y + 4]
        )

    def prompt_matches(self, prompt, expected):
        # OpenCode can append a newline when persisting bracketed paste.
        if isinstance(prompt, str) and expected is not None:
            return prompt.rstrip() == expected.rstrip()
        return prompt == expected

    def interrupt_pending(self, handle):
        return any("esc again to interrupt" in line for line in handle.terminal_screen()[0])

    def _event_from_payload(self, payload):
        # The plugin emits already-normalized events; only session-scope them.
        if self.session_id is None:
            self.session_id = payload.get("session_id")
        if payload.get("session_id") != self.session_id:
            return None
        return payload

    def snapshot(self):
        database = self.root / "data" / "opencode" / "opencode.db"
        if database.is_symlink() or not database.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("unsafe native session database")
        # A raw copy can omit WAL pages. SQLite's backup API takes a coherent
        # committed snapshot while the TUI still owns its connection.
        temporary = database.with_suffix(".snapshot")
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as source:
            with sqlite3.connect(temporary) as destination:
                source.backup(destination)
        try:
            # Package the backup without changing the live database.
            data = temporary.read_bytes()
            bundle = json.dumps(
                {
                    "version": 1,
                    "harness": self.name,
                    "session_id": self.session_id,
                    "files": {"data/opencode/opencode.db": base64.b64encode(data).decode()},
                }
            ).encode()
            if len(bundle) > MAX_SESSION_BYTES:
                raise ValueError("native session exceeds size limit")
            return bundle
        finally:
            temporary.unlink()

class OpenCodeAdapter(ChatCompletionsProtocol, HarnessAdapter):
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

        plugin_path.write_bytes((Path(__file__).parent / "pty" / "_opencode_pty_plugin.js").read_bytes())
        return plugin_path


__all__ = ["OpenCodeAdapter", "opencode_available"]
