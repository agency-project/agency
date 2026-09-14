"""Kimi Code interactive PTY adapter.

Verified against Kimi Code CLI 0.42.0. Its lifecycle hooks report a turn
identity on `TurnStarted` but not on `Stop`, so completion is reconciled
against the session's own `wire.jsonl` rather than trusted from the event.

Kimi's provider `type = "openai"` means its model traffic is plain Chat
Completions, served by the shared `ChatCompletionsBackend` seam.
"""

from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path

from .agharness_backend import AdapterRuntime, AttemptResult, agharness_backend
from .openai_protocol import ChatCompletionsBackend
from .pty_drivers import PtyDriver, run_pty_attempt
from .pty_session import MAX_SESSION_BYTES, restore_session, snapshot_session


def kimi_available() -> bool:
    return shutil.which("kimi") is not None


_MODEL_ALIAS = "agency-proxy"
_LIFECYCLE_EVENTS = (
    "SessionStart",
    "UserPromptSubmit",
    "TurnStarted",
    "Stop",
    "StopFailure",
    "Interrupt",
    "SessionEnd",
)


class KimiDriver(PtyDriver):
    """Kimi Code: CLI-assigned turn IDs, transcript-reconciled completion.

    `TurnStarted` carries `turn_id` and the submitted prompt; `Stop` carries
    neither, so it is stamped with the turn last started and only counts once
    `wire.jsonl` records a matching `turn.ended`. Turn IDs arrive as integers
    from the hooks and as strings inside the transcript, so every one of them
    is normalized to `str` before comparison.
    """

    name = "kimi"

    def __init__(self, adapter, runtime, root, session_id, blob, max_steps):
        self.started = False
        self._trusted_directory = False
        self._last_turn_id = None
        self._prior_blob = blob
        super().__init__(adapter, runtime, root, session_id, blob, max_steps)

    def _restore(self, session_id, blob):
        restore_session(self.root, self.name, session_id, blob)
        (self.root / "events").mkdir(exist_ok=True)

    def _configure(self, adapter, runtime, max_steps):
        adapter._write_kimi_config(
            self.root,
            runtime.harness_base_url,
            runtime.token,
            runtime.model or "default",
            max_steps,
            self._hook_commands(),
        )
        # KIMI_CODE_HOME relocates config, sessions, and credentials together,
        # so the attempt's whole footprint stays inside its isolated root.
        self.env.update(KIMI_CODE_HOME=str(self.root))
        self.argv += ["--auto", "--model", _MODEL_ALIAS]
        if self.session_id:
            self.argv += ["--session", self.session_id]

    def _hook_commands(self):
        hook_dir = Path(__file__).parent
        lifecycle = self.root / "pty_hook.py"
        lifecycle.write_bytes((hook_dir / "_pty_hook.py").read_bytes())
        permission = self.root / "agpolicy_hook.py"
        permission.write_bytes((hook_dir.parent / "_harness_permission_hook.py").read_bytes())
        run = "python3 " + shlex.quote(str(lifecycle))
        policy = "python3 " + shlex.quote(str(permission))
        commands = [(event, run) for event in _LIFECYCLE_EVENTS]
        commands += [
            (event, policy) for event in ("PreToolUse", "PostToolUse", "PostToolUseFailure")
        ]
        return commands

    def ready(self, handle):
        lines, _x, y, _generation = handle.terminal_screen()
        if any("Trust this folder?" in line for line in lines):
            # Same shape as Codex's trust prompt: the default row is already
            # selected, so Enter accepts and the composer follows. Answer it
            # once -- readiness is polled every 25ms, and the surplus Enters
            # would land in the composer as empty submissions.
            if not self._trusted_directory and any(
                "Trust this folder" in line and "❯" in line for line in lines
            ):
                self._trusted_directory = True
                handle.write_terminal(b"\r")
            return False
        # Require the real composer, never silence or scrollback.
        current = lines[y].strip()
        return current.startswith("│ >") and not current.removeprefix("│ >").rstrip("│ ").strip()

    @property
    def _transcript_path(self):
        if self.session_id is None:
            return None
        index = self.root / "session_index.jsonl"
        if not index.exists():
            return None
        for line in index.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("sessionId") == self.session_id:
                # sessionDir is Kimi's own absolute path; keep it inside the root.
                directory = Path(record["sessionDir"])
                if not directory.resolve().is_relative_to(self.root.resolve()):
                    raise RuntimeError("native transcript escaped isolated state")
                return directory / "agents" / "main" / "wire.jsonl"
        return None

    def _rows(self):
        path = self._transcript_path
        if path is None or not path.exists():
            return
        if path.stat().st_size > MAX_SESSION_BYTES:
            raise ValueError("native transcript exceeds size limit")
        with path.open("rb") as transcript:
            for line in transcript:
                if line.endswith(b"\n"):
                    yield json.loads(line)

    @staticmethod
    def _turn(value):
        return None if value is None else str(value)

    def _event_from_payload(self, payload):
        if payload.get("agent_id") or payload.get("subagent_name"):
            return None
        kind = payload.get("hook_event_name")
        session = payload.get("session_id")
        if kind == "SessionStart":
            if self.session_id is None:
                self.session_id = session
            self.started = True
            return None
        if session != self.session_id:
            return None
        if kind == "TurnStarted":
            # The only event carrying both the identity and the prompt.
            self._last_turn_id = self._turn(payload.get("turn_id"))
            return {
                "kind": "submit",
                "turn_id": self._last_turn_id,
                "prompt": payload.get("prompt"),
            }
        if kind == "Stop":
            # Stop reports no turn of its own; completed() refuses to honour
            # this until the transcript shows that turn actually ended.
            return {"kind": "stop", "turn_id": self._last_turn_id, "text": ""}
        if kind == "Interrupt":
            return {"kind": "interrupt", "turn_id": self._turn(payload.get("turn_id"))}
        if kind in {"StopFailure", "SessionEnd"}:
            reason = payload.get("reason") or payload.get("error") or "session ended"
            return {
                "kind": "error",
                "turn_id": self._turn(payload.get("turn_id")) or self._last_turn_id,
                "error": f"Kimi {kind}: {reason}",
            }
        return None

    def completed(self, event):
        super().completed(event)
        turn = event["turn_id"]
        if turn is None:
            return False
        text, usage = [], {"input_tokens": 0, "output_tokens": 0}
        ended = False
        for row in self._rows():
            kind = row.get("type")
            if kind == "context.append_loop_event":
                loop = row.get("event", {})
                if self._turn(loop.get("turnId")) != turn:
                    continue
                if loop.get("type") == "content.part":
                    part = loop.get("part", {})
                    if part.get("type") == "text":
                        text.append(part.get("text", ""))
                elif loop.get("type") == "step.end":
                    counts = loop.get("usage", {})
                    usage["input_tokens"] += (
                        counts.get("inputOther", 0)
                        + counts.get("inputCacheRead", 0)
                        + counts.get("inputCacheCreation", 0)
                    )
                    usage["output_tokens"] += counts.get("output", 0)
            elif kind == "turn.ended" and self._turn(row.get("turnId")) == turn:
                if row.get("reason") != "completed":
                    raise RuntimeError(f"Kimi turn ended: {row.get('reason')}")
                ended = True
        if not ended:
            return False
        event["text"] = "".join(text)
        event.update(usage)
        return True

    def snapshot(self):
        return snapshot_session(self.root, self.name, self.session_id)


class _KimiBackend(ChatCompletionsBackend, agharness_backend):
    _DEFAULT_BINARY = "kimi"
    _PTY_DRIVER = KimiDriver
    _PROVIDER_NAME = _MODEL_ALIAS

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

    def _write_kimi_config(self, config_home, base_url, token, model, max_steps, hooks):
        # type = "openai" is Kimi's OpenAI Chat Completions protocol, which
        # agproxy_llm already serves unchanged -- the same passthrough route
        # grok and opencode use, with no translation layer.
        lines = [
            f'default_model = "{_MODEL_ALIAS}"',
            'default_permission_mode = "auto"',
            "default_plan_mode = false",
            "telemetry = false",
            "",
            f"[providers.{_MODEL_ALIAS}]",
            'type = "openai"',
            f'base_url = "{base_url.rstrip("/")}/v1"',
            f"api_key = {json.dumps(token)}",
            "",
            f'[models."{_MODEL_ALIAS}"]',
            f'provider = "{_MODEL_ALIAS}"',
            f"model = {json.dumps(model)}",
            "max_context_size = 200000",
            'capabilities = ["tool_use"]',
        ]
        if max_steps is not None:
            lines += ["", "[loop_control]", f"max_steps_per_turn = {int(max_steps)}"]
        for event, command in hooks:
            lines += [
                "",
                "[[hooks]]",
                f'event = "{event}"',
                f"command = {json.dumps(command)}",
                "timeout = 30",
            ]
        (config_home / "config.toml").write_text("\n".join(lines) + "\n")


__all__ = ["_KimiBackend", "KimiDriver", "kimi_available"]
