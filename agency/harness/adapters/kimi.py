"""Kimi Code interactive PTY adapter.

Verified against Kimi Code CLI 0.42.0. Its lifecycle hooks report a turn
identity on `TurnStarted` but not on `Stop`, so completion is reconciled
against the session's own `wire.jsonl` rather than trusted from the event.

Kimi's provider `type = "openai"` means its model traffic is plain Chat
Completions, served by the shared `ChatCompletionsProtocol` seam.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shlex
import shutil
import time
from pathlib import Path, PurePosixPath

from .base import AdapterRuntime, AttemptResult, HarnessAdapter, fetch_context_limit
from .openai_chat_completions import ChatCompletionsProtocol
from .pty.driver import PtyDriver, run_pty_attempt
from .pty.execution import MAX_SESSION_BYTES, restore_session, snapshot_session


def kimi_available() -> bool:
    return shutil.which("kimi") is not None


_PROVIDER = "agency-proxy"
# Kimi's own session ids; also the guard that keeps one out of a path join.
_SESSION_ID_RE = re.compile(r"session_[A-Za-z0-9_-]{1,128}")
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
        self._relocate_session_index()

    def _relocate_session_index(self):
        """Repoint a restored session at this attempt's own root.

        Kimi stores absolute paths in two places -- `sessionDir` in
        `session_index.jsonl` and `agents.<name>.homedir` in each session's
        `state.json` -- both naming the config home that produced them. A
        bundle restored into a fresh home therefore points at directories that
        no longer exist. Only the `sessions/<workDirKey>/<sessionId>` tail is
        portable.
        """
        index = self.root / "session_index.jsonl"
        if not index.exists():
            return
        records, previous_roots = [], set()
        for line in index.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            parts = PurePosixPath(record.get("sessionDir", "")).parts
            if "sessions" in parts:
                cut = len(parts) - 1 - parts[::-1].index("sessions")
                previous_roots.add(str(PurePosixPath(*parts[:cut])))
                record["sessionDir"] = str(self.root.joinpath(*parts[cut:]))
            records.append(json.dumps(record))
        index.write_text("".join(line + "\n" for line in records))

        current = str(self.root)
        for previous in previous_roots:
            if not previous or previous == current:
                continue
            for path in (self.root / "sessions").rglob("*.json"):
                text = path.read_text()
                if previous in text:
                    path.write_text(text.replace(previous, current))

    def _configure(self, adapter, runtime, max_steps):
        adapter._write_kimi_config(
            self.root,
            runtime.harness_base_url,
            runtime.token,
            runtime.model or "default",
            max_steps,
            self._hook_commands(),
            context_limit=fetch_context_limit(runtime.harness_base_url, runtime.token),
        )
        # KIMI_CODE_HOME relocates config, sessions, and credentials together,
        # so the attempt's whole footprint stays inside its isolated root.
        self.env.update(KIMI_CODE_HOME=str(self.root))
        self.argv += ["--auto", "--model", runtime.model or "default"]
        if self.session_id:
            self.argv += ["--session", self.session_id]

    def _hook_commands(self):
        hook_dir = Path(__file__).parent
        lifecycle = self.root / "lifecycle_hook.py"
        lifecycle.write_bytes((hook_dir / "pty" / "_lifecycle_hook.py").read_bytes())
        permission = self.root / "agpolicy_hook.py"
        permission.write_bytes((hook_dir.parent / "_harness_permission_hook.py").read_bytes())
        run = "python3 " + shlex.quote(str(lifecycle))
        policy = "python3 " + shlex.quote(str(permission))
        commands = [(event, run) for event in _LIFECYCLE_EVENTS]
        commands += [
            (event, policy) for event in ("PreToolUse", "PostToolUse", "PostToolUseFailure")
        ]
        return commands

    def prepare_launch(self):
        """Pre-approve the workspace before Kimi binds its initial model.

        Agency always accepts Kimi's trust prompt. Kimi 0.42.0 can lose the
        active model when that prompt is accepted during TUI startup, so write
        the exact record its trust service would have written before launch.
        This runs after callers finalize ``cwd`` and also covers cold starts;
        resumed attempts normally already carry the record in their blob.
        """
        root = posixpath.abspath(self.cwd.replace("\\", "/"))
        normalized = root.rstrip("/")
        base = normalized.rsplit("/", 1)[-1]
        slug = re.sub(r"[^a-z0-9._-]+", "-", base.lower()).strip("-")[:40].strip("-")
        if slug in {"", ".", ".."}:
            slug = "workspace"
        digest = hashlib.sha256(normalized.encode()).hexdigest()[:12]
        marker = self.root / "workspace-trust" / f"wd_{slug}_{digest}"
        if marker.exists():
            return
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {"root": root, "trustedAt": int(time.time() * 1000)},
                separators=(",", ":"),
            )
        )

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
        # Located by layout rather than by session_index.jsonl, whose
        # sessionDir is absolute and belongs to whichever config home first
        # wrote it: sessions/<workDirKey>/<sessionId>/agents/main/wire.jsonl.
        if self.session_id is None or not _SESSION_ID_RE.fullmatch(self.session_id):
            return None
        sessions = self.root / "sessions"
        if not sessions.is_dir():
            return None
        for bucket in sorted(sessions.iterdir()):
            candidate = bucket / self.session_id / "agents" / "main" / "wire.jsonl"
            if candidate.is_file() and candidate.resolve().is_relative_to(self.root.resolve()):
                return candidate
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

    def _transcript_events(self):
        """Recover completion when Kimi omits its best-effort Stop hook.

        Kimi writes ``turn.ended`` to the durable wire transcript before it
        invokes the Stop hook. The hook can occasionally be absent on a
        resumed session, so use that authoritative record to wake the runner.
        ``completed()`` still validates the turn identity and end reason and
        reconstructs the answer from the transcript.
        """
        events = []
        for row in self._rows():
            if row.get("type") == "turn.ended":
                events.append(
                    {"kind": "stop", "turn_id": self._turn(row.get("turnId")), "text": ""}
                )
        return events

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


class KimiAdapter(ChatCompletionsProtocol, HarnessAdapter):
    _DEFAULT_BINARY = "kimi"
    _PTY_DRIVER = KimiDriver
    _PROVIDER_NAME = _PROVIDER

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

    def _write_kimi_config(
        self, config_home, base_url, token, model, max_steps, hooks, *, context_limit=None
    ):
        # type = "openai" is Kimi's OpenAI Chat Completions protocol, which
        # agproxy_llm already serves unchanged -- the same passthrough route
        # grok and opencode use, with no translation layer.
        # The alias has to BE the model name: a resumed session restores the
        # model it was bound to by name, and an alias the fresh config does not
        # define leaves Kimi with no LLM ("send /login to login") and silently
        # refuses every prompt.
        alias = json.dumps(model)
        lines = [
            f"default_model = {alias}",
            'default_permission_mode = "auto"',
            "default_plan_mode = false",
            "telemetry = false",
            "",
            f"[providers.{_PROVIDER}]",
            'type = "openai"',
            f'base_url = "{base_url.rstrip("/")}/v1"',
            f"api_key = {json.dumps(token)}",
            "",
            f"[models.{alias}]",
            f'provider = "{_PROVIDER}"',
            f"model = {alias}",
            # A real, per-model value when the host can supply one -- unlike
            # codex/grok/opencode this field already existed here, but it was
            # a static guess, wrong in either direction for whatever model is
            # actually proxied behind agency.
            f"max_context_size = {int(context_limit) if context_limit is not None else 200000}",
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


__all__ = ["KimiAdapter", "KimiDriver", "kimi_available"]
