"""Native TUI dialects for Codex, Grok Build, and OpenCode.

`PtyDriver` is the dialect contract `PtyExecution` drives. One subclass per
harness, mirroring this package's one-file-per-harness shape; `_HookPtyDriver`
carries what the hook-and-rollout CLIs (Codex, Grok) share.
"""

from __future__ import annotations

import base64
import json
import shlex
import sqlite3
import uuid
from pathlib import Path

from ..executable import HARNESS_PATH
from .pty_session import MAX_SESSION_BYTES, PtyExecution, restore_session, snapshot_session

_PROFILER_BRIDGE_TIMEOUT_S = 10.0


def driver_for(adapter, runtime, root, session_id, blob, max_steps):
    driver_class = getattr(adapter, "_PTY_DRIVER", None)
    if driver_class is None:
        raise ValueError(f"no PTY driver for harness: {adapter._DEFAULT_BINARY}")
    return driver_class(adapter, runtime, root, session_id, blob, max_steps)


def run_pty_attempt(adapter, runtime, *, prompt, resume_session_id, prior_session_blob, max_steps):
    from ..agharness import cleanup_config_home, materialize_config_home
    from ...native_harness.bridge_client import BridgeClient
    from ...native_harness.profiling import NativeProfiler

    key = (
        "pty",
        adapter._DEFAULT_BINARY,
        runtime.model,
        max_steps,
        runtime.has_sandbox_mcp_tools,
    )

    def factory():
        root = materialize_config_home(runtime.engine_name)
        bridge = BridgeClient(
            runtime.harness_base_url, runtime.token, timeout_s=_PROFILER_BRIDGE_TIMEOUT_S
        )
        try:
            driver = driver_for(
                adapter, runtime, root, resume_session_id, prior_session_blob, max_steps
            )
            profiler = NativeProfiler(bridge)
            try:
                profiler.enabled = bool(bridge.profiler_settings().get("enabled"))
            except Exception:
                profiler.enabled = False
            driver.profile_span = profiler.span
            return PtyExecution(driver, runtime, cleanup_callbacks=(bridge.close,))
        except BaseException:
            bridge.close()
            cleanup_config_home(root)
            raise

    return runtime.run_pty_execution(key, factory, prompt)


class PtyDriver:
    """One harness's TUI dialect: launch, readiness, events, and session state."""

    name = ""
    interrupt_key = b"\x1b"
    confirm_interrupt = False
    INPUT_TIMEOUT = 60.0
    START_TIMEOUT = 45.0
    ATTEMPT_TIMEOUT = 600.0
    # Whether terminal output alone should extend the attempt deadline.
    activity_extends_deadline = False

    def __init__(self, adapter, runtime, root, session_id, blob, max_steps):
        self.root = root
        self.session_id = session_id
        self.cwd = "/workspace"
        self.transcript_path = None
        self._restore(session_id, blob)
        self.env = {
            "PATH": HARNESS_PATH,
            "TERM": "xterm-256color",
            "COLORTERM": "truecolor",
            "HOME": str(root),
            "AGENCY_PTY_STATE": str(root),
            "AGPOLICY_STATE_DIR": str(root),
            "AGPOLICY_BASE_URL": runtime.harness_base_url,
            "AGPOLICY_TOKEN": runtime.token,
            "XDG_CONFIG_HOME": str(root / "config"),
            "XDG_DATA_HOME": str(root / "data"),
            "XDG_CACHE_HOME": str(root / "cache"),
            "XDG_STATE_HOME": str(root / "state"),
        }
        self.argv = [adapter.agconfig.harness_adapter.binary_path or self.name]
        self._configure(adapter, runtime, max_steps)

    def _restore(self, session_id, blob):
        restore_session(self.root, self.name, session_id, blob)
        (self.root / "events").mkdir()

    def _configure(self, adapter, runtime, max_steps):
        raise NotImplementedError

    def prepare_launch(self):
        """Materialize any driver state that depends on the final working directory."""

    def ready(self, handle):
        raise NotImplementedError

    def prompt_matches(self, prompt, expected):
        return prompt == expected

    def submission_marker(self, label):
        # Unique content also fences two submissions with identical user text,
        # which is what a CLI-assigned turn identity is matched against.
        return f"[Agency {label} {uuid.uuid4().hex}]"

    def begin_turn(self, prompt):
        """Return a turn identity this driver commits itself, else None."""
        return None

    def clear_input(self, handle, wait_until):
        # These CLIs clear the interrupted prompt. Ctrl+U clears any restored
        # editable draft without submitting it or sending a second interrupt.
        handle.write_terminal(b"\x15")

    def interrupt_pending(self, handle):
        return False

    def begin_interrupt(self, handle):
        """Record whatever an interrupt will later be recognized against."""

    def interrupted(self, handle):
        """Evidence of interruption that arrives outside the event stream."""
        return False

    def reap(self, handle):
        handle.close()

    def events(self):
        pending = []
        for path in sorted((self.root / "events").glob("*.json")):
            payload = json.loads(path.read_text())
            path.unlink()
            event = self._event_from_payload(payload)
            if event is not None:
                pending.append(event)
        pending.extend(self._transcript_events())
        return pending

    def _event_from_payload(self, payload):
        raise NotImplementedError

    def _transcript_events(self):
        return []

    def completed(self, event):
        if not isinstance(event.get("text"), str):
            raise RuntimeError(f"{self.name} completion has no final text")
        return True

    def snapshot(self):
        return snapshot_session(self.root, self.name, self.session_id)


class _HookPtyDriver(PtyDriver):
    """Codex and Grok: lifecycle hook files plus a native rollout transcript."""

    _HOOK_EVENTS = ()
    _HOOK_PATH = ()

    def __init__(self, adapter, runtime, root, session_id, blob, max_steps):
        self._rollout_offset = 0
        self._last_prompt = None
        self._last_turn_id = None
        self._transcript_turn_id = None
        super().__init__(adapter, runtime, root, session_id, blob, max_steps)

    def _rows(self):
        if not self.transcript_path or not self.transcript_path.exists():
            return
        if self.transcript_path.stat().st_size > MAX_SESSION_BYTES:
            raise ValueError("native transcript exceeds size limit")
        with self.transcript_path.open("rb") as transcript:
            for line in transcript:
                if line.endswith(b"\n"):
                    yield json.loads(line)

    def _write_hooks(self):
        hook_dir = Path(__file__).parent
        lifecycle = self.root / "pty_hook.py"
        lifecycle.write_bytes((hook_dir / "_pty_hook.py").read_bytes())
        permission = self.root / "agpolicy_hook.py"
        permission.write_bytes((hook_dir.parent / "_harness_permission_hook.py").read_bytes())

        def command(script):
            return [
                {"hooks": [{"type": "command", "command": "python3 " + shlex.quote(str(script))}]}
            ]

        hooks = {event: command(lifecycle) for event in self._HOOK_EVENTS}
        hooks.update({event: command(permission) for event in ["PreToolUse", "PostToolUse"]})
        path = self.root.joinpath(*self._HOOK_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"hooks": hooks}))

    def _event_from_payload(self, payload):
        if payload.get("agent_id") or payload.get("subagentType"):
            return None
        sid = payload.get("session_id", payload.get("sessionId"))
        if self.session_id is None:
            self.session_id = sid
            if self.name == "grok" and sid:
                # Grok's hooks omit transcript_path, but its ACP session log
                # has a deterministic location beneath GROK_HOME.
                self.transcript_path = (
                    self.root / "sessions" / "%2Fworkspace" / sid / "updates.jsonl"
                )
        if sid != self.session_id:
            return None
        transcript = payload.get("transcript_path", payload.get("transcriptPath"))
        if transcript:
            path = Path(transcript)
            if not path.resolve().is_relative_to(self.root.resolve()):
                raise RuntimeError("native transcript escaped isolated state")
            self.transcript_path = path
        kind = payload.get("hook_event_name") or {
            "session_start": "SessionStart",
            "user_prompt_submit": "UserPromptSubmit",
            "stop": "Stop",
            "stop_cancelled": "StopCancelled",
            "stop_failure": "StopFailure",
        }.get(payload.get("hookEventName"))
        turn = payload.get("turn_id", payload.get("promptId"))
        event = {"turn_id": turn}
        if kind == "UserPromptSubmit":
            prompt = self._submitted_prompt(payload.get("prompt"))
            event.update(kind="submit", prompt=prompt)
            self._last_prompt = prompt
            self._last_turn_id = turn
            return event
        if kind == "Stop":
            return self._stop_event(event, payload)
        if kind == "StopCancelled":
            if payload.get("reason") == "user_interrupt":
                event["kind"] = "interrupt"
            else:
                event.update(kind="error", error=f"Grok stopped: {payload.get('reason')}")
            return event
        if kind == "StopFailure":
            event.update(
                kind="error",
                error=f"Grok failed: {payload.get('errorDetails', payload.get('error'))}",
            )
            return event
        return None

    def _submitted_prompt(self, prompt):
        return prompt

    def _stop_event(self, event, payload):
        event.update(
            kind="stop",
            text=payload.get("last_assistant_message", payload.get("lastAssistantMessage", "")),
        )
        return event

    def _transcript_events(self):
        if not self.transcript_path or not self.transcript_path.exists():
            return []
        pending = []
        with self.transcript_path.open("rb") as transcript:
            transcript.seek(self._rollout_offset)
            for line in transcript:
                if not line.endswith(b"\n"):
                    break
                self._rollout_offset += len(line)
                self._scan_transcript_row(json.loads(line), pending)
        return pending

    def _scan_transcript_row(self, row, pending):
        if row.get("type") == "event_msg" and row.get("payload", {}).get("type") == "turn_aborted":
            pending.append({"kind": "interrupt", "turn_id": row["payload"].get("turn_id")})
        self._scan_dialect_row(row, pending)

    def _scan_dialect_row(self, row, pending):
        raise NotImplementedError


class CodexDriver(_HookPtyDriver):
    name = "codex"
    _HOOK_EVENTS = ("SessionStart", "UserPromptSubmit", "Stop")
    _HOOK_PATH = ("hooks.json",)

    def __init__(self, adapter, runtime, root, session_id, blob, max_steps):
        self._trusted_directory = False
        super().__init__(adapter, runtime, root, session_id, blob, max_steps)

    def _configure(self, adapter, runtime, max_steps):
        adapter._write_codex_config(
            self.root,
            runtime.harness_base_url,
            runtime.model or "default",
            has_sandbox_mcp_tools=runtime.has_sandbox_mcp_tools,
        )
        with (self.root / "config.toml").open("a") as config:
            config.write('\n[projects."/workspace"]\ntrust_level = "trusted"\n')
        self.env.update(CODEX_HOME=str(self.root), AGENCY_PROXY_API_KEY=runtime.token)
        # Codex has no CLI step-limit flag, as with its previous adapter.
        self.argv += [
            "--no-alt-screen",
            "--dangerously-bypass-approvals-and-sandbox",
            "--dangerously-bypass-hook-trust",
        ]
        if self.session_id:
            self.argv += ["resume", self.session_id]
        self._write_hooks()

    def ready(self, handle):
        lines, x, y, _generation = handle.terminal_screen()
        if not self._trusted_directory and any(
            "Do you trust the contents of this directory?" in line for line in lines
        ):
            if any(self.cwd in line for line in lines) and any(
                "1. Yes, continue" in line for line in lines
            ):
                self._trusted_directory = True
                handle.write_terminal(b"\r")
            return False
        # Require the actual composer, never silence or an old message in history.
        return lines[y].strip().startswith("›") and x <= 3

    def _stop_event(self, event, payload):
        event = super()._stop_event(event, payload)
        # Codex reports JSON null when a turn ends immediately after a
        # successful MCP submission. The protocol represents that as
        # an empty final string so the engine can consume the output
        # collected by submit_output.
        if event["text"] is None:
            event["text"] = ""
        return event

    def _scan_dialect_row(self, row, pending):
        payload = row.get("payload", {})
        if row.get("type") != "event_msg":
            return
        event_type = payload.get("type")
        if event_type in {"task_started", "turn_started"}:
            self._transcript_turn_id = payload.get("turn_id")
        elif event_type == "error" or (event_type == "task_complete" and payload.get("error")):
            # Native errors can omit turn_id; associate only with an observed
            # transcript start, never the host's current prompt (which may
            # have changed).
            pending.append(
                {
                    "kind": "error",
                    "turn_id": payload.get("turn_id") or self._transcript_turn_id,
                    "error": "Codex terminal error: "
                    + str(payload.get("error") or payload.get("message", "request failed")),
                }
            )

    def completed(self, event):
        super().completed(event)
        for row in self._rows():
            payload = row.get("payload", {})
            if (
                row.get("type") == "event_msg"
                and payload.get("type") == "task_complete"
                and payload.get("turn_id") == event["turn_id"]
            ):
                return True
        return False


class GrokDriver(_HookPtyDriver):
    name = "grok"
    interrupt_key = b"\x03"
    _HOOK_EVENTS = ("SessionStart", "UserPromptSubmit", "Stop", "StopFailure", "StopCancelled")
    _HOOK_PATH = ("hooks", "agency.json")

    def _configure(self, adapter, runtime, max_steps):
        adapter._write_grok_config(
            self.root, runtime.harness_base_url, runtime.token, runtime.model or "default"
        )
        self.env["GROK_HOME"] = str(self.root)
        self.argv += ["--no-alt-screen", "--always-approve", "--no-memory", "--no-plan"]
        if max_steps is not None:
            self.argv += ["--max-turns", str(max_steps)]
        if self.session_id:
            self.argv += ["--resume", self.session_id]
        self._write_hooks()

    def ready(self, handle):
        lines, _x, y, _generation = handle.terminal_screen()
        current = lines[y].strip()
        return current.startswith("│ ❯") and not current.removeprefix("│ ❯").rstrip("│ ").strip()

    def _submitted_prompt(self, prompt):
        if (
            isinstance(prompt, str)
            and prompt.startswith("<user_query>\n")
            and prompt.endswith("\n</user_query>")
        ):
            return prompt[len("<user_query>\n") : -len("\n</user_query>")]
        return prompt

    def _stop_event(self, event, payload):
        if payload.get("reason") != "end_turn":
            return None
        return super()._stop_event(event, payload)

    def interrupted(self, handle):
        """Recognize Grok's restored composer draft as interrupt evidence."""
        if self._last_prompt is None:
            return False
        marker = self._last_prompt.split("\n", 1)[0]
        pasted = f"│ ❯ [Pasted: {len(self._last_prompt.splitlines())} lines]"
        lines, _x, y, _generation = handle.terminal_screen()
        # Grok collapses a restored multiline paste into a token. The prompt
        # marker then appears only in the old transcript.
        return (
            self.ready(handle)
            or any("│ ❯ " + marker in line for line in lines)
            or pasted in lines[y]
        )

    def clear_input(self, handle, wait_until):
        # Grok restores a pre-response cancellation as a draft. Its idle
        # Ctrl+C clears that draft; Ctrl+U would trigger self-update.
        if self._last_prompt is None:
            return
        wait_until(lambda: self.interrupted(handle), "restored Grok input")
        if not self.ready(handle):
            handle.write_terminal(b"\x03")

    def _scan_dialect_row(self, row, pending):
        update = row.get("params", {}).get("update", {})
        if update.get("sessionUpdate") != "turn_completed":
            return
        reason = update.get("stop_reason")
        if (
            reason == "cancelled"
            and row["params"].get("_meta", {}).get("cancelTrigger") == "ctrl_c"
        ):
            pending.append({"kind": "interrupt", "turn_id": update.get("prompt_id")})
        elif reason != "end_turn":
            pending.append(
                {
                    "kind": "error",
                    "turn_id": update.get("prompt_id"),
                    "error": f"Grok turn ended: {reason}",
                }
            )

    def completed(self, event):
        super().completed(event)
        text = []
        for row in self._rows():
            params = row.get("params", {})
            update = params.get("update", {})
            if (
                params.get("_meta", {}).get("promptId") == event["turn_id"]
                and update.get("sessionUpdate") == "agent_message_chunk"
            ):
                content = update.get("content", {})
                if content.get("type") == "text":
                    text.append(content["text"])
            if (
                update.get("sessionUpdate") == "turn_completed"
                and update.get("prompt_id") == event["turn_id"]
            ):
                event["text"] = "".join(text)
                usage = update.get("usage", {})
                event.update(
                    input_tokens=usage.get("inputTokens", 0),
                    output_tokens=usage.get("outputTokens", 0),
                )
                return True
        return False


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
