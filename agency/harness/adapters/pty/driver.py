"""Shared PTY driver contract and invocation setup.

`PtyDriver` owns harness terminal and session semantics; `PtyExecution` drives
it. Concrete drivers live in their corresponding harness modules.
`_HookPtyDriver` shares hook and rollout behavior between Codex and Grok.
"""

from __future__ import annotations

import json
import shlex
import uuid
from pathlib import Path

from ...executable import HARNESS_PATH
from .execution import MAX_SESSION_BYTES, PtyExecution, restore_session, snapshot_session

_PROFILER_BRIDGE_TIMEOUT_S = 10.0


def driver_for(adapter, runtime, root, session_id, blob, max_steps):
    driver_class = getattr(adapter, "_PTY_DRIVER", None)
    if driver_class is None:
        raise ValueError(f"no PTY driver for harness: {adapter._DEFAULT_BINARY}")
    return driver_class(adapter, runtime, root, session_id, blob, max_steps)


def run_pty_attempt(adapter, runtime, *, prompt, resume_session_id, prior_session_blob, max_steps):
    from ...agharness import cleanup_config_home, materialize_config_home
    from ....native_harness.bridge_client import BridgeClient
    from ....native_harness.profiling import NativeProfiler

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
        lifecycle = self.root / "lifecycle_hook.py"
        lifecycle.write_bytes((hook_dir / "_lifecycle_hook.py").read_bytes())
        permission = self.root / "agpolicy_hook.py"
        permission.write_bytes(
            (hook_dir.parent.parent / "_harness_permission_hook.py").read_bytes()
        )

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
