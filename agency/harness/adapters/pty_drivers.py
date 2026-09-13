"""Native TUI dialects for Codex, Grok Build, and OpenCode."""

from __future__ import annotations

import json
import shlex
import sqlite3
from pathlib import Path

from ..executable import HARNESS_PATH
from .pty_session import MAX_SESSION_BYTES, PtyExecution, restore_session, snapshot_session

_PROFILER_BRIDGE_TIMEOUT_S = 10.0


def run_pty_attempt(adapter, runtime, *, prompt, resume_session_id, prior_session_blob, max_steps):
    from ..agharness import cleanup_config_home, materialize_config_home
    from ...native_harness.bridge_client import BridgeClient
    from ...native_harness.profiling import NativeProfiler

    root = materialize_config_home(runtime.engine_name)
    try:
        driver = PtyDriver(adapter, runtime, root, resume_session_id, prior_session_blob, max_steps)
    except BaseException:
        cleanup_config_home(root)
        raise
    # Reuse the existing span bridge, without enabling native Python's
    # automatic function sampler inside the external-harness daemon.
    bridge = BridgeClient(
        runtime.harness_base_url, runtime.token, timeout_s=_PROFILER_BRIDGE_TIMEOUT_S
    )
    profiler = NativeProfiler(bridge)
    try:
        try:
            profiler.enabled = bool(bridge.profiler_settings().get("enabled"))
        except Exception:
            profiler.enabled = False
        driver.profile_span = profiler.span
        return PtyExecution(driver, runtime).run(prompt)
    finally:
        bridge.close()


class PtyDriver:
    def _rows(self):
        if not self.transcript_path or not self.transcript_path.exists():
            return
        if self.transcript_path.stat().st_size > MAX_SESSION_BYTES:
            raise ValueError("native transcript exceeds size limit")
        with self.transcript_path.open("rb") as transcript:
            for line in transcript:
                if line.endswith(b"\n"):
                    yield json.loads(line)

    def __init__(self, adapter, runtime, root, session_id, blob, max_steps):
        self.name = adapter._DEFAULT_BINARY
        self.root = root
        self.session_id = session_id
        self.cwd = "/workspace"
        self.interrupt_key = b"\x1b" if self.name != "grok" else b"\x03"
        self.confirm_interrupt = self.name == "opencode"
        self.transcript_path = None
        self._rollout_offset = 0
        self._trusted_directory = False
        self._last_prompt = None
        self._last_turn_id = None
        self._transcript_turn_id = None
        restore_session(root, self.name, session_id, blob)
        (root / "events").mkdir()
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
        if self.name == "codex":
            adapter._write_codex_config(
                root,
                runtime.harness_base_url,
                runtime.model or "default",
                has_sandbox_mcp_tools=runtime.has_sandbox_mcp_tools,
            )
            with (root / "config.toml").open("a") as config:
                config.write('\n[projects."/workspace"]\ntrust_level = "trusted"\n')
            self.env.update(CODEX_HOME=str(root), AGENCY_PROXY_API_KEY=runtime.token)
            self.argv += [
                "--no-alt-screen",
                "--dangerously-bypass-approvals-and-sandbox",
                "--dangerously-bypass-hook-trust",
            ]
            if session_id:
                self.argv += ["resume", session_id]
            self._write_hooks(root / "hooks.json", ["SessionStart", "UserPromptSubmit", "Stop"])
        elif self.name == "grok":
            adapter._write_grok_config(
                root, runtime.harness_base_url, runtime.token, runtime.model or "default"
            )
            self.env["GROK_HOME"] = str(root)
            self.argv += ["--no-alt-screen", "--always-approve", "--no-memory", "--no-plan"]
            if max_steps is not None:
                self.argv += ["--max-turns", str(max_steps)]
            if session_id:
                self.argv += ["--resume", session_id]
            self._write_hooks(
                root / "hooks" / "agency.json",
                ["SessionStart", "UserPromptSubmit", "Stop", "StopFailure", "StopCancelled"],
            )
        else:
            plugin = adapter._write_agpolicy_plugin(root)
            adapter._write_opencode_config(
                root,
                runtime.harness_base_url,
                runtime.token,
                runtime.model or "default",
                plugin,
                max_steps,
            )
            self.env.update(
                OPENCODE_CONFIG=str(root / "opencode.json"),
                OPENCODE_DISABLE_AUTOUPDATE="true",
                OPENCODE_DISABLE_DEFAULT_PLUGINS="true",
            )
            self.argv += ["--auto"]
            if session_id:
                self.argv += ["--session", session_id]

    def _write_hooks(self, path, events):
        hook_dir = Path(__file__).parent
        lifecycle = self.root / "pty_hook.py"
        lifecycle.write_bytes((hook_dir / "_pty_hook.py").read_bytes())
        permission = self.root / "agpolicy_hook.py"
        permission.write_bytes((hook_dir.parent / "_harness_permission_hook.py").read_bytes())

        def command(script):
            return [
                {"hooks": [{"type": "command", "command": "python3 " + shlex.quote(str(script))}]}
            ]

        hooks = {event: command(lifecycle) for event in events}
        hooks.update({event: command(permission) for event in ["PreToolUse", "PostToolUse"]})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"hooks": hooks}))

    def ready(self, handle):
        lines, x, y, _generation = handle.terminal_screen()
        # Require the actual composer, never silence or an old message in history.
        current = lines[y].strip()
        if self.name == "codex":
            if not self._trusted_directory and any(
                "Do you trust the contents of this directory?" in line for line in lines
            ):
                if any(self.cwd in line for line in lines) and any(
                    "1. Yes, continue" in line for line in lines
                ):
                    self._trusted_directory = True
                    handle.write_terminal(b"\r")
                return False
            return current.startswith("›") and x <= 3
        if self.name == "grok":
            return (
                current.startswith("│ ❯") and not current.removeprefix("│ ❯").rstrip("│ ").strip()
            )
        empty_composer = current == "┃" or current.startswith("┃  Ask anything...")
        return empty_composer and any(
            "Build" in line and "Agency Proxy" in line for line in lines[max(0, y - 1) : y + 4]
        )

    def clear_input(self, handle, wait_until):
        # These CLIs clear the interrupted prompt. Ctrl+U clears any restored
        # editable draft without submitting it or sending a second interrupt.
        if self.name != "grok":
            handle.write_terminal(b"\x15")
        elif self._last_prompt is not None:
            # Grok restores a pre-response cancellation as a draft. Its idle
            # Ctrl+C clears that draft; Ctrl+U would trigger self-update.
            committed = False
            for row in self._rows():
                params = row.get("params", {})
                if params.get("_meta", {}).get("promptId") == self._last_turn_id and params.get(
                    "update", {}
                ).get("sessionUpdate") in {"agent_message_chunk", "tool_call"}:
                    committed = True
            if not committed:
                marker = self._last_prompt.split("\n", 1)[0]
                pasted = f"│ ❯ [Pasted: {len(self._last_prompt.splitlines())} lines]"

                def restored_input():
                    lines, _x, y, _generation = handle.terminal_screen()
                    # Grok collapses a restored multiline paste into a token.
                    # The prompt marker then appears only in the old transcript.
                    return (
                        self.ready(handle)
                        or any("│ ❯ " + marker in line for line in lines)
                        or pasted in lines[y]
                    )

                wait_until(restored_input, "restored Grok input")
                if not self.ready(handle):
                    handle.write_terminal(b"\x03")

    def interrupt_pending(self, handle):
        return any("esc again to interrupt" in line for line in handle.terminal_screen()[0])

    def events(self):
        pending = []
        for path in sorted((self.root / "events").glob("*.json")):
            payload = json.loads(path.read_text())
            path.unlink()
            if self.name == "opencode":
                if self.session_id is None:
                    self.session_id = payload.get("session_id")
                if payload.get("session_id") == self.session_id:
                    pending.append(payload)
                continue
            if payload.get("agent_id") or payload.get("subagentType"):
                continue
            sid = payload.get("session_id", payload.get("sessionId"))
            if self.session_id is None:
                self.session_id = sid
            if sid != self.session_id:
                continue
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
                prompt = payload.get("prompt")
                if (
                    self.name == "grok"
                    and isinstance(prompt, str)
                    and prompt.startswith("<user_query>\n")
                    and prompt.endswith("\n</user_query>")
                ):
                    prompt = prompt[len("<user_query>\n") : -len("\n</user_query>")]
                event.update(kind="submit", prompt=prompt)
                self._last_prompt = prompt
                self._last_turn_id = turn
            elif kind == "Stop" and (self.name != "grok" or payload.get("reason") == "end_turn"):
                text = payload.get(
                    "last_assistant_message", payload.get("lastAssistantMessage", "")
                )
                # Codex reports JSON null when a turn ends immediately after a
                # successful MCP submission. The protocol represents that as
                # an empty final string so the engine can consume the output
                # collected by submit_output.
                if self.name == "codex" and text is None:
                    text = ""
                event.update(
                    kind="stop",
                    text=text,
                )
            elif kind == "StopCancelled":
                if payload.get("reason") == "user_interrupt":
                    event["kind"] = "interrupt"
                else:
                    event.update(kind="error", error=f"Grok stopped: {payload.get('reason')}")
            elif kind == "StopFailure":
                event.update(
                    kind="error",
                    error=f"Grok failed: {payload.get('errorDetails', payload.get('error'))}",
                )
            else:
                continue
            pending.append(event)
        if (
            self.name in {"codex", "grok"}
            and self.transcript_path
            and self.transcript_path.exists()
        ):
            with self.transcript_path.open("rb") as transcript:
                transcript.seek(self._rollout_offset)
                for line in transcript:
                    if not line.endswith(b"\n"):
                        break
                    self._rollout_offset += len(line)
                    row = json.loads(line)
                    payload = row.get("payload", {})
                    if self.name == "codex" and row.get("type") == "event_msg":
                        event_type = payload.get("type")
                        if event_type in {"task_started", "turn_started"}:
                            self._transcript_turn_id = payload.get("turn_id")
                        elif event_type == "error":
                            # Native errors can omit turn_id; associate only
                            # with an observed transcript start, never the
                            # host's current prompt (which may have changed).
                            pending.append(
                                {
                                    "kind": "error",
                                    "turn_id": payload.get("turn_id") or self._transcript_turn_id,
                                    "error": "Codex terminal error: "
                                    + str(payload.get("message", "request failed")),
                                }
                            )
                    if row.get("type") == "event_msg" and payload.get("type") == "turn_aborted":
                        pending.append({"kind": "interrupt", "turn_id": payload.get("turn_id")})
                    params = row.get("params", {})
                    update = params.get("update", {})
                    if self.name == "grok" and update.get("sessionUpdate") == "turn_completed":
                        reason = update.get("stop_reason")
                        if (
                            reason == "cancelled"
                            and params.get("_meta", {}).get("cancelTrigger") == "ctrl_c"
                        ):
                            pending.append(
                                {"kind": "interrupt", "turn_id": update.get("prompt_id")}
                            )
                        elif reason != "end_turn":
                            pending.append(
                                {
                                    "kind": "error",
                                    "turn_id": update.get("prompt_id"),
                                    "error": f"Grok turn ended: {reason}",
                                }
                            )
        return pending

    def completed(self, event):
        if not isinstance(event.get("text"), str):
            raise RuntimeError(f"{self.name} completion has no final text")
        if self.name == "codex":
            if not self.transcript_path or not self.transcript_path.exists():
                return False
            for row in self._rows():
                payload = row.get("payload", {})
                if (
                    row.get("type") == "event_msg"
                    and payload.get("type") == "task_complete"
                    and payload.get("turn_id") == event["turn_id"]
                ):
                    return True
            return False
        if self.name == "grok":
            if not self.transcript_path or not self.transcript_path.exists():
                return False
            text = []
            for row in self._rows():
                params = row.get("params", {})
                update = params.get("update", {})
                meta = params.get("_meta", {})
                if (
                    meta.get("promptId") == event["turn_id"]
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
        return True

    def snapshot(self):
        if self.name == "opencode":
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
                import base64

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
        return snapshot_session(self.root, self.name, self.session_id)
