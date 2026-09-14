import base64
import json
import tomllib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agency.configs.agconfig import agconfig
from agency.harness.adapters.agharness_backend import AdapterRuntime, agharness_backend
from agency.harness.adapters.pty_drivers import OpencodeDriver, PtyDriver, driver_for
from agency.harness.adapters.pty_session import PtyExecution, restore_session, snapshot_session


@pytest.fixture
def runtime():
    return AdapterRuntime(
        agconfig=agconfig(),
        model="a-model",
        engine_name="test",
        harness_base_url="http://127.0.0.1:8766",
        token="attempt-token",
        syscall_policy=object(),
        register_control_handle=Mock(),
        register_redirect=Mock(),
    )


@pytest.mark.parametrize("name", ["codex", "grok", "opencode"])
def test_driver_has_only_interactive_launch_and_isolated_config(name, runtime, tmp_path):
    driver = driver_for(
        agharness_backend.for_config(name, runtime.agconfig), runtime, tmp_path, None, None, 4
    )
    assert driver.argv[0] == name
    assert not set(driver.argv) & {"exec", "run", "--json", "--format", "--prompt-file"}
    assert driver.env["HOME"] == str(tmp_path)
    assert driver.env["TERM"] == "xterm-256color"
    assert driver.env["AGPOLICY_TOKEN"] == runtime.token
    assert driver.cwd == "/workspace"
    if name == "codex":
        config = tomllib.loads((tmp_path / "config.toml").read_text())
        assert config["model_providers"]["agency-proxy"]["wire_api"] == "responses"
        assert config["mcp_servers"]["agency"] == {
            "url": f"{runtime.harness_base_url}/mcp",
            "bearer_token_env_var": "AGENCY_PROXY_API_KEY",
            "required": True,
            "default_tools_approval_mode": "approve",
        }
        assert "agency-sandbox" not in config["mcp_servers"]
        assert {"SessionStart", "Stop", "UserPromptSubmit", "PreToolUse", "PostToolUse"} == set(
            json.loads((tmp_path / "hooks.json").read_text())["hooks"]
        )
    elif name == "grok":
        assert driver.argv[-2:] == ["--max-turns", "4"]
        config = tomllib.loads((tmp_path / "config.toml").read_text())
        assert config["model"]["agency-proxy"]["api_backend"] == "chat_completions"
        assert "StopCancelled" in json.loads((tmp_path / "hooks/agency.json").read_text())["hooks"]
    else:
        config = json.loads((tmp_path / "opencode.json").read_text())
        assert config["agent"]["build"]["steps"] == 4
        assert config["provider"]["agency-proxy"]["options"]["apiKey"] == runtime.token
        assert "chat.message" in (tmp_path / "plugin/agpolicy_plugin.js").read_text()


def test_codex_registers_attempt_local_sandbox_mcp(runtime, tmp_path):
    runtime = replace(runtime, has_sandbox_mcp_tools=True)
    driver_for(
        agharness_backend.for_config("codex", runtime.agconfig),
        runtime,
        tmp_path,
        None,
        None,
        4,
    )
    config = tomllib.loads((tmp_path / "config.toml").read_text())
    assert config["mcp_servers"]["agency-sandbox"] == {
        "url": f"{runtime.harness_base_url}/sandbox/mcp",
        "bearer_token_env_var": "AGENCY_PROXY_API_KEY",
        "required": True,
        "default_tools_approval_mode": "approve",
    }


class FakeDriver(PtyDriver):
    """Subclasses the real contract so new driver hooks reach this double too."""

    name = "fake"
    cwd = "/workspace"
    argv = ["interactive-cli"]
    env = {"TERM": "xterm-256color"}
    session_id = "native-session"

    def __init__(self, root):
        self.root = root
        self.pending = []
        self.is_ready = True
        self.persisted = True

    def ready(self, handle):
        return self.is_ready

    def clear_input(self, handle, wait_until):
        handle.write_terminal(b"\x15")

    def events(self):
        result, self.pending = self.pending, []
        return result

    def completed(self, event):
        return self.persisted

    def snapshot(self):
        return b"durable-native-session"


def test_codex_transcript_error_keeps_original_turn(runtime, tmp_path):
    driver = driver_for(
        agharness_backend.for_config("codex", runtime.agconfig), runtime, tmp_path, None, None, 4
    )
    driver.transcript_path = tmp_path / "transcript.jsonl"
    driver.transcript_path.write_text(
        "\n".join(
            json.dumps({"type": "event_msg", "payload": p})
            for p in [
                {"type": "task_started", "turn_id": "old"},
                {"type": "error", "message": "bad request"},
            ]
        )
        + "\n"
    )
    events = driver.events()
    assert events == [
        {"kind": "error", "turn_id": "old", "error": "Codex terminal error: bad request"}
    ]
    assert driver.events() == []


def test_codex_accepts_empty_completion_after_mcp_submission(runtime, tmp_path):
    driver = driver_for(
        agharness_backend.for_config("codex", runtime.agconfig), runtime, tmp_path, None, None, 4
    )
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {"type": "task_complete", "turn_id": "turn"},
            }
        )
        + "\n"
    )
    (tmp_path / "events/stop.json").write_text(
        json.dumps(
            {
                "session_id": "session",
                "transcript_path": str(transcript),
                "hook_event_name": "Stop",
                "turn_id": "turn",
                "last_assistant_message": None,
            }
        )
    )

    event = driver.events()[0]
    assert event == {"turn_id": "turn", "kind": "stop", "text": ""}
    assert driver.completed(event)


@pytest.fixture
def execution(runtime, tmp_path):
    driver = FakeDriver(tmp_path)
    execution = PtyExecution(driver, runtime)
    execution.INPUT_TIMEOUT = 0.05
    execution.START_TIMEOUT = 0.05
    execution.ATTEMPT_TIMEOUT = 0.1
    handle = Mock(returncode=None)
    handle.is_paused.return_value = False
    execution.handle = handle
    execution._active = True
    execution._turn_id = "old-turn"
    execution._expected_prompt = "old prompt"
    return execution


@pytest.mark.parametrize("text", ["\x1b[31m", "hello\x00", "\x7f", "\r", " "])
def test_terminal_control_characters_cannot_be_injected(execution, text):
    assert not execution.redirect(text)
    execution.handle.write_terminal.assert_not_called()


@pytest.mark.parametrize("state", ["inactive", "paused", "completed"])
def test_redirect_rejects_unavailable_turn_without_writing(execution, state):
    if state == "inactive":
        execution._active = False
    elif state == "paused":
        execution.handle.is_paused.return_value = True
    else:
        execution.driver.pending.append({"kind": "stop", "turn_id": "old-turn", "text": "final"})
    assert not execution.redirect("replacement")
    execution.handle.write_terminal.assert_not_called()


def test_delayed_stop_and_wrong_prompt_do_not_acknowledge_current_turn(execution):
    execution._turn_id = None
    execution.driver.pending = [
        {"kind": "stop", "turn_id": "old-turn", "text": "stale"},
        {"kind": "submit", "turn_id": "wrong", "prompt": "different"},
    ]
    execution._poll()
    assert execution._turn_id is None
    assert execution._stop is None


def test_startup_failure_is_reported_before_any_turn_exists(execution):
    execution._turn_id = None
    execution._turn_started = False
    execution.driver.pending = [{"kind": "error", "turn_id": None, "error": "bad credential"}]
    execution._poll()
    assert execution._failure == "bad credential"


def test_turnless_error_cannot_fail_a_replacement_turn(execution):
    # A delayed error from a retired turn must not kill the redirect that
    # replaced it, which is still waiting to learn its own turn identity.
    execution._turn_id = None
    execution._turn_started = True
    execution.driver.pending = [{"kind": "error", "turn_id": None, "error": "stale failure"}]
    execution._poll()
    assert execution._failure is None


def test_redirect_interrupts_then_requires_exact_native_ack(execution):
    def write(data):
        if data == b"\x1b":
            execution.driver.pending.append({"kind": "interrupt", "turn_id": "old-turn"})
        elif data == b"\r":
            execution.driver.pending += [
                {"kind": "submit", "turn_id": "new-turn", "prompt": execution._expected_prompt},
                {"kind": "stop", "turn_id": "old-turn", "text": "stale"},
            ]

    execution.handle.write_terminal.side_effect = write
    assert execution.redirect("replacement\nwith second line")
    assert execution._turn_id == "new-turn"
    assert execution._stop is None
    calls = [call.args[0] for call in execution.handle.write_terminal.call_args_list]
    assert calls[0] == b"\x1b" and calls[-1] == b"\r"
    assert calls[-2].startswith(b"\x1b[200~[Agency redirect ")
    assert calls[-2].endswith(b"replacement\nwith second line\x1b[201~")


def test_completion_can_win_interrupt_race(execution):
    execution.handle.write_terminal.side_effect = lambda _: execution.driver.pending.append(
        {
            "kind": "stop",
            "turn_id": "old-turn",
            "text": "finished",
        }
    )
    assert not execution.redirect("too late")
    assert execution._stop["text"] == "finished"
    assert execution.handle.write_terminal.call_count == 1
    execution.handle.close.assert_not_called()


@pytest.mark.parametrize("ack_interrupt", [False, True])
def test_partial_redirect_reaps_process_before_returning_false(execution, ack_interrupt):
    if ack_interrupt:
        execution.handle.write_terminal.side_effect = lambda data: (
            execution.driver.pending.append({"kind": "interrupt", "turn_id": "old-turn"})
            if data == b"\x1b"
            else None
        )
    assert not execution.redirect("replacement")
    execution.handle.close.assert_called_once()
    assert not execution._active
    assert execution._failure


def test_pty_attempt_registers_controls_and_returns_only_native_completion(execution, monkeypatch):
    execution._active = False
    handle = execution.handle

    def write(data):
        if data == b"\r":
            execution.driver.pending += [
                {"kind": "submit", "turn_id": "fresh", "prompt": execution._expected_prompt},
                {"kind": "stop", "turn_id": "fresh", "text": "native final"},
            ]

    handle.write_terminal.side_effect = write
    launch = Mock(return_value=handle)
    monkeypatch.setattr("agency.harness.ptrace.supervisor.agProxyPtrace.launch", launch)
    result = execution.run("prompt")
    assert result.ok and result.final_text == "native final"
    assert result.session_blob == b"durable-native-session"
    assert launch.call_args.kwargs["pty_size"] == (120, 36)
    assert "stdin_data" not in launch.call_args.kwargs
    assert launch.call_args.kwargs["policy"] is execution.runtime.syscall_policy
    execution.runtime.register_control_handle.assert_called_once_with(handle)
    execution.runtime.register_redirect.assert_called_once()
    handle.wait.assert_not_called()
    handle.close.assert_called_once()
    assert not execution.driver.root.exists()


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_unsuccessful_attempt_always_closes_and_cleans_state(execution, monkeypatch, failure):
    if failure == "exit":
        execution.handle.returncode = 1
    else:
        execution.driver.is_ready = False
    monkeypatch.setattr(
        "agency.harness.ptrace.supervisor.agProxyPtrace.launch", Mock(return_value=execution.handle)
    )
    with pytest.raises(RuntimeError):
        execution.run("prompt")
    execution.handle.close.assert_called_once()
    assert not execution.driver.root.exists()


@pytest.mark.parametrize(
    "harness,path",
    [
        ("codex", "sessions/2026/09/09/rollout.jsonl"),
        ("grok", "sessions/session.json"),
        ("opencode", "data/opencode/opencode.db"),
    ],
)
def test_session_round_trip_excludes_auth_and_config(harness, path, tmp_path):
    root = tmp_path / "source"
    target = root / path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"native persisted conversation")
    (root / "auth.json").write_text("secret")
    blob = snapshot_session(
        root,
        harness,
        "session",
    )
    assert b"secret" not in blob and b"auth.json" not in blob
    destination = tmp_path / "restored"
    restore_session(destination, harness, "session", blob)
    assert (destination / path).read_bytes() == target.read_bytes()


@pytest.mark.parametrize(
    "path",
    ["/tmp/escape.json", "sessions/../../escape.json", "auth.json", "config.toml", "sessions/x.py"],
)
def test_restore_rejects_non_session_paths(tmp_path, path):
    blob = json.dumps(
        {
            "version": 1,
            "harness": "codex",
            "session_id": "s",
            "files": {path: base64.b64encode(b"payload").decode()},
        }
    ).encode()
    with pytest.raises(ValueError, match="path"):
        restore_session(tmp_path, "codex", "s", blob)


def test_snapshot_rejects_symlink(tmp_path):
    (tmp_path / "sessions").mkdir()
    (tmp_path / "auth.json").write_text("secret")
    (tmp_path / "sessions/leak.json").symlink_to(tmp_path / "auth.json")
    with pytest.raises(ValueError, match="unsafe"):
        snapshot_session(tmp_path, "codex", "s")


@pytest.mark.parametrize("session_id,blob", [("s", None), (None, b"data")])
def test_resume_requires_id_and_blob_together(tmp_path, session_id, blob):
    with pytest.raises(ValueError):
        restore_session(tmp_path, "codex", session_id, blob)


@pytest.mark.parametrize(
    "patch",
    [{"version": 2}, {"harness": "grok"}, {"session_id": "other"}, {"files": {}}, {"files": []}],
)
def test_restore_rejects_incompatible_or_empty_bundle(tmp_path, patch):
    bundle = {
        "version": 1,
        "harness": "codex",
        "session_id": "s",
        "files": {"sessions/session.json": "eA=="},
    }
    bundle.update(patch)
    with pytest.raises(ValueError):
        restore_session(tmp_path, "codex", "s", json.dumps(bundle).encode())
    assert not list(tmp_path.iterdir())


def test_restore_validates_all_paths_before_writing(tmp_path):
    bundle = {
        "version": 1,
        "harness": "codex",
        "session_id": "s",
        "files": {"sessions/session.json": "eA==", "../escape.json": "eA=="},
    }
    with pytest.raises(ValueError):
        restore_session(tmp_path, "codex", "s", json.dumps(bundle).encode())
    assert not list(tmp_path.iterdir())


def test_restore_rejects_symlinked_parent(tmp_path):
    root = tmp_path / "state"
    root.mkdir()
    (root / "sessions").symlink_to(tmp_path, target_is_directory=True)
    bundle = {
        "version": 1,
        "harness": "codex",
        "session_id": "s",
        "files": {"sessions/session.json": "eA=="},
    }
    with pytest.raises(ValueError, match="unsafe"):
        restore_session(root, "codex", "s", json.dumps(bundle).encode())
    assert not (tmp_path / "session.json").exists()


def test_sqlite_snapshot_includes_wal_without_mutating_live_database(runtime, tmp_path):
    import sqlite3

    driver = driver_for(
        agharness_backend.for_config("opencode", runtime.agconfig),
        runtime,
        tmp_path,
        None,
        None,
        None,
    )
    driver.session_id = "s"
    database = tmp_path / "data/opencode/opencode.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE messages (text)")
        connection.execute("INSERT INTO messages VALUES ('persisted response')")
        connection.commit()
        original = database.read_bytes()
        blob = driver.snapshot()
        assert database.read_bytes() == original
        restored = tmp_path / "restored"
        restore_session(restored, "opencode", "s", blob)
        with sqlite3.connect(restored / "data/opencode/opencode.db") as backup:
            assert backup.execute("SELECT text FROM messages").fetchone() == ("persisted response",)


def test_config_is_cleaned_even_when_process_close_raises(execution, monkeypatch):
    execution.handle.returncode = 1
    execution.handle.close.side_effect = RuntimeError("failed reap")
    monkeypatch.setattr(
        "agency.harness.ptrace.supervisor.agProxyPtrace.launch", Mock(return_value=execution.handle)
    )
    with pytest.raises(RuntimeError, match="failed reap"):
        execution.run("prompt")
    assert not execution.driver.root.exists()


def test_terminal_confirmation_is_required_before_second_escape(execution):
    execution.driver.confirm_interrupt = True
    execution.driver.interrupt_pending = Mock(return_value=True)
    escapes = []

    def write(data):
        if data == b"\x1b":
            escapes.append(data)
            if len(escapes) == 2:
                execution.driver.pending.append({"kind": "interrupt", "turn_id": "old-turn"})
        if data == b"\r":
            execution.driver.pending.append(
                {"kind": "submit", "turn_id": "fresh", "prompt": execution._expected_prompt}
            )

    execution.handle.write_terminal.side_effect = write
    assert execution.redirect("replacement")
    assert len(escapes) == 2
    execution.driver.interrupt_pending.assert_called_once()


@pytest.mark.parametrize("name", ["codex", "grok"])
def test_driver_fences_native_turns_and_rejects_child_events(name, runtime, tmp_path):
    driver = driver_for(
        agharness_backend.for_config(name, runtime.agconfig), runtime, tmp_path, None, None, None
    )
    if name == "codex":
        payload = {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "s",
            "turn_id": "t",
            "prompt": "prompt",
        }
        child = {**payload, "agent_id": "child", "session_id": "child-s"}
    else:
        payload = {
            "hookEventName": "user_prompt_submit",
            "sessionId": "s",
            "promptId": "t",
            "prompt": "<user_query>\nprompt\n</user_query>",
        }
        child = {**payload, "subagentType": "child", "sessionId": "child-s"}
    (tmp_path / "events/1.json").write_text(json.dumps(child))
    (tmp_path / "events/2.json").write_text(json.dumps(payload))
    assert driver.events() == [{"kind": "submit", "turn_id": "t", "prompt": "prompt"}]
    assert driver.session_id == "s"


@pytest.mark.parametrize("name", ["codex", "grok"])
def test_driver_requires_matching_complete_native_record_and_ignores_partial_tail(
    name, runtime, tmp_path
):
    driver = driver_for(
        agharness_backend.for_config(name, runtime.agconfig), runtime, tmp_path, None, None, None
    )
    path = tmp_path / "transcript.jsonl"
    driver.transcript_path = path
    event = {"kind": "stop", "turn_id": "t", "text": "clipped"}

    def complete(turn):
        if name == "codex":
            return {"type": "event_msg", "payload": {"type": "task_complete", "turn_id": turn}}
        return {"params": {"update": {"sessionUpdate": "turn_completed", "prompt_id": turn}}}

    path.write_text(json.dumps(complete("wrong")) + "\n" + json.dumps(complete("t")))
    assert not driver.completed(event)
    with path.open("a") as file:
        file.write("\n")
    assert driver.completed(event)


def test_grok_full_answer_is_read_from_committed_updates_not_clipped_stop_hook(runtime, tmp_path):
    driver = driver_for(
        agharness_backend.for_config("grok", runtime.agconfig), runtime, tmp_path, None, None, None
    )
    driver.transcript_path = tmp_path / "transcript.jsonl"
    text = "long answer " * 2000
    rows = [
        {
            "params": {
                "_meta": {"promptId": "t"},
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": text},
                },
            }
        },
        {
            "params": {
                "update": {
                    "sessionUpdate": "turn_completed",
                    "prompt_id": "t",
                    "usage": {"inputTokens": 10, "outputTokens": 20},
                }
            }
        },
    ]
    driver.transcript_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    event = {"kind": "stop", "turn_id": "t", "text": "clipped"}
    assert driver.completed(event)
    assert event["text"] == text
    assert (event["input_tokens"], event["output_tokens"]) == (10, 20)


def test_stream_cancellation_closes_upstream_generator():
    import asyncio
    from agency.harness.adapters.pty_session import stream_response

    closed = []

    async def upstream():
        try:
            yield {"type": "delta", "content": "uncommitted"}
            yield {"type": "done", "message": "committed"}
        finally:
            closed.append(True)

    async def run():
        stream = stream_response(
            SimpleNamespace(dispatch_stream_async=lambda *args: upstream()),
            "token",
            {},
            "model",
            lambda items, model: [items[0]["message"]],
        )
        assert await anext(stream) == "committed"
        await stream.aclose()

    asyncio.run(run())
    assert closed == [True]


def test_paused_time_does_not_consume_native_ack_deadline(execution, monkeypatch):
    from agency.harness.adapters import pty_session

    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        clock.now += seconds

    monkeypatch.setattr(
        pty_session, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep)
    )
    execution.handle.is_paused.side_effect = lambda: clock.now < 0.2
    execution._wait_until(lambda: clock.now >= 0.225, "resumed acknowledgment")
    assert clock.now > execution.INPUT_TIMEOUT


def test_grok_interrupt_accepts_an_already_cleared_input(runtime, tmp_path):
    driver = driver_for(
        agharness_backend.for_config("grok", runtime.agconfig), runtime, tmp_path, None, None, None
    )
    driver._last_prompt = "[Agency run test]\nprevious prompt"
    driver._last_turn_id = "turn"
    handle = SimpleNamespace(
        terminal_screen=lambda: (["│ ❯   │"], 4, 0, 1),
        write_terminal=Mock(),
    )

    def wait_until(predicate, description):
        assert predicate(), description

    driver.clear_input(handle, wait_until)
    handle.write_terminal.assert_not_called()


@pytest.mark.parametrize("suffix", ["\n", " \n"])
def test_opencode_acknowledges_native_trailing_whitespace(execution, suffix):
    execution.driver.name = "opencode"
    execution.driver.prompt_matches = OpencodeDriver.prompt_matches.__get__(execution.driver)
    execution._expected_prompt = "[Agency run test]\ncurrent instruction"
    execution._turn_id = None
    execution.driver.pending = [
        {
            "kind": "submit",
            "turn_id": "current",
            "prompt": execution._expected_prompt + suffix,
        }
    ]
    execution._poll()
    assert execution._turn_id == "current"


def test_grok_interrupt_clears_collapsed_multiline_paste(runtime, tmp_path):
    driver = driver_for(
        agharness_backend.for_config("grok", runtime.agconfig), runtime, tmp_path, None, None, None
    )
    driver._last_prompt = "\n".join(["[Agency run test]"] + ["previous line"] * 10)
    driver._last_turn_id = "turn"
    handle = SimpleNamespace(
        terminal_screen=lambda: (["│ ❯ [Pasted: 11 lines]  │"], 24, 0, 1),
        write_terminal=Mock(),
    )

    def wait_until(predicate, description):
        assert predicate(), description

    driver.clear_input(handle, wait_until)
    handle.write_terminal.assert_called_once_with(b"\x03")


# ---------------------------------------------------------------------------
# run_pty_attempt's profiler-status bridge timeout
# ---------------------------------------------------------------------------


def test_profiler_bridge_timeout_is_not_too_tight_for_container_startup():
    """Was 1s: tight enough that ordinary container-startup load made every
    failure of this status check silent (run_pty_attempt's blanket except
    around bridge.profiler_settings()) -- profiler.enabled would come back
    False with no error, and every harness:launch/startup_ready/submit/
    await_cli phase span for the whole attempt would silently become a
    no-op. Guarded directly since the failure mode has no visible symptom
    to catch it with otherwise."""
    from agency.harness.adapters.pty_drivers import _PROFILER_BRIDGE_TIMEOUT_S

    assert _PROFILER_BRIDGE_TIMEOUT_S >= 5.0
