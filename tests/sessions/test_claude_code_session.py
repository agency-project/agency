"""agency/sessions/claude_code.py against a scripted stand-in for `claude`.

The fake CLI (tests/fixtures/fake_claude_session.py) posts the same hook and
OTLP payloads the real CLI does; the assertions read the result back through
the live trajectory reader the web UI uses, not through private state.
"""

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pyte
import pytest

from agency.agpolicy import agpolicy
from agency.configs.agconfig import agconfig, dataloggerconfig
from agency.observability.agdatalogger import agDataLogger
from agency.sessions import claude_code
from agency.sessions import session as session_module
from agency.sessions.terminal import _Screen

from .helpers import (
    TerminalClient,
    actions,
    eventually,
    hook,
    linux_only,
    otlp,
    post,
    process_state,
    require,
    rows,
    trajectory,
    with_model_call,
)


def test_hook_endpoint_rejects_requests_without_the_session_token(make_session):
    session = make_session()
    assert hook(session, "Stop")[0] == 200
    for headers in ({}, {"Authorization": "Bearer wrong"}):
        status, _ = post(session, "/hooks", {"hook_event_name": "Stop"}, headers)
        assert status == 401
    assert post(session, "/v1/logs", {}, {"x-agency-token": "wrong"})[0] == 401


def test_hook_replies_are_empty_so_claude_keeps_its_own_permission_flow(make_session):
    session = make_session()
    for name in ("PreToolUse", "PermissionRequest"):
        status, body = hook(
            session, name, tool_name="Bash", tool_input={"command": "rm x"}, tool_use_id="t1"
        )
        assert (status, body) == (200, b"")


def test_session_registers_as_a_claude_code_agent_with_its_model(make_session):
    session = make_session()
    hook(session, "SessionStart", source="startup", model="claude-sonnet-5.5")
    agent = trajectory(session).agents["claude"]
    assert agent["identity"] == "execution_actor"
    assert agent["harness"] == "claude_code"
    assert agent["model"] == "claude-sonnet-5.5"


def test_tool_hooks_become_a_completed_trajectory_action(make_session):
    session = make_session()
    hook(session, "UserPromptSubmit", prompt="list files", prompt_id="p1")
    hook(session, "PreToolUse", tool_name="Bash", tool_input={"command": "ls"}, tool_use_id="t1")
    hook(
        session,
        "PostToolUse",
        tool_name="Bash",
        tool_input={"command": "ls"},
        tool_use_id="t1",
        tool_response={"stdout": "a.txt"},
        duration_ms=40,
    )
    [action] = [a for a in actions(trajectory(session)) if a["name"] == "Bash"]
    assert action["outcome"] == "success"
    assert action["command"] == "ls"
    assert "a.txt" in action["result"]
    assert action["model_tool_call_id"] == "t1"


def test_model_request_is_placed_at_end_minus_its_reported_duration(make_session):
    session = make_session()
    ended = time.time()
    status, body = otlp(
        session,
        "api_request",
        ended,
        model="claude-sonnet-5.5",
        duration_ms=1500,
        input_tokens=100,
        cache_read_tokens=20,
        cache_creation_tokens=5,
        output_tokens=30,
        request_id="req_123",
    )
    assert (status, json.loads(body)) == (200, {})
    [model_call] = actions(trajectory(session), "model")
    assert model_call["outcome"] == "success"
    assert model_call["started_ts"] == pytest.approx(ended - 1.5, abs=1e-3)
    assert model_call["end_ts"] == pytest.approx(ended, abs=1e-3)
    assert model_call["tokens"] == 125
    assert model_call["output_tokens"] == 30
    assert model_call["model"] == "claude-sonnet-5.5"


def test_failed_model_request_is_recorded_as_failed(make_session):
    session = make_session()
    otlp(session, "claude_code.api_error", time.time(), model="m", duration_ms=10, error="429")
    [model_call] = actions(trajectory(session), "model")
    assert model_call["outcome"] == "failed"


def test_rejected_permission_closes_the_tool_and_the_input_request(make_session):
    session = make_session()
    hook(session, "PreToolUse", tool_name="Write", tool_input={"file_path": "x"}, tool_use_id="w1")
    hook(session, "PermissionRequest", tool_name="Write", tool_input={"file_path": "x"})
    model = trajectory(session)
    assert model.signals["input:claude"]["active"]
    assert model.agents["claude"]["status"] == "waiting_input"
    otlp(
        session,
        "tool_decision",
        time.time(),
        tool_use_id="w1",
        decision="reject",
        source="user_reject",
    )
    model = trajectory(session)
    [action] = [a for a in actions(model) if a["name"] == "Write"]
    assert action["outcome"] != "running"
    assert "Rejected (user_reject)" in action["error"]
    assert not model.signals["input:claude"]["active"]
    # A user rejection ends the turn, and Claude fires no Stop hook for it.
    assert model.agents["claude"]["status"] == "agent_idle"


def test_cancelled_tool_closes_from_telemetry_and_ends_the_turn(make_session):
    # Escape on a running tool fires no hook; only the telemetry reports it.
    session = make_session()
    hook(session, "UserPromptSubmit", prompt="sleep", prompt_id="p1")
    hook(
        session,
        "PreToolUse",
        tool_name="Bash",
        tool_input={"command": "sleep 30"},
        tool_use_id="s1",
    )
    otlp(
        session,
        "tool_result",
        time.time(),
        tool_use_id="s1",
        success="false",
        error_type="ShellError",
    )
    model = trajectory(session)
    [action] = [a for a in actions(model) if a["name"] == "Bash"]
    assert "Cancelled by the user" in action["error"]
    assert model.agents["claude"]["status"] == "agent_idle"


def test_failure_telemetry_for_an_already_closed_tool_changes_nothing(make_session):
    session = make_session()
    hook(session, "UserPromptSubmit", prompt="ls", prompt_id="p1")
    hook(session, "PreToolUse", tool_name="Bash", tool_input={"command": "false"}, tool_use_id="f1")
    hook(session, "PostToolUseFailure", tool_name="Bash", tool_use_id="f1", error="exit 1")
    otlp(
        session,
        "tool_result",
        time.time(),
        tool_use_id="f1",
        success="false",
        error_type="ShellError",
    )
    model = trajectory(session)
    [action] = [a for a in actions(model) if a["name"] == "Bash"]
    assert action["error"] == "exit 1"
    assert model.agents["claude"]["status"] == "running_harness"


def test_viewer_sizes_are_clamped_rather_than_refused():
    assert session_module._dimension(1, 80) == 2
    assert session_module._dimension(5000, 80) == 1000
    assert session_module._dimension(None, 80) == 80
    assert session_module._dimension("nan", 24) == 24


def test_tool_without_any_completion_is_closed_after_the_turn_ends(make_session, monkeypatch):
    monkeypatch.setattr(session_module, "_UNREPORTED_TOOL_GRACE_S", 0.05)
    session = make_session()
    hook(
        session, "PreToolUse", tool_name="Bash", tool_input={"command": "sleep 9"}, tool_use_id="b1"
    )
    hook(session, "Stop", last_assistant_message="stopped")

    def closed():
        [action] = [a for a in actions(trajectory(session)) if a["name"] == "Bash"]
        assert "no completion reported" in action["error"]

    eventually(closed)
    assert trajectory(session).agents["claude"]["status"] == "agent_idle"


def test_settings_file_subscribes_http_hooks_without_exposing_the_token(make_session):
    session = make_session()
    path = session.run_dir / claude_code.SETTINGS_FILE
    settings = json.loads(path.read_text())
    assert set(settings["hooks"]) == set(claude_code.HOOK_EVENTS)
    handler = settings["hooks"]["PreToolUse"][0]["hooks"][0]
    assert handler["type"] == "http"
    assert handler["url"] == f"http://127.0.0.1:{session.port}/hooks"
    assert session.host_token not in path.read_text()
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert oct((session.run_dir / session_module.SESSION_FILE).stat().st_mode & 0o777) == "0o600"


@pytest.mark.parametrize(
    "cli_args",
    [
        [],
        ["--continue"],
        ["--resume", "native-session"],
        ["--resume", "native-session", "--fork-session"],
        ["--session-id", "8a62efba-931b-47c0-9a83-cce9d04b4afb"],
    ],
)
def test_native_conversation_selection_remains_owned_by_claude(make_session, cli_args):
    session = make_session(cli_args=cli_args)
    argv = session._cli_argv()
    assert argv == [
        session.binary,
        "--settings",
        str(session.run_dir / claude_code.SETTINGS_FILE),
        *cli_args,
    ]
    assert session.session_id not in argv


@pytest.mark.parametrize("form", ["file", "json", "equals", "last"])
@pytest.mark.parametrize("enforce", [False, True])
def test_explicit_settings_compose_user_fields_and_hooks(make_session, tmp_path, form, enforce):
    user_hook = {"hooks": [{"type": "command", "command": "printf user-hook"}]}
    source = {
        "permissions": {"defaultMode": "plan", "deny": ["Bash(rm *)"]},
        "future": {"opaque": [1, {"retained": True}]},
        "hooks": {"SessionStart": [user_hook, user_hook], "PreToolUse": [user_hook]},
    }
    path = tmp_path / "user-settings.json"
    original = json.dumps(source)
    path.write_text(original)
    arguments = {
        "file": ["--settings", path.name],
        "json": ["--settings", original],
        "equals": [f"--settings={original}"],
        "last": ["--settings", "ignored-missing.json", f"--settings={path.name}"],
    }[form]
    policy = agpolicy(tool_hooks={"Bash": lambda event: True}) if enforce else None
    session = make_session(cli_args=[*arguments, "--permission-mode", "manual"], policy=policy)
    merged = json.loads((session.run_dir / claude_code.SETTINGS_FILE).read_text())
    assert merged["permissions"] == source["permissions"]
    assert merged["future"] == source["future"]
    assert merged["hooks"]["SessionStart"][:2] == [user_hook, user_hook]
    assert merged["hooks"]["PreToolUse"][0] == user_hook
    agency_pre_tool = merged["hooks"]["PreToolUse"][1]["hooks"][0]
    assert agency_pre_tool["type"] == ("command" if enforce else "http")
    assert session._cli_argv() == [
        session.binary,
        "--settings",
        str(session.run_dir / claude_code.SETTINGS_FILE),
        "--permission-mode",
        "manual",
    ]
    assert path.read_text() == original


@pytest.mark.parametrize(
    "arguments",
    [
        ["--settings"],
        ["--settings", "missing.json"],
        ["--settings", "[]"],
        ["--settings", "{"],
        ["--settings", '{"future":NaN}'],
    ],
)
def test_malformed_explicit_settings_fail_before_launch(tmp_path, arguments):
    with pytest.raises(ValueError, match="--settings"):
        claude_code._additional_settings(arguments, tmp_path)


@pytest.mark.parametrize("source", ["123", "~/settings.json"])
def test_explicit_settings_file_names_are_not_parsed_as_json(tmp_path, source):
    path = tmp_path / source
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = {"permissions": {"defaultMode": "plan"}, "future": {"retained": True}}
    path.write_text(json.dumps(settings))
    assert claude_code._additional_settings(["--settings", source], tmp_path) == (settings, [])


def test_settings_option_after_cli_terminator_is_a_native_prompt(tmp_path):
    arguments = ["--", "--settings", "prompt text"]
    assert claude_code._additional_settings(arguments, tmp_path) == ({}, arguments)


def test_real_claude_keeps_user_startup_hook_and_agency_http_collection(
    make_session, monkeypatch, tmp_path
):
    binary = os.environ.get("AGENCY_CAPTURE_TEST_CLAUDE")
    if not binary:
        pytest.skip("set AGENCY_CAPTURE_TEST_CLAUDE for real local settings integration")
    marker = tmp_path / "user-hook.marker"
    command = shlex.join(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('user hook')",
            str(marker),
        ]
    )
    source = {
        "permissions": {"defaultMode": "plan"},
        "hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": command}]}]},
    }
    path = tmp_path / "custom.json"
    original = json.dumps(source)
    path.write_text(original)
    state = tmp_path / "claude-config"
    state.mkdir()
    (state / ".claude.json").write_text(
        json.dumps({"hasCompletedOnboarding": True, "theme": "dark"})
    )
    monkeypatch.setenv("HOME", str(state))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(state))
    session = make_session(
        binary=Path(binary),
        cli_args=[
            f"--settings={path}",
            "--setting-sources",
            "user",
            "--model",
            "gpt-6-luna",
            "-p",
            "fixture",
        ],
        policy=agpolicy(tool_hooks={"Bash": lambda event: True}),
    )
    events = []
    handle = session.handle_hook

    def record(payload):
        events.append(payload.get("hook_event_name"))
        return handle(payload)

    monkeypatch.setattr(session, "handle_hook", record)
    env = session._environment()
    env.update(
        ANTHROPIC_BASE_URL="http://127.0.0.1:1",
        ANTHROPIC_API_KEY="fixture-no-model",
        ANTHROPIC_AUTH_TOKEN="fixture-no-model",
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
        CLAUDE_CODE_USE_BEDROCK="0",
        CLAUDE_CODE_USE_VERTEX="0",
        CLAUDE_CODE_USE_FOUNDRY="0",
    )
    for key in ("LITE_LLM_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        env.pop(key, None)
    process = subprocess.Popen(
        session._cli_argv(),
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        eventually(lambda: require(marker.exists() and "UserPromptSubmit" in events), timeout=10)
        assert path.read_text() == original
        settings = json.loads((session.run_dir / claude_code.SETTINGS_FILE).read_text())
        assert settings["hooks"]["PreToolUse"][-1]["hooks"][0]["type"] == "command"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


@pytest.mark.parametrize(
    "optout",
    [
        ("--bare", None),
        ("--safe-mode", None),
        ("CLAUDE_CODE_SIMPLE", "1"),
        ("CLAUDE_CODE_SIMPLE", "true"),
        ("CLAUDE_CODE_SAFE_MODE", "1"),
        ("CLAUDE_CODE_SAFE_MODE", "TRUE"),
        ("settings", None),
    ],
)
@pytest.mark.parametrize("enforce", [False, True])
def test_explicit_hook_optouts_preserve_native_behavior_and_reject_tool_policy(
    tmp_path, monkeypatch, optout, enforce
):
    name, value = optout
    arguments = []
    if name.startswith("--"):
        arguments = [name]
    elif name == "settings":
        arguments = ["--settings", '{"disableAllHooks":true}']
    else:
        monkeypatch.setenv(name, value)
    kwargs = dict(
        cwd=tmp_path,
        binary="claude",
        cli_args=arguments,
        run_dir=tmp_path,
        ui_port=0,
        use_scope=False,
        policy=agpolicy(tool_hooks={"Bash": lambda event: True}) if enforce else None,
    )
    if enforce:
        with pytest.raises(ValueError, match="tool policy requires native hooks"):
            claude_code.ClaudeCodeSession(**kwargs)
    else:
        session = claude_code.ClaudeCodeSession(**kwargs)
        assert session.engine_coverage["tools"] == "unavailable"
        assert session.engine_coverage["turns"] == "unavailable"
        assert session.engine_coverage["llm"] == "claude_code_reported"
        if name == "settings":
            assert session._claude_settings["disableAllHooks"] is True
        else:
            assert session._claude_cli_args == arguments
    assert claude_code.ClaudeCodeSession.engine_coverage["tools"] == "hook_boundaries"


def test_hook_optout_keeps_independent_syscall_policy_available(tmp_path):
    session = claude_code.ClaudeCodeSession(
        cwd=tmp_path,
        binary="claude",
        cli_args=["--bare"],
        run_dir=tmp_path,
        ui_port=0,
        use_scope=False,
        policy=agpolicy(syscall_hooks={"execve": lambda event: True}),
    )
    assert session.trace and "execve" in session.runtime_config.ptrace.syscalls
    assert session.engine_coverage["tools"] == "unavailable"


def test_claude_environment_keeps_user_settings_and_drops_parent_session_markers(
    make_session, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://router.example")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent")
    env = make_session()._environment()
    assert env["ANTHROPIC_BASE_URL"] == "https://router.example"
    assert "CLAUDECODE" not in env and "CLAUDE_CODE_SESSION_ID" not in env
    assert env["OTEL_EXPORTER_OTLP_LOGS_PROTOCOL"] == "http/json"
    assert env["OTEL_METRICS_EXPORTER"] == "none"


# -- Through the real terminal endpoint, with the fake CLI -----------------------


@linux_only
def test_terminal_endpoint_requires_the_session_token(make_session):
    from websockets.exceptions import InvalidStatus
    from websockets.sync.client import connect

    session = make_session()
    with pytest.raises(InvalidStatus):
        connect(f"ws://127.0.0.1:{session.port}/terminal").recv(timeout=5)
    assert session.terminal is None  # nothing was spawned for the failed attempt


@linux_only
def test_fake_cli_session_end_to_end(make_session):
    session = make_session()
    client = TerminalClient(session)
    client.wait_for("fake claude ready")
    client.wait_for("hook reply 200 b''")
    client.wait_for("session leader True tty /dev/pts/")
    client.type(b"t")
    client.wait_for("tool done")
    client.type(b"m")
    client.wait_for("model done")
    model = eventually(lambda: with_model_call(trajectory(session)))
    [tool] = [a for a in actions(model) if a["name"] == "Bash"]
    assert tool["outcome"] == "success"
    assert model.agents["claude"]["model"] == "fake-model"
    client.type(b"q")
    eventually(lambda: require(session.ended.is_set()))
    client.pump()
    assert {"type": "exited", "returncode": 0} in client.status
    client.close()
    states = [
        json.loads(payload)["state"]
        for (payload,) in rows(session, "SELECT payload FROM events WHERE type='agent_state'")
    ]
    assert states[-1] == "agent_exit"


@linux_only
def test_reconnect_restores_a_usable_screen(make_session):
    session = make_session()
    first = TerminalClient(session)
    first.wait_for("fake claude ready")
    first.type(b"hello")
    first.wait_for("hello")
    first.close()
    second = TerminalClient(session)
    second.pump()
    assert "fake claude ready" in second.text()
    assert "hello" in second.text()
    assert (2004 << 5) in second.screen.mode  # bracketed paste survived
    second.type(b"t")  # and input still works on the restored screen
    second.wait_for("tool done")
    second.close()


@linux_only
def test_resizing_the_browser_resizes_the_cli(make_session):
    session = make_session()
    client = TerminalClient(session)
    client.wait_for("fake claude ready")
    client.type(json.dumps({"type": "resize", "cols": 120, "rows": 40}))
    client.screen.resize(40, 120)
    client.wait_for("size 120x40")
    client.close()


@linux_only
def test_pause_resume_and_stop_without_a_scope_use_the_process_group(make_session):
    session = make_session()
    client = TerminalClient(session)
    client.wait_for("fake claude ready")
    session.pause()
    eventually(lambda: require(process_state(session.terminal.pid) == "T"))
    assert trajectory(session).agents["claude"]["paused"]
    session.resume()
    eventually(lambda: require(process_state(session.terminal.pid) != "T"))
    session.stop()
    eventually(lambda: require(session.ended.is_set()))
    client.close()


def _scopes_available():
    return sys.platform.startswith("linux") and session_module._user_scope_available()


@pytest.mark.skipif(not _scopes_available(), reason="needs systemd user scopes")
def test_scope_freezes_and_kills_the_whole_session_tree(make_session):
    from agency.observability.profiler import agprof

    session = make_session(use_scope=True)
    client = TerminalClient(session)
    client.wait_for("fake claude ready")
    eventually(lambda: require(session.scope_dir is not None))
    assert agprof.container_registered(session.scope_unit)
    client.type(b"b")
    client.wait_for("background ")
    background = int(client.text().split("background ")[1].split()[0])
    # It left the CLI's process group, so only the cgroup still holds it.
    assert os.getpgid(background) != os.getpgid(session.terminal.pid)
    events = session.scope_dir / "cgroup.events"
    # cgroup v2 freezes asynchronously; `frozen` flips once every task stopped.
    session.pause()
    eventually(lambda: require("frozen 1" in events.read_text()))
    session.resume()
    eventually(lambda: require("frozen 0" in events.read_text()))
    session.stop(force=True)
    eventually(lambda: require(session.ended.is_set()))
    eventually(lambda: require(not Path(f"/proc/{background}").exists()))
    client.close()


# -- Web UI relay and controls ------------------------------------------------------


@linux_only
def test_web_ui_relays_the_terminal_only_for_loopback_pages_with_the_token(
    make_session, monkeypatch
):
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    import agency.observability.agwebui.server as server

    session = make_session()
    monkeypatch.setattr(server, "_run_dir", session.run_dir)
    with TestClient(server.app) as client:
        assert client.get("/api/terminal").json() == {"available": True, "label": "Claude Code"}
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect(
                "/ws/terminal", headers={"origin": "https://evil.example"}
            ) as ws:
                ws.receive_bytes()
        assert refused.value.code == 4403
        with client.websocket_connect(
            "/ws/terminal", headers={"origin": "http://127.0.0.1:7860"}
        ) as ws:
            ws.send_text(json.dumps({"token": "wrong", "cols": 80, "rows": 24}))
            with pytest.raises(WebSocketDisconnect) as rejected:
                ws.receive_bytes()
            assert rejected.value.code == 4401
        with client.websocket_connect(
            "/ws/terminal", headers={"origin": "http://localhost:9999"}
        ) as ws:
            ws.send_text(json.dumps({"token": session.ui_token, "cols": 80, "rows": 24}))
            screen = _Screen(80, 24)
            stream = pyte.ByteStream(screen)
            deadline = time.monotonic() + 10
            while "fake claude ready" not in "\n".join(screen.display):
                assert time.monotonic() < deadline
                message = ws.receive()
                if message.get("bytes"):
                    stream.feed(message["bytes"])
            ws.send_bytes(b"q")
    eventually(lambda: require(session.ended.is_set()))


def test_relay_controls_a_registered_standalone_session(tmp_path, monkeypatch):
    from agency.agent import agent
    from agency.observability.agwebui import relay as relay_module

    calls = []

    class Target:
        def pause(self):
            calls.append("pause")

        def resume(self):
            calls.append("resume")

        def stop(self, force=False):
            calls.append("kill" if force else "stop")

    monkeypatch.setattr(agent, "all", classmethod(lambda cls: []))
    relay_module.register_standalone_target("claude", Target())
    try:
        relay = relay_module.ProfilerRelay(tmp_path)
        monkeypatch.setattr(relay, "_event", lambda *_args: None)
        relay.publish()
        published = json.loads((tmp_path / ".session").read_text())
        assert published["capabilities"]["claude"] == {
            "queue_message": False,
            "redirect": [],
            "run": [],
        }
        for number, kind in enumerate(("pause", "resume", "stop", "kill")):
            receipt = relay.dispatch(
                {
                    "id": f"c{number}",
                    "type": kind,
                    "targets": ["claude"],
                    "relay_id": relay.identifier,
                }
            )
            assert receipt["state"] == "accepted", receipt
        assert calls == ["pause", "resume", "stop", "kill"]
        refused = relay.dispatch(
            {
                "id": "c9",
                "type": "queue_message",
                "message": "hi",
                "targets": ["claude"],
                "relay_id": relay.identifier,
            }
        )
        assert refused["state"] == "failed"
    finally:
        relay_module.unregister_standalone_target("claude")
        relay.close()


# -- Small changes to existing modules ------------------------------------------------


def test_data_logger_keeps_a_reported_timestamp(tmp_path):
    import sqlite3

    logger = agDataLogger(agconfig(dataloggerconfig(db_path=str(tmp_path / "db.sqlite3"))))
    logger.start()
    logger.record_event("agent_state", {"state": "x"}, timestamp=123.5, flush=True)
    from tests.exchange_helpers import record_exchange

    record_exchange(logger, "label", timestamp=124.0)
    logger.stop()
    with sqlite3.connect(tmp_path / "db.sqlite3") as connection:
        assert connection.execute("SELECT timestamp FROM events").fetchone() == (123.5,)
        assert connection.execute("SELECT timestamp FROM exchanges").fetchone() == (124.0,)


@linux_only
def test_profiler_attributes_no_network_to_a_host_network_scope(tmp_path):
    from agency.observability.profiler import agprof

    scope = tmp_path / "scope"
    scope.mkdir()
    (scope / "cgroup.procs").write_text(f"{os.getpid()}\n")
    agprof.container_started("host-scope", str(scope), host_network=True)
    agprof.container_started("container-scope", str(scope))
    try:
        start = len(agprof._samples)
        agprof._Sampler._tick_io_net(None, 1, "host-scope", str(scope), io_from_pids=False)
        agprof._Sampler._tick_io_net(None, 1, "container-scope", str(scope), io_from_pids=False)
        names = {name for _t, name, _value in agprof._samples[start:]}
        assert "cg:container-scope:net_rx" in names
        assert not any(name.startswith("cg:host-scope:") for name in names)
    finally:
        agprof.container_stopped("host-scope")
        agprof.container_stopped("container-scope")
    assert "host-scope" not in agprof._cg_host_network
