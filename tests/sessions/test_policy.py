"""Session hooks enforce Agency denials without approving native CLI tools."""

import json
import os
import socket
import subprocess
import sys

import pytest

from agency.agpolicy import agpolicy
from agency.sessions.claude_code import ClaudeCodeSession
from agency.sessions.codex import CodexSession
from agency.sessions.session import hook_command

from .helpers import hook, rows


@pytest.mark.parametrize("session_class", [ClaudeCodeSession, CodexSession])
def test_tool_policy_denial_reaches_cli_and_existing_sqlite_event(make_session, session_class):
    observed = []

    def deny(arguments):
        observed.append(arguments)
        return False, "fixture command blocked"

    session = make_session(session_class=session_class, policy=agpolicy(tool_hooks={"Bash": deny}))
    status, raw = hook(
        session,
        "PreToolUse",
        tool_name="Bash",
        tool_input={"command": "touch denied-file"},
        tool_use_id="denied-tool",
    )

    assert status == 200
    assert json.loads(raw) == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "fixture command blocked",
        }
    }
    assert observed == [{"command": "touch denied-file"}]
    [record] = rows(session, "SELECT payload FROM events WHERE type='tool_call'")
    payload = json.loads(record[0])
    assert payload["allowed"] is False and payload["reason"] == "fixture command blocked"
    assert payload["annotation"]["model_tool_call_id"] == "denied-tool"
    assert session._pending_tool_ids() == []


@pytest.mark.parametrize("session_class", [ClaudeCodeSession, CodexSession])
def test_allowed_policy_does_not_override_the_cli_permission_flow(make_session, session_class):
    session = make_session(
        session_class=session_class, policy=agpolicy(tool_hooks={"Bash": lambda arguments: True})
    )
    status, raw = hook(
        session,
        "PreToolUse",
        tool_name="Bash",
        tool_input={"command": "printf allowed"},
        tool_use_id="allowed-tool",
    )

    assert (status, raw) == (200, b""), "Agency must not emit a native allow decision"
    [record] = rows(session, "SELECT payload FROM events WHERE type='tool_call'")
    assert json.loads(record[0])["allowed"] is True
    assert session._pending_tool_ids() == ["allowed-tool"]


@pytest.mark.parametrize("session_class", [ClaudeCodeSession, CodexSession])
def test_command_hook_forwards_the_native_denial(make_session, session_class):
    session = make_session(
        session_class=session_class,
        policy=agpolicy(tool_hooks={"Bash": lambda arguments: (False, "denied by fixture")}),
    )
    command = hook_command()
    if session_class is ClaudeCodeSession:
        settings = json.loads((session.run_dir / "claude_settings.json").read_text())
        command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    completed = subprocess.run(
        command,
        shell=True,
        input=json.dumps(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": {"command": "false"},
                "tool_use_id": "relay-tool",
            }
        ).encode(),
        capture_output=True,
        env=session._environment(),
        timeout=10,
    )

    assert completed.returncode == 0 and completed.stderr == b""
    decision = json.loads(completed.stdout)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    assert decision["permissionDecisionReason"] == "denied by fixture"


@pytest.mark.parametrize(
    "enforce,event,exit_code",
    [
        (True, "PreToolUse", 2),
        (False, "PreToolUse", 0),
        (True, "PostToolUse", 0),
        (False, "PostToolUse", 0),
    ],
)
def test_codex_transport_failure_blocks_only_enforced_pre_tool_checks(enforce, event, exit_code):
    # A bound, non-listening socket deterministically rejects connections.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        env = {
            **os.environ,
            "AGENCY_SESSION_HOOK_URL": f"http://127.0.0.1:{reserved.getsockname()[1]}/hooks",
            "AGENCY_SESSION_TOKEN": "transport-fixture-token",
        }
        env.pop("AGENCY_SESSION_ENFORCE_POLICY", None)
        if enforce:
            env["AGENCY_SESSION_ENFORCE_POLICY"] = "1"
        completed = subprocess.run(
            hook_command(),
            shell=True,
            input=json.dumps({"hook_event_name": event}).encode(),
            capture_output=True,
            env=env,
            timeout=10,
        )

    assert completed.returncode == exit_code
    assert completed.stdout == b""
    assert b"agency hook relay" in completed.stderr


@pytest.mark.parametrize(
    "policy",
    [
        agpolicy(syscall_hooks={"execve": lambda event: True}),
        agpolicy(default_to_deny=True),
    ],
)
def test_syscall_policy_cannot_silently_run_without_tracing(tmp_path, policy):
    with pytest.raises(ValueError, match="syscall policy requires tracing"):
        CodexSession(
            cwd=tmp_path,
            binary=sys.executable,
            cli_args=[],
            run_dir=tmp_path / "logs",
            ui_port=0,
            use_scope=False,
            terminal_mode="native",
            trace=False,
            policy=policy,
        )


def test_observe_only_session_does_not_inherit_parent_policy_enforcement(make_session, monkeypatch):
    monkeypatch.setenv("AGENCY_SESSION_ENFORCE_POLICY", "1")
    session = make_session(session_class=CodexSession)
    inherited_flag = session._environment().get("AGENCY_SESSION_ENFORCE_POLICY")
    assert inherited_flag != "1"


@pytest.mark.parametrize("enforce", [True, False])
def test_unexpected_pre_tool_handler_failure_cannot_approve_an_enforced_tool(
    make_session, monkeypatch, enforce
):
    policy = agpolicy(tool_hooks={"Bash": lambda arguments: True}) if enforce else None
    session = make_session(session_class=CodexSession, policy=policy)

    def fail(*_args):
        raise RuntimeError("fixture service failed")

    monkeypatch.setattr(session, "_tool_started", fail)
    status, raw = hook(session, "PreToolUse", tool_name="Bash", tool_input={"command": "true"})
    assert status == 200
    if enforce:
        decision = json.loads(raw)["hookSpecificOutput"]
        assert decision["permissionDecision"] == "deny"
        assert decision["permissionDecisionReason"] == "Agency policy check failed"
    else:
        assert raw == b""


@pytest.mark.parametrize("enforce", [True, False])
def test_claude_uses_command_pre_tool_hook_only_for_explicit_policy(make_session, enforce):
    policy = agpolicy(tool_hooks={"Bash": lambda arguments: True}) if enforce else None
    session = make_session(session_class=ClaudeCodeSession, policy=policy)
    settings = json.loads((session.run_dir / "claude_settings.json").read_text())
    pre_tool = settings["hooks"]["PreToolUse"][0]["hooks"][0]
    post_tool = settings["hooks"]["PostToolUse"][0]["hooks"][0]

    assert pre_tool["type"] == ("command" if enforce else "http")
    assert post_tool["type"] == "http", "observational hooks should retain their normal transport"
    if enforce:
        assert pre_tool["command"] == hook_command()


def test_claude_enforced_command_hook_blocks_when_the_policy_service_is_unavailable(make_session):
    session = make_session(
        session_class=ClaudeCodeSession,
        policy=agpolicy(tool_hooks={"Bash": lambda arguments: True}),
    )
    settings = json.loads((session.run_dir / "claude_settings.json").read_text())
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        env = session._environment()
        env["AGENCY_SESSION_HOOK_URL"] = f"http://127.0.0.1:{reserved.getsockname()[1]}/hooks"
        completed = subprocess.run(
            command,
            shell=True,
            input=json.dumps(
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "Bash",
                    "tool_input": {"command": "true"},
                }
            ).encode(),
            capture_output=True,
            env=env,
            timeout=10,
        )

    assert completed.returncode == 2 and completed.stdout == b""
