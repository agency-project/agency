"""agency/sessions/codex.py against a scripted stand-in for `codex`.

Payload shapes follow Codex CLI 0.162.1 as observed: Claude-compatible hook
JSON, and OTLP/JSON log records that carry their time in `event.timestamp`.
"""

import json
import os
import subprocess
import time
import tomllib

import pytest

from agency.sessions import codex
from agency.sessions.codex import CodexSession

from .helpers import (
    FIXTURES,
    TerminalClient,
    actions,
    eventually,
    hook,
    linux_only,
    otlp,
    require,
    trajectory,
)


@pytest.fixture
def make_codex(make_session):
    def make(**kwargs):
        return make_session(CodexSession, FIXTURES / "fake_codex_session.py", **kwargs)

    return make


def overrides(argv):
    """`-c key=value` pairs in *argv*, values parsed as Codex parses them."""
    parsed = {}
    for flag, value in zip(argv, argv[1:]):
        if flag == "-c":
            key, _, raw = value.partition("=")
            parsed[key] = tomllib.loads(f"value = {raw}")["value"]
    return parsed


def codex_log(session, name, at, **attributes):
    return otlp(
        session,
        f"codex.{name}",
        at,
        time_in_attribute=True,
        **{"conversation.id": "conv-1", "model": "gpt-6.1-sol", **attributes},
    )


def test_hook_definitions_are_identical_across_sessions_so_one_trust_lasts(make_codex):
    first, second = make_codex(), make_codex()
    first_hooks = {k: v for k, v in overrides(first._cli_argv()).items() if k.startswith("hooks.")}
    second_hooks = {
        k: v for k, v in overrides(second._cli_argv()).items() if k.startswith("hooks.")
    }
    assert first_hooks == second_hooks
    assert set(first_hooks) == {f"hooks.{event}" for event in codex.HOOK_EVENTS}
    handler = first_hooks["hooks.PreToolUse"][0]["hooks"][0]
    # Codex clamps (and warns about) an Interrupt hook timeout above 3 s.
    assert handler == {"type": "command", "command": codex.hook_command(), "timeout": 3}


def test_secrets_stay_out_of_the_command_line(make_codex):
    session = make_codex()
    argv = session._cli_argv()
    assert session.host_token not in " ".join(argv)
    exporter = overrides(argv)["otel.exporter"]["otlp-http"]
    assert exporter == {"endpoint": f"http://127.0.0.1:{session.port}/v1/logs", "protocol": "json"}
    env = session._environment()
    assert env["OTEL_EXPORTER_OTLP_LOGS_HEADERS"] == f"x-agency-token={session.host_token}"
    assert env["AGENCY_SESSION_HOOK_URL"] == f"http://127.0.0.1:{session.port}/hooks"


def test_hook_relay_forwards_a_payload_and_makes_no_decision(make_codex):
    session = make_codex()
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "tool_use_id": "exec-1",
    }
    finished = subprocess.run(
        codex.hook_command(),
        shell=True,
        input=json.dumps(payload).encode(),
        capture_output=True,
        env=session._environment(),
        timeout=10,
    )
    assert (finished.returncode, finished.stdout) == (0, b"")
    [action] = actions(trajectory(session))
    assert (action["name"], action["command"], action["outcome"]) == ("Bash", "ls", "running")


def test_hook_relay_outside_a_session_does_nothing():
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"AGENCY_SESSION_HOOK_URL", "AGENCY_SESSION_TOKEN"}
    }
    finished = subprocess.run(
        codex.hook_command(), shell=True, input=b"{}", capture_output=True, env=environment
    )
    assert (finished.returncode, finished.stdout, finished.stderr) == (0, b"", b"")


def test_model_request_spans_request_start_to_response_completed(make_codex):
    session = make_codex()
    headers_at = time.time() - 3
    codex_log(
        session,
        "api_request",
        headers_at,
        duration_ms=500,
        attempt=0,
        **{"http.response.status_code": 200},
    )
    assert actions(trajectory(session), "model") == []  # not over until the stream ends
    codex_log(
        session,
        "sse_event",
        headers_at + 2,
        input_token_count=10748,
        output_token_count=41,
        ttft_ms=587,
        **{"event.kind": "response.completed"},
    )
    [model_call] = actions(trajectory(session), "model")
    assert model_call["started_ts"] == pytest.approx(headers_at - 0.5, abs=0.01)
    assert model_call["end_ts"] == pytest.approx(headers_at + 2, abs=0.01)
    assert (model_call["tokens"], model_call["output_tokens"]) == (10748, 41)
    assert model_call["model"] == "gpt-6.1-sol"


def test_rejected_http_status_is_a_failed_model_request(make_codex):
    session = make_codex()
    codex_log(
        session, "api_request", time.time(), duration_ms=80, **{"http.response.status_code": 429}
    )
    [model_call] = actions(trajectory(session), "model")
    assert model_call["outcome"] == "failed"


def test_denied_approval_closes_the_tool_from_telemetry(make_codex):
    session = make_codex()
    hook(
        session,
        "PreToolUse",
        tool_name="Bash",
        tool_input={"command": "rm x"},
        tool_use_id="exec-9",
    )
    hook(session, "PermissionRequest", tool_name="Bash", tool_input={"command": "rm x"})
    assert trajectory(session).agents["codex"]["status"] == "waiting_input"
    codex_log(
        session, "tool_decision", time.time(), call_id="exec-9", decision="denied", source="User"
    )
    model = trajectory(session)
    [action] = actions(model)
    assert "Rejected (User)" in action["error"]
    assert not model.signals["input:codex"]["active"]


def test_interrupt_hook_closes_running_tools_and_idles_the_agent(make_codex):
    session = make_codex()
    hook(session, "UserPromptSubmit", prompt="sleep", turn_id="t1")
    hook(
        session,
        "PreToolUse",
        tool_name="Bash",
        tool_input={"command": "sleep 30"},
        tool_use_id="exec-5",
    )
    hook(session, "Interrupt", turn_id="t1")
    model = trajectory(session)
    [action] = actions(model)
    assert action["error"] == "Interrupted by the user"
    assert model.agents["codex"]["status"] == "agent_idle"


def test_warns_when_the_users_own_codex_config_exports_telemetry(tmp_path):
    (tmp_path / "config.toml").write_text(
        '[otel]\nexporter = { otlp-http = { endpoint = "https://collector", protocol = "binary" } }\n'
    )
    assert codex._configured_otel_exporter({"CODEX_HOME": str(tmp_path)}) == "otlp-http"
    (tmp_path / "config.toml").write_text('model = "x"\n')
    assert codex._configured_otel_exporter({"CODEX_HOME": str(tmp_path)}) is None


@linux_only
def test_fake_codex_session_end_to_end(make_codex):
    session = make_codex()
    client = TerminalClient(session)
    client.wait_for("fake codex ready")
    client.wait_for("hook exit 0 output b''")  # the real relay ran and said nothing
    client.type(b"t")
    client.wait_for("tool done")
    client.type(b"m")
    client.wait_for("model done")
    client.type(b"p")
    client.wait_for("approval asked")
    client.type(b"n")
    client.wait_for("denied")

    def recorded():
        model = trajectory(session)
        names = sorted((a["name"], a["outcome"]) for a in actions(model) if a["kind"] != "model")
        require(len(names) == 2 and actions(model, "model"))
        return model, names

    model, tools = eventually(recorded)
    assert model.agents["codex"]["harness"] == "codex"
    assert model.agents["codex"]["model"] == "fake-codex-model"
    [denied] = [a for a in actions(model) if a["command"] == "rm -rf build"]
    assert "Rejected (User)" in denied["error"]
    client.type(b"q")
    eventually(lambda: require(session.ended.is_set()))
    client.close()
