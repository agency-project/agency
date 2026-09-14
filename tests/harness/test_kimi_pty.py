"""Kimi Code dialect: payload shapes captured from Kimi Code CLI 0.42.0."""

import json
import tomllib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agency.configs.agconfig import agconfig
from agency.harness.adapters.agharness_backend import AdapterRuntime, agharness_backend
from agency.harness.adapters.pty_drivers import driver_for
from agency.harness.adapters.pty_session import session_file_allowed
from pathlib import PurePosixPath

SESSION = "session_ef88f4ce-871e-48e3-b200-61666518f9fd"


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


@pytest.fixture
def driver(runtime, tmp_path):
    return driver_for(
        agharness_backend.for_config("kimi", runtime.agconfig), runtime, tmp_path, None, None, 4
    )


def test_launch_is_interactive_and_isolated(driver, runtime, tmp_path):
    assert driver.argv[0] == "kimi"
    assert "--auto" in driver.argv
    assert not set(driver.argv) & {"--prompt", "-p", "--output-format", "acp", "web"}
    # KIMI_CODE_HOME is the single switch relocating config, sessions and
    # credentials, so nothing lands in the real home directory.
    assert driver.env["KIMI_CODE_HOME"] == str(tmp_path)

    config = tomllib.loads((tmp_path / "config.toml").read_text())
    assert config["default_model"] == "agency-proxy"
    assert config["default_permission_mode"] == "auto"
    assert config["telemetry"] is False
    assert config["providers"]["agency-proxy"]["type"] == "openai"
    assert config["providers"]["agency-proxy"]["base_url"] == runtime.harness_base_url + "/v1"
    assert config["providers"]["agency-proxy"]["api_key"] == runtime.token
    assert config["loop_control"]["max_steps_per_turn"] == 4
    events = {hook["event"] for hook in config["hooks"]}
    assert {"SessionStart", "TurnStarted", "Stop", "Interrupt", "PreToolUse"} <= events


def test_resume_passes_the_native_session_flag(runtime, tmp_path):
    blob = json.dumps(
        {
            "version": 1,
            "harness": "kimi",
            "session_id": SESSION,
            "files": {"session_index.jsonl": ""},
        }
    ).encode()
    import base64

    payload = json.loads(blob)
    payload["files"]["session_index.jsonl"] = base64.b64encode(b"{}\n").decode()
    driver = driver_for(
        agharness_backend.for_config("kimi", runtime.agconfig),
        runtime,
        tmp_path,
        SESSION,
        json.dumps(payload).encode(),
        None,
    )
    assert driver.argv[-2:] == ["--session", SESSION]
    assert (tmp_path / "session_index.jsonl").read_bytes() == b"{}\n"


@pytest.mark.parametrize(
    "path,allowed",
    [
        ("session_index.jsonl", True),
        ("sessions/wd_work_4bcfda/session_x/agents/main/wire.jsonl", True),
        ("sessions/wd_work_4bcfda/session_x/state.json", True),
        ("config.toml", False),
        ("sessions/wd_work_4bcfda/session_x/logs/kimi-code.log", False),
        ("../escape.jsonl", False),
    ],
)
def test_only_session_state_is_portable(path, allowed):
    assert session_file_allowed("kimi", PurePosixPath(path)) is allowed


def test_trust_dialog_is_acknowledged_before_the_composer(driver):
    trust = [
        " Trust this folder?",
        "  ❯ Trust this folder",
        "    Don't trust",
    ]
    handle = SimpleNamespace(terminal_screen=lambda: (trust, 0, 0, 1), write_terminal=Mock())
    assert driver.ready(handle) is False
    handle.write_terminal.assert_called_once_with(b"\r")


@pytest.mark.parametrize(
    "line,expected",
    [
        (" │ >                        │ ", True),
        (" │ > already typing         │ ", False),
        (" ╭──────────────────────────╮ ", False),
        ("   No session yet — one will be created on your first message.", False),
    ],
)
def test_ready_requires_the_empty_composer(driver, line, expected):
    handle = SimpleNamespace(terminal_screen=lambda: ([line], 4, 0, 1), write_terminal=Mock())
    assert driver.ready(handle) is expected


def _hook(driver, **payload):
    (driver.root / "events" / f"{len(list((driver.root / 'events').glob('*')))}.json").write_text(
        json.dumps({"session_id": SESSION, "client_type": "kimi_code_cli", **payload})
    )


def test_turn_started_carries_the_identity_and_prompt(driver):
    _hook(driver, hook_event_name="SessionStart", source="startup")
    assert driver.events() == []
    assert driver.started is True
    assert driver.session_id == SESSION

    # UserPromptSubmit reports no turn at all, so it cannot acknowledge one.
    _hook(
        driver,
        hook_event_name="UserPromptSubmit",
        prompt=[{"type": "text", "text": "[Agency run x]\nhi"}],
    )
    _hook(
        driver,
        hook_event_name="TurnStarted",
        turn_id=0,
        origin_kind="user",
        prompt="[Agency run x]\nhi",
    )
    assert driver.events() == [{"kind": "submit", "turn_id": "0", "prompt": "[Agency run x]\nhi"}]


def test_turn_zero_survives_as_a_truthy_identity(driver):
    """Kimi numbers turns from 0; an integer 0 would never be adopted."""
    _hook(driver, hook_event_name="SessionStart", source="startup")
    driver.events()
    _hook(driver, hook_event_name="TurnStarted", turn_id=0, prompt="p")
    event = driver.events()[0]
    assert event["turn_id"] == "0"
    assert event["turn_id"]


def test_stop_is_stamped_with_the_turn_it_cannot_name(driver):
    _hook(driver, hook_event_name="SessionStart", source="startup")
    driver.events()
    _hook(driver, hook_event_name="TurnStarted", turn_id=7, prompt="p")
    driver.events()
    _hook(driver, hook_event_name="Stop", stop_hook_active=False)
    assert driver.events() == [{"kind": "stop", "turn_id": "7", "text": ""}]


def test_interrupt_and_failure_map_to_their_turn(driver):
    _hook(driver, hook_event_name="SessionStart", source="startup")
    driver.events()
    _hook(driver, hook_event_name="Interrupt", turn_id=3, reason="cancelled")
    _hook(driver, hook_event_name="StopFailure", turn_id=3, reason="model_error")
    assert driver.events() == [
        {"kind": "interrupt", "turn_id": "3"},
        {"kind": "error", "turn_id": "3", "error": "Kimi StopFailure: model_error"},
    ]


def test_other_sessions_are_ignored(driver):
    _hook(driver, hook_event_name="SessionStart", source="startup")
    driver.events()
    (driver.root / "events" / "foreign.json").write_text(
        json.dumps({"hook_event_name": "Stop", "session_id": "session_other"})
    )
    assert driver.events() == []


def _transcript(driver, rows):
    session_dir = driver.root / "sessions" / "wd_work_abc" / SESSION
    (session_dir / "agents" / "main").mkdir(parents=True, exist_ok=True)
    (driver.root / "session_index.jsonl").write_text(
        json.dumps({"sessionId": SESSION, "sessionDir": str(session_dir), "workDir": "/workspace"})
        + "\n"
    )
    (session_dir / "agents" / "main" / "wire.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )


def _loop(kind, turn, **event):
    return {"type": "context.append_loop_event", "event": {"type": kind, "turnId": turn, **event}}


def test_stop_is_not_completion_until_the_transcript_agrees(driver):
    driver.session_id = SESSION
    _transcript(
        driver,
        [_loop("content.part", "0", part={"type": "text", "text": "partial"})],
    )
    # The turn is still running: a Stop alone must not finish the attempt.
    assert driver.completed({"turn_id": "0", "text": ""}) is False


def test_completion_reads_answer_and_usage_from_the_transcript(driver):
    driver.session_id = SESSION
    _transcript(
        driver,
        [
            _loop("content.part", "0", part={"type": "text", "text": "hello "}),
            _loop("content.part", "0", part={"type": "text", "text": "world"}),
            _loop("content.part", "9", part={"type": "text", "text": "other turn"}),
            _loop(
                "step.end",
                "0",
                usage={
                    "inputOther": 10,
                    "inputCacheRead": 5,
                    "inputCacheCreation": 2,
                    "output": 7,
                },
            ),
            {"type": "turn.ended", "turnId": 0, "reason": "completed"},
        ],
    )
    event = {"turn_id": "0", "text": ""}
    assert driver.completed(event) is True
    assert event["text"] == "hello world"
    assert event["input_tokens"] == 17
    assert event["output_tokens"] == 7


def test_an_aborted_turn_is_a_failure_not_an_answer(driver):
    driver.session_id = SESSION
    _transcript(
        driver,
        [
            _loop("content.part", "0", part={"type": "text", "text": "half"}),
            {"type": "turn.ended", "turnId": 0, "reason": "cancelled"},
        ],
    )
    with pytest.raises(RuntimeError, match="cancelled"):
        driver.completed({"turn_id": "0", "text": ""})
