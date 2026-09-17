"""Claude's actual interrupt/paste/ack path on a scripted PTY, no model required."""

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agency.configs.agconfig import agconfig
from agency.harness.adapters.base import AdapterRuntime
from agency.harness.adapters.claude_code import ClaudeDriver
from agency.harness.adapters.pty.execution import PtyExecution
from agency.harness.ptrace.supervisor import ptrace_available

pytestmark = pytest.mark.skipif(not ptrace_available(), reason="Linux ptrace required")


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "mode",
    [
        "running",
        "restored",
        "exit",
        "unacknowledged",
        "finished",
        "finished_empty",
        "completion_race",
        "paused",
    ],
)
def test_claude_native_input_is_acknowledged_or_execution_is_retired(tmp_path, mode, monkeypatch):
    (tmp_path / "events").mkdir()
    (tmp_path / "agency-turn.json").write_text(json.dumps({"turn_id": None}))
    env = {**os.environ, "AGENCY_CLAUDE_STATE": str(tmp_path), "TEST_PTY_MODE": mode}
    script = Path(__file__).parents[1] / "fixtures" / "claude_pty_cli.py"
    adapter = SimpleNamespace(
        agconfig=agconfig(),
        prepare_pty=lambda *args, **kwargs: ([sys.executable, str(script)], env),
    )
    paused = threading.Event()

    def register_handle(handle):
        if mode == "paused":
            handle.pause()
            paused.set()

    runtime = AdapterRuntime(
        agconfig(),
        "model",
        "engine",
        "",
        "",
        SimpleNamespace(check=lambda *args: True),
        register_control_handle=register_handle,
    )
    driver = ClaudeDriver(adapter, runtime, tmp_path, None, None, None)
    execution = PtyExecution(driver, runtime)
    execution.INPUT_TIMEOUT = 1
    if mode == "paused":
        execution.START_TIMEOUT = 0.1
    # Keep the fixture's byte log after the real process cleanup for assertions.
    monkeypatch.setattr("agency.harness.agharness.cleanup_config_home", lambda path: None)
    results, errors = [], []

    def run():
        try:
            results.append(execution.run("original prompt"))
        except RuntimeError as exc:
            errors.append(str(exc))

    worker = threading.Thread(target=run)
    worker.start()
    try:
        if mode == "paused":
            assert paused.wait(3)
            assert execution.redirect("before startup") is False
            # Exceed the normal startup deadline while the harness is frozen.
            assert not threading.Event().wait(0.25)
            assert worker.is_alive()
            execution.handle.resume()
        deadline = time.monotonic() + 5
        while not execution._active and worker.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        message = "/literal text\nwith unicode: café\tand tabs"
        delivered = execution.redirect(message)
        worker.join(5)
        assert not worker.is_alive()
        data = (tmp_path / "input.bin").read_bytes()
        expected_input = b"\x1b[200~[Agency run]\noriginal prompt\x1b[201~\r"
        if mode not in {"finished", "finished_empty"}:
            expected_input += b"\x1b"  # Escape interrupts Claude's active turn.
        if mode == "restored":
            expected_input += b"\x1b\x1b"
        if mode not in {"finished", "finished_empty", "completion_race", "exit"}:
            expected_input += b"\x1b[200~[Agency redirect]\n" + message.encode() + b"\x1b[201~\r"
        assert data == expected_input
        assert b"\x03" not in data
        if mode in {"finished", "finished_empty", "completion_race"}:
            assert delivered is False
            assert len(results) == 1
            assert not errors
            if mode == "finished_empty":
                assert results[0].final_text == ""
        elif mode in {"exit", "unacknowledged"}:
            assert delivered is False
            assert errors
            assert execution.handle.returncode is not None
        else:
            assert delivered is True
            assert not errors
            assert results[0].final_text == "scripted final"
        assert execution.redirect("after completion") is False
    finally:
        if execution.handle is not None:
            execution.handle.kill()
        worker.join(5)


@pytest.mark.parametrize(
    "context_limit,expected_window",
    [
        (32_000, None),  # 28.8k after margin -- still below the flag's 100k floor
        (100_000, None),  # 90k after margin -- just below the floor
        (500_000, 450_000),
        (1_000_000, 900_000),
        (2_000_000, None),  # 1.8M after margin -- still above the flag's 1M ceiling
    ],
)
def test_autocompact_uses_a_safety_margin_and_claudes_accepted_range(
    tmp_path, monkeypatch, context_limit, expected_window
):
    (tmp_path / "events").mkdir()
    monkeypatch.setattr(
        "agency.harness.adapters.claude_code.fetch_context_limit", lambda *a, **k: context_limit
    )
    adapter = SimpleNamespace(
        agconfig=agconfig(),
        prepare_pty=lambda *args, **kwargs: (["claude"], {}),
    )
    runtime = AdapterRuntime(
        agconfig(),
        "model",
        "engine",
        "http://127.0.0.1:8766",
        "token",
        SimpleNamespace(check=lambda *args: True),
    )
    driver = ClaudeDriver(adapter, runtime, tmp_path, None, None, None)
    if expected_window is None:
        assert "--autocompact" not in driver.argv
    else:
        assert driver.argv[driver.argv.index("--autocompact") + 1] == str(expected_window)


def test_no_autocompact_flag_when_context_limit_lookup_fails(tmp_path, monkeypatch):
    (tmp_path / "events").mkdir()
    monkeypatch.setattr(
        "agency.harness.adapters.claude_code.fetch_context_limit", lambda *a, **k: None
    )
    adapter = SimpleNamespace(
        agconfig=agconfig(),
        prepare_pty=lambda *args, **kwargs: (["claude"], {}),
    )
    runtime = AdapterRuntime(
        agconfig(),
        "model",
        "engine",
        "http://127.0.0.1:8766",
        "token",
        SimpleNamespace(check=lambda *args: True),
    )
    driver = ClaudeDriver(adapter, runtime, tmp_path, None, None, None)
    assert "--autocompact" not in driver.argv
