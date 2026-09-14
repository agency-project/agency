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
from agency.harness.adapters.agharness_backend import AdapterRuntime
from agency.harness.adapters.claude_code import ClaudeDriver
from agency.harness.adapters.pty_session import PtyExecution
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
