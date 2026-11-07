"""The native CLI shares a terminal without an intermediate Agency PTY."""

import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agency.sessions.process import NativeProcess


@pytest.mark.parametrize("trace", [False, True])
def test_inherited_terminal_identity_input_and_foreground_are_preserved(tmp_path, trace):
    # The test supplies a controlling terminal, as a user's terminal emulator
    # would. Inside it, Agency must never call openpty or consume input.
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import json,os,sys\n"
        "from agency.configs.agconfig import agconfig\n"
        "from agency.sessions.process import NativeProcess\n"
        "from agency.harness.ptrace.supervisor import ptrace_available\n"
        "if sys.argv[1]=='1' and not ptrace_available():\n"
        " print('TRACE_UNAVAILABLE',flush=True);sys.exit(0)\n"
        "class Policy:\n"
        " def check(self,*args):return True\n"
        "terminal=os.ttyname(0);before=os.tcgetpgrp(0)\n"
        "def forbidden(*args):raise AssertionError('extra PTY allocated')\n"
        "os.openpty=forbidden\n"
        "child=\"import os,sys;print('CHILD',os.ttyname(0),os.isatty(1),os.isatty(2),os.tcgetpgrp(0)==os.getpgrp(),flush=True);print('INPUT',input(),flush=True);sys.exit(7)\"\n"
        "p=NativeProcess([sys.executable,'-c',child],dict(os.environ),os.getcwd(),config=agconfig(),policy=Policy(),trace=sys.argv[1]=='1')\n"
        "assert p.wait(15)==7\n"
        "print('RESTORED',terminal,os.tcgetpgrp(0)==before,flush=True)\n"
    )
    master, slave = os.openpty()
    process = subprocess.Popen(
        ["setsid", "-c", sys.executable, str(driver), str(int(trace))],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])},
    )
    expected_terminal = os.ttyname(slave)
    os.close(slave)
    received = bytearray()
    try:
        deadline = time.monotonic() + 20
        answered = False
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                received.extend(chunk)
            if b"TRACE_UNAVAILABLE" in received:
                pytest.skip("ptrace unavailable on this host")
            if b"CHILD" in received and not answered:
                os.write(master, b"native input\n")
                answered = True
            if b"RESTORED" in received:
                break
        assert process.wait(timeout=5) == 0, received.decode(errors="replace")
        text = received.decode(errors="replace")
        assert f"CHILD {expected_terminal} True True True" in text
        assert "INPUT native input" in text
        assert f"RESTORED {expected_terminal} True" in text
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)


def test_untraced_scope_setup_precedes_exec_and_path_uses_child_environment(tmp_path):
    marker = tmp_path / "executed"
    executable = tmp_path / "native-fixture"
    executable.write_text(f"#!/bin/sh\nprintf ready > {marker}\nexit 3\n")
    executable.chmod(0o755)
    seen = []

    def spawned(pid):
        assert not marker.exists(), "CLI executed before scope setup"
        assert Path(f"/proc/{pid}").exists()
        seen.append(pid)

    process = NativeProcess(
        [executable.name],
        {**os.environ, "PATH": str(tmp_path)},
        tmp_path,
        config=None,
        policy=None,
        trace=False,
        on_spawn=spawned,
    )
    assert process.wait(5) == 3
    assert seen == [process.pid]
    assert marker.read_text() == "ready"


def test_failed_scope_setup_reaps_stopped_bootstrap(tmp_path):
    seen = []

    def fail(pid):
        seen.append(pid)
        raise RuntimeError("scope fixture rejected")

    with pytest.raises(RuntimeError, match="scope fixture rejected"):
        NativeProcess(
            [sys.executable, "-c", "raise AssertionError('executed')"],
            dict(os.environ),
            tmp_path,
            config=None,
            policy=None,
            trace=False,
            on_spawn=fail,
        )
    assert not Path(f"/proc/{seen[0]}").exists()


def test_wait_timeout_and_close_do_not_leave_the_root_running(tmp_path):
    process = NativeProcess(
        [sys.executable, "-c", "import time;time.sleep(60)"],
        dict(os.environ),
        tmp_path,
        config=None,
        policy=None,
        trace=False,
    )
    assert process.wait(0.01) is None
    process.close()
    assert process.exited.is_set()
    assert process.returncode == -9
    assert not Path(f"/proc/{process.pid}").exists()


@pytest.mark.parametrize("trace", [False, True])
def test_root_exit_retires_background_children_without_changing_exit_status(tmp_path, trace):
    from agency.configs.agconfig import agconfig
    from agency.harness.ptrace.supervisor import ptrace_available

    if trace and not ptrace_available():
        pytest.skip("ptrace unavailable on this host")

    class Policy:
        def check(self, *_args):
            return True

    child_pid = tmp_path / "background.pid"
    spawned = []
    code = (
        "import subprocess,sys\n"
        "from pathlib import Path\n"
        f"p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session={trace})\n"
        f"Path({str(child_pid)!r}).write_text(str(p.pid))\n"
        "sys.exit(19)\n"
    )
    process = NativeProcess(
        [sys.executable, "-c", code],
        dict(os.environ),
        tmp_path,
        config=agconfig(),
        policy=Policy(),
        trace=trace,
        on_spawn=spawned.append,
    )
    try:
        assert process.wait(6) == 19
        assert spawned == [process.pid], "session scope callback ran for a descendant"
        pid = int(child_pid.read_text())
        status = Path(f"/proc/{pid}/stat")
        deadline = time.monotonic() + 2
        while status.exists():
            raw = status.read_text()
            if raw[raw.rfind(")") + 2] == "Z":
                break
            assert time.monotonic() < deadline, "background child still executing"
            time.sleep(0.01)
    finally:
        process.close()
