"""Inherited-terminal job control through a real interactive parent shell."""

from __future__ import annotations

import errno
import json
import os
import select
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _state(pid):
    return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]


@pytest.mark.parametrize(
    "trace,background_launch,agency_entry",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_shell_ctrl_z_fg_bg_and_background_read(tmp_path, trace, background_launch, agency_entry):
    from agency.harness.ptrace.supervisor import ptrace_available

    if trace and not ptrace_available():
        pytest.skip("ptrace unavailable")
    if agency_entry:
        from agency.sessions.session import _user_scope_available

        if not _user_scope_available():
            pytest.skip("systemd user scopes unavailable")
    cli = tmp_path / "cli.py"
    cli.write_text(
        f"#!{sys.executable}\n"
        + """
import os,signal,sys,termios,tty
from pathlib import Path
Path(os.environ['JOB_CONTROL_FIXTURE_ROOT']).joinpath('cli.pid').write_text(str(os.getpid()))
tty.setcbreak(0)
raw=termios.tcgetattr(0)
def continued(*_):
 print('JOB_CONT',os.tcgetpgrp(0)==os.getpgrp(),termios.tcgetattr(0)==raw,flush=True)
signal.signal(signal.SIGCONT,continued)
print('JOB_READY',flush=True)
while True:
 data=os.read(0,1)
 if data==b'q':break
 print('JOB_INPUT',data.hex(),flush=True)
print('CLI_FINISHED',flush=True)
sys.exit(7)
"""
    )
    cli.chmod(0o755)
    driver = tmp_path / "driver.py"
    driver.write_text("""
import os,sys
from pathlib import Path
from agency.configs.agconfig import agconfig
from agency.sessions.process import NativeProcess
root=Path(sys.argv[1]);root.joinpath('wrapper.pid').write_text(str(os.getpid()))
class Policy:
 def check(self,*_):return True
p=NativeProcess([sys.executable,str(root/'cli.py'),str(root)],dict(os.environ),str(root),config=agconfig(),policy=Policy(),trace=sys.argv[2]=='True')
try:assert p.wait(30)==7
finally:p.close()
print('JOB_FINISHED',flush=True)
""")
    master, slave = os.openpty()
    shell = subprocess.Popen(
        ["setsid", "-c", "bash", "--noprofile", "--norc", "-i"],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env={**os.environ, "PS1": "AGENCY_SHELL> ", "PROMPT_COMMAND": ""},
    )
    os.close(slave)
    received = bytearray()

    def until(marker, start=0, timeout=15):
        deadline = time.monotonic() + timeout
        while marker not in received[start:]:
            assert time.monotonic() < deadline, received.decode(errors="replace")
            if select.select([master], [], [], 0.05)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
                    chunk = b""
                assert chunk, received.decode(errors="replace")
                received.extend(chunk)
        return len(received)

    def command(value, marker=b"AGENCY_SHELL> "):
        start = len(received)
        os.write(master, value.encode() + b"\n")
        return until(marker, start)

    def stopped():
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            wrapper = tmp_path / "wrapper.pid"
            if wrapper.exists() and _state(int(wrapper.read_text())) == "T":
                return
            time.sleep(0.02)
        pytest.fail(received.decode(errors="replace"))

    try:
        until(b"AGENCY_SHELL> ")
        launch = shlex.join(
            [
                "env",
                f"PYTHONPATH={Path(__file__).parents[2]}",
                f"JOB_CONTROL_FIXTURE_ROOT={tmp_path}",
                sys.executable,
                str(driver),
                str(tmp_path),
                str(trace),
            ]
        )
        if agency_entry:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            launch = shlex.join(
                [
                    "env",
                    f"PYTHONPATH={Path(__file__).parents[2]}",
                    f"JOB_CONTROL_FIXTURE_ROOT={tmp_path}",
                    f"AGENCY_RUNS_ROOT={tmp_path / 'runs'}",
                    sys.executable,
                    "-m",
                    "agency.cli",
                    "codex",
                    "--codex",
                    str(cli),
                    "--cwd",
                    str(tmp_path),
                    "--port",
                    str(port),
                    "--trace",
                ]
            )
        if background_launch:
            command(launch + " &")
            stopped()  # SIGTTOU/SIGTTIN: never steal the shell's terminal.
            command("fg", b"JOB_READY")
        else:
            command(launch, b"JOB_READY")
        if agency_entry:
            info_path = next((tmp_path / "runs").rglob("interactive_session.json"))
            info = json.loads(info_path.read_text())
            assert info["trace"] and info["scope"]
            cli_pid = int((tmp_path / "cli.pid").read_text())
            assert info["cli_pid"] == cli_pid
            assert info["scope"] in Path(f"/proc/{cli_pid}/cgroup").read_text()
            assert info["scope"] not in Path(f"/proc/{info['runtime_pid']}/cgroup").read_text()
            (tmp_path / "wrapper.pid").write_text(str(info["runtime_pid"]))
        start = len(received)
        os.write(master, b"\x1a")
        until(b"AGENCY_SHELL> ", start)
        stopped()
        command("printf 'SHELL_USABLE\\n'")
        assert b"SHELL_USABLE" in received
        command("fg", b"JOB_CONT True True")
        start = len(received)
        os.write(master, b"x")
        until(b"JOB_INPUT 78", start)
        start = len(received)
        os.write(master, b"\x1a")
        until(b"AGENCY_SHELL> ", start)
        command("bg")
        stopped()  # Resumed background CLI reads stdin and gets SIGTTIN.
        command("printf 'SHELL_STILL_USABLE\\n'")
        command("fg", b"JOB_CONT True True")
        start = len(received)
        os.write(master, b"q")
        until(b"CLI_FINISHED", start)
        if not agency_entry:
            until(b"JOB_FINISHED", start)
        else:
            until(b"AGENCY_SHELL> ", start)
        command("exit 0", b"exit")
        assert shell.wait(timeout=5) == 0
    finally:
        for file in ("cli.pid", "wrapper.pid"):
            path = tmp_path / file
            if path.exists():
                try:
                    os.kill(int(path.read_text()), 9)
                except ProcessLookupError:
                    pass
        if shell.poll() is None:
            shell.kill()
            shell.wait()
        os.close(master)


@pytest.mark.parametrize("trace", [False, True])
def test_dashboard_pause_is_not_a_shell_job_stop(tmp_path, trace):
    from agency.configs.agconfig import agconfig
    from agency.harness.ptrace.supervisor import ptrace_available
    from agency.sessions.process import NativeProcess

    if trace and not ptrace_available():
        pytest.skip("ptrace unavailable")

    class Policy:
        def check(self, *_):
            return True

    import signal

    process = NativeProcess(
        [sys.executable, "-c", "import time;time.sleep(60)"],
        dict(os.environ),
        tmp_path,
        config=agconfig(),
        policy=Policy(),
        trace=trace,
    )
    try:
        for delay in (0, 0.05):
            process.signal_group(signal.SIGSTOP)
            time.sleep(delay)
            assert not process._job_stopped.is_set()
            process.signal_group(signal.SIGCONT)
            time.sleep(0.05)
            assert not process._job_stopped.is_set(), "late UI pause status became a shell stop"
        assert not process.exited.is_set()
    finally:
        process.close()
