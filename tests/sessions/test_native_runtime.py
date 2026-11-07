"""Native-session exit status and real workload-scope control contracts."""

import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agency.sessions import session as module


def test_resume_releases_signal_fallback_when_cgroup_writes_recover(make_session, monkeypatch):
    session = make_session()
    delivered = []
    session.terminal = SimpleNamespace(exited=threading.Event(), signal_group=delivered.append)
    monkeypatch.setattr(session, "_write_scope", lambda name, value: value == "0")
    try:
        session.pause()
        session.resume()
        assert delivered == [signal.SIGSTOP, signal.SIGCONT]
        assert not session.is_paused()
    finally:
        session.terminal = None


def test_failed_cgroup_thaw_does_not_claim_the_workload_resumed(make_session, monkeypatch):
    session = make_session()
    delivered = []
    session.terminal = SimpleNamespace(exited=threading.Event(), signal_group=delivered.append)
    monkeypatch.setattr(session, "_write_scope", lambda name, value: value == "1")
    try:
        session.pause()
        with pytest.raises(RuntimeError, match="unfreeze"):
            session.resume()
        assert session.is_paused()
        assert not delivered
    finally:
        session.terminal = None


def test_wait_failure_finishes_the_session_and_retires_background_work(make_session):
    session = make_session()
    delivered = []

    def wait():
        raise RuntimeError("fixture child drain failed")

    session.terminal = SimpleNamespace(
        exited=threading.Event(),
        signal_group=delivered.append,
        wait=wait,
        returncode=3,
        pid=0,
    )
    try:
        session._watch_process()
        assert session.ended.is_set()
        assert isinstance(session._process_error, RuntimeError)
        assert delivered == [signal.SIGKILL]
    finally:
        session.terminal = None


def test_supervision_failure_is_a_failed_dashboard_workload(make_session, monkeypatch):
    session = make_session()
    session.terminal_mode = "native"
    session._process_error = RuntimeError("fixture supervisor failed")
    session.ended.set()
    monkeypatch.setattr(session, "start", lambda: None)
    monkeypatch.setattr(session, "launch_native", lambda: None)
    with pytest.raises(RuntimeError, match="CLI supervision failed") as exc:
        session.run()
    assert exc.value.__cause__ is session._process_error


@pytest.mark.parametrize("partially_started", [False, True])
def test_start_failure_closes_partially_initialized_runtime(
    tmp_path, monkeypatch, partially_started
):
    from agency.configs.agconfig import agconfig, dataloggerconfig
    from agency.observability.agdatalogger import agDataLogger

    logger = agDataLogger(agconfig(dataloggerconfig(db_path=str(tmp_path / "global.sqlite3"))))
    logger.start()
    session = module.CliSession(
        cwd=tmp_path,
        binary=sys.executable,
        cli_args=[],
        run_dir=tmp_path / "logs",
        ui_port=0,
        use_scope=False,
        trace=False,
    )
    session.agent_name = "partial_start"

    def fail():
        raise RuntimeError("fixture startup failed")

    start = session.start
    monkeypatch.setattr(session, "_prepare", fail)
    monkeypatch.setattr(
        session, "start", (lambda: start(global_logger=logger)) if partially_started else fail
    )
    try:
        with pytest.raises(RuntimeError, match="fixture startup failed"):
            session.run()
        assert session.ended.is_set()
        if partially_started:
            assert not session._server_thread.is_alive()
            assert session._agent_logger._conn is None
    finally:
        session.close()
        logger.stop()


@pytest.mark.parametrize("code,expected", [(0, 0), (7, 7), (-15, 143)])
def test_entry_preserves_native_exit_status(monkeypatch, tmp_path, code, expected):
    from agency.observability.agwebui import agwebui

    class Session:
        label = "Fixture"

        def __init__(self, **kwargs):
            assert kwargs["terminal_mode"] == "native"
            self.terminal = SimpleNamespace(returncode=code)

        def run(self):
            pass

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(module, "_user_scope_available", lambda: False)
    monkeypatch.setattr(agwebui, "run", lambda callback, **kwargs: callback())
    assert (
        module.main(Session, "codex", ["--codex", sys.executable, "--cwd", str(tmp_path)])
        == expected
    )


def test_entry_reports_launch_failure_even_if_webui_records_and_swallows_it(monkeypatch, tmp_path):
    from agency.observability.agwebui import agwebui

    class Session:
        label = "Fixture"
        terminal = None

        def __init__(self, **kwargs):
            pass

        def run(self):
            raise RuntimeError("fixture launch failed")

    def webui(callback, **kwargs):
        try:
            callback()
        except RuntimeError:
            pass  # mirrors agwebui.run's workload-failure recording path

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(module, "_user_scope_available", lambda: False)
    monkeypatch.setattr(agwebui, "run", webui)
    assert module.main(Session, "codex", ["--codex", sys.executable, "--cwd", str(tmp_path)]) == 1


def test_native_cleanup_ignores_late_ctrl_c_and_restores_the_previous_handler(
    monkeypatch, tmp_path
):
    from agency.observability.agwebui import agwebui

    received = []
    cleaned = []
    original = signal.getsignal(signal.SIGINT)

    def previous(signum, frame):
        received.append(signum)

    class Session:
        label = "Fixture"

        def __init__(self, **kwargs):
            self.terminal = SimpleNamespace(returncode=None)

        def run(self):
            os.kill(os.getpid(), signal.SIGINT)  # Native child still live: preserve handler.
            self.terminal.returncode = 0

    def webui(callback, **kwargs):
        callback()
        # Two late terminal interrupts during held post-child finalization.
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGINT)
        cleaned.append(True)

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(module, "_user_scope_available", lambda: False)
    monkeypatch.setattr(agwebui, "run", webui)
    signal.signal(signal.SIGINT, previous)
    try:
        assert (
            module.main(Session, "codex", ["--codex", sys.executable, "--cwd", str(tmp_path)]) == 0
        )
        assert received == [signal.SIGINT]
        assert cleaned == [True]
        assert signal.getsignal(signal.SIGINT) is previous
    finally:
        signal.signal(signal.SIGINT, original)


@pytest.mark.skipif(not module._user_scope_available(), reason="systemd user scopes unavailable")
@pytest.mark.parametrize("trace", [False, True])
def test_real_native_scope_is_separate_and_freeze_resume_stop_controls_children(tmp_path, trace):
    from agency.harness.ptrace.supervisor import ptrace_available

    if trace and not ptrace_available():
        pytest.skip("ptrace unavailable")
    child = tmp_path / "child.py"
    child.write_text("""
import os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);root.joinpath('cli.pid').write_text(str(os.getpid()))
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
root.joinpath('detached.pid').write_text(str(p.pid))
print('NATIVE_SCOPE_READY',flush=True)
i=0
while True:
 i+=1;root.joinpath('heartbeat').write_text(str(i));time.sleep(.02)
""")
    controller = tmp_path / "controller.py"
    controller.write_text("""
import json,os,sys,time
from pathlib import Path
from agency.configs.agconfig import agconfig,dataloggerconfig
from agency.observability.agdatalogger import agDataLogger
from agency.sessions.session import CliSession
root=Path(sys.argv[1]);trace=sys.argv[2]=='True'
class Session(CliSession):
 agent_name='native_fixture';harness='codex';label='Native fixture'
 def _cli_argv(self):return [sys.executable,str(root/'child.py'),str(root)]
 def _cli_environment(self,env):pass
 def handle_log_records(self,body):pass
logger=agDataLogger(agconfig(dataloggerconfig(db_path=str(root/'global.sqlite3'))));logger.start()
s=Session(cwd=root,binary=sys.executable,cli_args=[],run_dir=root/'logs',ui_port=0,use_scope=True,trace=trace)
def eventually(check):
 deadline=time.monotonic()+10
 while time.monotonic()<deadline:
  if check():return
  time.sleep(.02)
 raise AssertionError('native scope did not reach expected state')
try:
 s.start(global_logger=logger);s.launch_native()
 eventually(lambda:(root/'heartbeat').exists())
 assert s.scope_dir is not None
 assert str(s.scope_dir).removeprefix('/sys/fs/cgroup') not in Path('/proc/self/cgroup').read_text()
 for f in ['cli.pid','detached.pid']:
  pid=int((root/f).read_text());assert s.scope_unit in Path(f'/proc/{pid}/cgroup').read_text()
 s.pause();eventually(lambda:'frozen 1' in (s.scope_dir/'cgroup.events').read_text())
 before=(root/'heartbeat').read_text();time.sleep(.15);assert (root/'heartbeat').read_text()==before
 s.resume();eventually(lambda:(root/'heartbeat').read_text()!=before)
 s.pause();s.stop(force=True);assert s.ended.wait(10)
 for f in ['cli.pid','detached.pid']:
  pid=int((root/f).read_text());status=Path(f'/proc/{pid}/stat')
  if status.exists():assert status.read_text().split(') ',1)[1][0]=='Z'
 root.joinpath('result.json').write_text(json.dumps({'trace':trace,'scope':s.scope_unit,'outside_scope':True,'freeze_resume_stop':True}))
 print('NATIVE_CONTROL_FINISHED',flush=True)
finally:s.close();logger.stop()
""")
    master, slave = os.openpty()
    process = subprocess.Popen(
        ["setsid", "-c", sys.executable, str(controller), str(tmp_path), str(trace)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])},
    )
    os.close(slave)
    received = bytearray()
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                received.extend(chunk)
            if b"NATIVE_CONTROL_FINISHED" in received:
                break
        assert process.wait(timeout=5) == 0, received.decode(errors="replace")
        assert b"NATIVE_SCOPE_READY" in received
        assert json.loads((tmp_path / "result.json").read_text())["freeze_resume_stop"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
