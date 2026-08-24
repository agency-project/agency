"""Tests for agutil utility helpers."""

import os
import signal
import threading
import time

import pytest

from agency.agutil import _strip_thinking, _extract_thinking, sigterm_as_exit


def test_strip_thinking_removes_think_tag():
    assert _strip_thinking("<think>reasoning</think>answer") == "answer"


def test_strip_thinking_removes_thinking_tag():
    assert _strip_thinking("<thinking>deep thought</thinking>result") == "result"


def test_strip_thinking_no_tag_unchanged():
    assert _strip_thinking("plain answer") == "plain answer"


def test_extract_thinking_returns_content():
    assert _extract_thinking("<think>my reasoning</think>answer") == "my reasoning"


def test_extract_thinking_no_tag_returns_empty():
    assert _extract_thinking("no thinking here") == ""


def test_extract_thinking_multiple_blocks():
    text = "<think>first</think>middle<think>second</think>end"
    result = _extract_thinking(text)
    assert "first" in result and "second" in result


class TestSigtermAsExit:
    def test_no_signal_received_event_stays_unset(self):
        with sigterm_as_exit() as received:
            pass
        assert not received.is_set()

    def test_restores_previous_handler_after_normal_exit(self):
        prev = signal.getsignal(signal.SIGTERM)
        with sigterm_as_exit():
            pass
        assert signal.getsignal(signal.SIGTERM) is prev

    def test_sigterm_raises_systemexit_and_sets_event(self):
        received_ref = {}
        with pytest.raises(SystemExit):
            with sigterm_as_exit() as received:
                received_ref["event"] = received
                os.kill(os.getpid(), signal.SIGTERM)
                # Should never reach here -- the handler raises SystemExit
                # synchronously as soon as the signal is delivered.
                time.sleep(5)
        assert received_ref["event"].is_set()

    def test_restores_previous_handler_even_after_sigterm(self):
        prev = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit):
            with sigterm_as_exit():
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)
        assert signal.getsignal(signal.SIGTERM) is prev

    def test_custom_label_used_in_message(self, capsys):
        with pytest.raises(SystemExit):
            with sigterm_as_exit("myapp"):
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)
        captured = capsys.readouterr()
        assert "[myapp] Received SIGTERM" in captured.out

    def test_noop_from_non_main_thread(self):
        """signal.signal() only works on the main thread -- a background
        thread must get a harmless no-op instead of a crash, with an Event
        that's simply never set."""
        results = {}

        def _worker():
            with sigterm_as_exit() as received:
                results["received"] = received
                results["ran"] = True

        t = threading.Thread(target=_worker)
        t.start()
        t.join(timeout=5)
        assert results.get("ran") is True
        assert not results["received"].is_set()


# ---------------------------------------------------------------------------
# Host-side UDS gateway: location, sun_path budget, and orphan reaping.
#
# These exist because a live socket was silently deleted out from under a
# running server by an external $TMPDIR cleaner, leaving every request across
# the bridge failing with a bare ENOENT. Each test below pins one link of that
# chain so it cannot be reintroduced.
# ---------------------------------------------------------------------------

import socket as _socket
import subprocess

from agency import agutil as _agutil
from agency.agutil import (
    UDS_SUN_PATH_MAX,
    agency_run_id,
    agharness_llm_gateway_dir,
    new_uds_path,
    pid_alive,
    uds_listener_is_live,
)

# Every prefix minting a socket in the gateway directory.
_UDS_PREFIXES = [
    "agllm_terminus",
    "agharness_messenger",
    "agmcp_server",
    "agproxy_llm",
    "agprof-ingest",
    "agproxy_ptrace",
    "native-entrypoint",
]


@pytest.fixture
def _fresh_gateway(monkeypatch):
    """Point the gateway root at a scratch dir and clear the per-process
    caches, so each test gets its own short, socket-safe root/run."""
    import shutil
    import tempfile
    from pathlib import Path

    # pytest's macOS tmp_path includes the test name and routinely exceeds
    # sockaddr_un.sun_path's 108-byte budget before the socket basename is
    # added.  Production deliberately uses /tmp/agency for the same reason.
    root = Path(tempfile.mkdtemp(prefix="agt-", dir="/tmp"))
    monkeypatch.setattr(_agutil, "agency_tmp_root", lambda: root)
    monkeypatch.setattr(_agutil, "_gateway_dir", None)
    monkeypatch.setattr(_agutil, "_gateway_reap_done", False)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _dead_pid() -> int:
    """A pid that has certainly exited (its child is reaped by wait())."""
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def test_gateway_dir_ignores_tmpdir(tmp_path, monkeypatch):
    """The gateway must NOT follow $TMPDIR. Sockets there are routinely
    deleted by scratch-space cleanup policies -- survivable for a temp file,
    fatal for a live socket."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(_agutil, "_gateway_dir", None)
    monkeypatch.setattr(_agutil, "_gateway_reap_done", True)

    gateway = agharness_llm_gateway_dir()

    assert str(tmp_path) not in str(gateway)
    assert str(gateway).startswith("/tmp/agency/gw/")


def test_gateway_dir_is_run_scoped_and_records_its_owner(_fresh_gateway):
    """Per-run scoping is what makes cleanup a single removal and stops one
    run's container from reaching another run's sockets (the directory is
    bind-mounted rw into every container)."""
    gateway = agharness_llm_gateway_dir()

    assert gateway.parent.name == "gw"
    assert gateway.name == agency_run_id()
    assert (gateway / "owner.pid").read_text().strip() == str(os.getpid())


def test_every_service_prefix_fits_the_sun_path_budget():
    """sockaddr_un.sun_path is 108 bytes -- a kernel ABI limit, unrelated to
    PATH_MAX -- so a legal file path can still be an unbindable socket path.

    Measured against the REAL production layout (pure string math, no
    directories created): a scratch root proves nothing, since the question is
    whether the *shipped* root leaves room. A relocation that eats the margin
    fails here, once, instead of at bind time in seven services.
    """
    run_dir = f"{_agutil.agency_tmp_root()}/gw/{agency_run_id()}"
    for prefix in _UDS_PREFIXES:
        path = f"{run_dir}/{prefix}-{'0' * 8}.sock"
        assert len(path) < UDS_SUN_PATH_MAX, f"{prefix}: {len(path)} bytes"
        # Real headroom, not a lucky fit: the previous $TMPDIR-based layout
        # already spent 94 of the 108 bytes.
        assert UDS_SUN_PATH_MAX - len(path) > 20, (
            f"{prefix} has only {UDS_SUN_PATH_MAX - len(path)}"
        )


def test_over_budget_socket_path_is_rejected_with_a_useful_error(tmp_path, monkeypatch):
    """The kernel's own failure is a bare `AF_UNIX path too long` raised
    several frames inside uvicorn, naming neither the path nor the limit."""
    deep = tmp_path / ("d" * 90)
    monkeypatch.setattr(_agutil, "agency_tmp_root", lambda: deep)
    monkeypatch.setattr(_agutil, "_gateway_dir", None)
    monkeypatch.setattr(_agutil, "_gateway_reap_done", True)

    with pytest.raises(RuntimeError, match=r"sun_path|108"):
        new_uds_path("agharness_messenger")


def test_uds_listener_is_live_rejects_a_deleted_socket(_fresh_gateway):
    path = new_uds_path("agllm_terminus")
    alive = threading.Thread(target=lambda: time.sleep(5), daemon=True)
    alive.start()

    assert not uds_listener_is_live(path, alive)  # file never created
    open(path, "w").close()
    assert uds_listener_is_live(path, alive)
    os.remove(path)  # exactly what an external reaper does
    assert not uds_listener_is_live(path, alive)


def test_uds_listener_is_live_rejects_a_dead_server_thread(_fresh_gateway):
    """The other half: uvicorn unlinks on shutdown, so a thread that died for
    any reason leaves the cached path pointing at nothing."""
    path = new_uds_path("agllm_terminus")
    open(path, "w").close()
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()

    assert not uds_listener_is_live(path, dead)
    assert not uds_listener_is_live(path, None)


def test_reap_removes_a_dead_runs_gateway_dir(_fresh_gateway):
    agharness_llm_gateway_dir()  # this run's dir, must survive
    orphan = _fresh_gateway / "gw" / "rdeadbeef"
    orphan.mkdir(parents=True)
    (orphan / "owner.pid").write_text(f"{_dead_pid()}\n")

    _agutil._gateway_reap_done = False
    _agutil._reap_orphaned_gateway_dirs()

    assert not orphan.exists()
    assert _agutil._gateway_dir.exists()


def test_reap_keeps_a_live_runs_gateway_dir(_fresh_gateway):
    agharness_llm_gateway_dir()
    live = _fresh_gateway / "gw" / "rliveproc"
    live.mkdir(parents=True)
    (live / "owner.pid").write_text(f"{os.getpid()}\n")

    _agutil._gateway_reap_done = False
    _agutil._reap_orphaned_gateway_dirs()

    assert live.exists()


def test_reap_keeps_a_dir_whose_socket_still_has_a_listener(_fresh_gateway):
    """An owner record can be stale or hand-copied. Deleting a directory whose
    sockets are still being served would recreate the exact failure this
    layout exists to prevent, so liveness on the socket itself vetoes."""
    agharness_llm_gateway_dir()
    stale_label = _fresh_gateway / "gw" / "rstalepid"
    stale_label.mkdir(parents=True)
    (stale_label / "owner.pid").write_text(f"{_dead_pid()}\n")
    server = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    server.bind(str(stale_label / "agllm_terminus-deadbeef.sock"))
    server.listen(1)
    try:
        _agutil._gateway_reap_done = False
        _agutil._reap_orphaned_gateway_dirs()
        assert stale_label.exists(), "reaped a directory that was still being served"
    finally:
        server.close()


def test_reap_leaves_dirs_with_no_owner_record(_fresh_gateway):
    """Never guess: a directory with no readable owner is left alone (a known,
    documented leak, exactly like container.py's unlabeled pre-existing
    images) rather than deleted on suspicion."""
    agharness_llm_gateway_dir()
    unknown = _fresh_gateway / "gw" / "rnoowner"
    unknown.mkdir(parents=True)

    _agutil._gateway_reap_done = False
    _agutil._reap_orphaned_gateway_dirs()

    assert unknown.exists()


def test_pid_alive_agrees_with_reality():
    assert pid_alive(os.getpid())
    assert not pid_alive(_dead_pid())
