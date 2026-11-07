"""PTY behaviour of agency/sessions/terminal.py, driven by small dummy programs."""

import os
import signal
import sys
import threading
import time

import pyte
import pytest

from agency.sessions.terminal import _Screen, render_screen

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="interactive sessions are Linux-only"
)


class Viewer:
    """What one browser sees: the attach snapshot, then the live stream."""

    def __init__(self, terminal, columns=80, rows=24):
        self.screen = _Screen(columns, rows)
        self.stream = pyte.ByteStream(self.screen)
        self.raw = bytearray()
        self.closed = threading.Event()
        self._lock = threading.Lock()
        snapshot = terminal.attach(self)
        self._feed(snapshot)

    def __call__(self, chunk):
        if chunk is None:
            self.closed.set()
        else:
            self._feed(chunk)

    def _feed(self, data):
        with self._lock:
            self.raw += data
            self.stream.feed(data)

    def text(self):
        with self._lock:
            return "\n".join(line.rstrip() for line in self.screen.display)

    def wait_for(self, needle, timeout=10):
        deadline = time.monotonic() + timeout
        while needle not in self.text():
            if time.monotonic() > deadline:
                raise AssertionError(f"{needle!r} never appeared:\n{self.text()}")
            time.sleep(0.02)


@pytest.fixture
def spawn(tmp_path):
    from agency.sessions.terminal import Terminal

    started = []

    def start(program, columns=80, rows=24):
        terminal = Terminal(
            [sys.executable, "-c", program],
            {**os.environ, "TERM": "xterm-256color"},
            str(tmp_path),
            columns,
            rows,
        )
        started.append(terminal)
        return terminal

    yield start
    for terminal in started:
        terminal.signal_group(signal.SIGKILL)
        terminal.wait(5)


def test_render_screen_round_trips_colours_wide_characters_and_cursor():
    original = _Screen(20, 4)
    pyte.ByteStream(original).feed(
        "plain \x1b[1;31mred\x1b[0m 你好\r\n"
        "\x1b[38;5;45m256\x1b[48;2;1;2;3mtrue\x1b[0m\r\n"
        "\x1b[7mreverse\x1b[0m\x1b[3;5H".encode()
    )
    redrawn = _Screen(20, 4)
    pyte.ByteStream(redrawn).feed(render_screen(original))
    assert redrawn.display == original.display
    for y in range(4):
        for x in range(20):
            assert redrawn.buffer[y][x] == original.buffer[y][x], (x, y)
    assert (redrawn.cursor.x, redrawn.cursor.y) == (original.cursor.x, original.cursor.y)


def test_render_screen_restores_input_modes_the_application_enabled():
    screen = _Screen(20, 4)
    pyte.ByteStream(screen).feed(b"\x1b[?2004h\x1b[?1h\x1b[?1000;1006h\x1b[?25l")
    rendered = render_screen(screen)
    for mode in (b"\x1b[?2004h", b"\x1b[?1h", b"\x1b[?1000h", b"\x1b[?1006h"):
        assert mode in rendered
    assert rendered.endswith(b"\x1b[?25l")


@linux_only
def test_child_is_session_leader_with_the_pty_as_controlling_terminal(spawn):
    terminal = spawn(
        "import os,sys\n"
        "print('leader', os.getsid(0) == os.getpid(), os.tcgetpgrp(0) == os.getpgrp(), flush=True)\n"
        "print('got', sys.stdin.readline().strip(), flush=True)\n"
    )
    viewer = Viewer(terminal)
    viewer.wait_for("leader True True")
    assert terminal.write(b"hello\r")
    viewer.wait_for("got hello")
    terminal.wait(5)
    assert terminal.returncode == 0
    assert viewer.closed.wait(5)


@linux_only
def test_resize_reaches_the_process_as_sigwinch(spawn):
    terminal = spawn(
        "import os,signal\n"
        "def report(*_):\n"
        "    size = os.get_terminal_size(0)\n"
        "    print(f'size {size.columns}x{size.lines}', flush=True)\n"
        "signal.signal(signal.SIGWINCH, report)\n"
        "print('ready', flush=True)\n"
        "while True: signal.pause()\n"
    )
    viewer = Viewer(terminal)
    viewer.wait_for("ready")
    terminal.resize(100, 30)
    viewer.wait_for("size 100x30")
    assert terminal.size == (100, 30)


@linux_only
def test_reconnect_snapshot_matches_what_a_connected_viewer_saw(spawn):
    # Clears and redraws after scrolling far past one screen: replaying a
    # bounded raw buffer could not reproduce this, the screen model can.
    terminal = spawn(
        "import sys,time\n"
        "for i in range(500): print(f'line {i}')\n"
        "sys.stdout.write('\\x1b[2J\\x1b[H\\x1b[?2004h\\x1b[1;33mstatus\\x1b[0m\\n'"
        " '\\x1b[44m box \\x1b[0m wide 你好\\n\\x1b[5;7Hcursor here')\n"
        "sys.stdout.write('\\x1b[5;3H'); sys.stdout.flush()\n"
        "time.sleep(60)\n"
    )
    first = Viewer(terminal)
    first.wait_for("cursor here")
    time.sleep(0.2)
    terminal.detach(first)
    second = Viewer(terminal)
    assert second.screen.display == first.screen.display
    assert (second.screen.cursor.x, second.screen.cursor.y) == (2, 4)
    assert second.screen.buffer[0][0].fg == "brown"  # SGR 33
    assert second.screen.buffer[1][1].bg == "blue"
    assert (2004 << 5) in second.screen.mode
    assert "line 499" not in second.text()


@linux_only
def test_cursor_position_query_is_answered_while_no_viewer_is_attached(spawn):
    terminal = spawn(
        "import os,sys,tty\n"
        "tty.setraw(0)\n"
        "os.write(1, b'\\x1b[3;5H\\x1b[6n')\n"
        "reply = b''\n"
        "while not reply.endswith(b'R'): reply += os.read(0, 1)\n"
        "os.write(1, b'\\r\\nreply ' + reply[1:])\n"
        "import time; time.sleep(60)\n"
    )
    # Read the model directly: attaching a viewer would itself stop Agency
    # from answering on the viewer's behalf.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with terminal._lock:
            text = "\n".join(terminal._screen.display)
        if "reply [3;5R" in text:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("cursor position query was never answered")


@linux_only
def test_input_to_a_stopped_process_never_blocks_the_writer(spawn):
    terminal = spawn("import time\nprint('ready', flush=True)\ntime.sleep(60)\n")
    viewer = Viewer(terminal)
    viewer.wait_for("ready")
    terminal.signal_group(signal.SIGSTOP)
    started = time.monotonic()
    delivered = [terminal.write(b"x" * 4096) for _ in range(64)]
    assert time.monotonic() - started < 2
    assert not all(delivered)
    terminal.signal_group(signal.SIGCONT)


@linux_only
def test_exit_status_reaches_listeners(spawn):
    terminal = spawn("import time,sys\ntime.sleep(0.3)\nsys.exit(3)\n")
    viewer = Viewer(terminal)
    assert viewer.closed.wait(10)
    assert terminal.returncode == 3
    assert not terminal.write(b"late")
