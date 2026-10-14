"""One interactive PTY whose screen survives browser reconnects.

The child process owns a real terminal (it is the session leader and the PTY
is its controlling terminal), so job control, SIGWINCH on resize and raw-mode
input behave exactly as in a desktop terminal emulator. Agency keeps a pyte
model of the screen alongside the raw stream: replaying raw output after a
reconnect is not enough, because a bounded buffer loses the escape sequences
that cleared the screen or switched terminal modes long ago.

Linux only: the controlling terminal is assigned by util-linux `setsid -c`.
"""

from __future__ import annotations

import fcntl
import os
import select
import struct
import subprocess
import termios
import threading

import pyte
import pyte.graphics

# Private modes the application may have switched on that a fresh browser
# terminal must also be in for keys, paste and mouse input to keep working.
# DECTCEM (25) and DECAWM (7) are restored separately: both default to on.
_REPLAYED_PRIVATE_MODES = (1, 66, 1000, 1002, 1003, 1004, 1005, 1006, 1015, 2004)
_ALTERNATE_SCREEN_MODES = (47, 1047, 1049)

_SGR_FOREGROUND = {name: code for code, name in pyte.graphics.FG_ANSI.items()}
_SGR_FOREGROUND.update({name: code for code, name in pyte.graphics.FG_AIXTERM.items()})
_SGR_BACKGROUND = {name: code for code, name in pyte.graphics.BG_ANSI.items()}
_SGR_BACKGROUND.update({name: code for code, name in pyte.graphics.BG_AIXTERM.items()})

_CURSOR_POSITION_QUERY = b"\x1b[6n"


class _Screen(pyte.Screen):
    def report_device_status(self, mode, private=False):
        # Status queries are answered by whichever terminal is attached (or
        # by Terminal itself when none is); the model only tracks content.
        return

    def write_process_input(self, data):
        return


def _colour(value, named_codes, extended_prefix):
    if value == "default":
        return None
    if value in named_codes:
        return str(named_codes[value])
    try:
        red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    except (ValueError, TypeError):
        return None
    return f"{extended_prefix};2;{red};{green};{blue}"


def _sgr(char) -> str:
    codes = ["0"]
    if char.bold:
        codes.append("1")
    if char.italics:
        codes.append("3")
    if char.underscore:
        codes.append("4")
    if char.blink:
        codes.append("5")
    if char.reverse:
        codes.append("7")
    if char.strikethrough:
        codes.append("9")
    foreground = _colour(char.fg, _SGR_FOREGROUND, "38")
    background = _colour(char.bg, _SGR_BACKGROUND, "48")
    if foreground:
        codes.append(foreground)
    if background:
        codes.append(background)
    return "\x1b[" + ";".join(codes) + "m"


def render_screen(screen: pyte.Screen) -> bytes:
    """Byte sequence that redraws *screen* on a freshly reset terminal."""
    private = {mode >> 5 for mode in screen.mode if mode & 31 == 0 and mode >> 5}
    out = ["\x1bc"]
    if private & set(_ALTERNATE_SCREEN_MODES):
        out.append("\x1b[?1049h")
    for mode in _REPLAYED_PRIVATE_MODES:
        if mode in private:
            out.append(f"\x1b[?{mode}h")
    # Draw with autowrap off so a full last column cannot scroll the screen.
    out.append("\x1b[?7l\x1b[H\x1b[2J")
    for y in range(screen.lines):
        row = screen.buffer[y]
        out.append(f"\x1b[{y + 1};1H")
        current = None
        for x in range(screen.columns):
            char = row[x]
            if char.data == "":
                continue  # right half of a wide character drawn just before
            attributes = _sgr(char)
            if attributes != current:
                out.append(attributes)
                current = attributes
            out.append(char.data)
    out.append("\x1b[0m")
    if (7 << 5) in screen.mode:
        out.append("\x1b[?7h")
    out.append(f"\x1b[{screen.cursor.y + 1};{screen.cursor.x + 1}H")
    out.append("\x1b[?25l" if screen.cursor.hidden else "\x1b[?25h")
    return "".join(out).encode()


class Terminal:
    """Spawn *argv* on a new PTY and fan its output out to listeners.

    Listeners are callables taking ``bytes`` (output) or ``None`` (the
    process exited). They run on the reader thread and must not block.
    """

    OUTPUT_READ_SIZE = 65536

    def __init__(self, argv, env, cwd, columns: int, rows: int) -> None:
        _validate_size(columns, rows)
        self._lock = threading.Lock()
        self._listeners: "set" = set()
        self._screen = _Screen(columns, rows)
        self._stream = pyte.ByteStream(self._screen)
        self._query_tail = b""
        self.exited = threading.Event()
        self.returncode: "int | None" = None
        master, slave = os.openpty()
        _set_window_size(master, columns, rows)
        try:
            # subprocess forks in C and never runs Python in the child, which
            # matters in this multi-threaded process. `setsid -c` makes the
            # child a session leader with the PTY (its stdin) as controlling
            # terminal; it then execs argv without forking again.
            self._process = subprocess.Popen(
                ["setsid", "-c", *argv],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                cwd=cwd,
                env=env,
                close_fds=True,
            )
        except BaseException:
            os.close(master)
            os.close(slave)
            raise
        os.close(slave)
        # Non-blocking so a frozen (paused) process cannot wedge a writer.
        os.set_blocking(master, False)
        self._master = master
        self.pid = self._process.pid
        self._reader = threading.Thread(target=self._read_loop, name="terminal-reader", daemon=True)
        self._reader.start()

    @property
    def size(self) -> "tuple[int, int]":
        with self._lock:
            return self._screen.columns, self._screen.lines

    def attach(self, listener) -> bytes:
        """Register *listener* and return a redraw of the current screen.

        Both happen under one lock with the reader, so no output is lost or
        duplicated between the snapshot and the live stream."""
        with self._lock:
            self._listeners.add(listener)
            return render_screen(self._screen)

    def detach(self, listener) -> None:
        with self._lock:
            self._listeners.discard(listener)

    def write(self, data: bytes) -> bool:
        """Send input; False when the PTY buffer is full (e.g. while paused)."""
        if self.exited.is_set():
            return False
        try:
            remaining = memoryview(data)
            while remaining:
                remaining = remaining[os.write(self._master, remaining) :]
            return True
        except BlockingIOError:
            return False
        except OSError:
            return False

    def resize(self, columns: int, rows: int) -> None:
        _validate_size(columns, rows)
        with self._lock:
            if self.exited.is_set():
                return
            # The kernel delivers SIGWINCH to the foreground process group.
            _set_window_size(self._master, columns, rows)
            self._screen.resize(lines=rows, columns=columns)

    def _read_loop(self) -> None:
        while True:
            try:
                ready, _, _ = select.select([self._master], [], [], 0.5)
                if not ready:
                    if self._process.poll() is not None and not _readable(self._master):
                        break
                    continue
                chunk = os.read(self._master, self.OUTPUT_READ_SIZE)
            except BlockingIOError:
                continue
            except OSError:
                # EIO: every slave descriptor closed, i.e. the process tree
                # that held the terminal is gone. Any other read error leaves
                # the terminal equally unusable, so it ends the session too.
                break
            if not chunk:
                break
            self._on_output(chunk)
        self.returncode = self._process.wait()
        with self._lock:
            listeners = list(self._listeners)
            self.exited.set()
        for listener in listeners:
            listener(None)
        os.close(self._master)

    def _on_output(self, chunk: bytes) -> None:
        with self._lock:
            self._stream.feed(chunk)
            listeners = list(self._listeners)
            cursor = (self._screen.cursor.y + 1, self._screen.cursor.x + 1)
        for listener in listeners:
            listener(chunk)
        if listeners:
            return
        # Nobody can answer a cursor-position query while detached; a CLI that
        # waits for the reply would otherwise stall until a browser returns.
        combined = self._query_tail + chunk
        self._query_tail = combined[-(len(_CURSOR_POSITION_QUERY) - 1) :]
        for _ in range(combined.count(_CURSOR_POSITION_QUERY)):
            self.write(f"\x1b[{cursor[0]};{cursor[1]}R".encode())

    def signal_group(self, signum: int) -> None:
        if self.exited.is_set():
            return
        try:
            os.killpg(self.pid, signum)
        except ProcessLookupError:
            pass

    def wait(self, timeout: "float | None" = None) -> "int | None":
        self.exited.wait(timeout)
        return self.returncode


def _validate_size(columns: int, rows: int) -> None:
    if not (2 <= columns <= 1000 and 2 <= rows <= 1000):
        raise ValueError("invalid terminal dimensions")


def _set_window_size(fd: int, columns: int, rows: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))


def _readable(fd: int) -> bool:
    try:
        return bool(select.select([fd], [], [], 0)[0])
    except OSError:
        return False
