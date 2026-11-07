"""Foreground ownership for a CLI using its caller's existing terminal."""

from __future__ import annotations

import logging
import os
import signal
import termios
import threading


class InheritedTerminal:
    """Hand a foreground terminal to a child process group, then restore it.

    This never creates a PTY, reads input, or redirects output. A launcher
    creates the child's process group before ``claim``. Background callers
    retain ordinary job-control behavior rather than taking over a terminal.
    """

    def __init__(self):
        self._fd = None
        self._lock = threading.RLock()
        if not os.isatty(0):
            return
        fd = os.dup(0)
        try:
            os.tcgetpgrp(fd)  # Verify that this is our controlling terminal.
            self._attributes = termios.tcgetattr(fd)
        except BaseException:
            os.close(fd)
            raise
        self._fd, self._foreground = fd, os.getpgrp()
        self._child_group = None
        self._job_attributes = None

    def claim(self, pid: int) -> bool:
        with self._lock:
            if self._fd is None or os.tcgetpgrp(self._fd) != self._foreground:
                return False
            self._update(pid, self._job_attributes)
            self._child_group = pid
            return True

    def _update(self, process_group: int, attributes=None) -> None:
        # Restoring ownership runs while Agency is a background process.
        # Block SIGTTOU only on this thread so tcsetpgrp cannot suspend the
        # runtime which provides the external resume/stop controls.
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTTOU})
        try:
            os.tcsetpgrp(self._fd, process_group)
            if attributes is not None:
                termios.tcsetattr(self._fd, termios.TCSANOW, attributes)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)

    def release(self) -> None:
        """Return the terminal to the wrapper without forgetting CLI settings."""
        with self._lock:
            if self._fd is None or os.tcgetpgrp(self._fd) != self._child_group:
                return
            self._job_attributes = termios.tcgetattr(self._fd)
            self._update(self._foreground, self._attributes)

    def suspend(self, pid: int) -> None:
        """Forward a real CLI job stop to the shell, then honor fg or bg.

        The main thread calls this after a reaper observes a kernel group
        stop. No signal handler or tracer callback performs terminal work.
        A shell's fg sets the wrapper foreground before SIGCONT; bg leaves
        it in the background, where a CLI read can naturally cause SIGTTIN.
        """
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("terminal job control requires the main thread")
        if self._fd is None:
            return
        self.release()
        os.kill(os.getpid(), signal.SIGSTOP)
        self.claim(pid)
        try:
            os.killpg(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass

    def restore(self) -> None:
        with self._lock:
            if self._fd is None:
                return
            fd = self._fd
            try:
                self.release()
            except (OSError, termios.error) as exc:
                # Closing the terminal during a session must not strand the
                # process reaper or hide the CLI's actual exit status.
                logging.getLogger(__name__).warning(
                    "Could not restore inherited terminal: %s", type(exc).__name__
                )
            finally:
                self._fd = None
                os.close(fd)
