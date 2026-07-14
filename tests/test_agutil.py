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
