"""Availability is separate from interactive execution (test_external_pty.py)."""

import shutil

from agency.harness.adapters.codex import codex_available


def test_codex_available_reflects_real_which():
    assert codex_available() == (shutil.which("codex") is not None)
