"""Availability is separate from interactive execution (test_external_pty.py)."""

import shutil

from agency.harness.adapters.grok import grok_available


def test_grok_available_reflects_real_which():
    assert grok_available() == (shutil.which("grok") is not None)
