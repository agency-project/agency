"""Availability is separate from interactive execution (test_external_pty.py)."""

import shutil

from agency.harness.adapters.opencode import opencode_available


def test_opencode_available_reflects_real_which():
    assert opencode_available() == (shutil.which("opencode") is not None)
