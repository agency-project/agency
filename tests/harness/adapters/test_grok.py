"""Availability is separate from interactive execution (test_external_pty.py)."""

import shutil
from unittest.mock import MagicMock

from agency.configs.agconfig import agconfig, harnessadapterconfig
from agency.harness.adapters.base import AdapterRuntime
from agency.harness.adapters.grok import GrokAdapter, GrokDriver, grok_available


def test_grok_available_reflects_real_which():
    assert grok_available() == (shutil.which("grok") is not None)


def _grok_argv(tmp_path, config):
    runtime = AdapterRuntime(
        config, "test-model", "agent", "http://daemon", "attempt-key", MagicMock()
    )
    return GrokDriver(GrokAdapter(config), runtime, tmp_path, None, None, None).argv


def test_grok_disallows_subagents_by_default(tmp_path):
    assert "--no-subagents" in _grok_argv(tmp_path, agconfig())


def test_grok_omits_no_subagents_when_allowed(tmp_path):
    argv = _grok_argv(tmp_path, agconfig(harnessadapterconfig(allow_subagents=True)))
    assert "--no-subagents" not in argv
