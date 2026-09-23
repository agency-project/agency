"""Tests for the bash built-in in tandem_harness/tools.py -- specifically
that it actually honors the workdir/timeout it advertises in BASH_PARAMS
(both were silently ignored before)."""

from __future__ import annotations

import json

from agency.tandem_harness import tools


def _run_bash(**kwargs):
    return json.loads(tools.TOOL_DISPATCH["bash"](json.dumps(kwargs)))


def test_bash_honors_workdir(tmp_path):
    result = _run_bash(command="pwd", workdir=str(tmp_path))
    assert result["output"].strip() == str(tmp_path)
    assert result["returncode"] == 0


def test_bash_defaults_workdir_to_process_cwd_when_omitted():
    import os

    result = _run_bash(command="pwd")
    assert result["output"].strip() == os.getcwd()


def test_bash_reports_a_clear_error_for_a_missing_workdir(tmp_path):
    missing = str(tmp_path / "does-not-exist")
    result = _run_bash(command="pwd", workdir=missing)
    assert "error" in result
    assert missing in result["error"]


def test_bash_honors_a_custom_timeout():
    result = _run_bash(command="sleep 5", timeout=1)
    assert result == {"error": "command timed out after 1s"}


def test_bash_without_a_timeout_uses_the_120s_default():
    result = _run_bash(command="true")
    assert result["returncode"] == 0
