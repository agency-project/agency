"""Tests for agharness.py -- the thin, engine-agnostic glue shared by every
agharness_backends/* concrete backend."""

from __future__ import annotations

from unittest.mock import MagicMock

from agency.harness import agharness


def test_mcp_config_preserves_host_entry_and_adds_separate_sandbox_endpoint():
    host = {
        "type": "http",
        "url": "http://harness/mcp",
        "headers": {"Authorization": "Bearer attempt"},
    }
    assert agharness.mcp_config_for("http://harness", "attempt") == {"mcpServers": {"agency": host}}
    assert agharness.mcp_config_for("http://harness", "attempt", has_sandbox_mcp_tools=True) == {
        "mcpServers": {
            "agency": host,
            "agency-sandbox": {**host, "url": "http://harness/sandbox/mcp"},
        }
    }


def _make_agent(agname="test-agent"):
    ag = MagicMock()
    ag.agname = agname
    return ag


def test_materialize_config_home_creates_isolated_directory():
    ag = _make_agent()
    d1 = agharness.materialize_config_home(ag)
    d2 = agharness.materialize_config_home(ag)
    assert d1.is_dir()
    assert d2.is_dir()
    assert d1 != d2  # each launch gets its own directory
    agharness.cleanup_config_home(d1)
    agharness.cleanup_config_home(d2)
    assert not d1.exists()
    assert not d2.exists()


def test_cleanup_config_home_is_idempotent(tmp_path):
    d = tmp_path / "nonexistent"
    agharness.cleanup_config_home(d)  # must not raise
