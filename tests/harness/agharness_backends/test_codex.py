"""Tests for the Codex CLI adapter's availability and output parser."""

from __future__ import annotations

import json
import tomllib

from agency.agconfig import agConfig

from agency.harness.adapters.codex import _CodexBackend, codex_available


def test_codex_available_reflects_real_which():
    import shutil

    assert codex_available() == (shutil.which("codex") is not None)


def test_parse_output_events_ignores_non_agent_message_items():
    stdout = "\n".join(
        [
            json.dumps(
                {"type": "item.completed", "item": {"type": "reasoning", "text": "thinking"}}
            ),
            json.dumps(
                {"type": "item.completed", "item": {"type": "agent_message", "text": "final"}}
            ),
        ]
    )
    assert _CodexBackend._parse_output_events(stdout) == "final"


def test_parse_output_events_falls_back_to_raw_text():
    assert _CodexBackend._parse_output_events("plain text") == "plain text"


def test_codex_config_routes_model_and_mcp_through_agency(tmp_path):
    backend = _CodexBackend(agConfig())
    backend._write_codex_config(tmp_path, "http://127.0.0.1:1234", "test-model")

    config = tomllib.loads((tmp_path / "config.toml").read_text())

    assert config["model"] == "test-model"
    assert config["web_search"] == "disabled"
    assert config["agents"]["enabled"] is False
    assert config["features"]["shell_tool"] is False
    assert config["features"]["unified_exec"] is False
    assert config["model_providers"]["agency-proxy"]["base_url"] == ("http://127.0.0.1:1234/v1")
    assert config["mcp_servers"]["agency"] == {
        "url": "http://127.0.0.1:1234/mcp",
        "bearer_token_env_var": "AGENCY_PROXY_API_KEY",
        "required": True,
        "default_tools_approval_mode": "approve",
    }


def test_parse_usage_events_sums_completed_turns():
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 7, "output_tokens": 3},
                }
            ),
        ]
    )

    assert _CodexBackend._parse_usage_events(stdout) == {
        "input_tokens": 17,
        "output_tokens": 5,
    }


def test_parse_unsupported_tool_events_rejects_non_mcp_actions():
    stdout = "\n".join(
        [
            json.dumps({"type": "item.completed", "item": {"type": "mcp_tool_call"}}),
            json.dumps({"type": "item.completed", "item": {"type": "command_execution"}}),
            json.dumps({"type": "item.completed", "item": {"type": "file_change"}}),
        ]
    )

    assert _CodexBackend._parse_unsupported_tool_events(stdout) == {
        "command_execution",
        "file_change",
    }
