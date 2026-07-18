"""Tests for agharness_backends/base.py: agharness_backend.for_config()
dispatch and the shared Fields/Config-view mechanics."""

from __future__ import annotations

import pytest

from agency.agconfig import agConfig
from agency.agharness_internal.agharness_backends.base import agharness_backend, agHarnessConfig
from agency.agharness_internal.agharness_backends.opencode import _OpencodeBackend
from agency.agharness_internal.agharness_backends.claude_code import _ClaudeCodeBackend
from agency.agharness_internal.agharness_backends.codex import _CodexBackend


class TestForConfigDispatch:
    def test_opencode_engine_returns_opencode_backend(self):
        backend = agharness_backend.for_config("opencode", agConfig())
        assert isinstance(backend, _OpencodeBackend)

    def test_claude_code_engine_returns_claude_code_backend(self):
        backend = agharness_backend.for_config("claude_code", agConfig())
        assert isinstance(backend, _ClaudeCodeBackend)

    def test_codex_engine_returns_codex_backend(self):
        backend = agharness_backend.for_config("codex", agConfig())
        assert isinstance(backend, _CodexBackend)

    def test_unknown_engine_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown harness engine"):
            agharness_backend.for_config("not-a-real-engine", agConfig())


class TestAgHarnessConfig:
    def test_sets_gateway_mode(self):
        cfg = agConfig(agHarnessConfig(gateway_mode="translate"))
        assert cfg.get("agharness", "gateway_mode") == "translate"

    def test_default_gateway_mode_is_passthrough(self):
        backend = agharness_backend.for_config("opencode", agConfig())
        assert backend.gateway_mode == "passthrough"

    def test_binary_path_override(self):
        cfg = agConfig(agHarnessConfig(binary_path="/custom/opencode"))
        backend = agharness_backend.for_config("opencode", cfg)
        assert backend.binary_path == "/custom/opencode"

    def test_unknown_field_rejected(self):
        with pytest.raises(TypeError):
            agHarnessConfig(not_a_real_field=1)


class TestBaseExecuteNotImplemented:
    def test_execute_raises_not_implemented(self):
        backend = agharness_backend(agConfig())
        with pytest.raises(NotImplementedError):
            backend.execute(None, None, None, None, skill=None)


class TestChangeConfigAndGetConfigCopy:
    def test_change_config_clones(self):
        backend = agharness_backend.for_config("opencode", agConfig())
        new_cfg = agConfig(agHarnessConfig(gateway_mode="translate"))
        backend.change_config(new_cfg)
        assert backend.gateway_mode == "translate"
        new_cfg.set("agharness", "gateway_mode", "passthrough")
        assert backend.gateway_mode == "translate"  # unaffected -- cloned

    def test_get_config_copy_returns_clone(self):
        backend = agharness_backend.for_config("opencode", agConfig())
        copy = backend.get_config_copy()
        copy.set("agharness", "gateway_mode", "translate")
        assert backend.gateway_mode == "passthrough"  # unaffected
