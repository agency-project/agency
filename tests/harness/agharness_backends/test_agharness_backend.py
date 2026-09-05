"""Tests for harness adapter selection, configuration, and daemon seam."""

from __future__ import annotations

import pytest

from agency.configs.agconfig import agconfig
from agency.harness.adapters.agharness_backend import (
    AdapterRuntime,
    agharness_backend,
)
from agency.harness.adapters.opencode import _OpencodeBackend
from agency.harness.adapters.claude_code import _ClaudeCodeBackend
from agency.harness.adapters.codex import _CodexBackend
from agency.harness.adapters.grok import _GrokBackend
from agency.harness.adapters.native import _NativeBackend


class TestForConfigDispatch:
    def test_native_engine_returns_native_backend(self):
        backend = agharness_backend.for_config("native", agconfig())
        assert isinstance(backend, _NativeBackend)

    def test_opencode_engine_returns_opencode_backend(self):
        backend = agharness_backend.for_config("opencode", agconfig())
        assert isinstance(backend, _OpencodeBackend)

    def test_claude_code_engine_returns_claude_code_backend(self):
        backend = agharness_backend.for_config("claude_code", agconfig())
        assert isinstance(backend, _ClaudeCodeBackend)

    def test_codex_engine_returns_codex_backend(self):
        backend = agharness_backend.for_config("codex", agconfig())
        assert isinstance(backend, _CodexBackend)

    def test_grok_engine_returns_grok_backend(self):
        backend = agharness_backend.for_config("grok", agconfig())
        assert isinstance(backend, _GrokBackend)

    def test_unknown_engine_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown harness"):
            agharness_backend.for_config("not-a-real-engine", agconfig())


class TestAgHarnessConfig:
    def test_binary_path_override(self):
        cfg = agconfig(binary_path="/custom/opencode")
        backend = agharness_backend.for_config("opencode", cfg)
        assert backend.agconfig.binary_path == "/custom/opencode"

    def test_unknown_field_rejected(self):
        with pytest.raises(TypeError):
            agconfig(not_a_real_field=1)


class TestBaseDaemonAttemptNotImplemented:
    def test_run_daemon_attempt_raises_not_implemented(self):
        backend = agharness_backend(agconfig())
        with pytest.raises(NotImplementedError):
            backend.run_daemon_attempt(
                AdapterRuntime(
                    agconfig=agconfig(),
                    model="test-model",
                    engine_name="test",
                    harness_base_url="http://harness.local",
                    token="test-token",
                    syscall_policy=object(),
                ),
                prompt="",
                resume_session_id=None,
                prior_session_blob=None,
                max_steps=None,
            )


class TestChangeConfigAndGetConfigCopy:
    def test_change_config_clones(self):
        backend = agharness_backend.for_config("opencode", agconfig())
        new_cfg = agconfig(binary_path="/custom/opencode")
        backend.change_config(new_cfg)
        assert backend.agconfig.binary_path == "/custom/opencode"
        new_cfg.binary_path = "/other/opencode"
        assert backend.agconfig.binary_path == "/custom/opencode"  # unaffected -- cloned

    def test_get_config_copy_returns_clone(self):
        backend = agharness_backend.for_config("opencode", agconfig())
        copy = backend.get_config_copy()
        copy.binary_path = "/custom/opencode"
        assert backend.agconfig.binary_path is None  # unaffected
