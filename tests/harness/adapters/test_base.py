"""Tests for harness adapter selection, configuration, and daemon seam."""

from __future__ import annotations

import pytest

from agency.configs.agconfig import agconfig, agentconfig, harnessadapterconfig
from agency.harness.adapters.base import (
    AdapterRuntime,
    HarnessAdapter,
)
from agency.harness.adapters.opencode import OpenCodeAdapter
from agency.harness.adapters.claude_code import ClaudeCodeAdapter
from agency.harness.adapters.codex import CodexAdapter
from agency.harness.adapters.grok import GrokAdapter
from agency.harness.adapters.kimi import KimiAdapter
from agency.harness.adapters.native import NativeAdapter
from agency.harness.adapters.tandem import TandemAdapter


class TestForConfigDispatch:
    def test_native_engine_returns_native_backend(self):
        backend = HarnessAdapter.for_config("native", agconfig())
        assert isinstance(backend, NativeAdapter)

    def test_opencode_engine_returns_opencode_backend(self):
        backend = HarnessAdapter.for_config("opencode", agconfig())
        assert isinstance(backend, OpenCodeAdapter)

    def test_claude_code_engine_returns_claude_code_backend(self):
        backend = HarnessAdapter.for_config("claude_code", agconfig())
        assert isinstance(backend, ClaudeCodeAdapter)

    def test_codex_engine_returns_codex_backend(self):
        backend = HarnessAdapter.for_config("codex", agconfig())
        assert isinstance(backend, CodexAdapter)

    def test_grok_engine_returns_grok_backend(self):
        backend = HarnessAdapter.for_config("grok", agconfig())
        assert isinstance(backend, GrokAdapter)

    def test_kimi_engine_returns_kimi_adapter(self):
        adapter = HarnessAdapter.for_config("kimi", agconfig())
        assert isinstance(adapter, KimiAdapter)

    def test_unknown_engine_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown harness"):
            HarnessAdapter.for_config("not-a-real-engine", agconfig())

    def test_tandem_engine_returns_tandem_backend(self):
        cfg = agconfig(
            agentconfig(harness="tandem"),
            harnessadapterconfig(supervisor_model="big-model"),
        )
        backend = HarnessAdapter.for_config("tandem", cfg)
        assert isinstance(backend, TandemAdapter)

    def test_tandem_without_supervisor_model_raises_value_error(self):
        cfg = agconfig(agentconfig(harness="tandem"))
        with pytest.raises(ValueError, match="supervisor_model"):
            HarnessAdapter.for_config("tandem", cfg)


class TestAgHarnessConfig:
    def test_binary_path_override(self):
        cfg = agconfig(harnessadapterconfig(binary_path="/custom/opencode"))
        backend = HarnessAdapter.for_config("opencode", cfg)
        assert backend.agconfig.harness_adapter.binary_path == "/custom/opencode"

    def test_unrecognized_namespace_rejected(self):
        with pytest.raises(TypeError):
            agconfig(object())


class TestBaseDaemonAttemptNotImplemented:
    def test_run_daemon_attempt_raises_not_implemented(self):
        backend = HarnessAdapter(agconfig())
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
        backend = HarnessAdapter.for_config("opencode", agconfig())
        new_cfg = agconfig(harnessadapterconfig(binary_path="/custom/opencode"))
        backend.change_config(new_cfg)
        assert backend.agconfig.harness_adapter.binary_path == "/custom/opencode"
        new_cfg.harness_adapter.binary_path = "/other/opencode"
        assert (
            backend.agconfig.harness_adapter.binary_path == "/custom/opencode"
        )  # unaffected -- cloned

    def test_get_config_copy_returns_clone(self):
        backend = HarnessAdapter.for_config("opencode", agconfig())
        copy = backend.get_config_copy()
        copy.harness_adapter.binary_path = "/custom/opencode"
        assert backend.agconfig.harness_adapter.binary_path is None  # unaffected
