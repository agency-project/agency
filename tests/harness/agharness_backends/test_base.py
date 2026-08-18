"""Tests for agharness_backends/base.py: agharness_backend.for_config()
dispatch, the shared Fields/Config-view mechanics, and the shared
`execute()` template method (retry loop, structured-output collection,
session capture, transcript building) every migrated backend inherits."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agency.agconfig import agConfig
from agency.agcontext import agcontext
from agency.agdata import agdata, agerror
from agency.harness.agharness_backends.base import (
    AttemptResult,
    agharness_backend,
    agHarnessConfig,
)
from agency.harness.agharness_backends.opencode import _OpencodeBackend
from agency.harness.agharness_backends.claude_code import _ClaudeCodeBackend
from agency.harness.agharness_backends.codex import _CodexBackend
from agency.harness.agharness_backends.grok import _GrokBackend
from agency.harness.agharness_backends.native import _NativeBackend
from agency.agskill import agskill


class TestForConfigDispatch:
    def test_native_engine_returns_native_backend(self):
        backend = agharness_backend.for_config("native", agConfig())
        assert isinstance(backend, _NativeBackend)

    def test_opencode_engine_returns_opencode_backend(self):
        backend = agharness_backend.for_config("opencode", agConfig())
        assert isinstance(backend, _OpencodeBackend)

    def test_claude_code_engine_returns_claude_code_backend(self):
        backend = agharness_backend.for_config("claude_code", agConfig())
        assert isinstance(backend, _ClaudeCodeBackend)

    def test_codex_engine_returns_codex_backend(self):
        backend = agharness_backend.for_config("codex", agConfig())
        assert isinstance(backend, _CodexBackend)

    def test_grok_engine_returns_grok_backend(self):
        backend = agharness_backend.for_config("grok", agConfig())
        assert isinstance(backend, _GrokBackend)

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
    def test_run_attempt_raises_not_implemented(self):
        """`execute()` is now a real shared template method (see this
        module's own docstring) -- the abstract hook a concrete backend
        must implement is `_run_attempt()`, not `execute()` itself."""
        backend = agharness_backend(agConfig())
        with pytest.raises(NotImplementedError):
            backend._run_attempt(
                None,
                None,
                None,
                None,
                None,
                prompt="",
                resume_session_id=None,
                prior_session_blob=None,
                max_steps=None,
            )

    def test_execute_returns_agerror_without_host_manager_or_launch(self):
        backend = agharness_backend(agConfig())
        skill = agskill(name="s", system_prompt="do the thing")
        result, ctx, delta = backend.execute(
            MagicMock(), agcontext(), agdata(x=1), None, skill=skill
        )
        assert isinstance(result, agerror)
        assert "host_manager/launch" in result.error


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


class _FakeBackend(agharness_backend):
    """Minimal concrete backend for exercising the SHARED `execute()`
    template method in isolation, with `_run_attempt()`'s return values
    scripted per call -- engine-specific behavior (argv/env, binary
    resolution, CLI output parsing) is each real backend's own concern and
    is tested in that backend's own test file (e.g. test_claude_code.py)."""

    engine_key = "fake"

    def __init__(self, agconfig, attempts):
        super().__init__(agconfig)
        self._attempts = list(attempts)
        self.calls = []

    def _run_attempt(
        self,
        ag,
        host_manager,
        harness_base_url,
        launch,
        skill,
        *,
        prompt,
        resume_session_id,
        prior_session_blob,
        max_steps,
    ):
        self.calls.append({"prompt": prompt, "resume_session_id": resume_session_id})
        return self._attempts.pop(0)


class _FakeHostManager:
    def __init__(self, transcript=None, collected=None):
        self._transcript = transcript
        self._collected = collected if collected is not None else {}

    def transcript_for_token(self, token):
        return self._transcript

    def collected_output(self, token):
        return self._collected


def _make_ag():
    ag = MagicMock()
    ag._harness_sessions = {}
    return ag


class TestSharedExecuteTemplate:
    def test_single_attempt_success_uses_final_text_result(self):
        attempt = AttemptResult(ok=True, final_text="hi there", input_tokens=3, output_tokens=2)
        backend = _FakeBackend(agConfig(), [attempt])
        ag = _make_ag()
        skill = agskill(name="s", system_prompt="do the thing")
        prev_ctx = agcontext()

        result, ctx, delta = backend.execute(
            ag,
            prev_ctx,
            agdata(task="go"),
            None,
            skill=skill,
            host_manager=_FakeHostManager(),
            harness_base_url="http://harness.local",
            launch=SimpleNamespace(token="tok-1"),
        )

        assert not isinstance(result, agerror)
        assert result.result == "hi there"
        assert ctx is prev_ctx
        assert ctx.total_input_tokens == 3
        assert ctx.total_output_tokens == 2
        # No transcript from the (fake) host_manager -- falls back to the
        # coarse [system, user, assistant] shape rather than an empty one.
        assert delta[0]["role"] == "system"
        assert delta[1]["role"] == "user"
        assert delta[2] == {"role": "assistant", "content": "hi there"}
        assert len(backend.calls) == 1

    def test_hard_launch_failure_returns_agerror_with_message(self):
        attempt = AttemptResult(ok=False, error_message="claude binary not found")
        backend = _FakeBackend(agConfig(), [attempt])
        ag = _make_ag()
        skill = agskill(name="s", system_prompt="do the thing")

        result, ctx, delta = backend.execute(
            ag,
            agcontext(),
            agdata(task="go"),
            None,
            skill=skill,
            host_manager=_FakeHostManager(),
            harness_base_url="http://harness.local",
            launch=SimpleNamespace(token="tok-1"),
        )

        assert isinstance(result, agerror)
        assert "claude binary not found" in result.error
        assert len(backend.calls) == 1

    def test_structured_output_missing_after_retries_exhausted_returns_agerror(self):
        attempts = [
            AttemptResult(ok=True, final_text="partial")
            for _ in range(3)  # 1 initial attempt + 2 retries
        ]
        backend = _FakeBackend(agConfig(), attempts)
        ag = _make_ag()
        skill = agskill(
            name="s",
            system_prompt="do the thing",
            output_schema=agdata(greeting=str, word_count=int),
            max_output_schema_retries=2,
        )
        host_manager = _FakeHostManager(collected={"greeting": "hi"})  # word_count never arrives

        result, ctx, delta = backend.execute(
            ag,
            agcontext(),
            agdata(task="go"),
            None,
            skill=skill,
            host_manager=host_manager,
            harness_base_url="http://harness.local",
            launch=SimpleNamespace(token="tok-1"),
        )

        assert isinstance(result, agerror)
        assert "word_count" in result.error
        assert len(backend.calls) == 3
        assert "still missing" in backend.calls[1]["prompt"].lower()

    def test_structured_output_completes_on_retry(self):
        backend = _FakeBackend(
            agConfig(),
            [
                AttemptResult(ok=True, final_text="partial"),
                AttemptResult(ok=True, final_text="done"),
            ],
        )
        ag = _make_ag()
        skill = agskill(
            name="s",
            system_prompt="do the thing",
            output_schema=agdata(greeting=str, word_count=int),
            max_output_schema_retries=3,
        )

        class _RetryingHostManager(_FakeHostManager):
            def __init__(self):
                super().__init__()
                self._calls = 0

            def collected_output(self, token):
                self._calls += 1
                if self._calls == 1:
                    return {"greeting": "hi"}
                return {"greeting": "hi", "word_count": 2}

        result, ctx, delta = backend.execute(
            ag,
            agcontext(),
            agdata(task="go"),
            None,
            skill=skill,
            host_manager=_RetryingHostManager(),
            harness_base_url="http://harness.local",
            launch=SimpleNamespace(token="tok-1"),
        )

        assert not isinstance(result, agerror), result
        assert result.greeting == "hi"
        assert result.word_count == 2
        assert len(backend.calls) == 2
        assert backend.calls[0]["resume_session_id"] is None

    def test_session_id_and_blob_are_captured_onto_agent(self):
        attempt = AttemptResult(
            ok=True, final_text="ok", session_id="sess-abc", session_blob=b"raw-session-bytes"
        )
        backend = _FakeBackend(agConfig(), [attempt])
        ag = _make_ag()
        skill = agskill(name="s", system_prompt="do the thing")

        backend.execute(
            ag,
            agcontext(),
            agdata(task="go"),
            None,
            skill=skill,
            host_manager=_FakeHostManager(),
            harness_base_url="http://harness.local",
            launch=SimpleNamespace(token="tok-1"),
        )

        import base64

        stored = ag._harness_sessions["fake"]
        assert stored["session_id"] == "sess-abc"
        assert base64.b64decode(stored["blob_b64"]) == b"raw-session-bytes"

    def test_second_attempt_resumes_first_attempts_session_id(self):
        backend = _FakeBackend(
            agConfig(),
            [
                AttemptResult(ok=True, final_text="partial", session_id="sess-1"),
                AttemptResult(ok=True, final_text="done", session_id="sess-1"),
            ],
        )
        ag = _make_ag()
        skill = agskill(
            name="s",
            system_prompt="do the thing",
            output_schema=agdata(greeting=str),
            max_output_schema_retries=2,
        )
        host_manager = _FakeHostManager()
        host_manager.collected_output = MagicMock(side_effect=[{}, {"greeting": "hi"}])

        backend.execute(
            ag,
            agcontext(),
            agdata(task="go"),
            None,
            skill=skill,
            host_manager=host_manager,
            harness_base_url="http://harness.local",
            launch=SimpleNamespace(token="tok-1"),
        )

        assert backend.calls[0]["resume_session_id"] is None
        assert backend.calls[1]["resume_session_id"] == "sess-1"

    def test_uses_host_manager_transcript_for_history_when_available(self):
        transcript = [
            {"role": "system", "content": "harness's own system prompt"},
            {"role": "user", "content": "run echo ok"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "1", "type": "function", "function": {"name": "Bash"}}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
            {"role": "assistant", "content": "the command printed ok"},
        ]
        attempt = AttemptResult(ok=True, final_text="the command printed ok")
        backend = _FakeBackend(agConfig(), [attempt])
        ag = _make_ag()
        skill = agskill(name="s", system_prompt="do the thing")
        prev_ctx = agcontext()

        result, ctx, delta = backend.execute(
            ag,
            prev_ctx,
            agdata(task="go"),
            None,
            skill=skill,
            host_manager=_FakeHostManager(transcript=transcript),
            harness_base_url="http://harness.local",
            launch=SimpleNamespace(token="tok-1"),
        )

        assert ctx.messages == transcript[1:]
        assert any(m.get("role") == "tool" for m in ctx.messages), "tool turn was flattened away"
        assert delta[0]["role"] == "system"
        assert delta[1:] == transcript[1:]
