"""Offline checks for examples/smoke_test_models.py -- config building, model
selection, .env loading, key redaction, and expected-failure judging. No
network access, no API keys:
the live stages only run when the script is executed directly."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from agency.llm.agllm import agllm
from agency.llm.anthropic import _AnthropicBackend
from agency.llm.openai import _OpenAICompatibleBackend
from agency.llm.openai_responses import _OpenAIResponsesBackend

_PATH = Path(__file__).parent.parent / "examples" / "smoke_test_models.py"
_spec = importlib.util.spec_from_file_location("agency_smoke_test_models", _PATH)
smt = importlib.util.module_from_spec(_spec)
# dataclasses resolve their defining module through sys.modules.
sys.modules[_spec.name] = smt
_spec.loader.exec_module(smt)


def test_targets_current_models_plus_astra_and_6_1_sol_on_responses():
    assert [(s.model, s.provider) for s in smt.MODELS] == [
        ("gpt-6-astra", "openai"),
        ("gpt-6-sol", "openai"),
        ("gpt-6.1-sol", "openai"),
        ("gpt-6-luna", "openai"),
        ("gpt-6-astra", "openai_responses"),
        ("gpt-6.1-sol", "openai_responses"),
        ("claude-fable-5-1", "anthropic"),
        ("claude-opus-5-5", "anthropic"),
        ("claude-sonnet-5-5", "anthropic"),
    ]


@pytest.mark.parametrize("spec", smt.MODELS, ids=lambda s: s.model)
def test_build_config_selects_expected_backend(monkeypatch, spec):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    cfg = smt.build_config(spec)
    backend = agllm.for_config(cfg)
    assert backend.model == spec.model
    if spec.provider == "openai":
        assert type(backend) is _OpenAICompatibleBackend
        assert cfg.llm.base_url == "https://api.openai.com/v1"
        assert cfg.llm.api_key == "test-openai"
    elif spec.provider == "openai_responses":
        assert type(backend) is _OpenAIResponsesBackend
        assert cfg.llm.base_url == "https://api.openai.com/v1"
        assert cfg.llm.api_key == "test-openai"
        assert cfg.llm.reasoning_effort is None
    else:
        assert type(backend) is _AnthropicBackend
        assert cfg.llm.base_url is None
        assert cfg.llm.reasoning_effort is None
        assert cfg.llm.api_key == "test-anthropic"


def test_gpt6_tool_capable_models_default_to_reasoning_effort_none():
    efforts = {
        s.model: smt.build_config(s).llm.reasoning_effort
        for s in smt.MODELS
        if s.provider == "openai"
    }
    assert efforts["gpt-6-sol"] == "none"
    assert efforts["gpt-6-luna"] == "none"
    # Astra and GPT-6.1 Sol reject "none"; their provider default must be
    # left alone.
    assert efforts["gpt-6-astra"] is None
    assert efforts["gpt-6.1-sol"] is None


def test_reasoning_effort_override_applies_to_openai_only():
    by_model = {s.model: s for s in smt.MODELS}
    assert (
        smt.build_config(by_model["gpt-6-luna"], reasoning_effort="low").llm.reasoning_effort
        == "low"
    )
    assert (
        smt.build_config(by_model["claude-opus-5-5"], reasoning_effort="low").llm.reasoning_effort
        is None
    )


def test_select_by_provider_and_model():
    assert {s.provider for s in smt.select_models("anthropic", None)} == {"anthropic"}
    assert [s.model for s in smt.select_models("openai_responses", None)] == [
        "gpt-6-astra",
        "gpt-6.1-sol",
    ]
    # One model name can run on both OpenAI backends.
    assert {s.provider for s in smt.select_models(None, ["gpt-6-astra"])} == {
        "openai",
        "openai_responses",
    }
    assert [s.model for s in smt.select_models(None, ["gpt-6-sol"])] == ["gpt-6-sol"]
    assert smt.select_models("anthropic", ["gpt-6-sol"]) == []
    with pytest.raises(SystemExit):
        smt.select_models(None, ["gpt-4o"])


def test_default_harnesses_are_native_plus_provider_wire():
    by_model = {s.model: s for s in smt.MODELS}
    assert smt.harnesses_for(by_model["gpt-6-luna"], "default") == ["native", "codex"]
    assert smt.harnesses_for(by_model["claude-opus-5-5"], "default") == ["native", "claude_code"]
    assert smt.harnesses_for(by_model["claude-opus-5-5"], "codex") == ["codex"]


def test_load_env_file_never_overrides_existing_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "already-set")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        '# comment\nOPENAI_API_KEY=from-file\nexport ANTHROPIC_API_KEY="quoted-value"\nNOEQUALS\n'
    )
    loaded = smt.load_env_file(env)
    assert loaded == ["ANTHROPIC_API_KEY"]
    assert smt.os.environ["OPENAI_API_KEY"] == "already-set"
    assert smt.os.environ["ANTHROPIC_API_KEY"] == "quoted-value"
    assert smt.load_env_file(tmp_path / "missing.env") == []


def test_redact_removes_configured_keys_and_key_shaped_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret-openai-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-abcdefghijklmnop")
    text = smt.redact(
        "bad key super-secret-openai-value / sk-ant-api03-abcdefghijklmnop / sk-proj-zzzzzzzzzzzz"
    )
    assert "super-secret" not in text
    assert "abcdefghijklmnop" not in text
    assert "zzzzzzzz" not in text


@pytest.mark.parametrize(
    "status,expected",
    [
        (400, "request rejected"),
        (401, "account access"),
        (403, "account access"),
        (404, "model unavailable"),
        (429, "rate limit/quota"),
        (529, "provider error"),
    ],
)
def test_classify_status(status, expected):
    assert smt.classify_status(status) == expected


ASTRA_CHAT_ERROR = (
    "Function tools with reasoning_effort are not supported for gpt-6-astra in /v1/chat/completions"
)


def test_only_astra_and_6_1_sol_on_chat_completions_are_expected_failures():
    expected = {(s.model, s.provider): s.expected_e2e_error for s in smt.MODELS}
    assert expected.pop(("gpt-6-astra", "openai")) == ASTRA_CHAT_ERROR
    assert expected.pop(("gpt-6.1-sol", "openai")) == ASTRA_CHAT_ERROR.replace(
        "gpt-6-astra", "gpt-6.1-sol"
    )
    assert set(expected.values()) == {None}


def test_expected_provider_error_in_upstream_record_is_xfail():
    # On the native wire the harness only sees a truncated stream; the
    # provider's message survives in the host's upstream record.
    harness_error = RuntimeError("peer closed connection without sending complete message body")
    upstream = [f"BadRequestError: Error code: 400 - {{'message': \"{ASTRA_CHAT_ERROR}.\"}}"]
    outcome, _detail = smt.judge_e2e_failure(harness_error, upstream, ASTRA_CHAT_ERROR)
    assert outcome == smt.XFAIL


def test_expected_provider_error_in_harness_error_is_xfail():
    harness_error = RuntimeError(f"HTTP 400: {ASTRA_CHAT_ERROR}. To use function tools, ...")
    outcome, _detail = smt.judge_e2e_failure(harness_error, [], ASTRA_CHAT_ERROR)
    assert outcome == smt.XFAIL


def test_different_failure_of_an_expected_failure_spec_still_fails():
    harness_error = RuntimeError("HTTP 401: invalid api key")
    outcome, detail = smt.judge_e2e_failure(
        harness_error, ["AuthenticationError: 401"], ASTRA_CHAT_ERROR
    )
    assert outcome is False
    assert "expected" in detail and "upstream" in detail


def test_failure_without_expectation_fails():
    outcome, _detail = smt.judge_e2e_failure(
        RuntimeError(ASTRA_CHAT_ERROR), [ASTRA_CHAT_ERROR], None
    )
    assert outcome is False


def test_unexpected_pass_of_an_expected_failure_spec_fails():
    outcome, detail = smt.judge_e2e_success(smt.SECRET_WORD, 1.0, ASTRA_CHAT_ERROR)
    assert outcome is False
    assert "expected to fail" in detail


def test_success_without_expectation_requires_the_secret_word():
    assert smt.judge_e2e_success(smt.SECRET_WORD, 1.0, None)[0] is True
    assert smt.judge_e2e_success("something else", 1.0, None)[0] is False


def test_upstream_error_log_keeps_only_failed_exchange_errors():
    log = smt.UpstreamErrorLog()
    log.record_llm_exchange(
        "a", exchange_type="llm_stream_success", response_chain=[("h", {"role": "assistant"})]
    )
    log.record_llm_exchange(
        "b", exchange_type="llm_stream_cancelled", response_chain=[("h", {"cancelled": True})]
    )
    log.record_llm_exchange(
        "c", exchange_type="llm_stream_error", response_chain=[("h", {"error": "boom"})]
    )
    assert log.errors == ["boom"]
