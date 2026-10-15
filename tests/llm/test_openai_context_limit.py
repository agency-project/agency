"""GPT-6 context limit: static max-input-token table + fetch_context_limit wiring."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agency.configs.agconfig import agconfig, llmconfig
from agency.llm.agllm import agllm
from agency.llm.openai import _known_openai_max_input_tokens
from agency.native_harness.compaction import should_compact

MAX_INPUT = 922_000


@pytest.mark.parametrize(
    "model",
    ["gpt-6-astra", "gpt-6-sol", "gpt-6.1-sol", "gpt-6-luna", "gpt-6-luna-2026-05-18"],
)
def test_known_models_resolve_to_max_input_not_total_window(model):
    assert _known_openai_max_input_tokens(model) == MAX_INPUT


@pytest.mark.parametrize(
    "model",
    [
        "gpt-6-luna-mini",
        "gpt-6-lunar",
        "gpt-6.1-luna",
        "gpt-6",
        "gpt-6-sol-pro",
        "openai.gpt-6-luna",
        "gpt-6-luna-20260518",
        "",
        None,
    ],
)
def test_lookalike_models_do_not_match(model):
    assert _known_openai_max_input_tokens(model) is None


def test_sol_and_6_1_sol_do_not_prefix_match_each_other():
    assert _known_openai_max_input_tokens("gpt-6-sol.1") is None
    assert _known_openai_max_input_tokens("gpt-6.1-sol-2026-05-18") == MAX_INPUT
    assert _known_openai_max_input_tokens("gpt-6-sol-6.1") is None


def _backend(provider, model, **extra):
    return agllm.for_config(
        agconfig(
            llmconfig(provider=provider, base_url="https://api.openai.com/v1", model=model, **extra)
        )
    )


@pytest.mark.parametrize("provider", ["openai", "openai_responses"])
class TestFetchContextLimit:
    def test_listing_without_metadata(self, provider):
        backend = _backend(provider, "gpt-6-luna")
        with patch.object(type(backend), "list_models", return_value=[]):
            assert backend.fetch_context_limit() == MAX_INPUT

    def test_listing_failure(self, provider):
        backend = _backend(provider, "gpt-6-luna")
        with patch.object(type(backend), "list_models", side_effect=RuntimeError("boom")):
            assert backend.fetch_context_limit() == MAX_INPUT

    def test_unknown_model_falls_back_to_default(self, provider):
        backend = _backend(provider, "some-other-model")
        with patch.object(type(backend), "list_models", side_effect=RuntimeError("boom")):
            assert backend.fetch_context_limit() == 200_000

    def test_explicit_override_wins_without_listing(self, provider):
        backend = _backend(provider, "gpt-6-luna", context_limit=300_000)
        with patch.object(type(backend), "list_models") as listing:
            assert backend.fetch_context_limit() == 300_000
        listing.assert_not_called()


def test_compaction_threshold_is_90_percent_of_max_input():
    assert should_compact(829_799, MAX_INPUT) is False
    assert should_compact(829_800, MAX_INPUT) is True
