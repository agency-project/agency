"""Tests for the OpenAI(-compatible) LLM backend (agency.agllm_backends.openai)."""

from __future__ import annotations

from unittest.mock import patch

import httpx

from agency.agconfig import agConfig
from agency.agllm_backends.openai import _OpenAICompatibleBackend


def _cfg(**fields) -> agConfig:
    """Test helper: wrap agllm_backend fields in an agConfig."""
    return agConfig({"agllm_backend": fields})


class TestOpenAICompatibleBackend:
    def test_make_client_passes_config_through(self):
        backend = _OpenAICompatibleBackend(_cfg(api_key="k", base_url="http://x/v1"))
        with patch("agency.agllm_backends.openai.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        MockCls.assert_called_once_with(
            api_key="k", base_url="http://x/v1", timeout=httpx.Timeout(5.0)
        )

    def test_make_client_defaults_api_key_to_empty(self):
        backend = _OpenAICompatibleBackend(_cfg(base_url="http://x/v1"))
        with patch("agency.agllm_backends.openai.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        MockCls.assert_called_once_with(
            api_key="EMPTY", base_url="http://x/v1", timeout=httpx.Timeout(5.0)
        )

    def test_tokenize_url_strips_v1_suffix(self):
        backend = _OpenAICompatibleBackend(_cfg(base_url="http://x:8000/v1"))
        assert backend.tokenize_url() == "http://x:8000"

    def test_tokenize_url_strips_trailing_slash(self):
        backend = _OpenAICompatibleBackend(_cfg(base_url="http://x:8000/v1/"))
        assert backend.tokenize_url() == "http://x:8000"

    def test_tokenize_url_none_when_no_base_url(self):
        assert _OpenAICompatibleBackend(_cfg()).tokenize_url() is None

    def test_tokenize_url_preserves_non_v1_path(self):
        backend = _OpenAICompatibleBackend(_cfg(base_url="http://x:8000/custom"))
        assert backend.tokenize_url() == "http://x:8000/custom"
