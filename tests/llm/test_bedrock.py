"""Tests for the Amazon Bedrock backends and the separate "Claude Platform on
AWS" backend (agency.llm.bedrock)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agency.configs.agconfig import agconfig, llmconfig
from agency.llm.bedrock import (
    _AnthropicAWSBackend,
    _AnthropicBedrockBackend,
    _BedrockConverseBackend,
    _OpenAICompatibleBedrockBackend,
    _is_anthropic_bedrock_model,
)


def _cfg(**fields) -> agconfig:
    """Test helper: build an agconfig with the given llmconfig fields."""
    return agconfig(llmconfig(**fields))


# ---------------------------------------------------------------------------
# _is_anthropic_bedrock_model
# ---------------------------------------------------------------------------


class TestIsAnthropicBedrockModel:
    @pytest.mark.parametrize(
        "model",
        [
            "anthropic.claude-sonnet-5",
            "us.anthropic.claude-sonnet-5",
            "eu.anthropic.claude-opus-4-8",
            "apac.anthropic.claude-haiku-4-5",
            "global.anthropic.claude-sonnet-5",
        ],
    )
    def test_matches_anthropic_ids(self, model):
        assert _is_anthropic_bedrock_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "nvidia.nemotron-super-3-120b",
            "qwen.qwen3-32b",
            "openai.gpt-oss-120b",
            "",
            None,
            "not-anthropic.claude-sonnet-5",
        ],
    )
    def test_rejects_non_anthropic_ids(self, model):
        assert _is_anthropic_bedrock_model(model) is False


# ---------------------------------------------------------------------------
# _OpenAICompatibleBedrockBackend
# ---------------------------------------------------------------------------


class TestOpenAICompatibleBedrockBackend:
    def test_direct_bedrock_api_key_uses_bedrock_runtime_openai_url(self):
        backend = _OpenAICompatibleBedrockBackend(
            _cfg(region="us-east-2", api_key="bedrock-api-key-abc123")
        )
        with patch("agency.llm.bedrock.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        MockCls.assert_called_once_with(
            api_key="bedrock-api-key-abc123",
            base_url="https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1",
            timeout=httpx.Timeout(5.0),
        )

    def test_raises_when_no_api_key_configured(self, monkeypatch):
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        backend = _OpenAICompatibleBedrockBackend(_cfg(region="us-west-2"))
        with pytest.raises(RuntimeError, match="bearer-token api_key"):
            backend.make_client(httpx.Timeout(5.0))

    def test_raises_on_colon_delimited_api_key(self):
        """An api_key containing a colon looks like an
        ACCESS_KEY_ID:SECRET_ACCESS_KEY SigV4 pair -- no longer supported."""
        backend = _OpenAICompatibleBedrockBackend(
            _cfg(region="us-east-1", api_key="AKIAEXAMPLE:secretvalue")
        )
        with pytest.raises(ValueError, match="SigV4"):
            backend.make_client(httpx.Timeout(5.0))

    def test_real_bedrock_api_key_format_uses_bearer_token(self):
        """Real AWS Bedrock API keys look like 'ABSK...', not
        'bedrock-api-key-...' — they must still be recognized as a direct
        bearer token (colon-free), not sent down the SigV4 path."""
        backend = _OpenAICompatibleBedrockBackend(
            _cfg(region="us-east-2", api_key="ABSKQmVkcm9ja0FQSUtleS1leGFtcGxl")
        )
        with patch("agency.llm.bedrock.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        MockCls.assert_called_once_with(
            api_key="ABSKQmVkcm9ja0FQSUtleS1leGFtcGxl",
            base_url="https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1",
            timeout=httpx.Timeout(5.0),
        )

    def test_reads_aws_bearer_token_bedrock_env_var_when_config_has_no_api_key(self, monkeypatch):
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "ABSKfromenv")
        backend = _OpenAICompatibleBedrockBackend(_cfg(region="us-east-2"))
        with patch("agency.llm.bedrock.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        MockCls.assert_called_once_with(
            api_key="ABSKfromenv",
            base_url="https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1",
            timeout=httpx.Timeout(5.0),
        )

    def test_config_api_key_takes_priority_over_env_var(self, monkeypatch):
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "ABSKfromenv")
        backend = _OpenAICompatibleBedrockBackend(
            _cfg(region="us-east-2", api_key="ABSKfromconfig")
        )
        with patch("agency.llm.bedrock.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        _, kwargs = MockCls.call_args
        assert kwargs["api_key"] == "ABSKfromconfig"

    def test_region_defaults_to_us_east_1(self):
        backend = _OpenAICompatibleBedrockBackend(_cfg(api_key="bedrock-api-key-x"))
        with patch("agency.llm.bedrock.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        _, kwargs = MockCls.call_args
        assert kwargs["base_url"] == "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"

    def test_tokenize_url_is_none(self):
        assert _OpenAICompatibleBedrockBackend(_cfg()).tokenize_url() is None

    def test_inherits_openai_compatible_list_models(self):
        """bedrock-runtime exposes an OpenAI-style /v1/models — should reuse the base class."""
        backend = _OpenAICompatibleBedrockBackend(
            _cfg(region="us-east-2", api_key="bedrock-api-key-x")
        )
        mock_client = MagicMock()
        mock_client.models.list.return_value = ["qwen.qwen3-32b"]
        with patch.object(backend, "make_client", return_value=mock_client):
            assert backend.list_models() == ["qwen.qwen3-32b"]


# ---------------------------------------------------------------------------
# _AnthropicBedrockBackend
# ---------------------------------------------------------------------------


class TestAnthropicBedrockBackend:
    def test_make_client_raises_when_anthropic_sdk_missing(self):
        backend = _AnthropicBedrockBackend(_cfg(region="us-east-2"))
        with patch("agency.llm.bedrock._anthropic_sdk", None):
            with pytest.raises(RuntimeError, match="pip install anthropic"):
                backend.make_client(httpx.Timeout(5.0))

    def test_make_client_constructs_anthropic_bedrock_with_region_and_timeout(self):
        backend = _AnthropicBedrockBackend(_cfg(region="eu-west-1", api_key="ABSKexample"))
        mock_sdk = MagicMock()
        mock_anthropic_client = MagicMock()
        mock_sdk.AnthropicBedrock.return_value = mock_anthropic_client
        with patch("agency.llm.bedrock._anthropic_sdk", mock_sdk):
            client = backend.make_client(httpx.Timeout(30.0))
        mock_sdk.AnthropicBedrock.assert_called_once_with(
            aws_region="eu-west-1", api_key="ABSKexample", timeout=httpx.Timeout(30.0)
        )
        assert client is mock_anthropic_client

    def test_make_client_defaults_region(self):
        backend = _AnthropicBedrockBackend(_cfg(api_key="ABSKexample"))
        mock_sdk = MagicMock()
        with patch("agency.llm.bedrock._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        mock_sdk.AnthropicBedrock.assert_called_once_with(
            aws_region="us-east-1", api_key="ABSKexample", timeout=httpx.Timeout(5.0)
        )

    def test_make_client_raises_when_no_api_key_configured(self, monkeypatch):
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        backend = _AnthropicBedrockBackend(_cfg())
        with patch("agency.llm.bedrock._anthropic_sdk", MagicMock()):
            with pytest.raises(RuntimeError, match="bearer-token api_key"):
                backend.make_client(httpx.Timeout(5.0))

    def test_make_client_raises_on_colon_delimited_api_key(self):
        backend = _AnthropicBedrockBackend(_cfg(api_key="AKIAEXAMPLE:secretvalue"))
        with patch("agency.llm.bedrock._anthropic_sdk", MagicMock()):
            with pytest.raises(ValueError, match="SigV4"):
                backend.make_client(httpx.Timeout(5.0))

    def test_list_models_returns_empty(self):
        assert _AnthropicBedrockBackend(_cfg()).list_models() == []

    def test_tokenize_url_is_none(self):
        assert _AnthropicBedrockBackend(_cfg()).tokenize_url() is None

    def test_known_context_limit_delegates_to_lookup(self):
        backend = _AnthropicBedrockBackend(_cfg())
        assert backend.known_context_limit("us.anthropic.claude-sonnet-5") == 1_000_000
        assert backend.known_context_limit("anthropic.claude-nonexistent-model") is None


# ---------------------------------------------------------------------------
# _BedrockConverseBackend
# ---------------------------------------------------------------------------


class TestBedrockConverseBackend:
    def test_make_client_sets_bearer_token_env_var(self, monkeypatch):
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        backend = _BedrockConverseBackend(_cfg(region="us-east-2", api_key="ABSKexample"))
        with patch("boto3.client") as MockClient:
            backend.make_client(httpx.Timeout(5.0))
        assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "ABSKexample"
        MockClient.assert_called_once()
        args, kwargs = MockClient.call_args
        assert args[0] == "bedrock-runtime"
        assert kwargs["region_name"] == "us-east-2"

    def test_make_client_raises_when_no_api_key_configured(self, monkeypatch):
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        backend = _BedrockConverseBackend(_cfg())
        with pytest.raises(RuntimeError, match="bearer-token api_key"):
            backend.make_client(httpx.Timeout(5.0))

    def test_make_client_raises_on_colon_delimited_api_key(self):
        backend = _BedrockConverseBackend(_cfg(api_key="AKIAEXAMPLE:secretvalue"))
        with pytest.raises(ValueError, match="SigV4"):
            backend.make_client(httpx.Timeout(5.0))

    def test_list_models_returns_empty(self):
        assert _BedrockConverseBackend(_cfg()).list_models() == []

    def test_tokenize_url_is_none(self):
        assert _BedrockConverseBackend(_cfg()).tokenize_url() is None


# ---------------------------------------------------------------------------
# _AnthropicAWSBackend (Claude Platform on AWS via AnthropicAWS client)
# ---------------------------------------------------------------------------


class TestAnthropicAWSBackend:
    def test_make_client_uses_anthropic_aws(self):
        backend = _AnthropicAWSBackend(
            _cfg(api_key="aws-api-key", region="us-east-2", workspace_id="wrkspc_test")
        )
        mock_sdk = MagicMock()
        mock_sdk.AnthropicAWS = MagicMock()
        with patch("agency.llm.bedrock._anthropic_sdk", mock_sdk):
            client = backend.make_client(httpx.Timeout(30.0))
        mock_sdk.AnthropicAWS.assert_called_once_with(
            timeout=httpx.Timeout(30.0),
            api_key="aws-api-key",
            aws_region="us-east-2",
            workspace_id="wrkspc_test",
        )
        assert client is mock_sdk.AnthropicAWS.return_value

    def test_env_var_fallbacks(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_AWS_API_KEY", "key-from-env")
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_from_env")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://aws-external-anthropic.us-east-2.api.aws")
        backend = _AnthropicAWSBackend(_cfg())
        mock_sdk = MagicMock()
        mock_sdk.AnthropicAWS = MagicMock()
        with patch("agency.llm.bedrock._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        _, kwargs = mock_sdk.AnthropicAWS.call_args
        assert kwargs["api_key"] == "key-from-env"
        assert kwargs["workspace_id"] == "wrkspc_from_env"
        assert kwargs["base_url"] == "https://aws-external-anthropic.us-east-2.api.aws"

    def test_list_models_calls_anthropic_aws(self):
        backend = _AnthropicAWSBackend(
            _cfg(api_key="k", workspace_id="w", base_url="https://example.test")
        )
        mock_sdk = MagicMock()
        mock_raw = MagicMock()
        mock_raw.models.list.return_value = ["claude-sonnet-5"]
        mock_sdk.AnthropicAWS.return_value = mock_raw
        with patch("agency.llm.bedrock._anthropic_sdk", mock_sdk):
            result = backend.list_models()
        assert result == ["claude-sonnet-5"]

    def test_tokenize_url_is_none(self):
        assert _AnthropicAWSBackend(_cfg()).tokenize_url() is None

    def test_known_context_limit_delegates_to_lookup(self):
        backend = _AnthropicAWSBackend(_cfg())
        assert backend.known_context_limit("claude-sonnet-5") == 1_000_000
