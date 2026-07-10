"""Tests for agllm_backend — backend dispatch, client construction, and the
Anthropic-on-Bedrock adapter (message/tool translation, streaming chunk
shim, non-streaming shim)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

import agency.agllm_backend as agllm_backend_module
from agency.agconfig import agConfig
from agency.agllm_backend import (
    BAD_REQUEST_EXCS,
    API_CONN_EXCS,
    agllm_backend,
    _AnthropicAWSBackend,
    _AnthropicBackend,
    _AnthropicBedrockBackend,
    _AnthropicBedrockChatClient,
    _AnthropicBedrockCompletions,
    _AnthropicNonStreamResponse,
    _BedrockSigV4Auth,
    _OpenAICompatibleBackend,
    _OpenAICompatibleBedrockBackend,
    _anthropic_stream_to_openai_chunks,
    _is_anthropic_bedrock_model,
    _known_anthropic_context_window,
    _openai_messages_to_anthropic,
    _openai_tools_to_anthropic,
)

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None


def _cfg(**fields) -> agConfig:
    """Test helper: wrap agllm_backend fields in an agConfig."""
    return agConfig({"agllm_backend": fields})


# ---------------------------------------------------------------------------
# _is_anthropic_bedrock_model
# ---------------------------------------------------------------------------

class TestIsAnthropicBedrockModel:
    @pytest.mark.parametrize("model", [
        "anthropic.claude-sonnet-5",
        "us.anthropic.claude-sonnet-5",
        "eu.anthropic.claude-opus-4-8",
        "apac.anthropic.claude-haiku-4-5",
        "global.anthropic.claude-sonnet-5",
    ])
    def test_matches_anthropic_ids(self, model):
        assert _is_anthropic_bedrock_model(model) is True

    @pytest.mark.parametrize("model", [
        "nvidia.nemotron-super-3-120b",
        "qwen.qwen3-32b",
        "openai.gpt-oss-120b",
        "",
        None,
        "not-anthropic.claude-sonnet-5",
    ])
    def test_rejects_non_anthropic_ids(self, model):
        assert _is_anthropic_bedrock_model(model) is False


# ---------------------------------------------------------------------------
# agllm_backend.for_config — dispatch
# ---------------------------------------------------------------------------

class TestForConfig:
    def test_plain_config_returns_openai_compatible(self):
        backend = agllm_backend.for_config(_cfg(api_key='k', model=''))
        assert isinstance(backend, _OpenAICompatibleBackend)
        assert not isinstance(backend, _OpenAICompatibleBedrockBackend)

    def test_no_provider_key_returns_openai_compatible(self):
        backend = agllm_backend.for_config(_cfg(model=''))
        assert isinstance(backend, _OpenAICompatibleBackend)

    def test_bedrock_non_anthropic_model_returns_mantle_backend(self):
        backend = agllm_backend.for_config(_cfg(provider='bedrock', region='us-east-2', model='nvidia.nemotron-super-3-120b'))
        assert isinstance(backend, _OpenAICompatibleBedrockBackend)

    def test_bedrock_anthropic_model_returns_anthropic_backend(self):
        backend = agllm_backend.for_config(_cfg(provider='bedrock', region='us-east-2', model='us.anthropic.claude-sonnet-5'))
        assert isinstance(backend, _AnthropicBedrockBackend)

    def test_bedrock_anthropic_bare_id_returns_anthropic_backend(self):
        backend = agllm_backend.for_config(_cfg(provider='bedrock', region='us-east-2', model='anthropic.claude-opus-4-8'))
        assert isinstance(backend, _AnthropicBedrockBackend)

    def test_config_stored_on_instance(self):
        cfg = _cfg(model="some-model")
        backend = agllm_backend.for_config(cfg)
        assert backend.model == "some-model"

    def test_anthropic_provider_returns_anthropic_backend(self):
        backend = agllm_backend.for_config(_cfg(provider='anthropic', model='claude-sonnet-5'))
        assert isinstance(backend, _AnthropicBackend)
        assert not isinstance(backend, _AnthropicBedrockBackend)

    def test_anthropic_aws_provider_returns_anthropic_aws_backend(self):
        backend = agllm_backend.for_config(_cfg(provider='anthropicAWS', model='claude-sonnet-5'))
        assert isinstance(backend, _AnthropicAWSBackend)
        assert not isinstance(backend, _AnthropicBackend)

    def test_anthropic_aws_snake_case_alias(self):
        backend = agllm_backend.for_config(_cfg(provider='anthropic_aws', model='claude-sonnet-5'))
        assert isinstance(backend, _AnthropicAWSBackend)


# ---------------------------------------------------------------------------
# agllm_backend base class defaults
# ---------------------------------------------------------------------------

class TestBaseBackendDefaults:
    def test_make_client_not_implemented(self):
        backend = agllm_backend(_cfg())
        with pytest.raises(NotImplementedError):
            backend.make_client(httpx.Timeout(10.0))

    def test_tokenize_url_defaults_to_none(self):
        assert agllm_backend(_cfg()).tokenize_url() is None

    def test_known_context_limit_defaults_to_none(self):
        assert agllm_backend(_cfg()).known_context_limit("anything") is None

    def test_list_models_delegates_to_make_client(self):
        backend = agllm_backend(_cfg())
        mock_client = MagicMock()
        mock_client.models.list.return_value = ["m1", "m2"]
        with patch.object(backend, "make_client", return_value=mock_client) as mock_make:
            result = backend.list_models()
        assert result == ["m1", "m2"]
        mock_make.assert_called_once()

    def test_list_models_propagates_exceptions(self):
        backend = agllm_backend(_cfg())
        with patch.object(backend, "make_client", side_effect=RuntimeError("offline")):
            with pytest.raises(RuntimeError):
                backend.list_models()


# ---------------------------------------------------------------------------
# _OpenAICompatibleBackend
# ---------------------------------------------------------------------------

class TestOpenAICompatibleBackend:
    def test_make_client_passes_config_through(self):
        backend = _OpenAICompatibleBackend(_cfg(api_key='k', base_url='http://x/v1'))
        with patch("agency.agllm_backend.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        MockCls.assert_called_once_with(api_key="k", base_url="http://x/v1", timeout=httpx.Timeout(5.0))

    def test_make_client_defaults_api_key_to_empty(self):
        backend = _OpenAICompatibleBackend(_cfg(base_url='http://x/v1'))
        with patch("agency.agllm_backend.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        MockCls.assert_called_once_with(api_key="EMPTY", base_url="http://x/v1", timeout=httpx.Timeout(5.0))

    def test_tokenize_url_strips_v1_suffix(self):
        backend = _OpenAICompatibleBackend(_cfg(base_url='http://x:8000/v1'))
        assert backend.tokenize_url() == "http://x:8000"

    def test_tokenize_url_strips_trailing_slash(self):
        backend = _OpenAICompatibleBackend(_cfg(base_url='http://x:8000/v1/'))
        assert backend.tokenize_url() == "http://x:8000"

    def test_tokenize_url_none_when_no_base_url(self):
        assert _OpenAICompatibleBackend(_cfg()).tokenize_url() is None

    def test_tokenize_url_preserves_non_v1_path(self):
        backend = _OpenAICompatibleBackend(_cfg(base_url='http://x:8000/custom'))
        assert backend.tokenize_url() == "http://x:8000/custom"


# ---------------------------------------------------------------------------
# _OpenAICompatibleBedrockBackend
# ---------------------------------------------------------------------------

class TestOpenAICompatibleBedrockBackend:
    def test_direct_bedrock_api_key_uses_mantle_url(self):
        backend = _OpenAICompatibleBedrockBackend(_cfg(region='us-east-2', api_key='bedrock-api-key-abc123'))
        with patch("agency.agllm_backend.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        MockCls.assert_called_once_with(
            api_key="bedrock-api-key-abc123",
            base_url="https://bedrock-mantle.us-east-2.api.aws/v1",
            timeout=httpx.Timeout(5.0),
        )

    def test_no_api_key_uses_token_generator_when_available(self, monkeypatch):
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        backend = _OpenAICompatibleBedrockBackend(_cfg(region='us-west-2'))
        fake_token_mod = SimpleNamespace(provide_token=lambda region: "generated-token")
        with patch.dict("sys.modules", {"aws_bedrock_token_generator": fake_token_mod}), \
             patch("agency.agllm_backend.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        MockCls.assert_called_once_with(
            api_key="generated-token",
            base_url="https://bedrock-mantle.us-west-2.api.aws/v1",
            timeout=httpx.Timeout(5.0),
        )

    def test_falls_back_to_sigv4_when_token_generator_unavailable(self, monkeypatch):
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        backend = _OpenAICompatibleBedrockBackend(_cfg(region='us-east-1'))
        with patch.dict("sys.modules", {"aws_bedrock_token_generator": None}), \
             patch("agency.agllm_backend.openai.OpenAI") as MockCls, \
             patch("agency.agllm_backend._BedrockSigV4Auth") as MockAuth:
            backend.make_client(httpx.Timeout(5.0))
        MockAuth.assert_called_once_with("us-east-1", api_key=None)
        _, kwargs = MockCls.call_args
        assert kwargs["api_key"] == "bedrock"
        assert kwargs["base_url"] == "https://bedrock-runtime.us-east-1.amazonaws.com"
        assert "http_client" in kwargs

    def test_explicit_colon_delimited_api_key_goes_to_sigv4_path(self):
        """An api_key containing a colon is an ACCESS_KEY_ID:SECRET_ACCESS_KEY
        pair for SigV4 signing, not a Mantle bearer token."""
        backend = _OpenAICompatibleBedrockBackend(_cfg(region='us-east-1', api_key='AKIAEXAMPLE:secretvalue'))
        with patch("agency.agllm_backend.openai.OpenAI") as MockCls, \
             patch("agency.agllm_backend._BedrockSigV4Auth") as MockAuth:
            backend.make_client(httpx.Timeout(5.0))
        MockAuth.assert_called_once_with("us-east-1", api_key="AKIAEXAMPLE:secretvalue")
        _, kwargs = MockCls.call_args
        assert kwargs["base_url"] == "https://bedrock-runtime.us-east-1.amazonaws.com"

    def test_real_bedrock_api_key_format_uses_mantle_bearer_token(self):
        """Real AWS Bedrock API keys look like 'ABSK...', not
        'bedrock-api-key-...' — they must still be recognized as a direct
        bearer token for Mantle (colon-free), not sent down the SigV4 path."""
        backend = _OpenAICompatibleBedrockBackend(_cfg(region='us-east-2', api_key='ABSKQmVkcm9ja0FQSUtleS1leGFtcGxl'))
        with patch("agency.agllm_backend.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        MockCls.assert_called_once_with(
            api_key="ABSKQmVkcm9ja0FQSUtleS1leGFtcGxl",
            base_url="https://bedrock-mantle.us-east-2.api.aws/v1",
            timeout=httpx.Timeout(5.0),
        )

    def test_reads_aws_bearer_token_bedrock_env_var_when_config_has_no_api_key(self, monkeypatch):
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "ABSKfromenv")
        backend = _OpenAICompatibleBedrockBackend(_cfg(region='us-east-2'))
        with patch("agency.agllm_backend.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        MockCls.assert_called_once_with(
            api_key="ABSKfromenv",
            base_url="https://bedrock-mantle.us-east-2.api.aws/v1",
            timeout=httpx.Timeout(5.0),
        )

    def test_config_api_key_takes_priority_over_env_var(self, monkeypatch):
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "ABSKfromenv")
        backend = _OpenAICompatibleBedrockBackend(_cfg(region='us-east-2', api_key='ABSKfromconfig'))
        with patch("agency.agllm_backend.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        _, kwargs = MockCls.call_args
        assert kwargs["api_key"] == "ABSKfromconfig"

    def test_region_defaults_to_us_east_1(self):
        backend = _OpenAICompatibleBedrockBackend(_cfg(api_key='bedrock-api-key-x'))
        with patch("agency.agllm_backend.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        _, kwargs = MockCls.call_args
        assert kwargs["base_url"] == "https://bedrock-mantle.us-east-1.api.aws/v1"

    def test_tokenize_url_is_none(self):
        assert _OpenAICompatibleBedrockBackend(_cfg()).tokenize_url() is None

    def test_inherits_openai_compatible_list_models(self):
        """Mantle exposes an OpenAI-style /v1/models — should reuse the base class."""
        backend = _OpenAICompatibleBedrockBackend(_cfg(region='us-east-2', api_key='bedrock-api-key-x'))
        mock_client = MagicMock()
        mock_client.models.list.return_value = ["qwen.qwen3-32b"]
        with patch.object(backend, "make_client", return_value=mock_client):
            assert backend.list_models() == ["qwen.qwen3-32b"]


# ---------------------------------------------------------------------------
# _AnthropicBedrockBackend
# ---------------------------------------------------------------------------

class TestAnthropicBedrockBackend:
    def test_make_client_raises_when_anthropic_sdk_missing(self):
        backend = _AnthropicBedrockBackend(_cfg(region='us-east-2'))
        with patch("agency.agllm_backend._anthropic_sdk", None):
            with pytest.raises(RuntimeError, match="pip install anthropic"):
                backend.make_client(httpx.Timeout(5.0))

    def test_make_client_constructs_anthropic_bedrock_with_region_and_timeout(self):
        backend = _AnthropicBedrockBackend(_cfg(region='eu-west-1'))
        mock_sdk = MagicMock()
        mock_anthropic_client = MagicMock()
        mock_sdk.AnthropicBedrock.return_value = mock_anthropic_client
        with patch("agency.agllm_backend._anthropic_sdk", mock_sdk):
            client = backend.make_client(httpx.Timeout(30.0))
        mock_sdk.AnthropicBedrock.assert_called_once_with(aws_region="eu-west-1", timeout=httpx.Timeout(30.0))
        assert isinstance(client, _AnthropicBedrockChatClient)

    def test_make_client_defaults_region(self):
        backend = _AnthropicBedrockBackend(_cfg())
        mock_sdk = MagicMock()
        with patch("agency.agllm_backend._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        mock_sdk.AnthropicBedrock.assert_called_once_with(aws_region="us-east-1", timeout=httpx.Timeout(5.0))

    def test_list_models_returns_empty(self):
        assert _AnthropicBedrockBackend(_cfg()).list_models() == []

    def test_tokenize_url_is_none(self):
        assert _AnthropicBedrockBackend(_cfg()).tokenize_url() is None

    def test_known_context_limit_delegates_to_lookup(self):
        backend = _AnthropicBedrockBackend(_cfg())
        assert backend.known_context_limit("us.anthropic.claude-sonnet-5") == 1_000_000
        assert backend.known_context_limit("anthropic.claude-nonexistent-model") is None


# ---------------------------------------------------------------------------
# _AnthropicAWSBackend (Claude Platform on AWS via AnthropicAWS client)
# ---------------------------------------------------------------------------

class TestAnthropicAWSBackend:
    def test_make_client_uses_anthropic_aws(self):
        backend = _AnthropicAWSBackend(_cfg(api_key='aws-api-key', region='us-east-2', workspace_id='wrkspc_test'))
        mock_sdk = MagicMock()
        mock_sdk.AnthropicAWS = MagicMock()
        with patch("agency.agllm_backend._anthropic_sdk", mock_sdk):
            client = backend.make_client(httpx.Timeout(30.0))
        mock_sdk.AnthropicAWS.assert_called_once_with(
            timeout=httpx.Timeout(30.0),
            api_key="aws-api-key",
            aws_region="us-east-2",
            workspace_id="wrkspc_test",
        )
        assert isinstance(client, _AnthropicBedrockChatClient)

    def test_env_var_fallbacks(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_AWS_API_KEY", "key-from-env")
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_from_env")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://aws-external-anthropic.us-east-2.api.aws")
        backend = _AnthropicAWSBackend(_cfg())
        mock_sdk = MagicMock()
        mock_sdk.AnthropicAWS = MagicMock()
        with patch("agency.agllm_backend._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        _, kwargs = mock_sdk.AnthropicAWS.call_args
        assert kwargs["api_key"] == "key-from-env"
        assert kwargs["workspace_id"] == "wrkspc_from_env"
        assert kwargs["base_url"] == "https://aws-external-anthropic.us-east-2.api.aws"

    def test_list_models_calls_anthropic_aws(self):
        backend = _AnthropicAWSBackend(_cfg(api_key='k', workspace_id='w', base_url='https://example.test'))
        mock_sdk = MagicMock()
        mock_raw = MagicMock()
        mock_raw.models.list.return_value = ["claude-sonnet-5"]
        mock_sdk.AnthropicAWS.return_value = mock_raw
        with patch("agency.agllm_backend._anthropic_sdk", mock_sdk):
            result = backend.list_models()
        assert result == ["claude-sonnet-5"]

    def test_tokenize_url_is_none(self):
        assert _AnthropicAWSBackend(_cfg()).tokenize_url() is None

    def test_known_context_limit_delegates_to_lookup(self):
        backend = _AnthropicAWSBackend(_cfg())
        assert backend.known_context_limit("claude-sonnet-5") == 1_000_000


# ---------------------------------------------------------------------------
# _AnthropicBackend (first-party API, not Bedrock)
# ---------------------------------------------------------------------------

class TestAnthropicBackend:
    def test_make_client_raises_when_anthropic_sdk_missing(self):
        backend = _AnthropicBackend(_cfg())
        with patch("agency.agllm_backend._anthropic_sdk", None):
            with pytest.raises(RuntimeError, match="pip install anthropic"):
                backend.make_client(httpx.Timeout(5.0))

    def test_make_client_uses_config_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
        backend = _AnthropicBackend(_cfg(api_key='sk-ant-from-config'))
        mock_sdk = MagicMock()
        with patch("agency.agllm_backend._anthropic_sdk", mock_sdk):
            client = backend.make_client(httpx.Timeout(30.0))
        mock_sdk.Anthropic.assert_called_once_with(api_key="sk-ant-from-config", timeout=httpx.Timeout(30.0))
        assert isinstance(client, _AnthropicBedrockChatClient)

    def test_make_client_falls_back_to_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
        backend = _AnthropicBackend(_cfg())
        mock_sdk = MagicMock()
        with patch("agency.agllm_backend._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        mock_sdk.Anthropic.assert_called_once_with(api_key="sk-ant-from-env", timeout=httpx.Timeout(5.0))

    def test_config_api_key_takes_priority_over_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
        backend = _AnthropicBackend(_cfg(api_key='sk-ant-from-config'))
        mock_sdk = MagicMock()
        with patch("agency.agllm_backend._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        _, kwargs = mock_sdk.Anthropic.call_args
        assert kwargs["api_key"] == "sk-ant-from-config"

    def test_list_models_returns_empty_when_sdk_missing(self):
        backend = _AnthropicBackend(_cfg())
        with patch("agency.agllm_backend._anthropic_sdk", None):
            assert backend.list_models() == []

    def test_list_models_calls_raw_client_not_chat_wrapper(self):
        backend = _AnthropicBackend(_cfg(api_key='sk-ant-x'))
        mock_sdk = MagicMock()
        mock_raw_client = MagicMock()
        mock_raw_client.models.list.return_value = ["claude-sonnet-5"]
        mock_sdk.Anthropic.return_value = mock_raw_client
        with patch("agency.agllm_backend._anthropic_sdk", mock_sdk):
            result = backend.list_models()
        assert result == ["claude-sonnet-5"]

    def test_tokenize_url_is_none(self):
        assert _AnthropicBackend(_cfg()).tokenize_url() is None

    def test_known_context_limit_delegates_to_lookup(self):
        backend = _AnthropicBackend(_cfg())
        assert backend.known_context_limit("claude-sonnet-5") == 1_000_000
        assert backend.known_context_limit("claude-nonexistent-model") is None

    def test_no_workspace_id_omits_default_headers(self, monkeypatch):
        """Claude Platform on AWS requires the header; plain api.anthropic.com
        doesn't use it — omit default_headers entirely rather than sending an
        empty/None header when no workspace ID is configured."""
        monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
        backend = _AnthropicBackend(_cfg(api_key='sk-ant-x'))
        mock_sdk = MagicMock()
        with patch("agency.agllm_backend._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        _, kwargs = mock_sdk.Anthropic.call_args
        assert "default_headers" not in kwargs

    def test_config_workspace_id_sent_as_header(self):
        """Claude Platform on AWS (short-term API key + ANTHROPIC_BASE_URL
        override) rejects requests with 400 'Missing anthropic-workspace-id
        header' unless this is sent explicitly — the plain client does not
        read ANTHROPIC_WORKSPACE_ID into a header on its own."""
        backend = _AnthropicBackend(_cfg(api_key='sk-ant-x', workspace_id='wrkspc_from_config'))
        mock_sdk = MagicMock()
        with patch("agency.agllm_backend._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        _, kwargs = mock_sdk.Anthropic.call_args
        assert kwargs["default_headers"] == {"anthropic-workspace-id": "wrkspc_from_config"}

    def test_workspace_id_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_from_env")
        backend = _AnthropicBackend(_cfg(api_key='sk-ant-x'))
        mock_sdk = MagicMock()
        with patch("agency.agllm_backend._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        _, kwargs = mock_sdk.Anthropic.call_args
        assert kwargs["default_headers"] == {"anthropic-workspace-id": "wrkspc_from_env"}

    def test_config_workspace_id_takes_priority_over_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_from_env")
        backend = _AnthropicBackend(_cfg(api_key='sk-ant-x', workspace_id='wrkspc_from_config'))
        mock_sdk = MagicMock()
        with patch("agency.agllm_backend._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        _, kwargs = mock_sdk.Anthropic.call_args
        assert kwargs["default_headers"] == {"anthropic-workspace-id": "wrkspc_from_config"}

    def test_list_models_also_sends_workspace_header(self):
        backend = _AnthropicBackend(_cfg(api_key='sk-ant-x', workspace_id='wrkspc_from_config'))
        mock_sdk = MagicMock()
        mock_sdk.Anthropic.return_value.models.list.return_value = []
        with patch("agency.agllm_backend._anthropic_sdk", mock_sdk):
            backend.list_models()
        _, kwargs = mock_sdk.Anthropic.call_args
        assert kwargs["default_headers"] == {"anthropic-workspace-id": "wrkspc_from_config"}


# ---------------------------------------------------------------------------
# _known_anthropic_context_window
# ---------------------------------------------------------------------------

class TestKnownAnthropicContextWindow:
    @pytest.mark.parametrize("model,expected", [
        ("us.anthropic.claude-sonnet-5", 1_000_000),
        ("anthropic.claude-sonnet-5", 1_000_000),
        ("eu.anthropic.claude-opus-4-8", 1_000_000),
        ("global.anthropic.claude-fable-5", 1_000_000),
        ("apac.anthropic.claude-haiku-4-5", 200_000),
        ("anthropic.claude-haiku-4-5-20251001-v1:0", 200_000),
        ("anthropic.claude-opus-4-5-20251101-v1:0", 1_000_000),
        ("claude-sonnet-5", 1_000_000),  # bare first-party ID, no Bedrock prefix
        ("claude-haiku-4-5", 200_000),
    ])
    def test_known_models_resolve(self, model, expected):
        assert _known_anthropic_context_window(model) == expected

    @pytest.mark.parametrize("model", [
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-instant-v1",
        "",
        None,
    ])
    def test_unknown_models_return_none(self, model):
        assert _known_anthropic_context_window(model) is None

    def test_does_not_prefix_match_unrelated_longer_name(self):
        """'claude-sonnet-5' must not accidentally match a model that merely
        starts with the same characters without a '-' boundary."""
        assert _known_anthropic_context_window("anthropic.claude-sonnet-50000") is None


# ---------------------------------------------------------------------------
# _openai_messages_to_anthropic
# ---------------------------------------------------------------------------

class TestOpenAIMessagesToAnthropic:
    def test_system_message_extracted(self):
        system, msgs = _openai_messages_to_anthropic([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ])
        assert system == "You are helpful."
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_multiple_system_messages_joined(self):
        system, _ = _openai_messages_to_anthropic([
            {"role": "system", "content": "Part 1."},
            {"role": "system", "content": "Part 2."},
            {"role": "user", "content": "hi"},
        ])
        assert system == "Part 1.\n\nPart 2."

    def test_no_system_message_returns_none(self):
        system, _ = _openai_messages_to_anthropic([{"role": "user", "content": "hi"}])
        assert system is None

    def test_empty_system_content_not_appended(self):
        system, _ = _openai_messages_to_anthropic([
            {"role": "system", "content": ""},
            {"role": "user", "content": "hi"},
        ])
        assert system is None

    def test_plain_assistant_text(self):
        _, msgs = _openai_messages_to_anthropic([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        assert msgs[1] == {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}

    def test_assistant_with_tool_call_becomes_tool_use_block(self):
        _, msgs = _openai_messages_to_anthropic([
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }],
            },
        ])
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == [{
            "type": "tool_use", "id": "call_1", "name": "get_weather",
            "input": {"city": "Paris"},
        }]

    def test_assistant_with_text_and_tool_call_both_present(self):
        _, msgs = _openai_messages_to_anthropic([
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant", "content": "Let me check.",
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }],
            },
        ])
        blocks = msgs[1]["content"]
        assert blocks[0] == {"type": "text", "text": "Let me check."}
        assert blocks[1]["type"] == "tool_use"

    def test_malformed_tool_call_arguments_become_empty_dict(self):
        _, msgs = _openai_messages_to_anthropic([
            {"role": "user", "content": "x"},
            {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "f", "arguments": "not json"},
                }],
            },
        ])
        assert msgs[1]["content"][0]["input"] == {}

    def test_assistant_no_content_no_tools_becomes_empty_string(self):
        _, msgs = _openai_messages_to_anthropic([
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": None},
        ])
        assert msgs[1] == {"role": "assistant", "content": ""}

    def test_tool_result_becomes_user_message_with_tool_result_block(self):
        _, msgs = _openai_messages_to_anthropic([
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "get_weather", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "Sunny, 20C", "tool_call_id": "call_1"},
        ])
        assert msgs[2] == {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "Sunny, 20C"}],
        }

    def test_consecutive_tool_results_merge_into_one_user_message(self):
        _, msgs = _openai_messages_to_anthropic([
            {"role": "user", "content": "x"},
            {
                "role": "assistant", "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "content": "result a", "tool_call_id": "c1"},
            {"role": "tool", "content": "result b", "tool_call_id": "c2"},
        ])
        # Only one user message should follow the assistant turn, with both results.
        tool_result_msgs = [m for m in msgs if m["role"] == "user" and isinstance(m["content"], list)]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0]["content"] == [
            {"type": "tool_result", "tool_use_id": "c1", "content": "result a"},
            {"type": "tool_result", "tool_use_id": "c2", "content": "result b"},
        ]

    def test_tool_result_after_assistant_creates_new_user_message(self):
        """A tool result immediately after an assistant text turn (not a user
        message with list content) must start a fresh user message."""
        _, msgs = _openai_messages_to_anthropic([
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "thinking out loud"},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ])
        assert msgs[2] == {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "result"}],
        }

    def test_unrecognized_role_dropped(self):
        _, msgs = _openai_messages_to_anthropic([
            {"role": "user", "content": "x"},
            {"role": "function_call_result_legacy", "content": "should be dropped"},
        ])
        assert len(msgs) == 1


# ---------------------------------------------------------------------------
# _openai_tools_to_anthropic
# ---------------------------------------------------------------------------

class TestOpenAIToolsToAnthropic:
    def test_none_returns_none(self):
        assert _openai_tools_to_anthropic(None) is None

    def test_empty_list_returns_none(self):
        assert _openai_tools_to_anthropic([]) is None

    def test_converts_openai_function_tool_shape(self):
        result = _openai_tools_to_anthropic([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }])
        assert result == [{
            "name": "get_weather",
            "description": "Get the weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }]

    def test_missing_parameters_defaults_to_empty_object_schema(self):
        result = _openai_tools_to_anthropic([{"type": "function", "function": {"name": "f"}}])
        assert result[0]["input_schema"] == {"type": "object", "properties": {}}

    def test_flat_tool_shape_without_function_wrapper(self):
        """Defensive fallback: a tool dict without a 'function' key is treated
        as already flat."""
        result = _openai_tools_to_anthropic([{"name": "f", "description": "d"}])
        assert result == [{"name": "f", "description": "d", "input_schema": {"type": "object", "properties": {}}}]

    def test_multiple_tools_converted_in_order(self):
        result = _openai_tools_to_anthropic([
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
        ])
        assert [t["name"] for t in result] == ["a", "b"]


# ---------------------------------------------------------------------------
# _anthropic_stream_to_openai_chunks
# ---------------------------------------------------------------------------

def _ev(**kwargs):
    return SimpleNamespace(**kwargs)


class TestAnthropicStreamToOpenAIChunks:
    def test_text_only_stream(self):
        stream = [
            _ev(type="message_start", message=_ev(usage=_ev(input_tokens=10))),
            _ev(type="content_block_start", index=0, content_block=_ev(type="text", text="")),
            _ev(type="content_block_delta", index=0, delta=_ev(type="text_delta", text="Hello")),
            _ev(type="content_block_delta", index=0, delta=_ev(type="text_delta", text=", world")),
            _ev(type="content_block_stop", index=0),
            _ev(type="message_delta", usage=_ev(output_tokens=5)),
            _ev(type="message_stop"),
        ]
        chunks = list(_anthropic_stream_to_openai_chunks(iter(stream)))
        contents = [c.choices[0].delta.content for c in chunks if c.choices]
        assert contents == ["Hello", ", world"]
        final = chunks[-1]
        assert final.usage.prompt_tokens == 10
        assert final.usage.completion_tokens == 5
        assert final.choices == []

    def test_thinking_delta_emits_reasoning_content(self):
        stream = [
            _ev(type="message_start", message=_ev(usage=None)),
            _ev(type="content_block_start", index=0, content_block=_ev(type="thinking", thinking="")),
            _ev(type="content_block_delta", index=0, delta=_ev(type="thinking_delta", thinking="pondering")),
            _ev(type="content_block_stop", index=0),
        ]
        chunks = list(_anthropic_stream_to_openai_chunks(iter(stream)))
        reasoning = [c.choices[0].delta.reasoning_content for c in chunks if c.choices]
        assert reasoning == ["pondering"]

    def test_tool_use_emits_single_chunk_with_full_arguments(self):
        """Regression test: tool-call JSON must arrive as ONE chunk with the
        complete concatenated arguments, not streamed fragment-by-fragment —
        see agllm_backend.py's docstring for why (a real bug found in
        production: fragments could be dropped by downstream batching)."""
        stream = [
            _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
            _ev(type="content_block_start", index=0,
                content_block=_ev(type="tool_use", id="toolu_1", name="get_weather")),
            _ev(type="content_block_delta", index=0, delta=_ev(type="input_json_delta", partial_json="")),
            _ev(type="content_block_delta", index=0, delta=_ev(type="input_json_delta", partial_json='{"city": ')),
            _ev(type="content_block_delta", index=0, delta=_ev(type="input_json_delta", partial_json='"Paris"}')),
            _ev(type="content_block_stop", index=0),
            _ev(type="message_delta", usage=_ev(output_tokens=1)),
        ]
        chunks = list(_anthropic_stream_to_openai_chunks(iter(stream)))
        tool_call_chunks = [c for c in chunks if c.choices and c.choices[0].delta.tool_calls]
        assert len(tool_call_chunks) == 1
        tc = tool_call_chunks[0].choices[0].delta.tool_calls[0]
        assert tc.index == 0
        assert tc.id == "toolu_1"
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"city": "Paris"}'

    def test_truncated_tool_use_is_flushed_not_dropped(self):
        """Regression: if the stream ends (e.g. stop_reason="max_tokens")
        while a tool_use block is still open, content_block_stop never fires
        for it. Previously this silently dropped the tool call entirely,
        producing a completely empty assistant turn with no error — the
        real-world symptom being a harness that reprompts forever because it
        thinks the model just didn't respond. The partial JSON must still be
        flushed so the caller sees a (possibly unparseable) tool call attempt
        instead of silence."""
        stream = [
            _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
            _ev(type="content_block_start", index=0,
                content_block=_ev(type="tool_use", id="toolu_1", name="return_env_requirements")),
            _ev(type="content_block_delta", index=0, delta=_ev(type="input_json_delta", partial_json='{"foo": ')),
            _ev(type="content_block_delta", index=0, delta=_ev(type="input_json_delta", partial_json='"bar')),
            # stream ends here — no content_block_stop, no message_stop event needed
            _ev(type="message_delta", usage=_ev(output_tokens=1)),
        ]
        chunks = list(_anthropic_stream_to_openai_chunks(iter(stream)))
        tool_call_chunks = [c for c in chunks if c.choices and c.choices[0].delta.tool_calls]
        assert len(tool_call_chunks) == 1
        tc = tool_call_chunks[0].choices[0].delta.tool_calls[0]
        assert tc.id == "toolu_1"
        assert tc.function.name == "return_env_requirements"
        assert tc.function.arguments == '{"foo": "bar'  # truncated, but present

    def test_text_then_tool_use_at_nonzero_index(self):
        stream = [
            _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
            _ev(type="content_block_start", index=0, content_block=_ev(type="text", text="")),
            _ev(type="content_block_delta", index=0, delta=_ev(type="text_delta", text="Checking...")),
            _ev(type="content_block_stop", index=0),
            _ev(type="content_block_start", index=1,
                content_block=_ev(type="tool_use", id="toolu_2", name="get_weather")),
            _ev(type="content_block_delta", index=1, delta=_ev(type="input_json_delta", partial_json='{"city":"NYC"}')),
            _ev(type="content_block_stop", index=1),
            _ev(type="message_delta", usage=_ev(output_tokens=1)),
        ]
        chunks = list(_anthropic_stream_to_openai_chunks(iter(stream)))
        text_chunks = [c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content]
        tool_chunks = [c for c in chunks if c.choices and c.choices[0].delta.tool_calls]
        assert text_chunks == ["Checking..."]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].choices[0].delta.tool_calls[0].index == 1
        assert tool_chunks[0].choices[0].delta.tool_calls[0].function.arguments == '{"city":"NYC"}'

    def test_content_block_stop_without_prior_tool_use_emits_nothing(self):
        """content_block_stop for a text block (never registered in
        tool_blocks) must not emit a spurious tool_call chunk."""
        stream = [
            _ev(type="message_start", message=_ev(usage=None)),
            _ev(type="content_block_start", index=0, content_block=_ev(type="text", text="")),
            _ev(type="content_block_delta", index=0, delta=_ev(type="text_delta", text="hi")),
            _ev(type="content_block_stop", index=0),
        ]
        chunks = list(_anthropic_stream_to_openai_chunks(iter(stream)))
        tool_chunks = [c for c in chunks if c.choices and c.choices[0].delta.tool_calls]
        assert tool_chunks == []

    def test_empty_stream_still_yields_final_usage_chunk(self):
        chunks = list(_anthropic_stream_to_openai_chunks(iter([])))
        assert len(chunks) == 1
        assert chunks[0].usage.prompt_tokens == 0
        assert chunks[0].usage.completion_tokens == 0

    def test_missing_usage_on_message_start_defaults_to_zero(self):
        stream = [_ev(type="message_start", message=_ev(usage=None))]
        chunks = list(_anthropic_stream_to_openai_chunks(iter(stream)))
        assert chunks[-1].usage.prompt_tokens == 0


# ---------------------------------------------------------------------------
# _AnthropicNonStreamResponse
# ---------------------------------------------------------------------------

class TestAnthropicNonStreamResponse:
    def test_extracts_text_blocks(self):
        message = _ev(content=[_ev(type="text", text="Hello "), _ev(type="text", text="world")])
        resp = _AnthropicNonStreamResponse(message)
        assert resp.choices[0].message.content == "Hello world"

    def test_ignores_non_text_blocks(self):
        message = _ev(content=[
            _ev(type="tool_use", id="t1", name="f", input={}),
            _ev(type="text", text="answer"),
        ])
        resp = _AnthropicNonStreamResponse(message)
        assert resp.choices[0].message.content == "answer"

    def test_no_text_blocks_returns_empty_string(self):
        message = _ev(content=[_ev(type="tool_use", id="t1", name="f", input={})])
        resp = _AnthropicNonStreamResponse(message)
        assert resp.choices[0].message.content == ""


# ---------------------------------------------------------------------------
# _AnthropicBedrockCompletions.create
# ---------------------------------------------------------------------------

class TestAnthropicBedrockCompletions:
    def test_streaming_call_translates_kwargs_and_wraps_stream(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = iter([])
        completions = _AnthropicBedrockCompletions(mock_client)

        result = completions.create(
            model="us.anthropic.claude-sonnet-5",
            messages=[{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}],
            stream=True,
            max_tokens=256,
            temperature=0.5,
            top_p=0.9,
            extra_body={"top_k": 40},
            tools=[{"type": "function", "function": {"name": "f"}}],
        )

        mock_client.messages.create.assert_called_once_with(
            stream=True,
            model="us.anthropic.claude-sonnet-5",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
            ]}],
            max_tokens=256,
            system=[{"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral"}}],
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            tools=[{"name": "f", "description": "", "input_schema": {"type": "object", "properties": {}}}],
        )
        # Streaming path returns the chunk-translating generator, not the raw stream.
        assert hasattr(result, "__iter__")
        assert not isinstance(result, MagicMock)

    def test_non_streaming_call_defaults_and_returns_wrapped_response(self):
        mock_message = _ev(content=[_ev(type="text", text="pong")])
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        completions = _AnthropicBedrockCompletions(mock_client)

        result = completions.create(
            model="us.anthropic.claude-sonnet-5",
            messages=[{"role": "user", "content": "ping"}],
        )

        mock_client.messages.create.assert_called_once_with(
            model="us.anthropic.claude-sonnet-5",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "ping", "cache_control": {"type": "ephemeral"}},
            ]}],
            max_tokens=128000,
        )
        assert isinstance(result, _AnthropicNonStreamResponse)
        assert result.choices[0].message.content == "pong"

    def test_no_messages_omits_last_message_cache_control(self):
        mock_message = _ev(content=[_ev(type="text", text="pong")])
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        completions = _AnthropicBedrockCompletions(mock_client)

        completions.create(
            model="us.anthropic.claude-sonnet-5",
            messages=[{"role": "system", "content": "Be terse."}],
        )

        mock_client.messages.create.assert_called_once_with(
            model="us.anthropic.claude-sonnet-5",
            messages=[],
            max_tokens=128000,
            system=[{"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral"}}],
        )

    def test_cache_control_lands_on_last_tool_result_block_not_first(self):
        # A tool-role message gets merged into the preceding user message's
        # content list by _openai_messages_to_anthropic; the cache_control
        # breakpoint must land on the *last* block of that list.
        mock_message = _ev(content=[_ev(type="text", text="pong")])
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        completions = _AnthropicBedrockCompletions(mock_client)

        completions.create(
            model="us.anthropic.claude-sonnet-5",
            messages=[
                {"role": "user", "content": "call the tool"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "t1", "function": {"name": "f", "arguments": "{}"}},
                ]},
                {"role": "tool", "tool_call_id": "t1", "content": "result-1"},
            ],
        )

        sent_messages = mock_client.messages.create.call_args.kwargs["messages"]
        tool_result_message = sent_messages[-1]
        assert tool_result_message["content"][-1] == {
            "type": "tool_result", "tool_use_id": "t1", "content": "result-1",
            "cache_control": {"type": "ephemeral"},
        }

    def test_cache_control_does_not_mutate_caller_messages(self):
        # anthropic_messages is a fresh structure built by
        # _openai_messages_to_anthropic, but guard against a regression where
        # cache_control gets written back into a shared/reused dict.
        mock_message = _ev(content=[_ev(type="text", text="pong")])
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        completions = _AnthropicBedrockCompletions(mock_client)

        original_messages = [{"role": "user", "content": "ping"}]
        completions.create(model="us.anthropic.claude-sonnet-5", messages=original_messages)

        assert original_messages == [{"role": "user", "content": "ping"}]

    def test_max_tokens_defaults_to_128000_when_omitted(self):
        # Regression: a 4096 default could truncate mid-tool-call on a large
        # structured tool argument, silently dropping the call entirely.
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _ev(content=[])
        _AnthropicBedrockCompletions(mock_client).create(
            model="m", messages=[{"role": "user", "content": "x"}],
        )
        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["max_tokens"] == 128000

    def test_extra_body_without_top_k_is_ignored(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _ev(content=[])
        _AnthropicBedrockCompletions(mock_client).create(
            model="m", messages=[{"role": "user", "content": "x"}],
            extra_body={"repetition_penalty": 1.1},
        )
        _, kwargs = mock_client.messages.create.call_args
        assert "top_k" not in kwargs

    def test_no_system_message_omits_system_kwarg(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _ev(content=[])
        _AnthropicBedrockCompletions(mock_client).create(
            model="m", messages=[{"role": "user", "content": "x"}],
        )
        _, kwargs = mock_client.messages.create.call_args
        assert "system" not in kwargs

    def test_ignores_unknown_openai_kwargs(self):
        """stream_options and other OpenAI-only kwargs must be swallowed, not
        forwarded to the Anthropic SDK (which would reject unknown params)."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _ev(content=[])
        _AnthropicBedrockCompletions(mock_client).create(
            model="m", messages=[{"role": "user", "content": "x"}],
            stream_options={"include_usage": True},
            frequency_penalty=0.1,
        )
        _, kwargs = mock_client.messages.create.call_args
        assert "stream_options" not in kwargs
        assert "frequency_penalty" not in kwargs


# ---------------------------------------------------------------------------
# _AnthropicBedrockChatClient
# ---------------------------------------------------------------------------

class TestAnthropicBedrockChatClient:
    def test_chat_completions_create_delegates_to_wrapped_client(self):
        mock_anthropic_client = MagicMock()
        mock_anthropic_client.messages.create.return_value = _ev(content=[_ev(type="text", text="hi")])
        client = _AnthropicBedrockChatClient(mock_anthropic_client)

        result = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
        assert result.choices[0].message.content == "hi"

    def test_close_calls_underlying_client_close(self):
        mock_anthropic_client = MagicMock()
        client = _AnthropicBedrockChatClient(mock_anthropic_client)
        client.close()
        mock_anthropic_client.close.assert_called_once()

    def test_close_is_noop_when_underlying_client_has_no_close(self):
        class NoClose:
            pass
        client = _AnthropicBedrockChatClient(NoClose())
        client.close()  # must not raise


# ---------------------------------------------------------------------------
# Exception-translation tuples
# ---------------------------------------------------------------------------

class TestExceptionTuples:
    def test_bad_request_excs_includes_openai(self):
        assert openai.BadRequestError in BAD_REQUEST_EXCS

    def test_api_conn_excs_includes_openai(self):
        assert openai.APIConnectionError in API_CONN_EXCS

    @pytest.mark.skipif(_anthropic_sdk is None, reason="anthropic package not installed")
    def test_bad_request_excs_includes_anthropic_when_installed(self):
        assert _anthropic_sdk.BadRequestError in BAD_REQUEST_EXCS

    @pytest.mark.skipif(_anthropic_sdk is None, reason="anthropic package not installed")
    def test_api_conn_excs_includes_anthropic_when_installed(self):
        assert _anthropic_sdk.APIConnectionError in API_CONN_EXCS


# ---------------------------------------------------------------------------
# _BedrockSigV4Auth
# ---------------------------------------------------------------------------

class TestBedrockSigV4Auth:
    def test_parses_access_and_secret_key(self):
        auth = _BedrockSigV4Auth("us-east-1", api_key="AKIA123:secretvalue")
        assert auth._creds.access_key == "AKIA123"
        assert auth._creds.secret_key == "secretvalue"
        assert auth._creds.token is None

    def test_parses_session_token(self):
        auth = _BedrockSigV4Auth("us-east-1", api_key="AKIA123:secretvalue:sessiontoken")
        assert auth._creds.token == "sessiontoken"

    def test_malformed_api_key_raises_value_error(self):
        with pytest.raises(ValueError, match="ACCESS_KEY_ID:SECRET_ACCESS_KEY"):
            _BedrockSigV4Auth("us-east-1", api_key="not-a-valid-key")

    def test_no_api_key_uses_boto3_session_credentials(self):
        fake_creds = MagicMock()
        with patch("boto3.Session") as MockSession:
            MockSession.return_value.get_credentials.return_value = fake_creds
            auth = _BedrockSigV4Auth("us-east-1")
        assert auth._creds is fake_creds
        MockSession.assert_called_once_with(region_name="us-east-1")

    def test_no_api_key_and_no_boto3_credentials_raises_runtime_error(self):
        with patch("boto3.Session") as MockSession:
            MockSession.return_value.get_credentials.return_value = None
            with pytest.raises(RuntimeError, match="No AWS credentials found"):
                _BedrockSigV4Auth("us-east-1")

    def test_auth_flow_signs_request_and_copies_headers(self):
        auth = _BedrockSigV4Auth("us-east-1", api_key="AKIA123:secretvalue")
        request = httpx.Request("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/model/x/invoke",
                                 content=b'{"a": 1}')

        def fake_add_auth(aws_req):
            aws_req.headers["Authorization"] = "AWS4-HMAC-SHA256 fake-signature"

        with patch("botocore.auth.SigV4Auth") as MockSigV4:
            MockSigV4.return_value.add_auth.side_effect = fake_add_auth
            gen = auth.auth_flow(request)
            signed_request = next(gen)

        assert signed_request.headers["Authorization"] == "AWS4-HMAC-SHA256 fake-signature"
