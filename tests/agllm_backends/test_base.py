"""Tests for agllm_backends/base.py: agllm_backend.for_config() dispatch, the
base class's default method implementations, change_config/get_config_copy,
and the cross-SDK exception-translation tuples."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from agency.agconfig import agConfig
from agency.agllm_backends.base import agllm_backend, BAD_REQUEST_EXCS, API_CONN_EXCS
from agency.agllm_backends.openai import _OpenAICompatibleBackend
from agency.agllm_backends.bedrock import (
    _OpenAICompatibleBedrockBackend,
    _AnthropicBedrockBackend,
    _AnthropicAWSBackend,
)
from agency.agllm_backends.anthropic import _AnthropicBackend

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None


def _cfg(**fields) -> agConfig:
    """Test helper: wrap agllm_backend fields in an agConfig."""
    return agConfig({"agllm_backend": fields})


# ---------------------------------------------------------------------------
# agllm_backend.for_config — dispatch
# ---------------------------------------------------------------------------


class TestForConfig:
    def test_plain_config_returns_openai_compatible(self):
        backend = agllm_backend.for_config(_cfg(api_key="k", model=""))
        assert isinstance(backend, _OpenAICompatibleBackend)
        assert not isinstance(backend, _OpenAICompatibleBedrockBackend)

    def test_no_provider_key_returns_openai_compatible(self):
        backend = agllm_backend.for_config(_cfg(model=""))
        assert isinstance(backend, _OpenAICompatibleBackend)

    def test_bedrock_non_anthropic_model_returns_mantle_backend(self):
        backend = agllm_backend.for_config(
            _cfg(provider="bedrock", region="us-east-2", model="nvidia.nemotron-super-3-120b")
        )
        assert isinstance(backend, _OpenAICompatibleBedrockBackend)

    def test_bedrock_anthropic_model_returns_anthropic_backend(self):
        backend = agllm_backend.for_config(
            _cfg(provider="bedrock", region="us-east-2", model="us.anthropic.claude-sonnet-5")
        )
        assert isinstance(backend, _AnthropicBedrockBackend)

    def test_bedrock_anthropic_bare_id_returns_anthropic_backend(self):
        backend = agllm_backend.for_config(
            _cfg(provider="bedrock", region="us-east-2", model="anthropic.claude-opus-4-8")
        )
        assert isinstance(backend, _AnthropicBedrockBackend)

    def test_config_stored_on_instance(self):
        cfg = _cfg(model="some-model")
        backend = agllm_backend.for_config(cfg)
        assert backend.model == "some-model"

    def test_anthropic_provider_returns_anthropic_backend(self):
        backend = agllm_backend.for_config(_cfg(provider="anthropic", model="claude-sonnet-5"))
        assert isinstance(backend, _AnthropicBackend)
        assert not isinstance(backend, _AnthropicBedrockBackend)

    def test_anthropic_aws_provider_returns_anthropic_aws_backend(self):
        backend = agllm_backend.for_config(_cfg(provider="anthropicAWS", model="claude-sonnet-5"))
        assert isinstance(backend, _AnthropicAWSBackend)
        assert not isinstance(backend, _AnthropicBackend)

    def test_anthropic_aws_snake_case_alias(self):
        backend = agllm_backend.for_config(_cfg(provider="anthropic_aws", model="claude-sonnet-5"))
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
# change_config / get_config_copy
# ---------------------------------------------------------------------------


class TestBackendChangeConfigAndGetConfigCopy:
    def test_change_config_replaces_agconfig(self):
        backend = agllm_backend(_cfg(temperature=0.7))
        backend.change_config(_cfg(temperature=0.2))
        assert backend.temperature == 0.2

    def test_change_config_clones_given_agconfig(self):
        backend = agllm_backend(_cfg())
        new_cfg = _cfg(temperature=0.2)
        backend.change_config(new_cfg)
        new_cfg.agllm_backend.temperature = 0.9
        assert backend.temperature == 0.2

    def test_get_config_copy_returns_clone_not_same_object(self):
        cfg = _cfg(temperature=0.7)
        backend = agllm_backend(cfg)
        copy = backend.get_config_copy()
        assert copy is not backend._agconfig

    def test_get_config_copy_reflects_current_values(self):
        backend = agllm_backend(_cfg(temperature=0.7))
        assert backend.get_config_copy().agllm_backend.temperature == 0.7

    def test_mutating_get_config_copy_does_not_affect_backend(self):
        backend = agllm_backend(_cfg(temperature=0.7))
        copy = backend.get_config_copy()
        copy.agllm_backend.temperature = 0.1
        assert backend.temperature == 0.7


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
